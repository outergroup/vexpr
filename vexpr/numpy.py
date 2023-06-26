from functools import partial

import numpy as np

import vexpr.base
from vexpr.base import CannotVectorize, ArrayShape
from vexpr.tree_util import is_leaf
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten



class Expression(vexpr.base.Expression):

    def __truediv__(self, other):
        return Divide((self, other))

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Multiply((self, other))

    def __pow__(self, other):
        return Power((self, other))

    def vectorize(self):
        return vectorize(self)


def to_shape(x):
    if isinstance(x, Expression):
        return x.return_shape
    elif isinstance(x, np.ndarray):
        return ArrayShape(x.shape)
    else:
        return ArrayShape(())


def pytree_skeleton(tree):
    return tree_map(lambda x: None, tree)


class Sum(Expression):
    def __init__(self, *operands):
        # Absorb any child sums, to support using the '+' operator to build up
        # sums.
        operands2 = []
        for op in operands:
            if isinstance(op, Sum):
                operands2.extend(op.operands)
            else:
                operands2.append(op)
        operands = tuple(operands2)

        shapes = tree_map(to_shape, operands)
        shape, *rest = shapes

        if not all(s == ArrayShape([]) for s in tree_leaves(shape)):
            raise ValueError(f"Expected all scalars, got {shape}")

        if not all(s == shape for s in rest):
            raise ValueError(f"Expected same shape for all operands, got {shapes}")

        super().__init__(*operands, return_shape=shape)

    def compute(self, symbols, *operands):
        return sum(operands)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        return reduction_vectorize(cls, VectorizedSum, *instances)


def vectorize(*vs):
    """
    The protocol:

    A call is only valid if the return type for each v has the same pytree
    structure.

    If any v is an expression, one of those expressions vectorizes this level
    (and presumably recursively calls vectorize on the next level). We iterate
    over them, choosing one of them eventually.

    If any v is a leaf literal, then every one is one, so we return Stack(*vs).

    Otherwise, either every v is a pytree container. We call vectorize on each
    child of this container, recursing down to the next level.
    """
    # TODO
    # tree_structure = tree_map(lambda *xs: (x.return_shape
    #                                        if isinstance(x, Expression)
    #                                        else type(x)),
    #                           vs)
    # tree_structure, *rest = tree_structure
    # if not all(s == tree_structure for s in rest):
    #     raise ValueError("vectorize must be called with the same pytree "
    #                      f"structure for every v, got {(tree_structure, *rest)}")

    # get unique types, preserving order
    expression_types = list(dict.fromkeys(type(v) for v in vs
                                          if isinstance(v, Expression)))

    if len(expression_types) == 0:
        if isinstance(vs[0], (tuple, list, dict)):
            f = lambda *xs: vectorize(*xs)

            # manually implement tree_children_map
            if isinstance(vs[0], tuple):
                return tuple(f(*xs) for xs in zip(*vs))
            elif isinstance(vs[0], list):
                return tuple(f(*xs) for xs in zip(*vs))
            else:
                return {k: f(*[d[k] for d in vs])
                        for k in vs[0].keys()}
        else:
            # all leafs
            return np.concatenate([(v
                                    if isinstance(v, np.array)
                                    else [v])
                                   for v in vs])
    else:
        for t in expression_types:
            try:
                return t.vectorize_level(*vs, allow_partial=False)
            except CannotVectorize:
                pass

        for t in expression_types:
            # TODO: what if I move the partial logic to here? then, for example,
            # when CDist does a partial vectorize, it returns the vectorized and
            # unvectorized parts. Then this logic will be usable by other
            # classes, and we will establish more of the control is in the hands
            # of `vectorize`.
            try:
                return t.vectorize_level(*vs, allow_partial=True)
            except CannotVectorize:
                pass

        return Stack(*vs)


class VectorizedSum(Expression):
    def __init__(self, vector, at_indices=None):
        """
        vector: a numpy array (or, more generally, a pytree of numpy arrays)

        If a pytree is provided, this loops over the arrays. A future version of
        vexpr might support a truly vectorized sume over multiple arrays.
        (Supporting multiple shapes, as long as the data shape on each is the
        same.)
        """
        self.at_indices = (np.array(at_indices)
                           if at_indices is not None
                           else None)

        data_shape = ArrayShape((self.at_indices.max() + 1,)
                                if at_indices is not None
                                else ())
        return_shape = tree_map(lambda _: data_shape, vector)
        super().__init__(vector, return_shape=return_shape)

    def compute(self, symbols, vector):
        if self.at_indices is not None:
            return tree_map(self.sum_at, vector)
        else:
            return tree_map(lambda arr: arr.sum(axis=-1),
                            vector)

    def sum_at(self, arr):
        out = np.zeros((*arr.shape[:-1], self.data_shape[-1]),
                       dtype=arr.dtype)
        np.add.at(out, (..., self.at_indices), arr)
        return out

    def sexpr(self):
        ret = super().sexpr()
        if self.at_indices is not None:
            ret = ret + (self.at_indices.tolist(),)
        return ret

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        return vectorized_reduction_vectorize(cls, *instances)

    def lift_shuffle(self):
        # assume any descendent shuffles have already been lifted when this
        # class called vectorize.
        return self, None


def invert_shuffle(indices):
    inverted_indices = np.zeros_like(indices)
    inverted_indices[indices] = np.arange(len(indices))
    return inverted_indices


class Multiply(Expression):
    """
    Multiply is treated as a binary operation, always taking two operands.
    Two '*' operators lead two nested Multiply expressions. This leads to
    vectorization behavior that is desirable when multiplying values of
    different types, for example multiplying a weight or a scale with a value
    derived from data.

    For different vectorization behavior, consider using Product.
    """
    def __init__(self, operands, lift_shuffle_from="rhs"):
        # current view: multiply is a binary operation. it has an lhs and a rhs.
        # When we collect a few things to multiply, the convention is to do
        #   Multiply((Multiply((scale, Stack(w1, w2, w3))), big_tensor))
        # so that the lhs resolves during partial evaluation.

        lhs_shape, rhs_shape = tree_map(to_shape, operands)

        def get_return_shape(lhs_data_shape, rhs_data_shape):
            if len(lhs_data_shape) == 0:
                return rhs_data_shape
            elif len(rhs_data_shape) == 0:
                return lhs_data_shape
            else:
                if rhs_data_shape != lhs_data_shape:
                    raise ValueError((rhs_data_shape, rhs_data_shape))
                return lhs_data_shape

        return_shape = tree_map(get_return_shape, lhs_shape, rhs_shape)

        super().__init__(operands, return_shape=return_shape)
        self.lift_shuffle_from = lift_shuffle_from

    def compute(self, symbols, operands):
        lhs, rhs = operands
        return tree_map(lambda l, r: l * r,
                        lhs, rhs)

    # @classmethod
    # def identity_args(cls, expr):
    #     return (1.0, expr)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        # TODO: if one operand is shape 1 and another is shape n, then we need
        # to repeat the former operand at this point so that we can concat them.
        # lhs, rhs = self.operands
        # if lhs.ndim == 0 and rhs.ndim > 0:
        #     lhs = Repeat(lhs, rhs.data_shape[-1])
        # elif lhs.ndim > 0 and rhs.ndim == 0:
        #     rhs = Repeat(rhs, lhs.data_shape[-1])

        # for now, detect this and throw an error

        # the job here is: figure out all the shapes, insert the proper
        # identities, and then call vectorize on children.

        children = []

        for instance in instances:
            if isinstance(instance, cls):
                # eat the instance, since we are incorporating it into a
                # vectorized operation
                children.append(instance.operands[0])
            else:
                # don't eat the instance, pass it through an identity function.
                # convention: 1.0 on the lhs.
                # TODO do something cleaner than shape.a
                children.append((tree_map(lambda shape: np.ones(shape.a),
                                          instance.return_shape),
                                 instance))

        return Multiply(vectorize(*children))

    def lift_shuffle(self):
        """
        a caller calls lift_shuffle when they have the opportunity to eat
        the shuffle. it is up to this class to decide whether it has a shuffle
        that can be eaten, and if that's actually desirable.
        """
        operands = self.operands[0]
        if isinstance(operands, Expression):
            return operands.lift_shuffle()

        lhs, rhs = operands

        lifted = False
        lift_from, push_to = ((lhs, rhs)
                              if self.lift_shuffle_from == "lhs"
                              else (rhs, lhs))

        # TODO switch to tree_map, verify all indices are same
        if lift_from.ndim > 0:
            try:
                lift_from, lifted_indices = lift_from.lift_shuffle()
                if lifted_indices is not None:
                    lifted = True
            except NotImplementedError:
                pass

        if not lifted:
            return self, None
        elif push_to.ndim > 0:
            push_to = shuffle(lhs, invert_shuffle(lifted_indices))

        operands = ((lift_from, push_to)
                    if self.lift_shuffle_from == "lhs"
                    else (push_to, lift_from))

        return Multiply(operands), lifted_indices


def vectorized_reduction_vectorize(cls, *instances):
    if len(instances) == 1:
        children = instances[0].operands[0]
        at_indices = None
    else:
        children = []
        at_indices = []
        offset = 0
        for instance in instances:
            if isinstance(instance, cls):
                child_instance = instance.operands[0]

                # eat the instance, since we are incorporating it into a
                # vectorized reduction
                children.append(child_instance)

                if instance.at_indices is not None:
                    at_indices += (instance.at_indices + offset).tolist()
                    offset += instance.at_indices.max() + 1
                else:
                    length = child_instance.data_shape[-1]
                    at_indices += [offset] * length
                    offset += 1
            else:
                children.append(instance)
                length = 1 if instance.ndim == 0 else instance.data_shape[-1]
                at_indices += list(range(offset, offset + length))
                offset += length

        vectorized_children = vectorize(*children)

        # Now lift a shuffle operation if it occurs amongst descendants, as we can
        # optimize them away.
        try:
            leaves, treedef = tree_flatten(vectorized_children)
            leaves, all_shuffle_indices = zip(*[leaf.lift_shuffle()
                                                for leaf in leaves])
            shuffle_indices, *rest = all_shuffle_indices

            if not all(si == shuffle_indices for si in rest):
                # if running on a pytree, we build for the case where the pytree
                # came from the user of the expression (e.g. from a symbol), and
                # hence we expect all leaves to have run the same code and thus
                # have the same shuffle operations. Other shuffle-lifting use
                # cases are not currently supported.
                print("Not lifting shuffle, didn't get same indices for pytree "
                      "leaves")
            elif shuffle_indices is not None:
                vectorized_children = tree_unflatten(treedef, leaves)
                if at_indices is not None:
                    # incorporate the shuffle into the at_indices
                    at_indices = at_indices[shuffle_indices]
                else:
                    # simply forget the shuffle, since the order doesn't matter
                    pass
        except NotImplementedError:
            pass

        return cls(vectorized_children, at_indices=at_indices)


def reduction_vectorize(cls, vectorized_cls, *instances):
    if len(instances) == 1:
        children = instances[0].operands
        at_indices = None
    else:
        children = []
        lengths = []
        for instance in instances:
            if isinstance(instance, cls):
                # eat the instance, since we are incorporating it into a
                # vectorized reduction
                children += instance.operands
                lengths.append(len(instance.operands))
            else:
                # don't eat the instance. implicitly insert a reduction of 1
                # element (i.e. an identity operation).
                children.append(instance)
                lengths.append(1)

        at_indices = []
        for i, length in enumerate(lengths):
            at_indices += [i] * length
        at_indices = np.array(at_indices)

    vectorized_children = vectorize(*children)

    # Now lift a shuffle operation if it occurs amongst descendants, as we can
    # optimize them away.
    if isinstance(vectorized_children, Expression):
        try:
            (vectorized_children,
             shuffle_indices) = vectorized_children.lift_shuffle()
            if shuffle_indices is not None:
                if at_indices is not None:
                    # incorporate the shuffle into the at_indices
                    at_indices = at_indices[shuffle_indices]
                else:
                    # simply forget the shuffle, since the order doesn't matter
                    pass
        except NotImplementedError:
            pass

    return vectorized_cls(vectorized_children, at_indices=at_indices)


# def vectorize_operands_separately(cls, *instances):
#     operand_lists = [[] for _ in range(len(instances[0].operands))]

#     for instance in instances:
#         if isinstance(instance, cls):
#             # eat the instance, since we are incorporating it into a
#             # vectorized operation
#             operands = instance.operands
#         else:
#             # don't eat the instance, pass it through an identity function.
#             operands = cls.identity_args(instance)

#         for operand_list, operand in zip(operand_lists, operands):
#             operand_list.append(operand)

#     vectorized_operands = tuple(vectorize(*operand_list)
#                                 for operand_list in operand_lists)

#     return vectorized_operands


class Product(Expression):
    def __init__(self, *operands):
        operands2 = []
        for op in operands:
            if isinstance(op, Product):
                operands2.extend(op.operands)
            else:
                operands2.append(op)
        operands = tuple(operands2)

        shapes = tree_map(to_shape, operands)
        shape, *rest = shapes

        if not all(s == ArrayShape([]) for s in tree_leaves(shape)):
            raise ValueError(f"Expected all scalars, got {shape}")

        if not all(s == shape for s in rest):
            raise ValueError(f"Expected same shape for all operands, got {shapes}")

        super().__init__(*operands, return_shape=shape)

    def compute(self, symbols, *operands):
        # the purpose of the Product expression isn't to be fast, it's to prepare
        # for optimization to the VectorizedProduct and VectorizedProductAt.
        prod, *remaining = operands
        for v in remaining:
            prod = prod * v
        return prod

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        return reduction_vectorize(cls, VectorizedProduct, *instances)


class VectorizedProduct(Expression):
    def __init__(self, vector, at_indices=None):
        self.at_indices = (np.array(at_indices)
                           if at_indices is not None
                           else None)
        data_shape = ArrayShape((self.at_indices.max() + 1,)
                                if at_indices is not None
                                else ())
        return_shape = tree_map(lambda _: data_shape, vector)
        super().__init__(vector, return_shape=return_shape)

    def compute(self, symbols, arr):
        if self.at_indices is not None:
            out = np.ones((*arr.shape[:-1], self.data_shape[-1]),
                          dtype=arr.dtype)
            np.multiply.at(out, (..., self.at_indices), arr)
            return out
        else:
            return arr.prod(axis=-1)

    def sexpr(self):
        ret = super().sexpr()
        if self.at_indices is not None:
            ret = ret + (self.at_indices.tolist(),)
        return ret

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        return vectorized_reduction_vectorize(cls, *instances)

    def lift_shuffle(self):
        # assume any descendant shuffles have already been lifted when this
        # class called vectorize.
        return self, None


class Divide(Expression):
    """
    The numerator is a pytree, while the denominator is always either a
    single scalar or a vector. Vectorization of multiple divides always leads to
    a vector denominator.
    """
    def __init__(self, operands):
        numerator_shape, denominator_shape = tree_map(to_shape, operands)

        if not is_leaf(denominator_shape):
            raise ValueError("Second operator must be leaf, not general pytree"
                             f", got {denominator_shape}")

        super().__init__(operands, return_shape=numerator_shape)

    def compute(self, symbols, operands):
        numerator, denominator = operands
        return tree_map(lambda num: num / denominator,
                        numerator)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        children = []

        for instance in instances:
            if isinstance(instance, cls):
                # eat the instance, since we are incorporating it into a
                # vectorized operation
                children.append(instance.operands[0])
            else:
                # don't eat the instance, pass it through an identity function.
                children.append((instance, np.ones(instance.data_shape)))

        return Divide(vectorize(*children))

    @classmethod
    def identity_args(cls, expr):
        return (expr, 1.0)


class Power(Expression):
    def __init__(self, operands):
        base_shape, exponent_shape = tree_map(to_shape, operands)
        super().__init__(operands, return_shape=base_shape)

    def compute(self, symbols, operands):
        base, exponent = operands
        return base ** exponent

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        children = []
        for instance in instances:
            if isinstance(instance, cls):
                # eat the instance, since we are incorporating it into a
                # vectorized operation
                children.append(instance.operands[0])
            else:
                # don't eat the instance, pass it through an identity function.
                # convention: 1.0 on the lhs.
                # TODO do something cleaner than shape.a
                children.append((instance,
                                 tree_map(lambda shape: np.ones(shape.a),
                                          instance.return_shape)))

        return Power(vectorize(*children))



class Stack(Expression):
    def __init__(self, *operands):
        shapes = tree_map(to_shape, operands)
        shape, *rest = shapes

        if not all(s == ArrayShape([]) for s in tree_leaves(shape)):
            raise ValueError(f"Expected all scalars, got {shape}")

        if not all(s == shape for s in rest):
            raise ValueError(f"Expected same shape for all operands, got {shapes}")

        return_data_shape = ArrayShape([len(operands)])
        return_shape = tree_map(lambda _: return_data_shape, shape)

        super().__init__(*operands, return_shape=return_shape)

    def compute(self, symbols, *operands):
        return np.stack(operands)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        if len(instances) == 1:
            instance = instances[0]
            children = instance.operands
        else:
            # TODO
            raise CannotVectorize()

        vectorized_children = None
        for child in children:
            if isinstance(child, Expression):
                try:
                    vectorized_children = type(child).vectorize_level(*children)
                    break
                except CannotVectorize:
                    pass

        if vectorized_children is None:
            vectorized_children = Stack(*children)

        return vectorized_children

    def lift_shuffle(self):
        # we're creating an array at this point, so it's a stop point for
        # searching for a shuffle operation.
        return self, None


class Concatenate(Expression):
    """
    Programmer's note: this takes multiple operands, not a tuple of
    operands, to indicate what would happen during vectorization. During
    vectorization there is no guarantee that the tuple would preserved, in fact
    it generally wouldn't be. In vexpr's design, vectorization never modifies a
    pytree's structure.
    """
    def __init__(self, *operands):
        operands_ = []
        for v in operands:
            if isinstance(v, Concatenate):
                operands_ += v.operands
            else:
                operands_.append(v)
        operands = tuple(operands_)

        operand_shapes = tree_map(to_shape, operands)
        operand_shapes_leaves, treedefs = zip(*[tree_flatten(operand_shape)
                                                for operand_shape in operand_shapes])
        treedef, *rest = treedefs
        if not all(td == treedef for td in rest):
            raise ValueError(f"Expected same tree shapes, got {treedefs}")

        length = 0
        for operand_leaves in operand_shapes_leaves:
            data_shape, *rest = operand_leaves
            if not all(s == data_shape for s in rest):
                raise ValueError(f"Expected same shape, got {operand_leaves}")
            length += data_shape[-1]

        return_data_shape = ArrayShape([length])
        return_shape = tree_unflatten(treedef,
                                      [return_data_shape
                                       for _ in operand_shapes_leaves[0]])

        super().__init__(*operands, return_shape=return_shape)

    def compute(self, symbols, *operands):
        return np.concatenate(operands)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        # example scenario: i have two vectorized CDists and I want to combine
        # them and vectorize again.

        raise NotImplementedError()


class Shuffle(Expression):
    def __init__(self, vector, indices):
        data_shape = ArrayShape([len(indices)])
        return_shape = tree_map(lambda _: data_shape,
                                vector)

        super().__init__(vector, return_shape=return_shape)
        self.indices = np.array(indices)

    def compute(self, symbols, arr):
        return arr[..., self.indices]

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        raise NotImplementedError()

    def lift_shuffle(self):
        # remove self from the tree
        return self.operands[0], self.indices

    def sexpr(self) -> tuple:
        return super().sexpr() + (self.indices.tolist(),)


def shuffle(vector, indices):
    if isinstance(vector, Stack):
        return Stack(*(vector.operands[i] for i in indices))
    else:
        return Shuffle(vector, indices)


class Symbol(Expression):
    def __init__(self, name, data_shape=ArrayShape(())):
        return_shape = tree_map(lambda _: data_shape,
                                name)
        super().__init__(return_shape=return_shape)
        self.name = name

    def _partial_evaluate(self, symbols):
        if self.name in symbols:
            return self.compute(symbols)
        else:
            return self

    def compute(self, symbols):
        return symbols[self.name]

    def sexpr(self):
        return ("Symbol", self.name)

    def __getitem__(self, indices):
        # Rather than creating a separate Select node, fuse it with the Symbol
        # node so that optimizations can capture the scenario where multiple
        # selections come from the same symbol.
        return SelectFromSymbol(self.name, indices)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        # In my use cases, symbols are individual values.
        #
        # Other plausible scenarios:
        # - Sum(Symbol("a") ** 2, Prod(Symbol("b"), Symbol("c")))
        #   -

        return Stack(*instances)


class CDist(Expression):
    def __init__(self, x1x2, metric="euclidean", split_lengths=None):
        """
        x1x2 is either a list/tuple containing (x1, x2), or an expression
        that returns (x1, x2).
        """
        x1_data_shape, x2_data_shape = tree_map(to_shape, x1x2)
        if x1_data_shape != x2_data_shape:
            raise ValueError("Expected same shapes, got "
                             f"{x1_data_shape} and {x2_data_shape}")
        data_shape = ArrayShape((len(split_lengths),)
                                if split_lengths is not None
                                else ())
        return_shape = tree_map(lambda _: data_shape,
                                x1_data_shape)
        super().__init__(x1x2, return_shape=return_shape)
        self.metric = metric
        self.split_lengths = split_lengths

    def compute(self, symbols, x1x2):
        x1, x2 = x1x2
        if self.split_lengths is None:
            return scipy.spatial.distance.cdist(x1, x2, self.metric)
        else:
            # TODO implement vectorized version for numpy
            return np.stack([scipy.spatial.distance.cdist(x1_, x2_, self.metric)
                             for x1_, x2_ in zip(x1.split(self.split_lengths),
                                                 x2.split(self.split_lengths))],
                            axis=-1)

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        if cls == CDist:
            raise ValueError("Don't instantiate CDist directly, use a derived "
                             "class")

        if all(isinstance(instance, cls)
               for instance in instances):
            return_shapes = tree_map(to_shape, [inst.operands[0]
                                                for inst in instances])

            x1_shapes, x2_shapes = list(zip(*return_shapes))
            split_lengths1 = [s[-1] for s in x1_shapes]
            split_lengths2 = [s[-1] for s in x2_shapes]
            if split_lengths1 != split_lengths2:
                raise ValueError("Expected same shape", split_lengths1,
                                 split_lengths2)
            split_lengths = split_lengths1

            vectorized_children = vectorize(*(inst.operands[0]
                                              for inst in instances))
            return cls(vectorized_children,
                       split_lengths=split_lengths)
        else:
            if not allow_partial:
                raise CannotVectorize()

            offset = 0
            these = []
            these_target_indices = []
            those = []
            those_target_indices = []
            for instance in instances:
                length = (instance.data_shape[-1] if instance.ndim > 0 else 1)
                indices = list(range(offset, offset + length))
                if isinstance(instance, cls):
                    these.append(instance)
                    these_target_indices += indices
                else:
                    those.append(instance)
                    those_target_indices += indices
                offset += length

            these_vectorized = cls.vectorize_level(*these)
            those_vectorized = type(those[0]).vectorize_level(*those)

            those_vectorized, shuffle_update = those_vectorized.lift_shuffle()
            if shuffle_update is not None:
                those_target_indices = those_target_indices[shuffle_update]
            shuffle_indices = invert_shuffle(these_target_indices
                                             + those_target_indices)
            vectorized = Concatenate(these_vectorized, those_vectorized)
            vectorized = shuffle(vectorized, shuffle_indices)
            return vectorized

    def lift_shuffle(self):
        return self, None


class CDistEuclidean(CDist):
    def __init__(self, operands, split_lengths=None):
        super().__init__(operands, metric="euclidean",
                         split_lengths=split_lengths)


class CDistCityBlock(CDist):
    def __init__(self, operands, split_lengths=None):
        super().__init__(operands, metric="cityblock",
                         split_lengths=split_lengths)


class SelectFromSymbol(Expression):
    def __init__(self, name, indices):
        shape = ArrayShape([len(indices)])
        return_shape = tree_map(lambda _: shape,
                                name)
        super().__init__(return_shape=return_shape)
        self.name = name
        self.indices = np.array(indices)

    def _partial_evaluate(self, symbols):
        if self.name in symbols:
            return self.compute(symbols)
        else:
            return self

    def compute(self, symbols):
        symbol = symbols[self.name]
        return symbol[..., self.indices]

    def clone(self):
        return type(self)(self.name, self.indices)

    def sexpr(self):
        return ("SelectFromSymbol", self.name, self.indices.tolist())

    def lift_shuffle(self):
        return self, None

    @classmethod
    def vectorize_level(cls, *instances, allow_partial=True):
        if all(isinstance(instance, SelectFromSymbol) for instance in instances):
            name = instances[0].name
            if all(instance.name == name for instance in instances[1:]):
                indices = np.concatenate([instance.indices for instance in instances])
                return SelectFromSymbol(name, indices)
        else:
            raise CannotVectorize


class ArrayMean(Expression):
    def __init__(self, arr):
        shapes = tree_map(to_shape, arr)
        shape, *rest = shapes
        return_data_shape = ArrayShape([])
        return_shape = tree_map(lambda _: return_data_shape, shape)
        super().__init__(arr, return_shape=return_shape)

    def compute(self, symbols, array):
        return array.mean()
