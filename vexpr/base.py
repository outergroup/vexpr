import abc
from pprint import pformat

from jax.tree_util import tree_map, tree_leaves


class Expression(abc.ABC):
    def __init__(self, *operands, return_shape=None):
        """
        An expression has a list of operands, each of which may be a value
        or an expression.
        """
        self.operands = operands

        if return_shape is None:
            raise ValueError("Must specify return_shape")
        self.return_shape = return_shape

        shape, *rest = tree_leaves(return_shape)
        if not all(s == shape for s in rest):
            # TODO it is weird to enforce this in the base class. better to have
            # other classes infer data_shape / ndim when needed and throw an
            # error if the input shape is invalid. some classes like Multiply
            # can support different shapes in different leafs just fine, no
            # reason to shut down that entire set of use cases.
            raise ValueError(f"All shapes must be same, got {return_shape}")
        self.data_shape = shape
        self.ndim = len(shape)


    def _partial_evaluate(self, symbols):
        operands = tree_map(lambda v: (v._partial_evaluate(symbols)
                                       if isinstance(v, Expression)
                                       else v),
                            self.operands)

        ready = all(not isinstance(v, Expression)
                    for v in tree_leaves(operands))
        if ready:
            return self.compute(symbols, *operands)
        else:
            return self.clone(*operands)

    def partial_evaluate(self, symbols):
        ret = self._partial_evaluate(symbols)
        if isinstance(ret, Expression):
            return ret
        else:
            # Always return an expression.
            return Constant(ret)

    def __call__(self, symbols):
        operands = tree_map(lambda v: (v(symbols)
                                       if isinstance(v, Expression)
                                       else v),
                            self.operands)
        return self.compute(symbols, *operands)

    def extra_clone_kwargs(self):
        return {}

    def clone(self, *operands):
        return type(self)(*operands, **self.extra_clone_kwargs())

    def apply(self, f):
        """
        Computes f(expression) for every expression in the tree.

        f either returns expression unchanged, or returns a new modified expression.
        """
        operands = tree_map(lambda v: (v._apply(f)
                                       if isinstance(v, Expression)
                                       else v),
                            self.operands)

        return f(self.clone(*operands))

    @abc.abstractmethod
    def compute(self, symbols, operands):
        raise NotImplementedError()

    def sexpr(self) -> tuple:
        append = tree_map(lambda v: (v.sexpr()
                                     if isinstance(v, Expression)
                                     else v),
                          self.operands)
        return ((type(self).__name__,)
                + append
                + tuple(self.extra_clone_kwargs().values()))

    def extra_repr(self):
        return ", ".join([f"{k}={repr(v)}"
                          for k, v in self.extra_clone_kwargs().items()])

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        """
        (adapted from pytorch)
        """
        child_lines = []
        for operand in self.operands:
            operand_str = pformat(operand) + ","
            operand_str = _addindent(operand_str, 2)
            child_lines.append(operand_str)

        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        lines = child_lines + extra_lines

        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            return False
        return self.sexpr() == other.sexpr()

    @classmethod
    def vectorize_instances(cls, *instances):
        raise CannotVectorize


class CannotVectorize(Exception): pass


class Constant(Expression):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def clone(self):
        return type(self)(self.v)

    def compute(self, symbols):
        return self.v


class ArrayShape:
    def __init__(self, a):
        self.a = tuple(a)

    def __getitem__(self, *args, **kwargs):
        return self.a.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.a)

    def __eq__(self, other):
        return self.a == other.a

    def __repr__(self):
        return self.__class__.__name__ + f"({self.a})"


def _addindent(s_, numSpaces):
    """
    (adapted from pytorch)
    """
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s
