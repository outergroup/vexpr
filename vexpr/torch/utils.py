import collections

import torch

import vexpr as vp
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.custom.torch as vctorch
import vexpr.vectorization as v
from vexpr.custom.torch.utils import maybe_shuffle


def identity(x): return x


def torch_stack_shape(initial_shape, num_elements, dim=0):
    if dim < 0:
        dim += len(initial_shape) + 1
    return (initial_shape[:dim]
            + type(initial_shape)([num_elements,])
            + initial_shape[dim:])


def torch_stack_shape2(shapes, dim=0):
    assert all(shape == shapes[0] for shape in shapes)
    return torch_stack_shape(shapes[0], len(shapes), dim)


def torch_cat_shape(child_shapes, dim=0):
    if dim < 0:
        dim += len(child_shapes[0])
    return (child_shapes[0][:dim]
            + (sum(shape[dim] for shape in child_shapes),)
            + child_shapes[0][dim + 1:])


def canonical_axis(axis, ndim):
    if axis is None:
        return None
    elif axis < 0:
        return axis + ndim
    else:
        return axis


def canonical_stack_dim(dim, ndim):
    if dim is None:
        return None
    elif dim < 0:
        return dim + ndim + 1
    else:
        return dim


def maybe_lift_scatter(input, dim, index, src, batch_shape=()):
    if (isinstance(src, vp.Vexpr)
        and src.op == p.scatter_p
        and src.args[1] == dim
        and isinstance(input, vp.Vexpr)
        and isinstance(src.args[0], vp.Vexpr)
        and input.op == src.args[0].op
        and input.op in (p.zeros_p, p.ones_p)):

        # Include only the indices that actually correspond to values, not to
        # filler zeros and ones.

        give_up = False
        inner_index = src.args[2]
        if (len(batch_shape) > 0
            and isinstance(inner_index, vp.Vexpr)
            and inner_index.op == p.expand_p
            and isinstance(inner_index.args[0], torch.Tensor)):
            inner_index = inner_index.args[0].view(-1)
        elif isinstance(inner_index, torch.Tensor):
            pass
        else:
            give_up = True

        if not give_up:
            index = index[inner_index]
            src = src.args[3]

    if len(batch_shape) > 0:
        num_indices = index.shape[-1]
        index = index.view((1,) * len(batch_shape) + (num_indices,))
        shape = batch_shape + (num_indices,)
        index = v.with_return_shape(
            vtorch.expand(index, shape),
            shape)

    return v.with_return_shape(
        vtorch.scatter(input, dim, index, src),
        v.shape(input))


def invert_shuffle(indices):
    inverted_indices = torch.zeros_like(torch.as_tensor(indices))
    inverted_indices[indices] = torch.arange(len(indices))
    return inverted_indices


def stack_remainder_then_combine(applicable, remainder, applicable_indices,
                                 remainder_indices, transform=identity,
                                 **stack_kwargs):
    if len(remainder) == 0:
        return applicable

    remainder = v.with_return_shape(
        vtorch.stack(remainder, **stack_kwargs),
        torch_stack_shape2([v.shape(r_expr)
                            for r_expr in remainder],
                           **stack_kwargs)
    )
    if transform is not None:
        remainder = transform(remainder)

    result_shape = torch_cat_shape([v.shape(applicable), v.shape(remainder)],
                                   **stack_kwargs)

    result = v.with_return_shape(
        vtorch.cat([applicable, remainder], **stack_kwargs),
        result_shape)

    return maybe_shuffle(result,
                         invert_shuffle(applicable_indices + remainder_indices),
                         **stack_kwargs)


def cat_remainder_then_combine(applicable, remainder, applicable_indices,
                               remainder_indices, transform=identity,
                               dim=0):
    if len(remainder) == 0:
        return applicable

    # suppose we've called pushthrough on a unary op that has already gotten the
    # treatment. now my "applicable" is matern, the exp and exp2 get cat'd. i
    # push a transform on this remainder, and nothing happens. no change. i cat
    # the result with the applicable, possibly factoring out the inner cat. now
    # i have something equal to before... but how do i know? and we hit these
    # weird questions, like maybe it *will* change if transfrom=None, but then
    # it goes back to what it was after the transform.

    # i'm beginning to think that comparison of vexprs is the solution. remove
    # change-tracking. 👍

    remainder = transform(
        v.with_return_shape(
            vtorch.cat(remainder, dim=dim),
            torch_cat_shape([v.shape(r_expr)
                             for r_expr in remainder],
                            dim=dim)
        )
    )

    result_shape = torch_cat_shape([v.shape(applicable), v.shape(remainder)],
                                   dim=dim)
    result = v.with_return_shape(vtorch.cat([applicable, remainder],
                                            dim=dim),
                                 result_shape)

    return maybe_shuffle(result,
                         invert_shuffle(applicable_indices + remainder_indices),
                         dim,
                         transform)


def push_stack_through_reduction(reduction_p, parallel_reduction, fill_value,
                                 expr, transform=identity, allow_partial=True):
    assert expr.op == p.stack_p

    exprs_to_stack = expr.args[0]
    all_reduction_operands = []
    lengths = []

    stack_dim = expr.kwargs.get("dim", 0)

    for i, child_expr in enumerate(exprs_to_stack):
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == reduction_p:

            r_axis = child_expr.kwargs.get("dim", None)
            if r_axis is None:
                raise NotImplementedError()

            r_arg0 = child_expr.args[0]
            if isinstance(r_arg0, vp.Vexpr):
                num_operands = v.shape(r_arg0)[r_axis]
            else:
                num_operands = len(r_arg0)
                # treat array_like as an implicit stack.
                r_arg0 = v.with_return_shape(vtorch.stack(r_arg0),
                                             dict(return_shape=torch_stack_shape(
                                                 v.shape(r_arg0[0]), num_operands)))

            if r_axis != stack_dim:
                r_arg0 = vtorch.moveaxis(r_arg0, r_axis, stack_dim)
                r_arg0 = transform(r_arg0)
                r_axis = stack_dim

            # Incorporate child_expr's computation into a vectorized reduction.
            all_reduction_operands.append(r_arg0)
            lengths.append(num_operands)
        else:
            # Pass child_expr through. Implement Identity as a reduction of 1
            # element.
            child_expr = v.with_return_shape(
                vtorch.stack([child_expr], dim=stack_dim),
                torch_stack_shape(v.shape(child_expr), 1, stack_dim)
            )
            child_expr = transform(child_expr)
            all_reduction_operands.append(child_expr)
            lengths.append(1)

    all_reduction_operands = transform(
        v.with_return_shape(
            vtorch.cat(all_reduction_operands,
                       dim=stack_dim),
            torch_cat_shape([v.shape(child_expr)
                             for child_expr in all_reduction_operands],
                            stack_dim)
        )
    )

    groups = list(collections.Counter(lengths).items())

    pre_shuffle_indices = []
    post_shuffle_indices_inverted = []
    for length, count in groups:
        base = 0
        for i, length2 in enumerate(lengths):
            if length2 == length:
                pre_shuffle_indices += list(range(base, base + length))
                post_shuffle_indices_inverted.append(i)
            base += length2

    pre_shuffle_indices = torch.tensor(pre_shuffle_indices)
    post_shuffle_indices = invert_shuffle(post_shuffle_indices_inverted)

    all_reduction_operands = maybe_shuffle(all_reduction_operands,
                                           pre_shuffle_indices,
                                           stack_dim,
                                           transform)

    result = parallel_reduction(all_reduction_operands, groups=groups,
                                dim=stack_dim)

    child_shape = next(v.shape(expr) for expr in exprs_to_stack
                       if isinstance(expr, vp.Vexpr))
    result_shape = torch_stack_shape(child_shape,
                                     len(exprs_to_stack),
                                     dim=stack_dim)
    return maybe_shuffle(v.with_return_shape(result, result_shape),
                         post_shuffle_indices,
                         stack_dim,
                         transform)
