from functools import partial

import numpy as np

import vexpr as vp
import vexpr.core as core
import vexpr.custom.numpy.primitives as cp
import vexpr.numpy as vnp
import vexpr.numpy.impls as impls
import vexpr.numpy.primitives as p
import vexpr.vectorization as v
from vexpr.numpy.utils import (
    np_stack_shape,
    np_stack_shape2,
    np_concatenate_shape,
)


PRIORITIZED_OPS = set([
    p.stack_p, p.concatenate_p,
    p.sum_p, p.prod_p, p.add_at_p, p.multiply_at_p,
    core.operator_add_p, core.operator_mul_p, core.operator_truediv_p,
    core.operator_matmul_p,
    cp.shuffle_p,
])

# It's easy to accidentally include a p.functionname rather than a
# p.functionname_p in PRIORITIZED_OPS.
assert all(isinstance(op, core.Primitive) for op in PRIORITIZED_OPS)


def identity(x): return x

def pushthrough_return_self(expr, transform=identity, allow_partial=True):
    return expr


def stack_pushthrough(expr, transform=identity):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))
    vexpr_ops = ([op for op in vexpr_ops if op in PRIORITIZED_OPS]
                 + [op for op in vexpr_ops if op not in PRIORITIZED_OPS])

    if len(vexpr_ops) > 0:
        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return push_stack_through_op(expr, op,
                                                 allow_partial=allow_partial,
                                                 transform=transform)
                except v.CannotVectorize:
                    pass

    raise v.CannotVectorize


def push_stack_through_op(expr, op, transform=identity, allow_partial=True):
    impl = impls.push_stack_through_op.get(op, None)
    if impl is None:
        print("No stack pushthrough support for", op)
        raise v.CannotVectorize
    else:
        return impl(expr, transform, allow_partial)


def concatenate_pushthrough(expr, transform=identity):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op
                                   for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))

    vexpr_ops = [op for op in vexpr_ops if op != core.symbol_p]
    vexpr_ops = ([op for op in vexpr_ops if op in PRIORITIZED_OPS]
                 + [op for op in vexpr_ops if op not in PRIORITIZED_OPS])

    if len(vexpr_ops) > 0:
        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return push_concatenate_through_op(
                        expr, op, allow_partial=allow_partial,
                        transform=transform)
                except v.CannotVectorize:
                    pass
    elif len(expr.args[0]) == 1:
        # If concatenate-ing one item, just return it.
        return transform(expr.args[0][0])

    raise v.CannotVectorize


def push_concatenate_through_op(expr, op, transform=identity, allow_partial=True):
    impl = impls.push_concatenate_through_op.get(op, None)
    if impl is None:
        print("No concatenate pushthrough support for", op)
        raise v.CannotVectorize
    else:
        return impl(expr, transform, allow_partial)


def moveaxis_pushthrough(expr, transform=identity):
    op = expr.args[0].op
    impl = impls.push_moveaxis_through_op.get(op, None)
    if impl is None:
        print("No moveaxis pushthrough support for", op)
        return expr
    else:
        return impl(expr, transform)



def canonical_axis(axis, ndim):
    if axis is None:
        return None
    elif axis < 0:
        return axis + ndim
    else:
        return axis


def push_concatenate_through_getitem(expr, transform=identity, allow_partial=True):
    assert expr.op == p.concatenate_p

    # Quick hacky assumptions: every child expr is selecting from a symbol, and
    # they are the same symbol.
    # TODO: if any child is not a getitem, then push through all the getitems,
    # concatenate with the non-getitems, and insert a shuffle after everything
    # to correct the order.
    if not all(sub_expr.op == core.operator_getitem_p
               for sub_expr in expr.args[0]):
        raise NotImplementedError()

    select_targets = [sub_expr.args[0]
                      for sub_expr in expr.args[0]]
    select_target = select_targets[0]
    if not all(target.op == core.symbol_p for target in select_targets):
        raise NotImplementedError()
    if not all(target == select_target
               for target in select_targets):
        raise NotImplementedError()

    selections = [sub_expr.args[1]
                  for sub_expr in expr.args[0]]

    all_select_axes = []
    all_select_indices = []

    for selection in selections:
        if isinstance(selection, tuple):
            # parse selection tuple to determine which axis contains a list of
            # indices
            select_axis = None
            select_indices = None
            for i, sel in enumerate(selection):
                if isinstance(sel, slice):
                    continue
                elif sel is Ellipsis:
                    break
                else:
                    if select_axis is not None:
                        raise NotImplementedError(
                            "Can't handle indexing on multiple axes yet.")
                    select_axis = i
                    select_indices = sel

            if select_axis is None:
                # an ellipsis was found, count from the back
                for i, sel in enumerate(reversed(selection)):
                    if isinstance(sel, slice):
                        continue
                    elif sel is Ellipsis:
                        if select_axis is None:
                            raise ValueError(
                                "Found multiple ellipses, or found no indices",
                                selection)
                    else:
                        if select_axis is not None:
                            raise NotImplementedError(
                                "Can't handle indexing on multiple axes yet.")
                        select_axis = -(i + 1)
                        select_indices = sel

        else:
            select_axis = 0
            select_indices = selection

        all_select_axes.append(select_axis)
        all_select_indices.append(select_indices)

    if not all(select_axis == all_select_axes[0] for select_axis in all_select_axes):
        raise NotImplementedError()
    select_axis = all_select_axes[0]

    axis = expr.kwargs.get("axis", 0)
    ndim = len(v.shape(select_target))
    if canonical_axis(axis, ndim) != canonical_axis(select_axis, ndim):
        raise NotImplementedError()
    else:
        # prefer the version that the user specified
        axis = select_axis

    indices = np.concatenate(all_select_indices)
    if axis == 0:
        new_selection = indices
    elif axis < 0:
        new_selection = (Ellipsis, indices,) + (slice(None),) * ((-axis) - 1)
    else:
        new_selection = ((slice(None),) * axis) + (indices,)

    return v.with_return_shape(
        select_target[new_selection],
        np_concatenate_shape([v.shape(child_expr)
                              for child_expr in expr.args[0]],
                             axis))


def push_moveaxis_through_stack(expr, transform=identity):
    assert expr.op == p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op == p.stack_p

    source = expr.args[1]
    stack_expr = expr.args[0]
    stack_axis = stack_expr.kwargs.get("axis", 0)

    orig_stack_shape = v.shape(stack_expr)
    ndim = len(orig_stack_shape)
    if source < 0: source += ndim
    if stack_axis < 0: stack_axis += ndim

    if stack_axis == source:
        dest = expr.args[2]
        new_shape = list(orig_stack_shape)
        tmp = new_shape[source]
        new_shape[source] = new_shape[dest]
        new_shape[dest] = tmp
        expr = transform(
            v.with_return_shape(vnp.stack(stack_expr.args[0], axis=dest),
                                tuple(new_shape))
        )

    return expr


def push_moveaxis_through_sum(expr, transform=identity):
    assert expr.op == p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op == p.sum_p

    sum_expr = expr.args[0]
    sum_arg0 = sum_expr.args[0]
    if not isinstance(sum_arg0, vp.Vexpr):
        sum_arg0 = v.with_return_shape(vnp.stack(sum_arg0),
                                       np_stack_shape(
                                           v.shape(sum_expr.args[0]),
                                           len(sum_arg0), axis=0))

    source = expr.args[1]
    dest = expr.args[2]
    sum_arg0 = transform(vnp.moveaxis(sum_arg0, source, dest))

    sum_axis = sum_expr.kwargs.get("axis", None)

    trace = list(range(len(v.shape(sum_expr))))
    trace.insert(dest, trace.pop(source))
    new_axis = trace.index(sum_axis)

    return transform(
        v.with_return_shape(
            vnp.sum(sum_arg0, axis=new_axis),
            v.shape(expr))
    )


def push_concatenate_through_stack(expr, transform=identity, allow_partial=True):
    assert expr.op == p.concatenate_p

    # initial hack: only do anything if everything is a stack.
    # thus, this eliminates the cat.
    if not all(isinstance(arg, vp.Vexpr) and arg.op == p.stack_p
               for arg in expr.args[0]):
        return expr

    axis = expr.kwargs.get("axis", 0)

    # (when i remove the hack, the result is going to be
    # shuffle(cat([stack, non-stack-exprs])), or remove the outer
    # cat of there are no non-stack-exprs.)

    all_stacked_vexprs = []
    for child_expr in expr.args[0]:
        assert child_expr.kwargs.get("axis", 0) == axis
        all_stacked_vexprs.extend(child_expr.args[0])

    assert all(child_expr.kwargs == expr.args[0][0].kwargs
               for child_expr in expr.args[0][1:])
    stack_kwargs = expr.args[0][0].kwargs

    return transform(
        v.with_return_shape(
            vnp.stack(all_stacked_vexprs, **stack_kwargs),
            np_stack_shape2([v.shape(d) for d in all_stacked_vexprs],
                            **stack_kwargs)
        )
    )


def stack_detect_shape(children):
    return v.with_return_shape(
        vnp.stack(children),
        np_stack_shape2([v.shape(child)
                         for child in children])
    )


def push_stack_through_reduction(reduction_p, parallel_reduction, expr,
                                 transform=identity, allow_partial=True):
    assert expr.op == p.stack_p

    exprs_to_stack = expr.args[0]
    all_reduction_operands = []
    at_indices = []

    stack_axis = expr.kwargs.get("axis", 0)

    for i, child_expr in enumerate(exprs_to_stack):
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == reduction_p:

            r_axis = child_expr.kwargs.get("axis", None)
            if r_axis is None:
                raise NotImplementedError()

            r_arg0 = child_expr.args[0]
            if isinstance(r_arg0, vp.Vexpr):
                num_operands = v.shape(r_arg0)[r_axis]
            else:
                num_operands = len(r_arg0)
                # treat array_like as an implicit stack.
                r_arg0 = v.with_return_shape(vnp.stack(r_arg0),
                                             dict(return_shape=np_stack_shape(
                                                 v.shape(r_arg0[0]), num_operands)))

            if r_axis != stack_axis:
                prev_shape = v.shape(r_arg0)
                r_arg0 = vnp.moveaxis(r_arg0, r_axis, stack_axis)
                r_arg0 = transform(r_arg0)
                r_axis = stack_axis

            # Incorporate child_expr's computation into a vectorized reduction.
            all_reduction_operands.append(r_arg0)
            at_indices += [i] * num_operands
        else:
            # Pass child_expr through. Implement Identity as a reduction of 1
            # element.
            child_expr = v.with_return_shape(
                vnp.stack([child_expr], **expr.kwargs),
                np_stack_shape(v.shape(child_expr), 1, stack_axis)
            )
            child_expr = transform(child_expr)
            all_reduction_operands.append(child_expr)
            at_indices.append(i)

    all_reduction_operands = transform(
        v.with_return_shape(
            vnp.concatenate(all_reduction_operands, **expr.kwargs),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in all_reduction_operands],
                                 stack_axis)
        )
    )

    at_indices = np.array(at_indices)

    if stack_axis == 0:
        pass
    elif stack_axis < 0:
        at_indices = (Ellipsis, at_indices,) + (slice(None),) * ((-stack_axis) - 1)
    else:
        at_indices = ((slice(None),) * stack_axis) + (at_indices,)

    child_shape = next(v.shape(expr) for expr in exprs_to_stack
                       if isinstance(expr, vp.Vexpr))

    result_shape = np_stack_shape(child_shape, len(exprs_to_stack),
                                  axis=stack_axis)

    return parallel_reduction(result_shape,
                              at_indices,
                              all_reduction_operands)


def parallel_sum(result_shape, indices, source):
    return v.with_return_shape(
        vnp.add_at(
            v.with_return_shape(vnp.zeros(result_shape),
                                result_shape),
            indices, source),
        result_shape)


def parallel_prod(result_shape, indices, source):
    return v.with_return_shape(
        vnp.multiply_at(
            v.with_return_shape(vnp.ones(result_shape),
                                result_shape),
            indices,
            source),
        result_shape)


def push_concatenate_through_truediv(expr, transform=identity, allow_partial=True):
    assert expr.op == p.concatenate_p

    # initial hack: assume all args are truediv
    #
    # TODO: add similar treatment of "*_through_mul", dividing by 1.0. It's
    # easier than mul, since it's unambiguous; we can always put 1.0 in the
    # denominator.
    assert all(isinstance(arg, vp.Vexpr) and arg.op == core.operator_truediv_p
               for arg in expr.args[0])

    num = []
    den = []
    for child_expr in expr.args[0]:
        num.append(child_expr.args[0])
        den.append(child_expr.args[1])

    axis = expr.kwargs.get("axis", 0)
    num_shape = np_concatenate_shape([v.shape(child_expr)
                                      for child_expr in num],
                                     axis=axis)
    den_shape = np_concatenate_shape([v.shape(child_expr)
                                      for child_expr in den],
                                     axis=axis)
    num = transform(
        v.with_return_shape(vnp.concatenate(num, axis=axis),
                            num_shape)
    )
    den = transform(
        v.with_return_shape(vnp.concatenate(den, axis=axis),
                            den_shape)
    )

    return transform(
        v.with_return_shape(
            num / den,
            np.broadcast_shapes(v.shape(num), v.shape(den)))
    )

def push_stack_through_mul(expr, transform=identity, allow_partial=True):
    assert expr.op == p.stack_p

    left = []
    right = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == core.operator_mul_p:
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
        else:
            # TODO: how to choose side? for example, some users do w * matrix,
            # while others do matrix * w. Here I want to append the 1 to the w.
            # It seems the way to do this is to check which symbols each
            # child_expr is derived from, then when we're doing an identity, use
            # the symbols *this* child_expr is derived from. if they aren't
            # derived from the same symbols, then there's no benefit to choosing
            # correctly.
            #
            # TODO: what if we're multiplying matrices with matrices? sometimes
            # we need to instead append a matrix of ones, etc.
            left.append(1.0)
            right.append(child_expr)

    kwargs = {}
    if "axis" in expr.kwargs:
        kwargs["axis"] = expr.kwargs["axis"]

    # decide whether ones need to be inserted into the shape to get proper
    # broadcasting
    left_shapes = [v.shape(child_expr) for child_expr in left]
    left_shape_before_stack = left_shapes[0]
    assert all(left_shape_before_stack == shape for shape in left_shapes)
    right_shapes = [v.shape(child_expr) for child_expr in right]
    right_shape_before_stack = right_shapes[0]
    assert all(right_shape_before_stack == shape for shape in right_shapes)
    discrepancy = len(right_shape_before_stack) - len(left_shape_before_stack)
    if discrepancy == 0:
        implicit_left_shape = left_shape_before_stack
        implicit_right_shape = right_shape_before_stack
    elif discrepancy > 0:
        implicit_left_shape = left_shape_before_stack + (1,) * discrepancy
        implicit_right_shape = right_shape_before_stack
    else:
        implicit_left_shape = left_shape_before_stack
        implicit_right_shape = right_shape_before_stack + (1,) * -discrepancy

    axis = expr.kwargs.get("axis", 0)
    actual_left_shape = np_stack_shape(left_shape_before_stack,
                                       len(left), axis)
    actual_right_shape = np_stack_shape(right_shape_before_stack,
                                        len(right), axis)
    implicit_left_shape = np_stack_shape(implicit_left_shape,
                                         len(left), axis)
    implicit_right_shape = np_stack_shape(implicit_right_shape,
                                          len(right), axis)

    implicit_neg_axis = (axis
                         if axis < 0
                         else -len(implicit_left_shape) + axis)

    left = transform(
        v.with_return_shape(
            vnp.stack(left, **kwargs),
            actual_left_shape))
    if implicit_left_shape[implicit_neg_axis:] \
       != actual_left_shape[implicit_neg_axis:]:
        new_shape = implicit_left_shape[implicit_neg_axis:]
        left = v.with_return_shape(
            vnp.reshape(left, new_shape),
            new_shape)

    right = transform(
        v.with_return_shape(
            vnp.stack(right, **kwargs),
            actual_right_shape))
    if implicit_right_shape[implicit_neg_axis:] \
       != actual_right_shape[implicit_neg_axis:]:
        new_shape = implicit_right_shape[implicit_neg_axis:]
        right = v.with_return_shape(
            vnp.reshape(right, new_shape),
            new_shape)

    return transform(
        v.with_return_shape(
            left * right,
            np.broadcast_shapes(v.shape(left), v.shape(right))
        )
    )


def push_concatenate_through_mul(expr, transform=identity, allow_partial=True):
    assert expr.op == p.concatenate_p

    axis = expr.kwargs.get("axis", 0)

    left = []
    right = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == core.operator_mul_p:
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
        else:
            # TODO: how to choose side? for example, some users do w * matrix,
            # while others do matrix * w. Here I want to append the 1 to the w.
            # It seems the way to do this is to check which symbols each
            # child_expr is derived from, then when we're doing an identity, use
            # the symbols *this* child_expr is derived from. if they aren't
            # derived from the same symbols, then there's no benefit to choosing
            # correctly.
            #
            # TODO: what if we're multiplying matrices with matrices? sometimes
            # we need to instead append a matrix of ones, etc.
            num_values = v.shape(child_expr)[axis]
            left.append(np.ones(num_values))
            right.append(child_expr)

    left = transform(
        v.with_return_shape(
            vnp.concatenate(left, axis=axis),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in left], axis=axis))
    )
    right = transform(
        v.with_return_shape(
            vnp.concatenate(right, axis=axis),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in right], axis=axis))
    )

    return transform(
        v.with_return_shape(
            left * right,
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in expr.args[0]],
                                 axis=axis)))


v.implicit_stack_ops.update({
    p.sum_p: stack_detect_shape,
    p.prod_p: stack_detect_shape,
})


impls.push_stack_through_op.update({
    core.symbol_p: pushthrough_return_self,
    p.sum_p: partial(push_stack_through_reduction, p.sum_p,
                     parallel_sum),
    p.prod_p: partial(push_stack_through_reduction, p.prod_p,
                      parallel_prod),
    core.operator_mul_p: push_stack_through_mul,
})

impls.push_concatenate_through_op.update({
    core.operator_getitem_p: push_concatenate_through_getitem,
    core.operator_truediv_p: push_concatenate_through_truediv,
    core.operator_mul_p: push_concatenate_through_mul,
    p.stack_p: push_concatenate_through_stack,
})


impls.push_moveaxis_through_op.update({
    p.stack_p: push_moveaxis_through_stack,
    p.sum_p: push_moveaxis_through_sum,
})


v.phase_ops[0] += [
    (p.stack_p, stack_pushthrough),
]

v.phase_ops[1] += [
    (p.stack_p, stack_pushthrough),
    (p.concatenate_p, concatenate_pushthrough),
    (p.moveaxis_p, moveaxis_pushthrough),
]
