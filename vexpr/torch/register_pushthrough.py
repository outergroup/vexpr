from functools import partial

import torch

import vexpr as vp
import vexpr.core as core
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as csp_p
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
from vexpr.custom.torch import cdist_multi

def canonical_axis(axis, ndim):
    if axis is None:
        return None
    elif axis < 0:
        return axis + ndim
    else:
        return axis

def push_concat_through_getitem(shapes, expr, allow_partial=True):
    assert expr.op is p.concat_p

    # Quick hacky assumptions: every child expr is selecting from a symbol, and
    # they are the same symbol.
    # TODO: if any child is not a getitem, then push through all the getitems,
    # concat with the non-getitems, and insert a shuffle after everything
    # to correct the order.
    if not all(sub_expr.op is core.operator_getitem_p
               for sub_expr in expr.args[0]):
        raise NotImplementedError()

    select_targets = [sub_expr.args[0]
                      for sub_expr in expr.args[0]]
    select_target = select_targets[0]
    if not all(target.op is core.symbol_p for target in select_targets):
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
            arr_shape = shapes[id(select_target)]

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
        all_select_indices.append(torch.as_tensor(select_indices))

    if not all(select_axis == all_select_axes[0] for select_axis in all_select_axes):
        raise NotImplementedError()
    select_axis = all_select_axes[0]

    axis = expr.kwargs.get("dim", 0)
    ndim = len(shapes[id(select_target)])
    if not canonical_axis(axis, ndim) == canonical_axis(select_axis, ndim):
        raise NotImplementedError()
    else:
        # prefer the version that the user specified
        axis = select_axis

    indices = torch.concat(all_select_indices)
    if axis == 0:
        new_selection = indices
    elif axis < 0:
        new_selection = (Ellipsis, indices,) + (slice(None),) * ((-axis) - 1)
    else:
        new_selection = ((slice(None),) * axis) + (indices,)

    return select_target[new_selection]

core.pushthrough_impls.update({
    (p.concat_p, core.operator_getitem_p): push_concat_through_getitem,
})



def push_moveaxis_through_stack(shapes, expr, allow_partial=True):
    assert expr.op is p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op is p.stack_p

    source = expr.args[1]
    stack_expr = expr.args[0]
    stack_axis = stack_expr.kwargs.get("dim", 0)

    ndim = len(shapes[id(stack_expr)])
    if source < 0: source += ndim
    if stack_axis < 0: stack_axis += ndim

    if stack_axis == source:
        dest = expr.args[2]
        return vtorch.stack(stack_expr.args[0], dim=dest)
    else:
        # don't attempt, leave the moveaxis where it is
        return expr


def push_moveaxis_through_sum(shapes, expr, allow_partial=True):
    assert expr.op is p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op is p.sum_p

    sum_expr = expr.args[0]
    sum_arg0 = sum_expr.args[0]
    if not isinstance(sum_arg0, vp.Vexpr):
        sum_arg0 = vtorch.stack(sum_arg0)

    source = expr.args[1]
    dest = expr.args[2]
    sum_arg0 = vtorch.moveaxis(sum_arg0, source, dest)
    sum_arg0 = core.pushthrough(shapes, sum_arg0, p.stack_p)

    sum_axis = sum_expr.kwargs.get("dim", None)

    trace = list(range(len(shapes[id(sum_expr)])))
    trace.insert(dest, trace.pop(source))
    new_axis = trace.index(sum_axis)

    return sum(sum_arg0, dim=new_axis)

core.pushthrough_impls.update({
    (p.moveaxis_p, p.stack_p): push_moveaxis_through_stack,
    (p.moveaxis_p, p.sum_p): push_moveaxis_through_sum,
})


def push_concat_through_stack(shapes, expr, allow_partial=True):
    assert expr.op is p.concat_p

    # initial hack: assume all args are stack_p
    assert all(isinstance(arg, vp.Vexpr) and arg.op is p.stack_p
               for arg in expr.args[0])
    assert expr.kwargs.get("dim", 0) == 0

    # (when i remove the hack, the result is going to be
    # shuffle(concat([stack, non-stack-exprs])), or remove the outer
    # concat of there are no non-stack-exprs.)

    # todo push vectorize through children
    all_stacked_vexprs = []
    for child_expr in expr.args[0]:
        all_stacked_vexprs.extend(child_expr.args[0])

    expr = vtorch.stack(all_stacked_vexprs)
    expr = core._vectorize(shapes, expr)
    return expr

def torch_stack_shape(initial_shape, num_elements, dim=0):
    """
    generated by github copilot
    """
    if dim < 0:
        dim += len(initial_shape) + 1
    return initial_shape[:dim] + (num_elements,) + initial_shape[dim:]

def push_stack_through_reduction(reduction_p, parallel_reduction, shapes, expr,
                                 allow_partial=True):
    assert expr.op is p.stack_p

    exprs_to_stack = expr.args[0]
    all_reduction_operands = []
    at_indices = []

    stack_axis = expr.kwargs.get("dim", 0)

    for i, child_expr in enumerate(exprs_to_stack):
        if isinstance(child_expr, vp.Vexpr) and child_expr.op is reduction_p:

            r_axis = child_expr.kwargs.get("dim", None)
            if r_axis is None:
                raise NotImplementedError()

            r_arg0 = child_expr.args[0]
            if isinstance(r_arg0, vp.Vexpr):
                num_operands = shapes[id(r_arg0)][r_axis]
            else:
                num_operands = len(r_arg0)
                # treat array_like as an implicit stack.
                r_arg0 = vtorch.stack(r_arg0)

            if r_axis != stack_axis:
                prev_shape = shapes[id(r_arg0)]
                r_arg0 = vtorch.moveaxis(r_arg0, r_axis, stack_axis)
                r_arg0 = core.pushthrough(shapes, r_arg0, child_expr.op)
                r_axis = stack_axis

            # Incorporate child_expr's computation into a vectorized reduction.
            all_reduction_operands.append(r_arg0)
            at_indices += [i] * num_operands
        else:
            # Pass child_expr through. Implement Identity as a reduction of 1
            # element.

            # TODO is this is gross to do a full vectorize here?
            child_expr = vtorch.stack([child_expr], dim=stack_axis)
            child_expr = core._vectorize(shapes, child_expr)
            all_reduction_operands.append(child_expr)
            at_indices.append(i)

    at_indices = torch.tensor(at_indices)

    child_shape = next(shapes[id(expr)] for expr in exprs_to_stack
                       if isinstance(expr, vp.Vexpr))
    result_shape = torch_stack_shape(child_shape, len(exprs_to_stack),
                                     dim=stack_axis)

    return parallel_reduction(
        result_shape, stack_axis, at_indices,
        core._vectorize(shapes, vtorch.concat(all_reduction_operands,
                                              dim=stack_axis)))

def parallel_sum(result_shape, dim, index, source):
    return vtorch.index_add(vtorch.zeros(result_shape), dim, index, source)

def parallel_prod(result_shape, dim, index, source):
    return vtorch.index_reduce(vtorch.ones(result_shape), dim, index, source,
                               "prod")

def push_concat_through_truediv(shapes, expr, allow_partial=True):
    assert expr.op is p.concat_p

    # initial hack: assume all args are truediv
    #
    # TODO: add similar treatment of "*_through_mul", dividing by 1.0. It's
    # easier than mul, since it's unambiguous; we can always put 1.0 in the
    # denominator.
    assert all(isinstance(arg, vp.Vexpr) and arg.op is core.operator_truediv_p
               for arg in expr.args[0])

    num = []
    den = []
    for child_expr in expr.args[0]:
        num.append(child_expr.args[0])
        den.append(child_expr.args[1])

    axis = expr.kwargs.get("dim", 0)
    num = core._vectorize(shapes, vtorch.concat(num, dim=axis))
    den = core._vectorize(shapes, vtorch.concat(den, dim=axis))

    return num / den

def push_stack_through_mul(shapes, expr, allow_partial=True):
    assert expr.op is p.stack_p

    left = []
    right = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) and child_expr.op is core.operator_mul_p:
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
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]

    # decide whether ones need to be inserted into the shape to get proper
    # broadcasting
    left_shapes = [shapes[id(child_expr)] for child_expr in left]
    left_shape_before_stack = left_shapes[0]
    assert all(left_shape_before_stack == shape for shape in left_shapes)
    right_shapes = [shapes[id(child_expr)] for child_expr in right]
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

    axis = expr.kwargs.get("dim", 0)
    actual_left_shape = torch_stack_shape(left_shape_before_stack,
                                       len(left), axis)
    actual_right_shape = torch_stack_shape(right_shape_before_stack,
                                        len(right), axis)
    implicit_left_shape = torch_stack_shape(implicit_left_shape,
                                         len(left), axis)
    implicit_right_shape = torch_stack_shape(implicit_right_shape,
                                          len(right), axis)

    implicit_neg_axis = (axis
                         if axis < 0
                         else -len(implicit_left_shape) + axis)

    left = core._vectorize(shapes, vtorch.stack(left, **kwargs))
    if implicit_left_shape[implicit_neg_axis:] \
       != actual_left_shape[implicit_neg_axis:]:
        left = vtorch.reshape(left, implicit_left_shape[implicit_neg_axis:])

    right = core._vectorize(shapes, vtorch.stack(right, **kwargs))
    if implicit_right_shape[implicit_neg_axis:] \
       != actual_right_shape[implicit_neg_axis:]:
        right = vtorch.reshape(right, implicit_right_shape[implicit_neg_axis:])

    return left * right

def push_concat_through_mul(shapes, expr, allow_partial=True):
    assert expr.op is p.concat_p

    axis = expr.kwargs.get("dim", 0)

    left = []
    right = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) and child_expr.op is core.operator_mul_p:
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
            num_values = shapes[id(child_expr)][axis]
            left.append(torch.ones(num_values))
            right.append(child_expr)

    left = core._vectorize(shapes, vtorch.concat(left, dim=axis))
    right = core._vectorize(shapes, vtorch.concat(right, dim=axis))

    return left * right

core.pushthrough_impls.update({
    (p.concat_p, p.stack_p): push_concat_through_stack,
    (p.stack_p, p.sum_p): partial(push_stack_through_reduction, p.sum_p,
                                  parallel_sum),
    (p.stack_p, p.prod_p): partial(push_stack_through_reduction, p.prod_p,
                                   parallel_prod),
    (p.concat_p, core.operator_truediv_p): push_concat_through_truediv,
    (p.stack_p, core.operator_mul_p): push_stack_through_mul,
    (p.concat_p, core.operator_mul_p): push_concat_through_mul,
})


def push_stack_through_cdist(shapes, expr, allow_partial=True):
    assert expr.op is p.stack_p
    assert all(child_expr.op is p.cdist_p for child_expr in expr.args[0])

    # TODO process this
    stack_axis = expr.kwargs.get("dim", 0)

    left = []
    right = []
    lengths = []
    ps = []
    child_matrix_shapes = []
    for child_expr in expr.args[0]:
        left.append(child_expr.args[0])
        right.append(child_expr.args[1])
        ps.append(child_expr.kwargs.get("p", 2))

        shape = shapes[id(child_expr.args[0])]
        length = shape[-1]
        lengths.append(length)
        child_matrix_shapes.append(shape[:-1])

    left = core._vectorize(shapes, vtorch.concat(left, dim=-1))
    right = core._vectorize(shapes, vtorch.concat(right, dim=-1))

    kwargs = dict(
        lengths=torch.tensor(lengths),
        ps=torch.tensor(ps),
    )
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]

    ret = cdist_multi(left, right, **kwargs)

    # Compute shape of result
    child_matrix_shape = child_matrix_shapes[0]
    assert all(shape == child_matrix_shape for shape in child_matrix_shapes[1:])

    axis = expr.kwargs.get("dim", 0)
    # use axis to determine result shape after stack
    result_shape = child_matrix_shape
    if axis < 0:
        axis += len(result_shape) + 1
    result_shape = result_shape[:axis] + (len(lengths),) + result_shape[axis:]
    shapes[id(ret)] = result_shape

    return ret

core.pushthrough_impls[(p.stack_p, p.cdist_p)] = push_stack_through_cdist


def push_concat_through_cdist_multi(shapes, expr, allow_partial=True):
    assert expr.op is p.concat_p
    assert all(child_expr.op is csp_p.cdist_multi_p for child_expr in expr.args[0])

    # TODO process this
    concat_axis = expr.kwargs.get("dim", 0)

    left = []
    right = []
    lengths = []
    axes = []
    ps = []
    for child_expr in expr.args[0]:
        left.append(child_expr.args[0])
        right.append(child_expr.args[1])
        lengths.append(child_expr.kwargs["lengths"])
        ps.append(child_expr.kwargs["ps"])
        axes.append(child_expr.kwargs.get("dim", None))

    canonicalized_axes = [(axis if axis is not None else 0)
                          for axis in axes]
    if not all(axis == canonicalized_axes[0]
               for axis in canonicalized_axes[1:]):
        raise ValueError("Expected same axes", axes)
    axis = axes[0]

    left = core._vectorize(shapes, vtorch.concat(left, dim=-1))
    right = core._vectorize(shapes, vtorch.concat(right, dim=-1))

    kwargs = dict(
        lengths=torch.concat(lengths),
        ps=torch.concat(ps),
    )
    if axis is not None:
        kwargs["dim"] = axis

    return cdist_multi(left, right, **kwargs)

core.pushthrough_impls[(p.concat_p, csp_p.cdist_multi_p)] = push_concat_through_cdist_multi
