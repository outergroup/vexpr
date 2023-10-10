import collections
from functools import partial

import torch

import vexpr as vp
import vexpr.core as core
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as csp_p
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.vectorization as v
from vexpr.vectorization import _vectorize, with_metadata
from vexpr.custom.torch.utils import (
    maybe_shuffle,
)
from vexpr.torch.utils import (
    canonical_axis,
    invert_shuffle,
    torch_cat_shape,
    torch_stack_shape,
    torch_stack_shape2,
    stack_remainder_then_combine,
    cat_remainder_then_combine,
    push_stack_through_reduction,
)


def push_cat_through_index_select(expr, allow_partial=True):
    assert expr.op == p.cat_p

    # Quick hacky assumptions: every child expr is selecting from a symbol, and
    # they are the same symbol.
    if not all(sub_expr.op == p.index_select_p
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

    all_select_dims = [sub_expr.args[1]
                       for sub_expr in expr.args[0]]
    assert all(d == all_select_dims[0] for d in all_select_dims)
    select_dim = all_select_dims[0]


    selections = [sub_expr.args[2]
                  for sub_expr in expr.args[0]]
    all_select_indices = []
    for selection in selections:
        select_indices = selection
        all_select_indices.append(torch.as_tensor(select_indices))

    dim = expr.kwargs.get("dim", 0)
    select_target_shape = v.shape(select_target)
    ndim = len(select_target_shape)
    if not canonical_axis(dim, ndim) == canonical_axis(select_dim, ndim):
        raise NotImplementedError()
    else:
        # prefer the version that the user specified
        dim = select_dim

    indices = torch.cat(all_select_indices)

    if torch.equal(indices, torch.arange(select_target_shape[dim])):
        # the vectorized index_select is selecting all elements, so it can be
        # factored out.
        return select_target

    return_shape = list(v.shape(select_target))
    return_shape[dim] = len(indices)
    return_shape = tuple(return_shape)
    return v.with_return_shape(
        vtorch.index_select(select_target, dim, indices),
        return_shape)


v.pushthrough_impls.update({
    (p.cat_p, p.index_select_p): push_cat_through_index_select,
})


def push_cat_through_getitem(expr, allow_partial=True):
    assert expr.op == p.cat_p

    # Quick hacky assumptions: every child expr is selecting from a symbol, and
    # they are the same symbol.
    # TODO: if any child is not a getitem, then push through all the getitems,
    # cat with the non-getitems, and insert a shuffle after everything
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
        all_select_indices.append(torch.as_tensor(select_indices))

    if not all(select_axis == all_select_axes[0] for select_axis in all_select_axes):
        raise NotImplementedError()
    select_axis = all_select_axes[0]

    axis = expr.kwargs.get("dim", 0)
    select_target_shape = v.shape(select_target)
    ndim = len(select_target_shape)
    if not canonical_axis(axis, ndim) == canonical_axis(select_axis, ndim):
        raise NotImplementedError()
    else:
        # prefer the version that the user specified
        axis = select_axis

    indices = torch.cat(all_select_indices)

    if torch.equal(indices, torch.arange(select_target_shape[axis])):
        # the vectorized getitem is selecting all elements, so it can be
        # factored out.
        return select_target

    if axis == 0:
        new_selection = indices
    elif axis < 0:
        new_selection = (Ellipsis, indices,) + (slice(None),) * ((-axis) - 1)
    else:
        new_selection = ((slice(None),) * axis) + (indices,)

    return v.with_return_shape(select_target[new_selection],
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in expr.args[0]],
                                               axis))

v.pushthrough_impls.update({
    (p.cat_p, core.operator_getitem_p): push_cat_through_getitem,
})


def push_moveaxis_through_stack(expr, allow_partial=True):
    assert expr.op == p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op == p.stack_p

    source = expr.args[1]
    stack_expr = expr.args[0]
    stack_axis = stack_expr.kwargs.get("dim", 0)

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
        return v.with_return_shape(vtorch.stack(stack_expr.args[0], dim=dest),
                                   tuple(new_shape))
    else:
        # don't attempt, leave the moveaxis where it is
        return expr


def push_cat_through_zeros_ones(zeros_ones_p, zeros_ones,
                                   expr, allow_partial=True):
    assert expr.op == p.cat_p

    # initial hack: assume all args are same op
    assert all(isinstance(arg, vp.Vexpr) and arg.op == zeros_ones_p
               for arg in expr.args[0])

    dim = expr.kwargs.get("dim", 0)
    child_shapes = [child_expr.args[0] for child_expr in expr.args[0]]
    result_shape = torch_cat_shape(child_shapes, dim)
    return v.with_return_shape(zeros_ones(result_shape),
                               result_shape)

v.pushthrough_impls.update({
    (p.cat_p, p.zeros_p): partial(push_cat_through_zeros_ones, p.zeros_p,
                                  vtorch.zeros),
    (p.cat_p, p.ones_p): partial(push_cat_through_zeros_ones, p.ones_p,
                                 vtorch.ones),
})


def push_cat_through_unsqueeze(expr, allow_partial=True):
    assert expr.op == p.cat_p
    assert all(isinstance(child_expr, vp.Vexpr)
               and child_expr.op == p.unsqueeze_p
               for child_expr in expr.args[0])

    unsqueeze_dims = [child_expr.args[1]
                      for child_expr in expr.args[0]]
    assert all(dim == unsqueeze_dims[0] for dim in unsqueeze_dims)
    unsqueeze_dim = unsqueeze_dims[0]

    grandchildren = [child_expr.args[0] for child_expr in expr.args[0]]
    grandchildren = v._vectorize(
        v.with_return_shape(vtorch.cat(grandchildren, **expr.kwargs),
                            torch_cat_shape([v.shape(gc)
                                             for gc in grandchildren],
                                            **expr.kwargs)))

    return_shape = v.shape(grandchildren)
    unsqueeze_dim_nonneg = (unsqueeze_dim
                            if unsqueeze_dim >= 0
                            else unsqueeze_dim + len(return_shape) + 1)
    return_shape = (return_shape[:unsqueeze_dim_nonneg]
                    + (1,)
                    + return_shape[unsqueeze_dim_nonneg:])

    return v.with_return_shape(
        vtorch.unsqueeze(grandchildren, unsqueeze_dim),
        return_shape
    )



def push_moveaxis_through_sum(expr, allow_partial=True):
    assert expr.op == p.moveaxis_p
    assert isinstance(expr.args[0], vp.Vexpr) and expr.args[0].op == p.sum_p

    sum_expr = expr.args[0]
    sum_arg0 = sum_expr.args[0]
    if not isinstance(sum_arg0, vp.Vexpr):
        sum_arg0 = v.with_return_shape(vtorch.stack(sum_arg0),
                                       torch_stack_shape(
                                           v.shape(sum_expr.args[0]),
                                           len(sum_arg0), dim=0))

    source = expr.args[1]
    dest = expr.args[2]
    sum_arg0 = vtorch.moveaxis(sum_arg0, source, dest)
    sum_arg0 = v.pushthrough(sum_arg0, p.stack_p)

    sum_axis = sum_expr.kwargs.get("dim", None)

    trace = list(range(len(v.shape(sum_expr))))
    trace.insert(dest, trace.pop(source))
    new_axis = trace.index(sum_axis)

    return sum(sum_arg0, dim=new_axis)

v.pushthrough_impls.update({
    (p.cat_p, p.unsqueeze_p): push_cat_through_unsqueeze,
    (p.moveaxis_p, p.stack_p): push_moveaxis_through_stack,
    (p.moveaxis_p, p.sum_p): push_moveaxis_through_sum,
})


def push_cat_through_stack(expr, allow_partial=True):
    assert expr.op == p.cat_p

    # initial hack: only do anything if everything is a stack
    if not all(isinstance(arg, vp.Vexpr) and arg.op == p.stack_p
               for arg in expr.args[0]):
        return expr

    dim = expr.kwargs.get("dim", 0)

    # (when i remove the hack, the result is going to be
    # shuffle(cat([stack, non-stack-exprs])), or remove the outer
    # cat of there are no non-stack-exprs.)

    # todo push vectorize through children
    all_stacked_vexprs = []
    for child_expr in expr.args[0]:
        assert child_expr.kwargs.get("dim", 0) == dim
        all_stacked_vexprs.extend(child_expr.args[0])

    assert all(child_expr.kwargs == expr.args[0][0].kwargs
               for child_expr in expr.args[0][1:])
    stack_kwargs = expr.args[0][0].kwargs

    expr = v.with_return_shape(
        vtorch.stack(all_stacked_vexprs, **stack_kwargs),
        torch_stack_shape2([v.shape(d) for d in all_stacked_vexprs],
                           **stack_kwargs)
    )
    expr = v._vectorize(expr)
    return expr

def push_cat_through_cat(expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dims = [child_expr.kwargs.get("dim", 0)
                for child_expr in expr.args[0]
                if isinstance(child_expr, vp.Vexpr)
                and child_expr.op == p.cat_p]
    assert all(cat_dim == cat_dims[0] for cat_dim in cat_dims)
    cat_dim = cat_dims[0]

    # todo push vectorize through children
    all_cat_vexprs = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == p.cat_p:
            all_cat_vexprs.extend(child_expr.args[0])
        else:
            all_cat_vexprs.append(child_expr)

    expr = vtorch.cat(all_cat_vexprs, dim=cat_dim)
    return v.with_return_shape(
        v._vectorize(expr),
        torch_cat_shape([v.shape(child_expr) for child_expr in all_cat_vexprs],
                        cat_dim)
    )


def push_cat_through_index_reduction(index_reduction_p, parallel_reduction,
                                     expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    index_reduction_dims = [child_expr.args[1]
                            for child_expr in expr.args[0]
                            if isinstance(child_expr, vp.Vexpr)
                            and child_expr.op == index_reduction_p]
    assert all(dim == index_reduction_dims[0] for dim in index_reduction_dims)
    index_reduction_dim = index_reduction_dims[0]

    target = []
    indices = []
    grandchildren = []
    base = 0
    for child_expr in expr.args[0]:
        child_shape = v.shape(child_expr)
        num_results = child_shape[cat_dim]
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == index_reduction_p:
            target.append(child_expr.args[0])
            grandchildren.append(child_expr.args[3])
            indices.append(child_expr.args[2] + base)
            base += num_results
        else:
            target.append(vtorch.zeros(child_shape))
            grandchildren.append(child_expr)
            indices.append(torch.arange(base, base + num_results))
            base += num_results

    target = v._vectorize(vtorch.cat(target, dim=cat_dim))
    indices = torch.cat(indices)
    grandchildren = v._vectorize(vtorch.cat(grandchildren, dim=cat_dim))

    return v.with_return_shape(parallel_reduction(target, index_reduction_dim,
                                                  indices, grandchildren),
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in expr.args[0]],
                                               dim=cat_dim))

def parallel_sum(target, dim, index, source):
    return vtorch.index_add(target, dim, index, source)

def parallel_prod(target, dim, index, source):
    return vtorch.index_reduce(target, dim, index, source, "prod")


def push_stack_through_unary_elementwise(op, expr, allow_partial=True):
    assert expr.op == p.stack_p

    applicable = []
    applicable_indices = []
    remainder = []
    remainder_indices = []
    for i, child_expr in enumerate(expr.args[0]):
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == op:
            applicable.append(child_expr)
            applicable_indices.append(i)
        else:
            remainder.append(child_expr)
            remainder_indices.append(i)

    grandchildren = [child_expr.args[0] for child_expr in applicable]
    grandchildren = v._vectorize(
        v.with_return_shape(vtorch.stack(grandchildren, **expr.kwargs),
                            torch_stack_shape2([v.shape(gc)
                                                for gc in grandchildren],
                                               **expr.kwargs)))

    return stack_remainder_then_combine(
        v.with_return_shape(
            vp.Vexpr(op, (grandchildren,), applicable[0].kwargs),
            v.shape(grandchildren)
        ),
        remainder,
        applicable_indices,
        remainder_indices,
        **expr.kwargs)


def push_cat_through_unary_elementwise(op, expr, allow_partial=True):
    assert expr.op == p.cat_p

    applicable = []
    applicable_indices = []
    remainder = []
    remainder_indices = []
    base = 0
    dim = expr.kwargs.get("dim", 0)
    for child_expr in expr.args[0]:
        num_indices = v.shape(child_expr)[dim]
        result_indices = list(range(base, base + num_indices))
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == op:
            applicable.append(child_expr)
            applicable_indices += result_indices
        else:
            if not allow_partial:
                raise v.CannotVectorize()
            remainder.append(child_expr)
            remainder_indices += result_indices
        base += num_indices

    grandchildren = [child_expr.args[0] for child_expr in applicable]
    grandchildren = v._vectorize(
        v.with_return_shape(vtorch.cat(grandchildren, **expr.kwargs),
                            torch_cat_shape([v.shape(gc)
                                             for gc in grandchildren],
                                            **expr.kwargs)))

    return cat_remainder_then_combine(
        v.with_return_shape(
            vp.Vexpr(op, (grandchildren,), applicable[0].kwargs),
            v.shape(grandchildren)
        ),
        remainder,
        applicable_indices,
        remainder_indices,
        **expr.kwargs)


def push_cat_through_truediv(expr, allow_partial=True):
    assert expr.op == p.cat_p

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

    axis = expr.kwargs.get("dim", 0)
    num_shape = torch_cat_shape([v.shape(child_expr) for child_expr in num],
                                dim=axis)
    den_shape = torch_cat_shape([v.shape(child_expr) for child_expr in den],
                                dim=axis)
    num = v.with_return_shape(v._vectorize(vtorch.cat(num, dim=axis)),
                              num_shape)
    den = v.with_return_shape(v._vectorize(vtorch.cat(den, dim=axis)),
                              den_shape)

    return v.with_return_shape(num / den,
                               torch.broadcast_shapes(v.shape(num),
                                                      v.shape(den)))

def push_stack_through_mul(expr, allow_partial=True):
    assert expr.op == p.stack_p

    left = []
    right = []
    identity = False
    actual_indices = []
    for i, child_expr in enumerate(expr.args[0]):
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == core.operator_mul_p:
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
            actual_indices.append(i)
        else:
            # TODO this is making lots of assumptions
            identity = True
            right.append(child_expr)

    kwargs = {}
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]

    left = v._vectorize(
        v.with_return_shape(
            vtorch.stack(left, **kwargs),
            torch_stack_shape2([v.shape(d) for d in left],
                               **expr.kwargs)))

    if identity:
        ones_shape = (len(expr.args[0]),)
        left = v.with_return_shape(
            vtorch.scatter(
                v.with_return_shape(vtorch.ones(ones_shape), ones_shape),
                0,
                torch.tensor(actual_indices),
                left),
            ones_shape)

    right = v._vectorize(
        v.with_return_shape(
            vtorch.stack(right, **kwargs),
            torch_stack_shape2([v.shape(d) for d in right],
                               **expr.kwargs)))

    dim = expr.kwargs.get("dim", 0)
    ndim = len(v.shape(right))
    if canonical_axis(dim, ndim) == ndim - 1:
        return v.with_return_shape(left * right,
                                   torch.broadcast_shapes(v.shape(left),
                                                          v.shape(right)))
    else:
        return v.with_return_shape(vctorch.mul_along_dim(left, right, dim=dim),
                                   v.shape(right))


def push_cat_through_mul(expr, allow_partial=True):
    assert expr.op == p.cat_p

    dim = expr.kwargs.get("dim", 0)
    child_shapes = [v.shape(child_expr) for child_expr in expr.args[0]]
    ndim = len(child_shapes[0])

    if canonical_axis(dim, ndim) != ndim - 1:
        raise NotImplementedError(
            "With Vexpr vectorize, don't use vector-tensor elementwise multiplication. "
            "Instead, use vexpr.custom.torch.mul_along_dim, which states your "
            "intent more clearly and is hence much easier to vectorize."
        )

    left = []
    right = []
    base = 0
    actual_indices = []
    identity = False
    for child_expr in expr.args[0]:
        n = v.shape(child_expr)[dim]
        if isinstance(child_expr, vp.Vexpr) and child_expr.op == core.operator_mul_p:
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
            actual_indices += list(range(base, base + n))
        else:
            # TODO: this is hardcoded to expect identity to occur on the left
            right.append(child_expr)
            identity = True
        base += n
    total_n = base

    left_shapes = [v.shape(child_expr) for child_expr in left]

    left = v._vectorize(v.with_return_shape(vtorch.cat(left, dim=dim),
                                            torch_cat_shape(left_shapes,
                                                            dim=dim)))
    if identity:
        ones_shape = (total_n,)
        left = v.with_return_shape(
            vtorch.scatter(
                v.with_return_shape(vtorch.ones(ones_shape),
                                    ones_shape),
                0,
                torch.tensor(actual_indices),
                left),
            ones_shape)

    right_shapes = [v.shape(child_expr) for child_expr in right]

    right = v._vectorize(v.with_return_shape(vtorch.cat(right, dim=dim),
                                             torch_cat_shape(right_shapes,
                                                             dim=dim)))

    return v.with_return_shape(left * right, torch_cat_shape(child_shapes,
                                                             dim=dim))

def push_cat_through_scatter(expr, allow_partial=True):
    assert expr.op == p.cat_p

    into = []
    indices = []
    sources = []

    dim = expr.kwargs.get("dim", 0)

    base = 0
    for child_expr in expr.args[0]:
        if not isinstance(child_expr, vp.Vexpr) or child_expr.op != p.scatter_p:
            raise NotImplementedError()

        ndim = len(v.shape(child_expr))
        assert canonical_axis(child_expr.args[1], ndim) == canonical_axis(dim, ndim)

        into.append(child_expr.args[0])
        child_indices = child_expr.args[2]
        if isinstance(child_indices, vp.Vexpr):
            if child_indices.op == p.expand_p:
                child_indices = child_indices.args[0]
            else:
                raise NotImplementedError(
                    "child_indices must be a tensor or a vtorch.expand Vexpr"
                )
        indices.append(base + child_indices)
        sources.append(child_expr.args[3])

        base += v.shape(child_expr)[dim]

    into_shape = torch_cat_shape([v.shape(into_expr) for into_expr in into],
                                 dim=dim)
    source_shape = torch_cat_shape([v.shape(source_expr) for source_expr in sources],
                                   dim=dim)

    into = v._vectorize(v.with_return_shape(vtorch.cat(into, dim=dim),
                                            into_shape))
    sources = v._vectorize(v.with_return_shape(vtorch.cat(sources, dim=dim),
                                               source_shape))
    indices = torch.cat(indices, dim=dim)
    batch_shape = v.shape(sources)[:-1]
    if len(batch_shape) > 0:
        num_indices = indices.shape[-1]
        indices = indices.view((1,) * (len(batch_shape)) + (num_indices,))
        indices = vtorch.expand(indices, batch_shape + (num_indices,))
    return v.with_return_shape(vtorch.scatter(into, dim, indices, sources),
                               into_shape)

v.pushthrough_impls.update({
    (p.cat_p, p.stack_p): push_cat_through_stack,
    (p.cat_p, p.cat_p): push_cat_through_cat,
    (p.stack_p, p.sum_p): partial(push_stack_through_reduction, p.sum_p,
                                  vctorch.sum_multi, 0.),
    (p.stack_p, p.prod_p): partial(push_stack_through_reduction, p.prod_p,
                                   vctorch.prod_multi, 1.),
    (p.stack_p, csp_p.fast_prod_positive_p): partial(
        push_stack_through_reduction, csp_p.fast_prod_positive_p,
        vctorch.fast_prod_positive_multi, 1.),
    (p.cat_p, p.index_add_p): partial(push_cat_through_index_reduction,
                                      p.index_add_p,
                                      parallel_sum),
    # TODO this is hardcoded to prod, but index reduce might use e.g. mean
    (p.cat_p, p.index_reduce_p): partial(push_cat_through_index_reduction,
                                         p.index_reduce_p,
                                         parallel_prod),
    (p.stack_p, p.exp_p): partial(
        push_stack_through_unary_elementwise, p.exp_p),
    (p.cat_p, p.exp_p): partial(
        push_cat_through_unary_elementwise, p.exp_p),
    (p.cat_p, core.operator_truediv_p): push_cat_through_truediv,
    (p.stack_p, core.operator_mul_p): push_stack_through_mul,
    (p.cat_p, core.operator_mul_p): push_cat_through_mul,
    (p.stack_p, core.operator_neg_p): partial(
        push_stack_through_unary_elementwise, core.operator_neg_p),
    (p.cat_p, core.operator_neg_p): partial(
        push_cat_through_unary_elementwise, core.operator_neg_p),
    (p.cat_p, p.scatter_p): push_cat_through_scatter,
})


def push_stack_through_cdist(expr, allow_partial=True):
    assert expr.op == p.stack_p

    stack_dim = expr.kwargs.get("dim", 0)

    left = []
    right = []
    lengths = []
    ps = []
    child_matrix_shapes = []

    applicable_indices = []
    remainder = []
    remainder_indices = []

    for i, child_expr in enumerate(expr.args[0]):
        if child_expr.op == p.cdist_p:
            applicable_indices.append(i)
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
            lengths.append(v.shape(child_expr.args[0])[-1])
            ps.append(child_expr.kwargs.get("p", 2))
            child_matrix_shapes.append(v.shape(child_expr))
        else:
            remainder.append(child_expr)
            remainder_indices.append(i)

    left = v._vectorize(vtorch.cat(left, dim=-1))
    right = v._vectorize(vtorch.cat(right, dim=-1))

    groups = list(collections.Counter(zip(lengths, ps)).items())

    pre_shuffle_indices = []
    post_shuffle_indices_inverted = []
    for (length, metric), count in groups:
        base = 0
        for i, (length2, metric2) in enumerate(zip(lengths, ps)):
            if length2 == length and metric2 == metric:
                pre_shuffle_indices += list(range(base, base + length))
                post_shuffle_indices_inverted.append(i)
            base += length2

    pre_shuffle_indices = torch.tensor(pre_shuffle_indices)
    post_shuffle_indices = invert_shuffle(post_shuffle_indices_inverted)

    left = maybe_shuffle(left, pre_shuffle_indices, dim=-1)
    right = maybe_shuffle(right, pre_shuffle_indices, dim=-1)

    kwargs = dict(
        groups = groups,
    )
    ndim = len(v.shape(expr))
    if canonical_axis(stack_dim, ndim) != canonical_axis(-3, ndim):
        raise ValueError(
            f"cdist_multi always uses a stack_dim of -3, got {stack_dim}"
        )

    # Compute shape of result
    child_matrix_shape = child_matrix_shapes[0]
    assert all(shape == child_matrix_shape for shape in child_matrix_shapes[1:])

    dim = expr.kwargs.get("dim", 0)
    # use dim to determine result shape after stack
    result_shape = child_matrix_shape
    if dim < 0:
        dim += len(result_shape) + 1
    result_shape = result_shape[:dim] + (len(lengths),) + result_shape[dim:]

    applicable =  v.with_return_shape(
        vctorch.cdist_multi(left, right, **kwargs),
        result_shape)
    applicable = maybe_shuffle(applicable, post_shuffle_indices, dim=dim)

    return stack_remainder_then_combine(
        applicable,
        remainder,
        applicable_indices,
        remainder_indices,
        **expr.kwargs)

v.pushthrough_impls[(p.stack_p, p.cdist_p)] = push_stack_through_cdist
