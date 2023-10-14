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
from vexpr.custom.torch.utils import (
    maybe_shuffle,
)
from vexpr.torch.utils import (
    invert_shuffle,
    canonical_axis,
    canonical_stack_dim,
    maybe_lift_scatter,
    torch_stack_shape2,
    torch_cat_shape,
    cat_remainder_then_combine,
    push_stack_through_reduction,
)


def push_cat_through_shuffle(expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    # get a list of shuffles with the same dim by wrapping everything else with
    # identity shuffles
    child_exprs = [(child_expr
                    if child_expr.op == csp_p.shuffle_p
                    and child_expr.kwargs.get("dim", 0) == cat_dim
                    else v.with_return_shape(vctorch.shuffle(
                            child_expr,
                            torch.arange(v.shape(child_expr)[cat_dim])),
                                             v.shape(child_expr)))
                   for child_expr in expr.args[0]]

    grandchildren = []
    indices = []
    base = 0
    for child_expr in child_exprs:
        grandchildren.append(child_expr.args[0])
        child_indices = child_expr.args[1]
        indices.append(child_indices + base)
        base += len(child_indices)

    result_shape = torch_cat_shape([v.shape(child_expr)
                                    for child_expr in child_exprs],
                                   cat_dim)

    ret = v._vectorize(
        v.with_return_shape(
            vtorch.cat(grandchildren, **expr.kwargs),
            result_shape
        )
    )
    indices = torch.cat(indices)

    return maybe_shuffle(ret, indices, **expr.kwargs)


v.pushthrough_impls[(p.cat_p, csp_p.shuffle_p)] = push_cat_through_shuffle


def push_cat_through_cdist_multi(expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    all_groups = [child_expr.kwargs["groups"]
                  for child_expr in expr.args[0]
                  if child_expr.op == csp_p.cdist_multi_p]

    groups = collections.Counter()
    for group in all_groups:
        groups.update(collections.Counter(dict(group)))
    groups = list(groups.items())

    left = []
    right = []

    applicable_exprs = []
    applicable_indices = []
    remainder = []
    remainder_indices = []
    base = 0
    for child_expr in expr.args[0]:
        num_indices = v.shape(child_expr)[cat_dim]
        result_indices = list(range(base, base + num_indices))
        if child_expr.op == csp_p.cdist_multi_p:
            applicable_indices += result_indices
            applicable_exprs.append(child_expr)
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
        else:
            if not allow_partial:
                raise v.CannotVectorize()
            remainder_indices += result_indices
            remainder.append(child_expr)
        base += num_indices

    left = v._vectorize(vtorch.cat(left, dim=-1))
    right = v._vectorize(vtorch.cat(right, dim=-1))

    pre_shuffle_indices = []
    post_shuffle_indices_inverted = []
    for (length, metric), count in groups:
        base = 0
        i = 0
        n_found = 0
        for child_groups in all_groups:
            for (length2, metric2), count2 in child_groups:
                group_length = length2 * count2
                if length2 == length and metric2 == metric:
                    pre_shuffle_indices += list(range(base, base + group_length))
                    post_shuffle_indices_inverted += list(range(i, i + count2))
                    n_found += group_length
                base += group_length
                i += count2
        assert n_found == length * count
    assert i == sum(count for _, count in groups)

    pre_shuffle_indices = torch.tensor(pre_shuffle_indices)
    post_shuffle_indices = invert_shuffle(post_shuffle_indices_inverted)

    left = maybe_shuffle(left, pre_shuffle_indices, dim=-1)
    right = maybe_shuffle(right, pre_shuffle_indices, dim=-1)

    kwargs = dict(
        groups = groups,
    )

    ndim = len(v.shape(left))
    if canonical_stack_dim(cat_dim, ndim) != canonical_stack_dim(-3, ndim):
        raise ValueError(
            f"cdist_multi always uses a stack_dim of -3, got {cat_dim}"
        )

    result_shape = torch_cat_shape([v.shape(child_expr)
                                    for child_expr in applicable_exprs],
                                   dim=cat_dim)
    applicable = v.with_return_shape(
        vctorch.cdist_multi(left, right, **kwargs),
        result_shape)

    applicable = maybe_shuffle(applicable, post_shuffle_indices,
                               dim=cat_dim)

    return cat_remainder_then_combine(
        applicable,
        remainder,
        applicable_indices,
        remainder_indices,
        **expr.kwargs)


v.pushthrough_impls[(p.cat_p, csp_p.cdist_multi_p)] = push_cat_through_cdist_multi


def push_cat_through_reduction_multi(reduction_multi_p, parallel_reduction,
                                     fill_value, expr, all_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    reduction_dims = [child_expr.kwargs.get("dim", None)
                      for child_expr in expr.args[0]
                      if isinstance(child_expr, vp.Vexpr)
                      and child_expr.op == reduction_multi_p]
    assert all(dim == reduction_dims[0] for dim in reduction_dims)
    reduction_dim = reduction_dims[0]

    all_groups = [child_expr.kwargs["groups"]
                  for child_expr in expr.args[0]
                  if child_expr.op == reduction_multi_p]

    all_groups = []
    lengths = []
    grandchildren = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == reduction_multi_p:
            grandchildren.append(child_expr.args[0])
            all_groups.append(child_expr.kwargs["groups"])
        else:
            grandchildren.append(child_expr)
            all_groups.append([(1, v.shape(child_expr)[cat_dim])])

    grandchildren = v._vectorize(vtorch.cat(grandchildren, dim=cat_dim))

    groups = collections.Counter()
    for group in all_groups:
        groups.update(collections.Counter(dict(group)))
    groups = list(groups.items())

    pre_shuffle_indices = []
    post_shuffle_indices_inverted = []
    for length, count in groups:
        base = 0
        i = 0
        n_found = 0
        for child_groups in all_groups:
            for length2, count2 in child_groups:
                group_length = length2 * count2
                if length2 == length:
                    pre_shuffle_indices += list(range(base, base + group_length))
                    post_shuffle_indices_inverted += list(range(i, i + count2))
                    n_found += group_length
                base += group_length
                i += count2
        assert n_found == length * count
    assert i == sum(count for _, count in groups)

    pre_shuffle_indices = torch.tensor(pre_shuffle_indices)
    post_shuffle_indices = invert_shuffle(post_shuffle_indices_inverted)

    grandchildren = maybe_shuffle(grandchildren, pre_shuffle_indices,
                                  reduction_dim)

    result_shape = torch_cat_shape([v.shape(child_expr)
                                    for child_expr in expr.args[0]],
                                   dim=cat_dim)
    result = v.with_return_shape(parallel_reduction(grandchildren,
                                                    groups=groups, dim=reduction_dim),
                                 result_shape)

    return maybe_shuffle(result, post_shuffle_indices, dim=cat_dim)


v.pushthrough_impls.update({
    (p.cat_p, csp_p.sum_multi_p): partial(
        push_cat_through_reduction_multi, csp_p.sum_multi_p, vctorch.sum_multi, 0.),
    (p.cat_p, csp_p.prod_multi_p): partial(
        push_cat_through_reduction_multi, csp_p.prod_multi_p, vctorch.prod_multi, 1.),
    (p.cat_p, csp_p.fast_prod_positive_multi_p): partial(
        push_cat_through_reduction_multi, csp_p.fast_prod_positive_multi_p,
        vctorch.fast_prod_positive_multi, 1.),
    (p.stack_p, csp_p.fast_prod_positive_p): partial(
        push_stack_through_reduction, csp_p.fast_prod_positive_p,
        vctorch.fast_prod_positive_multi, 1.)
})


def push_stack_through_mul_along_dim(expr, allow_partial=True):
    assert expr.op == p.stack_p

    stack_dim = expr.kwargs.get("dim", 0)

    mul_along_dim_dims = [child_expr.kwargs.get("dim", None)
                          for child_expr in expr.args[0]
                          if isinstance(child_expr, vp.Vexpr)
                          and child_expr.op == csp_p.mul_along_dim_p]
    assert all(dim == mul_along_dim_dims[0] for dim in mul_along_dim_dims)
    mul_along_dim_dim = mul_along_dim_dims[0]

    w = []
    t = []
    identity = False
    actual_indices = []
    for i, child_expr in enumerate(expr.args[0]):
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == csp_p.mul_along_dim_p:
            w.append(child_expr.args[0])
            t.append(child_expr.args[1])
            actual_indices.append(i)
        else:
            identity = True
            t.append(child_expr)

    kwargs = {}
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]

    w = v._vectorize(
        v.with_return_shape(
            vtorch.stack(w, dim=-1),
            torch_stack_shape2([v.shape(d) for d in w],
                               dim=-1)))

    if identity:
        ones_shape = v.shape(w)[:-1] + (len(expr.args[0]),)
        indices = torch.tensor(actual_indices)
        batch_shape = v.shape(w)[:-1]
        w = v.with_return_shape(
            maybe_lift_scatter(
                v.with_return_shape(vtorch.ones(ones_shape), ones_shape),
                -1,
                indices,
                w,
                batch_shape=batch_shape),
            ones_shape)

    t = v._vectorize(
        v.with_return_shape(
            vtorch.stack(t, **kwargs),
            torch_stack_shape2([v.shape(d) for d in t],
                               **expr.kwargs)))

    dim = expr.kwargs.get("dim", 0)
    ndim = len(v.shape(t))

    assert canonical_axis(dim, ndim) == canonical_axis(mul_along_dim_dim, ndim)

    return v.with_return_shape(vctorch.mul_along_dim(w, t,
                                                     dim=mul_along_dim_dim),
                               v.shape(t))


v.pushthrough_impls.update({
    (p.stack_p, csp_p.mul_along_dim_p): push_stack_through_mul_along_dim
})


def push_cat_through_mul_along_dim(expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    mul_along_dim_dims = [child_expr.kwargs.get("dim", None)
                          for child_expr in expr.args[0]
                          if isinstance(child_expr, vp.Vexpr)
                          and child_expr.op == csp_p.mul_along_dim_p]
    assert all(dim == mul_along_dim_dims[0] for dim in mul_along_dim_dims)
    mul_along_dim_dim = mul_along_dim_dims[0]

    w = []
    t = []
    base = 0
    actual_indices = []
    identity = False
    for child_expr in expr.args[0]:
        n = v.shape(child_expr)[cat_dim]
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == csp_p.mul_along_dim_p:
            w.append(child_expr.args[0])
            t.append(child_expr.args[1])
            actual_indices += list(range(base, base + n))
        else:
            t.append(child_expr)
            identity = True
        base += n
    total_n = base

    w_shapes = [v.shape(w_expr) for w_expr in w]
    w = v._vectorize(v.with_return_shape(vtorch.cat(w, dim=-1),
                                         torch_cat_shape(w_shapes,
                                                         dim=-1)))
    if identity:
        ones_shape = v.shape(w)[:-1] + (total_n,)
        indices = torch.tensor(actual_indices)
        batch_shape = v.shape(w)[:-1]
        w = v.with_return_shape(
            maybe_lift_scatter(
                v.with_return_shape(vtorch.ones(ones_shape), ones_shape),
                -1,
                indices,
                w,
                batch_shape=batch_shape),
            ones_shape)

    t_shapes = [v.shape(t_expr) for t_expr in t]
    t = v._vectorize(v.with_return_shape(vtorch.cat(t, dim=cat_dim),
                                         torch_cat_shape(t_shapes,
                                                         dim=cat_dim)))

    return v.with_return_shape(
        vctorch.mul_along_dim(w, t, dim=mul_along_dim_dim),
        v.shape(t))

v.pushthrough_impls.update({
    (p.cat_p, csp_p.mul_along_dim_p): push_cat_through_mul_along_dim
})


def push_cat_through_index_reduction_into(
        index_reduction_into_p, parallel_reduction,
        expr, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    index_reduction_dims = [child_expr.args[1]
                            for child_expr in expr.args[0]
                            if isinstance(child_expr, vp.Vexpr)
                            and child_expr.op == index_reduction_into_p]
    assert all(dim == index_reduction_dims[0] for dim in index_reduction_dims)
    index_reduction_dim = index_reduction_dims[0]

    indices = []
    grandchildren = []
    base = 0
    for child_expr in expr.args[0]:
        child_shape = v.shape(child_expr)
        num_results = child_shape[cat_dim]
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == index_reduction_into_p:
            grandchildren.append(child_expr.args[3])
            indices.append(child_expr.args[2] + base)
            base += num_results
        else:
            grandchildren.append(child_expr)
            indices.append(torch.arange(base, base + num_results))
            base += num_results

    indices = torch.cat(indices)
    grandchildren = v._vectorize(vtorch.cat(grandchildren, dim=cat_dim))
    num_sums = base
    return v.with_return_shape(parallel_reduction(num_sums, index_reduction_dim,
                                                  indices, grandchildren),
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in
                                                expr.args[0]],
                                               dim=cat_dim))

def parallel_sum(num_sums, dim, index, source):
    return vctorch.index_add_into_zeros(num_sums, dim, index, source)

def parallel_prod(num_reductions, dim, index, source):
    return vctorch.index_reduce_into_ones(num_reductions, dim, index, source,
                                          "prod")

v.pushthrough_impls.update({
    (p.cat_p, csp_p.index_add_into_zeros_p): partial(
        push_cat_through_index_reduction_into, csp_p.index_add_into_zeros_p,
        parallel_sum),
    # TODO this is hardcoded to prod, but index reduce might use e.g. mean
    (p.cat_p, csp_p.index_reduce_into_ones_p): partial(
        push_cat_through_index_reduction_into, csp_p.index_reduce_into_ones_p,
        parallel_prod),
})


def push_concat_through_heads_tails(expr, allow_partial=True):
    assert expr.op == p.cat_p

    if not all(isinstance(child_expr, vp.Vexpr)
               and child_expr.op == csp_p.heads_tails_p
               for child_expr in expr.args[0]):
        print("Warning: giving up on pushing concat through heads_tails")
        return expr

    if len(expr.args[0]) == 1:
        return v._vectorize(expr.args[0][0])

    cat_dim = expr.kwargs.get("dim", 0)
    alphas = []
    for child_expr in expr.args[0]:
        if (not isinstance(child_expr, vp.Vexpr)
            or child_expr.op != csp_p.heads_tails_p):
            raise NotImplementedError()

        alpha = child_expr.args[0]
        alpha_shape = v.shape(alpha)
        if len(v.shape(alpha)) == 0:
            alpha = v.with_return_shape(
                v._vectorize(vtorch.stack([alpha], dim=cat_dim)),
                (1,))
        alphas.append(alpha)

    alphas = v._vectorize(v.with_return_shape(
        vtorch.cat(alphas, dim=cat_dim),
        torch_cat_shape([v.shape(alpha)
                         for alpha in alphas],
                        dim=cat_dim)
        ))
    return v.with_return_shape(vctorch.heads_tails(alphas),
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in expr.args[0]],
                                               dim=cat_dim))

v.pushthrough_impls.update({
    (p.cat_p, csp_p.heads_tails_p): push_concat_through_heads_tails
})


def push_shuffle_through_truediv(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == core.operator_truediv_p

    indices = expr.args[1]

    num, den = expr.args[0].args
    num = v.pushthrough(
        v.with_return_shape(vctorch.shuffle(num, indices, **expr.kwargs),
                            v.shape(num)),
        num.op)
    den = v.pushthrough(
        v.with_return_shape(vctorch.shuffle(den, indices, **expr.kwargs),
                            v.shape(den)),
        den.op)

    return v.with_return_shape(
        num / den,
        v.shape(expr.args[0])
    )

v.pushthrough_impls.update({
    (csp_p.shuffle_p, core.operator_truediv_p): push_shuffle_through_truediv,
})


def push_shuffle_through_mul_along_dim(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == csp_p.mul_along_dim_p

    indices = expr.args[1]

    w, t = expr.args[0].args
    w = v.pushthrough(
        v.with_return_shape(vctorch.shuffle(w, indices, dim=-1),
                            v.shape(w)),
        w.op)
    t = v.pushthrough(
        v.with_return_shape(vctorch.shuffle(t, indices, **expr.kwargs),
                            v.shape(t)),
        t.op)

    return v.with_return_shape(
        vctorch.mul_along_dim(w, t, **expr.args[0].kwargs),
        v.shape(expr.args[0])
    )

v.pushthrough_impls.update({
    (csp_p.shuffle_p, csp_p.mul_along_dim_p): push_shuffle_through_mul_along_dim,
})


def push_shuffle_through_index_select(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == p.index_select_p

    shuffle_indices = expr.args[1]
    shuffle_dim = expr.kwargs.get("dim", 0)

    input, dim, index = expr.args[0].args
    assert dim == shuffle_dim
    index = index.index_select(dim, shuffle_indices)

    return v.with_return_shape(
        vtorch.index_select(input, dim, index),
        v.shape(expr.args[0])
    )

v.pushthrough_impls.update({
    (csp_p.shuffle_p, p.index_select_p): push_shuffle_through_index_select,
})


def push_shuffle_through_unsqueeze(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == p.unsqueeze_p
    unsqueeze_expr = expr.args[0]
    return v.with_return_shape(
        vtorch.unsqueeze(
            v.single_pushthrough(
                vctorch.shuffle(unsqueeze_expr.args[0],
                                expr.args[1],
                                **expr.kwargs)),
            unsqueeze_expr.args[1]),
        v.shape(expr.args[0]))

v.pushthrough_impls.update({
    (csp_p.shuffle_p, p.unsqueeze_p): push_shuffle_through_unsqueeze,
})

def identity_pushthrough(expr, allow_partial=True):
    return expr

def destroy_shuffle_pushthrough(expr, allow_partial=True):
    return expr.args[0]

v.pushthrough_impls.update({
    (csp_p.shuffle_p, csp_p.cdist_multi_p): identity_pushthrough,
    (csp_p.shuffle_p, csp_p.sum_multi_p): identity_pushthrough,
    (csp_p.shuffle_p, csp_p.fast_prod_positive_multi_p): identity_pushthrough,
    (csp_p.shuffle_p, p.cat_p): identity_pushthrough,
    (csp_p.shuffle_p, p.zeros_p): destroy_shuffle_pushthrough,
    (csp_p.shuffle_p, p.ones_p): destroy_shuffle_pushthrough,
})

def push_shuffle_through_unary_elementwise(op, expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == op
    child_expr = expr.args[0]
    shape = v.shape(child_expr)
    return v.with_return_shape(
        vp.Vexpr(
            op,
            (v.single_pushthrough(
                v.with_return_shape(
                    vctorch.shuffle(child_expr.args[0],
                                    expr.args[1],
                                    **expr.kwargs),
                    shape)),
             ),
            child_expr.kwargs),
        shape)

def push_shuffle_through_scatter(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == p.scatter_p
    scatter_expr = expr.args[0]
    input, dim, index, src = scatter_expr.args

    if isinstance(input, vp.Vexpr):
        input = v.single_pushthrough(
            vctorch.shuffle(input, expr.args[1], **expr.kwargs))
    elif isinstance(input, torch.Tensor):
        input = input.index_select(expr.get("dim", 0), expr.args[1])
    else:
        raise ValueError(f"Unexpected input type {type(input)}")

    unshuffle_indices = invert_shuffle(expr.args[1])
    if isinstance(index, vp.Vexpr):
        index = v.pushthrough(
            vtorch.index_select(unshuffle_indices,
                                expr.kwargs.get("dim", 0), index),
            index.op
        )
    elif isinstance(index, torch.Tensor):
        index = unshuffle_indices.index_select(expr.get("dim", 0), index)
    else:
        raise ValueError(f"Unexpected index type {type(index)}")

    shape = v.shape(scatter_expr)
    return v.with_return_shape(
        vtorch.scatter(
            input,
            dim,
            index,
            src
        ),
        shape)

v.pushthrough_impls.update({
    (csp_p.shuffle_p, p.scatter_p): push_shuffle_through_scatter,
})

def push_shuffle_through_shuffle(expr, allow_partial=True):
    assert expr.op == csp_p.shuffle_p
    assert expr.args[0].op == csp_p.shuffle_p
    # Reuse logic that collapses shuffles.
    expr = maybe_shuffle(expr.args[0], expr.args[1], **expr.kwargs)
    if isinstance(expr.args[0], vp.Vexpr) and expr.op == csp_p.shuffle_p:
        return v.with_return_shape(v.single_pushthrough(expr),
                                   v.shape(expr))
    else:
        return expr

v.pushthrough_impls.update({
    (csp_p.shuffle_p, csp_p.shuffle_p): push_shuffle_through_shuffle,
})
