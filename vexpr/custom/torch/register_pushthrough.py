from functools import partial

import torch

import vexpr as vp
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as csp_p
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.vectorization as v
from vexpr.custom.torch.utils import (
    split_and_stack_kwargs,
)
from vexpr.torch.utils import (
    torch_cat_shape,
    torch_stack_shape,
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

    if not torch.equal(indices, torch.arange(len(indices))):
        ret = v.with_return_shape(vctorch.shuffle(ret, indices, **expr.kwargs),
                                  result_shape)

    return ret


v.pushthrough_impls[(p.cat_p, csp_p.shuffle_p)] = push_cat_through_shuffle


def combine_split_and_stack(exprs, dim=0):
    assert all(expr.op == csp_p.split_and_stack_p
               for expr in exprs)

    lengths = []
    children = []
    for expr in exprs:
        assert expr.op == csp_p.split_and_stack_p
        lengths += expr.kwargs["lengths"]
        children.append(expr.args[0])

    children = v._vectorize(vtorch.cat(children, dim=dim))

    return vctorch.split_and_stack(
        children,
        **split_and_stack_kwargs(lengths),
        dim=dim,
    )


def push_cat_through_cdist_multi(expr, allow_partial=True):
    assert expr.op == p.cat_p

    metric = next(child_expr.kwargs.get("p", 2)
                  for child_expr in expr.args[0]
                  if child_expr.op == csp_p.cdist_multi_p)

    cat_dim = expr.kwargs.get("dim", 0)

    left = []
    right = []
    axes = []

    applicable_exprs = []
    applicable_indices = []
    remainder = []
    remainder_indices = []
    base = 0
    for child_expr in expr.args[0]:
        num_indices = v.shape(child_expr)[cat_dim]
        result_indices = list(range(base, base + num_indices))
        if (child_expr.op == csp_p.cdist_multi_p
            and child_expr.kwargs.get("p", 2) == metric):
            applicable_indices += result_indices
            applicable_exprs.append(child_expr)
            left.append(child_expr.args[0])
            right.append(child_expr.args[1])
            axes.append(child_expr.kwargs.get("dim", None))
        else:
            if not allow_partial:
                raise v.CannotVectorize()
            remainder_indices += result_indices
            remainder.append(child_expr)
        base += num_indices

    canonicalized_axes = [(axis if axis is not None else 0)
                          for axis in axes]
    if not all(axis == canonicalized_axes[0]
               for axis in canonicalized_axes[1:]):
        raise ValueError("Expected same axes", axes)
    axis = axes[0]

    left = combine_split_and_stack(left, dim=-1)
    right = combine_split_and_stack(right, dim=-1)

    kwargs = dict(p=metric)
    if axis is not None:
        kwargs["dim"] = axis

    return cat_remainder_then_combine(
        v.with_return_shape(
            vctorch.cdist_multi(left, right, **kwargs),
            torch_cat_shape([v.shape(child_expr)
                             for child_expr in applicable_exprs],
                            dim=cat_dim)),
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

    lengths = []
    grandchildren = []
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == reduction_multi_p:
            split_and_stack_expr = child_expr.args[0]
            assert split_and_stack_expr.op == csp_p.split_and_stack_p
            lengths += split_and_stack_expr.kwargs["lengths"]
            grandchildren.append(split_and_stack_expr.args[0])
        else:
            grandchildren.append(child_expr)
            lengths += [1] * v.shape(child_expr)[cat_dim]

    grandchildren = v._vectorize(vtorch.cat(grandchildren, dim=cat_dim))
    grandchildren = vctorch.split_and_stack(grandchildren,
                                            **split_and_stack_kwargs(lengths),
                                            fill_value=fill_value,
                                            dim=reduction_dim)

    return v.with_return_shape(parallel_reduction(grandchildren,
                                                  dim=reduction_dim),
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in expr.args[0]],
                                               dim=cat_dim))

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

    assert all(isinstance(child_expr, vp.Vexpr)
               and child_expr.op == csp_p.heads_tails_p
               for child_expr in expr.args[0])

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
