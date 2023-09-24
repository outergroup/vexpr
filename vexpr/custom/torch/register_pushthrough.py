from functools import partial

import torch

import vexpr as vp
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as csp_p
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.vectorization as v
from vexpr.torch.utils import (
    torch_cat_shape,
    torch_stack_shape,
    cat_remainder_then_combine,
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


def push_cat_through_cdist_multi(expr, allow_partial=True):
    assert expr.op == p.cat_p

    metric = next(child_expr.kwargs.get("p", 2)
                  for child_expr in expr.args[0]
                  if child_expr.op == csp_p.cdist_multi_p)

    cat_dim = expr.kwargs.get("dim", 0)

    left = []
    right = []
    lengths = []
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
            lengths += child_expr.kwargs["lengths"]
            axes.append(child_expr.kwargs.get("dim", None))
        else:
            remainder_indices += result_indices
            remainder.append(child_expr)
        base += num_indices

    canonicalized_axes = [(axis if axis is not None else 0)
                          for axis in axes]
    if not all(axis == canonicalized_axes[0]
               for axis in canonicalized_axes[1:]):
        raise ValueError("Expected same axes", axes)
    axis = axes[0]

    left = v._vectorize(vtorch.cat(left, dim=-1))
    right = v._vectorize(vtorch.cat(right, dim=-1))

    kwargs = dict(
        lengths=tuple(lengths),
        p=metric,
    )
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




# when concating multiple heads_tails, some might have an array of alphas while
# others have a single alpha. the trick, I suppose, is to stack the alphas
# before concating. so the pushthrough code needs to do a v.shape on the alpha,
# and for any that have ndim == 0, we do a stack pushthrough. Then we do a
# concat of all of them. Yeah.

# def push_concat_through_heads_tails(expr, allow_partial=True):
    
