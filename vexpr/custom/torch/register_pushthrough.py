from functools import partial

import torch

import vexpr as vp
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as csp_p
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.vectorization as v
from vexpr.torch.utils import torch_concat_shape, torch_stack_shape


def push_concat_through_cdist_multi(expr, allow_partial=True):
    assert expr.op == p.concat_p
    assert all(child_expr.op == csp_p.cdist_multi_p for child_expr in expr.args[0])

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

    left = v._vectorize(vtorch.concat(left, dim=-1))
    right = v._vectorize(vtorch.concat(right, dim=-1))

    kwargs = dict(
        lengths=torch.concat(lengths),
        ps=torch.concat(ps),
    )
    if axis is not None:
        kwargs["dim"] = axis

    return v.with_return_shape(
        vctorch.cdist_multi(left, right, **kwargs),
        torch_concat_shape([v.shape(child_expr)
                            for child_expr in expr.args[0]],
                           dim=concat_axis))

v.pushthrough_impls[(p.concat_p, csp_p.cdist_multi_p)] = push_concat_through_cdist_multi



def push_concat_through_index_reduction_into(
        index_reduction_into_p, parallel_reduction,
        expr, allow_partial=True):
    assert expr.op == p.concat_p

    concat_dim = expr.kwargs.get("dim", 0)

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
        num_results = child_shape[concat_dim]
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
    grandchildren = v._vectorize(vtorch.concat(grandchildren, dim=concat_dim))
    num_sums = base
    return v.with_return_shape(parallel_reduction(num_sums, index_reduction_dim,
                                                  indices, grandchildren),
                               torch_concat_shape([v.shape(child_expr)
                                                   for child_expr in
                                                   expr.args[0]],
                                                  dim=concat_dim))

def parallel_sum(num_sums, dim, index, source):
    return vctorch.index_add_into_zeros(num_sums, dim, index, source)

def parallel_prod(num_reductions, dim, index, source):
    return vctorch.index_reduce_into_ones(num_reductions, dim, index, source,
                                          "prod")

v.pushthrough_impls.update({
    (p.concat_p, csp_p.index_add_into_zeros_p): partial(push_concat_through_index_reduction_into,
                                         csp_p.index_add_into_zeros_p,
                                         parallel_sum),
    # TODO this is hardcoded to prod, but index reduce might use e.g. mean
    (p.concat_p, csp_p.index_reduce_into_ones_p): partial(push_concat_through_index_reduction_into,
                                            csp_p.index_reduce_into_ones_p,
                                            parallel_prod),
})
