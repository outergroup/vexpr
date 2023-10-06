import collections
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
    invert_shuffle,
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
        lengths += expr.kwargs["lengths"]
        children.append(expr.args[0])

    stack_dim = exprs[0].kwargs["stack_dim"]
    split_dim = exprs[0].kwargs["split_dim"]
    assert split_dim == dim

    children = v._vectorize(vtorch.cat(children, dim=dim))

    return vctorch.split_and_stack(
        children,
        **split_and_stack_kwargs(lengths,
                                 split_dim=split_dim,
                                 stack_dim=stack_dim)
    )


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

    child_pre_shuffles = []
    child_post_shuffles = []

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
            child_pre_shuffles.append(
                child_expr.kwargs.get("pre_shuffle_indices", None))
            child_post_shuffles.append(
                child_expr.kwargs.get("post_shuffle_indices", None))
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

    # Incorporate the previous shuffles
    if any(indices is not None for indices in child_pre_shuffles):
        all_child_pre_shuffle_indices = []
        base = 0
        for child_groups, shuffle_indices in zip(all_groups,
                                                 child_pre_shuffles):
            num_elements = sum(length * count
                               for (length, _), count in child_groups)
            if shuffle_indices is None:
                shuffle_indices = torch.arange(base, base + num_elements)
            else:
                shuffle_indices += base
            base += num_elements
            all_child_pre_shuffle_indices.append(shuffle_indices)
        all_child_pre_shuffle_indices = torch.cat(all_child_pre_shuffle_indices)

        if pre_shuffle_indices is None:
            pre_shuffle_indices = all_child_pre_shuffle_indices
        else:
            pre_shuffle_indices = all_child_pre_shuffle_indices[pre_shuffle_indices]

    if any(indices is not None for indices in child_post_shuffles):
        all_child_post_shuffle_indices = []
        base = 0
        for child_groups, shuffle_indices in zip(all_groups,
                                                 child_post_shuffles):
            num_elements = sum(count
                               for _, count in child_groups)
            if shuffle_indices is None:
                shuffle_indices = torch.arange(base, base + num_elements)
            else:
                shuffle_indices += base
            base += num_elements
            all_child_post_shuffle_indices.append(shuffle_indices)
        all_child_post_shuffle_indices = torch.cat(all_child_post_shuffle_indices)

        if post_shuffle_indices is None:
            post_shuffle_indices = all_child_post_shuffle_indices
        else:
            post_shuffle_indices = post_shuffle_indices[all_child_post_shuffle_indices]

    if torch.equal(pre_shuffle_indices, torch.arange(len(pre_shuffle_indices))):
        pre_shuffle_indices = None

    if torch.equal(post_shuffle_indices,
                   torch.arange(len(post_shuffle_indices))):
        post_shuffle_indices = None

    kwargs = dict(
        groups = groups,
    )
    if pre_shuffle_indices is not None:
        kwargs["pre_shuffle_indices"] = pre_shuffle_indices
    if post_shuffle_indices is not None:
        kwargs["post_shuffle_indices"] = post_shuffle_indices
    if "dim" in expr.kwargs:
        kwargs["stack_dim"] = expr.kwargs["dim"]

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
                                            **split_and_stack_kwargs(
                                                lengths,
                                                split_dim=reduction_dim,
                                                stack_dim=reduction_dim),
                                            fill_value=fill_value)

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
    w = v._vectorize(v.with_return_shape(vtorch.cat(w, dim=cat_dim),
                                            torch_cat_shape(w_shapes,
                                                            dim=cat_dim)))
    if identity:
        ones_shape = (total_n,)
        w = v.with_return_shape(
            vtorch.scatter(
                v.with_return_shape(vtorch.ones(ones_shape), ones_shape),
                0,
                torch.tensor(actual_indices),
                w),
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
