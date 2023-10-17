import collections
from functools import partial

import torch

import vexpr as vp
import vexpr.core as core
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.impls as cimpls
import vexpr.custom.torch.primitives as cp
import vexpr.torch as vtorch
import vexpr.torch.impls as impls
import vexpr.torch.primitives as p
import vexpr.vectorization as v
from vexpr.custom.torch.utils import (
    maybe_index_select,
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


def identity(x): return x


def shuffle_pushthrough(expr, transform=identity):
    op = expr.args[0].op
    impl = cimpls.push_shuffle_through_op.get(op, None)
    if impl is None:
        print("No shuffle pushthrough support for", op)
        raise v.CannotVectorize
    return impl(expr, transform)


def mul_along_dim_pushthrough(expr, transform=identity):
    op = expr.args[1].op
    impl = cimpls.push_mul_along_dim_through_op.get(op, None)
    if impl is None:
        print("No mul_along_dim pushthrough support for", op)
        raise v.CannotVectorize
    return impl(expr, transform)


def push_cat_through_shuffle(expr, transform=identity, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    # get a list of shuffles with the same dim by wrapping everything else with
    # identity shuffles
    child_exprs = [(child_expr
                    if child_expr.op == cp.shuffle_p
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

    ret = transform(
        v.with_return_shape(
            vtorch.cat(grandchildren, **expr.kwargs),
            result_shape
        )
    )
    indices = torch.cat(indices)

    return transform(
        maybe_shuffle(ret, indices, **expr.kwargs, transform=transform)
    )


def push_cat_through_cdist_multi(expr, transform=identity, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    all_groups = [child_expr.kwargs["groups"]
                  for child_expr in expr.args[0]
                  if child_expr.op == cp.cdist_multi_p]

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
        if child_expr.op == cp.cdist_multi_p:
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

    left = transform(vtorch.cat(left, dim=-1))
    right = transform(vtorch.cat(right, dim=-1))

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

    left = maybe_shuffle(left, pre_shuffle_indices, -1, transform)
    right = maybe_shuffle(right, pre_shuffle_indices, -1, transform)

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
    applicable = transform(
        v.with_return_shape(
            vctorch.cdist_multi(left, right, **kwargs),
            result_shape)
    )

    applicable = transform(
        maybe_shuffle(applicable, post_shuffle_indices,
                      cat_dim, transform)
    )

    return transform(
        cat_remainder_then_combine(
            applicable,
            remainder,
            applicable_indices,
            remainder_indices,
            **expr.kwargs)
    )


def push_cat_through_reduction_multi(reduction_multi_p, parallel_reduction,
                                     fill_value, expr, transform=identity,
                                     allow_partial=True):
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

    grandchildren = transform(vtorch.cat(grandchildren, dim=cat_dim))

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
                                  reduction_dim, transform)

    result_shape = torch_cat_shape([v.shape(child_expr)
                                    for child_expr in expr.args[0]],
                                   dim=cat_dim)
    result = v.with_return_shape(parallel_reduction(grandchildren,
                                                    groups=groups, dim=reduction_dim),
                                 result_shape)

    return maybe_shuffle(result, post_shuffle_indices, cat_dim, transform)



def push_stack_through_mul_along_dim(expr, transform=identity, allow_partial=True):
    assert expr.op == p.stack_p

    stack_dim = expr.kwargs.get("dim", 0)

    mul_along_dim_dims = [child_expr.kwargs.get("dim", None)
                          for child_expr in expr.args[0]
                          if isinstance(child_expr, vp.Vexpr)
                          and child_expr.op == cp.mul_along_dim_p]
    assert all(dim == mul_along_dim_dims[0] for dim in mul_along_dim_dims)
    mul_along_dim_dim = mul_along_dim_dims[0]

    w = []
    t = []
    identity = False
    actual_indices = []
    for i, child_expr in enumerate(expr.args[0]):
        if isinstance(child_expr, vp.Vexpr) \
           and child_expr.op == cp.mul_along_dim_p:
            w.append(child_expr.args[0])
            t.append(child_expr.args[1])
            actual_indices.append(i)
        else:
            identity = True
            t.append(child_expr)

    kwargs = {}
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]

    w = transform(
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

    t = transform(
        v.with_return_shape(
            vtorch.stack(t, **kwargs),
            torch_stack_shape2([v.shape(d) for d in t],
                               **expr.kwargs)))

    dim = expr.kwargs.get("dim", 0)
    ndim = len(v.shape(t))
    assert canonical_axis(dim, ndim) == canonical_axis(mul_along_dim_dim, ndim)

    ret = transform(
        v.with_return_shape(vctorch.mul_along_dim(w, t,
                                                  dim=mul_along_dim_dim),
                            v.shape(t))
    )

    return ret


def push_cat_through_mul_along_dim(expr, transform=identity, allow_partial=True):
    assert expr.op == p.cat_p

    cat_dim = expr.kwargs.get("dim", 0)

    mul_along_dim_dims = [child_expr.kwargs.get("dim", None)
                          for child_expr in expr.args[0]
                          if isinstance(child_expr, vp.Vexpr)
                          and child_expr.op == cp.mul_along_dim_p]
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
           and child_expr.op == cp.mul_along_dim_p:
            w.append(child_expr.args[0])
            t.append(child_expr.args[1])
            actual_indices += list(range(base, base + n))
        else:
            t.append(child_expr)
            identity = True
        base += n
    total_n = base

    w_shapes = [v.shape(w_expr) for w_expr in w]
    w = transform(v.with_return_shape(vtorch.cat(w, dim=-1),
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
    t = transform(v.with_return_shape(vtorch.cat(t, dim=cat_dim),
                                         torch_cat_shape(t_shapes,
                                                         dim=cat_dim)))

    ret = transform(
        v.with_return_shape(
            vctorch.mul_along_dim(w, t, dim=mul_along_dim_dim),
            v.shape(t))
    )

    return ret


def push_concat_through_heads_tails(expr, transform=identity, allow_partial=True):
    assert expr.op == p.cat_p

    if not all(isinstance(child_expr, vp.Vexpr)
               and child_expr.op == cp.heads_tails_p
               for child_expr in expr.args[0]):
        print("Warning: giving up on pushing concat through heads_tails")
        return expr

    if len(expr.args[0]) == 1:
        return transform(expr.args[0][0])

    cat_dim = expr.kwargs.get("dim", 0)
    alphas = []
    for child_expr in expr.args[0]:
        if (not isinstance(child_expr, vp.Vexpr)
            or child_expr.op != cp.heads_tails_p):
            raise NotImplementedError()

        alpha = child_expr.args[0]
        alpha_shape = v.shape(alpha)
        if len(v.shape(alpha)) == 0:
            alpha = v.with_return_shape(
                transform(vtorch.stack([alpha], dim=cat_dim)),
                (1,))
        alphas.append(alpha)

    alphas = transform(v.with_return_shape(
        vtorch.cat(alphas, dim=cat_dim),
        torch_cat_shape([v.shape(alpha)
                         for alpha in alphas],
                        dim=cat_dim)
        ))
    return v.with_return_shape(vctorch.heads_tails(alphas),
                               torch_cat_shape([v.shape(child_expr)
                                                for child_expr in expr.args[0]],
                                               dim=cat_dim))


def push_shuffle_through_truediv(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == core.operator_truediv_p

    indices = expr.args[1]

    num, den = expr.args[0].args
    num = transform(
        v.with_return_shape(vctorch.shuffle(num, indices, **expr.kwargs),
                            v.shape(num)))
    den = transform(
        v.with_return_shape(vctorch.shuffle(den, indices, **expr.kwargs),
                            v.shape(den)))

    return v.with_return_shape(
        num / den,
        v.shape(expr.args[0])
    )


def push_shuffle_through_mul_along_dim(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == cp.mul_along_dim_p

    indices = expr.args[1]

    w, t = expr.args[0].args
    w = transform(
        v.with_return_shape(vctorch.shuffle(w, indices, dim=-1),
                            v.shape(w)))
    t = transform(
        v.with_return_shape(vctorch.shuffle(t, indices, **expr.kwargs),
                            v.shape(t)))

    return v.with_return_shape(
        vctorch.mul_along_dim(w, t, **expr.args[0].kwargs),
        v.shape(expr.args[0])
    )


def push_shuffle_through_index_select(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == p.index_select_p

    shuffle_indices = expr.args[1]
    shuffle_dim = expr.kwargs.get("dim", 0)

    input, dim, index = expr.args[0].args
    assert dim == shuffle_dim
    index = index.index_select(dim, shuffle_indices)

    return maybe_index_select(input, dim, index)


def push_shuffle_through_unsqueeze(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == p.unsqueeze_p
    unsqueeze_expr = expr.args[0]
    return v.with_return_shape(
        vtorch.unsqueeze(
            transform(
                vctorch.shuffle(unsqueeze_expr.args[0],
                                expr.args[1],
                                **expr.kwargs)),
            unsqueeze_expr.args[1]),
        v.shape(expr.args[0]))



def identity_pushthrough(expr, transform=identity, allow_partial=True):
    return expr

def destroy_shuffle_pushthrough(expr, transform=identity, allow_partial=True):
    return expr.args[0]


def push_shuffle_through_unary_elementwise(op, expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == op
    child_expr = expr.args[0]
    shape = v.shape(child_expr)
    return v.with_return_shape(
        vp.Vexpr(
            op,
            (transform(
                v.with_return_shape(
                    vctorch.shuffle(child_expr.args[0],
                                    expr.args[1],
                                    **expr.kwargs),
                    shape)),
             ),
            child_expr.kwargs),
        shape)

def push_shuffle_through_scatter(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == p.scatter_p
    scatter_expr = expr.args[0]
    input, dim, index, src = scatter_expr.args

    if isinstance(input, vp.Vexpr):
        input = transform(
            vctorch.shuffle(input, expr.args[1], **expr.kwargs))
    elif isinstance(input, torch.Tensor):
        input = input.index_select(expr.get("dim", 0), expr.args[1])
    else:
        raise ValueError(f"Unexpected input type {type(input)}")

    expand = False
    if isinstance(index, vp.Vexpr):
        assert index.op == p.expand_p
        expand = True
        prev_expand_shape = index.args[1]
        index = index.args[0].view(-1)

    unshuffle_indices = invert_shuffle(expr.args[1])
    if isinstance(index, torch.Tensor):
        index = unshuffle_indices.index_select(expr.kwargs.get("dim", 0), index)
    else:
        raise ValueError(f"Unexpected index type {type(index)}")

    if expand:
        expand_shape = prev_expand_shape[:-1] + (len(index),)
        view_shape = (1,) * (len(expand_shape) - 1) + (len(index),)
        index = v.with_return_shape(
            vtorch.expand(index.view(view_shape), expand_shape),
            expand_shape)

    shape = v.shape(scatter_expr)
    return v.with_return_shape(
        vtorch.scatter(
            input,
            dim,
            index,
            src
        ),
        shape)


def push_shuffle_through_shuffle(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.shuffle_p
    assert expr.args[0].op == cp.shuffle_p
    # Reuse logic that collapses shuffles.
    expr = maybe_shuffle(expr.args[0], expr.args[1], **expr.kwargs,
                         transform=transform)
    if isinstance(expr.args[0], vp.Vexpr) and expr.op == cp.shuffle_p:
        return v.with_return_shape(transform(expr),
                                   v.shape(expr))
    else:
        return expr


def raise_cannot_vectorize(expr, transform=identity, allow_partial=True):
    raise v.CannotVectorize


def push_mul_along_dim_through_mul_along_dim(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.mul_along_dim_p
    w, t = expr.args
    assert t.op == cp.mul_along_dim_p

    outer_dim = expr.kwargs.get("dim", 0)
    inner_dim = expr.kwargs.get("dim", 0)
    assert outer_dim == inner_dim

    inner_w, inner_t = t.args

    expr = v.with_return_shape(
        vctorch.mul_along_dim(
            v.with_return_shape(
                vctorch.mul_along_dim(
                    w, inner_w,
                    dim=-1),
                torch.broadcast_shapes(v.shape(w),
                                       v.shape(inner_w))
            ),
            inner_t,
            **t.kwargs
        ),
        v.shape(inner_t))

    try:
        expr = transform(expr)
    except v.CannotVectorize:
        # we've successfully merged two mul_along_dim operations, so this
        # pushthrough had a benefit. Catch the exception here and don't rethrow.
        pass
    return expr


def push_mul_along_dim_through_sum(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.mul_along_dim_p
    w, t = expr.args
    assert t.op == p.sum_p

    sum_dim = t.kwargs.get("dim", None)
    mul_along_dim_dim = expr.kwargs.get("dim", 0)
    assert sum_dim == mul_along_dim_dim

    sum_children = t.args[0]

    # it would sometimes be faster to do vtorch.unsqueeze(w, -1) instead of the
    # expand below, but other pushthrough logic wants to be able to select
    # weights by index.
    expand_shape = v.shape(w) + (v.shape(sum_children)[sum_dim],)
    multiplied_children = transform(
        v.with_return_shape(
            vctorch.mul_along_dim(
                v.with_return_shape(
                    vtorch.expand(
                        v.with_return_shape(
                            vtorch.unsqueeze(w, -1),
                            v.shape(w) + (1,)
                        ),
                        expand_shape),
                    expand_shape
                ),
                sum_children,
                **expr.kwargs),
            v.shape(sum_children)
        )
    )

    expr = v.with_return_shape(
        vtorch.sum(multiplied_children, **t.kwargs),
        v.shape(expr)
    )

    sum_children = expr.args[0]
    while isinstance(sum_children, vp.Vexpr) and sum_children.op == cp.sum_multi_p:
        # eat this sum_multi, it's all just one sum now
        sum_children = sum_children.args[0]
        expr = v.with_return_shape(
            vtorch.sum(sum_children, **t.kwargs),
            v.shape(expr)
        )

        # then do a shuffle lift and destroy it, since order doesn't matter
        # during sum
        try:
            sum_children = v.lift(expr.args[0], cp.shuffle_p)
            while sum_children.op == cp.shuffle_p:
                sum_children = sum_children.args[0]
                expr = v.with_return_shape(
                    vtorch.sum(sum_children, **t.kwargs),
                    v.shape(expr)
                )
                sum_children = v.lift(sum_children, cp.shuffle_p)
        except v.CannotVectorize:
            pass

    return expr


def push_mul_along_dim_through_sum_multi(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.mul_along_dim_p
    w, t = expr.args
    assert t.op == cp.sum_multi_p

    # Repeat weights using index_select. We could try changing this to
    # repeat_interleave.
    indices = []
    i = 0
    for d, n in t.kwargs["groups"]:
        for _ in range(n):
            indices += [i] * d
            i += 1
    indices = torch.tensor(indices)
    w = maybe_index_select(w, -1, indices)

    sum_children = transform(
        v.with_return_shape(
            vctorch.mul_along_dim(
                w,
                t.args[0],
                **expr.kwargs),
            v.shape(t.args[0])
        )
    )

    ret = v.with_return_shape(
        vctorch.sum_multi(
            sum_children,
            **t.kwargs),
        v.shape(expr))

    # TODO now check if the children are a sum_multi and fold them in if so

    # TODO and that should include if the next reduction is a sum_multi. so this
    # is an opportunity to crush any shuffles in the middle.
    return ret


def push_mul_along_dim_through_fast_prod_positive_multi(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.mul_along_dim_p
    w, t = expr.args
    assert t.op == cp.fast_prod_positive_multi_p

    base = 0
    scatter_indices = []
    for d, n in t.kwargs["groups"]:
        # insert to the first element of every prod
        scatter_indices.append(torch.arange(base, base + d * n, d))
        base += d * n
    tot = base
    scatter_indices = torch.cat(scatter_indices)
    batch_shape = v.shape(w)[:-1]
    ones_shape = batch_shape + (tot,)
    w = maybe_lift_scatter(
        v.with_return_shape(
            vtorch.ones(ones_shape),
            ones_shape),
        -1, scatter_indices, w, batch_shape=batch_shape)

    prod_children = transform(
        v.with_return_shape(
            vctorch.mul_along_dim(
                w,
                t.args[0],
                **expr.kwargs),
            v.shape(t.args[0])
        )
    )

    ret = v.with_return_shape(
        vctorch.fast_prod_positive_multi(
            prod_children,
            **t.kwargs),
        v.shape(expr))

    # TODO now check if the children are a fast_prod_positive_multi and fold
    # them in if so
    return ret


def push_mul_along_dim_through_shuffle(expr, transform=identity, allow_partial=True):
    assert expr.op == cp.mul_along_dim_p
    w, t = expr.args
    assert t.op == cp.shuffle_p

    # apply inverted shuffle to w
    w = v.with_return_shape(
        vctorch.shuffle(w, invert_shuffle(t.args[1]), dim=-1),
        v.shape(w))

    expr = transform(
        v.with_return_shape(
            vctorch.mul_along_dim(w, t.args[0], **expr.kwargs),
            v.shape(expr))
    )

    return v.with_return_shape(
        vctorch.shuffle(expr, t.args[1], **t.kwargs),
        v.shape(expr))


def register_elementwise_op(op):
    cimpls.push_shuffle_through_op.update({
        op: partial(push_shuffle_through_unary_elementwise, op),
    })
    # cimpls.push_mul_along_dim_through_op.update({
    #     op: partial(push_shuffle_through_unary_elementwise, op),
    # })


v.unary_elementwise_registration_steps.append(register_elementwise_op)

impls.push_stack_through_op.update({
    cp.fast_prod_positive_p: partial(
        push_stack_through_reduction, cp.fast_prod_positive_p,
        vctorch.fast_prod_positive_multi, 1.),
    cp.fast_prod_positive_p: partial(
        push_stack_through_reduction, cp.fast_prod_positive_p,
        vctorch.fast_prod_positive_multi, 1.),
    cp.mul_along_dim_p: push_stack_through_mul_along_dim,
})

impls.push_cat_through_op.update({
    cp.shuffle_p: push_cat_through_shuffle,
    cp.cdist_multi_p: push_cat_through_cdist_multi,
    cp.sum_multi_p: partial(
        push_cat_through_reduction_multi, cp.sum_multi_p, vctorch.sum_multi, 0.),
    cp.prod_multi_p: partial(
        push_cat_through_reduction_multi, cp.prod_multi_p, vctorch.prod_multi, 1.),
    cp.fast_prod_positive_multi_p: partial(
        push_cat_through_reduction_multi, cp.fast_prod_positive_multi_p,
        vctorch.fast_prod_positive_multi, 1.),
    cp.mul_along_dim_p: push_cat_through_mul_along_dim,
    cp.heads_tails_p: push_concat_through_heads_tails,
})

cimpls.push_shuffle_through_op.update({
    core.operator_truediv_p: push_shuffle_through_truediv,
    cp.mul_along_dim_p: push_shuffle_through_mul_along_dim,
    p.index_select_p: push_shuffle_through_index_select,
    cp.cdist_multi_p: identity_pushthrough,
    cp.sum_multi_p: identity_pushthrough,
    cp.fast_prod_positive_multi_p: identity_pushthrough,
    p.cat_p: identity_pushthrough,
    p.zeros_p: destroy_shuffle_pushthrough,
    p.ones_p: destroy_shuffle_pushthrough,
    p.scatter_p: push_shuffle_through_scatter,
    cp.shuffle_p: push_shuffle_through_shuffle,
    p.unsqueeze_p: push_shuffle_through_unsqueeze,
})

cimpls.push_mul_along_dim_through_op.update({
    cp.mul_along_dim_p: push_mul_along_dim_through_mul_along_dim,
    p.sum_p: push_mul_along_dim_through_sum,
    cp.cdist_multi_p: raise_cannot_vectorize,
    p.exp_p: raise_cannot_vectorize,
    p.cat_p: raise_cannot_vectorize,
    cp.sum_multi_p: push_mul_along_dim_through_sum_multi,
    cp.fast_prod_positive_multi_p: push_mul_along_dim_through_fast_prod_positive_multi,
    cp.shuffle_p: push_mul_along_dim_through_shuffle,
})

v.phase_ops[1] += [
    (cp.shuffle_p, shuffle_pushthrough),
]

v.phase_ops[2] += [
    (cp.mul_along_dim_p, mul_along_dim_pushthrough),
]
