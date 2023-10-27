import collections
from functools import partial

import torch

import vexpr as vp
import vexpr.primitives as vpp
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as cp
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.vectorization as v


def lift_return_self(expr):
    return expr

v.lift_impls.update({
    (cp.cdist_multi_p, cp.shuffle_p): lift_return_self,
    (cp.sum_multi_p, cp.shuffle_p): lift_return_self,
    (cp.fast_prod_positive_multi_p, cp.shuffle_p): lift_return_self,
    (cp.shuffle_p, cp.shuffle_p): lift_return_self,
})


def lift_shuffle_from_cat(expr):
    assert expr.op == p.cat_p
    # for each child, determine if there's a shuffle. if there are any shuffles,
    # lift them, provide an implicit shuffle for each other child, concatenate
    # them, and wrap it around the cat.
    child_exprs = []
    found_shuffle = False
    for child_expr in expr.args[0]:
        if isinstance(child_expr, vp.Vexpr):
            child_expr = v.lift(child_expr, cp.shuffle_p)
            if child_expr.op == cp.shuffle_p:
                found_shuffle = True
        child_exprs.append(child_expr)

    if found_shuffle:
        cat_dim = expr.kwargs.get("dim", 0)
        base = 0
        outer_indices = []
        altered_child_exprs = []
        for child_expr in child_exprs:
            shape = v.shape(child_expr)
            n = shape[cat_dim]
            if isinstance(child_expr, vp.Vexpr) and child_expr.op == cp.shuffle_p:
                outer_indices.append(base + child_expr.args[1])
                altered_child_exprs.append(child_expr.args[0])
            else:
                outer_indices.append(torch.arange(base, base + n))
                altered_child_exprs.append(child_expr)
            base += n
        outer_indices = torch.cat(outer_indices)
        expr = v.with_return_shape(
            vctorch.shuffle(
                v.with_return_shape(
                    vtorch.cat(altered_child_exprs, **expr.kwargs),
                    v.shape(expr)
                ),
                outer_indices,
                **expr.kwargs
            ),
            v.shape(expr)
        )

    return expr

v.lift_impls.update({
    (p.cat_p, cp.shuffle_p): lift_shuffle_from_cat,
})


def lift_shuffle_from_unary_elementwise(op, expr):
    assert expr.op == op
    # do a pullthrough on the child. if it successfully pulls through a shuffle,
    # pull it up one level higher.
    child = expr.args[0]
    child = v.lift(child, cp.shuffle_p)
    if child.op == cp.shuffle_p:
        expr = v.with_return_shape(
            vctorch.shuffle(
                v.with_return_shape(
                    vp.Vexpr(op, (child.args[0],), expr.kwargs),
                    v.shape(expr)
                ),
                child.args[1],
                **child.kwargs
            ),
            v.shape(expr))

    return expr


def register_elementwise_op(op):
    v.lift_impls[(op, cp.shuffle_p)] = partial(
        lift_shuffle_from_unary_elementwise, op
    )


v.unary_elementwise_registration_steps.append(register_elementwise_op)

v.lift_impls.update({
    (p.exp_p, cp.shuffle_p): partial(lift_shuffle_from_unary_elementwise,
                                     p.exp_p),
    (vpp.operator_neg_p, cp.shuffle_p): partial(
        lift_shuffle_from_unary_elementwise, vpp.operator_neg_p),
})
