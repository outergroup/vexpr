from functools import partial

import vexpr as vp
import vexpr.torch as vtorch
import vexpr.core as core
import vexpr.torch.primitives as p
import vexpr.custom.torch.primitives as cp
import vexpr.vectorization
from vexpr.vectorization import _vectorize

PRIORITIZED_OPS = set([
    p.stack_p, p.cat_p,
    p.sum_p, p.prod_p, core.operator_add_p,
    core.operator_mul_p, core.operator_truediv_p,
    core.operator_matmul_p,
    p.index_add_p, p.index_reduce_p,
    cp.index_add_into_zeros_p, cp.index_reduce_into_ones_p,
    cp.sum_multi_p, cp.prod_multi_p, cp.fast_prod_positive_p,
    cp.fast_prod_positive_multi_p,
    cp.shuffle_p,
])


def stack_vectorize(expr):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))
    vexpr_ops = [op for op in vexpr_ops if op != core.symbol_p]
    vexpr_ops = ([op for op in vexpr_ops if op in PRIORITIZED_OPS]
                 + [op for op in vexpr_ops if op not in PRIORITIZED_OPS])

    if len(vexpr_ops) > 0:
        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return vexpr.vectorization.pushthrough(expr, op,
                                                           allow_partial=allow_partial)
                except vexpr.vectorization.CannotVectorize:
                    pass

    return expr


# TODO: obviously, understand which of these impls should be the same.
def cat_vectorize(expr):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op
                                   for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))

    vexpr_ops = [op for op in vexpr_ops if op != core.symbol_p]
    vexpr_ops = ([op for op in vexpr_ops if op in PRIORITIZED_OPS]
                 + [op for op in vexpr_ops if op not in PRIORITIZED_OPS])

    # TODO if any child ops are a cat with the same dim, absorb their
    # children

    # TODO: for any adjacent numpy arrays, perform a cat immediately?

    if len(vexpr_ops) > 0:
        # first pass, detect any stacks and vectorize them, attempting to make them disappear.
        if p.stack_p in vexpr_ops:
            child_exprs = []
            changed = False
            for child_expr in expr.args[0]:
                if child_expr.op == p.stack_p:
                    try:
                        child_expr2 = _vectorize(child_expr)
                        # TODO find a faster way of doing this. probably make
                        # _vectorize indicate whether anything changed, perhaps
                        # always throwing CannotVectorize if no change occurs.
                        if child_expr != child_expr2:
                            child_expr = child_expr2
                            changed = True
                    except vexpr.vectorization.CannotVectorize:
                        pass
                child_exprs.append(child_expr)
            if changed:
                return cat_vectorize(vtorch.cat(child_exprs, **expr.kwargs))

        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return vexpr.vectorization.pushthrough(
                        expr, op, allow_partial=allow_partial)
                except vexpr.vectorization.CannotVectorize:
                    pass

    if len(expr.args[0]) == 1:
        # If cat-ing one item, just return it.
        return expr.args[0][0]
    else:
        return expr

def reduction_vectorize(op, expr):
    assert expr.op == op
    # TODO handle dim?
    child_expr = expr.args[0]
    if not isinstance(child_expr, core.Vexpr):
        child_expr = vtorch.stack(expr.args[0])
    child_expr = _vectorize(child_expr)
    kwargs = {}
    if "dim" in expr.kwargs:
        kwargs["dim"] = expr.kwargs["dim"]
    return core.Vexpr(op, (child_expr,), kwargs)

vexpr.vectorization.vectorize_impls.update({
    p.stack_p: stack_vectorize,
    p.cat_p: cat_vectorize,
    p.sum_p: partial(reduction_vectorize, p.sum_p),
    p.prod_p: partial(reduction_vectorize, p.prod_p),
})
