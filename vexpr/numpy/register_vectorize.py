from functools import partial

import vexpr as vp
import vexpr.core as core
import vexpr.numpy as vnp
import vexpr.numpy.primitives as p
import vexpr.vectorization as v

def stack_vectorize(expr):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))
    vexpr_ops = [op for op in vexpr_ops if op != core.symbol_p]

    if len(vexpr_ops) > 0:
        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return v.pushthrough(expr, op,
                                            allow_partial=allow_partial)
                except v.CannotVectorize:
                    pass

    return expr


# TODO: obviously, understand which of these impls should be the same.
def concatenate_vectorize(expr):
    # get unique list of ops, preserving order
    vexpr_ops = list(dict.fromkeys(v.op
                                   for v in expr.args[0]
                                   if isinstance(v, vp.Vexpr)))

    vexpr_ops = [op for op in vexpr_ops if op != core.symbol_p]

    # TODO: in general, every user-provided stack (and maybe concatenate, and
    # other things?) should be pushed through before trying other stuff. they
    # just cause confusion. One solution is to do a first pass that is bottom
    # up, performing a vectorize on each stack, starting from the leaf nodes.
    # another solution is to add special logic to ~every passthrough that tries
    # to eliminate certain ops from the children before proceeding with specific
    # logic.


    # TODO if any child ops are a concatenate with the same axis, absorb their
    # children

    # TODO: for any adjacent numpy arrays, perform a concat immediately?

    if len(vexpr_ops) > 0:
        # first pass, detect any stacks and vectorize them, attempting to make them disappear.
        if p.stack_p in vexpr_ops:
            child_exprs = []
            changed = False
            for child_expr in expr.args[0]:
                if child_expr.op == p.stack_p:
                    try:
                        child_expr2 = v._vectorize(child_expr)
                        # TODO find a faster way of doing this. probably make
                        # _vectorize indicate whether anything changed, perhaps
                        # always throwing CannotVectorize if no change occurs.
                        if child_expr != child_expr2:
                            child_expr = child_expr2
                            changed = True
                    except v.CannotVectorize:
                        pass
                child_exprs.append(child_expr)
            if changed:
                expr = vnp.concatenate(child_exprs, **expr.kwargs)
                return concatenate_vectorize(expr)

        for allow_partial in (False, True):
            for op in vexpr_ops:
                try:
                    return v.pushthrough(expr, op,
                                         allow_partial=allow_partial)
                except v.CannotVectorize:
                    pass


    return expr

def reduction_vectorize(op, expr):
    assert expr.op == op
    # TODO handle axis?
    child_expr = expr.args[0]
    if not isinstance(child_expr, core.Vexpr):
        child_expr = vnp.stack(expr.args[0])
    child_expr = v._vectorize(child_expr)
    kwargs = {}
    if "axis" in expr.kwargs:
        kwargs["axis"] = expr.kwargs["axis"]
    return core.Vexpr(op, (child_expr,), kwargs)

v.vectorize_impls.update({
    p.stack_p: stack_vectorize,
    p.concatenate_p: concatenate_vectorize,
    p.sum_p: partial(reduction_vectorize, p.sum_p),
    p.prod_p: partial(reduction_vectorize, p.prod_p),
})
