import torch

import vexpr.vectorization as v
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as ct_p


def maybe_shuffle(expr, indices, dim=0):
    shape = v.shape(expr)

    # Fuse all shuffles
    while expr.op == ct_p.shuffle_p and expr.kwargs.get("dim", 0) == dim:
        indices = expr.args[1][indices]
        expr = expr.args[0]

    if torch.equal(indices, torch.arange(len(indices))):
        result = expr
    else:
        result = vctorch.shuffle(expr, indices, dim=dim)
        try:
            result = v.pushthrough(result, expr.op)
        except v.CannotVectorize:
            pass

    return v.with_return_shape(result, shape)
