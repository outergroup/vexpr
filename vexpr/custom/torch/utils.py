import torch

import vexpr as vp
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

        # If this is still a shuffle, try to lift a descendent shuffle and merge
        # it into this one. Repeat until no more are lifted. This happens, for
        # example, when there is a shuffle underneath a vtorch.cat. We can't
        # generally push a shuffle through a cat and have it still be a cat, but
        # we can pull a shuffle out.
        if isinstance(result, vp.Vexpr) and result.op == ct_p.shuffle_p:
            indices = result.args[1]

            try:
                changed = False
                while True:
                    child = result.args[0]
                    if isinstance(child, vp.Vexpr):
                        child = v.lift(child, ct_p.shuffle_p)
                        if (child.op == ct_p.shuffle_p
                            and (child.kwargs.get("dim", 0)
                                 == result.kwargs.get("dim", 0))):
                            indices = child.args[1][indices]
                            result = vctorch.shuffle(child.args[0],
                                                     indices,
                                                     **result.kwargs)
                            changed = True
                            continue
                    break

                if changed:
                    if torch.equal(indices, torch.arange(len(indices))):
                        result = result.args[0]

            except v.CannotVectorize:
                pass

    return v.with_return_shape(result, shape)
