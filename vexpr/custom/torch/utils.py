import torch

import vexpr as vp
import vexpr.vectorization as v
import vexpr.torch as vtorch
import vexpr.torch.primitives as p
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as cp


def maybe_index_select(input, dim, indices):
    dim_positive = (dim
                    if dim >= 0
                    else dim + len(v.shape(input)))

    finished = False
    while not finished:
        if input.op == p.index_select_p:
            assert input.args[1] == dim
            indices = input.args[2][indices]
            input = input.args[0]
        elif input.op == cp.shuffle_p:
            assert input.kwargs.get("dim", 0) == dim
            indices = input.args[1][indices]
            input = input.args[0]
        else:
            finished = True

    if torch.equal(indices, torch.arange(v.shape(input)[dim])):
        return input

    ret_shape = (v.shape(input)[:dim_positive]
                 + (len(indices),)
                 + v.shape(input)[dim_positive + 1:])
    ret = v.with_return_shape(vtorch.index_select(input, dim, indices),
                              ret_shape)
    return ret


def maybe_shuffle(expr, indices, dim=0):
    shape = v.shape(expr)

    # Fuse all shuffles
    while expr.op == cp.shuffle_p and expr.kwargs.get("dim", 0) == dim:
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
        if isinstance(result, vp.Vexpr) and result.op == cp.shuffle_p:
            indices = result.args[1]

            try:
                changed = False
                while True:
                    child = result.args[0]
                    if isinstance(child, vp.Vexpr):
                        child = v.lift(child, cp.shuffle_p)
                        if (child.op == cp.shuffle_p
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
