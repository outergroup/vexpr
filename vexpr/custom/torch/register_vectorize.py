import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as cp
import vexpr.vectorization as v


def heads_tails_vectorize(expr):
    return v.with_return_shape(vctorch.heads_tails(v._vectorize(expr.args[0])),
                               v.shape(expr))

v.vectorize_impls[cp.heads_tails_p] = heads_tails_vectorize


def mul_along_dim_vectorize(expr):
    ret = v.with_return_shape(
        vctorch.mul_along_dim(v._vectorize(expr.args[0]),
                              v._vectorize(expr.args[1]),
                              **expr.kwargs),
        v.shape(expr))

    # now push mul_along_dim through as low as it can go.
    try:
        ret = v.pushthrough(ret, ret.args[1].op)
    except v.CannotVectorize:
        pass

    return ret

v.vectorize_impls[cp.mul_along_dim_p] = mul_along_dim_vectorize
