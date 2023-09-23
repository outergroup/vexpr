import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as cp
import vexpr.vectorization as v


def heads_tails_vectorize(expr):
    return v.with_return_shape(vctorch.heads_tails(v._vectorize(expr.args[0])),
                               v.shape(expr))

v.vectorize_impls[cp.heads_tails_p] = heads_tails_vectorize
