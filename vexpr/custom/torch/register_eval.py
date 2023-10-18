import torch

from vexpr import core
import vexpr.torch.primitives as p
from vexpr.torch.register_eval import allow_listlike_arg0
from vexpr.torch.utils import torch_stack_shape
from . import primitives as cp


# TODO remove this and rely on index_select, merging all of their vectorization
# logic. (It won't be faster, but it creates a rising tide to lift all boats.)
def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        return arr.index_select(dim, indices)

core.eval_impls[cp.shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, groups):
    with torch.profiler.record_function("cdist_multi"):
        ret = []
        splits = [d * n for (d, p), n in groups]
        for ((d, metric), n), x1_, x2_ in zip(groups,
                                              x1.split(splits, dim=-1),
                                              x2.split(splits, dim=-1)):
            x1_ = x1_.view(x1_.shape[:-1] + (n, d)).movedim(-2, -3)
            x2_ = x2_.view(x2_.shape[:-1] + (n, d)).movedim(-2, -3)
            ret.append(torch.cdist(x1_, x2_, p=metric))

        ret = torch.cat(ret, dim=-3)

        return ret


core.eval_impls[cp.cdist_multi_p] = cdist_multi_impl


def fast_prod_positive_impl(x, dim=None, epsilon=1e-10):
    with torch.profiler.record_function("fast_prod_positive"):
        # Much faster than torch.prod in the backward pass, since it doesn't require
        # a GPU synchronize.
        # https://twitter.com/mrcslws/status/1589721597396815873
        return x.clamp(min=epsilon).log().sum(dim=dim).exp()


core.eval_impls[cp.fast_prod_positive_p] = allow_listlike_arg0(
    fast_prod_positive_impl)


def reduction_multi(reduction_f, x, groups, dim):
    if dim < 0:
        dim += x.ndim

    ret = []
    splits = [d * n for d, n in groups]
    for (d, n), x_ in zip(groups, x.split(splits, dim=dim)):
        if d == 1:
            ret.append(x_)
        else:
            view_shape = x_.shape[:dim] + (n, d) + x_.shape[dim + 1:]
            x_ = x_.view(view_shape)
            ret.append(reduction_f(x_, dim=dim+1))

    return torch.cat(ret, dim=dim)


def sum_multi_impl(x, groups, dim):
    with torch.profiler.record_function("sum_multi"):
        return reduction_multi(torch.sum, x, groups, dim)


def prod_multi_impl(x, groups, dim):
    with torch.profiler.record_function("prod_multi"):
        return reduction_multi(torch.prod, x, groups, dim)


def fast_prod_positive_multi_impl(x, groups, dim, epsilon=1e-10):
    with torch.profiler.record_function("fast_prod_positive_multi"):
        # return reduction_multi(
        #     partial(fast_prod_positive_impl, epsilon=epsilon), x, groups, dim
        # )

        # Below is equivalent to above. Using functools.partial causes
        # torch.compile to split out into a new CompiledFunction (as of PyTorch
        # 2.1)
        if dim < 0:
            dim += x.ndim

        ret = []
        splits = [d * n for d, n in groups]
        for (d, n), x_ in zip(groups, x.split(splits, dim=dim)):
            if d == 1:
                ret.append(x_)
            else:
                view_shape = x_.shape[:dim] + (n, d) + x_.shape[dim + 1:]
                x_ = x_.view(view_shape)
                ret.append(fast_prod_positive_impl(x_, dim=dim+1,
                                                   epsilon=epsilon))

        return torch.cat(ret, dim=dim)


core.eval_impls.update({
    # Ideally these would be implemented with functools.partial, but this causes
    # torch.compile to split out into a new CompiledFunction (as of PyTorch 2.1)
    cp.sum_multi_p: sum_multi_impl,
    cp.prod_multi_p: prod_multi_impl,
    cp.fast_prod_positive_multi_p: fast_prod_positive_multi_impl,
})


def mul_along_dim_impl(w, t, dim=0):
    with torch.profiler.record_function("mul_along_dim"):
        new_shape = [1] * t.dim()
        w_n = len(w.shape)
        if dim < 0:
            dim += t.dim()
        new_shape[dim - w_n + 1 : dim + 1] = w.shape
        return w.view(new_shape) * t

core.eval_impls[cp.mul_along_dim_p] = mul_along_dim_impl


def heads_tails_impl(alpha):
    with torch.profiler.record_function("heads_tails"):
        if isinstance(alpha, torch.Tensor) and alpha.ndim > 0:
            # Build [alpha1, 1-alpha1, alpha2, 1-alpha2, ...]
            return torch.stack([alpha, 1.0 - alpha], dim=-1).view(
                *alpha.shape[:-1], -1)
        else:
            return torch.stack([alpha, 1.0 - alpha])

core.eval_impls[cp.heads_tails_p] = heads_tails_impl
