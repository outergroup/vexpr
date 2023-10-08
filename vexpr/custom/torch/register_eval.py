from functools import partial

import torch

from vexpr import core
from vexpr.torch.register_eval import allow_listlike_arg0
from vexpr.torch.utils import torch_stack_shape
from . import primitives as p

# TODO remove this and rely on index_select, merging all of their vectorization
# logic. (It won't be faster, but it creates a rising tide to lift all boats.)
def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        return arr.index_select(dim, indices)

core.eval_impls[p.shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, groups, stack_dim=0):
    with torch.profiler.record_function("cdist_multi"):
        if stack_dim >= 0:
            batch_dim = stack_dim
        else:
            batch_dim = len(x1.shape[:-2]) + stack_dim + 1

        ret = []
        splits = [d * n for (d, p), n in groups]
        for ((d, metric), n), x1_, x2_ in zip(groups,
                                              x1.split(splits, dim=-1),
                                              x2.split(splits, dim=-1)):
            x1_ = x1_.view(x1_.shape[:-1] + (n, d)).movedim(-2, batch_dim)
            x2_ = x2_.view(x2_.shape[:-1] + (n, d)).movedim(-2, batch_dim)
            ret.append(torch.cdist(x1_, x2_, p=metric))

        ret = torch.cat(ret, dim=batch_dim)
        if stack_dim != batch_dim:
            ret = ret.movedim(batch_dim, stack_dim)

        return ret


core.eval_impls[p.cdist_multi_p] = cdist_multi_impl


def fast_prod_positive_impl(x, dim=None, epsilon=1e-10):
    with torch.profiler.record_function("fast_prod_positive"):
        # Much faster than torch.prod in the backward pass, since it doesn't require
        # a GPU synchronize.
        # https://twitter.com/mrcslws/status/1589721597396815873
        return x.clamp(min=epsilon).log().sum(dim=dim).exp()


core.eval_impls[p.fast_prod_positive_p] = allow_listlike_arg0(
    fast_prod_positive_impl)


def reduction_multi(reduction_f, name, x, groups, dim):
    with torch.profiler.record_function(name):
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


core.eval_impls.update({
    p.sum_multi_p: partial(reduction_multi, torch.sum, "sum_multi"),
    p.prod_multi_p: partial(reduction_multi, torch.prod, "prod_multi"),
    p.fast_prod_positive_multi_p: partial(
        reduction_multi, fast_prod_positive_impl, "fast_prod_positive_multi"),
})


def mul_along_dim_impl(w, t, dim=0):
    new_shape = [1] * t.dim()
    new_shape[dim] = len(w)
    return w.view(new_shape) * t

core.eval_impls[p.mul_along_dim_p] = mul_along_dim_impl


def index_add_into_zeros_impl(n_sums, dim, index, source, *args, **kwargs):
    with torch.profiler.record_function("index_add_into_zeros"):
        shape = list(source.shape)
        shape[dim] = n_sums
        shape = tuple(shape)
        return source.new_zeros(shape).index_add_(dim, index, source, *args,
                                                  **kwargs)

core.eval_impls[p.index_add_into_zeros_p] = index_add_into_zeros_impl


def index_reduce_into_ones_impl(n_reductions, dim, index, source, *args,
                                **kwargs):
    with torch.profiler.record_function("index_reduce_into_ones"):
        shape = list(source.shape)
        shape[dim] = n_reductions
        shape = tuple(shape)
        ret = torch.ones(shape, dtype=source.dtype, device=source.device)
        return ret.index_reduce(dim, index, source, *args, **kwargs)

core.eval_impls[p.index_reduce_into_ones_p] = index_reduce_into_ones_impl


def heads_tails_impl(alpha):
    with torch.profiler.record_function("heads_tails"):
        if isinstance(alpha, torch.Tensor) and alpha.ndim > 0:
            # Build [alpha1, 1-alpha1, alpha2, 1-alpha2, ...]
            return torch.stack([alpha, 1.0 - alpha], dim=-1).view(*alpha.shape[:-1], -1)
        else:
            return torch.stack([alpha, 1.0 - alpha])

core.eval_impls[p.heads_tails_p] = heads_tails_impl
