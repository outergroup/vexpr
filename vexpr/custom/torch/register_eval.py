import torch

from vexpr import core
from . import primitives as p

def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        selection = [slice(None)] * arr.ndim
        selection[dim] = indices
        return arr[selection]

core.eval_impls[p.shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, p, lengths, dim=0):
    with torch.profiler.record_function("cdist_multi"):
        max_length = max(lengths)
        n = max_length * len(lengths)

        indices = []
        base = 0
        for length in lengths:
            indices.append(torch.arange(base, base + length))
            base += max_length
        indices = torch.cat(indices)

        x1_ = x1.new_zeros((*x1.shape[:-1], n))
        x1_[..., indices] = x1
        x1_ = x1_.view((*x1.shape[:-1], len(lengths), max_length))

        x2_ = x2.new_zeros((*x2.shape[:-1], n))
        x2_[..., indices] = x2
        x2_ = x2_.view((*x2.shape[:-1], len(lengths), max_length))

        f = torch.vmap(torch.cdist, in_dims=(-2, -2), out_dims=dim)

        return f(x1_, x2_, p=p)

core.eval_impls[p.cdist_multi_p] = cdist_multi_impl

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
