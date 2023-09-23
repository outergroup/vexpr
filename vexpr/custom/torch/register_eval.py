import torch

from vexpr import core
from . import primitives as p

def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        selection = [slice(None)] * arr.ndim
        selection[dim] = indices
        return arr[selection]

core.eval_impls[p.shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, ps, lengths, dim=0):
    with torch.profiler.record_function("cdist_multi"):
        base = 0
        results = []
        for p, length in zip(ps, lengths):
            results.append(torch.cdist(x1[..., base:base+length],
                                       x2[..., base:base+length],
                                       p=p))
            base += length
        return torch.stack(results, dim=dim)

core.eval_impls[p.cdist_multi_p] = cdist_multi_impl

def index_add_into_zeros_impl(n_sums, dim, index, source, *args, **kwargs):
    with torch.profiler.record_function("index_add_into_zeros"):
        shape = list(source.shape)
        shape[dim] = n_sums
        shape = tuple(shape)
        ret = torch.zeros(shape, dtype=source.dtype, device=source.device)
        return ret.index_add(dim, index, source, *args, **kwargs)

core.eval_impls[p.index_add_into_zeros_p] = index_add_into_zeros_impl


def index_reduce_into_ones_impl(n_reductions, dim, index, source, *args, **kwargs):
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
