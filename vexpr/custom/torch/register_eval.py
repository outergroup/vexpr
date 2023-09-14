import torch

from vexpr import core
from . import primitives as p

def shuffle_impl(arr, indices, axis=0):
    selection = [slice(None)] * arr.ndim
    selection[axis] = indices
    return arr[selection]

core.eval_impls[p.shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, ps, lengths, dim=0):
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
    shape = list(source.shape)
    shape[dim] = n_sums
    shape = tuple(shape)
    ret = torch.zeros(shape)
    return ret.index_add(dim, index, source, *args, **kwargs)

core.eval_impls[p.index_add_into_zeros_p] = index_add_into_zeros_impl


def index_reduce_into_ones_impl(n_reductions, dim, index, source, *args, **kwargs):
    shape = list(source.shape)
    shape[dim] = n_reductions
    shape = tuple(shape)
    ret = torch.ones(shape)
    return ret.index_reduce(dim, index, source, *args, **kwargs)

core.eval_impls[p.index_reduce_into_ones_p] = index_reduce_into_ones_impl