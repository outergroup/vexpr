import torch

from vexpr import core
from .primitives import shuffle_p, cdist_multi_p

def shuffle_impl(arr, indices, axis=0):
    selection = [slice(None)] * arr.ndim
    selection[axis] = indices
    return arr[selection]

core.eval_impls[shuffle_p] = shuffle_impl


def cdist_multi_impl(x1, x2, ps, lengths, dim=0):
    base = 0
    results = []
    for p, length in zip(ps, lengths):
        results.append(torch.cdist(x1[..., base:base+length],
                                   x2[..., base:base+length],
                                   p=p))
        base += length
    return torch.stack(results, dim=dim)

core.eval_impls[cdist_multi_p] = cdist_multi_impl
