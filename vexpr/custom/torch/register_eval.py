import torch

from vexpr import core
from . import primitives as p

def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        selection = [slice(None)] * arr.ndim
        selection[dim] = indices
        return arr[selection]

core.eval_impls[p.shuffle_p] = shuffle_impl



def split_and_stack_impl(x, lengths, expanded_length, expanded_indices,
                         max_length, dim=0):
    """
    Technically, this function receives redundant information. It receives
    the list of lengths because this makes it easier for the vectorizer to
    concatenate calls to split_and_pad, so that it doesn't have to reconstruct
    the lengths from the expanded_indices.
    """
    if dim != -1:
        raise NotImplementedError()

    with torch.profiler.record_function("split_and_stack"):
        final_shape = (*x.shape[:-1], len(lengths), max_length)
        if expanded_length == x.shape[dim]:
            # no expansion needed
            return x.view(final_shape)

        x_expanded = x.new_zeros((*x.shape[:-1], expanded_length))
        x_expanded[..., expanded_indices] = x
        return x_expanded.view(final_shape)

core.eval_impls[p.split_and_stack_p] = split_and_stack_impl


def cdist_multi_impl(x1, x2, p, dim=0):
    with torch.profiler.record_function("cdist_multi"):
        f = torch.vmap(torch.cdist, in_dims=(-2, -2), out_dims=dim)
        return f(x1, x2, p=p)

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
