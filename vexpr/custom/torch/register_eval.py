import torch

from vexpr import core
from vexpr.torch.register_eval import allow_listlike_arg0
from . import primitives as p

def shuffle_impl(arr, indices, dim=0):
    with torch.profiler.record_function("shuffle"):
        selection = [slice(None)] * arr.ndim
        selection[dim] = indices
        return arr[selection]

core.eval_impls[p.shuffle_p] = shuffle_impl



def split_and_stack_impl(x, lengths, expanded_length, expanded_indices,
                         max_length, fill_value=0., split_dim=0, stack_dim=0):
    """
    Technically, this function receives redundant information. It receives
    the list of lengths because this makes it easier for the vectorizer to
    concatenate calls to split_and_pad, so that it doesn't have to reconstruct
    the lengths from the expanded_indices.
    """
    with torch.profiler.record_function("split_and_stack"):
        if split_dim < 0:
            split_dim += len(x.shape)
        if stack_dim < 0:
            stack_dim += len(x.shape)

        shape_after_expand = (*x.shape[:split_dim], len(lengths),
                              max_length,
                              *x.shape[split_dim + 1:])

        if expanded_length == x.shape[split_dim]:
            # no expansion needed
            x_expanded = x.view(shape_after_expand)
        else:
            expanded_shape = (*x.shape[:split_dim], expanded_length,
                              *x.shape[split_dim + 1:])

            x_expanded = x.new_full(expanded_shape, fill_value)
            selection = [slice(None)] * len(expanded_shape)
            selection[split_dim] = expanded_indices
            selection = tuple(selection)
            x_expanded[selection] = x

            x_expanded = x_expanded.view(shape_after_expand)

        if split_dim == stack_dim:
            return x_expanded
        else:
            # equivalent of x_expanded.moveaxis(-2, stack_dim), but with
            # permute. moveaxis is not supported with vmap, as of Pytorch 2.1.
            order = list(range(x_expanded.ndim))
            negative_index = -2 % x_expanded.ndim
            order.pop(negative_index)
            order.insert(stack_dim, negative_index)
            return x_expanded.permute(tuple(order))

core.eval_impls[p.split_and_stack_p] = split_and_stack_impl


def cdist_multi_impl(x1, x2, p, dim=0):
    with torch.profiler.record_function("cdist_multi"):
        if dim == 0:
            return torch.cdist(x1, x2, p=p)
        elif dim == -1:
            f = torch.vmap(torch.cdist, in_dims=(-2, -2), out_dims=dim)
            return f(x1, x2, p=p)
        else:
            raise NotImplementedError()

core.eval_impls[p.cdist_multi_p] = cdist_multi_impl


def sum_multi_impl(x, dim=None):
    with torch.profiler.record_function("sum_multi"):
        if dim is None:
            batch_dim = 0
        elif dim < 0:
            batch_dim = dim - 1
        else:
            batch_dim = dim

        # Use vmap so that "dim" can still be in the language of the sum. For
        # example, when performing a set of sums on tensors over dim 0, without
        # vmap we would make dim 0 a batch dimension and convert the dim to 1.
        f = torch.vmap(torch.sum, in_dims=batch_dim, out_dims=dim)
        return f(x, dim=dim)

core.eval_impls[p.sum_multi_p] = sum_multi_impl


def fast_prod_positive_impl(x, dim=None, epsilon=1e-10):
    with torch.profiler.record_function("fast_prod_positive"):
        # Much faster than torch.prod in the backward pass, since it doesn't require
        # a GPU synchronize.
        # https://twitter.com/mrcslws/status/1589721597396815873
        return x.clamp(min=epsilon).log().sum(dim=dim).exp()

core.eval_impls[p.fast_prod_positive_p] = allow_listlike_arg0(
    fast_prod_positive_impl)


def fast_prod_positive_multi_impl(x, dim=None, epsilon=1e-10):
    with torch.profiler.record_function("fast_prod_positive_multi"):
        if dim is None:
            batch_dim = 0
        elif dim < 0:
            batch_dim = dim - 1
        else:
            batch_dim = dim

        # Use vmap so that "dim" can still be in the language of the prod. For
        # example, when performing a set of prods on tensors over dim 0, without
        # vmap we would make dim 0 a batch dimension and convert the dim to 1.
        f = torch.vmap(fast_prod_positive_impl, in_dims=batch_dim,
                       out_dims=dim)
        return f(x, dim=dim, epsilon=epsilon)

core.eval_impls[p.fast_prod_positive_multi_p] = fast_prod_positive_multi_impl


def prod_multi_impl(x, dim=None):
    with torch.profiler.record_function("prod_multi"):
        if dim is None:
            batch_dim = 0
        elif dim < 0:
            batch_dim = dim - 1
        else:
            batch_dim = dim

        # Use vmap so that "dim" can still be in the language of the prod. For
        # example, when performing a set of prods on tensors over dim 0, without
        # vmap we would make dim 0 a batch dimension and convert the dim to 1.
        f = torch.vmap(torch.prod, in_dims=batch_dim, out_dims=dim)
        return f(x, dim=dim)

core.eval_impls[p.prod_multi_p] = prod_multi_impl


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
