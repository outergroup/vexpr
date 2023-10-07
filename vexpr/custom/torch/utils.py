import torch

import vexpr.vectorization as v
import vexpr.custom.torch as vctorch
import vexpr.custom.torch.primitives as ct_p


def maybe_shuffle(expr, indices, dim=0):
    shape = v.shape(expr)

    # Fuse all shuffles
    while expr.op == ct_p.shuffle_p and expr.kwargs.get("dim", 0) == dim:
        indices = expr.args[1][indices]
        expr = expr.args[0]

    if torch.equal(indices, torch.arange(len(indices))):
        result = expr
    else:
        result = vctorch.shuffle(expr, indices, dim=dim)

    return v.with_return_shape(result, shape)


def split_and_stack_kwargs(lengths, split_dim, stack_dim):
    max_length = max(lengths)
    expanded_length = max_length * len(lengths)

    expanded_indices = []
    base = 0
    for length in lengths:
        expanded_indices.append(torch.arange(base, base + length))
        base += max_length
    expanded_indices = torch.cat(expanded_indices)

    return dict(
        lengths=lengths,
        expanded_length=expanded_length,
        expanded_indices=expanded_indices,
        max_length=max_length,
        split_dim=split_dim,
        stack_dim=stack_dim,
    )


def split_and_stack_kwargs2(lengths, split_dim, stack_dim):
    max_length = max(lengths)
    expanded_length = max_length * len(lengths)

    stack_dim_indices = []
    split_dim_indices = []
    for i, length in enumerate(lengths):
        stack_dim_indices += [i] * length
        split_dim_indices.append(torch.arange(length))
    stack_dim_indices = torch.tensor(stack_dim_indices)
    split_dim_indices = torch.cat(split_dim_indices)

    return dict(
        lengths=lengths,
        expanded_length=expanded_length,
        stack_dim_indices=stack_dim_indices,
        split_dim_indices=split_dim_indices,
        max_length=max_length,
        split_dim=split_dim,
        stack_dim=stack_dim,
    )
