import torch

def split_and_stack_kwargs(lengths):
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
    )
