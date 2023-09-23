import torch

import vexpr.torch as vtorch
import vexpr.custom.torch as vctorch
import vexpr.vectorization as v

def torch_stack_shape(initial_shape, num_elements, dim=0):
    if dim < 0:
        dim += len(initial_shape) + 1
    return initial_shape[:dim] + (num_elements,) + initial_shape[dim:]


def torch_stack_shape2(shapes, dim=0):
    assert all(shape == shapes[0] for shape in shapes)
    return torch_stack_shape(shapes[0], len(shapes), dim)


def torch_cat_shape(child_shapes, dim=0):
    if dim < 0:
        dim += len(child_shapes[0])
    return (child_shapes[0][:dim]
            + (sum(shape[dim] for shape in child_shapes),)
            + child_shapes[0][dim + 1:])


def invert_shuffle(indices):
    inverted_indices = torch.zeros_like(torch.as_tensor(indices))
    inverted_indices[indices] = torch.arange(len(indices))
    return inverted_indices


def maybe_shuffle(expr, scrambled_indices, **kwargs):
    if torch.equal(torch.as_tensor(scrambled_indices),
                   torch.arange(len(scrambled_indices))):
        # the shuffle would be a no-op
        return expr
    else:
        indices = invert_shuffle(scrambled_indices)
        return v.with_return_shape(
            vctorch.shuffle(expr, indices, **kwargs),
            v.shape(expr))


def stack_remainder_then_combine(applicable, remainder, applicable_indices,
                                 remainder_indices, **stack_kwargs):
    if len(remainder) == 0:
        return applicable

    remainder = v._vectorize(
        v.with_return_shape(
            vtorch.stack(remainder, **stack_kwargs),
            torch_stack_shape2([v.shape(r_expr)
                                for r_expr in remainder],
                               **stack_kwargs)
        )
    )

    result_shape = torch_cat_shape([v.shape(applicable), v.shape(remainder)],
                                   **stack_kwargs)

    result = v.with_return_shape(
        vtorch.cat([applicable, remainder], **stack_kwargs),
        result_shape)

    return maybe_shuffle(result, applicable_indices + remainder_indices,
                         **stack_kwargs)


def cat_remainder_then_combine(applicable, remainder, applicable_indices,
                               remainder_indices, **cat_kwargs):
    if len(remainder) == 0:
        return applicable

    remainder = v._vectorize(
        v.with_return_shape(
            vtorch.cat(remainder, **cat_kwargs),
            torch_cat_shape([v.shape(r_expr)
                             for r_expr in remainder],
                            **cat_kwargs)
        )
    )

    result_shape = torch_cat_shape([v.shape(applicable), v.shape(remainder)],
                                   **cat_kwargs)
    result = v.with_return_shape(vtorch.cat([applicable, remainder],
                                            **cat_kwargs),
                                 result_shape)

    return maybe_shuffle(result, applicable_indices + remainder_indices,
                         **cat_kwargs)
