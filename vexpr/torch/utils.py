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
