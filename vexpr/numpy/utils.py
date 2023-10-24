def np_stack_shape(initial_shape, num_elements, axis=0):
    if axis < 0:
        axis += len(initial_shape) + 1
    return (initial_shape[:axis]
            + type(initial_shape)([num_elements,])
            + initial_shape[axis:])


def np_stack_shape2(shapes, axis=0):
    assert all(shape == shapes[0] for shape in shapes)
    return np_stack_shape(shapes[0], len(shapes), axis)


def np_concatenate_shape(child_shapes, axis=0):
    if axis < 0:
        axis += len(child_shapes[0])
    return (child_shapes[0][:axis]
            + (sum(shape[axis] for shape in child_shapes),)
            + child_shapes[0][axis + 1:])
