import numpy as np

import vexpr.numpy as vnp
import vexpr.numpy.impls as impls
import vexpr.numpy.primitives as np_p
import vexpr.custom.scipy.primitives as csp_p
import vexpr.vectorization as v
from vexpr import core
from vexpr.custom.scipy import cdist_multi
from vexpr.numpy.utils import (
    np_concatenate_shape,
)

from . import primitives as p


def identity(x): return x


def push_stack_through_cdist(expr, transform=identity, allow_partial=True):
    assert expr.op == np_p.stack_p
    assert all(child_expr.op == p.cdist_p for child_expr in expr.args[0])

    # TODO process this
    stack_axis = expr.kwargs.get("axis", 0)

    left = []
    right = []
    lengths = []
    child_matrix_shapes = []
    for child_expr in expr.args[0]:
        left.append(child_expr.args[0])
        right.append(child_expr.args[1])

        shape = v.shape(child_expr.args[0])
        length = shape[-1]
        lengths.append(length)
        child_matrix_shapes.append(v.shape(child_expr))

    left = transform(
        v.with_return_shape(
            vnp.concatenate(left, axis=-1),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in left],
                                 -1)
        )
    )
    right = transform(
        v.with_return_shape(
            vnp.concatenate(right, axis=-1),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in right],
                                 -1)
        )
    )

    kwargs = dict(
        lengths=np.array(lengths)
    )
    if "axis" in expr.kwargs:
        kwargs["axis"] = expr.kwargs["axis"]

    # Compute shape of result
    child_matrix_shape = child_matrix_shapes[0]
    assert all(shape == child_matrix_shape for shape in child_matrix_shapes[1:])

    axis = expr.kwargs.get("axis", 0)
    # use axis to determine result shape after stack
    result_shape = child_matrix_shape
    if axis < 0:
        axis += len(result_shape) + 1
    result_shape = result_shape[:axis] + (len(lengths),) + result_shape[axis:]

    return v.with_return_shape(
        cdist_multi(left, right, **kwargs),
        result_shape)


def push_concatenate_through_cdist_multi(expr, transform=identity, allow_partial=True):
    assert expr.op == np_p.concatenate_p
    assert all(child_expr.op == csp_p.cdist_multi_p for child_expr in expr.args[0])

    # TODO process this
    concat_axis = expr.kwargs.get("axis", 0)

    left = []
    right = []
    lengths = []
    axes = []
    for child_expr in expr.args[0]:
        left.append(child_expr.args[0])
        right.append(child_expr.args[1])
        lengths.append(child_expr.kwargs["lengths"])
        axes.append(child_expr.kwargs.get("axis", None))

    canonicalized_axes = [(axis if axis is not None else 0)
                          for axis in axes]
    if not all(axis == canonicalized_axes[0]
               for axis in canonicalized_axes[1:]):
        raise ValueError("Expected same axes", axes)
    axis = axes[0]

    left = transform(
        v.with_return_shape(
            vnp.concatenate(left, axis=-1),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in left],
                                 -1)
        )
    )
    right = transform(
        v.with_return_shape(
            vnp.concatenate(right, axis=-1),
            np_concatenate_shape([v.shape(child_expr)
                                  for child_expr in right],
                                 -1)
        )
    )

    kwargs = dict(
        lengths=np.concatenate(lengths)
    )
    if axis is not None:
        kwargs["axis"] = axis

    return v.with_return_shape(
        cdist_multi(left, right, **kwargs),
        np_concatenate_shape([v.shape(child_expr)
                              for child_expr in expr.args[0]],
                             axis=concat_axis)
    )


impls.push_stack_through_op.update({
    p.cdist_p: push_stack_through_cdist
})
impls.push_concatenate_through_op.update({
    csp_p.cdist_multi_p: push_concatenate_through_cdist_multi
})
