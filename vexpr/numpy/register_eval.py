import numpy as np

from vexpr import core
from . import primitives as p

core.shape_impls.update({
    np.ndarray: np.shape,
})

core.shape_impls.update({
    dtype: np.shape
    for _, dtypes in np.sctypes.items()
    for dtype in dtypes
})

core.eval_impls.update({
    p.zeros_p: np.zeros,
    p.ones_p: np.ones,
    p.stack_p: np.stack,
    p.concatenate_p: np.concatenate,
    p.exp_p: np.exp,
    p.log_p: np.log,
    p.sum_p: np.sum,
    p.prod_p: np.prod,
})

def add_at_impl(a, indices, b):
    np.add.at(a, indices, b)
    return a
def prod_at_impl(a, indices, b):
    np.prod.at(a, indices, b)
    return a
core.eval_impls.update({
    p.add_at_p: add_at_impl,
    p.prod_at_p: prod_at_impl,
})
