from vexpr import core
from .primitives import shuffle_p

def shuffle_impl(arr, indices, axis=0):
    selection = [slice(None)] * arr.ndim
    selection[axis] = indices
    return arr[selection]

core.eval_impls[shuffle_p] = shuffle_impl


def matern_impl(x, nu=2.5):
    pass