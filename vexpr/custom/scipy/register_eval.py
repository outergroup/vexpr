import numpy as np
import scipy

from vexpr import core
from .primitives import cdist_multi_p

def cdist_multi_impl(x1, x2, lengths, axis=0):
    base = 0
    results = []
    for length in lengths:
        results.append(scipy.spatial.distance.cdist(x1[..., base:base+length],
                                                    x2[..., base:base+length]))
        base += length
    return np.stack(results, axis=axis)

core.eval_impls[cdist_multi_p] = cdist_multi_impl
