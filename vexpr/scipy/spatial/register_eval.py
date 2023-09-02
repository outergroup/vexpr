import numpy as np
import scipy

import vexpr.core
from . import primitives as p


vexpr.core.eval_impls.update({
    p.cdist_p: scipy.spatial.distance.cdist
})
