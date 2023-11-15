from functools import partial

from . import primitives as p
from .core import with_metadata, VexprWithMetadata
from .transformations import bottom_up_transform


# Analogous to type annotations, used for visualization.
def visual_type(expr, t):
    return with_metadata(expr, dict(visual_type=t))

mixing_weight = partial(visual_type, t="mixing_weight")
location = partial(visual_type, t="location")
scale = partial(visual_type, t="scale")


def comment(expr, comment):
    return with_metadata(expr, dict(comment=comment))


optimize_impls = {}
def optimize_(expr):
    impl = optimize_impls.get(expr.op, None)
    return impl(expr) if impl is not None else expr


type_impls = {}
def propagate_type_info_(expr):
    impl = type_impls.get(expr.op, None)
    if impl is None:
        return expr
    t = impl(expr)
    if t is None:
        return expr
    return visual_type(expr, t)


optimize = partial(bottom_up_transform, optimize_)
propagate_types = partial(bottom_up_transform, propagate_type_info_)


def getitem_type(expr):
    source = expr.args[0]
    if isinstance(source, VexprWithMetadata):
        return source.metadata.get("visual_type", None)
    return None


type_impls.update({
    p.operator_getitem_p: getitem_type,
})
