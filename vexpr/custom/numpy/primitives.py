from vexpr.core import _p_and_constructor

shuffle_p, shuffle = _p_and_constructor("custom.numpy.shuffle")
matern_p, matern = _p_and_constructor("custom.numpy.matern")

__all__ = [
    "shuffle",
    "matern",
]
