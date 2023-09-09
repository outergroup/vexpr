from vexpr.core import _p_and_constructor

shuffle_p, shuffle = _p_and_constructor("custom.torch.shuffle")
cdist_multi_p, cdist_multi = _p_and_constructor("custom.torch.cdist_multi")

__all__ = [
    "shuffle",
    "cdist_multi",
]
