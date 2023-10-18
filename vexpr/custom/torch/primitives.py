from vexpr.core import _p_and_constructor

shuffle_p, shuffle = _p_and_constructor("custom.torch.shuffle")
cdist_multi_p, cdist_multi = _p_and_constructor("custom.torch.cdist_multi")
fast_prod_positive_p, fast_prod_positive = _p_and_constructor(
    "custom.torch.fast_prod_positive")
fast_prod_positive_multi_p, fast_prod_positive_multi = _p_and_constructor(
    "custom.torch.fast_prod_positive_multi")
sum_multi_p, sum_multi = _p_and_constructor("custom.torch.sum_multi")
prod_multi_p, prod_multi = _p_and_constructor("custom.torch.prod_multi")

mul_along_dim_p, mul_along_dim = _p_and_constructor(
    "custom.torch.mul_along_dim")

heads_tails_p, heads_tails = _p_and_constructor("custom.torch.heads_tails")

__all__ = [
    "shuffle",
    "cdist_multi",
    "fast_prod_positive",
    "fast_prod_positive_multi",
    "sum_multi",
    "prod_multi",
    "mul_along_dim",
    "heads_tails",
]
