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

# calls index_add, sizing the return value according to the input value's shape
# this is like doing a vtorch.index_add(vtorch.zeros(...), ...), but the shape
# of the zeros is dynamically chosen.
index_add_into_zeros_p, index_add_into_zeros = _p_and_constructor("custom.torch.index_add_into_zeros")
index_reduce_into_ones_p, index_reduce_into_ones = _p_and_constructor("custom.torch.index_reduce_into_ones")

heads_tails_p, heads_tails = _p_and_constructor("custom.torch.heads_tails")

__all__ = [
    "shuffle",
    "cdist_multi",
    "fast_prod_positive",
    "fast_prod_positive_multi",
    "sum_multi",
    "prod_multi",
    "mul_along_dim",
    "index_add_into_zeros",
    "index_reduce_into_ones",
    "heads_tails",
]
