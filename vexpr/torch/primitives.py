from vexpr.core import _p_and_constructor

# Array construction
zeros_p, zeros = _p_and_constructor("torch.zeros")
ones_p, ones = _p_and_constructor("torch.ones")
# Array manipulation
stack_p, stack = _p_and_constructor("torch.stack")
cat_p, cat = _p_and_constructor("torch.cat")
scatter_p, scatter = _p_and_constructor("torch.scatter")
moveaxis_p, moveaxis = _p_and_constructor("torch.moveaxis")
reshape_p, reshape = _p_and_constructor("torch.reshape")
# Unary operators
exp_p, exp = _p_and_constructor("torch.exp")
log_p, log = _p_and_constructor("torch.log")
# Reduction operators
sum_p, sum = _p_and_constructor("torch.sum")
prod_p, prod = _p_and_constructor("torch.prod")
# Indexed reductions
index_add_p, index_add = _p_and_constructor("torch.index_add")
index_reduce_p, index_reduce = _p_and_constructor("torch.index_reduce")

cdist_p, cdist = _p_and_constructor("torch.cdist")

__all__ = [
    "zeros",
    "ones",
    "stack",
    "cat",
    "scatter",
    "moveaxis",
    "reshape",
    "exp",
    "log",
    "sum",
    "prod",
    "cdist",
    "index_add",
    "index_reduce",
]
