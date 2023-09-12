from vexpr.core import _p_and_constructor

# Array construction
zeros_p, zeros = _p_and_constructor("numpy.zeros")
ones_p, ones = _p_and_constructor("numpy.ones")
# Array manipulation
stack_p, stack = _p_and_constructor("numpy.stack")
concatenate_p, concatenate = _p_and_constructor("numpy.concatenate")
moveaxis_p, moveaxis = _p_and_constructor("numpy.moveaxis")
reshape_p, reshape = _p_and_constructor("numpy.reshape")
# Unary operators
exp_p, exp = _p_and_constructor("numpy.exp")
log_p, log = _p_and_constructor("numpy.log")
# Reduction operators
sum_p, sum = _p_and_constructor("numpy.sum")
prod_p, prod = _p_and_constructor("numpy.prod")
# Indexed reductions
add_at_p, add_at = _p_and_constructor("numpy.add.at")
multiply_at_p, multiply_at = _p_and_constructor("numpy.multiply.at")


__all__ = [
    "zeros",
    "ones",
    "stack",
    "concatenate",
    "moveaxis",
    "reshape",
    "exp",
    "log",
    "sum",
    "prod",
    "add_at", # TODO somehow change to "add.at"
    "multiply_at", # TODO somehow change to "multiply.at"
]
