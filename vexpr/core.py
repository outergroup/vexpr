"""
Move to README or blog post:
JAX uses a tracer-based approach for defining Jaxprs. Vexprs, on the other hand,
are built up directly by the user.
"""

import inspect
from functools import partial
from pprint import pformat
from typing import NamedTuple


class Primitive(NamedTuple):
    name: str


class Vexpr(NamedTuple):
    op: Primitive
    args: tuple
    kwargs: dict

    def __call__(self, *args, **kwargs):
        return operator_call(self, *args, **kwargs)

    def __add__(self, other):
        return operator_add(self, other)

    def __mul__(self, other):
        return operator_mul(self, other)

    def __rmul__(self, other):
        return operator_mul(other, self)

    def __truediv__(self, other):
        return operator_truediv(self, other)

    def __pow__(self, other):
        return operator_pow(self, other)

    def __matmul__(self, other):
        return operator_matmul(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __getitem__(self, index):
        return operator_getitem(self, index)

    def __repr__(self):
        child_lines = []

        if len(self.args) == 1:
            child_lines += pformat(self.args[0]).split('\n')
        else:
            for arg in self.args:
                operand_str = pformat(arg) + ","
                child_lines += operand_str.split('\n')

        extra_lines = ([", ".join([f"{k}={repr(v)}"
                                   for k, v in self.kwargs.items()])]
                       if len(self.kwargs) > 0
                       else [])

        lines = child_lines + extra_lines

        main_str = self.op.name + '('
        if lines:
            if len(lines) == 1:
                main_str += lines[0]
            else:
                newline_space = "\n  "
                main_str += newline_space + newline_space.join(lines) + '\n'

        main_str += ')'
        return main_str


################################################################################
# Special primitives. The Vexpr interpreter is aware of these.
################################################################################


symbol_p = Primitive("symbol")
symbol = lambda name: Vexpr(symbol_p, (name,), {})
let_p = Primitive("let")
let = lambda bindings, expr: Vexpr(let_p, (bindings, expr), {})


################################################################################
# The Vexpr interpreter
################################################################################

eval_impls = {}
def call(expr, context):
    """
    This function implements the Vexpr interpreter.
    """
    if not isinstance(expr, Vexpr):
        raise ValueError(expr)

    if expr.op == symbol_p:
        result = context[expr.args[0]]
    elif expr.op == let_p:
        context2 = dict(context)
        for symbol, v in expr.args[0]:
            name = (symbol if isinstance(symbol, str) else symbol.args[0])
            context2[name] = call(v, context2)
        result = call(expr.args[1], context2)
    else:
        impl = eval_impls[expr.op]
        args = evaluate_args(expr.args, context)
        result = impl(*args, **expr.kwargs)

    return result


def evaluate_args(args, context, call_fn=call):
    container = type(args)
    return container((call_fn(arg, context)
                      if isinstance(arg, Vexpr)
                      else (evaluate_args(arg, context, call_fn)
                            if isinstance(arg, (list, tuple))
                            else arg))
                     for arg in args)


################################################################################
# Built-in primitives
################################################################################

def _p_and_constructor(name):
    p = Primitive(name)
    def construct_vexpr(*args, **kwargs):
        return Vexpr(p, args, kwargs)
    return p, construct_vexpr

# Python language primitives
operator_call_p, operator_call = _p_and_constructor("operator.call")
operator_add_p, operator_add = _p_and_constructor("operator.add")
operator_mul_p, operator_mul = _p_and_constructor("operator.mul")
operator_truediv_p, operator_truediv = _p_and_constructor("operator.truediv")
operator_pow_p, operator_pow = _p_and_constructor("operator.pow")
operator_matmul_p, operator_matmul = _p_and_constructor("operator.matmul")
operator_neg_p, operator_neg = _p_and_constructor("operator.neg")
operator_getitem_p, operator_getitem = _p_and_constructor("operator.getitem")

constant_p = Primitive("constant")
constant = lambda value: Vexpr(constant_p, (value,), {})

################################################################################
# Default Impls
################################################################################

import operator


eval_impls.update({
    operator_add_p: operator.add,
    operator_mul_p: operator.mul,
    operator_truediv_p: operator.truediv,
    operator_pow_p: operator.pow,
    operator_matmul_p: operator.matmul,
    operator_neg_p: operator.neg,
    operator_getitem_p: operator.getitem,
    operator_call_p: (operator.call  # This exists in Python 3.11+
                      if hasattr(operator, "call")
                      else lambda f, *args, **kwargs: f(*args, **kwargs)),
})


################################################################################
# Vexprs partial evaluation
################################################################################


# this is essentially an alternate interpreter which only evaluates an operator
# if its children can be evaluated, given the provided inputs.
def partial_eval_(expr, context):
    if not isinstance(expr, Vexpr):
        raise ValueError(expr)

    if expr.op == symbol_p:
        name = expr.args[0]
        if name in context:
            return context[expr.args[0]]
        else:
            return expr
    elif expr.op == let_p:
        context2 = dict(context)
        for symbol, v in expr.args[0]:
            name = (symbol if isinstance(symbol, str) else symbol.args[0])
            v = partial_eval_(v, context2)
            if not isinstance(v, Vexpr):
                context2[name] = v
        return partial_eval_(expr.args[1], context2)
    else:
        impl = eval_impls[expr.op]
        args = evaluate_args(expr.args, context, partial_eval_)

        ready = True
        for arg in args:
            if isinstance(arg, Vexpr):
                ready = False
            elif isinstance(arg, (list, tuple)):
                for subarg in arg:
                    if isinstance(subarg, Vexpr):
                        ready = False

        if ready:
            return impl(*args, **expr.kwargs)
        else:
            return Vexpr(expr.op, args, expr.kwargs)



################################################################################
# User interface
################################################################################

class VexprCaller:
    """
    This class enables calling a Vexpr with ordered args. It also provides a way
    of queueing alternate __call__ behavior.
    """
    def __init__(self, vexpr, arg_names):
        self.vexpr = vexpr
        self.arg_names = arg_names
        self.alternate_calls = []

    def clone(self):
        ret = VexprCaller(self.vexpr, self.arg_names)
        ret.alternate_calls = self.alternate_calls
        return ret

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            kwargs.update(dict(zip(self.arg_names, args)))

        if len(self.alternate_calls) > 0:
            call_fn = self.alternate_calls[0]
            return call_fn(self, 0, **kwargs)

        return eval(self.vexpr, kwargs)


def make_vexpr(f):
    arg_names = inspect.getfullargspec(f).args  # TODO allow kwargs?
    symbols = {name: symbol(name) for name in arg_names}
    # trace the function by calling it with symbols
    expression = f(*symbols.values())
    return VexprCaller(expression, arg_names)


def eval(expr, inputs, allow_partial=False):
    if allow_partial:
        return partial_eval_(expr, inputs)
    else:
        return call(expr, inputs)


def partial_eval(f, inputs):
    """
    If all symbols are specified, the Vexpr is fully evaluated.
    """
    if isinstance(f, VexprCaller):
        f = f.clone()
        f.vexpr = partial_eval_(f.vexpr, inputs)
        for name in inputs.keys():
            # Mimic behavior of functools.partial. Any arg after the first kwarg
            # can now only be specified as a kwarg.
            try:
                i = f.arg_names.index(name)
                del f.arg_names[i:]
            except ValueError:
                pass
        return f
    elif isinstance(f, Vexpr):
        return partial_eval_(f, inputs)
    else:
        raise ValueError(f)
