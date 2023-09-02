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

    def __call__(self, **kwargs):
        return call(self, kwargs)

    def __add__(self, other):
        return operator_add(self, other)

    def __mul__(self, other):
        return operator_mul(self, other)

    def __truediv__(self, other):
        return operator_truediv(self, other)

    def __pow__(self, other):
        return operator_pow(self, other)

    def __neg__(self):
        return operator_neg(self)

    def __getitem__(self, index):
        return operator_getitem(self, index)

    def __repr__(self):
        """
        (adapted from pytorch)
        """
        child_lines = []

        if len(self.args) == 1:
            child_lines.append(pformat(self.args[0]))
        else:
            for arg in self.args:
                operand_str = pformat(arg) + ","
                operand_str = _addindent(operand_str, 2)
                child_lines.append(operand_str)

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
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


def _addindent(s_, numSpaces):
    """
    (adapted from pytorch)
    """
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


################################################################################
# Special primitives. The Vexpr interpreter is aware of these.
################################################################################


symbol_p = Primitive("symbol")
symbol = lambda name: Vexpr(symbol_p, (name,), {})
let_p = Primitive("let")
let = lambda bindings, expr: Vexpr(let_p, (bindings, expr), {})


################################################################################
# The numpy Vexpr interpreter
################################################################################

eval_impls = {}
def call(expr, context, callback=None):
    """
    This function implements the Vexpr interpreter.
    """
    if not isinstance(expr, Vexpr):
        result = expr
    elif expr.op is symbol_p:
        result = context[expr.args[0]]
    elif expr.op is let_p:
        context2 = dict(context)
        for symbol, v in expr.args[0]:
            name = (symbol if isinstance(symbol, str) else symbol.args[0])
            context2[name] = call(v, context2, callback)
        result = call(expr.args[1], context2, callback)
    else:
        impl = eval_impls[expr.op]

        # Evaluate Vexprs in the arguments, down to one level deep. Thus we
        # allow lists of Vexprs as args. We could replace this with JAX's
        # tree_map if we want to support arbitrary pytrees, but for now we're
        # avoiding the JAX dependency. (Also that's awkward with Vexprs, which
        # are technically tuples, hence aren't pytree leafs by default.)
        f = partial(call, context=context, callback=callback)
        args = tuple((f(arg)
                     if isinstance(arg, Vexpr)
                     else (type(arg)((f(v)
                                     if isinstance(v, Vexpr)
                                     else v) for v in arg)
                         if isinstance(arg, (list, tuple))
                         else arg))
                     for arg in expr.args)

        # args = tree_map(f, expr.args,
        #                 is_leaf=lambda x: not isinstance(x, (list, tuple, dict)) or isinstance(x, Vexpr))
        # args = [call(arg, context, callback) for arg in expr.args]
        result = impl(*args, **expr.kwargs)

    if callback is not None:
        callback(expr, result)

    return result


shape_impls = {
    int: lambda _: (),
    float: lambda _: (),
}


def evaluate_shapes(expr, **example_inputs):
    shapes = {}
    def record_shape(sub_expr, result):
        shape_impl = shape_impls[type(result)]
        shapes[id(sub_expr)] = shape_impl(result)
    call(expr, example_inputs, record_shape)
    return shapes


################################################################################
# Built-in primitives
################################################################################

def _p_and_constructor(name):
    p = Primitive(name)
    def construct_vexpr(*args, **kwargs):
        return Vexpr(p, args, kwargs)
    return p, construct_vexpr

# Python language primitives
operator_add_p, operator_add = _p_and_constructor("operator.add")
operator_mul_p, operator_mul = _p_and_constructor("operator.mul")
operator_truediv_p, operator_truediv = _p_and_constructor("operator.truediv")
operator_pow_p, operator_pow = _p_and_constructor("operator.pow")
operator_neg_p, operator_neg = _p_and_constructor("operator.neg")
operator_getitem_p, operator_getitem = _p_and_constructor("operator.getitem")


################################################################################
# Default Impls
################################################################################

import operator

eval_impls.update({
    operator_add_p: operator.add,
    operator_mul_p: operator.mul,
    operator_truediv_p: operator.truediv,
    operator_pow_p: operator.pow,
    operator_neg_p: operator.neg,
    operator_getitem_p: operator.getitem,
})


################################################################################
# Vexpr's vectorizor
#
# To vectorize an expression, we start from the root node and for each arg we
# move downward through the tree, attempting to convert each level into an
# operation on a single array. The process typically starts by taking an
# expression and wrapping each of its aruments with a "stack" operation, then
# attempting a "pushthrough" of that stack operation through that argument's
# descendants. As the "stack" operation pushes through each level, its effect
# changes; for example, when you pushthrough a "stack" through an array of
# "sum"s, the logic should then pushthrough a "concatenate" through the sums'
# children. This pushthrough logic is specified by for op-child_op pairs via
# pushthrough_impls.
################################################################################


# {op: f(shapes, expr)}
vectorize_impls = {}

def _vectorize(shapes, expr):
    impl = vectorize_impls.get(expr.op, None)
    if impl is None:
        return expr
    else:
        return impl(shapes, expr)


# {(op, child_op): f(shapes, expr, allow_partial)}
pushthrough_impls = {}

def pushthrough(shapes, expr, child_op, allow_partial=True):
    impl = pushthrough_impls[(expr.op, child_op)]
    return impl(shapes, expr, allow_partial)


################################################################################
# Vectorize implementations for operators
################################################################################

def operator_vectorize(shapes, expr):
    args = tuple(_vectorize(shapes, arg) for arg in expr.args)
    return Vexpr(expr.op, args, {})

vectorize_impls.update({
    operator_add_p: operator_vectorize,
    operator_mul_p: operator_vectorize,
    operator_truediv_p: operator_vectorize,
    operator_pow_p: operator_vectorize,
    operator_neg_p: operator_vectorize,
    operator_getitem_p: operator_vectorize,
})

# TODO needed? rename to CannotPush?
class CannotVectorize(Exception):
    pass


################################################################################
# User interface
################################################################################

def make_vexpr(f):
    arg_names = inspect.getfullargspec(f).args  # TODO allow kwargs?
    symbols = {name: symbol(name) for name in arg_names}
    expression = f(*symbols.values())
    return expression

def vectorize(f):
    if not isinstance(f, Vexpr):
        f = make_vexpr(f)
    return LazyVectorize(f)

class LazyVectorize:
    def __init__(self, vexpr):
        self.vexpr = vexpr
        self.vectorized = None

    def __call__(self, **kwargs):
        if self.vectorized is None:
            shapes = {}
            def record_shape(sub_expr, result):
                shape_impl = shape_impls[type(result)]
                shapes[id(sub_expr)] = shape_impl(result)
            result = call(self.vexpr, kwargs, record_shape)
            self.vectorized = _vectorize(shapes, self.vexpr)
            return result
        else:
            return self.vectorized(**kwargs)
