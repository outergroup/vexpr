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
# The numpy Vexpr interpreter
################################################################################

eval_impls = {}
def call(expr, context, callback=None):
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
})


################################################################################
# Vexprs partial evaluation
################################################################################

# this is essentially an alternate interpreter which only evaluates an operator
# if its children can be evaluated, given the provided inputs.
def partial_evaluate_(expr, context):
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
            v = partial_evaluate_(v, context2)
            if not isinstance(v, Vexpr):
                context2[name] = v
        return partial_evaluate_(expr.args[1], context2)
    else:
        impl = eval_impls[expr.op]

        # Evaluate Vexprs in the arguments, down to one level deep. Thus we
        # allow lists of Vexprs as args. We could replace this with JAX's
        # tree_map if we want to support arbitrary pytrees, but for now we're
        # avoiding the JAX dependency. (Also that's awkward with Vexprs, which
        # are technically tuples, hence aren't pytree leafs by default.)
        f = partial(partial_evaluate_, context=context)
        args = tuple((f(arg)
                     if isinstance(arg, Vexpr)
                     else (type(arg)((f(v)
                                      if isinstance(v, Vexpr)
                                      else v) for v in arg)
                           if isinstance(arg, (list, tuple))
                           else arg))
                     for arg in expr.args)

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
        print("No vectorization support for", expr.op)
        return expr
    else:
        return impl(shapes, expr)


# {(op, child_op): f(shapes, expr, allow_partial)}
pushthrough_impls = {}

def pushthrough(shapes, expr, child_op, allow_partial=True):
    impl = pushthrough_impls.get((expr.op, child_op), None)
    if impl is None:
        print("No vectorization support for", expr.op, child_op)
        raise CannotVectorize()

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
    operator_matmul_p: operator_vectorize,
    operator_neg_p: operator_vectorize,
    operator_getitem_p: operator_vectorize,
})

# TODO needed? rename to CannotPush?
class CannotVectorize(Exception):
    pass


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

        return self.vexpr(**kwargs)


def call_and_vectorize(vexpr_caller, i_alternate, **kwargs):
    shapes = {}
    def record_shape(sub_expr, result):
        shape_impl = shape_impls[type(result)]
        shapes[id(sub_expr)] = shape_impl(result)
    result = call(vexpr_caller.vexpr, kwargs, record_shape)
    vexpr_caller.vexpr = _vectorize(shapes, vexpr_caller.vexpr)
    del vexpr_caller.alternate_calls[i_alternate]
    return result


def make_vexpr(f):
    arg_names = inspect.getfullargspec(f).args  # TODO allow kwargs?
    symbols = {name: symbol(name) for name in arg_names}
    # trace the function by calling it with symbols
    expression = f(*symbols.values())
    return VexprCaller(expression, arg_names)


def vectorize(f):
    if isinstance(f, VexprCaller):
        f = f.clone()
    elif isinstance(f, Vexpr):
        f = VexprCaller(f, [])
    else:
        f = make_vexpr(f)

    f.alternate_calls.append(call_and_vectorize)
    return f


def partial_evaluate(f, inputs):
    """
    If all symbols are specified, the Vexpr is fully evaluated.
    """
    if isinstance(f, VexprCaller):
        f = f.clone()
        f.vexpr = partial_evaluate_(f.vexpr, inputs)
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
        return partial_evaluate_(f, inputs)
    else:
        raise ValueError(f)
