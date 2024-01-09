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

    def update_args(self, args):
        return Vexpr(self.op, args, self.kwargs)

    def new(self, op, args, kwargs):
        return Vexpr(op, args, kwargs)

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
        try:
            impl = repr_impls[self.op]
        except KeyError:
            impl = repr_impls["default"]
        return impl(self)


class VexprWithMetadata(Vexpr):
    metadata: dict

    def __new__(cls, op, args, kwargs, metadata):
        self = super().__new__(cls, op, args, kwargs)
        self.metadata = metadata
        return self

    def new(self, op, args, kwargs):
        return VexprWithMetadata(op, args, kwargs, self.metadata)

    def update_args(self, args):
        return VexprWithMetadata(self.op, args, self.kwargs, self.metadata)


def with_metadata(expr: Vexpr, metadata: dict):
    if isinstance(expr, VexprWithMetadata):
        metadata = {**expr.metadata, **metadata}
    return VexprWithMetadata(expr.op, expr.args, expr.kwargs, metadata)


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


def comparable(v):
    for t, convert in to_comparable_conversions.items():
        if isinstance(v, t):
            return convert(v)
    return v

def comparable_hashable(v):
    for t, convert in to_hashable_conversions.items():
        if isinstance(v, t):
            return convert(v)
    for t, convert in to_comparable_conversions.items():
        if isinstance(v, t):
            return convert(v)
    return v

# modified by vexpr.numpy, vexpr.torch, etc
to_comparable_conversions = {
    Primitive: lambda v: v,
    tuple: lambda v: tuple(comparable(v) for v in v),
    list: lambda v: list(comparable(v) for v in v),
}

to_hashable_conversions = {
    dict: lambda d: tuple(sorted((k, comparable_hashable(x))
                                 for k, x in d.items())),
    tuple: lambda v: tuple(comparable_hashable(v) for v in v),
    list: lambda v: tuple(comparable_hashable(v) for v in v),
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
operator_call_p, operator_call = _p_and_constructor("operator.call")
operator_add_p, operator_add = _p_and_constructor("operator.add")
operator_mul_p, operator_mul = _p_and_constructor("operator.mul")
operator_truediv_p, operator_truediv = _p_and_constructor("operator.truediv")
operator_pow_p, operator_pow = _p_and_constructor("operator.pow")
operator_matmul_p, operator_matmul = _p_and_constructor("operator.matmul")
operator_neg_p, operator_neg = _p_and_constructor("operator.neg")
operator_getitem_p, operator_getitem = _p_and_constructor("operator.getitem")

value_p = Primitive("value")
value = lambda v: Vexpr(value_p, (v,), {})


################################################################################
# Default repr Impls
################################################################################

repr_impls = {}

unquoted_string_p = Primitive("unquoted_string")
unquoted_string = lambda s: Vexpr(unquoted_string_p, (s,), {})

def comment_repr(s):
    """
    Users may reassign this function
    """
    return f"# {s}"


def print_comments(repr_fn):
    def repr_fn_with_comments(expr):
        repr_txt = repr_fn(expr)
        if isinstance(expr, VexprWithMetadata) \
           and "comment" in expr.metadata:
            repr_txt = f"\n {comment_repr(expr.metadata['comment'])}\n{repr_txt}"
        return repr_txt
    return repr_fn_with_comments


@print_comments
def default_vexpr_repr(expr):
    if expr.op in repr_impls:
        return repr_impls

    child_lines = []

    use_initial_newline = True

    for i, arg in enumerate(expr.args):
        operand_str = pformat(arg)
        operand_lines = operand_str.split('\n')

        if len(operand_lines) > 1:
            if operand_lines[0].startswith('(') \
               or operand_lines[0].startswith('[') \
               or operand_lines[0].startswith('{'):
                use_initial_newline = False
                operand_lines = ([operand_lines[0][:1]]
                                 + [" " + operand_lines[0][1:]]
                                 + operand_lines[1:])

        if i < len(expr.args) - 1:
            operand_lines[-1] += ","
        child_lines += operand_lines

    extra_lines = ([", ".join([f"{k}={repr(v)}"
                               for k, v in expr.kwargs.items()])]
                   if len(expr.kwargs) > 0
                   else [])

    if len(extra_lines) == 0:
        lines = child_lines
    else:
        if len(child_lines) == 1:
            lines = [child_lines[0] + ","] + extra_lines
        elif len(child_lines) > 1:
            if "=" in child_lines[-1]:
                # if the final child_line is just kwargs from a deeper vexpr,
                # don't create a newline.
                lines = (child_lines[:-1]
                         + [child_lines[-1] + ", " + extra_lines[0]]
                         + extra_lines[1:])
            else:
                lines = (child_lines[:-1]
                         + [child_lines[-1] + ", "]
                         + extra_lines)
        else:
            lines = extra_lines

    main_str = expr.op.name + '('
    if len(lines) > 1 and use_initial_newline:
        main_str += "\n  "
    main_str += "\n  ".join(lines)

    main_str += ')'
    return main_str


def infix_repr(separator):
    @print_comments
    def infix_repr_impl(expr):
        left = pformat(expr.args[0])
        right = pformat(expr.args[1])

        left_lines = left.split("\n")
        if len(left_lines) == 1:
            return f"{left} {separator} {right}"
        else:
            return f"{left}\n{separator} {right}"
    return infix_repr_impl


repr_impls.update({
    "default": default_vexpr_repr,
    symbol_p: lambda expr: expr.args[0],
    unquoted_string_p: lambda expr: expr.args[0],
    operator_add_p: infix_repr("+"),
    operator_mul_p: infix_repr("*"),
    operator_truediv_p: infix_repr("/"),
    operator_pow_p: infix_repr("**"),
    operator_matmul_p: infix_repr("@"),
    operator_neg_p: print_comments(lambda expr: f"-{pformat(expr.args[0])}"),
    operator_getitem_p: print_comments(
        lambda expr: f"{pformat(expr.args[0])}[{pformat(expr.args[1])}]"
    ),
})


################################################################################
# Default eval Impls
################################################################################

import operator


eval_impls.update({
    value_p: lambda v: v,
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
def partial_eval_(expr, context, after_eval=None, to_eval_ready=None):
    if not isinstance(expr, Vexpr):
        raise ValueError(expr)

    recurse = partial(partial_eval_, after_eval=after_eval,
                      to_eval_ready=to_eval_ready)

    if expr.op == symbol_p:
        name = expr.args[0]
        if name in context:
            ret = context[expr.args[0]]
        else:
            ret = expr
    elif expr.op == let_p:
        context2 = dict(context)
        for symbol, v in expr.args[0]:
            name = (symbol if isinstance(symbol, str) else symbol.args[0])
            v = recurse(v, context2)
            if not isinstance(v, Vexpr):
                context2[name] = v
        ret = recurse(expr.args[1], context2)
    else :
        args = evaluate_args(expr.args, context, recurse)
        e_args = to_eval_ready(args) if to_eval_ready is not None else args

        ready = True
        for arg in e_args:
            if isinstance(arg, Vexpr):
                ready = False
            elif isinstance(arg, (list, tuple)):
                for subarg in arg:
                    if isinstance(subarg, Vexpr):
                        ready = False

        if ready and expr.op in eval_impls:
            impl = eval_impls[expr.op]
            ret = impl(*e_args, **expr.kwargs)
        else:
            ret = expr.new(expr.op, args, expr.kwargs)

    if after_eval is not None and not isinstance(ret, Vexpr):
        ret = after_eval(expr, ret)

    return ret



# Wrap each value in a "value" Vexpr so that its metadata is preserved.
partial_eval_with_metadata = partial(
    partial_eval_,
    after_eval=lambda expr, ret: expr.new(value_p, (ret,), {}),
    to_eval_ready=partial(evaluate_args, context={}, call_fn=partial_eval_)
)


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
        f.vexpr = partial_eval_with_metadata(f.vexpr, inputs)
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
        return partial_eval_with_metadata(f, inputs)
    else:
        raise ValueError(f)
