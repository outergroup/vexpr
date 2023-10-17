from functools import partial

import vexpr as vp
from vexpr import Vexpr, core

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


class VexprWithMetadata(Vexpr):
    metadata: dict

    def __new__(cls, op, args, kwargs, metadata):
        self = super().__new__(cls, op, args, kwargs)
        self.metadata = metadata
        return self


def with_metadata(expr: Vexpr, metadata: dict):
    return VexprWithMetadata(*expr, metadata)


def with_return_shape(expr: Vexpr, return_shape: tuple):
    return with_metadata(expr, dict(return_shape=return_shape))


shape_impls = {
    int: lambda _: (),
    float: lambda _: (),
    VexprWithMetadata: lambda expr: expr.metadata["return_shape"],
}


def shape(expr):
    impl = shape_impls[type(expr)]
    return impl(expr)


def evaluate_args(args, context):
    args_with_metadata = []
    evaluated_args = []

    for arg in args:
        if isinstance(arg, Vexpr):
            arg, v = traced_call(arg, context)
            args_with_metadata.append(arg)
            evaluated_args.append(v)
        elif isinstance(arg, (list, tuple)):
            (sub_args_with_metadata,
             sub_evaluated_args) = evaluate_args(arg, context)
            args_with_metadata.append(sub_args_with_metadata)
            evaluated_args.append(sub_evaluated_args)
        else:
            evaluated_args.append(arg)
            args_with_metadata.append(arg)

    container = type(args)
    return container(args_with_metadata), container(evaluated_args)


def traced_call(expr, context):
    """
    This function implements the Vexpr interpreter.
    """
    if not isinstance(expr, Vexpr):
        raise ValueError(expr)

    if expr.op == core.symbol_p:
        result = context[expr.args[0]]
    elif expr.op == core.let_p:
        raise NotImplementedError()
    else:
        args_with_metadata, evaluated_args = evaluate_args(expr.args, context)
        expr = Vexpr(expr.op, args_with_metadata, expr.kwargs)
        impl = core.eval_impls[expr.op]
        result = impl(*evaluated_args, **expr.kwargs)

    expr = with_metadata(expr, dict(return_shape=shape(result)))

    return expr, result


# {op: f(shapes, expr)}
vectorize_impls = {}

def _vectorize(expr):
    # assert isinstance(expr, VexprWithMetadata)

    impl = vectorize_impls.get(expr.op, None)
    if impl is None:
        print("No vectorization support for", expr.op)
        return expr
    else:
        return impl(expr)


# {(op, child_op): f(expr, allow_partial)}
pushthrough_impls = {}

def pushthrough(expr, child_op, allow_partial=True):
    # assert isinstance(expr, VexprWithMetadata)

    impl = pushthrough_impls.get((expr.op, child_op), None)
    if impl is None:
        print("No pushthrough support for", expr.op, child_op)
        raise CannotVectorize()

    return impl(expr, allow_partial)

def single_pushthrough(expr, allow_partial=True):
    return pushthrough(expr, expr.args[0].op, allow_partial=allow_partial)


# {(op, child_op): f(expr)}
lift_impls = {}

def lift(expr, child_op):
    impl = lift_impls.get((expr.op, child_op), None)
    if impl is None:
        print("No lift support for", expr.op, child_op)
        raise CannotVectorize()

    return impl(expr)


unary_elementwise_registration_steps = []

def register_unary_elementwise_op(op):
    for register in unary_elementwise_registration_steps:
        register(op)


def _vectorize2(expr):
    """
    Function that orchestrates the vectorization process.
    """
    catch = [True, True, False]
    while True:
        iteration_prev_expr = expr
        group_i = 0
        while group_i < len(phase_ops):
            group_prev_expr = expr
            expr = recursive_pushthrough(phase_ops[group_i], expr, 
                                         always_catch=catch[group_i])
            if vp.comparable(expr) == vp.comparable(group_prev_expr):
                group_i += 1
        if vp.comparable(expr) == vp.comparable(iteration_prev_expr):
            break

    return expr


# Each phase is a list of (op, pushthrough) pairs
phase_ops = [
    # stack
    [],
    # concatenate, stack, index_select, shuffle
    [],
    # multiply
    [],
]

def recursive_pushthrough(ops, expr, always_catch=True):
    ops = dict(ops)

    def recursive_pushthrough_nocatch(expr):
        transform = (partial(recursive_pushthrough, ops, always_catch=True)
                     if always_catch
                     else recursive_pushthrough_nocatch)
        if expr.op in ops:
            pushthrough = ops[expr.op]
            return pushthrough(expr, transform)
        else:
            return with_return_shape(
                Vexpr(expr.op,
                      recursively_transform_args(expr.args, transform),
                      expr.kwargs),
                shape(expr))

    try:
        return recursive_pushthrough_nocatch(expr)
    except CannotVectorize:
        return expr


################################################################################
# Vectorize implementations for operators
################################################################################

def unary_elementwise_vectorize(expr):
    return Vexpr(expr.op, (_vectorize(expr.args[0]),), expr.kwargs)

def operator_vectorize(expr):
    args = tuple(_vectorize(arg) for arg in expr.args)
    return Vexpr(expr.op, args, {})

def identity(expr): return expr

vectorize_impls.update({
    core.symbol_p: identity,
    core.operator_add_p: operator_vectorize,
    core.operator_mul_p: operator_vectorize,
    core.operator_truediv_p: operator_vectorize,
    core.operator_pow_p: operator_vectorize,
    core.operator_matmul_p: operator_vectorize,
    core.operator_neg_p: unary_elementwise_vectorize,
    core.operator_getitem_p: operator_vectorize,
})


class CannotVectorize(Exception):
    pass



################################################################################
# User interface
################################################################################


def recursively_transform_args(args, transform):
    new_args = []
    for arg in args:
        if isinstance(arg, Vexpr):
            arg = transform(arg)
        elif isinstance(arg, (list, tuple)):
            arg = recursively_transform_args(arg, transform)
        new_args.append(arg)
    return type(args)(new_args)


def strip_metadata(expr):
    return Vexpr(expr.op, recursively_transform_args(expr.args,
                                                     strip_metadata),
                 expr.kwargs)


def call_and_vectorize(vexpr_caller, i_alternate, **kwargs):
    # Hack to detect whether to use old or new vectorize
    if sum(len(ops) for ops in phase_ops) > 0:
        _vec = _vectorize2
    else:
        _vec = _vectorize

    expr, result = traced_call(vexpr_caller.vexpr, kwargs)
    vexpr_caller.vexpr = strip_metadata(_vec(expr))
    del vexpr_caller.alternate_calls[i_alternate]
    return result


def vectorize(f_or_expr, example_inputs=None):
    if example_inputs is not None:
        # Hack to detect whether to use old or new vectorize
        if sum(len(ops) for ops in phase_ops) > 0:
            _vec = _vectorize2
        else:
            _vec = _vectorize
        if isinstance(f_or_expr, Vexpr):
            expr, result = traced_call(f_or_expr, example_inputs)
            return strip_metadata(_vec(expr))
        else:
            if isinstance(f_or_expr, core.VexprCaller):
                vexpr_caller = f_or_expr.clone()
            else:
                vexpr_caller = core.make_vexpr(f_or_expr)

            expr, result = traced_call(vexpr_caller.vexpr, example_inputs)
            vexpr_caller.vexpr = strip_metadata(_vec(expr))
            return vexpr_caller
    else:
        # Lazily vectorize when inputs are provided
        if isinstance(f_or_expr, core.VexprCaller):
            vexpr_caller = f_or_expr.clone()
        elif isinstance(f_or_expr, Vexpr):
            vexpr_caller = core.VexprCaller(f_or_expr, [])
        else:
            vexpr_caller = core.make_vexpr(f_or_expr)

        vexpr_caller.alternate_calls.append(call_and_vectorize)
        return vexpr_caller
