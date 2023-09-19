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


def evaluate_args(args, context, container=tuple):
    args_with_metadata = []
    evaluated_args = []

    for arg in args:
        if isinstance(arg, Vexpr):
            arg, v = traced_call(arg, context)
            args_with_metadata.append(arg)
            evaluated_args.append(v)
        elif isinstance(arg, (list, tuple)):
            (sub_args_with_metadata,
             sub_evaluated_args) = evaluate_args(arg, context,
                                                 container=type(arg))
            args_with_metadata.append(sub_args_with_metadata)
            evaluated_args.append(sub_evaluated_args)
        else:
            evaluated_args.append(arg)
            args_with_metadata.append(arg)

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
        print("No vectorization support for", expr.op, child_op)
        raise CannotVectorize()

    return impl(expr, allow_partial)


################################################################################
# Vectorize implementations for operators
################################################################################

def operator_vectorize(expr):
    args = tuple(_vectorize(arg) for arg in expr.args)
    return Vexpr(expr.op, args, {})

vectorize_impls.update({
    core.operator_add_p: operator_vectorize,
    core.operator_mul_p: operator_vectorize,
    core.operator_truediv_p: operator_vectorize,
    core.operator_pow_p: operator_vectorize,
    core.operator_matmul_p: operator_vectorize,
    core.operator_neg_p: operator_vectorize,
    core.operator_getitem_p: operator_vectorize,
})

# TODO needed? rename to CannotPush?
class CannotVectorize(Exception):
    pass



################################################################################
# User interface
################################################################################


def args_with_stripped_metadata(args):
    new_args = []
    for arg in args:
        if isinstance(arg, Vexpr):
            arg = strip_metadata(arg)
        elif isinstance(arg, (list, tuple)):
            arg = args_with_stripped_metadata(arg)
        new_args.append(arg)
    return type(args)(new_args)


def strip_metadata(expr):
    return Vexpr(expr.op, args_with_stripped_metadata(expr.args), expr.kwargs)


def call_and_vectorize(vexpr_caller, i_alternate, **kwargs):
    expr, result = traced_call(vexpr_caller.vexpr, kwargs)
    vexpr_caller.vexpr = strip_metadata(_vectorize(expr))
    del vexpr_caller.alternate_calls[i_alternate]
    return result


def vectorize(f):
    if isinstance(f, core.VexprCaller):
        f = f.clone()
    elif isinstance(f, Vexpr):
        f = core.VexprCaller(f, [])
    else:
        f = core.make_vexpr(f)

    f.alternate_calls.append(call_and_vectorize)
    return f
