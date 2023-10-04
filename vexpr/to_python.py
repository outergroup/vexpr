import numbers

from vexpr.core import Vexpr, symbol_p, eval_impls


class Code:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def contains_vexpr(v):
    if isinstance(v, Vexpr):
        return True
    if isinstance(v, (list, tuple)):
        return any(contains_vexpr(child) for child in v)
    if isinstance(v, dict):
        return any(contains_vexpr(child) for child in v.values())
    return False


def move_user_objects_out(v):
    constants = {}
    def move_out_helper(v):
        if isinstance(v, Vexpr):
            return Vexpr(v.op,
                         [move_out_helper(child)
                          for child in v.args],
                         {k: move_out_helper(child)
                          for k, child in v.kwargs.items()})
        elif isinstance(v, (numbers.Number, str, type(slice), type(None),
                            type(Ellipsis))):
            return v
        elif isinstance(v, (list, tuple)):
            if contains_vexpr(v):
                return type(v)(move_out_helper(child) for child in v)
        elif isinstance(v, dict):
            if contains_vexpr(v):
                return {k: move_out_helper(child)
                        for k, child in v.items()}

        name = f"v{len(constants)}"
        constants[name] = v
        return Code(name)
    return move_out_helper(v), constants


def get_args_unique_ops(args):
    ops = set()

    for arg in args:
        if isinstance(arg, Vexpr):
            ops |= set([arg.op])
            descendent_ops = get_args_unique_ops(arg.args)
            ops |= descendent_ops
        elif isinstance(arg, (list, tuple)):
            descendent_ops = get_args_unique_ops(arg)
            ops |= descendent_ops

    return ops


def to_list_of_strings(args, name_for_op):
    results = []

    for arg in args:
        if isinstance(arg, Vexpr):
            expr = arg
            if expr.op == symbol_p:
                name = expr.args[0]
                results.append(f"symbols[{repr(name)}]")
            else:
                f_name = name_for_op[expr.op]
                f_args = to_list_of_strings(expr.args, name_for_op)
                for k, v in expr.kwargs.items():
                    f_args.append(f"{k}={repr(v)}")
                s_args = ", ".join(f_args)
                results.append(f"{f_name}({s_args})")
        elif isinstance(arg, (list, tuple)):
            s_contents = ", ".join(to_list_of_strings(arg, name_for_op))
            if isinstance(arg, list):
                results.append(f"[{s_contents}]")
            else:
                results.append(f"({s_contents})")
        else:
            results.append(repr(arg))

    return results


def to_python(expr):
    """
    Returns a Python function that implements expr, completely factoring
    Vexpr out of the logic. This is useful for tools like `torch.compile` which
    are intended to trace functions directly, not to trace Vexpr interpreters.
    """
    expr_without_data, constants = move_user_objects_out(expr)
    unique_ops = get_args_unique_ops([expr_without_data])
    name_for_op = {op: op.name.replace(".", "_") for op in unique_ops}

    # Ensure uniqueness of names. If there's a good reason for names to collide,
    # this function needs logic added to create unique names.
    assert len(set(name_for_op.values())) == len(name_for_op)

    provided_functions = {}
    for op, name in name_for_op.items():
        if op != symbol_p:
            provided_functions[name] = eval_impls[op]

    provided_locals = {**provided_functions, **constants}

    s_expression = to_list_of_strings([expr_without_data], name_for_op)[0]
    s_function = "def generated_func(symbols): return " + s_expression
    print(provided_locals)
    print(s_function)
    exec(s_function, provided_locals)

    return provided_locals["generated_func"]


__all__ = [
    "to_python",
]
