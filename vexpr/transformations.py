import vexpr.primitives as p
from vexpr.core import Vexpr


def bottom_up_transform_args(transform, args):
    new_args = []
    for arg in args:
        if isinstance(arg, Vexpr):
            arg = transform(
                arg.update_args(bottom_up_transform_args(transform, arg.args))
            )
        elif isinstance(arg, (list, tuple)):
            arg = bottom_up_transform_args(transform, arg)
        new_args.append(arg)
    return type(args)(new_args)


def bottom_up_transform(transform, expr):
    return bottom_up_transform_args(transform, [expr])[0]


def bottom_up_transform_leafs_impl(transform, args):
    if isinstance(args, dict):
        return type(args)(
            (k, bottom_up_transform_leafs_impl(transform, v))
            for k, v in args.items()
        )
    elif isinstance(args, (list, tuple)):
        new_args = []
        for arg in args:
            if isinstance(arg, Vexpr):
                if arg.op == p.unquoted_string_p:
                    pass
                else:
                    arg = arg.new(arg.op,
                                  bottom_up_transform_leafs_impl(transform, arg.args),
                                  bottom_up_transform_leafs_impl(transform, arg.kwargs))
            elif isinstance(arg, (list, tuple, dict)):
                arg = bottom_up_transform_leafs_impl(transform, arg)
            else:
                arg = transform(arg)
            new_args.append(arg)
        return type(args)(new_args)
    else:
        return transform(args)


def transform_leafs(transform, expr):
    """
    Like a tree_map, but called only on values in Vexpr args and kwargs.
    """
    return bottom_up_transform_leafs_impl(transform, [expr])[0]
