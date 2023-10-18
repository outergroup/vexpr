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
