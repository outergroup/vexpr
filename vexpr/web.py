import vexpr as vp
import vexpr.core as core


def alias_values(expr):
    aliases = []
    values = []

    def alias_if_value(expr):
         if expr.op == vp.primitives.value_p:
             alias = None
             if isinstance(expr, core.VexprWithMetadata) \
                and "visual_type" in expr.metadata:

                 vtype = expr.metadata["visual_type"]
                 if vtype == "mixing_weight":
                     alias = f"$W{len(aliases)}"
                 elif vtype == "location":
                     alias = f"$L{len(aliases)}"
                 elif vtype == "scale":
                     alias = f"$S{len(aliases)}"

             if alias is None:
                 alias = f"$U{len(aliases)}"
             aliases.append(alias)
             values.append(expr.args[0].tolist())
             return vp.symbol(alias)
         else:
             return expr

    aliased_expr = vp.bottom_up_transform(alias_if_value, expr)

    return aliased_expr, aliases, values
