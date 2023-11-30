import os
import uuid
from pkg_resources import resource_string

import vexpr as vp
import vexpr.core as core


def header_content():
    vexpr_js = resource_string(
        'vexpr', os.path.join('package_data', 'vexpr.js')
    ).decode('utf-8')
    d3_js = resource_string(
        'vexpr', os.path.join('package_data', 'd3.min.js')
    ).decode('utf-8')

    return f"""
    <style>
    div.vexpr-output svg {{
      max-width: initial;
    }}
    </style>
    <script>
      const vexpr_undef_define = ("function"==typeof define);
      let vexpr_prev_define = undefined;
      if (vexpr_undef_define) {{
        vexpr_prev_define = define;
        define = undefined;
      }}
      {d3_js}
      {vexpr_js}
      if (vexpr_undef_define) {{
        define = vexpr_prev_define;
      }}
    </script>
    """



def time_control(class_name):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0],
              component = vexpr.timeControl()
                .onupdate(timestep => {{
                  d3.select(container).select(".timeState").node()._vexprState.selectTimestep(
                    timestep
                  );
                }});

        renderTimeFunctions.push(function(min, max, curr) {{
          d3.select(element).datum({{min: min, max: max, curr: curr}}).call(component);
        }});
      }})();
    """




def position_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0],
              component = vexpr.positionView()
              .scale(d3.scaleLinear().domain([-1.3, 0.3]).range([0, 400]));

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model["{key}"]).call(component);
        }});
      }})();
    """


def expression_view(class_name, keys, text):
    return f"""
      (function() {{
        const kernelExpr = `{text}`,
              element = container.querySelectorAll(".{class_name}")[0],
              kernelKeys = {repr(keys)},
              component = vexpr.expressionView(kernelExpr, kernelKeys);

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model).call(component);
        }});
      }})();
    """

def scalar_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0],
              component = vexpr.scalarView()
              .scale(d3.scaleLog().domain([1e-7, 5e0]).range([0, 400]))
              .eps(1e-7)
              .exponentFormat(true)
              .padRight(55)
              .height(12)
              .width(200)
              .fontSize(13)
              .tfrac(2.7/3);

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model["{key}"]).call(component);
        }});
      }})();
    """


def visualize_timeline_html(html_preamble, components, headers, encoded_data):
    element_id = str(uuid.uuid1())
    components_str = "\n".join(components)

    return f"""
    {html_preamble(element_id)}
    <script>
    (function() {{
      let renderTimestepFunctions = [],
          renderTimeFunctions = [],
          timeStateComponent = vexpr.hiddenTimeState()
          .renderTimestep(model => {{
            renderTimestepFunctions.forEach(render => render(model))
          }})
          .renderTime((min, max, curr) => {{
            renderTimeFunctions.forEach(render => render(min, max, curr))
          }});

      const container = document.getElementById("{element_id}");

      {components_str}

      const headers = {repr(headers)};
      const timesteps = `{encoded_data}`
       .replace(/\\n$/, '') // strip any newline at the end of the string
       .split('\\n').map(
           row => new Float32Array(Uint8Array.from(atob(row), c => c.charCodeAt(0)).buffer)
         ).map(
           row => Object.fromEntries(
             headers.map((header, i) => [header, row[i]])
         ));

      d3.select(container).datum(timesteps).call(timeStateComponent);
    }})();
    </script>
    """


def full_html(body):
    return f"""<!doctype html>
    <html>
    <head>
    {header_content()}
    </head>
    <body>
    {body}
    </body>
    </html>"""




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
             values.append(str(expr.args[0].item()))
             return vp.symbol(alias)
         else:
             return expr

    aliased_expr = vp.bottom_up_transform(alias_if_value, expr)

    return aliased_expr, aliases, values
