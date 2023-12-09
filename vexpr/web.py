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
</script>"""



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


def position_distribution_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0],
              component = vexpr.scalarDistributionView()
              .scale(d3.scaleLinear().domain([-1.5, 1.5]).range([0, 400]))
              .fixedMin(-1.5)
              .fixedMax(1.5)
              .padRight(8)
              .height(12)
              .fontSize(13)
              .tfrac(2.7/3);

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model["{key}"]).call(component);
        }});
      }})();
    """


def position_distribution_list_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0];
        renderTimestepFunctions.push(function (model) {{
            const pointsLists = model["{key}"],
                  min = d3.min(pointsLists, points => d3.min(points)),
                  max = d3.max(pointsLists, points => d3.max(points)),
              component = vexpr.scalarDistributionListView()
              .scale(d3.scaleLinear().domain([min, max]).range([0, 232]))
              .useDataMin(true)
              .useDataMax(true)
              .height(12)
              .fontSize(13);
          d3.select(element).datum({{pointsLists, min, max}}).call(component);
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


def expression_distribution_view(class_name, keys, text):
    return f"""
      (function() {{
        const kernelExpr = `{text}`,
              element = container.querySelectorAll(".{class_name}")[0],
              kernelKeys = {repr(keys)},
              component = vexpr.expressionView(kernelExpr, kernelKeys).valueType("scalarDistribution");

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model).call(component);
        }});
      }})();
    """


def expression_distribution_list_view(class_name, keys, text):
    return f"""
      (function() {{
        const kernelExpr = `{text}`,
              element = container.querySelectorAll(".{class_name}")[0],
              kernelKeys = {repr(keys)},
              component = vexpr.expressionView(kernelExpr, kernelKeys).valueType("scalarDistributionList");

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
              .fontSize(13)
              .tfrac(2.7/3);

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model["{key}"]).call(component);
        }});
      }})();
    """


def scalar_distribution_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0],
              component = vexpr.scalarDistributionView()
              .scale(d3.scaleLog().domain([1e-7, 5e0]).range([0, 400]))
              .fixedMin(1e-7)
              .exponentFormat(true)
              .padRight(8)
              .height(12)
              .fontSize(13)
              .tfrac(2.7/3);

        renderTimestepFunctions.push(function (model) {{
          d3.select(element).datum(model["{key}"]).call(component);
        }});
      }})();
    """


def scalar_distribution_list_view(class_name, key):
    return f"""
      (function() {{
        const element = container.querySelectorAll(".{class_name}")[0];
        renderTimestepFunctions.push(function(model) {{
          const pointsLists = model["{key}"];

          let min = d3.min(pointsLists, points => d3.min(points)),
              max = d3.max(pointsLists, points => d3.max(points));

         if (min == max) {{
           // Need a valid scale, so these need to be different.
           // Visualize each point as a low value.
           max *= 10;
         }}

         const component = vexpr.scalarDistributionListView()
              .scale(d3.scaleLog().domain([min, max]).range([0, 215]))
              .useDataMin(true)
              .useDataMax(true)
              .exponentFormat(true)
              .height(12)
              .fontSize(13);

          d3.select(element).datum({{pointsLists, min, max}}).call(component);
        }});
      }})();
    """


def scalar_timesteps_js(headers):
    return f"""
      const headers = {repr(headers)};
      const timesteps = encodedData
       .replace(/\\n$/, '') // strip any newline at the end of the string
       .split('\\n')
       .map(
          row => new Float32Array(Uint8Array.from(atob(row), c => c.charCodeAt(0)).buffer)
        ).map(
          row => Object.fromEntries(
            headers.map((header, i) => [header, row[i]])
        ));
    """


def scalar_snapshot_js(headers):
    return f"""
      {scalar_timesteps_js(headers)}
      if (timesteps.length != 1) {{
        throw "Expected single line";
      }}
      const snapshot = timesteps[0];
    """


def scalar_distribution_timesteps_js(headers, num_values_per_param):
    return f"""
      const numValuesPerParam = {num_values_per_param};
      const headers = {repr(headers)};
      const timesteps = encodedData
       .replace(/\\n$/, '') // strip any newline at the end of the string
       .split('\\n')
       .map(
         row => new Float32Array(Uint8Array.from(atob(row), c => c.charCodeAt(0)).buffer))
       .map(
           row => {{
             // Split row into numValuesPerParam chunks
              const chunks = [];
              for (let i = 0; i < row.length; i += numValuesPerParam) {{
                chunks.push(row.slice(i, i + numValuesPerParam));
              }}

             return Object.fromEntries(
               headers.map((header, i) => [header, chunks[i]]));
          }});
    """


def scalar_distribution_list_snapshot_js(headers, num_values_per_param):
    return f"""
      const numValuesPerParam = {repr(num_values_per_param)};
      const headers = {repr(headers)};
      const snapshot = Object.fromEntries(
        headers.map((header, i) => [header, []]));
      const models = encodedData
       .replace(/\\n$/, '') // strip any newline at the end of the string
       .split('\\n')
       .map(
         row => new Float32Array(Uint8Array.from(atob(row), c => c.charCodeAt(0)).buffer))
       .forEach(
           (row, iRow) => {{
             // Split row into numValuesPerParam chunks
              let iCol = 0;
              for (let i = 0; i < row.length; i += numValuesPerParam[iRow]) {{
                snapshot[headers[iCol]].push(row.slice(i, i + numValuesPerParam[iRow]));
                iCol++;
              }}
          }});
    """


def scalar_distribution_snapshot_js(headers, num_values_per_param):
    return f"""
      {scalar_distribution_timesteps_js(headers, num_values_per_param)}
      if (timesteps.length != 1) {{
        throw "Expected single line";
      }}
      const snapshot = timesteps[0];
    """


def visualize_timeline_html(html_preamble, encoded_to_timesteps, components, encoded_data):
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

  const encodedData = `{encoded_data}`;
  {encoded_to_timesteps}

  d3.select(container).datum(timesteps).call(timeStateComponent);
}})();
</script>"""


def visualize_snapshot_html(html_preamble, encoded_to_snapshot, components, encoded_data):
    element_id = str(uuid.uuid1())
    components_str = "\n".join(components)

    return f"""
{html_preamble(element_id)}
<script>
(function() {{
  let renderTimestepFunctions = [];

  const container = document.getElementById("{element_id}");

  {components_str}

  const encodedData = `{encoded_data}`;
  {encoded_to_snapshot}
  renderTimestepFunctions.forEach(render => render(snapshot));
}})();
</script>"""


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
             values.append(expr.args[0].tolist())
             return vp.symbol(alias)
         else:
             return expr

    aliased_expr = vp.bottom_up_transform(alias_if_value, expr)

    return aliased_expr, aliases, values
