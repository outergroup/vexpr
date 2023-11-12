import os
import uuid

from pkg_resources import resource_string

from IPython.display import HTML, display, update_display



def init_notebook_mode():
    vexpr_js = resource_string(
        'vexpr', os.path.join('package_data', 'vexpr.js')
    ).decode('utf-8')
    d3_js = resource_string(
        'vexpr', os.path.join('package_data', 'd3.min.js')
    ).decode('utf-8')

    display(HTML(f"""
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

      if (window.vexprQueue) {{
        window.vexprQueue.forEach(([k, f]) => f());
        window.vexprQueue = null;
      }}
    </script>
    """))


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


def js_refresh(element_id, csv_txt):
    return f"""
    (function() {{
      function update() {{
        document.getElementById("{element_id}")._vexprState.refresh(`{csv_txt}`);
      }}

      if (window.vexpr) {{
        update();
      }} else {{
          if (!window.vexprQueue) {{
            window.vexprQueue = [];
          }}

          // Remove stale refreshes. (Avoid queueing a huge unnecessary work task)
          const k = "{element_id} update";
          window.vexprQueue = window.vexprQueue.filter(([k2, f]) => k2 != k);
          window.vexprQueue.push([k, update]);
      }}
    }})();
    </script>
    """


def visualize_timeline(html_preamble, components, csv_txt):
    element_id = str(uuid.uuid1())
    components_str = "\n".join(components)

    s = f"""
    {html_preamble(element_id)}
    <script>
    function initialize() {{
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
      container._vexprState = {{
        refresh: function(csv) {{
          const timesteps = d3.csvParse(csv, d => {{
            const row = {{}};
            for (const [key, value] of Object.entries(d)) {{
              row[key] = parseFloat(value);
            }}
            return row;
          }});

          d3.select(container).datum(timesteps).call(timeStateComponent);
        }}
      }};

    {components_str}
    }}

    if (window.vexpr) {{
      initialize();
    }} else {{
        if (!window.vexprQueue) {{
          window.vexprQueue = [];
        }}

        window.vexprQueue.push(["", initialize]);
    }}
    </script>
    """

    display(HTML(s))

    # Reuse element ID as display ID, because it's convenient.
    display(HTML("<script>" + js_refresh(element_id, csv_txt) + "</script>"),
            display_id=element_id)
    return element_id


def update_timeline(element_id, csv_txt):
    # Reuse element ID as display ID, because it's convenient.
    update_display(HTML("<script>" + js_refresh(element_id, csv_txt) + "</script>"),
                   display_id=element_id)
