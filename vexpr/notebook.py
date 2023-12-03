import uuid

import vexpr.web
from IPython.display import HTML, display, update_display



def init_notebook_mode():
    display(HTML(
        vexpr.web.header_content() + """
    <script>
      if (window.vexprQueue) {{
        window.vexprQueue.forEach(([k, f]) => f());
        window.vexprQueue = null;
      }}
    </script>
    """))


def js_refresh(element_id, encoded_data):
    return f"""
    (function() {{
      function update() {{
        document.getElementById("{element_id}")._vexprState.refresh(`{encoded_data}`);
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


def visualize_snapshot(html_preamble, encoded_to_snapshot, components, encoded_data):
    element_id = str(uuid.uuid1())
    components_str = "\n".join(components)

    s = f"""
    {html_preamble(element_id)}
    <script>
    function initialize() {{
      let renderTimestepFunctions = [];

      const container = document.getElementById("{element_id}");
      container._vexprState = {{
        refresh: function(encodedData) {{
          {encoded_to_snapshot}
          renderTimestepFunctions.forEach(render => render(snapshot));
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
    display(HTML("<script>" + js_refresh(element_id, encoded_data) + "</script>"),
            display_id=element_id)
    return element_id


def visualize_timeline(html_preamble, encoded_to_timesteps, components, encoded_data):
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
        refresh: function(encodedData) {{
          {encoded_to_timesteps}
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
    display(HTML("<script>" + js_refresh(element_id, encoded_data) + "</script>"),
            display_id=element_id)
    return element_id


def update_timeline(element_id, encoded_data):
    # Reuse element ID as display ID, because it's convenient.
    update_display(HTML("<script>" + js_refresh(element_id, encoded_data) + "</script>"),
                   display_id=element_id)


def update_snapshot(element_id, encoded_data):
    # This is currently identical to update_timeline, but that fact might change.
    return update_timeline(element_id, encoded_data)
