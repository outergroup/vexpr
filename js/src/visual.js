import * as d3 from "d3";
import { play_button_svg, pause_button_svg, restart_button_svg } from "./buttons";


const anim_t = 50;
const param_color = "black";

function hiddenTimeState() {
  let renderTimestep,
      renderTime;

  function render(selection) {
    selection.selectAll(".timeState")
      .data(d => [d])
      .join(enter => {
        const timeState = enter.append("span").attr("class", "timeState");
        timeState.each(timesteps => {
          // Hack: access timeState from here, since "this" is null,
          // but we only want to run this when timeState isn't an
          // empty node.
          timeState.node()._vexprState = {
            userSelectedTimestep: null,
          };
        });
        return timeState;
      })
      .call(timeState => {
        timeState.each(timesteps => {
          let selectedTimestep = timeState.node()._vexprState.userSelectedTimestep;
          if (selectedTimestep == null) {
            selectedTimestep = timesteps.length;
            renderTimestep(timesteps[selectedTimestep - 1]);
          }
          renderTime(1, timesteps.length, selectedTimestep);

          timeState.node()._vexprState.selectTimestep = function(timestep) {
            timeState.node()._vexprState.userSelectedTimestep = timestep;
            renderTimestep(timesteps[Math.floor(timestep) - 1]);
            renderTime(1, timesteps.length, timestep);
          }
        });
      });
  }

  render.renderTimestep = function(_) {
    if (!arguments.length) return renderTimestep;
    renderTimestep = _;
    return render;
  }

  render.renderTime = function(_) {
    if (!arguments.length) return renderTime;
    renderTime = _;
    return render;
  }

  return render;
}

function scalarView() {
  let scale = d3.scaleLinear(),
      eps = 0,
      exponentFormat = false,
      height = 30,
      width = 400,
      fontSize = "13px",
      padRight = 30,
      tfrac = 22.5/30;

  function render(selection) {
    selection.each(function(d) {
      let x = d < 0 ? scale(d) : scale(eps);
      let rectWidth = Math.abs(scale(d) - scale(eps)),
          padding = height / 6;
      d3.select(this)
        .selectAll("svg")
        .data([d])
        .join(enter =>
          enter.append("svg")
            .attr("height", height)
            .style("vertical-align", "middle")
            .call(svg => {
              svg.append("title");

              svg.append('rect')
                .attr('y', padding)
                .attr('height', height - padding*2)
                .style('fill', param_color);

              svg.append('text')
                .attr('y', height * tfrac)
                .attr('fill', param_color)
                .style("font-size", fontSize);

              return svg;
            }))
        .call(svg => {
          svg.transition()
            .duration(anim_t)
            .ease(d3.easeLinear)
            .attr("width", scale(d) + padRight)

          svg.select("title")
            .text(d => d.toPrecision(5));

          svg.select('rect')
            .transition()
            .duration(anim_t)
            .ease(d3.easeLinear)
            .attr('x', x)
            .attr('width', rectWidth);

          svg.select('text')
            .text(exponentFormat ? d.toExponential(1) : d.toPrecision(2))
            .attr('text-anchor', d < 0 ? 'end' : 'start')
            .transition()
            .duration(anim_t)
            .ease(d3.easeLinear)
            .attr('x', d < 0 ? scale(d) - 5 : scale(eps) + rectWidth + 5);

          return svg;
        })
    });
  }

  render.scale = function(value) {
    if (!arguments.length) return scale;
    scale = value;
    return render;
  };

  render.eps = function(value) {
    if (!arguments.length) return eps;
    eps = value;
    return render;
  };

  render.exponentFormat = function(value) {
    if (!arguments.length) return exponentFormat;
    exponentFormat = value;
    return render;
  };

  render.width = function(value) {
    if (!arguments.length) return width;
    width = value;
    return render;
  };

  render.height = function(value) {
    if (!arguments.length) return height;
    height = value;
    return render;
  };

  render.fontSize = function(value) {
    if (!arguments.length) return fontSize;
    fontSize = value;
    return render;
  };

  render.padRight = function(value) {
    if (!arguments.length) return padRight;
    padRight = value;
    return render;
  }

  render.tfrac = function(value) {
    if (!arguments.length) return tfrac;
    tfrac = value;
    return render;
  }

  return render;
}

function positionView() {
  let scale = d3.scaleLinear();

  function render(selection) {
    selection.each(function(d) {
      d3.select(this).selectAll("svg")
        .data([d])
        .join(enter => {
          let svg = enter.append("svg")
              .attr('width', 400)
              .attr('height', 50)
              .style("vertical-align", "middle");

          // Add horizontal number line
          svg.append('line')
            .attr('x1', scale.range()[0])
            .attr('y1', 25)
            .attr('x2', scale.range()[1])
            .attr('y2', 25)
            .style('stroke', 'gray')
            .style('stroke-width', 2);

          // Add xticks
          scale.ticks(5).forEach(d => {
            svg.append('line')
              .attr('x1', scale(d))
              .attr('y1', 20)
              .attr('x2', scale(d))
              .attr('y2', 30)
              .style('stroke', 'gray')
              .style('stroke-width', 2);

            svg.append('text')
              .attr('x', scale(d))
              .attr('y', 50)
              .attr('fill', 'gray')
              .attr('text-anchor', 'middle')
              .text(d.toPrecision(2));
          });

          svg.append('g')
            .attr("class", "position")
            .call(position => {
              position.append("line")
                .attr('y1', 16)
                .attr('y2', 34)
                .style('stroke', param_color)
                .style('stroke-width', 6);
              position.append("text")
                .attr("class", "position-label")
                .attr('y', 10)
                .attr('fill', param_color)
                .style("text-anchor", 'middle')
                .style("font-size", "13px");
            });

          svg.append('title');
          return enter;
        })
        .call(svg => {
          svg.select(".position")
            .call(position => {
              position.transition()
                .duration(anim_t)
                .ease(d3.easeLinear)
                .attr('transform', d => `translate(${scale(d)},0)`)
            })
            .select(".position-label")
            .text(d => d.toPrecision(2));
;
          svg.select("title")
            .text(d => d.toPrecision(8));

          return svg;
        })

    });
  }

  render.scale = function(value) {
    if (!arguments.length) return scale;
    scale = value;
    return render;
  };

  return render;
}

function mixingWeightView() {
  let scale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, 100]);

  function my(selection) {
    selection.selectAll("svg")
      .data(d => [d])
      .join(enter =>
        enter.append("svg")
          .attr("width", 100)
          .attr("height", 10)
          .call(svg => {
            let g = svg.append("g");

            g.append("rect")
              .attr("x", 0)
              .attr("y", 3)
              .attr("width", 100)
              .attr("height", 6)
              .attr("fill", "transparent")
              .attr("stroke", "black");

            g.append("rect")
              .attr("class", "value")
              .attr("x", 0)
              .attr("y", 3)
              .attr("height", 6)
              .attr("fill", param_color);

            g.append("title");

            return svg;
          }))
      .select("g")
      .call(g => {
        g.select("title")
          .text(d => d.toPrecision(5));

        g.select("rect.value")
          .transition()
          .duration(anim_t)
          .ease(d3.easeLinear)
          .attr("width", d => scale(d));
      });
  }

  my.scale = function(_) {
    if (!arguments.length) return scale;
    scale = _;
    return my;
  };

  return my;
}

function timeControl() {
  let onupdate;

  const width = 645,
        padding = {top: 30, right: 95, left: 105, bottom: 60},
        slider_width = width - padding.right - padding.left - 35;

  function renderValue(selection) {
    const slider_container = selection.select(".slider_container_container")
          .select(".slider_container");

    slider_container.select(".slider")
      .property("value", d => d.curr);
    slider_container.select(".slider_text")
      .style("left", d => {
        const scale = d3.scaleLinear()
              .domain([d.min, d.max])
              .range([-32, slider_width - 13 - 32]);
        return `${scale(d.curr)}px`;
      })
      .text(d => `Step ${Math.floor(d.curr)}`);
  }

  function render(selection) {
    selection.each(function(d) {

      // Create a slider for selecting the current timestep

      const container = d3.select(this);

      d3.select(this).selectAll(".slider_container_container")
        .data([d])
        .join(enter => {
          const slider_container_container = enter.append("div")
                .attr("class", "slider_container_container"),
                slider_container = slider_container_container.append("div")
                .attr("class", "slider_container")
                .style("display", "inline-block")
                .style("width", `${slider_width}px`)
                .style("position", "relative")
                .style("margin-left", `${padding.left + 15}px`)
                .style("margin-top", padding.top + "px");

          const slider_ = slider_container.append("input")
                .attr("class", "slider")
                .style("display", "inline")
                .attr("type", "range")
                .attr("id", "timestep")
                .attr("step", "any")
                .style("width", "100%");

          slider_container.append("span")
            .attr("class", "slider_text")
            .style("position", "absolute")
            .style("top", "-20px")
            .style("width", "80px")
            .style("text-align", "center");

          slider_container_container.each(function(d) {
            const slider_container_container = d3.select(this),
                  slider_ = slider_container_container
                  .select(".slider_container")
                  .select(".slider");

            /**
             * Slider logic
             */
            (function() {
              // This is per instance state that should only be created
              // on enter. Thus it needs to run inside of enter.each.
              let stopped = true,
                  pointer_down = false,
                  vStart,
                  tStart = null;

              function restart() {
                slider_.node().value = 1;
                play();
              }

              function getMax() {
                return parseInt(slider_.node().getAttribute("max"));
              }

              function set_stopped_button() {
                pause_button.style("display", "none");
                if (slider_.node().value == getMax()) {
                  play_button.style("display", "none");
                  restart_button.style("display", "inline");
                } else {
                  play_button.style("display", "inline");
                  restart_button.style("display", "none");
                }
              }

              function set_playing_button() {
                play_button.style("display", "none");
                if (slider_.node().value == getMax()) {
                  pause_button.style("display", "none");
                  restart_button.style("display", "inline");
                } else {
                  pause_button.style("display", "inline");
                  restart_button.style("display", "none");
                }
              }

              function pause() {
                stopped = true;
              }

              function play() {
                vStart = parseFloat(slider_.node().value);
                tStart = null;

                if (vStart >= getMax()) {
                  vStart = getMax();
                  stopped = true;
                } else {
                  stopped = false;
                }

                function nextFrame(timestamp) {
                  if (!stopped && !pointer_down) {
                    if (tStart === null) tStart = timestamp;
                    let elapsed = timestamp - tStart;
                    let value = vStart + elapsed / 50;

                    if (value >= getMax()) {
                      value = getMax();
                      stopped = true;
                    }

                    let d = selection.datum();
                    d.curr = value;
                    renderValue(selection.datum(d))
                    slider_.node()._vp_onupdate(value);
                  }

                  if (stopped) {
                    set_stopped_button();
                  } else if (!pointer_down) {
                    requestAnimationFrame(nextFrame);
                  }
                }

                if (!stopped) {
                  requestAnimationFrame(nextFrame);
                  play_button.style("display", "none");
                  pause_button.style("display", "inline");
                  restart_button.style("display", "none");
                }
              }

              let play_button = slider_container_container.append("img")
                  .style("display", "none")
                  .style("margin-left", "10px")
                  .style("margin-top", "-10px")
                  .on("click", play)
                  .attr("width", 13)
                  .attr("src", "data:image/svg+xml;base64," + btoa(play_button_svg)),
                  pause_button = slider_container_container.append("img")
                  .style("display", "none")
                  .style("margin-left", "8px")
                  .style("margin-top", "-10px")
                  .on("click", pause)
                  .attr("width", 15)
                  .attr("src", "data:image/svg+xml;base64," + btoa(pause_button_svg)),
                  restart_button = slider_container_container.append("img")
                  .style("display", "inline")
                  .style("margin-left", "8px")
                  .style("margin-top", "-10px")
                  .on("click", restart)
                  .attr("width", 15)
                  .attr("src", "data:image/svg+xml;base64," + btoa(restart_button_svg));

              slider_
                .on("input", () => {
                  let v = parseFloat(slider_.node().value);
                  if (v >= getMax()) {
                    v = getMax();
                    stopped = true;
                  }

                  if (stopped) {
                    set_stopped_button();
                  } else {
                    vStart = parseFloat(slider_.node().value);
                    tStart = null;
                    play();
                    set_playing_button();
                  }
                  slider_.node()._vp_onupdate(v);

                  let d = selection.datum();
                  d.curr = v;
                  renderValue(selection.datum(d))
                })
                .on("pointerdown", () => {
                  pointer_down = true;
                })
                .on("pointerup", () => {
                  pointer_down = false;
                  if (!stopped) play();
                });
            })();
          });

          return slider_container_container;
        })
        .call(slider_container_container => {
          const slider_container = slider_container_container.select(".slider_container"),
                slider_ = slider_container.select(".slider");

          slider_
            .attr("min", d => d.min)
            .attr("max", d => d.max);

          slider_.node()._vp_onupdate = onupdate;
          renderValue(selection);
        });
    });
  }

  render.onupdate = function(_) {
    if (!arguments.length) return onupdate;
    onupdate = _;
    return render;
  };

  return render;
}


function expressionView(expr, keys) {
  const exprHTML = (function() {
    let exprHTML = expr;

    exprHTML = expr
      .replace(
        // Detect and color the comments
        /#.*\n/g, match => `<span style="color:green;">${match}</span>`)
      .replace(
        // Detect and color strings
        /'.*?'/g, match => `<span style="color:brown;">${match}</span>`);

    keys.forEach(key => {
      if (key.startsWith("$W")) {
        exprHTML = exprHTML.replace(
          key, `<span class="weight-value" style="color:blue;" data-key="${key}"></span>`);
      } else if (key.startsWith("$S")) {
        exprHTML = exprHTML.replace(
          key, `<span class="scale-value" style="color:red;" data-key="${key}"></span>`);
      } else if (key.startsWith("$LS")) {
        exprHTML = exprHTML.replace(
          key, `<span class="lengthscale-value" style="color:red;" data-key="${key}"></span>`);
      } else {
        exprHTML = exprHTML.replace(
          key, `<span class="parameter-value" style="color:blue;" data-key="${key}"></span>`);
      }
    })

    exprHTML = `<span style='color:gray;'>${exprHTML}</span>`;
    return exprHTML;
  })();

  function render(selection) {
    selection.each(function(model) {
      const expression = d3.select(this)
            .selectAll(".expression")
            .data([model])
            .join(enter => enter.append("span")
                  .attr("class", "expression")
                  .html(d => exprHTML));

      expression.selectAll("span.scale-value").each(function(_) {
        const scaleValue = d3.select(this),
              k = scaleValue.attr("data-key");
        scaleValue.datum(model[k])
          .call(scalarView()
                .scale(d3.scaleLinear().domain([0, 2.5]).range([0, 200]))
                .height(12)
                .width(200)
                .fontSize(10));
      });
      expression.selectAll("span.lengthscale-value").each(function(_) {
        let lengthscaleValue = d3.select(this);
        let k = lengthscaleValue.attr("data-key");
        lengthscaleValue.datum(model[k])
          .call(scalarView()
                .scale(d3.scaleLinear().domain([0, 3.9]).range([0, 200]))
                .height(12)
                .width(200)
                .fontSize(9));
      });
      expression.selectAll("span.weight-value").each(function(_) {
        let weightValue = d3.select(this);
        let k = weightValue.attr("data-key");
        weightValue.datum(model[k])
          .call(mixingWeightView());
      });
    });
  }

  return render;
}


export { hiddenTimeState, scalarView, expressionView, positionView, timeControl };
