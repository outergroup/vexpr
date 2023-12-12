import * as d3 from "d3";
import { play_button_svg, pause_button_svg, restart_button_svg } from "./buttons";


const anim_t = 200;

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
            renderedTimestep: null,
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
            let state = timeState.node()._vexprState;

            const tDiscrete = Math.floor(timestep);
            if (tDiscrete == timesteps.length) {
              state.userSelectedTimestep = null;
            } else {
              state.userSelectedTimestep = tDiscrete;
            }

            if (tDiscrete != state.renderedTimestep) {
              renderTimestep(timesteps[tDiscrete - 1]);
              state.renderedTimestep = tDiscrete;
            }

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
                .style('fill', "currentColor");

              svg.append('text')
                .attr('y', height * tfrac)
                .attr('fill', "currentColor")
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

function scalarDistributionView() {
  let scale = d3.scaleLinear(),
      exponentFormat = false,
      height = 30,
      fontSize = "13px",
      useDataMin = false,
      useDataMax = false,
      cnvMult = 4,
      pointRadius = 2.5,
      maxWidth = 300;

  let cnv_x, cnv_y, cnv_y05, cnv_y0, cnv_y1;
  function onScaleUpdate() {
    cnv_x = d3.scaleLinear()
        .domain(scale.domain())
        .range([cnvMult * (1 + scale.range()[0]), cnvMult * (scale.range()[1] + 1)]);
    cnv_y = d3.scaleLinear()
        .domain([0, 1])
        .range([0, cnvMult * height]);
    cnv_y05 = cnv_y(0.5);
    cnv_y0 = cnv_y(0);
    cnv_y1 = cnv_y(1);
  }

  onScaleUpdate();

  function fmin(points) {
    if (useDataMin) {
      return d3.min(points);
    } else {
      return scale.domain()[0];
    }
  }

  function fmax(points) {
    if (useDataMax) {
      return d3.max(points);
    } else {
      return scale.domain()[1];
    }
  }

  const fmintext = points => exponentFormat
        ? fmin(points).toExponential(1)
        : toPrecisionThrifty(fmin(points), 2),
        fmaxtext = points => exponentFormat
        ? fmax(points).toExponential(1)
        : toPrecisionThrifty(fmax(points), 2);

  function draw(canvasNode, div, points) {
    const min = useDataMin
      ? d3.min(points)
      : scale.domain()[0],
      max = useDataMax
      ? d3.max(points)
      : scale.domain()[1],
      width = scale(max) - scale(min);

    if (!canvasNode._vexprInitialized || (useDataMin || useDataMax)) {
      div.select(".mintext")
      // If this text is rapidly animating, the varying precision to
      // toPrecisionThrifty causes too much flickering.
        .text(useDataMin ? min.toPrecision(2)
                         : toPrecisionThrifty(min, 2));
      div.select(".maxtext")
        .text(useDataMax ? max.toPrecision(2)
                         : toPrecisionThrifty(max, 2));
      div.select(".canvasContainer")
        .style("width", `${width}px`);
      canvasNode._vexprInitialized = true;
    }

    let ctx = canvasNode.getContext("2d");
    ctx.clearRect(0, 0, canvasNode.width, canvasNode.height)
    ctx.fillStyle = "blue";
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 0;
    for (let i = 0; i < points.length; i++) {
      ctx.beginPath();
      ctx.arc(cnv_x(points[i]), cnv_y05, pointRadius*cnvMult, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function render(selection) {
    selection.selectAll(".visualization")
      .data(d => [d])
      .join(enter => enter.append("div")
            .attr("class", "visualization")
            .style("position", "relative")
            .style("vertical-align", "middle")
            .style("display", "inline-block")
            .style("padding-right", "12px")
            .call(div => {
              div.append('span')
                .attr("class", "mintext")
                .style("font-size", fontSize)
                .style("color", "gray")
                .style("position", "relative")
                .text(fmintext);

              div.append("div")
              .attr("class", "canvasContainer")
              .style("display", "inline-block")
              .style("position", "relative")
              .style("overflow", "hidden")
              .style("left", "4px")
              .style("height", `${height}px`)
              .style("border-left", "2px solid gray")
              .style("border-right", "2px solid gray")
              .style("background-color", "silver")
              .append("canvas")
                .attr("height", cnvMult * height)
                .style("height", `${height}px`)
                .attr("width", cnvMult * maxWidth)
                .style("width", `${maxWidth}px`)
                .style("vertical-align", "top");

              div.append('span')
                .attr("class", "maxtext")
                .style("font-size", fontSize)
                .style("color", "gray")
                .style("position", "relative")
                .style("left", "8px")
                .text(fmaxtext);
            }))
      .call(div => {
        div.select(".canvasContainer").select("canvas").each(function(points) {
          const canvas = d3.select(this);
          if (this._vexprPrevPoints) {
            const interpolator = d3.interpolateNumberArray(this._vexprPrevPoints, points);

            let timer = d3.timer((elapsed) => {
              const t = Math.min(1, d3.easeCubic(elapsed / anim_t));
              draw(this, div, interpolator(t));
              if (t === 1) {
                timer.stop();
              }
            });

          } else {
            draw(this, div, points);
          }

          this._vexprPrevPoints = points;

        });
      });
  }

  render.scale = function(value) {
    if (!arguments.length) return scale;
    scale = value;
    onScaleUpdate();
    return render;
  };

  render.useDataMin = function(value) {
    if (!arguments.length) return useDataMin;
    useDataMin = value;
    return render;
  };

  render.useDataMax = function(value) {
    if (!arguments.length) return useDataMax;
    useDataMax = value;
    return render;
  };

  render.exponentFormat = function(value) {
    if (!arguments.length) return exponentFormat;
    exponentFormat = value;
    return render;
  };

  render.height = function(value) {
    if (!arguments.length) return height;
    height = value;
    onScaleUpdate();
    return render;
  };

  render.fontSize = function(value) {
    if (!arguments.length) return fontSize;
    fontSize = value;
    return render;
  };

  return render;
}

function toPrecisionThrifty(d, precision) {
  const fullPrecision = d.toPrecision(precision),
        parsedPrecise = parseFloat(fullPrecision);

  for (let i = 1; i < precision; i++) {
    const candidate = d.toPrecision(i);
    if (parseFloat(candidate) == parsedPrecise) {
      return candidate;
    }
  }

  return fullPrecision;
}

function scalarDistributionListView() {
  let scale = d3.scaleLinear(),
      exponentFormat = false,
      height = 30,
      fontSize = "13px",
      rowHeight = 2,
      pointRadius = 1,
      useDataMin = false,
      useDataMax = false,
      cnvMult = 4;

  function render(selection) {
    const fmin = useDataMin
          ? pointsListsData => pointsListsData.min
          : pointsListsData => scale.domain()[0],
          fmax = useDataMax
          ? pointsListsData => pointsListsData.max
          : pointsListsData => scale.domain()[1],
          fxmax = pointsListsData => scale(fmax(pointsListsData)),
          fxmin = pointsListsData => scale(fmin(pointsListsData)),
          fwidth = pointsListsData => fxmax(pointsListsData) - fxmin(pointsListsData),
          fheight = pointsListsData => pointsListsData.pointsLists.length * rowHeight,
          fmintext = pointsListsData => exponentFormat
          ? fmin(pointsListsData).toExponential(1)
          : toPrecisionThrifty(fmin(pointsListsData), 2),
          fmaxtext = pointsListsData => exponentFormat
          ? fmax(pointsListsData).toExponential(1)
          : toPrecisionThrifty(fmax(pointsListsData), 2);

    selection.selectAll(".visualizationContainer")
      .data(d => [d])
      .join(enter => enter.append("div")
            .attr("class", "visualizationContainer")
            .style("display", "inline-block")
            .style("vertical-align", "middle")
            .style("border", "1px solid black")
            .style("border-radius", "3px")
            .style("padding-left", "5px")
            .style("padding-right", "12px")
            .style("margin-top", "1px")
            .style("margin-bottom", "1px")
            .call(div => {
              div.append('span')
                .attr('class', 'min-text')
                .style("font-size", fontSize)
                .style("color", "gray")
                .style("position", "relative")
                .style("top", d => `-${d.pointsLists.length * rowHeight / 2 - 5}px`)
                .text(fmintext);

              div.append("canvas")
                .style("position", "relative")
                .style("left", "4px");

              div.append('span')
                .attr('class', 'max-text')
                .style("font-size", fontSize)
                .style("color", "gray")
                .style("position", "relative")
                .style("top", pointsListsData =>
                  `-${pointsListsData.pointsLists.length * rowHeight / 2 - 5}px`)
                .style("left", "8px")
                .text(fmaxtext);
            }))
      .call(div => {
        div.style("height", pointsListsData => `${fheight(pointsListsData)}px`);

        if (useDataMin) {
          div.select(".min-text")
            .text(fmintext);
        }

        if (useDataMax) {
          div.select(".max-text")
            .text(fmaxtext);
        }

        let canvas = div.select("canvas")
          .attr("width", d => cnvMult * (fwidth(d) + 2))
          .attr("height", d => cnvMult * fheight(d))
          .style("width", d => `${fwidth(d) + 2}px`)
          .style("height", d => `${fheight(d)}px`);

        canvas.each(function(pointsListsData) {
          let ctx = this.getContext("2d");

          const cnv_x = d3.scaleLinear()
            .domain(scale.domain())
            .range([cnvMult * (1 + scale.range()[0]), cnvMult * (scale.range()[1] + 1)]),
                cnv_y = d3.scaleLinear()
            .domain([0, pointsListsData.pointsLists.length - 1])
            .range([cnvMult * 0.5, cnvMult * rowHeight * (pointsListsData.pointsLists.length + 0.5)]);

          // draw silver rect
          ctx.fillStyle = "silver";
          ctx.fillRect(0, 0, cnvMult * fwidth(pointsListsData), cnvMult * fheight(pointsListsData));

          ctx.strokeStyle = "gray";
          ctx.lineWidth = 2 * cnvMult;

          ctx.beginPath();
          ctx.moveTo(cnv_x(fmin(pointsListsData)), cnv_y(0));
          ctx.lineTo(cnv_x(fmin(pointsListsData)), cnv_y(pointsListsData.pointsLists.length - 1));
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(cnv_x(fmax(pointsListsData)), cnv_y(0));
          ctx.lineTo(cnv_x(fmax(pointsListsData)), cnv_y(pointsListsData.pointsLists.length - 1));
          ctx.stroke();

          ctx.fillStyle = "blue";
          ctx.globalAlpha = 0.4;

          pointsListsData.pointsLists.forEach((points, i) => {
            points.forEach(point => {
              ctx.beginPath();
              ctx.arc(cnv_x(point), cnv_y(i), pointRadius*cnvMult, 0, 2 * Math.PI);
              ctx.fill();
            });
          });
        });
      });
  }

  render.scale = function(value) {
    if (!arguments.length) return scale;
    scale = value;
    return render;
  };

  render.useDataMin = function(value) {
    if (!arguments.length) return useDataMin;
    useDataMin = value;
    return render;
  };

  render.useDataMax = function(value) {
    if (!arguments.length) return useDataMax;
    useDataMax = value;
    return render;
  };

  render.exponentFormat = function(value) {
    if (!arguments.length) return exponentFormat;
    exponentFormat = value;
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
                .style('stroke', 'currentColor')
                .style('stroke-width', 6);
              position.append("text")
                .attr("class", "position-label")
                .attr('y', 10)
                .attr('fill', 'currentColor')
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
              .attr("stroke", "currentColor");

            g.append("rect")
              .attr("class", "value")
              .attr("x", 0)
              .attr("y", 3)
              .attr("height", 6)
              .attr("fill", 'currentColor');

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

  const padding = {top: 5, right: 0, left: 8, bottom: 10},
        msPerStep = 200;

  function renderValue(selection) {
    const slider_container = selection.select(".slider_container_container")
          .select(".slider_container");

    slider_container.select(".slider")
      .property("value", d => d.curr);
    slider_container.select(".slider_text")
      .style("left", d => {
        const pctScale = d3.scaleLinear()
        .domain([d.min, d.max])
        .range([0, 100]),
              pxScale = d3.scaleLinear()
        .domain([d.min, d.max])
        .range([-30, -45]);
        return `calc(${pctScale(d.curr)}% + ${pxScale(d.curr)}px)`;
      })
      .text(d => {
        const step = Math.floor(d.curr),
              stepLabel = d.labels !== undefined ? d.labels[step - d.min] : step;
        return `Step ${stepLabel}`;
      });
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
                .style("width", `calc(100% - ${25 + padding.left}px)`)
                .style("position", "relative")
                .style("margin", `${padding.top}px ${padding.right}px ${padding.bottom}px ${padding.left}px`);

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
            .style("width", "80px")  // avoid text wrapping when it's near an edge
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
                  vStart;

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

                if (vStart >= getMax()) {
                  vStart = getMax();
                  stopped = true;
                } else {
                  stopped = false;
                }

                if (!stopped) {
                  let timer = d3.timer((elapsed) => {
                    if (!stopped && !pointer_down) {
                      let value = vStart + elapsed / msPerStep;
  
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
                    }
  
                    if (stopped || pointer_down) {
                      timer.stop();
                    }
                  });

                  play_button.style("display", "none");
                  pause_button.style("display", "inline");
                  restart_button.style("display", "none");
                }
              }

              let play_button = slider_container_container.append("span")
                  .style("display", "none")
                  .style("margin-left", "10px")
                  .style("margin-top", "-10px")
                  .on("click", play)
                  .html(play_button_svg),
                  pause_button = slider_container_container.append("span")
                  .style("display", "none")
                  .style("margin-left", "8px")
                  .style("margin-top", "-10px")
                  .on("click", pause)
                  .html(pause_button_svg),
                  restart_button = slider_container_container.append("span")
                  .style("display", "inline")
                  .style("margin-left", "8px")
                  .style("margin-top", "-10px")
                  .on("click", restart)
                  .html(restart_button_svg);

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
          key, `<span class="mixing-weight-value" data-key="${key}"></span>`);
      } else if (key.startsWith("$S")) {
        exprHTML = exprHTML.replace(
          key, `<span class="scale-value" data-key="${key}"></span>`);
      } else {
        exprHTML = exprHTML.replace(
          key, `<span class="parameter-value" data-key="${key}"></span>`);
      }
    });

    // Find all text that hasn't yet been wrapped in spans. This gives us the
    // ability to style this text without affecting the style of the other
    // spans, enabling them to inherit dark mode colors.
    const parsed = (new DOMParser()).parseFromString(exprHTML, 'text/html');
    exprHTML = "";
    parsed.body.childNodes.forEach(node => {
      if (node.nodeType == Node.TEXT_NODE) {
        exprHTML += `<span class="vexpr-code">${node.textContent}</span>`;
      } else {
        exprHTML += node.outerHTML;
      }
    });

    return exprHTML;
  })();

  let valueType = "scalar",
      asynchronous = false,
      onfinished = () => null,
      scaleComponent,
      mixingWeightComponent;

  function initializeComponents() {
    if (valueType == "scalar") {
      scaleComponent = scalarView()
        .scale(d3.scaleLinear().domain([0, 2.5]).range([0, 200]))
        .height(12)
        .fontSize(10);
      mixingWeightComponent = mixingWeightView();
    } else if (valueType == "scalarDistribution") {
      scaleComponent = scalarDistributionView()
        .scale(d3.scaleLinear().domain([0, 2.5]).range([0, 200]))
        .useDataMax(true)
        .height(12)
        .fontSize(10);
      mixingWeightComponent = scalarDistributionView()
        .scale(d3.scaleLinear().domain([0, 1]).range([0, 50]))
        .height(12)
        .fontSize(10);
    } else if (valueType == "scalarDistributionList") {
      scaleComponent = undefined;
      mixingWeightComponent = undefined;
    }
  }

  initializeComponents();

  function render(selection) {
    selection.each(function(model) {
      const expression = d3.select(this)
            .selectAll(".expression")
            .data([model])
            .join(enter => enter.append("span")
                  .attr("class", "expression")
                  .html(d => exprHTML));

      if (valueType == "scalar") {
        expression.selectAll("span.scale-value").each(function(_) {
          const scaleValue = d3.select(this),
                k = scaleValue.attr("data-key");
  
          scaleValue.datum(model[k])
            .call(scaleComponent);
        });
        expression.selectAll("span.mixing-weight-value").each(function(_) {
          let weightValue = d3.select(this);
          let k = weightValue.attr("data-key");
          weightValue.datum(model[k])
            .call(mixingWeightComponent);
        });
      } else if (valueType == "scalarDistribution") {
          expression.selectAll("span.scale-value").each(function(_) {
            const scaleValue = d3.select(this),
                  k = scaleValue.attr("data-key");
    
            scaleValue.datum(model[k])
              .call(scaleComponent);
          });
          expression.selectAll("span.mixing-weight-value").each(function(_) {
            let weightValue = d3.select(this);
            let k = weightValue.attr("data-key");
            weightValue.datum(model[k])
              .call(mixingWeightComponent);
          });
      } else if (valueType == "scalarDistributionList") {

        let scaleValue = expression.selectAll("span.scale-value"),
            scaleKeys = scaleValue.nodes().map(n => n.getAttribute("data-key")),
            allPointsLists = scaleKeys.map(k => model[k]),
            mins = allPointsLists.map(pointsLists => d3.min(pointsLists, points => d3.min(points))),
            maxs = allPointsLists.map(pointsLists => d3.max(pointsLists, points => d3.max(points))),
            globalMin = d3.min(mins),
            globalMax = d3.max(maxs),
            data = allPointsLists.map((pointsLists, i) => {
              return {min: mins[i], max: maxs[i],
                      pointsLists};
            });

        let scaleComponent = scalarDistributionListView()
          .scale(d3.scaleLinear().domain([0, globalMax]).range([0, 200]))
          .height(10)
          .useDataMax(true)
          .fontSize(10);

        let topLevelFinished = [false, false];

        if (asynchronous) {
          let finished = new Array(data.length).fill(false);
          scaleValue.data(data)
            .each(function(d, i) {
              setTimeout(() => {
                d3.select(this).call(scaleComponent);
                finished[i] = true;
                if (finished.every(d => d)) {
                  topLevelFinished[0] = true;
                  if (topLevelFinished.every(d => d)) onfinished();
                }
              }, 0);
            });
        } else {
          scaleValue.data(data).call(scaleComponent);
        }

        let mixingWeightComponent = scalarDistributionListView()
          .scale(d3.scaleLinear().domain([0, 1]).range([0, 50]))
          .height(10)
          .fontSize(10);

        let mixingWeightValue = expression.selectAll("span.mixing-weight-value"),
            mixingWeightKeys = mixingWeightValue.nodes().map(n => n.getAttribute("data-key")),
            mwData = mixingWeightKeys.map(
              k => { return {pointsLists: model[k]}; });

        if (asynchronous) {
          let finished = new Array(mwData.length).fill(false);
          mixingWeightValue.data(mwData)
            .each(function(d, i) {
              setTimeout(() => {
                d3.select(this).call(mixingWeightComponent);
                finished[i] = true;
                if (finished.every(d => d)) {
                  topLevelFinished[1] = true;
                  if (topLevelFinished.every(d => d)) onfinished();
                }
              }, 0);
            });
        } else {
          mixingWeightValue.data(mwData).call(mixingWeightComponent);
        }
      }
    });
  }

  render.valueType = function(_) {
    if (!arguments.length) return valueType;
    valueType = _;
    initializeComponents();
    return render;
  };

  render.asynchronous = function(_) {
    if (!arguments.length) return asynchronous;
    asynchronous = _;
    return render;
  };

  render.onfinished = function(_) {
    if (!arguments.length) return onfinished;
    onfinished = _;
    return render;
  };

  return render;
}


export {
  expressionView,
  hiddenTimeState,
  positionView,
  scalarDistributionListView,
  scalarDistributionView,
  scalarView,
  timeControl,
};
