(function () {
  var root = document.getElementById("multimetric-chart");
  var dataNode = document.getElementById("dashboard-multimetric-data");
  var payload = dataNode ? JSON.parse(dataNode.textContent) : null;
  if (payload) {
  var METRIC_CONFIG = {
    payout: {
      standard: {title: "Per-Era Net Return", tickformat: ".2%", hoverformat: ".2%"},
      cumulative: {title: "Cumulative Wealth (1.0 Stake)", tickformat: ".3f", hoverformat: ".3f"}
    },
    corr20: {standard: {title: "Per-Era CORR (20D)", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative CORR (20D)", tickformat: ".4f", hoverformat: ".4f"}},
    mmc20:  {standard: {title: "Per-Era MMC (20D)", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative MMC (20D)", tickformat: ".4f", hoverformat: ".4f"}},
    corr60: {standard: {title: "Per-Era CORR (60D)", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative CORR (60D)", tickformat: ".4f", hoverformat: ".4f"}},
    mmc60:  {standard: {title: "Per-Era MMC (60D)", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative MMC (60D)", tickformat: ".4f", hoverformat: ".4f"}},
    bmc:    {standard: {title: "Per-Era BMC", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative BMC", tickformat: ".4f", hoverformat: ".4f"}},
    cwmm:   {standard: {title: "Per-Era CWMM", tickformat: ".4f", hoverformat: ".4f"},
              cumulative: {title: "Cumulative CWMM", tickformat: ".4f", hoverformat: ".4f"}}
  };
  var currentMetric = "payout";
  var currentView = "standard";
  var mounted = false;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function stressShapes() {
    var shapes = [];
    var eras = payload.eras;
    var mask = payload.meta_downside_mask || [];
    var start = null;
    for (var i = 0; i <= mask.length; i++) {
      if (mask[i] && start === null) { start = i; }
      else if (!mask[i] && start !== null) {
        shapes.push({type: "rect", xref: "x", yref: "paper", x0: eras[start],
                      x1: eras[i - 1], y0: 0, y1: 1,
                      fillcolor: "rgba(248, 81, 73, 0.10)", line: {width: 0},
                      layer: "below"});
        start = null;
      }
    }
    return shapes;
  }

  function applyState() {
    var metric = payload.metrics[currentMetric] || {};
    var cfg = METRIC_CONFIG[currentMetric][currentView];
    var traces = [];
    var ids = Object.keys(metric).sort();
    for (var i = 0; i < ids.length; i++) {
      var series = metric[ids[i]];
      traces.push({
        x: payload.eras,
        y: series[currentView],
        mode: "lines",
        name: series.label,
        hovertemplate: "%{y:" + cfg.hoverformat + "}" + "<extra>" + esc(series.label) + "</extra>"
      });
    }
    var layout = {
      template: "plotly_dark",
      showlegend: false,
      margin: {l: 20, r: 20, t: 50, b: 20},
      xaxis: {title: "Era"},
      yaxis: {title: cfg.title, tickformat: cfg.tickformat},
      shapes: stressShapes()
    };
    if (!mounted) {
      Plotly.newPlot(root, traces, layout);
      mounted = true;
    } else {
      Plotly.react(root, traces, layout);
    }
  }

  var controls = document.createElement("div");
  controls.style.cssText = "display:flex; gap:1rem; align-items:center; margin-bottom:0.5rem;";
  var select = document.createElement("select");
  select.innerHTML = '<option value="payout">Net Payout Return</option>'
    + '<option value="corr20">CORR (20D)</option>'
    + '<option value="mmc20">MMC (20D)</option>'
    + '<option value="corr60">CORR (60D)</option>'
    + '<option value="mmc60">MMC (60D)</option>'
    + '<option value="bmc">BMC</option>'
    + '<option value="cwmm">CWMM</option>';
  select.addEventListener("change", function () {
    currentMetric = select.value;
    applyState();
  });
  var stdButton = document.createElement("button");
  stdButton.textContent = "Standard View";
  var cumButton = document.createElement("button");
  cumButton.textContent = "Cumulative View";
  stdButton.addEventListener("click", function () { currentView = "standard"; applyState(); });
  cumButton.addEventListener("click", function () { currentView = "cumulative"; applyState(); });
  controls.appendChild(select);
  controls.appendChild(stdButton);
  controls.appendChild(cumButton);
  root.parentNode.insertBefore(controls, root);
  applyState();
  }
})();
