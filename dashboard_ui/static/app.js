/* dashboard_ui/static/app.js
   Vanilla SVG renderer for the executive report. Reads the #dashboard-data
   JSON node and renders the five chart sections into #dashboard-root.
   All geometry mirrors the tested Python reference in dashboard_ui/charts.py
   (data_to_svg_path / svg_area_path / cumulative_series / drawdown_series). */
(function () {
  "use strict";

  var root = document.getElementById("dashboard-root");
  var dataNode = document.getElementById("dashboard-data");
  if (!root || !dataNode) return;
  var payload;
  try {
    payload = JSON.parse(dataNode.textContent);
  } catch (err) {
    return; /* corrupt payload: keep the static report readable */
  }

  var METRIC_CONFIG = {
    payout: {standard: {label: "Per-Era Net Return", percent: true},
             cumulative: {label: "Cumulative Wealth (1.0 Stake)", percent: false}},
    corr20: {standard: {label: "Per-Era CORR (20D)", percent: false},
             cumulative: {label: "Cumulative CORR (20D)", percent: false}},
    mmc20:  {standard: {label: "Per-Era MMC (20D)", percent: false},
             cumulative: {label: "Cumulative MMC (20D)", percent: false}},
    corr60: {standard: {label: "Per-Era CORR (60D)", percent: false},
             cumulative: {label: "Cumulative CORR (60D)", percent: false}},
    mmc60:  {standard: {label: "Per-Era MMC (60D)", percent: false},
             cumulative: {label: "Cumulative MMC (60D)", percent: false}},
    bmc:    {standard: {label: "Per-Era BMC", percent: false},
             cumulative: {label: "Cumulative BMC", percent: false}},
    cwmm:   {standard: {label: "Per-Era CWMM", percent: false},
             cumulative: {label: "Cumulative CWMM", percent: false}}
  };
  var COLORS = ["#58a6ff", "#3fb950", "#d29922", "#a371f7", "#f85149", "#79c0ff", "#f0883e"];
  var TS = {width: 800, height: 320, pad: {top: 24, right: 24, bottom: 40, left: 56}};
  var LB = {width: 800, height: 420, pad: {top: 16, right: 40, bottom: 24, left: 190}};
  var DD = {width: 800, height: 240, pad: {top: 24, right: 24, bottom: 40, left: 56}};
  var currentMetric = "payout";
  var currentView = "standard";
  var eras = payload.eras || [];
  var stressMask = payload.meta_downside_mask || [];
  var metrics = payload.metrics || {};
  var crosshair = null;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function fmt(v, percent) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return percent ? (v * 100).toFixed(2) + "%" : v.toFixed(4);
  }

  function cumulativeSeries(values, payout) {
    var out = [];
    var acc = payout ? 1.0 : 0.0;
    for (var i = 0; i < values.length; i++) {
      acc = payout ? acc * (1.0 + values[i]) : acc + values[i];
      out.push(acc);
    }
    return out;
  }

  function drawdownSeries(cumulative) {
    var out = [];
    var peak = -Infinity;
    for (var i = 0; i < cumulative.length; i++) {
      if (cumulative[i] > peak) peak = cumulative[i];
      out.push(peak > 0 ? cumulative[i] / peak - 1.0 : 0.0);
    }
    return out;
  }

  function globalYRange(seriesList) {
    var lo = Infinity, hi = -Infinity;
    for (var s = 0; s < seriesList.length; s++) {
      var arr = seriesList[s];
      for (var i = 0; i < arr.length; i++) {
        var v = arr[i];
        if (!isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo)) return {min: 0.0, max: 1.0};
    return {min: lo, max: hi};
  }

  function dataToSvgPath(values, yMin, yMax, width, height, pad) {
    if (!values || !values.length) return "";
    var span = yMax - yMin;
    if (Math.abs(span) < 1e-12) { yMin -= 1.0; yMax += 1.0; span = 2.0; }
    var innerW = width - pad.left - pad.right;
    var innerH = height - pad.top - pad.bottom;
    var pts = [];
    var n = values.length;
    var denom = Math.max(1, n - 1);
    for (var i = 0; i < n; i++) {
      var x = pad.left + (i / denom) * innerW;
      var y = pad.top + (1.0 - (values[i] - yMin) / span) * innerH;
      pts.push(x.toFixed(1) + "," + y.toFixed(1));
    }
    return "M " + pts.join(" L ");
  }

  function svgAreaPath(values, yMin, yMax, yBase, width, height, pad) {
    var line = dataToSvgPath(values, yMin, yMax, width, height, pad);
    if (!line) return "";
    var span = yMax - yMin;
    if (Math.abs(span) < 1e-12) { yMin -= 1.0; yMax += 1.0; span = 2.0; }
    var innerH = height - pad.top - pad.bottom;
    var yBaseSvg = pad.top + (1.0 - (yBase - yMin) / span) * innerH;
    var innerW = width - pad.left - pad.right;
    var denom = Math.max(1, values.length - 1);
    var x0 = pad.left;
    var xN = pad.left + ((values.length - 1) / denom) * innerW;
    return line + " L " + xN.toFixed(1) + "," + yBaseSvg.toFixed(1) +
           " L " + x0.toFixed(1) + "," + yBaseSvg.toFixed(1) + " Z";
  }

  function activeSeries() {
    var metric = metrics[currentMetric] || {};
    var ids = Object.keys(metric).sort();
    var series = [];
    for (var i = 0; i < ids.length; i++) {
      var entry = metric[ids[i]];
      var standard = entry.standard || [];
      var values = currentView === "cumulative"
        ? cumulativeSeries(standard, currentMetric === "payout")
        : standard;
      series.push({id: ids[i], label: entry.label, values: values,
                   color: COLORS[i % COLORS.length]});
    }
    return series;
  }

  function stressShapes(svg) {
    var innerW = TS.width - TS.pad.left - TS.pad.right;
    var denom = Math.max(1, eras.length - 1);
    for (var i = 0; i < stressMask.length; i++) {
      if (!stressMask[i]) continue;
      var start = i;
      while (i + 1 < stressMask.length && stressMask[i + 1]) i++;
      var x0 = TS.pad.left + (start / denom) * innerW;
      var x1 = TS.pad.left + (i / denom) * innerW;
      var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x0.toFixed(1));
      rect.setAttribute("width", (x1 - x0).toFixed(1));
      rect.setAttribute("y", "0");
      rect.setAttribute("height", String(TS.height));
      rect.setAttribute("fill", "rgba(248, 81, 73, 0.10)");
      svg.appendChild(rect);
    }
  }

  function gridLines(svg, yMin, yMax, pad, fmtVal) {
    var span = yMax - yMin;
    var ticks = (yMin <= 0.0 && yMax >= 0.0)
      ? [yMax, 0.0, yMin]
      : [yMax, (yMin + yMax) / 2.0, yMin];
    var innerH = TS.height - pad.top - pad.bottom;
    for (var k = 0; k < ticks.length; k++) {
      var val = ticks[k];
      var y = pad.top + (1.0 - (val - yMin) / span) * innerH;
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "grid-line");
      line.setAttribute("x1", String(pad.left));
      line.setAttribute("x2", String(TS.width - pad.right));
      line.setAttribute("y1", y.toFixed(1));
      line.setAttribute("y2", y.toFixed(1));
      svg.appendChild(line);
      var label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "axis-text");
      label.setAttribute("x", "4");
      label.setAttribute("y", (y - 4).toFixed(1));
      label.textContent = fmtVal(val);
      svg.appendChild(label);
    }
  }

  function renderTimeseries() {
    var svg = document.getElementById("timeseries-svg");
    if (!svg) return;
    var series = activeSeries();
    var cfg = METRIC_CONFIG[currentMetric][currentView];
    var labelSpan = document.getElementById("axis-label");
    if (labelSpan) labelSpan.textContent = cfg.label;
    var range = globalYRange(series.map(function (s) { return s.values; }));
    if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1.0; range.max += 1.0; }
    svg.textContent = "";
    stressShapes(svg);
    gridLines(svg, range.min, range.max, TS.pad, function (v) { return fmt(v, cfg.percent); });
    for (var i = 0; i < series.length; i++) {
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", dataToSvgPath(series[i].values, range.min, range.max,
                                           TS.width, TS.height, TS.pad));
      path.setAttribute("stroke", series[i].color);
      path.setAttribute("class", "series-line");
      path.setAttribute("data-model", series[i].id);
      svg.appendChild(path);
    }
    crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line");
    crosshair.setAttribute("class", "crosshair");
    crosshair.setAttribute("visibility", "hidden");
    crosshair.setAttribute("y1", String(TS.pad.top));
    crosshair.setAttribute("y2", String(TS.height - TS.pad.bottom));
    svg.appendChild(crosshair);
  }

  function renderLeaderboard() {
    var svg = document.getElementById("leaderboard-svg");
    if (!svg) return;
    var rows = payload.leaderboard || [];
    svg.textContent = "";
    if (!rows.length) {
      svg.appendChild(textNode(svg, "No models recorded yet", LB.width / 2, LB.height / 2));
      return;
    }
    var sorted = rows.slice().sort(function (a, b) {
      return (a.sharpe === null ? -Infinity : a.sharpe) - (b.sharpe === null ? -Infinity : b.sharpe);
    });
    var innerW = LB.width - LB.pad.left - LB.pad.right;
    var barH = 24, gap = 8;
    var totalH = Math.max(LB.height, LB.pad.top + LB.pad.bottom + sorted.length * (barH + gap));
    svg.setAttribute("viewBox", "0 0 " + LB.width + " " + totalH);
    var maxX = Math.max.apply(null, sorted.map(function (r) { return r.sharpe === null ? 0 : r.sharpe; }));
    if (!(maxX > 0)) maxX = 1.0;
    for (var i = 0; i < sorted.length; i++) {
      var row = sorted[i];
      var y = LB.pad.top + i * (barH + gap);
      var w = row.sharpe === null ? 0 : Math.max(0, row.sharpe / maxX) * innerW;
      var bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bar.setAttribute("x", String(LB.pad.left));
      bar.setAttribute("y", String(y));
      bar.setAttribute("width", w.toFixed(1));
      bar.setAttribute("height", String(barH));
      bar.setAttribute("rx", "3");
      bar.setAttribute("class", row.champion ? "bar champion-bar" : "bar");
      svg.appendChild(bar);
      var label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "bar-label");
      label.setAttribute("x", String(LB.pad.left - 8));
      label.setAttribute("y", (y + barH / 2 + 4).toFixed(1));
      label.setAttribute("text-anchor", "end");
      label.textContent = row.label;
      svg.appendChild(label);
      if (row.sharpe !== null) {
        var val = document.createElementNS("http://www.w3.org/2000/svg", "text");
        val.setAttribute("class", "bar-value");
        val.setAttribute("x", (LB.pad.left + w + 6).toFixed(1));
        val.setAttribute("y", (y + barH / 2 + 4).toFixed(1));
        val.textContent = row.sharpe.toFixed(3);
        svg.appendChild(val);
        if (row.ci_low !== null && row.ci_high !== null) {
          var xL = LB.pad.left + Math.max(0, row.ci_low / maxX) * innerW;
          var xH = LB.pad.left + Math.max(0, row.ci_high / maxX) * innerW;
          var whisker = document.createElementNS("http://www.w3.org/2000/svg", "line");
          whisker.setAttribute("x1", xL.toFixed(1));
          whisker.setAttribute("x2", xH.toFixed(1));
          whisker.setAttribute("y1", (y + barH / 2).toFixed(1));
          whisker.setAttribute("y2", (y + barH / 2).toFixed(1));
          whisker.setAttribute("class", "ci-whisker");
          svg.appendChild(whisker);
        }
      }
    }
    var hurdle = payload.hurdle_sharpe;
    if (hurdle !== null && hurdle !== undefined) {
      var hx = LB.pad.left + (hurdle / maxX) * innerW;
      var hline = document.createElementNS("http://www.w3.org/2000/svg", "line");
      hline.setAttribute("class", "hurdle-line");
      hline.setAttribute("x1", hx.toFixed(1));
      hline.setAttribute("x2", hx.toFixed(1));
      hline.setAttribute("y1", String(LB.pad.top));
      hline.setAttribute("y2", String(LB.height - LB.pad.bottom));
      svg.appendChild(hline);
      var htext = document.createElementNS("http://www.w3.org/2000/svg", "text");
      htext.setAttribute("class", "hurdle-text");
      htext.setAttribute("x", (hx + 4).toFixed(1));
      htext.setAttribute("y", String(LB.pad.top + 12));
      htext.textContent = "tier-4 hurdle " + hurdle.toFixed(2);
      svg.appendChild(htext);
    }
  }

  function renderSimilarity() {
    var host = document.getElementById("similarity-host");
    if (!host) return;
    var sim = payload.similarity || {labels: [], matrix: []};
    host.textContent = "";
    if (!sim.matrix.length) {
      host.appendChild(emptyNote("Similarity matrix unavailable without local v5.3 assets"));
      return;
    }
    var table = document.createElement("table");
    table.setAttribute("class", "similarity");
    var head = document.createElement("thead");
    var hr = document.createElement("tr");
    hr.appendChild(document.createElement("th"));
    for (var j = 0; j < sim.labels.length; j++) {
      var th = document.createElement("th");
      th.textContent = sim.labels[j];
      hr.appendChild(th);
    }
    head.appendChild(hr);
    table.appendChild(head);
    var body = document.createElement("tbody");
    for (var i = 0; i < sim.matrix.length; i++) {
      var tr = document.createElement("tr");
      var rowLabel = document.createElement("th");
      rowLabel.textContent = sim.labels[i];
      tr.appendChild(rowLabel);
      for (var k = 0; k < sim.matrix[i].length; k++) {
        var td = document.createElement("td");
        var v = sim.matrix[i][k];
        var isNum = v !== null && v !== undefined && isFinite(v);
        td.textContent = isNum ? v.toFixed(3) : "—";
        var alpha = isNum ? (0.05 + 0.85 * Math.abs(v)) : 0.0;
        var color = isNum ? (v < 0 ? "248, 81, 73" : "88, 166, 255") : "110, 118, 129";
        td.style.backgroundColor = "rgba(" + color + ", " + alpha.toFixed(2) + ")";
        if (i === 0 || k === 0) td.setAttribute("class", "highlight");
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    table.appendChild(body);
    host.appendChild(table);
  }

  function renderDrawdown() {
    var svg = document.getElementById("drawdown-svg");
    if (!svg) return;
    svg.textContent = "";
    var payout = metrics.payout || {};
    var ids = Object.keys(payout).sort();
    if (!eras.length || !ids.length) {
      svg.appendChild(textNode(svg, "Timeseries data unavailable without local v5.3 assets",
                               DD.width / 2, DD.height / 2));
      return;
    }
    var paths = [];
    for (var i = 0; i < ids.length; i++) {
      var standard = payout[ids[i]].standard || [];
      paths.push(drawdownSeries(cumulativeSeries(standard, true)));
    }
    var range = globalYRange(paths);
    if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1.0; range.max += 1.0; }
    for (var k = 0; k < paths.length; k++) {
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", svgAreaPath(paths[k], range.min, range.max, 0.0,
                                         DD.width, DD.height, DD.pad));
      path.setAttribute("class", "drawdown-area");
      path.setAttribute("fill", "rgba(248, 81, 73, 0.15)");
      path.setAttribute("stroke", COLORS[k % COLORS.length]);
      svg.appendChild(path);
    }
  }

  function textNode(svg, text, x, y) {
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", String(x));
    t.setAttribute("y", String(y));
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("class", "empty-note");
    t.textContent = text;
    return t;
  }

  function emptyNote(text) {
    var p = document.createElement("p");
    p.setAttribute("class", "empty-note");
    p.textContent = text;
    return p;
  }

  function eraIndexFromX(x) {
    if (!eras.length) return -1;
    var innerW = TS.width - TS.pad.left - TS.pad.right;
    var t = Math.round((x - TS.pad.left) / (innerW / Math.max(1, eras.length - 1)));
    return Math.min(Math.max(t, 0), eras.length - 1);
  }

  function attachTooltip() {
    var svg = document.getElementById("timeseries-svg");
    var tip = document.getElementById("timeseries-tooltip");
    if (!svg || !tip || !eras.length) return;
    svg.addEventListener("mousemove", function (ev) {
      var rect = svg.getBoundingClientRect();
      var x = (ev.clientX - rect.left) * (TS.width / rect.width);
      var t = eraIndexFromX(x);
      if (t < 0 || !crosshair) { tip.hidden = true; return; }
      crosshair.setAttribute("visibility", "visible");
      var innerW = TS.width - TS.pad.left - TS.pad.right;
      var cx = TS.pad.left + (t / Math.max(1, eras.length - 1)) * innerW;
      crosshair.setAttribute("x1", cx.toFixed(1));
      crosshair.setAttribute("x2", cx.toFixed(1));
      var series = activeSeries();
      var cfg = METRIC_CONFIG[currentMetric][currentView];
      var lines = ["<b>Era " + esc(eras[t]) + "</b>"];
      for (var i = 0; i < series.length; i++) {
        lines.push('<span style="color:' + series[i].color + '">\u25CF</span> ' +
                   esc(series[i].label) + ": " + fmt(series[i].values[t], cfg.percent));
      }
      tip.innerHTML = lines.join("<br>");
      tip.hidden = false;
      tip.style.left = Math.min(x + 14, TS.width - 240) + "px";
      tip.style.top = "8px";
    });
    svg.addEventListener("mouseleave", function () {
      if (crosshair) crosshair.setAttribute("visibility", "hidden");
      tip.hidden = true;
    });
  }

  var select = document.getElementById("metric-select");
  var stdBtn = document.getElementById("view-standard");
  var cumBtn = document.getElementById("view-cumulative");
  if (select) select.addEventListener("change", function () {
    currentMetric = select.value;
    renderTimeseries();
  });
  if (stdBtn) stdBtn.addEventListener("click", function () { currentView = "standard"; renderTimeseries(); });
  if (cumBtn) cumBtn.addEventListener("click", function () { currentView = "cumulative"; renderTimeseries(); });

  attachTooltip();
  renderTimeseries();
  renderLeaderboard();
  renderSimilarity();
  renderDrawdown();
})();
