(function () {
  "use strict";

  var root = document.getElementById("dashboard-root");
  var dataNode = document.getElementById("dashboard-data");
  if (!root || !dataNode) return;

  var payload;
  try {
    payload = JSON.parse(dataNode.textContent);
  } catch (err) {
    return;
  }

  var metricSpecs = payload.metric_specs || [];
  var rowFields = payload.row_fields || [];
  var rows = (payload.rows || []).map(function (record, rowIndex) {
    if (!Array.isArray(record)) return record;
    var row = {};
    rowFields.forEach(function (field, index) { row[field] = record[index]; });
    row.model_id = (payload.model_ids || [])[rowIndex];
    return row;
  });
  var state = {
    metric: (payload.meta || {}).default_rank_metric || "mmc",
    cohort: "all",
    search: "",
    shortcut: null,
    selected: null
  };
  var cohortColors = {trained: "#4f46e5", heuristic: "#64748b", benchmark: "#9aa5b1", full: "#a96500"};
  var chartColors = ["#f5b921", "#7edaa3", "#ff7a7a", "#9d8cff", "#53b5d8", "#ef9d5a"];
  var metricConfig = {
    payout: {standard: {label: "Per-Era Net Return", percent: true}, cumulative: {label: "Cumulative Wealth", percent: false}},
    corr20: {standard: {label: "Per-Era CORR (20D)", percent: false}, cumulative: {label: "Cumulative CORR (20D)", percent: false}},
    mmc20: {standard: {label: "Per-Era MMC (20D)", percent: false}, cumulative: {label: "Cumulative MMC (20D)", percent: false}},
    corr60: {standard: {label: "Per-Era CORR (60D)", percent: false}, cumulative: {label: "Cumulative CORR (60D)", percent: false}},
    mmc60: {standard: {label: "Per-Era MMC (60D)", percent: false}, cumulative: {label: "Cumulative MMC (60D)", percent: false}},
    bmc: {standard: {label: "Per-Era BMC", percent: false}, cumulative: {label: "Cumulative BMC", percent: false}},
    cwmm: {standard: {label: "Per-Era CWMM", percent: false}, cumulative: {label: "Cumulative CWMM", percent: false}}
  };
  var timeseriesPadding = {top: 24, right: 24, bottom: 40, left: 56};
  var drawdownPadding = {top: 24, right: 24, bottom: 40, left: 56};
  var currentMetric = "payout";
  var currentView = "standard";
  var eras = payload.eras || [];
  var metrics = payload.metrics || {};
  var crosshair = null;
  var previousFocus = null;

  function byId(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
  }

  function finite(value) { return value !== null && value !== undefined && isFinite(Number(value)); }

  function decodeSeries(encoded, scale) {
    if (Array.isArray(encoded)) return encoded.map(function (value) { return finite(value) ? Number(value) / scale : null; });
    if (typeof encoded !== "string") return [];
    try {
      var binary = atob(encoded), result = [];
      for (var offset = 0; offset + 3 < binary.length; offset += 4) {
        var raw = binary.charCodeAt(offset) + binary.charCodeAt(offset + 1) * 256 + binary.charCodeAt(offset + 2) * 65536 + binary.charCodeAt(offset + 3) * 16777216;
        if (raw >= 2147483648) raw -= 4294967296;
        result.push(raw === -2147483648 ? null : raw / scale);
      }
      return result;
    } catch (err) {
      return [];
    }
  }

  function fmt(value, percent) {
    if (!finite(value)) return "&#8212;";
    return percent ? (Number(value) * 100).toFixed(2) + "%" : Number(value).toFixed(4);
  }

  function safeText(value) {
    return value === null || value === undefined || value === "" ? "&#8212;" : esc(value);
  }

  function tooltipLine(color, label, value) {
    return "<span class=\"tooltip-series\"><i style=\"--tooltip-color:" + color + "\"></i><span>" + esc(label) + ": " + value + "</span></span>";
  }

  function compactModelLabel(value) {
    var raw = String(value === null || value === undefined ? "" : value);
    var row = rows.filter(function (item) { return item.model_id === raw; })[0];
    if (row && row.cohort === "trained" && row.run_name) return row.run_name + " / " + raw.slice(0, 8);
    return raw.length > 24 ? raw.slice(0, 12) + "..." + raw.slice(-4) : raw;
  }

  function chartContainer(element) {
    return element.closest(".chart-box") || element.parentElement;
  }

  function showChartTooltip(tooltip, event, content, element) {
    var container = chartContainer(element);
    if (!tooltip || !container || !content) return;
    tooltip.innerHTML = content;
    tooltip.hidden = false;
    var rect = container.getBoundingClientRect();
    var x = event.clientX - rect.left + 14;
    var y = event.clientY - rect.top + 14;
    var maxX = Math.max(8, container.clientWidth - tooltip.offsetWidth - 8);
    var maxY = Math.max(8, container.clientHeight - tooltip.offsetHeight - 8);
    tooltip.style.left = Math.max(8, Math.min(x, maxX)) + "px";
    tooltip.style.top = Math.max(8, Math.min(y, maxY)) + "px";
  }

  function hideChartTooltip(tooltip) {
    if (tooltip) tooltip.hidden = true;
  }

  function bindPointerTooltip(element, tooltip, contentFactory) {
    if (!element || !tooltip || element.getAttribute("data-tooltip-bound")) return;
    element.setAttribute("data-tooltip-bound", "true");
    element.style.touchAction = "pan-y";
    var update = function (event) {
      showChartTooltip(tooltip, event, contentFactory(event), element);
    };
    element.addEventListener("pointermove", update);
    element.addEventListener("pointerdown", update);
    element.addEventListener("pointerleave", function () { hideChartTooltip(tooltip); });
    element.addEventListener("pointercancel", function () { hideChartTooltip(tooltip); });
  }

  function pointIndexFromEvent(event, svg, length, padding) {
    if (!length) return -1;
    var rect = svg.getBoundingClientRect();
    var viewX = (event.clientX - rect.left) * (800 / rect.width);
    var innerWidth = 800 - padding.left - padding.right;
    var index = Math.round((viewX - padding.left) / (innerWidth / Math.max(1, length - 1)));
    return Math.max(0, Math.min(index, length - 1));
  }

  function specFor(metric) {
    for (var i = 0; i < metricSpecs.length; i++) if (metricSpecs[i].name === metric) return metricSpecs[i];
    return {name: metric, label: metric, higher_is_better: true, direction: "higher"};
  }

  function metricValue(row, metric) {
    var index = (payload.metric_fields || []).indexOf(metric);
    var value = index < 0 || !row.values ? null : row.values[index];
    return value === null || value === undefined ? null : Number(value) / (payload.row_value_scale || 1);
  }

  function ciValue(row, index) {
    var value = row.ci && row.ci[index];
    if (value === null || value === undefined) return null;
    return index < 2 ? Number(value) / (payload.row_value_scale || 1) : value;
  }

  function rankValue(row, metric) {
    var index = (payload.metric_fields || []).indexOf(metric);
    var rowIndex = rows.indexOf(row), ranks = (payload.rank_values || [])[rowIndex];
    return index < 0 || !ranks ? null : ranks[index];
  }

  function compareRows(a, b, metric) {
    var spec = specFor(metric), ar = rankValue(a, metric), br = rankValue(b, metric);
    if (finite(ar) && finite(br)) return Number(ar) === Number(br) ? compareIds(a.model_id, b.model_id) : (Number(ar) < Number(br) ? -1 : 1);
    if (finite(ar)) return -1;
    if (finite(br)) return 1;
    var av = metricValue(a, metric), bv = metricValue(b, metric);
    if (!finite(av) && !finite(bv)) return compareIds(a.model_id, b.model_id);
    if (!finite(av)) return 1;
    if (!finite(bv)) return -1;
    var delta = Number(av) - Number(bv);
    if (!spec.higher_is_better) delta = -delta;
    return delta === 0 ? compareIds(a.model_id, b.model_id) : (delta > 0 ? -1 : 1);
  }

  function compareIds(a, b) { var left = String(a), right = String(b); return left === right ? 0 : (left < right ? -1 : 1); }

  function compareLowerMetric(a, b, metric) {
    var av = metricValue(a, metric), bv = metricValue(b, metric);
    if (finite(av) && finite(bv)) return Number(av) === Number(bv) ? 0 : (Number(av) < Number(bv) ? -1 : 1);
    return finite(av) ? -1 : (finite(bv) ? 1 : 0);
  }

  function strictlyBeats(a, b, metric) {
    return b.cohort === "benchmark" && ((payload.strict_beats || {})[metric] || []).indexOf(a.model_id) !== -1;
  }

  function sortedRows(input, metric) {
    return input.slice().sort(function (a, b) {
      if (a.cohort === "full" && b.cohort !== "full") return 1;
      if (b.cohort === "full" && a.cohort !== "full") return -1;
      return compareRows(a, b, metric || state.metric);
    });
  }

  function searchable(row) {
    if (!state.search) return true;
    return [row.model_id, row.run_name, row.family, row.backend, row.preset, row.feature_set]
      .join(" ").toLowerCase().indexOf(state.search.toLowerCase()) !== -1;
  }

  function baseVisibleRows() {
    return rows.filter(function (row) { return (state.cohort === "all" || row.cohort === state.cohort) && searchable(row); });
  }

  function bestRow(input, cohort, metric) {
    return sortedRows(input.filter(function (row) { return row.cohort === cohort && finite(metricValue(row, metric)); }), metric)[0] || null;
  }

  function filteredRows() {
    var visible = baseVisibleRows();
    if (state.shortcut === "top10") return sortedRows(visible.filter(function (row) { return row.cohort !== "full"; })).slice(0, 10);
    if (state.shortcut === "beats") {
      var benchmark = bestRow(rows, "benchmark", state.metric);
      if (!benchmark) return [];
      return sortedRows(visible.filter(function (row) { return row.cohort === "trained" && strictlyBeats(row, benchmark, state.metric); }));
    }
    if (state.shortcut === "robust") {
      return visible.filter(function (row) { return row.cohort !== "full"; }).sort(function (a, b) {
        var aScore = Object.keys(a.robustness || {}).filter(function (key) { return a.robustness[key] === true; }).length;
        var bScore = Object.keys(b.robustness || {}).filter(function (key) { return b.robustness[key] === true; }).length;
        return bScore - aScore || compareLowerMetric(a, b, "std_corr") || compareLowerMetric(a, b, "max_drawdown") || compareIds(a.model_id, b.model_id);
      });
    }
    return sortedRows(visible);
  }

  function valueText(row, metric) {
    return fmt(metricValue(row, metric), metric === "cagr_1y" || metric === "max_drawdown");
  }

  function medalRank(row) {
    if (row.cohort !== "trained") return null;
    var candidates = sortedRows(rows.filter(function (item) {
      return item.cohort === "trained" && finite(metricValue(item, state.metric));
    }));
    var index = candidates.indexOf(row);
    return index >= 0 && index < 3 ? index + 1 : null;
  }

  function rankCell(row) {
    var rank = rankValue(row, state.metric);
    var rankText = rank === null || rank === undefined ? "&#8212;" : "#" + rank;
    var medal = medalRank(row);
    if (!medal) return rankText;
    var medalName = medal === 1 ? "gold" : medal === 2 ? "silver" : "bronze";
    return "<span class=\"medal medal-" + medalName + "\" title=\"Top trained model by selected metric\">" + medal + "</span><small class=\"medal-rank\">" + rankText + "</small>";
  }

  function ciText(row) {
    var low = ciValue(row, 0), high = ciValue(row, 1);
    if (!finite(low) || !finite(high)) return valueText(row, "corr_sharpe_ac");
    return valueText(row, "corr_sharpe_ac") + " [" + fmt(low, false) + "&#8211;" + fmt(high, false) + "]";
  }

  function marker(row) {
    if (row.cohort === "heuristic") return "<span class=\"type-marker heuristic\">&#9670; heuristic</span>";
    if (row.cohort === "benchmark") return "<span class=\"type-marker benchmark\">&#8212; benchmark</span>";
    if (row.cohort === "full") return "<span class=\"type-marker full\">&#9671; full lineage</span>";
    return "<span class=\"type-marker trained\">&#9679; trained</span>";
  }

  function renderMasterLeaderboard() {
    var host = byId("leaderboard-table"), empty = byId("leaderboard-empty"), count = byId("row-count");
    if (!host || !empty) return;
    var visible = filteredRows();
    if (count) count.textContent = visible.length + " of " + rows.length + " rows";
    empty.hidden = visible.length !== 0;
    if (!visible.length) { host.innerHTML = ""; return; }
    var spec = specFor(state.metric), showRankedMetric = ["cagr_1y", "mmc", "corr", "corr_sharpe_ac", "max_drawdown"].indexOf(state.metric) === -1;
    var rankedLabel = state.metric === "mmc" ? "RANKED: MMC (CORE)" : "RANKED: " + spec.label;
    var html = "<table class=\"tournament-table\"><thead><tr>" +
      "<th>Rank</th><th>Model</th><th>Type</th><th" + (state.metric === "cagr_1y" ? " class=\"selected-head\"" : "") + ">" + (state.metric === "cagr_1y" ? esc(rankedLabel) : "CAGR 1Y") + "</th><th" + (state.metric === "mmc" ? " class=\"selected-head\"" : "") + ">" + (state.metric === "mmc" ? esc(rankedLabel) : "MMC (CORE)") + "</th><th" + (state.metric === "corr" ? " class=\"selected-head\"" : "") + ">" + (state.metric === "corr" ? esc(rankedLabel) : "CORR") + "</th><th" + (state.metric === "corr_sharpe_ac" ? " class=\"selected-head\"" : "") + ">" + (state.metric === "corr_sharpe_ac" ? esc(rankedLabel) : "CORR Sharpe + CI") + "</th><th" + (state.metric === "max_drawdown" ? " class=\"selected-head\"" : "") + ">" + (state.metric === "max_drawdown" ? esc(rankedLabel) : "Max DD") + "</th>" +
      (showRankedMetric ? "<th class=\"selected-head\">" + esc(rankedLabel) + "</th>" : "") +
      "<th>G:P</th><th>Eras</th></tr></thead><tbody>";
    visible.forEach(function (row) {
      var rowClass = row.champion ? " champion-row" : "";
      if (row.cohort === "full") rowClass += " full-row";
      if (row.cohort === "benchmark") rowClass += " benchmark-row";
      if (row.cohort === "heuristic") rowClass += " heuristic-row";
      html += "<tr class=\"model-row" + rowClass + "\" data-model-id=\"" + esc(row.model_id) + "\" tabindex=\"0\">";
      html += "<td class=\"rank-cell mono\">" + rankCell(row) + "</td><td class=\"model-cell\"><strong>" + esc(row.run_name || row.model_id) + "</strong>";
      html += row.champion ? " <span class=\"champion-mark\">CHAMPION</span>" : "";
      html += "<small>" + esc(row.model_id) + "</small></td><td>" + marker(row) + "</td>";
      html += "<td class=\"num\">" + valueText(row, "cagr_1y") + "</td><td class=\"num\">" + valueText(row, "mmc") + "</td>";
      html += "<td class=\"num\">" + valueText(row, "corr") + "</td><td class=\"num\">" + ciText(row) + "</td>";
      html += "<td class=\"num\">" + valueText(row, "max_drawdown") + "</td>";
      if (showRankedMetric) html += "<td class=\"num selected-value\">" + valueText(row, state.metric) + "</td>";
      html += "<td class=\"num\">" + valueText(row, "gain_to_pain_ratio") + "</td><td class=\"num\">" + (finite(ciValue(row, 2)) ? ciValue(row, 2) : "&#8212;") + "</td></tr>";
    });
    host.innerHTML = html + "</tbody></table>";
    host.querySelectorAll(".model-row").forEach(function (node) {
      node.addEventListener("click", function () { openDossier(node.getAttribute("data-model-id")); });
      node.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openDossier(node.getAttribute("data-model-id")); } });
    });
  }

  function edgeText(trained, baseline, metric) {
    if (!trained) return "&#8212;";
    if (!baseline) return "&#8212;";
    var spec = specFor(metric), value = Number(metricValue(trained, metric)), base = Number(metricValue(baseline, metric));
    var edge = spec.higher_is_better ? value - base : base - value;
    var text = (edge >= 0 ? "+" : "") + fmt(edge, false);
    if (base !== 0) text += " (" + (edge / Math.abs(base) * 100 >= 0 ? "+" : "") + (edge / Math.abs(base) * 100).toFixed(1) + "%)";
    return text;
  }

  function edgeClass(trained, baseline, metric) {
    if (!trained || !baseline) return "missing";
    var spec = specFor(metric), value = Number(metricValue(trained, metric)), base = Number(metricValue(baseline, metric));
    var edge = spec.higher_is_better ? value - base : base - value;
    return edge < 0 ? "negative" : "positive";
  }

  function renderAdvantage() {
    var host = byId("advantage-strip"); if (!host) return;
    var metric = state.metric, spec = specFor(metric), trained = bestRow(rows, "trained", metric), heuristic = bestRow(rows, "heuristic", metric), benchmark = bestRow(rows, "benchmark", metric);
    host.innerHTML = "<div class=\"advantage-title\"><span>ML ADVANTAGE</span><small>Best trained vs simple alternatives &middot; " + esc(spec.label) + "</small></div>" +
      "<div class=\"advantage-track\"><div><span>Trained</span><strong>" + (trained ? valueText(trained, metric) : "&#8212;") + "</strong><small>" + esc(trained ? trained.run_name : "Unavailable") + "</small></div>" +
      "<div><span>Heuristic</span><strong>" + (heuristic ? valueText(heuristic, metric) : "&#8212;") + "</strong><small>" + esc(heuristic ? heuristic.run_name : "Unavailable") + "</small></div>" +
      "<div><span>Benchmark</span><strong>" + (benchmark ? valueText(benchmark, metric) : "&#8212;") + "</strong><small>" + esc(benchmark ? benchmark.run_name : "Unavailable") + "</small></div></div>" +
      "<div class=\"advantage-edges\"><span>Edge vs heuristic <b class=\"" + edgeClass(trained, heuristic, metric) + "\">" + edgeText(trained, heuristic, metric) + "</b></span><span>Edge vs benchmark <b class=\"" + edgeClass(trained, benchmark, metric) + "\">" + edgeText(trained, benchmark, metric) + "</b></span></div>";
  }

  function textNode(svg, text, x, y) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", "text"); node.setAttribute("x", x); node.setAttribute("y", y); node.setAttribute("class", "empty-note"); node.setAttribute("text-anchor", "middle"); node.textContent = text; return node;
  }

  function renderLandscape() {
    var svg = byId("landscape-svg"); if (!svg) return;
    var points = payload.landscape || rows.filter(function (row) {
      return row.cohort !== "full" && finite(metricValue(row, "corr")) && finite(metricValue(row, "mmc"));
    }).map(function (row) {
      return {model_id: row.model_id, cohort: row.cohort, corr: metricValue(row, "corr"), mmc: metricValue(row, "mmc"), champion: row.champion};
    });
    svg.textContent = "";
    if (!points.length) { svg.appendChild(textNode(svg, "Landscape unavailable", 380, 180)); return; }
    var width = 760, height = 360, pad = {left: 70, right: 24, top: 20, bottom: 48};
    var xs = points.map(function (point) { return Number(point.corr); }), ys = points.map(function (point) { return Number(point.mmc); });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs), ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    if (xmin === xmax) { xmin -= 1; xmax += 1; } else { var xp = (xmax - xmin) * .08; xmin -= xp; xmax += xp; }
    if (ymin === ymax) { ymin -= 1; ymax += 1; } else { var yp = (ymax - ymin) * .08; ymin -= yp; ymax += yp; }
    function x(value) { return pad.left + (value - xmin) / (xmax - xmin) * (width - pad.left - pad.right); }
    function y(value) { return height - pad.bottom - (value - ymin) / (ymax - ymin) * (height - pad.top - pad.bottom); }
    var axis = document.createElementNS("http://www.w3.org/2000/svg", "path"); axis.setAttribute("d", "M " + pad.left + " " + pad.top + " V " + (height - pad.bottom) + " H " + (width - pad.right)); axis.setAttribute("class", "chart-axis"); svg.appendChild(axis);
    var guideVertical = document.createElementNS("http://www.w3.org/2000/svg", "line"); guideVertical.setAttribute("class", "hover-guide vertical"); guideVertical.setAttribute("visibility", "hidden"); guideVertical.setAttribute("y1", pad.top); guideVertical.setAttribute("y2", height - pad.bottom); svg.appendChild(guideVertical);
    var guideHorizontal = document.createElementNS("http://www.w3.org/2000/svg", "line"); guideHorizontal.setAttribute("class", "hover-guide horizontal"); guideHorizontal.setAttribute("visibility", "hidden"); guideHorizontal.setAttribute("x1", pad.left); guideHorizontal.setAttribute("x2", width - pad.right); svg.appendChild(guideHorizontal);
    var landscapeTip = byId("landscape-tooltip");
    points.forEach(function (point) {
      var px = x(Number(point.corr)), py = y(Number(point.mmc)), shape, color = cohortColors[point.cohort] || cohortColors.benchmark;
      if (point.cohort === "heuristic") { shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon"); shape.setAttribute("points", px + "," + (py - 6) + " " + (px + 6) + "," + py + " " + px + "," + (py + 6) + " " + (px - 6) + "," + py); }
      else if (point.cohort === "benchmark") { shape = document.createElementNS("http://www.w3.org/2000/svg", "rect"); shape.setAttribute("x", px - 5); shape.setAttribute("y", py - 5); shape.setAttribute("width", 10); shape.setAttribute("height", 10); }
      else { shape = document.createElementNS("http://www.w3.org/2000/svg", "circle"); shape.setAttribute("cx", px); shape.setAttribute("cy", py); shape.setAttribute("r", point.champion ? 7 : 5); }
      shape.setAttribute("fill", color); shape.setAttribute("class", point.champion ? "landscape-point champion-point" : "landscape-point"); shape.setAttribute("data-model-id", point.model_id); shape.addEventListener("click", function () { openDossier(point.model_id); });
      shape.setAttribute("tabindex", "0"); shape.setAttribute("role", "button"); shape.setAttribute("aria-label", compactModelLabel(point.model_id) + ", " + point.cohort + ", CORR " + fmt(point.corr, false) + ", MMC " + fmt(point.mmc, false)); shape.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openDossier(point.model_id); } });
      shape.addEventListener("pointerenter", function () { guideVertical.setAttribute("x1", px); guideVertical.setAttribute("x2", px); guideVertical.setAttribute("visibility", "visible"); guideHorizontal.setAttribute("y1", py); guideHorizontal.setAttribute("y2", py); guideHorizontal.setAttribute("visibility", "visible"); shape.classList.add("is-hovered"); });
      shape.addEventListener("pointerleave", function () { guideVertical.setAttribute("visibility", "hidden"); guideHorizontal.setAttribute("visibility", "hidden"); shape.classList.remove("is-hovered"); });
      bindPointerTooltip(shape, landscapeTip, function () {
        return tooltipLine(cohortColors[point.cohort] || cohortColors.benchmark, compactModelLabel(point.model_id), esc(point.cohort)) + "<br>CORR " + fmt(point.corr, false) + "<br>MMC " + fmt(point.mmc, false);
      });
      var title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = point.model_id + " / CORR " + fmt(point.corr, false) + " / MMC " + fmt(point.mmc, false); shape.appendChild(title); svg.appendChild(shape);
    });
    [xmin, (xmin + xmax) / 2, xmax].forEach(function (value) { var tick = textNode(svg, fmt(value, false), x(value), height - 28); tick.setAttribute("class", "axis-tick"); svg.appendChild(tick); });
    [ymax, (ymax + ymin) / 2, ymin].forEach(function (value) { var tick = textNode(svg, fmt(value, false), pad.left - 10, y(value) + 4); tick.setAttribute("class", "axis-tick"); tick.setAttribute("text-anchor", "end"); svg.appendChild(tick); });
    var landscapeLegend = byId("landscape-legend"); if (landscapeLegend) landscapeLegend.innerHTML = [{name: "Trained", color: cohortColors.trained}, {name: "Heuristic", color: cohortColors.heuristic}, {name: "Benchmark", color: cohortColors.benchmark}].map(function (item) { return "<span class=\"legend-item\"><i style=\"--legend-color:" + item.color + "\"></i><b>" + item.name + "</b></span>"; }).join("");
    svg.appendChild(textNode(svg, "CORR", width / 2, height - 10)); var ylabel = textNode(svg, "MMC", 16, height / 2); ylabel.setAttribute("transform", "rotate(-90 16 " + height / 2 + ")"); svg.appendChild(ylabel);
  }

  function renderProfile(row) {
    var svg = byId("profile-svg"), label = byId("profile-label"); if (!svg || !label) return;
    svg.textContent = "";
    if (!row) { label.textContent = "Select a row"; svg.appendChild(textNode(svg, "Select a model", 380, 180)); return; }
    label.textContent = row.run_name || row.model_id;
    var available = metricSpecs.filter(function (spec) { return finite(metricValue(row, spec.name)); });
    if (!available.length) { svg.appendChild(textNode(svg, "Metric profile unavailable", 380, 180)); return; }
    var max = Math.max.apply(null, available.map(function (spec) { return Math.abs(Number(metricValue(row, spec.name))); }).concat([.000001]));
    var zero = 210, barWidth = 430, top = 20, rowHeight = 22;
    var profileTip = byId("profile-tooltip");
    available.forEach(function (spec, index) {
      var value = Number(metricValue(row, spec.name)), y = top + index * rowHeight, width = Math.abs(value) / max * barWidth, x = value >= 0 ? zero : zero - width;
      var bar = document.createElementNS("http://www.w3.org/2000/svg", "rect"); bar.setAttribute("x", x); bar.setAttribute("y", y); bar.setAttribute("width", width); bar.setAttribute("height", 13); bar.setAttribute("class", value >= 0 ? "profile-bar positive" : "profile-bar negative"); svg.appendChild(bar);
      bindPointerTooltip(bar, profileTip, function () {
        return tooltipLine(value >= 0 ? "var(--gold)" : "var(--coral)", spec.label, "Value " + fmt(value, spec.name === "cagr_1y" || spec.name === "max_drawdown")) + "<br>" + (spec.higher_is_better ? "Higher is better" : "Lower is better");
      });
      var name = textNode(svg, spec.label, 198, y + 11); name.setAttribute("text-anchor", "end"); svg.appendChild(name); var valueTextNode = textNode(svg, fmt(value, spec.name === "cagr_1y" || spec.name === "max_drawdown"), Math.min(720, zero + width + 8), y + 11); valueTextNode.setAttribute("class", "profile-value"); svg.appendChild(valueTextNode);
    });
    var baseline = document.createElementNS("http://www.w3.org/2000/svg", "line"); baseline.setAttribute("x1", zero); baseline.setAttribute("x2", zero); baseline.setAttribute("y1", 12); baseline.setAttribute("y2", top + available.length * rowHeight); baseline.setAttribute("class", "profile-baseline"); svg.appendChild(baseline);
  }

  function renderDossier(row) {
    var host = byId("model-dossier"), rowIndex = rows.indexOf(row), packed = (payload.details || [])[rowIndex] || [[], [], null, null];
    var scorecardValues = {}, provenanceValues = {};
    (packed[0] || []).forEach(function (pair) { scorecardValues[pair[0]] = pair[1]; });
    (packed[1] || []).forEach(function (pair) { provenanceValues[pair[0]] = pair[1]; });
    var detail = {scorecard_values: scorecardValues, provenance_values: provenanceValues, evidence_ref: packed[2], reason: packed[3]}; if (!host) return;
    var ranks = metricSpecs.filter(function (spec) { return rankValue(row, spec.name) !== null && rankValue(row, spec.name) !== undefined; }).map(function (spec) { return "<span>" + esc(spec.label) + " <b>#" + rankValue(row, spec.name) + "</b></span>"; }).join("");
    var scorecard = {};
    (payload.metric_fields || []).forEach(function (key) { scorecard[key] = metricValue(row, key); });
    scorecard.corr_sharpe_ac_ci_low = ciValue(row, 0);
    scorecard.corr_sharpe_ac_ci_high = ciValue(row, 1);
    scorecard.corr_sharpe_ac_n_eras = ciValue(row, 2);
    (payload.scorecard_fields || []).forEach(function (key, index) { scorecard[key] = detail.scorecard_values ? detail.scorecard_values[index] : null; });
    var keys = Object.keys(scorecard).filter(function (key) { return scorecard[key] !== null && scorecard[key] !== undefined; }).sort();
    var scorecardHtml = keys.length ? keys.map(function (key) { var value = scorecard[key], display = key.indexOf("cagr") !== -1 || key.indexOf("drawdown") !== -1 ? fmt(value, true) : (finite(value) ? fmt(value, false) : esc(value)); return "<tr><th>" + esc(key.replace(/_/g, " ")) + "</th><td class=\"num\">" + display + "</td></tr>"; }).join("") : "<tr><td colspan=\"2\" class=\"missing\">&#8212;</td></tr>";
    var provenance = {backend: row.backend, preset: row.preset, feature_set: row.feature_set, feature_subset: row.feature_subset, targets: row.targets, neutralization_proportion: row.neutralization_proportion, oof_device: row.oof_device};
    (payload.provenance_fields || []).forEach(function (key, index) { provenance[key] = detail.provenance_values[index]; });
    var provenanceKeys = Object.keys(provenance).sort(), provenanceHtml = provenanceKeys.map(function (key) { var value = provenance[key]; if (value === null || value === undefined) return ""; value = Array.isArray(value) ? value.join(", ") : value; return "<div><span>" + esc(key.replace(/_/g, " ")) + "</span><strong>" + esc(value) + "</strong></div>"; }).join("");
    if (!provenanceHtml) provenanceHtml = "<p class=\"missing\">Provenance unavailable</p>";
    var evidenceRef = detail.evidence_ref;
    if (!evidenceRef && row.cohort === "trained") evidenceRef = "registry/" + row.model_id + "/run.json";
    var ref = evidenceRef ? "<a href=\"" + esc(evidenceRef) + "\">Open immutable evidence</a>" : "<span class=\"missing\">Evidence link unavailable</span>";
    var meta = payload.meta || {}, window = meta.evaluation_window || {};
    host.innerHTML = "<div class=\"drawer-kicker\">" + esc(row.cohort) + " &middot; " + esc(row.status || "RESEARCH") + "</div><h2 id=\"drawer-title\">" + esc(row.run_name || row.model_id) + "</h2><p class=\"drawer-id mono\">" + esc(row.model_id) + "</p>" +
      "<div class=\"drawer-meta\"><span>Suite " + safeText(meta.suite_version) + "</span><span>Window " + safeText(window.start) + " &rarr; " + safeText(window.end) + "</span><span>Timestamp " + safeText(provenance.timestamp) + "</span></div>" +
      "<h3>Rank across metrics</h3><div class=\"rank-list\">" + (ranks || "<span>&#8212;</span>") + "</div><h3>Full scorecard evidence</h3><table class=\"dossier-table\"><tbody>" + scorecardHtml + "</tbody></table>" +
      "<h3>Provenance / lineage</h3><div class=\"provenance-list\">" + provenanceHtml + "</div><p class=\"evidence-link\">" + ref + "</p><p class=\"drawer-note\">Missing values remain unavailable; this dossier contains offline evaluation evidence only.</p>";
  }

  function openDossier(modelId) {
    var row = rows.filter(function (item) { return item.model_id === modelId; })[0]; if (!row) return;
    previousFocus = document.activeElement;
    state.selected = modelId; renderDossier(row); renderProfile(row); var drawer = byId("model-drawer"); if (drawer) { drawer.hidden = false; document.body.classList.add("drawer-open"); }
    var closeButton = document.querySelector(".drawer-close"); if (closeButton) closeButton.focus();
  }

  function closeDossier() { var drawer = byId("model-drawer"); if (!drawer || drawer.hidden) return; drawer.hidden = true; document.body.classList.remove("drawer-open"); if (previousFocus && previousFocus.isConnected && typeof previousFocus.focus === "function") previousFocus.focus(); previousFocus = null; }

  function trapDrawerFocus(event) {
    var drawer = byId("model-drawer"); if (!drawer || drawer.hidden || event.key !== "Tab") return;
    var focusable = drawer.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])"); if (!focusable.length) return;
    var first = focusable[0], last = focusable[focusable.length - 1]; if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }

  function populateToolbar() {
    var select = byId("rank-by");
    if (select) {
      select.innerHTML = metricSpecs.map(function (spec) { return "<option value=\"" + esc(spec.name) + "\">" + esc(spec.label) + "</option>"; }).join(""); select.value = state.metric;
      select.addEventListener("change", function () { state.metric = select.value; state.shortcut = null; renderAll(); });
    }
    var search = byId("model-search"); if (search) search.addEventListener("input", function () { state.search = search.value; renderMasterLeaderboard(); });
    document.querySelectorAll(".cohort-button").forEach(function (button) { button.addEventListener("click", function () { state.cohort = button.getAttribute("data-cohort"); state.shortcut = null; document.querySelectorAll(".cohort-button").forEach(function (item) { var active = item === button; item.classList.toggle("active", active); item.setAttribute("aria-pressed", String(active)); }); document.querySelectorAll(".shortcut-button").forEach(function (item) { item.classList.remove("active"); item.setAttribute("aria-pressed", "false"); }); renderAll(); }); });
    document.querySelectorAll(".shortcut-button:not([disabled])").forEach(function (button) { button.addEventListener("click", function () { var value = button.getAttribute("data-shortcut"); state.shortcut = state.shortcut === value ? null : value; document.querySelectorAll(".shortcut-button").forEach(function (item) { var active = item === button && state.shortcut !== null; item.classList.toggle("active", active); item.setAttribute("aria-pressed", String(active)); }); renderAll(); }); });
    document.querySelectorAll("[data-drawer-close]").forEach(function (button) { button.addEventListener("click", closeDossier); }); document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeDossier(); trapDrawerFocus(event); });
  }

  function renderAll() {
    var direction = byId("rank-direction"); if (direction) direction.innerHTML = specFor(state.metric).higher_is_better ? "&uarr; higher is better" : "&darr; lower is better";
    renderMasterLeaderboard(); renderAdvantage(); renderLandscape();
    var selected = rows.filter(function (row) { return row.model_id === state.selected; })[0] || filteredRows()[0] || null; renderProfile(selected);
    if (selected && state.selected === selected.model_id && byId("model-drawer") && !byId("model-drawer").hidden) renderDossier(selected);
    renderTimeseries(); renderSimilarity(); renderDrawdown();
  }

  function globalYRange(seriesList) {
    var values = []; seriesList.forEach(function (series) { series.forEach(function (value) { if (finite(value)) values.push(Number(value)); }); });
    return values.length ? {min: Math.min.apply(null, values), max: Math.max.apply(null, values)} : {min: 0, max: 1};
  }

  function dataToSvgPath(values, yMin, yMax, width, height, pad) {
    if (!values || !values.length) return ""; if (Math.abs(yMax - yMin) < 1e-12) { yMin -= 1; yMax += 1; }
    var innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom, denom = Math.max(1, values.length - 1), commands = [], open = false;
    values.forEach(function (value, index) { if (!finite(value)) { open = false; return; } var x = pad.left + index / denom * innerW, y = pad.top + (1 - (Number(value) - yMin) / (yMax - yMin)) * innerH; commands.push((open ? "L " : "M ") + x.toFixed(1) + "," + y.toFixed(1)); open = true; }); return commands.join(" ");
  }

  function cumulativeSeries(values, payout) { var result = [], acc = payout ? 1 : 0, available = true; values.forEach(function (value) { if (!available || !finite(value)) { result.push(null); available = false; return; } acc = payout ? acc * (1 + Number(value)) : acc + Number(value); result.push(acc); }); return result; }
  function drawdownSeries(values) { var result = [], peak = -Infinity, available = true; values.forEach(function (value) { if (!available || !finite(value)) { result.push(null); available = false; return; } peak = Math.max(peak, Number(value)); result.push(peak > 0 ? Number(value) / peak - 1 : 0); }); return result; }
  function svgAreaPath(values, yMin, yMax, yBase, width, height, pad) {
    if (!values || !values.length) return ""; if (Math.abs(yMax - yMin) < 1e-12) { yMin -= 1; yMax += 1; }
    var innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom, denom = Math.max(1, values.length - 1), base = pad.top + (1 - (yBase - yMin) / (yMax - yMin)) * innerH, parts = [], segment = [];
    function flush() { if (!segment.length) return; var points = segment.map(function (entry) { var x = pad.left + entry.index / denom * innerW, y = pad.top + (1 - (Number(entry.value) - yMin) / (yMax - yMin)) * innerH; return x.toFixed(1) + "," + y.toFixed(1); }); var firstX = pad.left + segment[0].index / denom * innerW, lastX = pad.left + segment[segment.length - 1].index / denom * innerW; parts.push("M " + points.join(" L ") + " L " + lastX.toFixed(1) + "," + base.toFixed(1) + " L " + firstX.toFixed(1) + "," + base.toFixed(1) + " Z"); segment = []; }
    values.forEach(function (value, index) { if (!finite(value)) { flush(); return; } segment.push({index: index, value: value}); }); flush(); return parts.join(" ");
  }

  function activeSeries() {
    if (metrics && metrics.series) {
      var scale = Number(metrics.scale) || 1;
      var seriesIds = metrics.model_ids || [];
      if (!seriesIds.length) seriesIds = (metrics.model_indices || []).map(function (modelIndex) { return (payload.model_ids || [])[modelIndex]; });
      return seriesIds.map(function (id, index) {
        var encoded = (metrics.series[currentMetric] || [])[index] || [];
        var standard = decodeSeries(encoded, scale);
        return {id: id, label: compactModelLabel((metrics.labels || [])[index] || id), values: currentView === "cumulative" ? cumulativeSeries(standard, currentMetric === "payout") : standard, color: chartColors[index % chartColors.length]};
      });
    }
    var metric = metrics[currentMetric] || {};
    return Object.keys(metric).sort().map(function (id, index) { var standard = decodeSeries(metric[id].standard || [], 1), values = currentView === "cumulative" ? cumulativeSeries(standard, currentMetric === "payout") : standard; return {id: id, label: metric[id].label, values: values, color: chartColors[index % chartColors.length]}; });
  }

  function renderTimeseries() {
    var svg = byId("timeseries-svg"); if (!svg) return; var series = activeSeries(), range = globalYRange(series.map(function (item) { return item.values; })); if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1; range.max += 1; } svg.textContent = "";
    var config = (metricConfig[currentMetric] || metricConfig.payout)[currentView];
    var innerWidth = 800 - timeseriesPadding.left - timeseriesPadding.right;
    var innerHeight = 320 - timeseriesPadding.top - timeseriesPadding.bottom;
    [range.max, (range.max + range.min) / 2, range.min].forEach(function (value) {
      var y = timeseriesPadding.top + (1 - (value - range.min) / (range.max - range.min)) * innerHeight;
      var grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
      grid.setAttribute("x1", timeseriesPadding.left); grid.setAttribute("x2", 800 - timeseriesPadding.right);
      grid.setAttribute("y1", y.toFixed(1)); grid.setAttribute("y2", y.toFixed(1)); grid.setAttribute("class", "chart-grid-line"); svg.appendChild(grid);
      var tick = textNode(svg, fmt(value, config.percent), 48, y + 4);
      tick.setAttribute("class", "axis-tick"); tick.setAttribute("text-anchor", "end"); svg.appendChild(tick);
    });
    var xIndices = eras.length > 1 ? [0, Math.floor((eras.length - 1) / 2), eras.length - 1] : [0];
    xIndices.forEach(function (index) {
      var x = timeseriesPadding.left + index / Math.max(1, eras.length - 1) * innerWidth;
      var tick = textNode(svg, eras[index] || "", x, 300); tick.setAttribute("class", "axis-tick"); svg.appendChild(tick);
    });
    var yTitle = textNode(svg, config.label, 13, 160); yTitle.setAttribute("id", "timeseries-y-axis"); yTitle.setAttribute("class", "axis-title"); yTitle.setAttribute("transform", "rotate(-90 13 160)"); svg.appendChild(yTitle);
    var xTitle = textNode(svg, "Evaluation era", 420, 318); xTitle.setAttribute("id", "timeseries-x-axis"); xTitle.setAttribute("class", "axis-title"); svg.appendChild(xTitle);
    series.forEach(function (item) { var path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", dataToSvgPath(item.values, range.min, range.max, 800, 320, timeseriesPadding)); path.setAttribute("stroke", item.color); path.setAttribute("class", "series-line"); svg.appendChild(path); });
    crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line"); crosshair.setAttribute("class", "crosshair"); crosshair.setAttribute("visibility", "hidden"); crosshair.setAttribute("y1", timeseriesPadding.top); crosshair.setAttribute("y2", 320 - timeseriesPadding.bottom); svg.appendChild(crosshair);
    var axis = byId("axis-label"), axisTitle = byId("timeseries-axis-label"); if (axis) axis.textContent = config.label; if (axisTitle) axisTitle.textContent = config.label;
    var legend = byId("timeseries-legend"); if (legend) legend.innerHTML = series.map(function (item) { return "<span class=\"legend-item\"><i style=\"--legend-color:" + item.color + "\"></i><b>" + esc(item.label) + "</b></span>"; }).join("");
  }

  function attachTimeseriesTooltip() {
    var svg = byId("timeseries-svg"), tip = byId("timeseries-tooltip");
    if (!svg || !tip || svg.getAttribute("data-tooltip-bound")) return;
    svg.setAttribute("data-tooltip-bound", "true");
    var update = function (event) {
      if (!eras.length || !crosshair) return;
      var index = pointIndexFromEvent(event, svg, eras.length, timeseriesPadding);
      var centerX = timeseriesPadding.left + index / Math.max(1, eras.length - 1) * (800 - timeseriesPadding.left - timeseriesPadding.right);
      crosshair.setAttribute("visibility", "visible"); crosshair.setAttribute("x1", centerX); crosshair.setAttribute("x2", centerX);
      var config = (metricConfig[currentMetric] || metricConfig.payout)[currentView];
      var lines = ["<strong>Era " + esc(eras[index]) + "</strong>"];
      activeSeries().forEach(function (item) { lines.push(tooltipLine(item.color, item.label, fmt(item.values[index], config.percent))); });
      showChartTooltip(tip, event, lines.join("<br>"), svg);
    };
    svg.addEventListener("pointermove", update);
    svg.addEventListener("pointerdown", update);
    svg.addEventListener("pointerleave", function () { if (crosshair) crosshair.setAttribute("visibility", "hidden"); hideChartTooltip(tip); });
    svg.addEventListener("pointercancel", function () { if (crosshair) crosshair.setAttribute("visibility", "hidden"); hideChartTooltip(tip); });
  }

  function renderSimilarity() {
    var host = byId("similarity-host"); if (!host) return; var similarity = payload.similarity || {labels: [], matrix: []}; if (!similarity.matrix || !similarity.matrix.length) { host.innerHTML = "<p class=\"empty-note\">Similarity matrix unavailable</p>"; return; }
    var html = "<table class=\"similarity\"><thead><tr><th></th>" + similarity.labels.map(function (label) { return "<th>" + esc(label) + "</th>"; }).join("") + "</tr></thead><tbody>"; similarity.matrix.forEach(function (row, index) { html += "<tr><th>" + esc(similarity.labels[index]) + "</th>" + row.map(function (value, column) { var diagonal = index === column; return diagonal ? "<td class=\"diagonal\" aria-label=\"Self similarity\"><span></span></td>" : "<td data-sim-row=\"" + index + "\" data-sim-column=\"" + column + "\" style=\"--matrix-alpha:" + (finite(value) ? (0.12 + 0.5 * Math.abs(Number(value))).toFixed(2) : "0") + ";--matrix-color:" + (finite(value) && Number(value) < 0 ? "var(--coral)" : "var(--mint)") + "\">" + (finite(value) ? Number(value).toFixed(3) : "&#8212;") + "</td>"; }).join("") + "</tr>"; }); host.innerHTML = html + "</tbody></table>";
    var similarityTip = byId("similarity-tooltip");
    host.querySelectorAll("td[data-sim-row]").forEach(function (cell) {
      bindPointerTooltip(cell, similarityTip, function () {
        var rowIndex = Number(cell.getAttribute("data-sim-row")), column = Number(cell.getAttribute("data-sim-column")), value = similarity.matrix[rowIndex][column];
        return "<strong>Signal similarity</strong><br>" + tooltipLine(finite(value) && Number(value) < 0 ? "var(--coral)" : "var(--mint)", compactModelLabel(similarity.labels[rowIndex]) + " / " + compactModelLabel(similarity.labels[column]), "Rank correlation " + (finite(value) ? Number(value).toFixed(4) : "&#8212;"));
      });
    });
  }

  function renderDrawdown() {
    var svg = byId("drawdown-svg"); if (!svg) return;
    var seriesIds = metrics.model_ids || [], modelIndices = metrics.model_indices || [], ids = seriesIds.length ? seriesIds : modelIndices.map(function (index) { return (payload.model_ids || [])[index]; }), scale = Number(metrics.scale) || 1;
    var payout = metrics.series ? (metrics.series.payout || []).map(function (values) { return decodeSeries(values, scale); }) : ids.map(function (id) { return decodeSeries((metrics.payout[id] || {}).standard || [], 1); });
    var paths = payout.map(function (standard) { return drawdownSeries(cumulativeSeries(standard || [], true)); }), range = globalYRange(paths); if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1; range.max += 1; } svg.textContent = "";
    var innerWidth = 800 - drawdownPadding.left - drawdownPadding.right, innerHeight = 240 - drawdownPadding.top - drawdownPadding.bottom;
    [range.max, (range.max + range.min) / 2, range.min].forEach(function (value) { var y = drawdownPadding.top + (1 - (value - range.min) / (range.max - range.min)) * innerHeight; var grid = document.createElementNS("http://www.w3.org/2000/svg", "line"); grid.setAttribute("x1", drawdownPadding.left); grid.setAttribute("x2", 800 - drawdownPadding.right); grid.setAttribute("y1", y.toFixed(1)); grid.setAttribute("y2", y.toFixed(1)); grid.setAttribute("class", "chart-grid-line"); svg.appendChild(grid); var tick = textNode(svg, fmt(value, true), 48, y + 4); tick.setAttribute("class", "axis-tick"); tick.setAttribute("text-anchor", "end"); svg.appendChild(tick); });
    var xIndices = eras.length > 1 ? [0, Math.floor((eras.length - 1) / 2), eras.length - 1] : [0]; xIndices.forEach(function (index) { var x = drawdownPadding.left + index / Math.max(1, eras.length - 1) * innerWidth; var tick = textNode(svg, eras[index] || "", x, 226); tick.setAttribute("class", "axis-tick"); svg.appendChild(tick); });
    var yTitle = textNode(svg, "Drawdown", 13, 120); yTitle.setAttribute("class", "axis-title"); yTitle.setAttribute("transform", "rotate(-90 13 120)"); svg.appendChild(yTitle); var xTitle = textNode(svg, "Evaluation era", 420, 238); xTitle.setAttribute("class", "axis-title"); svg.appendChild(xTitle);
    var colors = chartColors, drawdownTip = byId("drawdown-tooltip"); paths.forEach(function (path, index) { var area = document.createElementNS("http://www.w3.org/2000/svg", "path"); area.setAttribute("d", svgAreaPath(path, range.min, range.max, 0, 800, 240, drawdownPadding)); area.setAttribute("class", "drawdown-area"); area.setAttribute("stroke", colors[index % colors.length]); area.setAttribute("fill", colors[index % colors.length]); svg.appendChild(area); });
    var crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line"); crosshair.setAttribute("class", "crosshair"); crosshair.setAttribute("visibility", "hidden"); crosshair.setAttribute("y1", drawdownPadding.top); crosshair.setAttribute("y2", 240 - drawdownPadding.bottom); svg.appendChild(crosshair);
    var overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect"); overlay.setAttribute("class", "chart-hit-area"); overlay.setAttribute("x", drawdownPadding.left); overlay.setAttribute("y", drawdownPadding.top); overlay.setAttribute("width", innerWidth); overlay.setAttribute("height", innerHeight); overlay.setAttribute("fill", "transparent"); svg.appendChild(overlay);
    var update = function (event) { if (!eras.length) return; var eraIndex = pointIndexFromEvent(event, svg, eras.length, drawdownPadding), x = drawdownPadding.left + eraIndex / Math.max(1, eras.length - 1) * innerWidth; crosshair.setAttribute("x1", x); crosshair.setAttribute("x2", x); crosshair.setAttribute("visibility", "visible"); var lines = ["<strong>Era " + esc(eras[eraIndex]) + "</strong>"]; paths.forEach(function (path, index) { lines.push(tooltipLine(colors[index % colors.length], compactModelLabel((metrics.labels || [])[index] || ids[index]), fmt(path[eraIndex], true))); }); showChartTooltip(drawdownTip, event, lines.join("<br>"), svg); };
    overlay.style.touchAction = "pan-y"; overlay.addEventListener("pointermove", update); overlay.addEventListener("pointerdown", update); overlay.addEventListener("pointerleave", function () { crosshair.setAttribute("visibility", "hidden"); hideChartTooltip(drawdownTip); }); overlay.addEventListener("pointercancel", function () { crosshair.setAttribute("visibility", "hidden"); hideChartTooltip(drawdownTip); });
    var legend = byId("drawdown-legend"); if (legend) legend.innerHTML = paths.map(function (_, index) { return "<span class=\"legend-item\"><i style=\"--legend-color:" + colors[index % colors.length] + "\"></i><b>" + esc(compactModelLabel((metrics.labels || [])[index] || ids[index])) + "</b></span>"; }).join("");
  }

  function renderLeaderboard() { renderMasterLeaderboard(); }

  var metricSelect = byId("metric-select"), standardButton = byId("view-standard"), cumulativeButton = byId("view-cumulative");
  if (metricSelect) metricSelect.addEventListener("change", function () { currentMetric = metricSelect.value; renderTimeseries(); });
  if (standardButton) standardButton.addEventListener("click", function () { currentView = "standard"; standardButton.classList.add("active"); cumulativeButton.classList.remove("active"); renderTimeseries(); });
  if (cumulativeButton) cumulativeButton.addEventListener("click", function () { currentView = "cumulative"; cumulativeButton.classList.add("active"); standardButton.classList.remove("active"); renderTimeseries(); });
  document.querySelectorAll(".primary-nav .nav-item").forEach(function (item) {
    item.addEventListener("click", function () {
      document.querySelectorAll(".primary-nav .nav-item").forEach(function (navItem) { navItem.classList.remove("active"); });
      item.classList.add("active");
    });
  });

  populateToolbar();
  renderAll();
  attachTimeseriesTooltip();
})();
