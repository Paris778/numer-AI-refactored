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
  var rows = (payload.rows || []).map(function (record) {
    if (!Array.isArray(record)) return record;
    var row = {};
    rowFields.forEach(function (field, index) { row[field] = record[index]; });
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

  function byId(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#x27;");
  }

  function finite(value) { return value !== null && value !== undefined && isFinite(Number(value)); }

  function fmt(value, percent) {
    if (!finite(value)) return "&#8212;";
    return percent ? (Number(value) * 100).toFixed(2) + "%" : Number(value).toFixed(4);
  }

  function safeText(value) {
    return value === null || value === undefined || value === "" ? "&#8212;" : esc(value);
  }

  function specFor(metric) {
    for (var i = 0; i < metricSpecs.length; i++) if (metricSpecs[i].name === metric) return metricSpecs[i];
    return {name: metric, label: metric, higher_is_better: true, direction: "higher"};
  }

  function metricValue(row, metric) {
    var index = (payload.metric_fields || []).indexOf(metric);
    return index < 0 || !row.values ? null : row.values[index];
  }

  function rankValue(row, metric) {
    var index = (payload.metric_fields || []).indexOf(metric);
    var rowIndex = rows.indexOf(row), ranks = (payload.rank_values || [])[rowIndex];
    return index < 0 || !ranks ? null : ranks[index];
  }

  function compareRows(a, b, metric) {
    var spec = specFor(metric), av = metricValue(a, metric), bv = metricValue(b, metric);
    if (!finite(av) && !finite(bv)) return String(a.model_id).localeCompare(String(b.model_id));
    if (!finite(av)) return 1;
    if (!finite(bv)) return -1;
    var delta = Number(av) - Number(bv);
    if (!spec.higher_is_better) delta = -delta;
    return delta === 0 ? String(a.model_id).localeCompare(String(b.model_id)) : (delta > 0 ? -1 : 1);
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
      var baseline = Number(metricValue(benchmark, state.metric)), spec = specFor(state.metric);
      return sortedRows(visible.filter(function (row) {
        if (row.cohort !== "trained" || !finite(metricValue(row, state.metric))) return false;
        return spec.higher_is_better ? Number(metricValue(row, state.metric)) > baseline : Number(metricValue(row, state.metric)) < baseline;
      }));
    }
    if (state.shortcut === "robust") {
      return visible.filter(function (row) { return row.cohort !== "full"; }).sort(function (a, b) {
        var aScore = Object.keys(a.robustness || {}).filter(function (key) { return a.robustness[key] === true; }).length;
        var bScore = Object.keys(b.robustness || {}).filter(function (key) { return b.robustness[key] === true; }).length;
        return bScore - aScore || compareRows(a, b, state.metric);
      });
    }
    return sortedRows(visible);
  }

  function valueText(row, metric) {
    return fmt(metricValue(row, metric), metric === "cagr_1y" || metric === "max_drawdown");
  }

  function ciText(row) {
    var low = row.ci && row.ci[0], high = row.ci && row.ci[1];
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
    var spec = specFor(state.metric), html = "<table class=\"tournament-table\"><thead><tr>" +
      "<th>Rank</th><th>Model</th><th>Type</th><th class=\"selected-head\">RANKED: " + esc(spec.label) + "</th>" +
      "<th>CORR</th><th>MMC (CORE)</th><th>CORR Sharpe + CI</th><th>CAGR / G:P</th><th>Max DD</th><th>Eras</th></tr></thead><tbody>";
    visible.forEach(function (row) {
      var rowClass = row.champion ? " champion-row" : "";
      if (row.cohort === "full") rowClass += " full-row";
      var rowRank = rankValue(row, state.metric), rank = rowRank !== null && rowRank !== undefined ? "#" + rowRank : "&#8212;";
      html += "<tr class=\"model-row" + rowClass + "\" data-model-id=\"" + esc(row.model_id) + "\" tabindex=\"0\">";
      html += "<td class=\"rank-cell mono\">" + rank + "</td><td class=\"model-cell\"><strong>" + esc(row.run_name || row.model_id) + "</strong>";
      html += row.champion ? " <span class=\"champion-mark\">CHAMPION</span>" : "";
      html += "<small>" + esc(row.model_id) + "</small></td><td>" + marker(row) + "</td>";
      html += "<td class=\"num selected-value\">" + valueText(row, state.metric) + "</td><td class=\"num\">" + valueText(row, "corr") + "</td>";
      html += "<td class=\"num\">" + valueText(row, "mmc") + "</td><td class=\"num\">" + ciText(row) + "</td>";
      html += "<td class=\"num\"><span>" + valueText(row, "cagr_1y") + "</span><span class=\"secondary-number\">" + valueText(row, "gain_to_pain_ratio") + "</span></td>";
      html += "<td class=\"num\">" + valueText(row, "max_drawdown") + "</td><td class=\"num\">" + (row.ci && finite(row.ci[2]) ? row.ci[2] : "&#8212;") + "</td></tr>";
    });
    host.innerHTML = html + "</tbody></table>";
    host.querySelectorAll(".model-row").forEach(function (node) {
      node.addEventListener("click", function () { openDossier(node.getAttribute("data-model-id")); });
      node.addEventListener("keydown", function (event) { if (event.key === "Enter" || event.key === " ") openDossier(node.getAttribute("data-model-id")); });
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
    points.forEach(function (point) {
      var px = x(Number(point.corr)), py = y(Number(point.mmc)), shape, color = cohortColors[point.cohort] || cohortColors.benchmark;
      if (point.cohort === "heuristic") { shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon"); shape.setAttribute("points", px + "," + (py - 6) + " " + (px + 6) + "," + py + " " + px + "," + (py + 6) + " " + (px - 6) + "," + py); }
      else if (point.cohort === "benchmark") { shape = document.createElementNS("http://www.w3.org/2000/svg", "rect"); shape.setAttribute("x", px - 5); shape.setAttribute("y", py - 5); shape.setAttribute("width", 10); shape.setAttribute("height", 10); }
      else { shape = document.createElementNS("http://www.w3.org/2000/svg", "circle"); shape.setAttribute("cx", px); shape.setAttribute("cy", py); shape.setAttribute("r", point.champion ? 7 : 5); }
      shape.setAttribute("fill", color); shape.setAttribute("class", point.champion ? "landscape-point champion-point" : "landscape-point"); shape.setAttribute("data-model-id", point.model_id); shape.addEventListener("click", function () { openDossier(point.model_id); });
      var title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = point.model_id + " / CORR " + fmt(point.corr, false) + " / MMC " + fmt(point.mmc, false); shape.appendChild(title); svg.appendChild(shape);
    });
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
    available.forEach(function (spec, index) {
      var value = Number(metricValue(row, spec.name)), y = top + index * rowHeight, width = Math.abs(value) / max * barWidth, x = value >= 0 ? zero : zero - width;
      var bar = document.createElementNS("http://www.w3.org/2000/svg", "rect"); bar.setAttribute("x", x); bar.setAttribute("y", y); bar.setAttribute("width", width); bar.setAttribute("height", 13); bar.setAttribute("class", value >= 0 ? "profile-bar positive" : "profile-bar negative"); svg.appendChild(bar);
      var name = textNode(svg, spec.label, 198, y + 11); name.setAttribute("text-anchor", "end"); svg.appendChild(name); var valueTextNode = textNode(svg, fmt(value, spec.name === "cagr_1y" || spec.name === "max_drawdown"), Math.min(720, zero + width + 8), y + 11); valueTextNode.setAttribute("class", "profile-value"); svg.appendChild(valueTextNode);
    });
    var baseline = document.createElementNS("http://www.w3.org/2000/svg", "line"); baseline.setAttribute("x1", zero); baseline.setAttribute("x2", zero); baseline.setAttribute("y1", 12); baseline.setAttribute("y2", top + available.length * rowHeight); baseline.setAttribute("class", "profile-baseline"); svg.appendChild(baseline);
  }

  function renderDossier(row) {
    var host = byId("model-dossier"), rowIndex = rows.indexOf(row), packed = (payload.details || [])[rowIndex] || [[], [], null, null];
    var detail = {scorecard_values: packed[0] || [], provenance_values: packed[1] || [], evidence_ref: packed[2], reason: packed[3]}; if (!host) return;
    var ranks = metricSpecs.filter(function (spec) { return rankValue(row, spec.name) !== null && rankValue(row, spec.name) !== undefined; }).map(function (spec) { return "<span>" + esc(spec.label) + " <b>#" + rankValue(row, spec.name) + "</b></span>"; }).join("");
    var scorecard = {};
    (payload.metric_fields || []).forEach(function (key, index) { scorecard[key] = row.values ? row.values[index] : null; });
    scorecard.corr_sharpe_ac_ci_low = row.ci ? row.ci[0] : null;
    scorecard.corr_sharpe_ac_ci_high = row.ci ? row.ci[1] : null;
    scorecard.corr_sharpe_ac_n_eras = row.ci ? row.ci[2] : null;
    (payload.scorecard_fields || []).forEach(function (key, index) { scorecard[key] = detail.scorecard_values ? detail.scorecard_values[index] : null; });
    var keys = Object.keys(scorecard).filter(function (key) { return scorecard[key] !== null && scorecard[key] !== undefined; }).sort();
    var scorecardHtml = keys.length ? keys.map(function (key) { var value = scorecard[key], display = key.indexOf("cagr") !== -1 || key.indexOf("drawdown") !== -1 ? fmt(value, true) : (finite(value) ? fmt(value, false) : esc(value)); return "<tr><th>" + esc(key.replace(/_/g, " ")) + "</th><td class=\"num\">" + display + "</td></tr>"; }).join("") : "<tr><td colspan=\"2\" class=\"missing\">&#8212;</td></tr>";
    var provenance = {backend: row.backend, preset: row.preset, feature_set: row.feature_set, feature_subset: row.feature_subset, targets: row.targets, neutralization_proportion: row.neutralization_proportion, oof_device: row.oof_device};
    (payload.provenance_fields || []).forEach(function (key, index) { provenance[key] = detail.provenance_values[index]; });
    var provenanceKeys = Object.keys(provenance).sort(), provenanceHtml = provenanceKeys.map(function (key) { var value = provenance[key]; if (value === null || value === undefined) return ""; value = Array.isArray(value) ? value.join(", ") : value; return "<div><span>" + esc(key.replace(/_/g, " ")) + "</span><strong>" + esc(value) + "</strong></div>"; }).join("");
    if (!provenanceHtml) provenanceHtml = "<p class=\"missing\">Provenance unavailable</p>";
    var evidenceRef = detail.evidence_ref;
    if (!evidenceRef && (row.source === "trained" || row.source === "trained_legacy")) evidenceRef = "registry/" + row.model_id + "/run.json";
    var ref = evidenceRef ? "<a href=\"" + esc(evidenceRef) + "\">Open immutable evidence</a>" : "<span class=\"missing\">Evidence link unavailable</span>";
    var meta = payload.meta || {}, window = meta.evaluation_window || {};
    host.innerHTML = "<div class=\"drawer-kicker\">" + esc(row.cohort || row.source) + " &middot; " + esc(row.status || "RESEARCH") + "</div><h2 id=\"drawer-title\">" + esc(row.run_name || row.model_id) + "</h2><p class=\"drawer-id mono\">" + esc(row.model_id) + "</p>" +
      "<div class=\"drawer-meta\"><span>Suite " + safeText(meta.suite_version) + "</span><span>Window " + safeText(window.start) + " &rarr; " + safeText(window.end) + "</span><span>Timestamp " + safeText(provenance.timestamp) + "</span></div>" +
      "<h3>Rank across metrics</h3><div class=\"rank-list\">" + (ranks || "<span>&#8212;</span>") + "</div><h3>Full scorecard evidence</h3><table class=\"dossier-table\"><tbody>" + scorecardHtml + "</tbody></table>" +
      "<h3>Provenance / lineage</h3><div class=\"provenance-list\">" + provenanceHtml + "</div><p class=\"evidence-link\">" + ref + "</p><p class=\"drawer-note\">Missing values remain unavailable; this dossier contains offline evaluation evidence only.</p>";
  }

  function openDossier(modelId) {
    var row = rows.filter(function (item) { return item.model_id === modelId; })[0]; if (!row) return;
    state.selected = modelId; renderDossier(row); renderProfile(row); var drawer = byId("model-drawer"); if (drawer) { drawer.hidden = false; document.body.classList.add("drawer-open"); }
  }

  function closeDossier() { var drawer = byId("model-drawer"); if (drawer) { drawer.hidden = true; document.body.classList.remove("drawer-open"); } }

  function populateToolbar() {
    var select = byId("rank-by");
    if (select) {
      select.innerHTML = metricSpecs.map(function (spec) { return "<option value=\"" + esc(spec.name) + "\">" + esc(spec.label) + "</option>"; }).join(""); select.value = state.metric;
      select.addEventListener("change", function () { state.metric = select.value; state.shortcut = null; renderAll(); });
    }
    var search = byId("model-search"); if (search) search.addEventListener("input", function () { state.search = search.value; renderMasterLeaderboard(); });
    document.querySelectorAll(".cohort-button").forEach(function (button) { button.addEventListener("click", function () { state.cohort = button.getAttribute("data-cohort"); state.shortcut = null; document.querySelectorAll(".cohort-button").forEach(function (item) { item.classList.toggle("active", item === button); }); renderAll(); }); });
    document.querySelectorAll(".shortcut-button:not([disabled])").forEach(function (button) { button.addEventListener("click", function () { var value = button.getAttribute("data-shortcut"); state.shortcut = state.shortcut === value ? null : value; document.querySelectorAll(".shortcut-button").forEach(function (item) { item.classList.toggle("active", item === button && state.shortcut !== null); }); renderAll(); }); });
    document.querySelectorAll("[data-drawer-close]").forEach(function (button) { button.addEventListener("click", closeDossier); }); document.addEventListener("keydown", function (event) { if (event.key === "Escape") closeDossier(); });
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
    var innerW = width - pad.left - pad.right, innerH = height - pad.top - pad.bottom, denom = Math.max(1, values.length - 1), points = [];
    values.forEach(function (value, index) { var x = pad.left + index / denom * innerW, y = pad.top + (1 - (Number(value) - yMin) / (yMax - yMin)) * innerH; points.push(x.toFixed(1) + "," + y.toFixed(1)); }); return "M " + points.join(" L ");
  }

  function cumulativeSeries(values, payout) { var result = [], acc = payout ? 1 : 0; values.forEach(function (value) { acc = payout ? acc * (1 + Number(value)) : acc + Number(value); result.push(acc); }); return result; }
  function drawdownSeries(values) { var result = [], peak = -Infinity; values.forEach(function (value) { peak = Math.max(peak, Number(value)); result.push(peak > 0 ? Number(value) / peak - 1 : 0); }); return result; }
  function svgAreaPath(values, yMin, yMax, yBase, width, height, pad) { var line = dataToSvgPath(values, yMin, yMax, width, height, pad); if (!line) return ""; if (Math.abs(yMax - yMin) < 1e-12) { yMin -= 1; yMax += 1; } var innerH = height - pad.top - pad.bottom, innerW = width - pad.left - pad.right, denom = Math.max(1, values.length - 1), base = pad.top + (1 - (yBase - yMin) / (yMax - yMin)) * innerH, xN = pad.left + (values.length - 1) / denom * innerW; return line + " L " + xN.toFixed(1) + "," + base.toFixed(1) + " L " + pad.left.toFixed(1) + "," + base.toFixed(1) + " Z"; }

  function activeSeries() {
    if (metrics && metrics.series) {
      var scale = Number(metrics.scale) || 1;
      return (metrics.model_ids || []).map(function (id, index) {
        var encoded = (metrics.series[currentMetric] || [])[index] || [];
        var standard = encoded.map(function (value) { return finite(value) ? Number(value) / scale : 0; });
        return {id: id, label: (metrics.labels || [])[index] || id, values: currentView === "cumulative" ? cumulativeSeries(standard, currentMetric === "payout") : standard, color: ["#4f46e5", "#0f766e", "#c58b26", "#b42318", "#64748b"][index % 5]};
      });
    }
    var metric = metrics[currentMetric] || {};
    return Object.keys(metric).sort().map(function (id, index) { var standard = metric[id].standard || [], values = currentView === "cumulative" ? cumulativeSeries(standard, currentMetric === "payout") : standard; return {id: id, label: metric[id].label, values: values, color: ["#4f46e5", "#0f766e", "#c58b26", "#b42318", "#64748b"][index % 5]}; });
  }

  function renderTimeseries() {
    var svg = byId("timeseries-svg"); if (!svg) return; var series = activeSeries(), range = globalYRange(series.map(function (item) { return item.values; })); if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1; range.max += 1; } svg.textContent = "";
    series.forEach(function (item) { var path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", dataToSvgPath(item.values, range.min, range.max, 800, 320, timeseriesPadding)); path.setAttribute("stroke", item.color); path.setAttribute("class", "series-line"); svg.appendChild(path); });
    crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line"); crosshair.setAttribute("class", "crosshair"); crosshair.setAttribute("visibility", "hidden"); crosshair.setAttribute("y1", timeseriesPadding.top); crosshair.setAttribute("y2", 320 - timeseriesPadding.bottom); svg.appendChild(crosshair);
    var axis = byId("axis-label"), config = (metricConfig[currentMetric] || metricConfig.payout)[currentView]; if (axis) axis.textContent = config.label;
  }

  function attachTimeseriesTooltip() {
    var svg = byId("timeseries-svg"), tip = byId("timeseries-tooltip");
    if (!svg || !tip || svg.getAttribute("data-tooltip-bound")) return;
    svg.setAttribute("data-tooltip-bound", "true");
    svg.addEventListener("mousemove", function (event) {
      if (!eras.length || !crosshair) return;
      var rect = svg.getBoundingClientRect(), viewX = (event.clientX - rect.left) * (800 / rect.width);
      var innerWidth = 800 - timeseriesPadding.left - timeseriesPadding.right;
      var index = Math.round((viewX - timeseriesPadding.left) / (innerWidth / Math.max(1, eras.length - 1)));
      index = Math.max(0, Math.min(index, eras.length - 1));
      var centerX = timeseriesPadding.left + index / Math.max(1, eras.length - 1) * innerWidth;
      crosshair.setAttribute("visibility", "visible"); crosshair.setAttribute("x1", centerX); crosshair.setAttribute("x2", centerX);
      var config = (metricConfig[currentMetric] || metricConfig.payout)[currentView];
      var lines = ["<strong>Era " + esc(eras[index]) + "</strong>"];
      activeSeries().forEach(function (item) { lines.push(esc(item.label) + ": " + fmt(item.values[index], config.percent)); });
      tip.innerHTML = lines.join("<br>"); tip.hidden = false;
      tip.style.left = Math.min(Math.max(8, viewX + 12), 800 - 220) + "px"; tip.style.top = "8px";
    });
    svg.addEventListener("mouseleave", function () { if (crosshair) crosshair.setAttribute("visibility", "hidden"); tip.hidden = true; });
  }

  function renderSimilarity() {
    var host = byId("similarity-host"); if (!host) return; var similarity = payload.similarity || {labels: [], matrix: []}; if (!similarity.matrix || !similarity.matrix.length) { host.innerHTML = "<p class=\"empty-note\">Similarity matrix unavailable</p>"; return; }
    var html = "<table class=\"similarity\"><thead><tr><th></th>" + similarity.labels.map(function (label) { return "<th>" + esc(label) + "</th>"; }).join("") + "</tr></thead><tbody>"; similarity.matrix.forEach(function (row, index) { html += "<tr><th>" + esc(similarity.labels[index]) + "</th>" + row.map(function (value) { return "<td>" + (finite(value) ? Number(value).toFixed(3) : "&#8212;") + "</td>"; }).join("") + "</tr>"; }); host.innerHTML = html + "</tbody></table>";
  }

  function renderDrawdown() {
    var svg = byId("drawdown-svg"); if (!svg) return; var ids = metrics.model_ids || [], scale = Number(metrics.scale) || 1, payout = metrics.series ? (metrics.series.payout || []).map(function (values) { return values.map(function (value) { return finite(value) ? Number(value) / scale : 0; }); }) : ids.map(function (id) { return (metrics.payout[id] || {}).standard || []; }), paths = payout.map(function (standard) { return drawdownSeries(cumulativeSeries(standard || [], true)); }), range = globalYRange(paths); if (Math.abs(range.max - range.min) < 1e-12) { range.min -= 1; range.max += 1; } svg.textContent = "";
    paths.forEach(function (path, index) { var area = document.createElementNS("http://www.w3.org/2000/svg", "path"); area.setAttribute("d", svgAreaPath(path, range.min, range.max, 0, 800, 240, drawdownPadding)); area.setAttribute("class", "drawdown-area"); area.setAttribute("stroke", ["#4f46e5", "#0f766e", "#c58b26"][index % 3]); svg.appendChild(area); });
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
