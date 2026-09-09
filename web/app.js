(function () {
  "use strict";

  var bundle = JSON.parse(document.getElementById("bundle-data").textContent);
  var images = JSON.parse(document.getElementById("image-data").textContent);

  /* Bundled cases arrive as sprite sheets; an uploaded study arrives as typed
     arrays parsed from its NIfTI. Both answer the same accessors below. */
  var cases = bundle.cases.map(function (c) {
    var copy = JSON.parse(JSON.stringify(c));
    copy.source = "sprite";
    copy.hasReference = true;
    return copy;
  });

  var state = {
    caseIndex: 0,
    slice: 0,
    mode: "overlay",       // overlay | prediction | reference | difference | ct
    playing: false,
    timer: null,
    opacity: 0.38,
    gain: 1.0
  };

  var sprites = {};
  var maskCache = {};
  var rgbCache = {};

  var viewport = document.getElementById("viewport");
  var vctx = viewport.getContext("2d");
  var scratch = document.createElement("canvas");
  var sctx = scratch.getContext("2d", { willReadFrequently: true });
  var layer = document.createElement("canvas");
  var lctx = layer.getContext("2d");

  function currentCase() { return cases[state.caseIndex]; }
  function byId(id) { return document.getElementById(id); }

  function band(dice) {
    if (dice >= 0.92) { return { key: "strong", label: "Strong" }; }
    if (dice >= 0.85) { return { key: "fair", label: "Fair" }; }
    if (dice >= 0.70) { return { key: "weak", label: "Weak" }; }
    return { key: "poor", label: "Failed" };
  }

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function hsl(name, alpha) {
    var value = token(name);
    return alpha === undefined ? "hsl(" + value + ")" : "hsl(" + value + " / " + alpha + ")";
  }
  function colorRgb(name) {
    var key = name + token(name);
    if (rgbCache[key]) { return rgbCache[key]; }
    var probe = document.createElement("canvas").getContext("2d");
    probe.fillStyle = hsl(name);
    probe.fillRect(0, 0, 1, 1);
    var d = probe.getImageData(0, 0, 1, 1).data;
    rgbCache[key] = [d[0], d[1], d[2]];
    return rgbCache[key];
  }

  /* ---------------- slice accessors ---------------- */

  function loadSprites(caseId, done) {
    if (sprites[caseId]) { done(sprites[caseId]); return; }
    var set = {}, pending = 3;
    ["ct", "pred", "ref"].forEach(function (kind) {
      var img = new Image();
      img.onload = img.onerror = function () {
        pending--;
        if (pending === 0) { sprites[caseId] = set; done(set); }
      };
      img.src = images[caseId + "_" + kind];
      set[kind] = img;
    });
  }

  function sliceRect(kase, index) {
    return {
      sx: (index % kase.cols) * kase.shape[0],
      sy: Math.floor(index / kase.cols) * kase.shape[0 + 1] * 0 + Math.floor(index / kase.cols) * kase.shape[1],
      w: kase.shape[0],
      h: kase.shape[1]
    };
  }

  function maskBits(kase, kind, index) {
    if (kase.source === "array") {
      if (kind !== "pred" || !kase.pred) { return null; }
      var n = kase.shape[0] * kase.shape[1];
      return kase.pred.subarray(index * n, (index + 1) * n);
    }
    var key = kase.case_id + "|" + kind + "|" + index;
    if (maskCache[key]) { return maskCache[key]; }
    var set = sprites[kase.case_id];
    if (!set || !set[kind] || !set[kind].width) { return null; }
    var r = sliceRect(kase, index);
    scratch.width = r.w; scratch.height = r.h;
    sctx.clearRect(0, 0, r.w, r.h);
    sctx.drawImage(set[kind], r.sx, r.sy, r.w, r.h, 0, 0, r.w, r.h);
    var data = sctx.getImageData(0, 0, r.w, r.h).data;
    var bits = new Uint8Array(r.w * r.h);
    for (var p = 0, q = 0; p < data.length; p += 4, q++) { bits[q] = data[p] > 127 ? 1 : 0; }
    maskCache[key] = bits;
    return bits;
  }

  function drawCt(kase) {
    var w = kase.shape[0], h = kase.shape[1];
    vctx.imageSmoothingEnabled = true;
    vctx.filter = "contrast(" + state.gain + ") brightness(" + (0.65 + 0.35 * state.gain) + ")";
    if (kase.source === "array") {
      var n = w * h;
      var slice = kase.ct.subarray(state.slice * n, (state.slice + 1) * n);
      var out = lctx.createImageData(w, h);
      for (var i = 0; i < n; i++) {
        var v = slice[i], o = i * 4;
        out.data[o] = v; out.data[o + 1] = v; out.data[o + 2] = v; out.data[o + 3] = 255;
      }
      layer.width = w; layer.height = h;
      lctx.putImageData(out, 0, 0);
      vctx.drawImage(layer, 0, 0, viewport.width, viewport.height);
    } else {
      var set = sprites[kase.case_id];
      if (!set || !set.ct || !set.ct.width) { vctx.filter = "none"; return false; }
      var r = sliceRect(kase, state.slice);
      vctx.drawImage(set.ct, r.sx, r.sy, r.w, r.h, 0, 0, viewport.width, viewport.height);
    }
    vctx.filter = "none";
    return true;
  }

  function paintFill(kase, kind, colorToken, alpha) {
    var bits = maskBits(kase, kind, state.slice);
    if (!bits) { return; }
    var w = kase.shape[0], h = kase.shape[1];
    var rgb = colorRgb(colorToken);
    var out = lctx.createImageData(w, h);
    for (var i = 0; i < bits.length; i++) {
      if (!bits[i]) { continue; }
      var o = i * 4;
      out.data[o] = rgb[0]; out.data[o + 1] = rgb[1]; out.data[o + 2] = rgb[2]; out.data[o + 3] = 255;
    }
    layer.width = w; layer.height = h;
    lctx.putImageData(out, 0, 0);
    vctx.globalAlpha = alpha;
    vctx.drawImage(layer, 0, 0, viewport.width, viewport.height);
    vctx.globalAlpha = 1;
  }

  function paintOutline(kase, kind, colorToken) {
    var bits = maskBits(kase, kind, state.slice);
    if (!bits) { return; }
    var w = kase.shape[0], h = kase.shape[1];
    var rgb = colorRgb(colorToken);
    var out = lctx.createImageData(w, h);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var i = y * w + x;
        if (!bits[i]) { continue; }
        var edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
          !bits[i - 1] || !bits[i + 1] || !bits[i - w] || !bits[i + w];
        if (!edge) { continue; }
        var o = i * 4;
        out.data[o] = rgb[0]; out.data[o + 1] = rgb[1]; out.data[o + 2] = rgb[2]; out.data[o + 3] = 235;
      }
    }
    layer.width = w; layer.height = h;
    lctx.putImageData(out, 0, 0);
    vctx.drawImage(layer, 0, 0, viewport.width, viewport.height);
  }

  function paintDifference(kase) {
    var pred = maskBits(kase, "pred", state.slice);
    var ref = maskBits(kase, "ref", state.slice);
    if (!pred || !ref) { return; }
    var w = kase.shape[0], h = kase.shape[1];
    var over = colorRgb("--pred"), miss = colorRgb("--miss");
    var out = lctx.createImageData(w, h);
    for (var i = 0; i < pred.length; i++) {
      if (pred[i] === ref[i]) { continue; }
      var o = i * 4, c = pred[i] ? over : miss;
      out.data[o] = c[0]; out.data[o + 1] = c[1]; out.data[o + 2] = c[2]; out.data[o + 3] = 205;
    }
    layer.width = w; layer.height = h;
    lctx.putImageData(out, 0, 0);
    vctx.drawImage(layer, 0, 0, viewport.width, viewport.height);
  }

  function render() {
    var kase = currentCase();
    vctx.setTransform(1, 0, 0, 1, 0, 0);
    vctx.fillStyle = hsl("--well");
    vctx.fillRect(0, 0, viewport.width, viewport.height);
    if (!drawCt(kase)) { return; }

    var hasPred = kase.source === "array" ? !!kase.pred : true;
    if (state.mode === "overlay") {
      if (hasPred) { paintFill(kase, "pred", "--pred", state.opacity); }
      if (kase.hasReference) { paintOutline(kase, "ref", "--ref"); }
    } else if (state.mode === "prediction" && hasPred) {
      paintFill(kase, "pred", "--pred", Math.max(state.opacity, 0.5));
    } else if (state.mode === "reference" && kase.hasReference) {
      paintFill(kase, "ref", "--ref", state.opacity);
    } else if (state.mode === "difference" && kase.hasReference && hasPred) {
      paintDifference(kase);
    }
    updateHud();
  }

  /* ---------------- HUD, series plot ---------------- */

  function seriesFor(kase) {
    if (kase.hasReference && kase.per_slice_dice) {
      return { values: kase.per_slice_dice, title: "Dice by slice", max: 1, unit: "Dice",
               note: "Click the plot to jump to a slice. Dips mark the slices worth reading first." };
    }
    return { values: kase.per_slice_area || [], title: "Liver area by slice", max: null, unit: "ml",
             note: "Liver area per slice from the returned mask. Click the plot to jump to a slice." };
  }

  function updateHud() {
    var kase = currentCase();
    var series = seriesFor(kase);
    var value = series.values[state.slice];
    var z = (state.slice * kase.spacing_mm[2]).toFixed(1);
    byId("hud-case").textContent = kase.case_id + "  ·  " + kase.split.toUpperCase();
    byId("hud-slice").textContent = "slice " + (state.slice + 1) + " / " + kase.shape[2] + "   z " + z + " mm";
    byId("hud-dice").textContent = (value === null || value === undefined)
      ? "no liver on this slice"
      : (series.unit === "Dice" ? "slice Dice " + value.toFixed(3) : "liver area " + value.toFixed(1) + " ml");
    byId("slice-readout").textContent =
      (state.slice + 1) + "/" + kase.shape[2] + "  ·  " + kase.spacing_mm.join(" × ") + " mm";
    drawSeries();
  }

  function drawSeries() {
    var kase = currentCase();
    var series = seriesFor(kase);
    var values = series.values;
    var svg = byId("spark");
    byId("series-title").textContent = series.title;
    byId("series-note").textContent = series.note;

    var present = values.filter(function (v) { return v !== null && v !== undefined; });
    if (!present.length) {
      svg.innerHTML = '<text x="8" y="40" font-size="11" font-family="var(--mono)" fill="' +
        hsl("--muted-foreground") + '">no series for this study</text>';
      byId("spark-note").textContent = "—";
      return;
    }
    var w = 300, h = 74, pad = 6;
    var top = series.max !== null ? series.max : Math.max.apply(null, present) * 1.1;
    var lo = series.max !== null ? Math.max(0, Math.min.apply(null, present) - 0.05) : 0;
    function X(i) { return pad + (w - 2 * pad) * i / Math.max(1, values.length - 1); }
    function Y(v) { return h - 14 - (h - 22) * (v - lo) / Math.max(1e-6, top - lo); }

    var parts = [];
    parts.push('<line x1="' + pad + '" y1="' + Y(top) + '" x2="' + (w - pad) + '" y2="' + Y(top) +
      '" stroke="' + hsl("--border") + '" stroke-width="1"/>');
    var path = "", area = "", started = false;
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      if (v === null || v === undefined) { started = false; continue; }
      var x = X(i).toFixed(1), y = Y(v).toFixed(1);
      path += (started ? "L" : "M") + x + "," + y;
      area += (started ? "L" : "M" + x + "," + (h - 14) + "L") + x + "," + y;
      started = true;
    }
    parts.push('<path d="' + area + '" fill="' + hsl("--primary", 0.12) + '" stroke="none"/>');
    parts.push('<path d="' + path + '" fill="none" stroke="' + hsl("--primary") + '" stroke-width="1.6" stroke-linejoin="round"/>');
    parts.push('<line x1="' + X(state.slice).toFixed(1) + '" y1="4" x2="' + X(state.slice).toFixed(1) +
      '" y2="' + (h - 14) + '" stroke="' + hsl("--pred") + '" stroke-width="1.2"/>');
    var current = values[state.slice];
    if (current !== null && current !== undefined) {
      parts.push('<circle cx="' + X(state.slice).toFixed(1) + '" cy="' + Y(current).toFixed(1) +
        '" r="3" fill="' + hsl("--pred") + '"/>');
    }
    parts.push('<text x="' + pad + '" y="' + (h - 2) + '" font-size="9" font-family="var(--mono)" fill="' +
      hsl("--muted-foreground") + '">1</text>');
    parts.push('<text x="' + (w - pad) + '" y="' + (h - 2) + '" text-anchor="end" font-size="9" font-family="var(--mono)" fill="' +
      hsl("--muted-foreground") + '">' + values.length + '</text>');
    parts.push('<text x="' + (w - pad) + '" y="' + (Y(top) + 10) + '" text-anchor="end" font-size="9" font-family="var(--mono)" fill="' +
      hsl("--muted-foreground") + '">' + (series.unit === "Dice" ? "Dice 1.00" : top.toFixed(1) + " ml") + "</text>");
    svg.innerHTML = parts.join("");
    byId("spark-note").textContent = present.length + " slices with liver";
  }

  /* ---------------- panels ---------------- */

  function renderStudyList() {
    var host = byId("study-list");
    host.innerHTML = "";
    cases.forEach(function (kase, index) {
      var el = document.createElement("button");
      el.className = "study";
      el.setAttribute("role", "option");
      el.setAttribute("aria-selected", index === state.caseIndex ? "true" : "false");
      var right, sub;
      if (kase.hasReference) {
        var b = band(kase.metrics.dice);
        right = '<span class="band ' + b.key + '"><span class="dot"></span>' + kase.metrics.dice.toFixed(3) + "</span>";
        sub = "<span>" + b.label + "</span>";
      } else {
        right = '<span class="band strong tabular">' + (kase.measurements ? kase.measurements.volume_ml.toFixed(0) + " ml" : "—") + "</span>";
        sub = "<span>no reference</span>";
      }
      var review = readReview(kase.case_id);
      el.innerHTML =
        '<span class="study-top"><span class="study-id">' + kase.case_id + "</span>" + right + "</span>" +
        '<span class="study-sub"><span class="tag">' + kase.split + "</span>" + sub +
        '<span class="tabular">' + kase.runtime_seconds.toFixed(2) + " s</span>" +
        (review.status !== "pending" ? '<span class="tag">' + review.status + "</span>" : "") + "</span>";
      el.addEventListener("click", function () { selectCase(index); });
      host.appendChild(el);
    });
    byId("worklist-count").textContent = cases.length + (cases.length === 1 ? " study" : " studies");
  }

  function renderMetrics() {
    var kase = currentCase();
    var pill = byId("metric-band");
    var rows;

    if (kase.hasReference) {
      var m = kase.metrics, b = band(m.dice);
      pill.textContent = b.label;
      pill.className = "pill";
      pill.style.color = hsl("--band-" + b.key);
      pill.style.borderColor = hsl("--band-" + b.key, 0.4);
      pill.style.background = hsl("--band-" + b.key, 0.12);
      rows = [
        ["Dice", m.dice.toFixed(3), ""],
        ["IoU", m.iou.toFixed(3), ""],
        ["Precision", m.precision.toFixed(3), ""],
        ["Recall", m.recall.toFixed(3), ""],
        ["HD95", m.hd95_mm.toFixed(2), "mm"],
        ["ASSD", m.assd_mm.toFixed(2), "mm"],
        ["Predicted volume", m.pred_volume_ml.toFixed(1), "ml"],
        ["Reference volume", m.ref_volume_ml.toFixed(1), "ml"],
        ["Volume error", (m.volume_error_ml > 0 ? "+" : "") + m.volume_error_ml.toFixed(1), "ml"],
        ["Components", String(m.n_components), ""],
        ["Inference", kase.runtime_seconds.toFixed(2), "s"]
      ];
      byId("metric-note").textContent =
        "Computed in the source voxel grid after inverting preprocessing — the same mask a viewer would open.";
    } else {
      var meas = kase.measurements || {};
      pill.textContent = "No reference";
      pill.className = "pill";
      pill.style.color = "";
      pill.style.borderColor = "";
      pill.style.background = "";
      rows = [
        ["Liver volume", meas.volume_ml !== undefined ? meas.volume_ml.toFixed(1) : "—", "ml"],
        ["Liver voxels", meas.voxels !== undefined ? meas.voxels.toLocaleString() : "—", ""],
        ["Threshold", meas.threshold !== undefined ? String(meas.threshold) : "—", ""],
        ["Volume", kase.shape.join(" × "), "vox"],
        ["Spacing", kase.spacing_mm.join(" × "), "mm"],
        ["Inference", kase.runtime_seconds.toFixed(2), "s"]
      ];
      byId("metric-note").textContent =
        "Uploaded study: no expert mask exists, so overlap metrics cannot be computed. Volume comes from the returned mask.";
    }

    byId("metric-list").innerHTML = rows.map(function (r) {
      return "<dt>" + r[0] + "</dt><dd>" + r[1] + (r[2] ? ' <span class="unit">' + r[2] + "</span>" : "") + "</dd>";
    }).join("");

    if (kase.warnings && kase.warnings.length) {
      byId("metric-note").innerHTML = "<b>Warnings:</b> " + kase.warnings.join("; ");
    }
  }

  function renderAllCases() {
    byId("all-cases").innerHTML = cases.filter(function (k) { return k.hasReference; }).map(function (k) {
      var m = k.metrics, b = band(m.dice);
      return "<tr><td>" + k.case_id + '</td><td><span class="tag">' + k.split + "</span></td>" +
        '<td class="num" style="color:' + hsl("--band-" + b.key) + '">' + m.dice.toFixed(3) + "</td>" +
        '<td class="num">' + m.iou.toFixed(3) + "</td>" +
        '<td class="num">' + m.precision.toFixed(3) + "</td>" +
        '<td class="num">' + m.recall.toFixed(3) + "</td>" +
        '<td class="num">' + m.hd95_mm.toFixed(2) + " mm</td>" +
        '<td class="num">' + m.assd_mm.toFixed(2) + " mm</td>" +
        '<td class="num">' + m.pred_volume_ml.toFixed(0) + " ml</td>" +
        '<td class="num">' + k.runtime_seconds.toFixed(2) + " s</td></tr>";
    }).join("");
  }

  function renderProvenance() {
    var kase = currentCase();
    byId("prov-release").innerHTML = [
      ["Model version", kase.model_version || bundle.model_version],
      ["Checkpoint SHA-256", bundle.checkpoint_sha256.slice(0, 16) + "…"],
      ["Device", bundle.device],
      ["Exported", bundle.generated_utc],
      ["Selected study", kase.case_id + " (" + kase.split + ")"],
      ["Volume", kase.shape.join(" × ") + " voxels"],
      ["Source spacing", kase.spacing_mm.join(" × ") + " mm"]
    ].map(function (r) { return "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>"; }).join("");

    var c = bundle.config;
    byId("prov-config").innerHTML = [
      ["Orientation", "RAS canonical"],
      ["Working spacing", c.spacing_mm.join(" × ") + " mm"],
      ["HU window", c.window_hu[0] + " … " + c.window_hu[1]],
      ["Patch size", c.patch_size.join(" × ")],
      ["Feature widths", c.channels.join(" · ")],
      ["Window overlap", c.overlap + " gaussian"],
      ["Decision threshold", c.threshold],
      ["Postprocessing", "largest component · fill holes"]
    ].map(function (r) { return "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>"; }).join("");
  }

  /* ---------------- reader decision ---------------- */

  function storageKey(caseId) { return "liver3d.review." + caseId; }

  function readReview(caseId) {
    try {
      var raw = window.localStorage.getItem(storageKey(caseId));
      return raw ? JSON.parse(raw) : { status: "pending", events: [] };
    } catch (e) {
      return { status: "pending", events: [] };
    }
  }

  function writeReview(caseId, record) {
    try { window.localStorage.setItem(storageKey(caseId), JSON.stringify(record)); } catch (e) { /* private mode */ }
  }

  function renderReview() {
    var kase = currentCase();
    var record = readReview(kase.case_id);
    var pill = byId("review-state");
    var labels = { pending: "Pending", accepted: "Accepted", corrected: "Corrected", rejected: "Rejected" };
    pill.textContent = labels[record.status] || "Pending";
    pill.className = "pill" + (record.status === "pending" ? " review" : "");
    pill.style.color = ""; pill.style.background = ""; pill.style.borderColor = "";
    if (record.status === "accepted") {
      pill.style.color = hsl("--band-strong");
      pill.style.background = hsl("--band-strong", 0.12);
      pill.style.borderColor = hsl("--band-strong", 0.4);
    } else if (record.status === "rejected") {
      pill.style.color = hsl("--band-poor");
      pill.style.background = hsl("--band-poor", 0.12);
      pill.style.borderColor = hsl("--band-poor", 0.4);
    }
    document.querySelectorAll("[data-decision]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", btn.getAttribute("data-decision") === record.status ? "true" : "false");
    });

    var events = [{ when: (kase.received_at || bundle.generated_utc.slice(11, 19)), what: "job_completed", who: kase.model_version || bundle.model_version }]
      .concat(record.events || []);
    byId("audit").innerHTML = events.map(function (e) {
      return '<li><span class="when">' + e.when + '</span><span class="what">' + e.what + "</span><span>" + e.who + "</span></li>";
    }).join("");
  }

  document.querySelectorAll("[data-decision]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var kase = currentCase();
      var decision = btn.getAttribute("data-decision");
      var record = readReview(kase.case_id);
      record.status = decision;
      record.events = (record.events || []).concat([{
        when: new Date().toTimeString().slice(0, 8),
        what: "review_recorded · " + decision,
        who: "clinician"
      }]).slice(-6);
      writeReview(kase.case_id, record);
      renderReview();
      renderStudyList();
    });
  });

  /* ---------------- case switching ---------------- */

  function selectCase(index) {
    state.caseIndex = index;
    var kase = currentCase();
    var series = seriesFor(kase).values;
    var withLiver = [];
    series.forEach(function (v, i) { if (v !== null && v !== undefined && v > 0) { withLiver.push(i); } });
    state.slice = withLiver.length ? withLiver[Math.floor(withLiver.length / 2)] : Math.floor(kase.shape[2] / 2);

    var range = byId("slice");
    range.max = String(kase.shape[2] - 1);
    range.value = String(state.slice);

    syncModeButtons();
    renderStudyList();
    renderMetrics();
    renderReview();
    renderProvenance();
    if (kase.source === "sprite") {
      loadSprites(kase.case_id, function () { render(); });
    } else {
      render();
    }
    drawSurface();
  }

  /* ---------------- controls ---------------- */

  function syncModeButtons() {
    var kase = currentCase();
    var hasPred = kase.source === "array" ? !!kase.pred : true;
    document.querySelectorAll("[data-mode]").forEach(function (btn) {
      var mode = btn.getAttribute("data-mode");
      var supported = true;
      if (mode === "reference" || mode === "difference") { supported = !!kase.hasReference; }
      if (mode === "prediction" || mode === "overlay") { supported = hasPred || mode === "overlay"; }
      btn.disabled = !supported;
      btn.title = supported ? "" : "not available for this study (no reference mask)";
      if (!supported && state.mode === mode) { state.mode = "overlay"; }
    });
    document.querySelectorAll("[data-mode]").forEach(function (btn) {
      btn.setAttribute("aria-checked", btn.getAttribute("data-mode") === state.mode ? "true" : "false");
    });
    byId("legend-diff").hidden = state.mode !== "difference";
    byId("legend-overlay").hidden = state.mode === "difference";
    byId("opacity").disabled = state.mode === "difference" || state.mode === "ct";
  }

  document.querySelectorAll("[data-mode]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (btn.disabled) { return; }
      state.mode = btn.getAttribute("data-mode");
      syncModeButtons();
      render();
    });
  });

  byId("slice").addEventListener("input", function (e) {
    state.slice = parseInt(e.target.value, 10);
    render();
  });
  byId("opacity").addEventListener("input", function (e) {
    state.opacity = parseInt(e.target.value, 10) / 100;
    render();
  });
  byId("gain").addEventListener("input", function (e) {
    state.gain = parseInt(e.target.value, 10) / 100;
    render();
  });
  byId("reset-view").addEventListener("click", function () {
    state.mode = "overlay"; state.opacity = 0.38; state.gain = 1.0;
    byId("opacity").value = "38"; byId("gain").value = "100";
    setPlaying(false);
    syncModeButtons();
    render();
  });

  var playBtn = byId("play");
  function setPlaying(on) {
    state.playing = on;
    playBtn.textContent = on ? "❚❚ Pause" : "▶ Cine";
    playBtn.setAttribute("aria-pressed", on ? "true" : "false");
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    if (on) {
      state.timer = setInterval(function () {
        var kase = currentCase();
        state.slice = (state.slice + 1) % kase.shape[2];
        byId("slice").value = String(state.slice);
        render();
      }, 90);
    }
  }
  playBtn.addEventListener("click", function () { setPlaying(!state.playing); });

  byId("spark").addEventListener("click", function (e) {
    var kase = currentCase();
    var rect = e.currentTarget.getBoundingClientRect();
    var fraction = (e.clientX - rect.left) / rect.width;
    state.slice = Math.max(0, Math.min(kase.shape[2] - 1, Math.round(fraction * (kase.shape[2] - 1))));
    byId("slice").value = String(state.slice);
    render();
  });

  viewport.addEventListener("wheel", function (e) {
    e.preventDefault();
    var kase = currentCase();
    var step = e.deltaY > 0 ? 1 : -1;
    state.slice = Math.max(0, Math.min(kase.shape[2] - 1, state.slice + step));
    byId("slice").value = String(state.slice);
    render();
  }, { passive: false });

  document.addEventListener("keydown", function (e) {
    var tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") { return; }
    var kase = currentCase();
    if (e.key === "ArrowLeft") { state.slice = Math.max(0, state.slice - 1); }
    else if (e.key === "ArrowRight") { state.slice = Math.min(kase.shape[2] - 1, state.slice + 1); }
    else if (e.key === " " && tag !== "BUTTON") { setPlaying(!state.playing); e.preventDefault(); return; }
    else { return; }
    byId("slice").value = String(state.slice);
    e.preventDefault();
    render();
  });

  /* ---------------- tabs ---------------- */

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
  function showTab(tab) {
    tabs.forEach(function (other) {
      var selected = other === tab;
      other.setAttribute("aria-selected", selected ? "true" : "false");
      byId(other.getAttribute("aria-controls")).hidden = !selected;
    });
    if (tab.id === "tab-surface") { drawSurface(); }
  }
  tabs.forEach(function (tab) { tab.addEventListener("click", function () { showTab(tab); }); });

  /* ---------------- 3D surface ---------------- */

  var surface = { host: null, canvas: null, ctx: null, yaw: -0.6, pitch: 0.3, zoom: 1, w: 0, h: 0, dpr: 1 };

  function initSurface() {
    surface.host = byId("surface");
    surface.canvas = document.createElement("canvas");
    surface.canvas.style.width = "100%";
    surface.canvas.style.height = "100%";
    surface.canvas.style.display = "block";
    surface.canvas.style.cursor = "grab";
    surface.canvas.style.touchAction = "none";
    surface.host.appendChild(surface.canvas);
    surface.ctx = surface.canvas.getContext("2d");

    var dragging = false, lastX = 0, lastY = 0;
    surface.canvas.addEventListener("pointerdown", function (e) {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      surface.canvas.setPointerCapture(e.pointerId);
      surface.canvas.style.cursor = "grabbing";
    });
    surface.canvas.addEventListener("pointermove", function (e) {
      if (!dragging) { return; }
      surface.yaw += (e.clientX - lastX) * 0.01;
      surface.pitch = Math.max(-1.45, Math.min(1.45, surface.pitch + (e.clientY - lastY) * 0.01));
      lastX = e.clientX; lastY = e.clientY;
      drawSurface();
    });
    function release() { dragging = false; surface.canvas.style.cursor = "grab"; }
    surface.canvas.addEventListener("pointerup", release);
    surface.canvas.addEventListener("pointercancel", release);
    surface.canvas.addEventListener("wheel", function (e) {
      e.preventDefault();
      surface.zoom = Math.max(0.6, Math.min(3, surface.zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
      drawSurface();
    }, { passive: false });
    surface.host.addEventListener("keydown", function (e) {
      var step = 0.12;
      if (e.key === "ArrowLeft") { surface.yaw -= step; }
      else if (e.key === "ArrowRight") { surface.yaw += step; }
      else if (e.key === "ArrowUp") { surface.pitch = Math.max(-1.45, surface.pitch - step); }
      else if (e.key === "ArrowDown") { surface.pitch = Math.min(1.45, surface.pitch + step); }
      else { return; }
      e.preventDefault();
      drawSurface();
    });
    window.addEventListener("resize", function () { drawSurface(); });
  }

  function surfaceMessage(ctx, text) {
    ctx.fillStyle = hsl("--muted-foreground");
    ctx.font = "12px " + token("--sans");
    ctx.fillText(text, 16, 28);
  }

  function drawSurface() {
    if (!surface.host || byId("panel-surface").hidden) { return; }
    var kase = currentCase();
    var mesh = kase.mesh;
    var ctx = surface.ctx;
    surface.dpr = Math.min(window.devicePixelRatio || 1, 2);
    surface.w = surface.host.clientWidth;
    surface.h = surface.host.clientHeight;
    if (!surface.w || !surface.h) { return; }
    surface.canvas.width = Math.round(surface.w * surface.dpr);
    surface.canvas.height = Math.round(surface.h * surface.dpr);
    ctx.setTransform(surface.dpr, 0, 0, surface.dpr, 0, 0);
    ctx.clearRect(0, 0, surface.w, surface.h);
    if (!mesh) {
      surfaceMessage(ctx, kase.source === "array"
        ? "Surface is built by the export script; run scripts/export_viewer_data.py for this study."
        : "No surface for this study.");
      return;
    }

    var n = mesh.x.length, tris = mesh.i.length;
    var radius = 1;
    for (var v = 0; v < n; v++) {
      var d = Math.sqrt(mesh.x[v] * mesh.x[v] + mesh.y[v] * mesh.y[v] + mesh.z[v] * mesh.z[v]);
      if (d > radius) { radius = d; }
    }
    var ca = Math.cos(surface.yaw), sa = Math.sin(surface.yaw);
    var cb = Math.cos(surface.pitch), sb = Math.sin(surface.pitch);
    var scale = (Math.min(surface.w, surface.h) * 0.40 / radius) * surface.zoom;
    var cx = surface.w / 2, cy = surface.h / 2;

    var rx = new Float32Array(n), ry = new Float32Array(n), rz = new Float32Array(n);
    for (var p = 0; p < n; p++) {
      var X = mesh.x[p] * ca - mesh.y[p] * sa;
      var Y = mesh.x[p] * sa + mesh.y[p] * ca;
      var Y2 = Y * cb - mesh.z[p] * sb;
      var Z2 = Y * sb + mesh.z[p] * cb;
      rx[p] = cx + X * scale; ry[p] = cy - Z2 * scale; rz[p] = Y2;
    }
    var order = new Int32Array(tris), depth = new Float32Array(tris);
    for (var t = 0; t < tris; t++) {
      order[t] = t;
      depth[t] = (rz[mesh.i[t]] + rz[mesh.j[t]] + rz[mesh.k[t]]) / 3;
    }
    Array.prototype.sort.call(order, function (a, b) { return depth[b] - depth[a]; });

    var base = colorRgb("--pred");
    var light = [0.35, -0.78, 0.52];
    for (var s = 0; s < tris; s++) {
      var idx = order[s], i0 = mesh.i[idx], i1 = mesh.j[idx], i2 = mesh.k[idx];
      var ux = mesh.x[i1] - mesh.x[i0], uy = mesh.y[i1] - mesh.y[i0], uz = mesh.z[i1] - mesh.z[i0];
      var wx = mesh.x[i2] - mesh.x[i0], wy = mesh.y[i2] - mesh.y[i0], wz = mesh.z[i2] - mesh.z[i0];
      var nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
      var len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nx /= len; ny /= len; nz /= len;
      var NX = nx * ca - ny * sa;
      var NY = nx * sa + ny * ca;
      var NY2 = NY * cb - nz * sb;
      var NZ2 = NY * sb + nz * cb;
      var shade = 0.36 + 0.64 * Math.abs(NX * light[0] + NY2 * light[1] + NZ2 * light[2]);
      var fill = "rgb(" + Math.round(base[0] * shade) + "," + Math.round(base[1] * shade) + "," + Math.round(base[2] * shade) + ")";
      ctx.beginPath();
      ctx.moveTo(rx[i0], ry[i0]);
      ctx.lineTo(rx[i1], ry[i1]);
      ctx.lineTo(rx[i2], ry[i2]);
      ctx.closePath();
      ctx.fillStyle = fill; ctx.strokeStyle = fill; ctx.lineWidth = 0.6;
      ctx.fill(); ctx.stroke();
    }

    var barPx = 50 * scale;
    ctx.strokeStyle = hsl("--muted-foreground");
    ctx.fillStyle = hsl("--muted-foreground");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(16, surface.h - 20); ctx.lineTo(16 + barPx, surface.h - 20);
    ctx.moveTo(16, surface.h - 24); ctx.lineTo(16, surface.h - 16);
    ctx.moveTo(16 + barPx, surface.h - 24); ctx.lineTo(16 + barPx, surface.h - 16);
    ctx.stroke();
    ctx.font = "10px " + token("--mono");
    ctx.fillText("50 mm", 16, surface.h - 28);
    ctx.textAlign = "right";
    ctx.fillText(tris.toLocaleString() + " triangles · voxel " + kase.spacing_mm.join(" × ") + " mm", surface.w - 16, surface.h - 20);
    ctx.textAlign = "left";
  }

  /* ---------------- NIfTI reader ---------------- */

  var DTYPES = {
    2: { bytes: 1, read: function (dv, o) { return dv.getUint8(o); } },
    4: { bytes: 2, read: function (dv, o, le) { return dv.getInt16(o, le); } },
    8: { bytes: 4, read: function (dv, o, le) { return dv.getInt32(o, le); } },
    16: { bytes: 4, read: function (dv, o, le) { return dv.getFloat32(o, le); } },
    64: { bytes: 8, read: function (dv, o, le) { return dv.getFloat64(o, le); } },
    256: { bytes: 1, read: function (dv, o) { return dv.getInt8(o); } },
    512: { bytes: 2, read: function (dv, o, le) { return dv.getUint16(o, le); } },
    768: { bytes: 4, read: function (dv, o, le) { return dv.getUint32(o, le); } }
  };

  function parseNifti(buffer) {
    var dv = new DataView(buffer);
    var le = true;
    var sizeof = dv.getInt32(0, true);
    if (sizeof !== 348) {
      le = false;
      sizeof = dv.getInt32(0, false);
      if (sizeof !== 348) { throw new Error("not a NIfTI-1 volume (bad header size)"); }
    }
    var dims = [];
    for (var i = 0; i < 8; i++) { dims.push(dv.getInt16(40 + i * 2, le)); }
    if (dims[0] < 3) { throw new Error("expected a 3D volume, header says " + dims[0] + "D"); }
    var datatype = dv.getInt16(70, le);
    var spec = DTYPES[datatype];
    if (!spec) { throw new Error("unsupported NIfTI datatype code " + datatype); }
    var pixdim = [];
    for (var p = 0; p < 8; p++) { pixdim.push(dv.getFloat32(76 + p * 4, le)); }
    var voxOffset = Math.max(352, Math.round(dv.getFloat32(108, le)));
    var slope = dv.getFloat32(112, le) || 1;
    var inter = dv.getFloat32(116, le) || 0;

    var nx = dims[1], ny = dims[2], nz = dims[3];
    var count = nx * ny * nz;
    if (!count || count > 90e6) { throw new Error("volume too large for the browser viewer (" + count + " voxels)"); }
    var values = new Float32Array(count);
    var offset = voxOffset;
    for (var v = 0; v < count; v++) {
      values[v] = spec.read(dv, offset, le) * slope + inter;
      offset += spec.bytes;
    }
    return {
      shape: [nx, ny, nz],
      spacing: [Math.abs(pixdim[1]) || 1, Math.abs(pixdim[2]) || 1, Math.abs(pixdim[3]) || 1],
      values: values
    };
  }

  function gunzipIfNeeded(buffer, name) {
    var bytes = new Uint8Array(buffer);
    var gzipped = bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
    if (!gzipped) { return Promise.resolve(buffer); }
    if (typeof DecompressionStream === "undefined") {
      return Promise.reject(new Error("this browser cannot unzip .nii.gz - upload an uncompressed .nii"));
    }
    var stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Response(stream).arrayBuffer();
  }

  function windowToBytes(values, lo, hi) {
    var out = new Uint8Array(values.length);
    var span = Math.max(1e-6, hi - lo);
    for (var i = 0; i < values.length; i++) {
      var v = (values[i] - lo) / span;
      out[i] = v <= 0 ? 0 : (v >= 1 ? 255 : Math.round(v * 255));
    }
    return out;
  }

  /* ---------------- upload workflow ---------------- */

  var uploadFile = null;

  function log(message, kind) {
    var list = byId("upload-log");
    var li = document.createElement("li");
    li.className = kind || "";
    li.innerHTML = '<span class="when">' + new Date().toTimeString().slice(0, 8) + "</span><span>" + message + "</span>";
    list.appendChild(li);
    list.scrollTop = list.scrollHeight;
  }

  function endpoint() { return byId("api-endpoint").value.replace(/\/+$/, ""); }
  function apiKey() { return byId("api-key").value.trim(); }

  function rememberSettings() {
    try {
      window.localStorage.setItem("liver3d.endpoint", byId("api-endpoint").value);
      window.localStorage.setItem("liver3d.key", byId("api-key").value);
    } catch (e) { /* private mode */ }
  }

  function restoreSettings() {
    try {
      var e = window.localStorage.getItem("liver3d.endpoint");
      var k = window.localStorage.getItem("liver3d.key");
      if (e) { byId("api-endpoint").value = e; }
      if (k) { byId("api-key").value = k; }
    } catch (e) { /* private mode */ }
  }

  function mixedContentBlocked() {
    return window.location.protocol === "https:" && endpoint().indexOf("http://") === 0;
  }

  function setVerdict(kind, headline, detail) {
    var box = byId("verdict");
    box.hidden = false;
    box.className = "verdict " + kind;
    byId("verdict-headline").textContent = headline;
    byId("verdict-detail").innerHTML = detail || "";
  }

  function checkService() {
    if (mixedContentBlocked()) {
      log("this page is served over HTTPS, so the browser blocks calls to a local HTTP service", "err");
      setVerdict("warn", "Upload needs the local page",
        "Open <code>liver3d/viewer.html</code> from disk (or serve it over http) and the upload path works against the local service.");
      return Promise.resolve(false);
    }
    log("GET " + endpoint() + "/healthz");
    return fetch(endpoint() + "/healthz").then(function (r) { return r.json(); }).then(function (payload) {
      log("200 · status " + payload.status + " · model_loaded " + payload.model_loaded + " · queue " + payload.queue_depth, "ok");
      return true;
    }).catch(function (err) {
      log("service unreachable: " + err.message, "err");
      setVerdict("warn", "Segmentation service is not running",
        "Start it, then try again:<br><code>set LIVER3D_CHECKPOINT=outputs/demo/best.pt</code><br>" +
        "<code>set LIVER3D_API_KEYS=demo-key:clinician</code><br>" +
        "<code>uvicorn src.api.service:app --port 8000</code>");
      return false;
    });
  }

  function pollJob(jobId, tries) {
    tries = tries || 0;
    return fetch(endpoint() + "/v1/segmentations/" + jobId, { headers: { "X-API-Key": apiKey() } })
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        if (payload.status === "completed" || payload.status === "failed" || payload.status === "rejected") {
          return payload;
        }
        if (tries > 600) { throw new Error("job did not finish in time"); }
        return new Promise(function (resolve) { setTimeout(resolve, 500); }).then(function () {
          return pollJob(jobId, tries + 1);
        });
      });
  }

  function runUpload() {
    if (!uploadFile) { log("choose a .nii or .nii.gz volume first", "err"); return; }
    var btn = byId("upload-run");
    btn.disabled = true;
    btn.textContent = "Segmenting…";
    byId("verdict").hidden = true;
    rememberSettings();

    var parsed = null;
    var jobId = null;

    checkService().then(function (ok) {
      if (!ok) { throw new Error("service unavailable"); }
      log("reading " + uploadFile.name + " (" + (uploadFile.size / 1048576).toFixed(1) + " MB) in the browser");
      return uploadFile.arrayBuffer();
    }).then(function (buffer) {
      return gunzipIfNeeded(buffer, uploadFile.name);
    }).then(function (buffer) {
      parsed = parseNifti(buffer);
      log("parsed volume " + parsed.shape.join(" × ") + " · spacing " +
        parsed.spacing.map(function (s) { return s.toFixed(2); }).join(" × ") + " mm", "ok");
      var form = new FormData();
      form.append("file", uploadFile, uploadFile.name);
      log("POST " + endpoint() + "/v1/segmentations");
      return fetch(endpoint() + "/v1/segmentations", {
        method: "POST", headers: { "X-API-Key": apiKey() }, body: form
      });
    }).then(function (response) {
      return response.json().then(function (payload) { return { status: response.status, payload: payload }; });
    }).then(function (res) {
      if (res.status !== 202) {
        throw new Error(res.status + " · " + (res.payload.detail || "upload refused"));
      }
      jobId = res.payload.job_id;
      log("202 · job " + jobId + " queued", "ok");
      return pollJob(jobId);
    }).then(function (result) {
      if (result.status !== "completed") {
        log(result.status + " · " + (result.error_message || "no output"), "err");
        setVerdict("bad", result.status === "rejected" ? "Study rejected before inference" : "Inference failed",
          (result.finding || result.error_message || "") + (result.trace_id ? " (trace " + result.trace_id + ")" : ""));
        return null;
      }
      log("200 · completed in " + result.runtime_seconds + " s", "ok");
      log("GET " + result.mask_url);
      return fetch(endpoint() + result.mask_url, { headers: { "X-API-Key": apiKey() } })
        .then(function (r) { return r.arrayBuffer(); })
        .then(function (buffer) { return gunzipIfNeeded(buffer, "mask.nii.gz"); })
        .then(function (buffer) { return { result: result, mask: parseNifti(buffer) }; });
    }).then(function (payload) {
      if (!payload) { return; }
      addUploadedCase(parsed, payload.mask, payload.result);
    }).catch(function (err) {
      if (err.message !== "service unavailable") { log("failed: " + err.message, "err"); }
      if (byId("verdict").hidden) {
        setVerdict("bad", "Could not segment this study", err.message);
      }
    }).then(function () {
      btn.disabled = false;
      btn.textContent = "Segment volume";
    });
  }

  function addUploadedCase(volume, mask, result) {
    var nx = volume.shape[0], ny = volume.shape[1], nz = volume.shape[2];
    if (mask.shape.join() !== volume.shape.join()) {
      log("returned mask " + mask.shape.join(" × ") + " does not match the volume — not loaded", "err");
      setVerdict("bad", "Mask geometry does not match the study", "The exported mask must sit on the source voxel grid.");
      return;
    }
    var window_hu = bundle.config.window_hu;
    var ct = windowToBytes(volume.values, window_hu[0], window_hu[1]);
    var bits = new Uint8Array(mask.values.length);
    for (var i = 0; i < mask.values.length; i++) { bits[i] = mask.values[i] > 0.5 ? 1 : 0; }

    var voxelMl = volume.spacing[0] * volume.spacing[1] * volume.spacing[2] / 1000;
    var perSlice = [];
    for (var z = 0; z < nz; z++) {
      var count = 0, base = z * nx * ny;
      for (var p = 0; p < nx * ny; p++) { count += bits[base + p]; }
      perSlice.push(count === 0 ? null : count * voxelMl);
    }

    var kase = {
      case_id: uploadFile.name.replace(/\.nii(\.gz)?$/i, "").slice(0, 28),
      split: "uploaded",
      source: "array",
      hasReference: false,
      shape: [nx, ny, nz],
      spacing_mm: volume.spacing.map(function (s) { return Math.round(s * 1000) / 1000; }),
      ct: ct,
      pred: bits,
      per_slice_area: perSlice,
      measurements: result.measurements || {},
      metrics: null,
      mesh: null,
      warnings: result.warnings || [],
      runtime_seconds: result.runtime_seconds || 0,
      model_version: result.model_version,
      received_at: new Date().toTimeString().slice(0, 8)
    };
    cases.push(kase);
    selectCase(cases.length - 1);
    showTab(byId("tab-slices"));

    var meas = kase.measurements;
    var empty = meas.empty_prediction;
    var detail = [
      "<b>" + (meas.volume_ml !== undefined ? meas.volume_ml.toFixed(1) : "0") + " ml</b> across " +
        perSlice.filter(function (v) { return v !== null; }).length + " of " + nz + " slices",
      "model <code>" + (result.model_version || "unknown") + "</code> at threshold " + (meas.threshold !== undefined ? meas.threshold : "—"),
      "runtime " + result.runtime_seconds + " s · job <code>" + result.job_id + "</code>"
    ];
    if (kase.warnings.length) { detail.push("<b>warnings:</b> " + kase.warnings.join("; ")); }
    detail.push("Loaded into slice review. This is a proposed mask: accept, correct or reject it.");
    setVerdict(empty ? "bad" : "good",
      result.finding || (empty ? "No liver segmented" : "Liver segmented"),
      detail.join("<br>"));
    log(empty ? "no liver voxels returned — flagged for review" : "liver volume " + meas.volume_ml.toFixed(1) + " ml", empty ? "err" : "ok");
  }

  byId("upload-file").addEventListener("change", function (e) {
    uploadFile = e.target.files && e.target.files[0] ? e.target.files[0] : null;
    byId("upload-name").textContent = uploadFile ? uploadFile.name + " · " + (uploadFile.size / 1048576).toFixed(1) + " MB" : "no file chosen";
    byId("upload-run").disabled = !uploadFile;
  });
  byId("upload-run").addEventListener("click", runUpload);
  byId("upload-check").addEventListener("click", function () { checkService(); });

  var drop = byId("dropzone");
  ["dragenter", "dragover"].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) {
    var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) { return; }
    uploadFile = file;
    byId("upload-name").textContent = file.name + " · " + (file.size / 1048576).toFixed(1) + " MB";
    byId("upload-run").disabled = false;
    log("selected " + file.name);
  });

  /* ---------------- boot ---------------- */

  byId("meta-model").textContent = bundle.model_version;
  byId("meta-sha").textContent = bundle.checkpoint_sha256.slice(0, 12);
  byId("meta-device").textContent = bundle.device.toUpperCase();

  restoreSettings();
  initSurface();
  renderAllCases();
  selectCase(0);

  if (mixedContentBlocked()) {
    byId("upload-note").innerHTML =
      "This page is served over HTTPS, so the browser will block calls to a local HTTP service. " +
      "Open <code>liver3d/viewer.html</code> from disk to upload a study.";
  }

  function repaint() { rgbCache = {}; render(); drawSurface(); }
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) { mq.addEventListener("change", repaint); }
  }
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
