/* Ghost race: five versions of the controller replayed together from their logged telemetry.
   Positions are each race's own line position and lateral offset (rebuilt in tools/site_data.py); ranking and gaps use the
   unwrapped centreline station, a common axis, because each version's racing line has its own length. */
(function () {
  "use strict";
  var host = document.querySelector(".player[data-player]");
  if (!host) return;
  var ROAR = (window.ROAR = window.ROAR || {});
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var KEYS = ["v1", "v2", "v3", "v4", "v5"];
  var COL = { v1: "#5f6873", v2: "#7d8691", v3: "#9da4ad", v4: "#c3c8ce", v5: "#ffffff" };
  var RAMP = ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"];
  var $ = function (s) { return host.parentNode.querySelector(s); };
  var map = $(".pmap"), cv = $(".pmap canvas"), ctx = cv.getContext("2d");
  var runs = {}, track = null, sel = {}, focus = "v5", t = 0, playing = false, speed = 8, last = 0, vis = false;
  var started = false, dpr = 1, W = 0, H = 0, view = null, stat = null, tmax = 0, mono = {};
  KEYS.forEach(function (k) { sel[k] = true; });

  /* ---------- data ---------- */
  function ready() {
    return track && KEYS.every(function (k) { return runs[k]; });
  }
  function load() {
    var base = host.getAttribute("data-base") || "assets/data/";
    ROAR.load(base + "track.js", function () { track = ROAR.track; start(); });
    ROAR.load(base + "runs-v5.js", function () {
      runs.v5 = ROAR.runs.v5; prep("v5"); start();
      ["v4", "v3", "v2", "v1"].forEach(function (k) {
        ROAR.load(base + "runs-" + k + ".js", function () { runs[k] = ROAR.runs[k]; prep(k); chipsEnable(); start(); });
      });
    });
  }
  function prep(k) {
    var r = runs[k], m = new Array(r.s.length), mx = -1e9;
    for (var i = 0; i < r.s.length; i++) { mx = Math.max(mx, r.s[i]); m[i] = mx; }
    mono[k] = m; tmax = Math.max(tmax, r.T);
  }
  function at(arr, tt, dt) { // linear interpolation in a 10 Hz array
    var f = Math.min(Math.max(tt / dt, 0), arr.length - 1), i = Math.floor(f), g = f - i;
    return i >= arr.length - 1 ? arr[arr.length - 1] : arr[i] * (1 - g) + arr[i + 1] * g;
  }
  function timeAt(k, s) { // time at which run k reaches station s (dm)
    var m = mono[k], lo = 0, hi = m.length - 1;
    if (s <= m[0]) return 0;
    if (s >= m[hi]) return runs[k].T;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (m[mid] <= s) lo = mid; else hi = mid; }
    var g = (s - m[lo]) / Math.max(m[hi] - m[lo], 1);
    return (lo + g) * runs[k].dt;
  }

  /* ---------- geometry ---------- */
  function fit() {
    var xs = track.left[0].concat(track.right[0]), ys = track.left[1].concat(track.right[1]);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var padx = Math.min(96, W * 0.2), pady = 28, k = Math.min((W - 2 * padx) / (x1 - x0), (H - 2 * pady - 26) / (y1 - y0));
    view = { k: k, ox: (W - k * (x1 - x0)) / 2 - k * x0, oy: (H - 26 - k * (y1 - y0)) / 2 + k * y1 };
  }
  var X = function (x) { return view.ox + view.k * x; }, Y = function (y) { return view.oy - view.k * y; };

  var layer = document.createElement("canvas");
  function drawStatic() {
    layer.width = cv.width; layer.height = cv.height;
    var c = layer.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = "#14171a"; c.fillRect(0, 0, W, H);
    var L = track.left, R = track.right;
    c.beginPath();
    for (var i = 0; i < L[0].length; i++) c[i ? "lineTo" : "moveTo"](X(L[0][i]), Y(L[1][i]));
    c.closePath();
    c.moveTo(X(R[0][0]), Y(R[1][0]));
    for (i = 0; i < R[0].length; i++) c.lineTo(X(R[0][i]), Y(R[1][i]));
    c.closePath();
    c.fillStyle = "#262a30"; c.fill("evenodd");
    c.strokeStyle = "#858c94"; c.lineWidth = 1;
    [L, R].forEach(function (w) {
      c.beginPath();
      for (var j = 0; j < w[0].length; j++) c[j ? "lineTo" : "moveTo"](X(w[0][j]), Y(w[1][j]));
      c.closePath(); c.stroke();
    });
    c.strokeStyle = "#3a4048"; c.lineWidth = 1; c.beginPath();
    for (i = 0; i < track.line[0].length; i++) c[i ? "lineTo" : "moveTo"](X(track.line[0][i]), Y(track.line[1][i]));
    c.closePath(); c.stroke();
    c.font = '500 12px "Saira", system-ui, sans-serif'; c.fillStyle = "#b9bfc6"; c.strokeStyle = "#858c94";
    track.corners.forEach(function (cn) {
      var px = X(cn[1]), py = Y(cn[2]), dx = cn[3][0] * 10 * view.k * 0.55, dy = -cn[3][1] * 10 * view.k * 0.55;
      if (Math.hypot(dx, dy) < 18) { dx = dx || 24; }
      var lx = px + dx, ly = py + dy;
      c.beginPath(); c.moveTo(px, py); c.lineTo(lx, ly); c.stroke();
      c.textBaseline = "middle";
      var tw = c.measureText(cn[0]).width, left = dx < 0;
      if (left && lx - 4 - tw < 6) left = false; // would leave the canvas: put the label on the other side
      if (!left && lx + 4 + tw > W - 6) left = true;
      c.textAlign = left ? "right" : "left";
      c.fillText(cn[0], lx + (left ? -4 : 4), ly);
    });
    var sb = 5000 * view.k; // 500 m in dm
    c.strokeStyle = "#858c94"; c.beginPath(); c.moveTo(W - 20 - sb, H - 18); c.lineTo(W - 20, H - 18); c.stroke();
    c.textAlign = "center"; c.fillStyle = "#858c94"; c.fillText("500 m", W - 20 - sb / 2, H - 28);
  }
  function resize() {
    if (!track) return;
    var r = map.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(280, r.width); H = Math.max(280, r.height);
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    fit(); drawStatic(); draw();
  }

  /* ---------- drawing ---------- */
  function draw() {
    if (!view || !runs[focus]) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(layer, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var fr = runs[focus], dt = fr.dt, ft = Math.min(t, fr.T);
    var vmin = Math.min.apply(null, fr.v) / 10, vmax = Math.max.apply(null, fr.v) / 10;
    var a = Math.max(0, ft - 12), step = 0.2, px = null;
    ctx.lineWidth = W < 520 ? 2 : 3; ctx.lineCap = "round";
    for (var tt = a; tt <= ft; tt += step) {
      var x = X(at(fr.x, tt, dt)), y = Y(at(fr.y, tt, dt));
      if (px) {
        var sp = at(fr.v, tt, dt) / 10, b = Math.min(4, Math.max(0, Math.floor((sp - vmin) / (vmax - vmin) * 5)));
        ctx.strokeStyle = RAMP[b]; ctx.beginPath(); ctx.moveTo(px[0], px[1]); ctx.lineTo(x, y); ctx.stroke();
      }
      px = [x, y];
    }
    var order = KEYS.filter(function (k) { return sel[k] && runs[k]; });
    var lead = leader();
    order.sort(function (p, q) { return (p === lead) - (q === lead) || KEYS.indexOf(p) - KEYS.indexOf(q); });
    order.forEach(function (k) {
      var r = runs[k], tt = Math.min(t, r.T), x = X(at(r.x, tt, r.dt)), y = Y(at(r.y, tt, r.dt));
      ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.2832);
      if (k === "v5") { ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#181b1f"; ctx.stroke(); }
      else { ctx.fillStyle = "#181b1f"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = COL[k]; ctx.stroke(); }
      if (W >= 520 || k === lead || k === focus) {
        ctx.font = '600 12px "Saira", system-ui, sans-serif'; ctx.fillStyle = "#eef0f2"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(k, x + 11, y - 8);
      }
    });
  }

  /* ---------- tower and HUD ---------- */
  var towerEl = $(".tower ol"), lapEl = $(".tower .lap"), clkEl = $(".tower .clk"), hudEl = $(".hud");
  function sNow(k) { var r = runs[k]; return at(mono[k], Math.min(t, r.T), r.dt); }
  function ranked() { // finished cars first, by finish time; the others by distance covered on the common axis
    return KEYS.filter(function (k) { return sel[k] && runs[k]; }).map(function (k) {
      return { k: k, s: sNow(k), fin: t >= runs[k].T, T: runs[k].T };
    }).sort(function (a, b) { return a.fin !== b.fin ? (a.fin ? -1 : 1) : a.fin ? a.T - b.T : b.s - a.s; });
  }
  function leader() { var r = ranked(); return r.length ? r[0].k : null; }
  function fmt(x) { return x.toFixed(2); }
  function tower() {
    var rows = ranked(), lead = rows.length ? rows[0].k : null;
    rows.forEach(function (r) { r.gap = r.k === lead ? 0 : Math.max(0, Math.min(t, r.T) - timeAt(lead, r.s)); });
    var html = "";
    rows.forEach(function (r, i) {
      var right = r.fin ? fmt(r.T) : (r.k === lead ? "Leader" : "+" + fmt(r.gap));
      html += '<li class="' + (r.k === lead ? "lead" : "") + '" style="--c:' + COL[r.k] + '"><span>' + (i + 1) +
        '</span><i class="sw"></i><span class="v">' + r.k + ' <small>' + runs[r.k].label + '</small></span><span class="gap">' + right + "</span></li>";
    });
    towerEl.innerHTML = html;
    var Lc = track.len * 10, ls = lead ? sNow(lead) : 0;
    lapEl.textContent = "Lap " + Math.min(3, Math.floor(ls / Lc) + 1) + " / 3";
    clkEl.textContent = "t " + fmt(t) + " s";
  }
  function hud() {
    var r = runs[focus]; if (!r) return;
    var tt = Math.min(t, r.T), dt = r.dt, v = at(r.v, tt, dt) / 10, vr = at(r.vref, tt, dt) / 10;
    hudEl.querySelector(".sp").innerHTML = v.toFixed(1) + "<small>m/s, plan " + vr.toFixed(1) + "</small>";
    hudEl.querySelector(".thr b").style.transform = "scaleX(" + (at(r.thr, tt, dt) / 100) + ")";
    hudEl.querySelector(".brk b").style.transform = "scaleX(" + (at(r.brk, tt, dt) / 100) + ")";
    hudEl.querySelector(".thr .val").textContent = Math.round(at(r.thr, tt, dt)) + " %";
    hudEl.querySelector(".brk .val").textContent = Math.round(at(r.brk, tt, dt)) + " %";
    var st = at(r.st, tt, dt) / 100;
    hudEl.querySelector(".str b").style.transform = "scaleX(" + Math.min(1, Math.abs(st) / 0.35) + ")";
    hudEl.querySelector(".str b").style.marginLeft = st < 0 ? "0" : "";
    hudEl.querySelector(".str .val").textContent = (st >= 0 ? "+" : "") + st.toFixed(2);
    hudEl.querySelector(".slp .val").textContent = r.beta ? (at(r.beta, tt, dt) / 10).toFixed(1) + "°" : "n/a";
    hudEl.querySelector(".who").textContent = focus + " " + r.label + " s: " + r.what;
  }

  /* ---------- controls ---------- */
  var scrub = $(".scrub input"), playBtn = $(".play"), speedSel = $(".speed"), focusSel = $(".focus"), tickBox = $(".scrub .ticks");
  var ICON_PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2v12l10-6z" fill="currentColor"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 2h4v12H3zM9 2h4v12H9z" fill="currentColor"/></svg>';
  function lapOf(tt) {
    var l = runs.v5 ? runs.v5.laps : [0, 0, 0]; return tt < l[0] ? 1 : tt < l[1] ? 2 : 3;
  }
  function sync(frame) {
    scrub.value = t;
    scrub.setAttribute("aria-valuetext", fmt(t) + " s, lap " + lapOf(t));
    tower(); hud(); if (frame !== false) draw();
  }
  function play(on) {
    playing = on; playBtn.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute("aria-label", on ? "Pause" : "Play"); playBtn.setAttribute("aria-pressed", on ? "true" : "false");
    if (on) { if (t >= tmax) t = 0; last = 0; requestAnimationFrame(loop); }
  }
  var lastTower = 0;
  function loop(ts) {
    if (!playing) return;
    if (!vis) { last = 0; requestAnimationFrame(loop); return; }
    if (last) t = Math.min(tmax, t + (ts - last) / 1000 * speed);
    last = ts;
    draw();
    if (ts - lastTower > 90 || t >= tmax) { lastTower = ts; scrub.value = t; tower(); hud(); }
    if (t >= tmax) { play(false); sync(); return; }
    requestAnimationFrame(loop);
  }
  function seek(x) { t = Math.min(tmax, Math.max(0, x)); sync(); }
  function ticks() {
    var r = runs[focus]; if (!r) return;
    var h = "", L = r.laps;
    [L[0], L[1]].forEach(function (x) { h += '<i class="lap" style="left:' + (x / tmax * 100) + '%"></i>'; });
    r.corners.forEach(function (c) { c[1].forEach(function (x) { h += '<i style="left:' + (x / tmax * 100) + '%"></i>'; }); });
    tickBox.innerHTML = h;
  }
  function chipsEnable() {
    host.parentNode.querySelectorAll(".chips input").forEach(function (i) { if (runs[i.value]) i.disabled = false; });
    host.parentNode.querySelectorAll(".chips label small").forEach(function (s) {
      var k = s.getAttribute("data-k"); if (runs[k]) s.textContent = runs[k].label;
    });
  }
  function bind() {
    scrub.max = tmax; scrub.step = 0.05;
    scrub.addEventListener("input", function () { seek(parseFloat(scrub.value)); });
    playBtn.addEventListener("click", function () { play(!playing); });
    $(".restart").addEventListener("click", function () { seek(0); });
    speedSel.addEventListener("change", function () { speed = parseFloat(speedSel.value); });
    focusSel.addEventListener("change", function () { focus = focusSel.value; ticks(); sync(); });
    host.parentNode.querySelectorAll(".chips input").forEach(function (i) {
      i.addEventListener("change", function () {
        sel[i.value] = i.checked;
        if (!KEYS.some(function (k) { return sel[k]; })) { sel[i.value] = true; i.checked = true; }
        sync();
      });
    });
    host.parentNode.addEventListener("keydown", function (e) {
      if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type === "range" && /Arrow/.test(e.key)) return;
      var k = e.key;
      if (k === " " || k === "k") { if (e.target.tagName === "BUTTON") return; e.preventDefault(); play(!playing); }
      else if (k === "ArrowRight") { e.preventDefault(); seek(t + (e.shiftKey ? 30 : 5)); }
      else if (k === "ArrowLeft") { e.preventDefault(); seek(t - (e.shiftKey ? 30 : 5)); }
      else if (k === "Home") { e.preventDefault(); seek(0); }
      else if (k === "End") { e.preventDefault(); seek(tmax); }
      else if (/^[1-5]$/.test(k)) { var i = host.parentNode.querySelector('.chips input[value="v' + k + '"]'); if (i && !i.disabled) { i.checked = !i.checked; i.dispatchEvent(new Event("change")); } }
    });
    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(map); else window.addEventListener("resize", resize);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        vis = es[0].isIntersecting;
        if (vis && !started && !reduce && ready()) { started = true; play(true); }
        if (!vis && playing) { playing = false; play(false); }
      }, { threshold: 0.5 }).observe(host);
    } else vis = true;
  }
  var bound = false;
  function start() {
    if (!track || !runs.v5) return;
    var loadEl = $(".loading"); if (loadEl) loadEl.remove();
    if (!bound) { bound = true; bind(); }
    resize(); ticks(); sync(); buildTable();
    if (vis && !started && !reduce && ready()) { started = true; play(true); }
  }
  function buildTable() {
    var el = $(".lapstable"); if (!el || !ready()) return;
    var h = "<table><thead><tr><th>Version</th><th class=n>Lap 1</th><th class=n>Lap 2</th><th class=n>Lap 3</th><th class=n>Race</th></tr></thead><tbody>";
    KEYS.forEach(function (k) {
      var r = runs[k], L = r.laps;
      h += "<tr><td>" + k + " (" + r.label + " s)</td><td class=n>" + fmt(L[0]) + "</td><td class=n>" + fmt(L[1] - L[0]) + "</td><td class=n>" +
        fmt(L[2] - L[1]) + "</td><td class=n>" + fmt(r.T) + "</td></tr>";
    });
    el.innerHTML = h + "</tbody></table>";
  }
  ROAR.player = { seek: seek, play: play };
  ROAR.whenNear(host, load, 900);
})();
