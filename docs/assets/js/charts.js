/* Statistics charts: inline SVG drawn from window.ROAR.stats (written by tools/site_data.py). No chart library.
   Each chart draws in once when it enters the viewport (stroke and opacity only; instant under reduced motion). */
(function () {
  "use strict";
  var hosts = [].slice.call(document.querySelectorAll("[data-chart]"));
  if (!hosts.length) return;
  var ROAR = (window.ROAR = window.ROAR || {});
  var NS = "http://www.w3.org/2000/svg";
  var C = { ink: "#1c2228", fence: "#56606b", rule: "#dde2e7", fast: "#5b4bd0", kerb: "#c41e2a", brake: "#b8461a", band: "#9ec5f480" };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) { var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = s; return t; }
  function scale(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); }; }
  function nice(lo, hi, n) {
    var step = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(step))), m = step / p;
    var s = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p, out = [];
    for (var v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(6));
    return out;
  }
  /* frame: svg with margins, y axis grid, returns scales */
  function frame(host, o) {
    var w = o.w || 720, h = o.h || 380, m = Object.assign({ l: 52, r: 20, t: 16, b: 44 }, o.m || {});
    if (o.ylabel) m.t = Math.max(m.t, 34);
    var svg = el("svg", { viewBox: "0 0 " + w + " " + h, role: "img", "aria-label": o.label, preserveAspectRatio: "xMidYMid meet" });
    var ax = el("g", { "class": "ax" }, svg);
    var x = o.x, y = scale(o.y[0], o.y[1], h - m.b, m.t);
    nice(o.y[0], o.y[1], o.yticks || 5).forEach(function (v) {
      el("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }, ax);
      txt(ax, m.l - 8, y(v) + 4, o.yfmt ? o.yfmt(v) : v, { "text-anchor": "end" });
    });
    if (o.ylabel) txt(svg, 4, 14, o.ylabel, { "text-anchor": "start" });
    host.appendChild(svg);
    return { svg: svg, w: w, h: h, m: m, x: x, y: y, ax: ax };
  }
  function line(f, pts, attrs) {
    var d = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join("");
    var p = el("path", Object.assign({ d: d, "class": "line", fill: "none", stroke: C.ink }, attrs || {}), f.svg);
    try { p.style.setProperty("--len", Math.ceil(p.getTotalLength()) + 1); } catch (e) { p.style.setProperty("--len", 3000); }
    return p;
  }
  var tip;
  function tipFor(host) {
    if (!tip || tip.parentNode !== host) { tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip); }
    return tip;
  }
  function dot(f, host, cx, cy, r, fill, html, d, extra) {
    var c = el("circle", Object.assign({ cx: cx, cy: cy, r: r, fill: fill, "class": "dot", tabindex: 0, "aria-label": html.replace(/<[^>]+>/g, "") }, extra || {}), f.svg);
    c.style.setProperty("--d", d || 0);
    var show = function (e) {
      var t = tipFor(host); t.innerHTML = html; t.classList.add("on");
      var hb = host.getBoundingClientRect(), cb = c.getBoundingClientRect();
      var px = (e && e.clientX ? e.clientX : cb.left + cb.width / 2) - hb.left, py = cb.top - hb.top;
      t.style.left = Math.min(Math.max(px - t.offsetWidth / 2, 0), hb.width - t.offsetWidth) + "px";
      t.style.top = (py - t.offsetHeight - 10) + "px";
    };
    c.addEventListener("pointerenter", show); c.addEventListener("pointermove", show); c.addEventListener("focus", show);
    c.addEventListener("pointerleave", function () { tipFor(host).classList.remove("on"); });
    c.addEventListener("blur", function () { tipFor(host).classList.remove("on"); });
    return c;
  }
  function dataTable(host, head, rows) {
    var d = document.createElement("details"); d.className = "data";
    var h = "<summary>Data table</summary><div class='table-scroll'><table><thead><tr>" + head.map(function (x, i) { return "<th" + (i ? " class=n" : "") + ">" + x + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r) { h += "<tr>" + r.map(function (x, i) { return "<td" + (i ? " class=n" : "") + ">" + x + "</td>"; }).join("") + "</tr>"; });
    d.innerHTML = h + "</tbody></table></div>";
    host.parentNode.insertBefore(d, host.nextSibling);
  }

  /* ---------- charts ---------- */
  var charts = {};

  charts.versions = function (host, st) {
    var V = st.versions, f = frame(host, { label: "Race time by version", y: [316, 343], x: null, yfmt: function (v) { return v + " s"; }, m: { l: 62 } });
    var x = scale(0, V.length - 1, f.m.l + 40, f.w - f.m.r - 60), y = f.y;
    f.ax.appendChild(Object.assign(el("line", { x1: f.m.l, x2: f.w - f.m.r, y1: y(st.winner), y2: y(st.winner), stroke: "#56606b", "stroke-dasharray": "5 4", "stroke-width": 1 }), {}));
    txt(f.svg, f.m.l + 8, y(st.winner) + 20, "Summer 2026 winner's code, run in our simulator: " + st.winner.toFixed(2) + " s", { "text-anchor": "start" });
    line(f, V.map(function (v, i) { return [x(i), y(v.t)]; }), { "stroke-width": 2 });
    V.forEach(function (v, i) {
      var last = i === V.length - 1;
      dot(f, host, x(i), y(v.t), last ? 7 : 5, last ? C.fast : C.ink,
        "<b>" + v.key + " " + v.label + " s</b><br>" + v.what + "<br>median of " + v.gate + " races · " + v.sha, i * 140);
      var lab = txt(f.svg, x(i), y(v.t) - 14, v.label, { "text-anchor": "middle", "class": "lab", style: "fill:" + (last ? C.fast : C.ink) + ";font-weight:600" });
      lab.style.setProperty("--d", i * 140);
      txt(f.svg, x(i), f.h - 18, v.key, { "text-anchor": "middle" });
    });
    dataTable(host, ["Version", "Median (s)", "Races", "What changed", "Commit"], V.map(function (v) { return [v.key, v.label, v.gate, v.what, v.sha]; }));
  };

  charts.qv = function (host, st) {
    var Q = st.qv, f = frame(host, { label: "Race time against MPC speed weight", y: [319.9, 322.5], yticks: 5, yfmt: function (v) { return v.toFixed(1); }, ylabel: "single race, s", m: { l: 60 } });
    var x = scale(0, Q.length - 1, f.m.l + 30, f.w - f.m.r - 20), y = f.y;
    line(f, Q.map(function (q, i) { return [x(i), y(q[1])]; }));
    Q.forEach(function (q, i) {
      el("rect", { x: x(i) - 12, y: y(q[1] + 0.05), width: 24, height: y(q[1] - 0.05) - y(q[1] + 0.05), fill: "#1c222818" }, f.svg);
      dot(f, host, x(i), y(q[1]), q[0] === 128 ? 7 : 5, q[0] === 128 ? C.fast : C.ink, "<b>q_v = " + q[0] + "</b><br>" + q[1].toFixed(2) + " s, one race", i * 90);
      txt(f.svg, x(i), f.h - 18, q[0], { "text-anchor": "middle" });
    });
    el("circle", { cx: x(6), cy: y(Q[6][1]), r: 12, fill: "none", stroke: C.fast, "stroke-width": 1.5 }, f.svg);
    txt(f.svg, x(6), y(Q[6][1]) + 30, "shipped", { "text-anchor": "middle", style: "fill:" + C.fast + ";font-weight:600" });
    txt(f.svg, f.w / 2, f.h - 2, "MPC speed weight q_v (log scale). Grey bars: race-to-race noise of about 0.05 s.", { "text-anchor": "middle" });
    dataTable(host, ["q_v", "Race time (s)"], Q.map(function (q) { return [q[0], q[1].toFixed(2)]; }));
  };

  charts.results = function (host, st) {
    var R = st.results.concat(st.release_gate.map(function (g) { return [g[0], g[1], 0, "finished", 1]; }));
    var f = frame(host, { label: "Every race in our results log", y: [315, 420], yfmt: function (v) { return v; }, ylabel: "race time, s", w: 960, h: 400, m: { l: 52, b: 96 } });
    var x = scale(0, R.length - 1, f.m.l + 6, f.w - f.m.r - 6), y = f.y, y2 = function (v) { return y(Math.min(v, 418)); };
    var rail = f.h - 70;
    txt(f.svg, f.m.l, f.h - 12, "Hollow: did not finish. Cross: collision. Violet: the release gate's 20 races.", { "text-anchor": "start" });
    R.forEach(function (r, i) {
      var tm = r[1], fin = r[3] === "finished" && tm, gate = r[4] === 1, coll = r[2] > 0;
      var tip = "<b>" + r[0] + "</b><br>" + (tm ? tm.toFixed(2) + " s" : "no time") + " · " + r[3] + (coll ? " · collisions " + r[2] : "");
      if (!fin) { dot(f, host, x(i), rail - 8, 3.5, "none", tip, i * 2, { stroke: C.fence, "stroke-width": 1.2 }); return; }
      if (tm > 418) { dot(f, host, x(i), y(418), 3.5, C.ink, tip, i * 2); return; }
      if (coll) {
        var c = el("path", { d: "M" + (x(i) - 4) + " " + (y2(tm) - 4) + "l8 8m0-8l-8 8", stroke: C.kerb, "stroke-width": 2, "class": "dot", fill: "none" }, f.svg);
        c.style.setProperty("--d", i * 2); return;
      }
      dot(f, host, x(i), y2(tm), gate ? 4 : 3, gate ? C.fast : C.ink, tip, i * 2);
    });
    txt(f.svg, f.m.l, f.h - 36, "Race number in file order, oldest left. Times above 418 s sit on the top edge; the starter code took " + st.starter.toFixed(2) + " s.", { "text-anchor": "start" });
    var n = st.results.filter(function (r) { return r[3] === "finished"; }).length;
    dataTable(host, ["Races in results.csv", "Finished", "With a collision"], [[st.results.length, n, st.results.filter(function (r) { return r[2] > 0; }).length]]);
  };

  charts.sectors = function (host, st) {
    var T = st.table, S = st.sectors, h = "<div class='table-scroll'><table class='grid'><thead><tr><th>Version</th>";
    S.forEach(function (s) { h += "<th class=n>" + s + "</th>"; });
    h += "<th class=n>Lap 2</th></tr></thead><tbody>";
    var best = {}; S.forEach(function (s) { best[s] = Math.min.apply(null, T.map(function (r) { return r.sect[s]; })); });
    var bestLap = Math.min.apply(null, T.map(function (r) { return r.lap2; }));
    T.forEach(function (r, i) {
      h += "<tr><td><b>" + r.key + "</b> " + r.label + "</td>";
      S.concat(["__lap"]).forEach(function (s) {
        var v = s === "__lap" ? r.lap2 : r.sect[s], prev = i ? (s === "__lap" ? T[i - 1].lap2 : T[i - 1].sect[s]) : null;
        var b = s === "__lap" ? bestLap : best[s], cls = v === b ? "p" : prev === null ? "" : v < prev ? "g" : v > prev ? "y" : "";
        var d = prev === null ? "" : "<small>" + (v - prev >= 0 ? "+" : "−") + Math.abs(v - prev).toFixed(2) + "</small>";
        h += "<td class='cell n " + cls + "'>" + v.toFixed(2) + d + "</td>";
      });
      h += "</tr>";
    });
    host.innerHTML = h + "</tbody></table></div>";
  };

  charts.plan = function (host, st) {
    var P = st.plan, f = frame(host, { label: "Planned and driven speed over a flying lap", y: [0, 80], w: 960, h: 420, m: { l: 52, b: 120, t: 34 }, ylabel: "speed, m/s" });
    var L = P.x[P.x.length - 1], x = scale(0, L, f.m.l, f.w - f.m.r), y = f.y;
    Object.keys(P.bands).forEach(function (k, i) {
      var b = P.bands[k]; el("rect", { x: x(b[0]), y: f.m.t, width: Math.max(2, x(b[1]) - x(b[0])), height: f.h - f.m.b - f.m.t + 72, fill: "#cfd5db66" }, f.svg);
      txt(f.svg, x((b[0] + b[1]) / 2), 14 + (i % 2) * 13, k, { "text-anchor": "middle", style: "font-size:12px" });
    });
    var gap = "M" + P.x.map(function (d, i) { return x(d).toFixed(1) + " " + y(P.vref[i]).toFixed(1); }).join("L") + "L" +
      P.x.slice().reverse().map(function (d, i) { var j = P.x.length - 1 - i; return x(d).toFixed(1) + " " + y(P.v[j]).toFixed(1); }).join("L") + "Z";
    el("path", { d: gap, fill: C.band, "class": "dot" }, f.svg);
    line(f, P.x.map(function (d, i) { return [x(d), y(P.vref[i])]; }), { stroke: C.fence, "stroke-dasharray": "5 4", "stroke-width": 1.5 });
    line(f, P.x.map(function (d, i) { return [x(d), y(P.v[i])]; }), { "stroke-width": 2 });
    var by = f.h - 70, bh = 36;
    txt(f.svg, f.m.l, by - 8, "brake", { "text-anchor": "start" });
    P.x.forEach(function (d, i) { if (P.brk[i] > 0.05) el("rect", { x: x(d), y: by + bh * (1 - P.brk[i]), width: Math.max(1.5, x(3)), height: bh * P.brk[i], fill: C.brake, "class": "dot" }, f.svg); });
    nice(0, L, 5).forEach(function (v) { txt(f.svg, x(v), f.h - 14, v, { "text-anchor": "middle" }); });
    txt(f.svg, f.w - f.m.r, f.h - 2, "distance along the racing line, m", { "text-anchor": "end" });
    txt(f.svg, x(P.x[P.x.length - 1]) - 6, y(P.vref[P.vref.length - 1]) - 8, "plan (TOPP-RA)", { "text-anchor": "end" });
    txt(f.svg, x(P.x[P.x.length - 1]) - 6, y(P.v[P.v.length - 1]) + 18, "driven", { "text-anchor": "end", style: "font-weight:600;fill:" + C.ink });
    dataTable(host, ["Measure", "Value"], [["Mean gap, plan minus driven (m/s)", P.mean_gap], ["Largest gap (m/s)", P.max_gap], ["Largest gap at line distance (m)", P.max_gap_at]]);
  };

  charts.clear = function (host, st) {
    var cl = st.clear, bins = [], nb = 20;
    for (var i = 0; i < nb; i++) bins.push({ a: 0, b: 0, o: 0 });
    cl.forEach(function (c) {
      var k = Math.min(nb - 1, Math.floor(c[0] / 0.01)), loc = c[1] >= 1680 && c[1] <= 1700 ? "a" : c[1] >= 5400 && c[1] <= 5415 ? "b" : "o";
      bins[k][loc]++;
    });
    var ymax = Math.max.apply(null, bins.map(function (b) { return b.a + b.b + b.o; })) + 1;
    var f = frame(host, { label: "Closest wall approach in the 20 release-gate races", y: [0, ymax], yticks: ymax, w: 720, h: 340, m: { l: 40, b: 56 }, ylabel: "races" });
    var x = scale(0, 0.2, f.m.l, f.w - f.m.r), y = f.y, bw = x(0.01) - x(0) - 3;
    el("rect", { x: x(0), y: f.m.t, width: x(0.1) - x(0), height: f.h - f.m.b - f.m.t, fill: "#c41e2a14" }, f.svg);
    txt(f.svg, x(0.05), f.m.t + 16, "crashed cars touched here", { "text-anchor": "middle", style: "fill:" + C.kerb });
    bins.forEach(function (b, i) {
      var y0 = y(0), parts = [[b.a, "#1c2228", "Lesmo 1"], [b.b, "#56606b", "Final hairpin"], [b.o, "#9aa3ad", "elsewhere"]];
      parts.forEach(function (p) {
        if (!p[0]) return;
        var h = y(0) - y(p[0]);
        var r = el("rect", { x: x(i * 0.01) + 1.5, y: y0 - h, width: bw, height: h - 1, fill: p[1], "class": "bar" }, f.svg);
        r.style.setProperty("--d", i * 40);
        r.appendChild(Object.assign(el("title", {}), { textContent: p[2] + ": " + p[0] + " race(s) at " + (i / 100).toFixed(2) + " to " + ((i + 1) / 100).toFixed(2) + " m" }));
        y0 -= h;
      });
    });
    el("line", { x1: x(0.1), x2: x(0.1), y1: f.m.t, y2: f.h - f.m.b, stroke: C.kerb, "stroke-width": 2 }, f.svg);
    txt(f.svg, x(0.1) + 6, f.m.t + 32, "0.10 m shipping rule", { "text-anchor": "start", style: "fill:" + C.kerb + ";font-weight:600" });
    for (var v = 0; v <= 0.2001; v += 0.04) txt(f.svg, x(v), f.h - 30, v.toFixed(2), { "text-anchor": "middle" });
    txt(f.svg, f.w / 2, f.h - 8, "closest approach of the car body to a wall, m (0.01 m bins)   ■ Lesmo 1   ■ Final hairpin   ■ elsewhere", { "text-anchor": "middle" });
    dataTable(host, ["Statistic", "Value"], [["Closest of 20 races (m)", st.min_clear.toFixed(3)], ["Median (m)", st.median_clear.toFixed(3)], ["Races under 0.10 m", cl.filter(function (c) { return c[0] < 0.1; }).length]]);
  };

  charts.probe = function (host, st) {
    var rows = st.probe.map(function (p) { return [p[0], p[1], "inside, probed"]; }).concat([["Bottom curve (sec8)", 0.6, "none within 0.6 m"]]).concat(st.probe_out.map(function (p) { return [p[0] + " (outside wall)", p[1], "outside, probed"]; }));
    var f = frame(host, { label: "Static probe of the car's own poses", y: [0, 1], yticks: 1, w: 720, h: 60 + rows.length * 36, m: { l: 190, t: 10, b: 36 } });
    f.ax.innerHTML = "";
    var x = scale(0, 0.8, f.m.l, f.w - f.m.r);
    [0, 0.2, 0.4, 0.6, 0.8].forEach(function (v) { el("line", { x1: x(v), x2: x(v), y1: 6, y2: f.h - f.m.b, stroke: C.rule }, f.ax); txt(f.ax, x(v), f.h - 14, v.toFixed(1) + " m", { "text-anchor": "middle" }); });
    rows.forEach(function (r, i) {
      var yy = 14 + i * 36, none = r[2].indexOf("none") === 0, out = r[2].indexOf("outside") === 0;
      txt(f.svg, f.m.l - 10, yy + 18, r[0], { "text-anchor": "end", style: "fill:#1c2228" });
      var b = el("rect", { x: x(0), y: yy + 6, width: x(r[1]) - x(0), height: 18, fill: none ? "#cfd5db" : out ? C.fence : C.ink, "class": "bar hb" }, f.svg);
      b.style.setProperty("--d", i * 90);
      txt(f.svg, x(r[1]) + 8, yy + 20, none ? "no touch within 0.6 m →" : r[1].toFixed(2) + " m", { "text-anchor": "start", style: "font-weight:600;fill:#1c2228" });
    });
    dataTable(host, ["Where", "Closest touch (m)"], rows.map(function (r) { return [r[0], r[2].indexOf("none") === 0 ? "none within 0.6" : r[1].toFixed(2)]; }));
  };

  charts.telemetry = function (host, st) {
    var h = "<div class='table-scroll'><table><thead><tr><th>Version</th><th class=n>Race (s)</th><th class=n>Top speed (m/s)</th><th class=n>Mean speed (m/s)</th><th class=n>Full throttle</th><th class=n>Braking</th><th class=n>Plan gap, lap (m/s)</th><th class=n>Peak sideslip</th></tr></thead><tbody>";
    st.table.forEach(function (r) {
      h += "<tr><td><b>" + r.key + "</b></td><td class=n>" + r.label + "</td><td class=n>" + r.top.toFixed(1) + "</td><td class=n>" + r.mean.toFixed(2) + "</td><td class=n>" + r.thr.toFixed(1) + " %</td><td class=n>" + r.brk.toFixed(1) + " %</td><td class=n>" + r.gap.toFixed(2) + "</td><td class=n>" + (r.beta === null ? "n/a" : r.beta.toFixed(1) + "°") + "</td></tr>";
    });
    host.innerHTML = h + "</tbody></table></div>";
  };

  /* ---------- boot ---------- */
  function draw(host, st) {
    var fn = charts[host.getAttribute("data-chart")];
    if (!fn || host.__drawn) return;
    host.__drawn = true;
    try { fn(host, st); } catch (e) { console.warn("chart " + host.getAttribute("data-chart") + ": " + e.message); return; }
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !("IntersectionObserver" in window)) host.classList.add("in");
    else {
      var io = new IntersectionObserver(function (es) {
        if (es[0].isIntersecting) { host.classList.add("in"); io.disconnect(); }
      }, { threshold: 0.25 });
      io.observe(host);
    }
  }
  function boot(host) {
    ROAR.whenNear(host, function () {
      ROAR.load(host.getAttribute("data-base") ? host.getAttribute("data-base") + "stats.js" : "assets/data/stats.js", function () { draw(host, ROAR.stats); });
    }, 500);
  }
  hosts.forEach(boot);
})();
