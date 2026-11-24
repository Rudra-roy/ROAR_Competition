"""Race logs -> docs/assets/data/*.js for the project site (docs/index.html, docs/book.html). Training-time tool.

  python tools/site_data.py

Reads the race logs listed in VERSIONS (logs/ is git-ignored, so the generated files are what is committed), the wall map
and the committed racing lines of each version (git show), and writes plain <script> data files (window.ROAR.*): they load
from file:// and from GitHub Pages alike. Everything the site states is either computed here from a log or is a literal
from the experiment log below, with its line number. Assertions stop the build if a file is not the race it claims to be.
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "competition_code")]
from envelope import station  # noqa: E402
from make_media import ANCHOR, BANDS, LABEL_OFF  # noqa: E402
from media_data import CORNERS, DATA, Race  # noqa: E402

OUT = os.path.join(ROOT, "docs", "assets", "data")
DT = 0.05
# (key, gated median, commit of the racing line, race log, what changed, gate size). The 341.20 commit exists on origin
# as 780b4bb (the same tree as the first gate's 72afe93). Each log's time equals its version's gate median exactly.
VERSIONS = [
    ("v1", 341.20, "780b4bb", "logs/final8e_2.run.csv", "early MPC with a TOPP-RA speed plan", 5),
    ("v2", 322.50, "462bc0b", "logs/ref/gk4w_1.run.csv", "learned corrections and the kink exit", 5),
    ("v3", 319.85, "c1c1009", "../slots/wt2/logs/gq2.run.csv", "MPC speed weight 2 to 128 and a per-corner re-tune", 5),
    ("v4", 319.15, "4683eb1", "logs/ref_ship/gb2.run.csv", "corner exits and a re-tune", 5),
    ("v5", 318.40, "d6f964a", "logs/capture/cap3.run.csv", "real-wall corridor map", 20),
]
# From the experiment log kept while building the entry (a local file, not published; line numbers as of 2026-11-24).
LEDGER = {
    "starter_s": 1658.90,  # results.csv row "baseline", ledger L26
    "winner_s": 319.75,  # the Summer 2026 first-place entry's code, run in our simulator (L221, README)
    "qv_sweep": [[2, 322.10], [4, 321.60], [8, 321.25], [16, 321.10], [32, 320.65], [64, 320.55], [128, 320.35],
                 [256, 320.30], [512, 320.35]],  # L275-276, one race per value
    "gated": [[383.20, "L130"], [364.50, "L138"], [360.10, "L147"], [340.60, "L156"], [341.20, "L168"],
              [340.55, "L183"], [333.90, "L197"], [331.80, "L216"], [326.75, "L243"], [322.95, "L257"],
              [322.50, "L263"], [322.35, "L270"], [319.85, "L278"], [319.15, "L289"], [318.40, "L346"]],
    "networks_off": [["all four active", 319.15], ["network 0 off", 319.60], ["network 2 off", 319.45],
                     ["network 4 off", 319.35], ["network 5 off", 319.85]],  # L294
    "probe_m": [["Final hairpin apex", 0.18], ["Lesmo 1", 0.26], ["c1329", 0.26], ["First-chicane exit", 0.26],
                ["Kink", 0.40]],  # README "The car's own poses, probed"; Bottom curve: none within 0.6 m
    "probe_out_m": [["Lesmo 1", 0.60], ["Final hairpin", 0.75]],
}
SECTORS = ["First chicane", "c1329", "Lesmo 1", "Kink", "Bottom curve", "Final hairpin"]
BUDGET = 150_000  # bytes per data file


def git_line(sha, tmp):
    p = os.path.join(tmp, sha + ".npz")
    with open(p, "wb") as f:
        f.write(subprocess.run(["git", "-C", ROOT, "show", sha + ":competition_code/data/racing_line.npz"], check=True,
                               capture_output=True).stdout)
    return p


def unwrapped_station(xy, centre, Lc):
    s, _ = station(xy, centre)
    d = np.diff(s)
    d = np.where(d < -Lc / 2, d + Lc, np.where(d > Lc / 2, d - Lc, d))
    return np.r_[0.0, np.cumsum(d)], s[0]


def cross(t, su, target):
    """Time at which the (non-decreasing) station su reaches `target`."""
    return float(np.interp(target, np.maximum.accumulate(su), t))


def wrap(assign, obj):
    head = "(window.ROAR=window.ROAR||{});"
    if "." in assign:
        head += "ROAR.%s=ROAR.%s||{};" % (assign.split(".")[0], assign.split(".")[0])
    return head + "ROAR.%s=%s;\n" % (assign, json.dumps(obj, separators=(",", ":")))


def write(name, assign, obj):
    s = wrap(assign, obj)
    assert not re.search(r"/home/|mt-labpc", s), name + " leaks a local path"
    assert len(s) < BUDGET, "%s is %d bytes (budget %d)" % (name, len(s), BUDGET)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(s)
    return len(s)


RELEASE = "17ee5c3"  # the commit whose files the book quotes (and links to)
CODE = re.compile(r"<!-- code:([^:]+):(\d+)-(\d+) -->.*?<!-- /code -->", re.S)


def fill_code(html_path):
    """Replace every <!-- code:PATH:A-B --> region with that exact line range of PATH at RELEASE, HTML-escaped."""
    import html as htmllib
    src = open(html_path).read()

    def one(m):
        path, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        body = subprocess.run(["git", "-C", ROOT, "show", "%s:%s" % (RELEASE, path)], check=True, capture_output=True,
                              text=True).stdout.split("\n")[a - 1:b]
        assert body and any(x.strip() for x in body), "%s:%d-%d is empty" % (path, a, b)
        url = "https://github.com/Rudra-roy/ROAR_Competition/blob/%s/%s#L%d-L%d" % (RELEASE, path, a, b)
        return ('<!-- code:%s:%d-%d -->\n<div class="codeblock"><div class="hd"><a href="%s" rel="noopener">%s</a> at %s, lines %d to %d</div>'
                '<pre tabindex="0"><code>%s</code></pre></div>\n<!-- /code -->' % (path, a, b, url, path, RELEASE, a, b,
                                                                              htmllib.escape("\n".join(body))))
    out = CODE.sub(one, src)
    if out != src:
        open(html_path, "w").write(out)
    return len(CODE.findall(out))


def main():
    os.makedirs(OUT, exist_ok=True)
    centre = np.load(os.path.join(DATA, "centreline.npy"))
    Lc = float(np.linalg.norm(np.roll(centre, -1, 0) - centre, axis=1).sum())
    tmp = tempfile.mkdtemp()
    lines = {v[2]: git_line(v[2], tmp) for v in VERSIONS}
    v5 = Race(os.path.join(ROOT, VERSIONS[-1][3]), line=lines[VERSIONS[-1][2]])
    bb = np.r_[v5.track.xy.min(0), v5.track.xy.max(0)]
    ox, oy = float(np.floor(bb[0] - 40)), float(np.floor(bb[1] - 40))
    dm = lambda a, o: np.round((np.asarray(a) - o) * 10).astype(int).tolist()  # noqa: E731

    # ---- corners: where each named corner is on the v5 line and on the centreline (station, mod Lc)
    mids = {k: 0.5 * (a + b) for k, (a, b) in CORNERS.items()}
    mids["Bottom curve"] = ANCHOR["Bottom curve (sec8)"]
    corner_xy, corner_st = {}, {}
    for k, ls in mids.items():
        p = np.array([[np.interp(ls, v5.track.s, v5.track.xy[:, 0]), np.interp(ls, v5.track.s, v5.track.xy[:, 1])]])
        corner_xy[k] = p[0]
        corner_st[k] = float(station(p, centre)[0][0])
    assert corner_xy["Final hairpin"][0] < corner_xy["Lesmo 1"][0], "map orientation (hairpin must be left of Lesmo 1)"

    # ---- track: walls from the wall map along the centreline normals, the v5 line, corner labels
    z = np.load(os.path.join(ROOT, "tools", "corridor_map.npz"))
    st = z["station"]
    seg = np.roll(centre, -1, 0) - centre
    el = np.linalg.norm(seg, axis=1)
    cs = np.r_[0.0, np.cumsum(el)[:-1]]
    grid = np.arange(0, Lc, 4.0)
    cx, cy = np.interp(grid, cs, centre[:, 0], period=Lc), np.interp(grid, cs, centre[:, 1], period=Lc)
    ang = np.arctan2(np.interp(grid, cs, seg[:, 1] / el, period=Lc), np.interp(grid, cs, seg[:, 0] / el, period=Lc))
    nx, ny = -np.sin(ang), np.cos(ang)
    left = np.nan_to_num(np.interp(grid, st, z["left"], period=Lc), nan=8.0, posinf=8.0).clip(0, 9)
    right = np.nan_to_num(np.interp(grid, st, z["right"], period=Lc), nan=8.0, posinf=8.0).clip(0, 9)
    lx, ly = cx + left * nx, cy + left * ny
    rx, ry = cx - right * nx, cy - right * ny
    line_pts = v5.track.xy[::4]
    track = {
        "origin": [ox, oy], "len": round(Lc, 1),
        "left": [dm(lx, ox), dm(ly, oy)], "right": [dm(rx, ox), dm(ry, oy)],
        "line": [dm(line_pts[:, 0], ox), dm(line_pts[:, 1], oy)],
        "corners": [[k, int(round((corner_xy[k][0] - ox) * 10)), int(round((corner_xy[k][1] - oy) * 10)),
                     list(LABEL_OFF.get(k if k != "Bottom curve" else "Bottom curve (sec8)", (0, 0)))] for k in
                    ["First chicane", "c1329", "Lesmo 1", "Kink", "Bottom curve", "Final hairpin"]],
    }
    write("track.js", "track", track)

    # ---- runs
    runs, tab = {}, []
    for key, median, sha, path, what, n_gate in VERSIONS:
        r = Race(os.path.join(ROOT, path), line=lines[sha])
        raw = np.genfromtxt(os.path.join(ROOT, path), delimiter=",", names=True)
        T = (raw["tick"][-1] + 1) * DT
        assert abs(T - median) < 0.006, "%s: log time %.2f != gated median %.2f" % (key, T, median)
        t = r.t
        su, s_grid = unwrapped_station(r.xy, centre, Lc)
        laps = [cross(t, su, Lc), cross(t, su, 2 * Lc), T]
        corners = []
        for k in SECTORS:
            rel = (corner_st[k] - s_grid) % Lc
            corners.append([k, [round(cross(t, su, rel + m * Lc), 2) for m in range(3)]])
        idx = np.r_[np.arange(0, len(t), 2), len(t) - 1]  # 10 Hz
        beta = np.degrees(raw["beta"][idx]) * 10 if "beta" in raw.dtype.names else None
        runs[key] = dict(label="%.2f" % median, sha=sha, what=what, gate=n_gate, T=round(T, 2), dt=0.1, n=len(idx),
                         laps=[round(x, 2) for x in laps], x=dm(r.xy[idx, 0], ox), y=dm(r.xy[idx, 1], oy),
                         s=np.round(su[idx] * 10).astype(int).tolist(),
                         v=np.round(r.v[idx] * 10).astype(int).tolist(), vref=np.round(r.v_ref[idx] * 10).astype(int).tolist(),
                         thr=np.round(raw["throttle"][idx] * 100).astype(int).tolist(),
                         brk=np.round(raw["brake"][idx] * 100).astype(int).tolist(),
                         st=np.round(raw["steer"][idx] * 100).astype(int).tolist(),
                         beta=None if beta is None else np.round(beta).astype(int).tolist(), corners=corners)
        # sectors of lap 2 (by centreline station, so versions with different lines compare)
        order = sorted(SECTORS, key=lambda k: (corner_st[k] - s_grid) % Lc)
        cst = [(corner_st[k] - s_grid) % Lc for k in order]
        bounds = [0.0] + [0.5 * (cst[i] + cst[i + 1]) for i in range(len(cst) - 1)] + [Lc]
        # a lap here starts at the start grid; the first sector wraps: start the sector list at the first boundary
        times = [cross(t, su, Lc + b) for b in bounds]
        sect = {k: round(times[i + 1] - times[i], 2) for i, k in enumerate(order)}
        assert abs(sum(sect.values()) - (laps[1] - laps[0])) < 0.05, key + ": sector sum != lap 2"
        gap = (r.v_ref - r.v)[r.seg == 2]
        tab.append(dict(key=key, label="%.2f" % median, top=round(float(r.v.max()), 1), mean=round(float(r.v.mean()), 2),
                        thr=round(float((raw["throttle"] >= 0.99).mean() * 100), 1),
                        brk=round(float((raw["brake"] > 0).mean() * 100), 1), gap=round(float(gap.mean()), 2),
                        beta=None if "beta" not in raw.dtype.names else round(float(np.degrees(np.abs(raw["beta"]).max())), 1),
                        lap2=round(laps[1] - laps[0], 2), sect=sect))
    # timing-tower consistency on the common axis (centreline station): at its own finish station, each car's time minus
    # the leader's time at that station must equal its race-time deficit, to within the finish-line slack
    lead, checks = runs["v5"], {}
    ls_t = np.arange(lead["n"]) * 0.1
    for key in ("v1", "v2", "v3", "v4"):
        r = runs[key]
        s_end = r["s"][-1] / 10.0
        deficit_at_station = r["T"] - float(np.interp(s_end * 10, lead["s"], ls_t))
        full = r["T"] - lead["T"]
        assert abs(deficit_at_station - full) < 0.25, "%s: gap on the station axis %.2f vs time deficit %.2f" % (
            key, deficit_at_station, full)
        checks[key] = round(deficit_at_station - full, 3)
    for key, r in runs.items():
        write("runs-%s.js" % key, "runs." + key, r)

    # ---- statistics
    gate = list(csv.DictReader(open(os.path.join(ROOT, "logs", "gate_g2.csv"))))
    times = np.array([float(g["sim_time_s"]) for g in gate])
    clear = np.array([float(g["clearance_m"]) for g in gate])
    assert len(gate) == 20 and abs(np.median(times) - 318.40) < 1e-6
    assert all(int(g["collisions"]) == 0 and g["status"] == "finished" for g in gate)
    assert abs(clear.min() - 0.130) < 1e-6
    results = [[r["label"], float(r["sim_time_s"]) if r["sim_time_s"] else None, int(r["collisions"] or 0), r["status"]]
               for r in csv.DictReader(open(os.path.join(ROOT, "results.csv")))]
    base = [r for r in results if r[0] == "baseline"]
    assert base and abs(base[0][1] - LEDGER["starter_s"]) < 1e-6
    # plan versus driven on one flying lap (Race.seg == 2), v5; the README states mean 3.0 and max 9.9 m/s
    m = v5.seg == 2
    o = np.argsort(v5.line_s[m])
    ls, vv, rr = v5.line_s[m][o], v5.v[m][o], v5.v_ref[m][o]
    brk = np.genfromtxt(os.path.join(ROOT, VERSIONS[-1][3]), delimiter=",", names=True)["brake"][m][o]
    gx = np.arange(0.0, v5.L, 3.0)
    gap = rr - vv
    assert round(float(gap.mean()), 1) == 3.0 and round(float(gap.max()), 1) == 9.9, "plan gap differs from the README"
    stats = dict(
        median=float(np.median(times)), tmin=float(times.min()), tmax=float(times.max()),
        collisions=0, min_clear=float(clear.min()), median_clear=float(np.median(clear)),
        clear=[[round(float(c), 3), int(g["at_line_s"])] for c, g in zip(clear, gate)],
        starter=LEDGER["starter_s"], winner=LEDGER["winner_s"], vs_winner=round(318.40 - LEDGER["winner_s"], 2),
        mean_v5=tab[-1]["mean"], mean_v1=tab[0]["mean"],
        versions=[dict(key=v[0], label="%.2f" % v[1], t=v[1], sha=v[2], what=v[4], gate=v[5]) for v in VERSIONS],
        qv=LEDGER["qv_sweep"], gated=LEDGER["gated"], networks=LEDGER["networks_off"],
        probe=LEDGER["probe_m"], probe_out=LEDGER["probe_out_m"],
        results=results, release_gate=[[g["label"], float(g["sim_time_s"])] for g in gate],
        plan=dict(x=gx.tolist(), v=np.round(np.interp(gx, ls, vv), 1).tolist(), vref=np.round(np.interp(gx, ls, rr), 1).tolist(),
                  brk=np.round(np.interp(gx, ls, brk), 2).tolist(), mean_gap=round(float(gap.mean()), 1),
                  max_gap=round(float(gap.max()), 1), max_gap_at=int(ls[np.argmax(gap)]),
                  bands={k: list(v) for k, v in BANDS.items()}),
        table=tab, sectors=SECTORS,
    )
    n = write("stats.js", "stats", stats)
    book = os.path.join(ROOT, "docs", "book.html")
    if os.path.exists(book):
        print("filled %d code excerpts in docs/book.html" % fill_code(book))
    print("wrote %d runs, track and stats (%d bytes); station length %.1f m" % (len(runs), n, Lc))
    for k, r in runs.items():
        print("  %s T=%.2f n=%d finish-gap check %s s" % (k, r["T"], r["n"], checks.get(k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
