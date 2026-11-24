"""README figures from one logged race (dark panels that read on GitHub's light and dark themes). Training-time tool.

  python tools/make_media.py RUN.run.csv OUTDIR [--only track,speed,progress,pipeline,replay]

Palette and sizes follow the README design plan: one slate panel colour with a baked 1 px edge, a single-hue blue speed
ramp, orange for the brake channel, violet only for our best time (never in a figure that also has blue). The ramp and
the blue/orange pair passed the dataviz validator against the panel colour (see the plan); violet/blue fails it, hence
the rule. Text is Noto Sans SemiCondensed when installed, DejaVu Sans otherwise. Needs ImageMagick (`convert`) for the edge.

The GIFs are encoded with ffmpeg from frames (docs/img/):
  lap_replay.gif   python tools/make_media.py RUN.csv OUT --only replay --printed 318.40, then
                   ffmpeg -framerate 12 -i OUT/replay_frames/f_%04d.png -vf "split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 lap_replay.gif
  sim_timelapse.gif  every 4th chase frame of tools/capture.py --every 0.5 (one per 2 s of simulated time), numbered
                   h_0000.png ..., then ffmpeg -framerate 12 -i h_%04d.png -vf "scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 sim_timelapse.gif
  (dithering makes photographic frames about 3x larger: 17 MB at 640 px / 128 colours / bayer dither.)
Stills: tools/pick_stills.py names the frames; convert FRAME.png -strip -colors 256 PNG8:OUT.png.
"""
import argparse
import os
import subprocess
import sys

import matplotlib
import matplotlib.transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from media_data import DATA, Race  # noqa: E402

PANEL, EDGE = "#181b1f", "#30363d"
INK1, INK2, INK3 = "#eef0f2", "#b9bfc6", "#858c94"
GRID, AXIS, ROAD = "#2a2f35", "#3a4048", "#262a30"
SPEED_RAMP = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DRIVEN, BRAKE, BEST = "#3987e5", "#d95926", "#9085e9"
DPI = 200
CMAP = LinearSegmentedColormap.from_list("speed", SPEED_RAMP)

# Corners as the README names them: line_s windows (m) of the racing line. Apex windows are from the wall probes
# (README "Real walls"); the bottom curve is tuning section 8 (vehicle_params.json a_lat_sections).
BANDS = {
    "First chicane": (880.0, 1040.0),
    "c1329": (1325.0, 1350.0),
    "Lesmo 1": (1670.0, 1712.0),
    "Kink": (2670.0, 2725.0),
    "Bottom curve (sec8)": (3840.6, 4678.5),
    "Final hairpin": (5390.0, 5460.0),
}


def _font():
    base = "/usr/share/fonts/truetype/noto/NotoSans-SemiCondensed"
    for suffix in ("", "Medium"):
        if os.path.exists(base + suffix + ".ttf"):
            font_manager.fontManager.addfont(base + suffix + ".ttf")
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Noto Sans" if "Noto Sans" in names else "DejaVu Sans"


def _panel(w_px, h_px):
    fig = plt.figure(figsize=(w_px / DPI, h_px / DPI), dpi=DPI, facecolor=PANEL)
    return fig


def _save(fig, path):
    fig.savefig(path, dpi=DPI, facecolor=PANEL)
    plt.close(fig)
    # a 1 px edge of GitHub's own dark border colour keeps the panel visible on the dark theme (size unchanged)
    subprocess.run(["convert", path, "-shave", "1x1", "-bordercolor", EDGE, "-border", "1", "-colors", "256", path],
                   check=True)


def _title(fig, title, sub, x_px=50):
    W, H = fig.get_size_inches() * DPI
    fig.text(x_px / W, 1 - 34 / H, title, color=INK1, fontsize=13, fontweight="medium", va="top")
    fig.text(x_px / W, 1 - 34 / H - 44 / H * 1.0, sub, color=INK2, fontsize=10, va="top")


def _flying(r):
    """Ticks of one whole pass of the racing line (the middle one): a flying lap, line_s 0 to the line's length."""
    return np.where(r.seg == 2)[0]


def _segments(xy):
    pts = xy.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def _track_band(ax, lw):
    c = np.load(os.path.join(DATA, "centreline.npy"))
    c = np.vstack([c, c[:1]])
    ax.plot(c[:, 0], c[:, 1], color=ROAD, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=1)


def _at(r, s):
    return np.array([np.interp(s, r.track.s, r.track.xy[:, 0]), np.interp(s, r.track.s, r.track.xy[:, 1])])


ANCHOR = {"Bottom curve (sec8)": 3937.0}
START_S = 5501.0  # line_s of the start grid (run.csv row 0)  # where a corner's leader line lands (line_s), default the band's middle
# label anchor offsets in metres (dx, dy) from the corner centre on the racing line; tuned on the rendered figure
LABEL_OFF = {
    "First chicane": (-100, 170),
    "c1329": (-200, 120),
    "Lesmo 1": (120, -40),
    "Kink": (190, 60),
    "Bottom curve (sec8)": (0, -150),
    "Final hairpin": (-230, -100),
}


def track_speed(r, out):
    i = _flying(r)
    xy, v = r.xy[i], r.v[i]
    W, H = 1200, 1760
    fig = _panel(W, H)
    _title(fig, "A flying lap, coloured by speed",
           "Driven path of one whole lap of the line; corners are named as in the text.")
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.axis("off")
    _track_band(ax, 6.0)
    lc = LineCollection(_segments(xy), cmap=CMAP, linewidth=1.5, capstyle="round", zorder=3)
    lc.set_array(0.5 * (v[:-1] + v[1:]))
    lc.set_clim(v.min(), v.max())
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.12)
    for name, (a, b) in BANDS.items():
        p = _at(r, ANCHOR.get(name, 0.5 * (a + b)))
        dx, dy = LABEL_OFF[name]
        q = p + np.array([dx, dy])
        ax.annotate(name, xy=p, xytext=q, color=INK1, fontsize=9.5,
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"), va="center",
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.6, shrinkA=2, shrinkB=2), zorder=5)
    p0 = _at(r, START_S)
    ax.annotate("start / finish", xy=p0, xytext=p0 + np.array([200, -60]), color=INK1, fontsize=9.5, va="center",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.6, shrinkA=2, shrinkB=2), zorder=5)
    # colour key
    kax = fig.add_axes([0.05, 0.055, 0.26, 0.008])
    kax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CMAP)
    kax.set_yticks([])
    kax.set_xticks([0, 127, 255])
    kax.set_xticklabels(["%.0f" % v.min(), "%.0f" % (0.5 * (v.min() + v.max())), "%.0f" % v.max()],
                        color=INK3, fontsize=9)
    kax.tick_params(length=0)
    for s in kax.spines.values():
        s.set_visible(False)
    fig.text(0.05, 0.085, "speed (m/s)", color=INK3, fontsize=9, va="bottom")
    # scale bar, 500 m
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    sb = [xmax - 0.04 * (xmax - xmin) - 500, xmax - 0.04 * (xmax - xmin)]
    ax.plot(sb, [ymin + 0.03 * (ymax - ymin)] * 2, color=INK3, lw=1.0, solid_capstyle="butt")
    ax.text(np.mean(sb), ymin + 0.045 * (ymax - ymin), "500 m", color=INK3, fontsize=9, ha="center", va="bottom")
    _save(fig, out)


def speed_trace(r, out):
    i = _flying(r)
    s, v, vr = r.line_s[i], r.v[i], r.v_ref[i]
    t = np.genfromtxt(r.csv, delimiter=",", names=True)
    brake = t["brake"][i]
    W, H = 1760, 880
    fig = _panel(W, H)
    gap = vr - v
    _title(fig, "Planned and driven speed over a flying lap",
           "The car brakes where the plan does and runs below it on the way out of corners: mean gap %.1f m/s, "
           "largest %.1f m/s." % (gap.mean(), gap.max()))
    gs = fig.add_gridspec(2, 1, left=0.075, right=0.80, top=0.76, bottom=0.12, height_ratios=[4.6, 1.0], hspace=0.12)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    for ax in (a1, a2):
        ax.set_facecolor(PANEL)
        for k in ("top", "right", "left", "bottom"):
            ax.spines[k].set_visible(False)
        ax.tick_params(length=0, colors=INK3, labelsize=9)
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        ax.set_xlim(0, r.L)
    o = np.argsort(s)
    for name, (a, b) in BANDS.items():
        for ax in (a1, a2):
            ax.axvspan(a, b, color=ROAD, lw=0, zorder=0)
        a1.text(0.5 * (a + b), 1.12 if name == "c1329" else 1.03, name, color=INK3, fontsize=8.5, ha="center",
                va="bottom", transform=a1.get_xaxis_transform())
    a1.plot(s[o], vr[o], color=INK2, lw=0.65, zorder=3)
    a1.plot(s[o], v[o], color=DRIVEN, lw=0.8, zorder=4)
    a1.set_ylim(0, 80)
    a1.set_yticks([0, 20, 40, 60, 80])
    a1.set_xticklabels([])
    a1.set_ylabel("speed (m/s)", color=INK3, fontsize=9)
    a2.fill_between(s[o], 0, brake[o], color=BRAKE, alpha=0.12, lw=0)
    a2.plot(s[o], brake[o], color=BRAKE, lw=0.8)
    a2.set_ylim(0, 1.05)
    a2.set_yticks([0, 1])
    a2.set_ylabel("brake", color=INK3, fontsize=9)
    a2.set_xticks(np.arange(0, 5001, 1000))
    a2.set_xlabel("distance along the line (m)", color=INK3, fontsize=9)
    # direct labels at the right end
    a1.text(1.01, np.interp(r.L - 1, s[o], vr[o]) / 80 + 0.02, "plan (TOPP-RA)", color=INK2, fontsize=9.5,
            transform=a1.transAxes, va="center")
    a1.text(1.01, np.interp(r.L - 1, s[o], v[o]) / 80 - 0.04, "driven", color=INK1, fontsize=9.5,
            transform=a1.transAxes, va="center")
    _save(fig, out)


def progress(out):
    pts = [("341.20", 341.20), ("322.50", 322.50), ("319.85", 319.85), ("319.15", 319.15), ("318.40", 318.40)]
    notes = {0: "early MPC with\nTOPP-RA plan", 1: "learned\ncorrections,\nkink exit",
             2: "MPC speed weight\nq_v 2 to 128,\nper-corner re-tune", 3: "corner exits,\nre-tune",
             4: "real-wall\ncorridor map"}
    W, H = 1760, 720
    fig = _panel(W, H)
    _title(fig, "Race time across versions", "Median of 5 gate races per version (20 for the last). Lower is faster.")
    ax = fig.add_axes([0.07, 0.30, 0.62, 0.48])
    ax.set_facecolor(PANEL)
    for k in ("top", "right", "left", "bottom"):
        ax.spines[k].set_visible(False)
    ax.tick_params(length=0, colors=INK3, labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    xs = np.arange(len(pts))
    ys = np.array([p[1] for p in pts])
    ax.plot(xs, ys, color=INK3, lw=0.8, zorder=3)
    ax.scatter(xs[:-1], ys[:-1], s=22, color=INK3, edgecolor=PANEL, linewidth=1.0, zorder=4)
    ax.scatter(xs[-1:], ys[-1:], s=38, color=BEST, edgecolor=PANEL, linewidth=1.0, zorder=5)
    ax.axhline(319.75, color=INK2, lw=0.5, zorder=2)
    ax.set_ylim(314, 344)
    ax.set_yticks(range(315, 345, 5))
    ax.set_xlim(-0.4, len(pts) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([notes[k] for k in xs], color=INK3, fontsize=8.5)
    ax.set_ylabel("race time, 3 laps (s)", color=INK3, fontsize=9)
    for k, (lab, y) in enumerate(pts):
        last = k == len(pts) - 1
        if last:  # below its point: above it the label would sit on the benchmark line
            ax.text(k, y - 1.2, lab + " s", color=INK1, fontsize=9.5, ha="center", va="top")
        else:
            ax.text(k if k == 0 else k + 0.07, y + 1.3, lab, color=INK2, fontsize=9.5,
                    ha="center" if k == 0 else "left", va="bottom")
    ax.text(1.02, 319.75, "Summer 2026 first-place entry,\nrun in our simulator: 319.75 s", color=INK2, fontsize=9.5,
            va="center", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    _save(fig, out)


def pipeline(out):
    """How the pieces fit: offline tools build the inputs, the controller runs them every tick. Drawn, not Mermaid, so it
    renders in every viewer (a Mermaid block did not render for the user)."""
    from matplotlib.patches import FancyBboxPatch
    W, H = 1760, 600
    fig = _panel(W, H)
    _title(fig, "How the pieces fit", "Offline tools build the inputs; the controller runs them every tick.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # pixel coordinates, y down
    ax.axis("off")
    BW, BH = 206, 84
    col = lambda c: 130 + (c - 1) * 240  # noqa: E731  box centre x for column c
    top, bot = 225, 470
    boxes = {
        "W": (col(1), top, "Wall map", "ray casts + probes"),
        "A": (col(2), top, "Racing line", "mincurv_line.py"),
        "B": (col(3), top, "TOPP-RA speed plan", "build_profile.py"),
        "I": (col(4.2), top, "Learned line offset", "ilc.py"),
        "P": (col(5.2), top, "Per-corner tuning", "corner_opt.py"),
        "R": (col(6.2), top, "Section networks", "six, tools/rl/"),
        "L": (col(3.6), bot, "Line + offset", "committed data"),
        "M": (col(5.2), bot, "Slip-aware MPC", "OSQP, 60 steps, 3 s"),
        "N": (col(6.2), bot, "Corrections", "from the networks"),
        "O": (col(7.2), bot, "Steering,\nthrottle, brake", ""),
    }
    for y, label in ((top, "Offline, in tools/"), (bot, "Every tick, in competition_code/")):
        ax.add_patch(FancyBboxPatch((24, y - 66), W - 48, 132, boxstyle="round,pad=0,rounding_size=10", fc="none",
                                    ec=AXIS, lw=0.6))
        ax.text(40, y - 78, label, color=INK2, fontsize=9, va="bottom")
    for k, (x, y, a_, b_) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=8",
                                    fc="#20242a", ec=AXIS, lw=0.8, zorder=3))
        ax.text(x, y - (11 if b_ else 0), a_, color=INK1, fontsize=7.5, ha="center", va="center", zorder=4,
                linespacing=1.25)
        if b_:
            ax.text(x, y + 17, b_, color=INK3, fontsize=7, ha="center", va="center", zorder=4)

    def arrow(a_, b_, side="r"):
        (x0, y0), (x1, y1) = boxes[a_][:2], boxes[b_][:2]
        if side == "r":  # box to box on the same row
            p0, p1 = (x0 + BW / 2, y0), (x1 - BW / 2, y1)
        else:  # top row down to the bottom row
            p0, p1 = (x0, y0 + BH / 2), (x1 - (30 if x1 > x0 + 5 else -30 if x1 < x0 - 5 else 0), y1 - BH / 2)
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=INK3, lw=0.9, shrinkA=0, shrinkB=2,
                                                          mutation_scale=11), zorder=2)

    arrow("W", "A"); arrow("A", "B")
    arrow("B", "L", "d"); arrow("I", "L", "d"); arrow("P", "M", "d"); arrow("R", "N", "d")
    arrow("L", "M"); arrow("M", "N"); arrow("N", "O")
    _save(fig, out)


def _replay_frames(r, outdir, printed=None, nframes=216, hold=24):
    os.makedirs(outdir, exist_ok=True)
    W, H = 520, 680
    t_end = r.total_time
    tgrid = np.linspace(0, t_end, nframes)
    ticks = np.minimum((tgrid / 0.05).astype(int), len(r.t) - 1)
    vmin, vmax = float(r.v.min()), float(r.v.max())
    for n, k in enumerate(list(ticks) + [ticks[-1]] * hold):
        fig = _panel(W * 2, H * 2)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
        ax.set_facecolor(PANEL)
        ax.set_aspect("equal")
        ax.axis("off")
        _track_band(ax, 3.0)
        lo = max(0, k - int(10 / 0.05))
        seg = r.xy[lo:k + 1]
        if len(seg) > 1:
            lc = LineCollection(_segments(seg), cmap=CMAP, linewidth=1.4, capstyle="round", zorder=3)
            lc.set_array(0.5 * (r.v[lo:k][:len(seg) - 1] + r.v[lo + 1:k + 1][:len(seg) - 1]))
            lc.set_clim(vmin, vmax)
            ax.add_collection(lc)
        ax.plot(*r.xy[k], "o", color=INK1, ms=4.0, mec=PANEL, mew=1.0, zorder=6)
        ax.set_xlim(r.xy[:, 0].min() - 80, r.xy[:, 0].max() + 80)
        ax.set_ylim(r.xy[:, 1].min() - 80, r.xy[:, 1].max() + 80)
        fin = n >= nframes
        rows = [("lap", "%d of 3" % r.lap[k]), ("time", "%.1f s" % r.t[k]), ("speed", "%.1f m/s" % r.v[k])]
        if fin:
            rows[1] = ("finished", "%.2f s" % (printed or t_end))  # the runner's printed time, not the last logged tick
        for j, (lab, val) in enumerate(rows):
            fig.text(0.05, 0.955 - 0.032 * j, lab, color=INK3, fontsize=7.5, va="top")
            fig.text(0.28, 0.955 - 0.032 * j, val, color=INK1, fontsize=7.5, va="top", ha="right")
        fig.savefig(os.path.join(outdir, "f_%04d.png" % n), dpi=DPI // 2, facecolor=PANEL)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("outdir")
    ap.add_argument("--only", default="track,speed,progress")
    ap.add_argument("--printed", type=float, help="the time the runner printed for this race (s), shown on the replay's last frame")
    a = ap.parse_args(argv)
    _font()
    os.makedirs(a.outdir, exist_ok=True)
    r = Race(a.run)
    r.csv = a.run
    only = a.only.split(",")
    if "track" in only:
        track_speed(r, os.path.join(a.outdir, "track_speed.png"))
    if "speed" in only:
        speed_trace(r, os.path.join(a.outdir, "speed_trace.png"))
    if "progress" in only:
        progress(os.path.join(a.outdir, "progress.png"))
    if "pipeline" in only:
        pipeline(os.path.join(a.outdir, "pipeline.png"))
    if "replay" in only:
        _replay_frames(r, os.path.join(a.outdir, "replay_frames"), a.printed)


if __name__ == "__main__":
    main()
