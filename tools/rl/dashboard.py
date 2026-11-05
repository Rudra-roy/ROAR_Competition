"""Live training dashboard for tools/rl/train_es.py runs.

  python tools/rl/dashboard.py es3            (refreshes every 10 s; close the window to quit)

Panels: lap-time gain of the current policy vs the plain controller per generation; crash rate of the trial variants
against its target; adaptive penalty and exploration size; the current best policy's behaviour by slip angle
(throttle correction and steering correction, evaluated on a clean reference race: a policy that "knows when to stop"
shows the throttle correction falling as slip rises); gain per corner for the latest generation.
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "competition_code"))
from submission import RESIDUAL_AHEAD_M, ResidualPolicy  # noqa: E402
from track import load_track  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "es3"
RUN = os.path.join(ROOT, "rl", NAME)
SEG_NAMES = ["back str.", "hairpin", "chicane", "1329", "1697", "kink", "bottom"]
SLIP_BINS = [(0, 5), (5, 10), (10, 15), (15, 40)]
_track = load_track(os.path.join(ROOT, "competition_code", "data", "racing_line.npz"))
_ref = np.genfromtxt(os.path.join(ROOT, "competition_code", "data", "rl_reference.run.csv"), delimiter=",", names=True)


def read_log():
    rows = [r for r in csv.reader(open(os.path.join(RUN, "log.csv"))) if r and r[0] != "base"]
    out = []
    for r in rows:
        if len(r) < 9:
            continue
        g = int(r[0])
        gain = float(r[4]) if r[4] not in ("", "nan") else np.nan
        out.append(dict(g=g, gain=gain, coll=int(r[2]), crash=float(r[5]), pen=float(r[6]), mat=int(r[7]),
                        sigma=float(r[8]) if len(r) > 8 else np.nan,
                        times=json.loads(r[3]), ref=json.loads(r[9]) if len(r) > 9 else None))
    return out


def slip_response():
    f = os.path.join(RUN, "best.npz")
    if not os.path.exists(f):
        return None
    pol, tr, d = ResidualPolicy(f), _track, _ref
    L = tr.length
    m = (d["s_total"] > L) & (d["s_total"] < 2 * L)
    X = []
    for i in np.flatnonzero(m)[::2]:
        ks = [np.interp((d["line_s"][i] + a) % L, tr.s, tr.kappa, period=L) for a in RESIDUAL_AHEAD_M]
        X.append((abs(np.degrees(d["beta"][i])), abs(np.interp(d["line_s"][i], tr.s, tr.kappa, period=L)) > 0.004,
                  *pol(ResidualPolicy.features(d["v"][i], d["beta"][i], d["r"][i], d["e_y"][i] - d["d_ref"][i],
                                               d["e_psi"][i], ks, d["a_cmd"][i], d["kappa_cmd"][i]))))
    X = np.array(X)
    res = []
    for lo, hi in SLIP_BINS:
        k = (X[:, 1] > 0) & (X[:, 0] >= lo) & (X[:, 0] < hi)
        res.append((X[k, 3].mean() if k.any() else np.nan, np.abs(X[k, 2]).mean() if k.any() else np.nan))
    return np.array(res)


def draw(fig, axes):
    rows = read_log()
    for ax in axes.ravel():
        ax.clear()
    if not rows:
        axes[0, 0].text(0.5, 0.5, "waiting for the first generation...", ha="center")
        return
    g = np.array([r["g"] for r in rows])
    a = axes[0, 0]
    gain = np.array([r["gain"] for r in rows])
    a.plot(g, gain, "o-", color="#1f6fb4")
    a.axhline(0, color="#888", lw=0.8)
    a.set_title("lap-time gain of the policy vs plain controller (s, per 2 laps)")
    a.set_xlabel("generation")
    best = np.nanmax(gain) if np.isfinite(gain).any() else np.nan
    a.text(0.02, 0.92, "best clean gain %+.2f s   latest %+.2f s" % (best, gain[-1]), transform=a.transAxes)
    a = axes[0, 1]
    a.bar(g, [r["crash"] for r in rows], color="#d9822b", alpha=0.8, label="trial crash rate")
    tgt = [0.30 + (0.10 - 0.30) * min(r["mat"] / 30, 1) for r in rows]
    a.plot(g, tgt, "k--", lw=1, label="target")
    a.set_ylim(0, 1)
    a.set_title("crash rate of trial variants")
    a.legend(loc="upper right")
    a = axes[1, 0]
    a.plot(g, [r["pen"] for r in rows], "o-", color="#b4321f", label="crash penalty (s)")
    a2 = a.twinx()
    a2.plot(g, [r["sigma"] for r in rows], "s--", color="#555", label="exploration size")
    a.set_title("adaptive penalty (red) and exploration size (grey)")
    a.set_xlabel("generation")
    a = axes[1, 1]
    sr = slip_response()
    if sr is not None:
        x = np.arange(len(SLIP_BINS))
        a.bar(x - 0.2, sr[:, 0], 0.4, color="#2b9348", label="throttle corr. (m/s^2)")
        a3 = a.twinx()
        a3.bar(x + 0.2, sr[:, 1] * 1000, 0.4, color="#7b2cbf", label="|steer corr.| (mrad)")
        a.set_xticks(x)
        a.set_xticklabels(["%d-%d deg" % b for b in SLIP_BINS])
        a.axhline(0, color="#888", lw=0.8)
        a.set_title("best policy by slip angle: green should FALL as slip rises")
        a.legend(loc="upper left")
        a3.legend(loc="upper right")
    a = axes[2, 0]
    last = rows[-1]
    if last["ref"] is not None and not last["coll"]:
        t, ref = np.array(last["times"], float), np.array(last["ref"], float)
        n = len(SEG_NAMES)
        per = (ref - t).reshape(-1, n).mean(0) if len(t) % n == 0 else ref - t
        a.bar(range(len(per)), per, color=["#2b9348" if x >= 0 else "#b4321f" for x in per])
        a.set_xticks(range(len(per)))
        a.set_xticklabels(SEG_NAMES[:len(per)], rotation=20)
        a.axhline(0, color="#888", lw=0.8)
        a.set_title("gen %d: time gained per corner (s per lap, green = faster)" % last["g"])
    a = axes[2, 1]
    a.axis("off")
    a.text(0.02, 0.9, "run %s   generation %d   maturity %d" % (NAME, last["g"], last["mat"]), fontsize=12)
    a.text(0.02, 0.7, "policy crashes this gen: %d   trial crash rate %.0f %%" % (last["coll"], 100 * last["crash"]),
           fontsize=11)
    a.text(0.02, 0.5, "penalty %.1f s   exploration %.3f" % (last["pen"], last["sigma"]), fontsize=11)
    a.text(0.02, 0.3, "best clean gain so far %+.2f s per 2 laps (%+.2f s per lap)" % (best, best / 2), fontsize=11)
    fig.tight_layout()


def main():
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.canvas.manager.set_window_title("ROAR RL training: %s" % NAME)
    while plt.fignum_exists(fig.number):
        try:
            draw(fig, axes)
        except Exception as e:  # a half-written log line: try again next refresh
            print("refresh failed: %r" % e)
        fig.canvas.draw_idle()
        plt.pause(10)


if __name__ == "__main__":
    main()
