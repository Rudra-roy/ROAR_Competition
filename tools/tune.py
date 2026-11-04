"""Tuning driver: edit vehicle_params.json -> rebuild profile -> commit -> tools/eval.sh -> summarise.

  python tools/tune.py <label> <runs> [speed_scale=0.9] [brake_scale=1.1] [sec7=30.1] [--dry] [--keep-going]

secN=value sets a_lat_sections[N]; mpc.key=value sets vehicle_params["mpc"][key]. A profile rebuild runs when brake_scale, brake_ellipse_c, brake_model or a section changes; non-numeric values stay strings. Each run is a
separate `eval.sh 1 <label>` with ROAR_LOG=1; its run.csv is kept as logs/<label>_<i>.run.csv and summarised
(max|e_y|, top speed, crash line_s, per-section sim time vs predicted). --dry edits + rebuilds + predicts only.
Stops after the first unclean run unless --keep-going.
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_code", "data")
PARAMS = os.path.join(DATA, "vehicle_params.json")
PROFILE = os.path.join(DATA, "speed_profile.npz")
DT = 0.05


def git(*a):
    return subprocess.run(["git", "-C", ROOT] + list(a), check=True, capture_output=True, text=True).stdout.strip()


def sections(p):
    return np.array([s for s, _ in p["a_lat_sections"]])


def sec_index(starts, line_s):
    i = np.searchsorted(starts, line_s, side="right") - 1
    return np.where(i < 0, len(starts) - 1, i)  # before the first start = wrapped last section


def predicted(p):
    """Total and per-section sim time of the scaled profile (constant accel between grid points)."""
    d = np.load(PROFILE)
    s, v = d["s"], p["speed_scale"] * d["v"]
    dt = 2 * np.diff(s) / (v[:-1] + v[1:])
    sys.path.insert(0, os.path.join(ROOT, "competition_code"))
    from track import load_track
    L = load_track(os.path.join(DATA, "racing_line.npz")).length
    idx = sec_index(sections(p), (d["start_s"] + 0.5 * (s[:-1] + s[1:])) % L)
    return dt.sum(), np.bincount(idx, dt, minlength=len(p["a_lat_sections"]))


def summarise(run_csv, p):
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    jumps = np.flatnonzero(np.diff(d["s_total"]) < -50)  # respawn at the start
    idx = sec_index(sections(p), d["line_s"])
    t_sec = np.bincount(idx, minlength=len(p["a_lat_sections"])) * DT
    # car offset from the centreline (walls ~4.15-4.7 m for the car centre; the line itself is within 4.0)
    sys.path[:0] = [os.path.join(ROOT, "competition_code"), os.path.join(ROOT, "tools")]
    from export_line import offsets
    from track import load_track
    tr = load_track(os.path.join(DATA, "racing_line.npz"))
    ls = d["line_s"][::2]
    psi = np.interp(ls, tr.s, np.unwrap(tr.psi), period=tr.length)
    xy = np.c_[np.interp(ls, tr.s, tr.xy[:, 0], period=tr.length), np.interp(ls, tr.s, tr.xy[:, 1], period=tr.length)]
    xy += d["e_y"][::2, None] * np.c_[-np.sin(psi), np.cos(psi)]
    dc = np.abs(offsets(xy, np.load(os.path.join(DATA, "centreline.npy")))[0])
    return dict(max_dc=float(dc.max()), dc_at=float(ls[np.argmax(dc)]),max_ey=float(np.abs(d["e_y"]).max()), ey_at=float(d["line_s"][np.argmax(np.abs(d["e_y"]))]),
                vmax=float(d["v"].max()), crashes=[round(float(d["line_s"][j]), 1) for j in jumps],
                t_sec=t_sec, t_total=len(d) * DT)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry, keep_going = "--dry" in sys.argv, "--keep-going" in sys.argv
    label, runs, sets = args[0], int(args[1]), args[2:]
    with open(PARAMS) as f:
        p = json.load(f)
    rebuild = False
    for kv in sets:
        k, val = kv.split("=")
        if k.startswith("sec"):
            p["a_lat_sections"][int(k[3:])][1] = float(val)
            rebuild = True
        elif k.startswith("ctf"):  # ctfN=0.7: corner throttle floor of section N (corner_throttle_floor.thr_sec)
            cf = p["corner_throttle_floor"]
            ts = cf.setdefault("thr_sec", [cf["thr"]] * len(p["a_lat_sections"]))
            ts[int(k[3:])] = float(val)
        elif k.startswith("brk"):  # brkN=1.3: braking scale of section N (brake_sec_scale)
            bss = p.setdefault("brake_sec_scale", [1.0] * len(p["a_lat_sections"]))
            bss[int(k[3:])] = float(val)
            rebuild = True
        elif k.startswith("pursuit."):  # pursuit.la0 / pursuit.la_k; pursuit.off=1 removes it
            if k == "pursuit.off":
                p.pop("pursuit", None)
            else:
                p.setdefault("pursuit", {})[k[8:]] = float(val)
        elif k.startswith("mpc."):
            p["mpc"][k[4:]] = float(val)
        else:
            try:
                p[k] = float(val)
            except ValueError:  # e.g. brake_model=pedal_map
                p[k] = val
            rebuild |= k in ("brake_scale", "brake_ellipse_c", "brake_model")
    with open(PARAMS, "w") as f:
        json.dump(p, f, indent=2)
    if rebuild:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_profile.py")], check=True,
                             capture_output=True, text=True).stdout
        print("\n".join(l for l in out.splitlines() if l.startswith(("lap", "total", "top"))))
    tp, tsec_p = predicted(p)
    print("%s: predicted %.2f s at speed_scale %.2f; per section %s" % (label, tp, p["speed_scale"],
                                                                       np.round(tsec_p, 1).tolist()))
    if dry:
        return
    if git("status", "--porcelain", PARAMS, PROFILE):
        git("add", PARAMS, PROFILE)
        git("commit", "-m", "tune: %s (%s)\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
            % (label, " ".join(sets)))
    env = dict(os.environ, ROAR_LOG="1")
    for i in range(1, runs + 1):
        for _ in range(2):  # one retry when the sim server hangs (client never connects): not a car result
            for f in ("run.csv", "hud.csv"):
                if os.path.exists(os.path.join(ROOT, "logs", f)):
                    os.remove(os.path.join(ROOT, "logs", f))
            out = subprocess.run([os.path.join(ROOT, "tools", "eval.sh"), "1", label], env=env, capture_output=True,
                                 text=True).stdout.strip()
            with open(os.path.join(ROOT, "results.csv")) as f:
                row = list(csv.reader(f))[-1]
            if row[5] not in ("crashed_client", "server_hang"):
                break
            print("infra failure (%s), retrying" % row[5], flush=True)
        if not os.path.exists(os.path.join(ROOT, "logs", "run.csv")):
            print("%s run %d: %s, no run.csv" % (label, i, row[5]), flush=True)
            break
        run_csv = os.path.join(ROOT, "logs", "%s_%d.run.csv" % (label, i))
        shutil.copy(os.path.join(ROOT, "logs", "run.csv"), run_csv)
        hud_csv = os.path.join(ROOT, "logs", "hud.csv")  # ROAR_HUD_LOG=1: keep the HUD log next to the run log
        if os.path.exists(hud_csv):
            shutil.copy(hud_csv, run_csv.replace(".run.csv", ".hud.csv"))
        m = summarise(run_csv, p)
        line = ("%s run %d [%s]: %s sim=%s coll=%s | max|e_y| %.2f @ %.0f, max|d_centre| %.2f @ %.0f, vmax %.1f, crashes at %s | "
                "sec time %s (pred %s)" % (label, i, row[0], row[5], row[3] or "NA", row[4], m["max_ey"],
                                           m["ey_at"], m["max_dc"], m["dc_at"], m["vmax"], m["crashes"], np.round(m["t_sec"], 1).tolist(),
                                           np.round(tsec_p, 1).tolist()))
        print(out.splitlines()[-1] if out else "", "\n" + line, flush=True)
        with open(os.path.join(ROOT, "logs", "tune_summary.txt"), "a") as f:
            f.write(line + "\n")
        if (row[5] != "finished" or row[4] != "0") and not keep_going:
            break  # step already failed: the next run would not change the verdict


if __name__ == "__main__":
    main()
