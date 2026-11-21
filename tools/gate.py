"""Parallel release gate: N full races of the COMMITTED setup, wall clearance measured on every race.

  python tools/gate.py LABEL --n 20 [--slots 1,2,3,4] [--windowed 2] [--walls tools/corridor_map.npz --wall-min 0.10]

Why it exists: the earlier gates were five races from scratch scripts, and a gate once raced the wrong files. Here
  * the repo's competition_code/ must be clean and committed; every slot worktree (../slots/wtK) is checked out at
    that exact commit (detached) and its data files are compared with the repo's, byte for byte, before any race;
  * every race uses the unmodified runner through tools/eval_slot.sh (which stops at the first collision);
  * each finished race is scored against the wall map (tools/body_margin.WallMargin on the committed line): the
    closest the car body came to a wall, and where;
  * --windowed K opens the server window for the first K races (the organizers run with the viewer on; every gate
    until now was offscreen). Needs a display.

PASS = every race finished with 0 collisions AND the closest wall approach over all races >= --wall-min.
Writes logs/gate_<LABEL>.csv (label, slot, windowed, sim_time, collisions, status, clearance_m, at_line_s).
"""
import argparse
import csv
import filecmp
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "competition_code")]
DATA = os.path.join("competition_code", "data")


def verdict(rows, wall_min, need_walls=True):
    """rows: dicts(status, coll, clearance). -> (passed, summary text)."""
    bad = [r for r in rows if r["status"] != "finished" or int(r["coll"]) != 0]
    clr = [r["clearance"] for r in rows if r["clearance"] is not None and np.isfinite(r["clearance"])]
    worst = min(clr) if clr else float("nan")
    missing = need_walls and len(clr) < len(rows) - len(bad)
    ok = not bad and not missing and (not need_walls or worst >= wall_min)
    return ok, "%d/%d finished clean; closest wall approach %.2f m (need >= %.2f)" % (
        len(rows) - len(bad), len(rows), worst, wall_min)


def sha_of(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def prepare_slots(slots):
    """Check every slot out at the repo's HEAD and prove its race files equal the repo's. -> HEAD sha."""
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "competition_code"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("competition_code/ has uncommitted changes: commit the setup first, the gate races a commit\n" + dirty)
    sha = sha_of(ROOT)
    for s in slots:
        wt = os.path.join(PARENT, "slots", "wt%d" % s)
        subprocess.run(["git", "-C", wt, "checkout", "-q", "-f", "--detach", sha], check=True)
        if sha_of(wt) != sha:
            sys.exit("slot %d is not at %s" % (s, sha[:8]))
        for dp, _, files in os.walk(os.path.join(ROOT, DATA)):
            for f in files:
                a = os.path.join(dp, f)
                b = os.path.join(wt, os.path.relpath(a, ROOT))
                if not os.path.exists(b) or not filecmp.cmp(a, b, shallow=False):
                    sys.exit("slot %d: %s differs from the repo's" % (s, os.path.relpath(a, ROOT)))
    return sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--slots", default="1,2,3,4")
    ap.add_argument("--windowed", type=int, default=0)
    ap.add_argument("--walls")
    ap.add_argument("--wall-min", type=float, default=0.10)
    a = ap.parse_args(argv)
    slots = [int(s) for s in a.slots.split(",")]
    sha = prepare_slots(slots)
    print("gate %s: %d races from %s on slots %s (%d windowed)" % (a.label, a.n, sha[:8], slots, a.windowed), flush=True)
    wm = None
    if a.walls:
        from body_margin import WallMargin
        wm = WallMargin(a.walls, a.wall_min)
    line = os.path.join(ROOT, DATA, "racing_line.npz")
    rows, procs, queue = [], {}, list(range(a.n))
    while queue or procs:
        while queue and len(procs) < len(slots):
            i = queue.pop(0)
            slot = [s for s in slots if s not in [v[1] for v in procs.values()]][0]
            wt = os.path.join(PARENT, "slots", "wt%d" % slot)
            label = "%s_%02d" % (a.label, i + 1)
            env = dict(os.environ)
            if i < a.windowed:
                env["ROAR_RENDER"] = ""
            pr = subprocess.Popen([os.path.join(ROOT, "tools", "eval_slot.sh"), str(slot), wt, label], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            procs[i] = (pr, slot, label, wt, i < a.windowed)
            time.sleep(20)
        for i, (pr, slot, label, wt, win) in list(procs.items()):
            if pr.poll() is None:
                continue
            out = (pr.stdout.read() or "").strip().splitlines()
            f = out[-1].split(",") if out else [label, "", "0", "crashed"]
            sim, coll, status = f[1], int(f[2] or 0), f[3]
            clr, at = None, None
            run_csv = os.path.join(wt, "logs", label + ".run.csv")
            if wm is not None and status == "finished" and os.path.exists(run_csv):
                per, at = wm.excess(run_csv, line)
                clr = a.wall_min - float(per.max())
            rows.append(dict(label=label, slot=slot, windowed=win, sim=sim, coll=coll, status=status, clearance=clr,
                             at=at))
            print("  %s slot %d%s: %s s, %d collisions, %s%s" % (
                label, slot, " (window)" if win else "", sim or "-", coll, status,
                ", closest wall %.2f m @ line_s %.0f" % (clr, at) if clr is not None else ""), flush=True)
            del procs[i]
        time.sleep(3)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "gate_%s.csv" % a.label), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "slot", "windowed", "sim_time_s", "collisions", "status", "clearance_m", "at_line_s"])
        for r in sorted(rows, key=lambda r: r["label"]):
            w.writerow([r["label"], r["slot"], int(r["windowed"]), r["sim"], r["coll"], r["status"],
                        "" if r["clearance"] is None else "%.3f" % r["clearance"],
                        "" if r["at"] is None else "%.0f" % r["at"]])
    times = np.array([float(r["sim"]) for r in rows if r["status"] == "finished" and r["sim"]])
    ok, text = verdict(rows, a.wall_min, need_walls=wm is not None)
    if len(times):
        text += "; time median %.2f min %.2f max %.2f s" % (np.median(times), times.min(), times.max())
    print("GATE %s @ %s: %s" % ("PASS" if ok else "FAIL", sha[:8], text), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
