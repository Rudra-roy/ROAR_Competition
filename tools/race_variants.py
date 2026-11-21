"""Race setups (data directories) in parallel sim slots and score each one's wall clearance.

  python tools/race_variants.py DIR [DIR ...] [--slots 1,2,3,4] [--walls tools/corridor_map.npz --wall-min 0.10]

Each DIR holds any of racing_line.npz, line_offset.npz, vehicle_params.json, residual.npz (what it lacks comes from the
repo's committed competition_code/data). Per variant, in a free slot: the slot worktree is reset to the repo's HEAD,
the files are copied over it, the speed profile is rebuilt there, one race runs through tools/eval_slot.sh (stops at
the first collision). Prints label, sim time, collisions, status and the closest wall approach of the car body.
For exploring: a setup that will be submitted goes through tools/gate.py from a commit instead.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "competition_code")]
FILES = ("racing_line.npz", "line_offset.npz", "vehicle_params.json", "residual.npz")


def stage(wt, d, sha):
    subprocess.run(["git", "-C", wt, "checkout", "-q", "-f", "--detach", sha], check=True)
    data = os.path.join(wt, "competition_code", "data")
    for f in FILES:
        if os.path.exists(os.path.join(d, f)):
            shutil.copy(os.path.join(d, f), data)
    out = subprocess.run([os.path.join(PARENT, ".venv", "bin", "python"), os.path.join(wt, "tools", "build_profile.py")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("build_profile failed for %s: %s" % (d, out.stderr[-300:]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--slots", default="1,2,3,4")
    ap.add_argument("--walls")
    ap.add_argument("--wall-min", type=float, default=0.10)
    a = ap.parse_args(argv)
    slots = [int(s) for s in a.slots.split(",")]
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    wm = None
    if a.walls:
        from body_margin import WallMargin
        wm = WallMargin(a.walls, a.wall_min)
    queue, procs = list(a.dirs), {}
    while queue or procs:
        while queue and len(procs) < len(slots):
            d = queue.pop(0).rstrip("/")
            slot = [s for s in slots if s not in [v[1] for v in procs.values()]][0]
            wt = os.path.join(PARENT, "slots", "wt%d" % slot)
            stage(wt, d, sha)
            label = os.path.basename(d)
            procs[d] = (subprocess.Popen([os.path.join(ROOT, "tools", "eval_slot.sh"), str(slot), wt, label],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True), slot, label, wt)
            time.sleep(20)
        for d, (pr, slot, label, wt) in list(procs.items()):
            if pr.poll() is None:
                continue
            out = (pr.stdout.read() or "").strip().splitlines()
            f = out[-1].split(",") if out else [label, "", "0", "crashed"]
            extra = ""
            run_csv = os.path.join(wt, "logs", label + ".run.csv")
            if wm is not None and os.path.exists(run_csv):
                line = os.path.join(d, "racing_line.npz")
                per, at = wm.excess(run_csv, line if os.path.exists(line) else None)
                extra = "  closest wall %+.2f m @ line_s %.0f" % (a.wall_min - float(per.max()), at)
            print("%-14s %7s s  coll %s  %s%s" % (label, f[1] or "-", f[2], f[3], extra), flush=True)
            del procs[d]
        time.sleep(3)


if __name__ == "__main__":
    main()
