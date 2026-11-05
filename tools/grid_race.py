"""Race a list of parameter variants in parallel sim slots (tools/eval_slot.sh via corner_opt.run_batch).

  python tools/grid_race.py <name> <base vehicle_params.json> "k=v k=v" "k=v" ... [--slots 1,2,3,4]

Keys: any top-level numeric param (brake_scale, brake_ellipse_c, speed_scale, ...), secN (a_lat), brkN (brake scale
of section N), ctfN (corner throttle floor of section N), mpc.KEY. "" = the base itself. Prints one row per variant
(time, status, collision section) and writes corner_opt/<name>/grid.csv.
"""
import csv
import json
import os
import sys

import corner_opt as co


def apply_overrides(p0, spec):
    p = json.loads(json.dumps(p0))
    n = len(p["a_lat_sections"])
    for kv in spec.split():
        k, v = kv.split("=")
        if k.startswith("sec"):
            p["a_lat_sections"][int(k[3:])][1] = float(v)
        elif k.startswith("brk"):
            p.setdefault("brake_sec_scale", [1.0] * n)[int(k[3:])] = float(v)
        elif k.startswith("ctf"):
            cf = p.setdefault("corner_throttle_floor", {"thr": 0.5, "ay_min": 8.0})
            cf.setdefault("thr_sec", [cf["thr"]] * n)[int(k[3:])] = float(v)
        elif k.startswith("mpc."):
            p["mpc"][k[4:]] = float(v)
        else:
            p[k] = float(v)
    return p


def main(argv):
    slots = [1, 2, 3, 4]
    if "--slots" in argv:
        i = argv.index("--slots")
        slots = [int(s) for s in argv[i + 1].split(",")]
        argv = argv[:i] + argv[i + 2:]
    name, base, specs = argv[0], argv[1], argv[2:]
    p0 = json.load(open(base))
    cands = [apply_overrides(p0, s) for s in specs]
    wts = {s: os.path.join(co.PARENT, "slots", "wt%d" % s) for s in slots}
    out = os.path.join(co.ROOT, "corner_opt", name)
    os.makedirs(out, exist_ok=True)
    res = co.run_batch(cands, slots, wts, name, 0)
    with open(os.path.join(out, "grid.csv"), "a", newline="") as f:
        w = csv.writer(f)
        for spec, (label, r, status) in zip(specs, res):
            t = "%.2f" % r[3] if r else ""
            coll = r[1] if r else ""
            w.writerow([label, spec, status, t, coll])
            print("%-14s %-8s %7s  coll_sec %-3s  %s" % (label, status, t, coll, spec or "(base)"))


if __name__ == "__main__":
    main(sys.argv[1:])
