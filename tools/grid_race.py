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
        if k == "ppuse":
            on = {int(x) for x in v.split(".")}
            p["pursuit"]["use_sec"] = [1 if j in on else 0 for j in range(n)]
            continue
        if k.startswith("sec"):
            p["a_lat_sections"][int(k[3:])][1] = float(v)
        elif k.startswith("brk"):
            p.setdefault("brake_sec_scale", [1.0] * n)[int(k[3:])] = float(v)
        elif k.startswith("ctf"):
            cf = p.setdefault("corner_throttle_floor", {"thr": 0.5, "ay_min": 8.0})
            cf.setdefault("thr_sec", [cf["thr"]] * n)[int(k[3:])] = float(v)
        elif k.startswith("mpc."):
            p["mpc"][k[4:]] = float(v)
        elif k.startswith("tc."):  # traction control: tc.slip_on=8 tc.slip_full=16 tc.t_min=0.4
            p.setdefault("traction_control", {"slip_on": 8.0, "slip_full": 16.0, "t_min": 0.4})[k[3:]] = float(v)
        elif k == "pp.on":  # hybrid pure-pursuit steering with the winner's lookahead table (metres)
            p["pursuit"] = {"la_v": [0.0, 25.0, 30.6, 36.1, 44.4, 50.0, 55.6, 69.4, 83.3],
                            "la_m": [12.0, 19.0, 23.2, 29.6, 38.0, 46.5, 54.9, 63.4, 73.9],
                            "gain_sec": [1.0] * n, "la_sec": [1.0] * n}
        elif k == "ppuse":  # ppuse=2.4.5.7.8.10: sections that steer with pursuit (others keep the MPC)
            on = {int(x) for x in v.split(".")}
            p["pursuit"]["use_sec"] = [1 if j in on else 0 for j in range(n)]
            continue
        elif k.startswith("ppla"):  # ppla=0.8: scale every lookahead
            p["pursuit"]["la_m"] = [x * float(v) for x in p["pursuit"]["la_m"]]
        elif k.startswith("ppg"):  # ppgN=1.1: steering gain of section N; ppg=1.1 all sections
            if k == "ppg":
                p["pursuit"]["gain_sec"] = [float(v)] * n
            else:
                p["pursuit"]["gain_sec"][int(k[3:])] = float(v)
        elif k.startswith("pb."):  # pedal_binary option: pb.dwell=3, pb.ay_min=8 (pb.on=1 with defaults)
            pb = p.setdefault("pedal_binary", {})
            if k != "pb.on":
                pb[k[3:]] = float(v)
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
