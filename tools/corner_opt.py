"""Per-corner optimisation of the throttle floor, braking scale and corner grip, raced in parallel sim slots.

Each corner section k has knobs x_k = (thr_k: corner throttle floor, brk_k: braking scale, grip_k: a_lat). The lap
time splits into per-section times t_k (tools/tune.py's sections), and a section's knobs mostly move its own t_k, so
every race scores all sections at once. A batch = the incumbent (candidate 0, re-validated every batch) plus N-1
candidates whose sections are perturbed independently (one random knob step per section, with per-section momentum).
Update: for each section take the clean candidate with the lowest t_k (a collision in a section marks it infeasible
for that candidate; sections after an abort are unknown). The new combo becomes the incumbent only after it raced
clean as candidate 0 with a better total; otherwise the changed sections step back half way. Everything is logged to
corner_opt/<name>/history.csv and state.json; the incumbent's data files are written to corner_opt/<name>/best/.

  python tools/corner_opt.py <name> --slots 1,2,3 --batches 8 [--resume]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "competition_code", "data")
DT = 0.05
KNOBS = ("thr", "brk", "grip")
STEP0 = {"thr": 0.15, "brk": 0.08, "grip": 0.8}       # initial perturbation size
BOUNDS = {"thr": (0.0, 1.0), "brk": (0.8, 1.4), "grip": (16.0, 31.0)}
SLACK_SECTIONS = ()  # sections with no corner (straights) are detected from the profile and left alone


def sections(p):
    return [s for s, _ in p["a_lat_sections"]]


def sec_index(starts, line_s):
    i = np.searchsorted(starts, line_s, side="right") - 1
    return np.where(i < 0, len(starts) - 1, i)


def race_result(run_csv, p):
    """(t_sec array, collision section or -1, clean bool, max |d_centre|) from a run.csv (aborted runs allowed)."""
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    if d.ndim == 0 or len(d) < 100:
        return None
    jumps = np.flatnonzero(np.diff(d["s_total"]) < -50)
    starts = sections(p)
    idx = sec_index(starts, d["line_s"])
    nsec = len(starts)
    t_sec = np.bincount(idx, minlength=nsec) * DT
    coll = int(idx[jumps[0]]) if len(jumps) else -1
    return t_sec, coll, len(jumps) == 0, float(len(d) * DT)


def write_candidate(wt, p):
    """Write the candidate params into a slot checkout and rebuild its speed profile there."""
    with open(os.path.join(wt, "competition_code", "data", "vehicle_params.json"), "w") as f:
        json.dump(p, f, indent=2)
    out = subprocess.run([os.path.join(PARENT, ".venv", "bin", "python"), os.path.join(wt, "tools", "build_profile.py")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("build_profile failed in %s: %s" % (wt, out.stderr[-400:]))


def apply(p0, X):
    """Params with the knob matrix X (nsec x 3) applied."""
    p = json.loads(json.dumps(p0))
    n = len(p["a_lat_sections"])
    cf = p.setdefault("corner_throttle_floor", {"thr": 0.5, "ay_min": 8.0})
    cf["thr_sec"] = [float(np.clip(X[k, 0], *BOUNDS["thr"])) for k in range(n)]
    p["brake_sec_scale"] = [float(np.clip(X[k, 1], *BOUNDS["brk"])) for k in range(n)]
    for k in range(n):
        p["a_lat_sections"][k][1] = float(np.clip(X[k, 2], *BOUNDS["grip"]))
    return p


def knobs_of(p):
    n = len(p["a_lat_sections"])
    cf = p.get("corner_throttle_floor", {"thr": 0.5})
    thr = cf.get("thr_sec", [cf.get("thr", 0.5)] * n)
    brk = p.get("brake_sec_scale", [1.0] * n)
    return np.array([[thr[k], brk[k], p["a_lat_sections"][k][1]] for k in range(n)], float)


def corner_mask(p):
    """Sections that contain a corner (the profile is below 0.98 v_max somewhere inside)."""
    d = np.load(os.path.join(DATA, "speed_profile.npz"))
    sys.path.insert(0, os.path.join(ROOT, "competition_code"))
    from track import load_track
    L = load_track(os.path.join(DATA, "racing_line.npz")).length
    idx = sec_index(sections(p), (d["start_s"] + d["s"]) % L)
    vmin = np.array([d["v"][idx == k].min() if (idx == k).any() else 99 for k in range(len(sections(p)))])
    return vmin < 0.98 * p["v_max"]


def run_batch(cands, slots, wts, name, batch):
    """Race candidates in parallel slots (round-robin over slots); -> list of (label, result|None, status)."""
    results = [None] * len(cands)
    procs = {}
    queue = list(range(len(cands)))
    while queue or procs:
        while queue and len(procs) < len(slots):
            i = queue.pop(0)
            slot = [s for s in slots if s not in [v[1] for v in procs.values()]][0]
            wt = wts[slot]
            write_candidate(wt, cands[i])
            label = "%s_b%d_c%d" % (name, batch, i)
            pr = subprocess.Popen([os.path.join(ROOT, "tools", "eval_slot.sh"), str(slot), wt, label],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            procs[i] = (pr, slot, label, wt)
            time.sleep(20)  # stagger the server start-ups
        for i, (pr, slot, label, wt) in list(procs.items()):
            if pr.poll() is not None:
                line = (pr.stdout.read() or "").strip().splitlines()
                status = line[-1].split(",")[-1] if line else "crashed"
                run_csv = os.path.join(wt, "logs", label + ".run.csv")
                res = race_result(run_csv, cands[i]) if os.path.exists(run_csv) else None
                results[i] = (label, res, status)
                del procs[i]
        time.sleep(3)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slots", default="1,2,3")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--params", help="starting vehicle_params.json (default: the repo's)")
    a = ap.parse_args()
    slots = [int(s) for s in a.slots.split(",")]
    wts = {s: os.path.join(PARENT, "slots", "wt%d" % s) for s in slots}
    out = os.path.join(ROOT, "corner_opt", a.name)
    os.makedirs(os.path.join(out, "best"), exist_ok=True)
    state_f = os.path.join(out, "state.json")
    with open(a.params or os.path.join(DATA, "vehicle_params.json")) as f:
        p0 = json.load(f)
    if a.resume and os.path.exists(state_f):
        st = json.load(open(state_f))
        X, X_best, step, mom, best_t, batch0 = (np.array(st["X"]), np.array(st["X_best"]), st["step"], np.array(st["mom"]),
                                              st["best_t"], st["batch"] + 1)
    else:
        X = knobs_of(p0)
        X_best, step, mom, best_t, batch0 = X.copy(), {k: [STEP0[k]] * len(X) for k in KNOBS}, np.zeros_like(X), None, 0
    active = corner_mask(p0)
    nsec, ncand = len(X), len(slots)
    rng = np.random.default_rng(batch0)
    hist = open(os.path.join(out, "history.csv"), "a", newline="")
    hw = csv.writer(hist)
    if hist.tell() == 0:
        hw.writerow(["batch", "cand", "label", "status", "t_total", "coll_sec", "knobs_json", "t_sec_json"])
    print("sections active:", np.flatnonzero(active).tolist(), " incumbent knobs:\n", np.round(X, 2))
    for batch in range(batch0, batch0 + a.batches):
        cands_X = [X.copy()]
        for c in range(1, ncand):
            Xc = X.copy()
            for k in np.flatnonzero(active):
                j = rng.integers(3)
                kn = KNOBS[j]
                d = mom[k, j] if (mom[k, j] != 0 and rng.random() < 0.6) else rng.choice([-1, 1]) * step[kn][k]
                Xc[k, j] = np.clip(Xc[k, j] + d, *BOUNDS[kn])
            cands_X.append(Xc)
        cands = [apply(p0, Xc) for Xc in cands_X]
        t0 = time.time()
        res = run_batch(cands, slots, wts, a.name, batch)
        print("batch %d: %.0f s" % (batch, time.time() - t0))
        # score
        T = np.full((ncand, nsec), np.nan)
        clean = [False] * ncand
        for i, (label, r, status) in enumerate(res):
            if r is None:
                hw.writerow([batch, i, label, status, "", "", json.dumps(cands_X[i].round(3).tolist()), ""])
                print("  cand %d %s: %s (no result)" % (i, label, status))
                continue
            t_sec, coll, ok, t_total = r
            clean[i] = ok
            valid = np.ones(nsec, bool)
            if coll >= 0:
                order = (np.arange(nsec) - coll) % nsec
                valid[order > 0] = False  # sections after the collision never ran
                valid[coll] = False
            T[i, valid] = t_sec[valid]
            hw.writerow([batch, i, label, status, t_total, coll, json.dumps(cands_X[i].round(3).tolist()), json.dumps(t_sec.round(2).tolist())])
            print("  cand %d %s: %s total %.2f coll_sec %d" % (i, label, status, t_total, coll))
        hist.flush()
        # incumbent validation
        r0 = res[0][1]
        if r0 is not None and clean[0]:
            if best_t is None or r0[3] < best_t - 0.01:
                best_t, X_best = r0[3], X.copy()
                for fn in ("vehicle_params.json", "speed_profile.npz"):
                    shutil.copy(os.path.join(wts[slots[0]], "competition_code", "data", fn), os.path.join(out, "best", fn))
                print("  incumbent validated: %.2f (new best)" % r0[3])
            else:
                print("  incumbent validated: %.2f (best stays %.2f)" % (r0[3], best_t))
        elif r0 is not None:
            print("  incumbent COLLIDED in section %d: stepping its changed sections back half way" % r0[1])
            X = 0.5 * (X + X_best)
        # per-section selection among candidates with a valid, clean-in-section time
        X_new = X.copy()
        for k in np.flatnonzero(active):
            col = T[:, k]
            if np.all(np.isnan(col)):
                continue
            i_best = int(np.nanargmin(col))
            if i_best != 0 and not np.isnan(col[0]) and col[i_best] < col[0] - 0.01:
                d = cands_X[i_best][k] - X[k]
                X_new[k] = cands_X[i_best][k]
                mom[k] = d
                print("  sec %2d: cand %d better by %.2f s -> %s" % (k, i_best, col[0] - col[i_best], np.round(X_new[k], 2)))
            else:
                mom[k] = 0.0
                for kn in KNOBS:
                    step[kn][k] = max(step[kn][k] * 0.7, 0.25 * STEP0[kn])
        X = X_new
        json.dump(dict(X=X.tolist(), X_best=X_best.tolist(), step=step, mom=mom.tolist(), best_t=best_t, batch=batch),
                  open(state_f, "w"), indent=1)
    print("done. best %.2f, knobs:\n%s" % (best_t if best_t else float("nan"), np.round(X_best, 2)))


if __name__ == "__main__":
    main()
