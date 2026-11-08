"""Evolution strategies on the residual networks, every candidate scored on a FULL 3-lap race.

Why: train_es.py scores flying laps from a placed start; on the q_v 128 MPC its best policy was 0.9 s faster per two
flying laps but 0.7 s SLOWER in full races (es6, 2026-11-08). Here the score is the race itself.

Each generation races the current networks plus PAIRS antithetic perturbations (theta +- sigma eps) in parallel sim
slots (tools/eval_slot.sh, the unmodified runner). Every trained network k is graded on its own lap segment
[cuts[k], next cut) summed over the race, so one race scores all networks at once (they mostly move their own
segment). A candidate that collides, stops early, or sweeps its car body more than --body-max beyond the reference
races (tools/body_margin.py) is ranked last in every segment. Update per network: centred ranks of the segment time,
plain gradient step of --lr in units of sigma. The current networks are re-raced every generation; a new best (clean,
total time) is written to rl/<name>/best.npz. Frozen networks (not in --train) keep their shipped weights.
--elite S: a clean candidate at least S seconds faster than the current networks replaces them (a (1+lambda) step);
the gradient step on its own moved the networks only 0.3 sigma in 3 generations (res1).

  python tools/rl/race_es.py <name> --slots 1,2,3,4 --gens 12 [--pairs 3] [--sigma 0.03] [--lr 0.5]
         [--train 0,2,4,5] [--elite 0.15] [--body-ref "logs/ref/*.csv" --body-max 0.55] [--resume]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARENT = os.path.dirname(ROOT)
DATA = os.path.join(ROOT, "competition_code", "data")
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "competition_code")]
KEYS = ("W1", "b1", "W2", "b2")
DT = 0.05


def flat(res, nets):
    return np.concatenate([np.asarray(res[k][j], float).ravel() for j in nets for k in KEYS])


def unflat(theta, res, nets):
    out = {k: np.array(res[k], float, copy=True) for k in res}
    i = 0
    for j in nets:
        for k in KEYS:
            n = out[k][j].size
            out[k][j] = theta[i:i + n].reshape(out[k][j].shape)
            i += n
    return out


def segment_times(run_csv, cuts, laps_m):
    """(per-segment race time [len(cuts)], clean) from a run.csv; segment k = line_s in [cuts[k], cuts[k+1])."""
    d = np.genfromtxt(run_csv, delimiter=",", names=True)
    if d.ndim == 0 or len(d) < 100:
        return None, False
    jumps = np.flatnonzero(np.diff(d["s_total"]) < -50)
    clean = len(jumps) == 0 and d["s_total"][-1] >= laps_m - 50
    k = (np.searchsorted(cuts, d["line_s"], side="right") - 1) % len(cuts)
    return np.bincount(k, minlength=len(cuts)) * DT, clean


def centred_ranks(x):
    """Ranks scaled to [-0.5, 0.5]; higher = better (x is a cost: lower is better)."""
    r = np.argsort(np.argsort(-np.asarray(x, float)))
    return r / max(len(x) - 1, 1) - 0.5


def race_all(cands, slots, name, gen, body, cuts, laps_m):
    """Race every candidate residual dict; -> list of (seg_times|None, ok, total)."""
    out = [None] * len(cands)
    procs, queue = {}, list(range(len(cands)))
    while queue or procs:
        while queue and len(procs) < len(slots):
            i = queue.pop(0)
            slot = [s for s in slots if s not in [v[1] for v in procs.values()]][0]
            wt = os.path.join(PARENT, "slots", "wt%d" % slot)
            np.savez(os.path.join(wt, "competition_code", "data", "residual.npz"), **cands[i])
            label = "%s_g%d_c%d" % (name, gen, i)
            pr = subprocess.Popen([os.path.join(ROOT, "tools", "eval_slot.sh"), str(slot), wt, label],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            procs[i] = (pr, slot, label, wt)
            time.sleep(20)
        for i, (pr, slot, label, wt) in list(procs.items()):
            if pr.poll() is None:
                continue
            f = os.path.join(wt, "logs", label + ".run.csv")
            seg, ok = segment_times(f, cuts, laps_m) if os.path.exists(f) else (None, False)
            wide = ""
            if seg is not None and ok and body is not None:
                bm, bmax = body
                per, _ = bm.excess(f, os.path.join(wt, "competition_code", "data", "racing_line.npz"))
                if per.max() > bmax:
                    ok, wide = False, " body %.2f m" % per.max()
            total = float(seg.sum()) if seg is not None else float("nan")
            out[i] = (seg, ok, total)
            print("  %s: %s %.2f%s" % (label, "clean" if ok else "FAIL", total, wide), flush=True)
            del procs[i]
        time.sleep(3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slots", default="1,2,3,4")
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=0.5, help="step per generation, in units of sigma")
    ap.add_argument("--train", default="0,2,4,5", help="networks to train (the shipped zeroed ones stay off)")
    ap.add_argument("--body-ref")
    ap.add_argument("--body-max", type=float, default=0.55)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--elite", type=float, default=0.0, help="adopt the best clean candidate outright when it beats the "
                    "current networks by at least this many seconds (0 = gradient steps only)")
    a = ap.parse_args(argv)
    slots = [int(s) for s in a.slots.split(",")]
    nets = [int(x) for x in a.train.split(",")]
    out = os.path.join(ROOT, "rl", a.name)
    os.makedirs(out, exist_ok=True)
    res0 = dict(np.load(os.path.join(DATA, "residual.npz")))
    cuts = np.asarray(res0["cuts"], float)
    from track import load_track
    laps_m = 3 * load_track(os.path.join(DATA, "racing_line.npz")).length
    body = None
    if a.body_ref:
        from body_margin import BodyMargin
        bm = BodyMargin(sorted(glob.glob(a.body_ref)))
        bm.starts = cuts  # grade the body excess per residual segment
        body = (bm, a.body_max)
    st_f = os.path.join(out, "state.npz")
    if a.resume and os.path.exists(st_f):
        st = np.load(st_f)
        theta, g0, best = st["theta"], int(st["gen"]) + 1, float(st["best"])
        rng = np.random.default_rng(g0)
    else:
        theta, g0, best = flat(res0, nets), 0, np.inf
        rng = np.random.default_rng(0)
    sizes = [sum(np.asarray(res0[k][j]).size for k in KEYS) for j in nets]
    offs = np.r_[0, np.cumsum(sizes)]
    log = open(os.path.join(out, "log.csv"), "a")
    for g in range(g0, g0 + a.gens):
        t0 = time.time()
        eps = rng.normal(0, 1, (a.pairs, theta.size))
        thetas = [theta] + [theta + a.sigma * e for e in eps] + [theta - a.sigma * e for e in eps]
        r = race_all([unflat(t, res0, nets) for t in thetas], slots, a.name, g, body, cuts, laps_m)
        seg0, ok0, tot0 = r[0]
        if ok0 and tot0 < best:
            best = tot0
            np.savez(os.path.join(out, "best.npz"), **unflat(theta, res0, nets))
        # per-network gradient from its own segment's time; failed candidates rank last everywhere
        step = np.zeros_like(theta)
        pert = r[1:]
        for jj, j in enumerate(nets):
            cost = [s[j] if (s is not None and ok) else np.inf for s, ok, _ in pert]
            if not np.isfinite(cost).any():
                continue
            rk = centred_ranks(np.where(np.isfinite(cost), cost, 1e9))
            w = rk[:a.pairs] - rk[a.pairs:]  # antithetic: + and - of the same eps
            gk = (w[:, None] * eps[:, offs[jj]:offs[jj + 1]]).sum(0) / a.pairs
            step[offs[jj]:offs[jj + 1]] = gk
        okt = [(x[2], i) for i, x in enumerate(pert) if x[1]]
        if a.elite > 0 and ok0 and okt and min(okt)[0] <= tot0 - a.elite:
            i = min(okt)[1]
            theta = thetas[1 + i]
            print("  adopting candidate %d (%.2f vs current %.2f)" % (1 + i, pert[i][2], tot0), flush=True)
        else:
            theta = theta + a.lr * a.sigma * step
        n_ok = sum(1 for _, ok, _ in pert if ok)
        print("gen %d: current %s %.2f | best %.2f | clean %d/%d | %.0f s" % (
            g, "clean" if ok0 else "FAIL", tot0, best, n_ok, len(pert), time.time() - t0), flush=True)
        log.write(json.dumps(dict(gen=g, cur=tot0, cur_ok=bool(ok0), best=best,
                                  totals=[x[2] for x in pert], ok=[bool(x[1]) for x in pert])) + "\n")
        log.flush()
        np.savez(st_f, theta=theta, gen=g, best=best)


if __name__ == "__main__":
    main()
