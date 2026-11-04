"""Evolution-strategies training of the residual policy (competition_code ResidualPolicy) on parallel sim slots.

  python tools/rl/train_es.py <name> --slots 1,2,3,4 --gens 60 [--pop 8] [--sigma 0.1] [--lr 0.05] [--resume]

The lap is cut into SEGMENTS, each from 120 m before one corner's braking zone to the same point before the next:
braking in, the corner, and the whole straight after it, so a slow exit costs time exactly as in a race; the segment
times add up to about one lap. Fitness of a weight vector = -(sum of segment times) - COLL_PEN per collision.
Antithetic ES with centred-rank shaping and Adam. Each generation re-scores the current mean, which is saved as
rl/<name>/best.npz whenever it improves and is clean. Logs rl/<name>/log.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARENT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "competition_code"))
from submission import RESIDUAL_AHEAD_M  # noqa: E402

CUTS = [660.0, 1130.0, 1480.0, 2420.0, 3600.0, 3810.0, 5120.0]  # line_s, 120 m before each corner's braking zone
SEGMENTS = [[a, b] for a, b in zip(CUTS, CUTS[1:] + CUTS[:1])]
NF, NH = 7 + len(RESIDUAL_AHEAD_M), 16
COLL_PEN = 20.0
SHAPES = [("W1", (NH, NF)), ("b1", (NH,)), ("W2", (2, NH)), ("b2", (2,))]


def unflat(theta):
    out, i = {}, 0
    for k, sh in SHAPES:
        n = int(np.prod(sh))
        out[k] = theta[i:i + n].reshape(sh).tolist()
        i += n
    return out


def n_params():
    return sum(int(np.prod(sh)) for _, sh in SHAPES)


class Worker:
    def __init__(self, slot):
        wt = os.path.join(PARENT, "slots", "wt%d" % slot)
        self.p = subprocess.Popen([os.path.join(PARENT, ".venv", "bin", "python"), "-u",
                                   os.path.join(ROOT, "tools", "rl", "worker.py"), str(slot), wt],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.path.join(wt, "logs", "rl_worker.log"), "a"), text=True, cwd=PARENT)
        line = self.p.stdout.readline()
        if "ready" not in line:
            raise RuntimeError("worker %d failed: %r" % (slot, line))

    def submit(self, weights):
        self.p.stdin.write(json.dumps({"weights": weights, "episodes": SEGMENTS}) + "\n")
        self.p.stdin.flush()

    def result(self):
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def evaluate(workers, thetas):
    """Score each theta (None = policy off) on all segments, spread over the workers."""
    res = [None] * len(thetas)
    queue = list(range(len(thetas)))
    busy = {}
    while queue or busy:
        for w in workers:
            if w not in busy and queue:
                i = queue.pop(0)
                w.submit(None if thetas[i] is None else unflat(thetas[i]))
                busy[w] = i
        for w in list(busy):  # results come back in submission order per worker
            r = w.result()
            res[busy.pop(w)] = r
    return res


def fitness(r):
    return -sum(r["times"]) - COLL_PEN * sum(r["coll"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slots", default="1,2,3,4")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)  # antithetic pairs x 2
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "rl", a.name)
    os.makedirs(out, exist_ok=True)
    st_f = os.path.join(out, "state.npz")
    n = n_params()
    rng = np.random.default_rng(0)
    if a.resume and os.path.exists(st_f):
        st = np.load(st_f)
        theta, m, v, g0, best = st["theta"], st["m"], st["v"], int(st["gen"]) + 1, float(st["best"])
        rng = np.random.default_rng(g0)
    else:
        theta = rng.normal(0, 0.3, n)
        theta[-(2 * NH + 2):] = 0.0  # output layer zero: the policy starts as "no correction"
        m, v, g0, best = np.zeros(n), np.zeros(n), 0, -np.inf
    workers = [Worker(int(s)) for s in a.slots.split(",")]
    log = open(os.path.join(out, "log.csv"), "a", newline="")
    lw = csv.writer(log)
    try:
        if g0 == 0:
            base = evaluate(workers, [None])[0]
            print("baseline (policy off): %.2f s, coll %d, segments %s" % (sum(base["times"]), sum(base["coll"]),
                                                                        np.round(base["times"], 2).tolist()), flush=True)
            lw.writerow(["base", sum(base["times"]), sum(base["coll"]), json.dumps(base["times"])])
        for g in range(g0, g0 + a.gens):
            t0 = time.time()
            eps = rng.normal(0, 1, (a.pop // 2, n))
            thetas = [theta] + [theta + a.sigma * e for e in eps] + [theta - a.sigma * e for e in eps]
            rs = evaluate(workers, thetas)
            f = np.array([fitness(r) for r in rs])
            f_mean, fp, fm = f[0], f[1:1 + len(eps)], f[1 + len(eps):]
            ranks = np.argsort(np.argsort(np.r_[fp, fm])) / (2 * len(eps) - 1) - 0.5
            grad = ((ranks[:len(eps)] - ranks[len(eps):])[:, None] * eps).sum(0) / (len(eps) * a.sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + a.lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            r0 = rs[0]
            clean0 = not any(r0["coll"])
            if clean0 and f_mean > best:
                best = f_mean
                np.savez(os.path.join(out, "best.npz"), **{k: np.array(x) for k, x in unflat(thetas[0]).items()})
            print("gen %3d: mean %.2f s (coll %d) | pop best %.2f s | pop coll %d | best clean %.2f | %.0f s" % (
                g, -f_mean if clean0 else sum(r0["times"]), sum(r0["coll"]), -max(f[1:]),
                sum(sum(r["coll"]) for r in rs[1:]), -best, time.time() - t0), flush=True)
            lw.writerow([g, sum(r0["times"]), sum(r0["coll"]), json.dumps(r0["times"]), -max(f[1:])])
            log.flush()
            np.savez(st_f, theta=theta, m=m, v=v, gen=g, best=best)
    finally:
        for w in workers:
            w.close()


if __name__ == "__main__":
    main()
