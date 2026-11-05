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

CUTS = [660.0, 1130.0, 1480.0, 2420.0, 3600.0, 5120.0]  # line_s on straights, ~120 m before each braking zone
# (the bottom curve c3835 + c3993-4113 is one segment: a cut inside it started the car mid-slide and it crashed)
SEGMENTS = [[a, b] for a, b in zip(CUTS, CUTS[1:] + CUTS[:1])]
NF, NH = 7 + len(RESIDUAL_AHEAD_M), 16
COLL_PEN0, COLL_PEN_MIN, COLL_PEN_MAX = 5.0, 2.0, 80.0
TARGET_COLL_RATE = (0.30, 0.10)  # population crash rate the penalty steers toward: early (exploring) -> mature
MATURE_GENS = 30
PERT = (0.06, 0.7, 0.03)  # start randomisation: speed +-6 %, lateral +-0.7 m, heading +-0.03 rad
REPS = 2  # randomised starts per segment per evaluation (same draws for every variant in a generation)
SIGMA_MIN, SIGMA_MAX = 0.02, 0.3
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

    def submit(self, weights, episodes):
        self.p.stdin.write(json.dumps({"weights": weights, "episodes": episodes}) + "\n")
        self.p.stdin.flush()

    def result(self):
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def evaluate(workers, thetas, episodes=None):
    """Score each theta (None = policy off) on the episodes (default: the plain segments), spread over the workers."""
    episodes = episodes or SEGMENTS
    res = [None] * len(thetas)
    queue = list(range(len(thetas)))
    busy = {}
    while queue or busy:
        for w in workers:
            if w not in busy and queue:
                i = queue.pop(0)
                w.submit(None if thetas[i] is None else unflat(thetas[i]), episodes)
                busy[w] = i
        for w in list(busy):  # results come back in submission order per worker
            r = w.result()
            res[busy.pop(w)] = r
    return res


class Reward:
    """Evolving reward. A crash costs pen * (1 + severity) plus the rest of the segment at the baseline pace (so
    crashing later beats crashing sooner: a gradient even when everything crashes); severity = min(impulse / 2000, 2),
    a scrape vs a hard hit. pen adapts every generation toward a target population crash rate that tightens as the
    model matures (0.30 -> 0.10 over MATURE_GENS generations, counted only while the policy improves): early on
    crashes are cheap enough to explore near the limit, later they dominate."""

    def __init__(self, base_times, pen=COLL_PEN0, gen=0):
        self.base = np.asarray(base_times, float)  # one per episode (segments x REPS when randomised)
        self.pen, self.gen = pen, gen

    def __call__(self, r):
        return float(self.per_segment(r).sum())

    def per_segment(self, r):
        """Reward of each segment (time + crash cost, negated)."""
        t = np.asarray(r["times"], float)
        c = np.asarray(r["coll"], bool)
        frac = np.clip(np.asarray(r.get("frac", [1.0] * len(t)), float), 0.0, 1.0)
        sev = np.minimum(np.asarray(r.get("impulse", [0.0] * len(t)), float) / 2000.0, 2.0)
        rest = np.where(c, (1.0 - frac) * self.base, 0.0)
        cost = self.pen * (1.0 + sev) * (0.5 + 0.5 * (1.0 - frac))  # the earlier in the segment, the dearer
        return -(t + rest + cost * c)

    def target(self):
        a, b = TARGET_COLL_RATE
        return a + (b - a) * min(self.gen / MATURE_GENS, 1.0)

    def adapt(self, crash_rate, improving):
        if crash_rate > self.target():
            self.pen = min(self.pen * 1.3, COLL_PEN_MAX)
        elif crash_rate < 0.5 * self.target():
            self.pen = max(self.pen * 0.9, COLL_PEN_MIN)
        if improving:
            self.gen += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--slots", default="1,2,3,4")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)  # antithetic pairs x 2
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="start the mean from a saved policy (best.npz)")
    ap.add_argument("--view", type=int, default=0, help="slot whose server opens a window with a chase camera")
    a = ap.parse_args()
    if a.view:
        os.environ["ROAR_RL_VIEW"] = str(a.view)
    out = os.path.join(ROOT, "rl", a.name)
    os.makedirs(out, exist_ok=True)
    st_f = os.path.join(out, "state.npz")
    n = n_params()
    rng = np.random.default_rng(0)
    if a.resume and os.path.exists(st_f):
        st = np.load(st_f)
        theta, m, v, g0, best = st["theta"], st["m"], st["v"], int(st["gen"]) + 1, float(st["best"])
        rng = np.random.default_rng(g0)
        reward = Reward(st["base"], float(st["pen"]), int(st["mat"]))
        sigma = float(st["sigma"]) if "sigma" in st else a.sigma
    else:
        theta = rng.normal(0, 0.3, n)
        theta[-(2 * NH + 2):] = 0.0  # output layer zero: the policy starts as "no correction"
        if a.init:
            f0 = np.load(a.init)
            theta = np.concatenate([np.asarray(f0[k], float).ravel() for k, _ in SHAPES])
        m, v, g0, best = np.zeros(n), np.zeros(n), 0, -np.inf
        sigma = a.sigma
    workers = [Worker(int(s)) for s in a.slots.split(",")]
    log = open(os.path.join(out, "log.csv"), "a", newline="")
    lw = csv.writer(log)
    try:
        if g0 == 0:
            base = evaluate(workers, [None])[0]
            print("baseline (policy off): %.2f s, coll %d, segments %s" % (sum(base["times"]), sum(base["coll"]),
                                                                        np.round(base["times"], 2).tolist()), flush=True)
            lw.writerow(["base", sum(base["times"]), sum(base["coll"]), json.dumps(base["times"])])
            if any(base["coll"]):
                raise RuntimeError("the baseline crashes in a segment: fix the segment cuts first")
            reward = Reward(base["times"])
        for g in range(g0, g0 + a.gens):
            t0 = time.time()
            eps = rng.normal(0, 1, (a.pop // 2, n))
            thetas = [theta] + [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps]
            # this generation's randomised starts (shared by every variant), plus the policy-off reference on them
            draws = rng.uniform(-1, 1, (len(SEGMENTS) * REPS, 3)) * np.array(PERT)
            episodes = [[s0, s1, d.tolist()] for (s0, s1), d in zip(SEGMENTS * REPS, draws)]
            rs = evaluate(workers, [None] + thetas, episodes)
            ref, rs = rs[0], rs[1:]
            reward.base = np.where(np.asarray(ref["coll"]), np.asarray(ref["times"]) * 1.0 + 5.0, np.asarray(ref["times"]))
            f = np.array([reward(r) for r in rs])
            f_mean = f[0]
            # per-corner shaping: rank the variants within each segment, then sum the ranks, so a gain in any corner
            # counts the same (the corner with the biggest absolute gain would otherwise drown out the rest)
            F = np.array([reward.per_segment(r) for r in rs[1:]])  # (pop, segments)
            seg_ranks = np.argsort(np.argsort(F, axis=0), axis=0) / (len(F) - 1) - 0.5
            score = seg_ranks.sum(1)
            ranks = np.argsort(np.argsort(score)) / (len(score) - 1) - 0.5
            grad = ((ranks[:len(eps)] - ranks[len(eps):])[:, None] * eps).sum(0) / (len(eps) * a.sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + a.lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            r0 = rs[0]
            clean0 = not any(r0["coll"])
            crash_rate = float(np.mean([any(r["coll"]) for r in rs[1:]]))
            gain = float(np.sum(np.asarray(ref["times"]) - np.asarray(r0["times"])))  # vs policy off, same starts
            improving = clean0 and gain > best
            if improving:
                best = gain
                np.savez(os.path.join(out, "best.npz"), **{k: np.array(x) for k, x in unflat(thetas[0]).items()})
            print("gen %3d: mean gains %+.2f s vs policy off on the same %d starts (coll %d, ref coll %d) | crash rate "
                  "%.2f (target %.2f) | penalty %.1f | sigma %.3f | maturity %d | best clean gain %+.2f | %.0f s" % (
                      g, gain, len(episodes), sum(r0["coll"]), sum(ref["coll"]), crash_rate, reward.target(),
                      reward.pen, sigma, reward.gen, best, time.time() - t0), flush=True)
            lw.writerow([g, sum(r0["times"]), sum(r0["coll"]), json.dumps(r0["times"]), gain, crash_rate,
                         reward.pen, reward.gen, sigma, json.dumps(ref["times"])])
            log.flush()
            reward.adapt(crash_rate, improving)
            # exploration adapts to the crash rate (the penalty alone cannot: how far the variants stray sets it)
            if crash_rate > 2 * reward.target():
                sigma = max(sigma * 0.8, SIGMA_MIN)
            elif crash_rate < 0.5 * reward.target():
                sigma = min(sigma * 1.1, SIGMA_MAX)
            np.savez(st_f, theta=theta, m=m, v=v, gen=g, best=best, base=reward.base, pen=reward.pen, mat=reward.gen,
                     sigma=sigma)
    finally:
        for w in workers:
            w.close()


if __name__ == "__main__":
    main()
