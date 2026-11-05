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
RISK_W = (0.5, 4.0)  # near-miss weight (s of reward per unit of risk): early -> mature
PERT = (0.06, 0.7, 0.03)  # start randomisation: speed +-6 %, lateral +-0.7 m, heading +-0.03 rad
REPS = 2  # randomised starts per segment per evaluation (same draws for every variant in a generation)
SIGMA_MIN, SIGMA_MAX = 0.02, 0.3
LAP_START = 4700.0  # line_s on the back straight: the one placement per lap (straight, steady speed)
LAP_PERT = (0.03, 0.3, 0.01)  # mild start randomisation for laps
NOISE_STD = 0.004  # steering wobble (AR(1)) in laps: trajectory variety without teleports
SHAPES = [("W1", (NH, NF)), ("b1", (NH,)), ("W2", (2, NH)), ("b2", (2,))]


ENSEMBLE = False  # set by --ensemble: one network per lap section (sections start at the sorted CUTS)
SECT_CUTS = sorted(CUTS)
TRACK_L = 5588.0


def unflat(theta):
    n1 = n_params()
    if ENSEMBLE and theta.size == len(SECT_CUTS) * n1:
        parts = [unflat_one(theta[k * n1:(k + 1) * n1]) for k in range(len(SECT_CUTS))]
        out = {key: [p[key] for p in parts] for key in parts[0]}
        out["cuts"] = SECT_CUTS
        return out
    return unflat_one(theta)


def lap_segment_experts():
    """Section network driving each timed lap segment (segments start at LAP_START + 100 m, then at each cut)."""
    t0 = LAP_START + 100.0
    starts = [t0] + sorted(CUTS, key=lambda c: (c - t0) % TRACK_L)
    return [(int(np.searchsorted(SECT_CUTS, st % TRACK_L, side="right")) - 1) % len(SECT_CUTS) for st in starts]


def unflat_one(theta):
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
        while True:  # skip library chatter until the worker's JSON ready line (or its exit)
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker %d exited during start-up (see slots/wt%d/logs/rl_worker.log)" % (slot, slot))
            if line.startswith("{") and "ready" in line:
                break

    info = ""

    def submit(self, weights, episodes):
        job = {"weights": weights, "info": Worker.info}
        job.update({"laps": episodes, "cuts": CUTS} if episodes and len(episodes[0]) == 4 and not
                   isinstance(episodes[0][1], (int, float)) else {"episodes": episodes})
        self.p.stdin.write(json.dumps(job) + "\n")
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
    """Score each theta (None = policy off) on the episodes (default: the plain segments). Laps are dispatched one
    (theta, lap) at a time, so every slot stays busy until the generation is done (no idle slot waiting)."""
    episodes = episodes or SEGMENTS
    lap_mode = len(episodes[0]) == 4 and not isinstance(episodes[0][1], (int, float))
    jobs = [(i, [ep]) for i in range(len(thetas)) for ep in episodes] if lap_mode else \
        [(i, episodes) for i in range(len(thetas))]
    parts = [[None] * (len(episodes) if lap_mode else 1) for _ in thetas]
    queue, busy = list(range(len(jobs))), {}
    while queue or busy:
        for w in workers:
            if w not in busy and queue:
                j = queue.pop(0)
                i, eps = jobs[j]
                w.submit(None if thetas[i] is None else unflat(thetas[i]), eps)
                busy[w] = j
        import select
        ready, _, _ = select.select([w.p.stdout for w in busy], [], [])
        for w in list(busy):
            if w.p.stdout in ready:
                j = busy.pop(w)
                i, eps = jobs[j]
                r = w.result()
                if lap_mode:
                    parts[i][episodes.index(eps[0])] = r["laps"][0]
                else:
                    parts[i][0] = r
    return [{"laps": p} if lap_mode else p[0] for p in parts]


def lap_to_segments(laps):
    """Flatten lap results to per-segment entries for Reward: splits as times; the crash segment carries the crash;
    segments never reached have time None (Reward fills in the policy-off pace)."""
    times, coll, imp, frac, risk = [], [], [], [], []
    for lp in laps:
        rk = lp.get("risk") or [0.0] * len(lp["splits"])
        for k, sp in enumerate(lp["splits"]):
            risk.append(float(rk[k]) if k < len(rk) else 0.0)
            crash = lp["coll_seg"] == k
            times.append(None if (sp is None and not crash) else (sp if sp is not None else 0.0))
            coll.append(bool(crash))
            imp.append(lp["impulse"] if crash else 0.0)
            frac.append(lp["frac"] if crash else 1.0)
    return {"times": times, "coll": coll, "impulse": imp, "frac": frac, "risk": risk}


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
        """Reward of each segment (time + crash cost, negated). A time of None = never reached (after a crash
        earlier in the lap): counted at the policy-off pace."""
        t = np.array([self.base[i] if x is None else x for i, x in enumerate(r["times"])], float)
        c = np.asarray(r["coll"], bool)
        frac = np.clip(np.asarray(r.get("frac", [1.0] * len(t)), float), 0.0, 1.0)
        sev = np.minimum(np.asarray(r.get("impulse", [0.0] * len(t)), float) / 2000.0, 2.0)
        rest = np.where(c, (1.0 - frac) * self.base, 0.0)
        cost = self.pen * (1.0 + sev) * (0.5 + 0.5 * (1.0 - frac))  # the earlier in the segment, the dearer
        risk = np.asarray(r.get("risk", [0.0] * len(t)), float)  # near misses: slip / line excess, every tick
        return -(t + rest + cost * c + self.risk_w() * risk)

    def risk_w(self):
        a, b = RISK_W
        return a + (b - a) * min(self.gen / MATURE_GENS, 1.0)

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
    ap.add_argument("--laps", action="store_true", help="continuous flying laps instead of teleported segments")
    ap.add_argument("--view", type=int, default=0, help="slot whose server opens a window with a chase camera")
    ap.add_argument("--ensemble", action="store_true", help="one network per lap section, each graded on its own")
    ap.add_argument("--train", help="ensemble: comma list of section networks to train (others frozen)")
    ap.add_argument("--pen-max", type=float, default=COLL_PEN_MAX, help="cap on the adaptive crash penalty")
    a = ap.parse_args()
    global ENSEMBLE, COLL_PEN_MAX
    ENSEMBLE = a.ensemble
    COLL_PEN_MAX = a.pen_max
    if a.view:
        os.environ["ROAR_RL_VIEW"] = str(a.view)
    out = os.path.join(ROOT, "rl", a.name)
    os.makedirs(out, exist_ok=True)
    st_f = os.path.join(out, "state.npz")
    n = n_params() * (len(SECT_CUTS) if a.ensemble else 1)
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
            if "theta" in f0:
                one = np.asarray(f0["theta"], float)
            elif np.asarray(f0["W1"]).ndim == 3:  # an ensemble file: its section networks in order
                n1k = n_params()
                one = np.concatenate([np.concatenate([np.asarray(f0[k], float)[j].ravel() for k, _ in SHAPES])
                                      for j in range(np.asarray(f0["W1"]).shape[0])])
            else:
                one = np.concatenate([np.asarray(f0[k], float).ravel() for k, _ in SHAPES])
            theta = np.tile(one, n // one.size) if one.size < n else one  # each section network starts as the policy
        m, v, g0, best = np.zeros(n), np.zeros(n), 0, -np.inf
        sigma = a.sigma
    best_theta, bad_gens = (theta.copy() if a.init else None), 0
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
            Worker.info = "TRAINING %s  gen %d  best gain %+.2f s/2 laps  penalty %.1f  explore %.3f" % (
                a.name, g, best if np.isfinite(best) else 0.0, reward.pen, sigma)
            eps = rng.normal(0, 1, (a.pop // 2, n))
            if a.ensemble and a.train:  # frozen section networks: no perturbation, no update
                n1 = n_params()
                live = {int(x) for x in a.train.split(",")}
                for k in range(len(SECT_CUTS)):
                    if k not in live:
                        eps[:, k * n1:(k + 1) * n1] = 0.0
            thetas = [theta] + [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps]
            # this generation's randomised starts (shared by every variant), plus the policy-off reference on them
            if a.laps:  # continuous flying laps: one placement on the back straight, splits per corner
                draws = rng.uniform(-1, 1, (REPS, 3)) * np.array(LAP_PERT)
                seeds = rng.integers(1, 2 ** 31, REPS)
                episodes = [[LAP_START, d.tolist(), NOISE_STD, int(sd)] for d, sd in zip(draws, seeds)]
                rs = [lap_to_segments(r["laps"]) for r in evaluate(workers, [None] + thetas, episodes)]
            else:
                draws = rng.uniform(-1, 1, (len(SEGMENTS) * REPS, 3)) * np.array(PERT)
                episodes = [[s0, s1, d.tolist()] for (s0, s1), d in zip(SEGMENTS * REPS, draws)]
                rs = evaluate(workers, [None] + thetas, episodes)
            ref, rs = rs[0], rs[1:]
            rt = np.array([np.nan if x is None else x for x in ref["times"]], float)
            prev = reward.base if len(reward.base) == len(rt) else np.full(len(rt), np.nanmean(rt))
            reward.base = np.where(np.isnan(rt) | np.asarray(ref["coll"]), prev, rt)
            f = np.array([reward(r) for r in rs])
            f_mean = f[0]
            # per-corner shaping: rank the variants within each segment, then sum the ranks, so a gain in any corner
            # counts the same (the corner with the biggest absolute gain would otherwise drown out the rest)
            F = np.array([reward.per_segment(r) for r in rs[1:]])  # (pop, segments)
            seg_ranks = np.argsort(np.argsort(F, axis=0), axis=0) / (len(F) - 1) - 0.5
            score = seg_ranks.sum(1)
            ranks = np.argsort(np.argsort(score)) / (len(score) - 1) - 0.5
            if ENSEMBLE and a.laps:  # each section network graded on its own segments only
                n1, K = n_params(), len(SECT_CUTS)
                seg_exp = lap_segment_experts()
                col_exp = np.array([seg_exp[j % len(seg_exp)] for j in range(F.shape[1])])
                grad = np.zeros(n)
                for k in range(K):
                    Sk = F[:, col_exp == k].sum(1)
                    rk = np.argsort(np.argsort(Sk)) / (len(Sk) - 1) - 0.5
                    blk = slice(k * n1, (k + 1) * n1)
                    grad[blk] = ((rk[:len(eps)] - rk[len(eps):])[:, None] * eps[:, blk]).sum(0) / (len(eps) * sigma)
            else:
                grad = ((ranks[:len(eps)] - ranks[len(eps):])[:, None] * eps).sum(0) / (len(eps) * sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + a.lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            r0 = rs[0]
            clean0 = not any(r0["coll"])
            crash_rate = float(np.mean([any(r["coll"]) for r in rs[1:]]))
            gain = (float(np.sum(reward.base) - np.sum(r0["times"])) if not any(r0["coll"])
                    else float("nan"))  # vs policy off on the same starts
            improving = clean0 and gain > best
            if improving:
                best = gain
                best_theta, bad_gens = thetas[0].copy(), 0
            elif any(r0["coll"]):
                bad_gens += 1
                if bad_gens >= 3 and best_theta is not None:  # the mean drifted into crashing: roll back to the best
                    print("mean crashed 3 generations running: rolling back to the best clean policy", flush=True)
                    theta, m, v, bad_gens = best_theta.copy(), np.zeros(n), np.zeros(n), 0
                    sigma = max(sigma, 0.05)
                    reward.pen = max(reward.pen * 0.5, COLL_PEN_MIN)
                np.savez(os.path.join(out, "best.npz"), **{k: np.array(x) for k, x in unflat(thetas[0]).items()})
            r0["times"] = [x if x is not None else 0.0 for x in r0["times"]]
            risk0 = float(np.sum(r0.get("risk", [0.0])))
            print("near-miss risk of the mean policy %.2f (weight %.2f)" % (risk0, reward.risk_w()), flush=True)
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
