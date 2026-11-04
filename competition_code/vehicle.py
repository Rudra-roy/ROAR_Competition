"""Vehicle actuation maps (runtime + offline). Convention: kappa = -K(v)*steer, positive steer = right."""
import bisect
import json

import numpy as np


class Vehicle:
    def __init__(self, params):
        self.p = params
        self.a_brake = params["a_brake"]
        self.brake_scale = params.get("brake_scale", 1.0)
        self._sv = list(params["steer_gain_v"])
        self._sk = list(params["steer_gain_k"])
        secs = params["a_lat_sections"]
        self._starts = [s for s, _ in secs]
        self._vals = [a for _, a in secs]
        self.a_lat_max = max(self._vals)
        # per-section braking scale (same sections as a_lat_sections; absent = 1 everywhere): a corner can brake
        # harder with the pedal than the global limit (a_brake * brake_scale) without it applying to every corner
        bss = params.get("brake_sec_scale")
        self._bss = [float(x) for x in bss] if bss else [1.0] * len(secs)
        if len(self._bss) != len(secs):
            raise ValueError("brake_sec_scale needs one value per a_lat section")
        pm = params.get("pedal_map")
        self._pm = None
        if pm:  # measured steady-state accel on one signed pedal axis u = throttle - brake (exclusive pedals)
            u = np.r_[-np.asarray(pm["brk"][:0:-1], float), pm["thr"]]
            tab = np.c_[np.asarray(pm["a_brk"], float)[:, :0:-1], np.asarray(pm["a_thr"], float)]
            for j in range(tab.shape[1] - 2, -1, -1):  # strictly increasing in u per speed, so it inverts
                tab[:, j] = np.minimum(tab[:, j], tab[:, j + 1] - 1e-3)
            self._pm = (np.asarray(pm["v"], float), u, tab)

    def a_brake_max(self, v):
        """Largest deceleration (> 0) at speed v. Default: the constant a_brake * brake_scale. brake_model
        "pedal_map": brake_scale * the measured full-brake decel of the pedal map, linear in v^2 between grid speeds
        (clamped outside), so the offline profile can use the same function as linear constraints (brake_lines)."""
        if self.p.get("brake_model") != "pedal_map":
            return self.a_brake * self.brake_scale
        pm = self.p["pedal_map"]
        return self.brake_scale * float(np.interp(v * v, np.square(pm["v"]), -np.asarray(pm["a_brk"])[:, -1]))

    def brake_lines(self):
        """None for the constant model; else [(c, d)] with a_brake_max(v) = min_i(c_i + d_i v^2) on the grid range:
        the chords between grid speeds. Exact only if the table is concave in v^2 (checked)."""
        if self.p.get("brake_model") != "pedal_map":
            return None
        pm = self.p["pedal_map"]
        x, y = np.square(pm["v"]), -self.brake_scale * np.asarray(pm["a_brk"])[:, -1]
        d = np.diff(y) / np.diff(x)
        if (np.diff(d) > 1e-12).any():
            raise ValueError("full-brake decel is not concave in v^2: chords would not bound it")
        return [(float(y[i] - d[i] * x[i]), float(d[i])) for i in range(len(d))]

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def _smooth(self, v):
        return self.p["steer_gain_k0"] / (1.0 + self.p["steer_gain_c"] * v * v)

    def K(self, v):
        v = max(v, 1.0)
        if self._sv[0] <= v <= self._sv[-1]:
            return float(np.interp(v, self._sv, self._sk))
        ve, ke = (self._sv[0], self._sk[0]) if v < self._sv[0] else (self._sv[-1], self._sk[-1])
        return self._smooth(v) * ke / self._smooth(ve)

    def a_drive(self, v):
        return min(a - b * v * v for a, b in self.p["a_drive_lines"])

    def scrub_chords(self):
        """Tyre scrub: cornering at lateral accel x costs S(x) of the drive accel (fitted at full throttle,
        tools/fit_scrub.py). Returned as secant chords [(m, c)], S(x) <= max_j(m_j x + c_j) (convex, conservative);
        [(0, 0)] when the model is off."""
        sc = self.p.get("scrub")
        if not sc:
            return [(0.0, 0.0)]
        x = np.asarray(sc["x"], float)
        y = np.asarray(sc["s"], float)
        m = np.diff(y) / np.diff(x)
        return [(0.0, 0.0)] + [(float(mi), float(yi - mi * xi)) for mi, xi, yi in zip(m, x[:-1], y[:-1])]

    def scrub(self, a_lat_abs):
        return max(max(m * a_lat_abs + c for m, c in self.scrub_chords()), 0.0)

    def kappa_max(self, v):
        v = max(v, 1.0)
        return min(self.K(v), self.a_lat_max / (v * v))

    def steer_for(self, kappa, v):
        u = float(np.clip(-kappa / self.K(v), -1.0, 1.0))
        cap = self.p.get("steer_cap")  # [[v, |steer| max], ...]: past the tyres' peak more steer only scrubs speed
        if cap:
            c = float(np.interp(v, [a for a, _ in cap], [b for _, b in cap]))
            u = min(max(u, -c), c)
        return u

    def pedals_for(self, a_cmd, v, a_y=0.0, s_line=None):
        """(throttle, brake) for a longitudinal accel command at speed v. a_y: current lateral accel magnitude.
        Option "corner_throttle_floor": {"thr": t, "ay_min": a} - while cornering (a_y > a) hold the throttle at
        >= t and take the extra deceleration from the brake pedal: a closed throttle is engine braking on the driven
        wheels (-13 m/s^2 at 50 m/s here), which unloads the rear mid-corner; the pedal brakes all four wheels.
        "thr_sec" (one value per a_lat section) overrides t per section when s_line is given."""
        cf = self.p.get("corner_throttle_floor")
        if cf and self._pm is not None and a_y > cf["ay_min"] and a_cmd < 0:
            t = float(cf["thr"])
            if s_line is not None and cf.get("thr_sec"):
                t = float(cf["thr_sec"][bisect.bisect_right(self._starts, s_line) - 1])
            vg, ug, tab = self._pm
            i = int(np.clip(np.searchsorted(vg, v) - 1, 0, len(vg) - 2))
            f = min(max((v - vg[i]) / (vg[i + 1] - vg[i]), 0.0), 1.0)
            row = (1 - f) * tab[i] + f * tab[i + 1]   # accel over u = -1..1 (brake..throttle), throttle 0 at u = 0
            thr_u, brk_u = ug[ug >= 0], -ug[ug <= 0][::-1]  # 0..1 each
            a_thr = row[ug >= 0]
            a_brk = row[ug <= 0][::-1] - row[ug == 0][0]  # the pedal's own contribution (<= 0), engine braking removed
            a_floor = float(np.interp(t, thr_u, a_thr))
            if a_cmd >= a_floor:  # mild: throttle alone, at or above the floor
                return float(np.interp(a_cmd, a_thr, thr_u)), 0.0
            need = a_cmd - a_floor  # what the pedal has to add (<= 0)
            if need >= a_brk[-1]:
                return t, float(np.interp(need, a_brk[::-1], brk_u[::-1]))
            # beyond full pedal: give the rest back to the throttle (more engine braking), pedal stays full
            return float(np.clip(np.interp(a_cmd - a_brk[-1], a_thr, thr_u), 0.0, t)), 1.0
        if self._pm is not None:
            # The measured map is continuous and monotone in u at every speed (no gap), so the inverse is one
            # pedal setting: linear in v between grid rows, then interp a -> u (clamped to full brake/throttle).
            vg, ug, tab = self._pm
            i = int(np.clip(np.searchsorted(vg, v) - 1, 0, len(vg) - 2))
            f = min(max((v - vg[i]) / (vg[i + 1] - vg[i]), 0.0), 1.0)
            u = float(np.interp(a_cmd, (1 - f) * tab[i] + f * tab[i + 1], ug))
            return (u, 0.0) if u >= 0 else (0.0, -u)
        if a_cmd >= 0:
            ad = self.a_drive(v)
            th = (1.0 if a_cmd > 0 else 0.0) if ad <= 0 else a_cmd / ad
            return (1.0 if th >= 0.95 else float(np.clip(th, 0, 1))), 0.0
        return 0.0, float(np.clip(-a_cmd / (self.a_brake * self.brake_scale), 0, 1))

    def a_lat(self, s_line):
        i = bisect.bisect_right(self._starts, s_line) - 1  # -1 wraps to last section
        return self._vals[i]

    def brake_sec(self, s_line):
        """Braking scale of the section at s_line (1.0 unless brake_sec_scale is set)."""
        return self._bss[bisect.bisect_right(self._starts, s_line) - 1]


def load_profile(path, speed_scale=1.0):
    """v_ref(s_total) from speed_profile.npz: linear interp, clamped at the ends, scaled."""
    d = np.load(path)
    s, v = d["s"], d["v"]
    return lambda s_total: speed_scale * float(np.interp(s_total, s, v))
