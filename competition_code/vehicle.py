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

    def kappa_max(self, v):
        v = max(v, 1.0)
        return min(self.K(v), self.a_lat_max / (v * v))

    def steer_for(self, kappa, v):
        return float(np.clip(-kappa / self.K(v), -1.0, 1.0))

    def pedals_for(self, a_cmd, v):
        if a_cmd >= 0:
            ad = self.a_drive(v)
            th = (1.0 if a_cmd > 0 else 0.0) if ad <= 0 else a_cmd / ad
            return (1.0 if th >= 0.95 else float(np.clip(th, 0, 1))), 0.0
        return 0.0, float(np.clip(-a_cmd / (self.a_brake * self.brake_scale), 0, 1))

    def a_lat(self, s_line):
        i = bisect.bisect_right(self._starts, s_line) - 1  # -1 wraps to last section
        return self._vals[i]
