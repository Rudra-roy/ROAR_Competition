"""Run one race with the UNMODIFIED competition runner (competition_code/competition_runner.py).

  python tools/sim/race.py            # from the repo root, CARLA server already up on port 2000

Only two things are patched, both outside the runner: the CARLA client timeout (5 s -> 120 s, because the first ticks
under Proton are slow) and, if ROAR_PORT is set, the server port (for parallel servers). Simulated time does not
depend on wall-clock speed (synchronous mode), so the score is the same as under the official runner.
Env: ROAR_CODE (competition_code dir to race, default this repo's), ROAR_PORT (default 2000),
ROAR_HUD=1 (live speed/lap/progress overlay in the viewer, dev only), ROAR_LOG=1 (per-tick log to logs/run.csv).
"""
import asyncio
import os
import sys

import carla

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code = os.path.abspath(os.environ.get("ROAR_CODE", os.path.join(REPO, "competition_code")))
port = int(os.environ.get("ROAR_PORT", "2000"))
sys.path.insert(0, code)
os.chdir(code)
_init, _timeout = carla.Client.__init__, carla.Client.set_timeout
carla.Client.__init__ = lambda self, host="127.0.0.1", p=2000, *a, **k: _init(self, host, port if p == 2000 else p, *a, **k)
carla.Client.set_timeout = lambda self, t: _timeout(self, max(t, 120.0))
import competition_runner  # noqa: E402

if os.environ.get("ROAR_HUD") == "1":
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import hud  # noqa: E402
    hud.install()
asyncio.run(competition_runner.main())
