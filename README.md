# ROAR Competition entry: slip-aware MPC with learned corrections (Monza)

**Result:** 318.40 s median over 20 races (318.30 to 318.40 s), **0 collisions**, the car body never closer than 0.13 m to a
wall on the wall map (`tools/gate.py`, 4 of the 20 with the server window open). The map is ray-cast, with collision-probe
edges at the inside of Lesmo 1, the final hairpin, c1329, the first-chicane exit and the kink; the car's own poses were
then probed (static, collision mesh): closest touch 0.18 m (hairpin apex), 0.26 m at Lesmo 1, c1329 and the chicane exit,
0.40 m at the kink, none within 0.6 m at sec8; outside walls 0.60 m (Lesmo 1) and 0.75 m (hairpin). Elsewhere the walls are
ray-cast only. Cars that crashed in our tests touched at 0.0-0.1 m. One machine only: results elsewhere are untested,
3 laps of Monza, CARLA 0.9.12, run with the unmodified `competition_runner.py` and `infrastructure.py`.

Documentation for the competition itself: https://roar.gitbook.io/roar-competition-documentation/

## Running it (official way)

The submission is `competition_code/submission.py` with its modules and data in `competition_code/`. On a machine
where the Monza CARLA server already runs (e.g. Windows, as the organizers run it), it runs like the starter code:

```bash
pip install -r requirements.txt          # numpy, scipy, osqp (Python 3.8)
# start the Monza CARLA server, then:
cd competition_code
python competition_runner.py             # prints "Solution finished in 318.3... seconds"
```

Runtime dependencies are numpy, scipy and osqp only (plus `roar_py_interface` from ROAR_PY). The submission reads
sensors through `get_last_gym_observation()` and never calls `receive_observation()` or `world.step()`.
All data it needs is committed (table below), so nothing has to be generated or trained to reproduce the result.

## Reproducing the result on Linux (step by step)

Tested on Ubuntu 22.04, NVIDIA RTX 5060 Ti / RTX 4070 SUPER, X11. Monza ships as a Windows CARLA 0.9.12 build, so the
server runs under GE-Proton 9; the Python client runs natively in a Python 3.8 venv. Needs a Vulkan GPU driver
(`vulkaninfo --summary`), git, curl, [uv](https://astral.sh/uv), about 10 GB of disk and about 3 GB of free RAM per
running server. No sudo or Steam.

**1. Folder layout** (the scripts assume these paths; each can be overridden, see step 6):

```
~/Documents/roar_racing/
├── .venv/              Python 3.8 venv (step 4)
├── ROAR_PY/            git clone of augcog/ROAR_PY
├── ROAR_Competition/   this repo, branch hybrid-mpc
└── Monza_V1.1/Monza/   the extracted map (contains CarlaUE4.exe, CarlaUE4/, Engine/)
~/tools/GE-Proton9-27/  Proton
~/tools/pfx/            Proton prefix (created on first launch)
```

**2. Map.** Download Monza from https://roar.berkeley.edu/monza-map/ and extract it to
`~/Documents/roar_racing/Monza_V1.1/Monza/`.

**3. GE-Proton 9** (10 and 11 need glibc 2.38 and fail on Ubuntu 22.04):

```bash
mkdir -p ~/tools/pfx && cd ~/tools
curl -LO https://github.com/GloriousEggroll/proton-ge-custom/releases/download/GE-Proton9-27/GE-Proton9-27.tar.gz
tar xf GE-Proton9-27.tar.gz && rm GE-Proton9-27.tar.gz
```

**4. Code and Python environment** (carla 0.9.12 wheels exist only for Python <= 3.8):

```bash
cd ~/Documents/roar_racing
git clone https://github.com/augcog/ROAR_PY.git
git clone -b hybrid-mpc https://github.com/Rudra-roy/ROAR_Competition.git
uv venv --python 3.8 .venv
source .venv/bin/activate
(cd ROAR_PY && uv pip install -r requirements.txt -e roar_py_core -e roar_py_carla)
uv pip install -r ROAR_Competition/requirements.txt
python -c "import carla, roar_py_interface, osqp, scipy; print('ok')"
```

`requirements.txt` pins what the submission needs (numpy 1.24.4, scipy 1.10.1, osqp 1.1.3; osqp must be 1.x);
carla 0.9.12 and gymnasium 0.28.1 come from ROAR_PY. For the development tools and tests (not needed to race):
`uv pip install -r ROAR_Competition/requirements-dev.txt`, then `PYTHONPATH= python -m pytest tests/ -q` from the
repo root (162 tests, about 5 minutes).

**5. One race, watching it.** Start the server outside the venv (Proton needs the system python >= 3.9), wait for
port 2000, then run the race wrapper from the venv:

```bash
cd ~/Documents/roar_racing/ROAR_Competition
deactivate 2>/dev/null; tools/sim/run_monza.sh &           # opens a 1280x720 CarlaUE4 window
until ss -ltn | grep -q ':2000 '; do sleep 2; done          # about 10-30 s
source ../.venv/bin/activate
python tools/sim/race.py                                    # ROAR_HUD=1 adds a speed/lap/progress overlay
```

`tools/sim/race.py` runs the unmodified `competition_runner.py`. It only raises the CARLA client timeout from 5 s to
120 s (the first ticks under Proton are slower than 5 s). The sim runs in synchronous mode, so wall-clock speed does
not change the score. Expected output: `Solution finished in 318.3... seconds` (318.30 to 318.40 s), and no
`major collision` line. Stop the server afterwards with
`WINEPREFIX=~/tools/pfx/pfx ~/tools/GE-Proton9-27/files/bin/wineserver -k`.

**6. The 5-race check.** `tools/eval.sh` starts a server, races, kills the server, and repeats:

```bash
ROAR_SERVER_ARGS=-RenderOffScreen tools/eval.sh 5 check    # appends git_sha,label,run,sim_time_s,collisions,status
cat results.csv                                             # to results.csv; logs in logs/check_N.*
```

To race in parallel (one server per slot on port 2000+3·slot, offscreen; leave about 3 GB of RAM per slot):
`tools/eval_slot.sh <slot> <checkout dir> <label>` prints `label,sim_time,collisions,status`, for example
`for k in 1 2 3 4; do tools/eval_slot.sh $k . run_$k & sleep 25; done; wait`.

Paths can be overridden with `ROAR_MONZA` (map folder), `ROAR_PROTON` (Proton folder), `ROAR_PFX` (prefix) and
`ROAR_VENV` (venv, default `../.venv`).

**Troubleshooting.** Use `-dx12` (already set; DX11 asserts) and never `-quality-level=Low` (missing materials crash
the map). `time-out of 5000ms while waiting for the simulator`: run through `tools/sim/race.py`, not the runner
directly. If a client is killed mid-race the server stays in synchronous mode and hangs: stop it with `wineserver -k`
as above and start again. Server logs: `Monza_V1.1/Monza/CarlaUE4/Saved/Logs/CarlaUE4.log`.

## Data files

All in `competition_code/data/`, committed:

| File | What it is |
|---|---|
| `racing_line.npz` | the racing line (points + its spline) |
| `speed_profile.npz` | the time-optimal speed plan along the line (TOPP-RA, 3 laps from a standing start) |
| `line_offset.npz` | a learned lateral offset of the line, per metre (see below) |
| `vehicle_params.json` | fitted car model: pedal map, steering, tyres, per-corner grip, braking and throttle settings, controller weights |
| `residual.npz` | learned per-section correction networks (see below) |
| `centreline.npy` | track centreline (used by the tools, not loaded at run time) |

## How it works

1. **Racing line.** A minimum-curvature line (`tools/mincurv_line.py`), widened corner by corner. The waypoint polygon
   is not the real wall layout (fences and obstacles sit inside it), so the room is bounded by what was measured:
   first the space cars have driven without contact, and at Lesmo 1 and the final hairpin the real walls
   (`--corridor`, below). At the diagonal kink, the bottom curve (sec8) and the first chicane the curvature cost
   also rises from the apex to the exit (`--exit-weight`, `SHIP_EXIT_STEPS`), so the line does its turning early and
   runs straighter where the car is on full power (0.45 + 0.25 + 0.05 s per race).
   **Real walls** (`tools/map_corridor.py` -> `tools/corridor_map.npz`): ray casts sideways from the centreline every
   metre, corrected with `tools/probe_obstacle.py`, which teleports the car through a recorded race's poses shifted
   toward the corner's inside and finds the first touch with the simulator's own collision mesh (the rays miss
   the lower part of some fences by 0.1-0.25 m; crash replays touch at 0.0-0.1 m, so the probe is validated).
   `tools/correct_corridor.py` merges probe results into the map. The line's bounds are the wall minus the car
   body's measured reach (slides included) minus a gap (`SHIP_CORRIDOR_WINDOWS` in `mincurv_line.py`).
2. **Speed plan.** TOPP-RA over the line with per-section lateral grip, a fitted drive-force limit and braking that
   shares grip with cornering (friction ellipse), offline in `tools/build_profile.py`.
3. **Controller: slip-aware MPC** (`competition_code/tyre_mpc.py`). A linear time-varying MPC (OSQP, 60 steps,
   3 s) on a dynamic bicycle model with saturating tyres and braking load transfer, fitted to race logs
   (`tools/fit_tyres.py`). It tracks the line's course rather than the car's nose, limits tyre slip angles, and
   shrinks its braking limit as cornering load rises, so it brakes in a straight line and turns while easing off.
   Its speed weight (`mpc.q_v` = 128) makes it hold the planned speed through corners: with the earlier weight of 2
   it traded speed for line accuracy and ran 3-9 m/s below the plan in every corner (2.3 s per race).
   `competition_code/mpc.py` is the earlier kinematic MPC, kept as an alternative (`mpc.model`).
4. **Learned line offset** (`tools/ilc.py`). Iterative learning over repeated laps: the car's measured lateral error
   at each metre of the lap is fed back into the reference, so the driven path lands on the intended line.
5. **Per-corner tuning** (`tools/corner_opt.py`, `tools/grid_race.py`). Mid-corner throttle floor (so the driven
   wheels keep pulling instead of engine-braking the rear loose), braking strength and corner grip, optimised per
   section by racing four simulations in parallel. A change is kept only if the car's body (its four corners from
   the logged heading, not just its centre) stays within the body positions of earlier clean races
   (`tools/body_margin.py`, `corner_opt.py --body-ref`), or with `--walls` stays at least `--wall-min` (0.10 m) from
   the measured walls: sliding harder through Lesmo was faster but swept the body 0.3-0.4 m further inside, and one
   such run hit the apex. Final releases are checked with `tools/gate.py`: N full races of the committed setup
   (slot checkouts verified byte-identical, some with the server window open), the closest wall approach of every race.
6. **Learned corrections** (`tools/rl/`). Six small neural networks, one per section of the lap, add bounded
   corrections to the MPC's steering and acceleration (at most 0.03 rad and 4 m/s²). They were trained with
   evolution strategies on continuous flying laps in the simulator, each graded on its own section's time, crashes
   and near-misses. Only the sections whose networks proved faster in full races are active; the others output zero.
   At run time they are a few hundred numbers evaluated with numpy.

## Repository layout

- `competition_code/` the submission (runner and infrastructure unchanged from upstream).
- `tools/sim/` the Linux server launcher (`run_monza.sh`) and race wrapper (`race.py`).
- `tools/` development tools: car-model fitting, line and speed-plan generation, the learning loops, race
  evaluation (`eval.sh`, `race_guard.sh`, `eval_slot.sh` for parallel simulator slots), analysis.
- `tests/` unit tests: `PYTHONPATH= python -m pytest tests/ -q` (about 5 minutes; needs `requirements-dev.txt`).

## Credits

- The racing line was first built from the line of the Summer 2026 first-place entry
  (github.com/micahc123/roar2026), itself built on github.com/MightyMark3/ROAR_1_S25. The corner widening used the
  path that ROAR_1_S25's car drove in our simulator (recorded telemetry) as evidence of where the road is clear.
  No code from either repository is included. Neither carries a licence, so we credit the authors here and
  disclose this use to the organizers with the submission. The same was done with the recorded path of the Summer
  2026 first-place entry's car in our simulator (telemetry only): it is part of the driven-path evidence for the exit
  steps of the line (`--winner`) and the benchmark the lap was compared against section by section. The wall map and
  the obstacle probe use only our own simulator runs.
- ROAR_PY, CARLA and the competition starter code: the ROAR team at UC Berkeley.
