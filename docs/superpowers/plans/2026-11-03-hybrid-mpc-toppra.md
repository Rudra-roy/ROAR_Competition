# Hybrid MPC + TOPP-RA Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the starter `submission.py` with a controller that follows the hand-tuned racing line using a long-horizon MPC, under a TOPP-RA time-optimal speed profile built from the car's *measured* power, braking and grip limits. Target: finish 3 laps with 0 collisions, under 330 s first, then beat 319.65 s (Summer 2026 winner).

**Architecture:** Offline, export the racing line, identify the car (full-throttle acceleration curve, braking, top speed, steer→curvature gain) and run TOPP-RA over 3 laps to get `v_ref(s)`; ship both as data files. Online, each 0.05 s tick: localize on the line (arc length `s`, lateral error `e_y`, heading error `e_psi`), solve a linear-time-varying MPC (OSQP, 60 steps = 3 s ≈ 210 m at top speed) for curvature and acceleration, and map those to steer/throttle/brake through the identified vehicle model.

**Tech Stack:** Python 3.8 (`.venv`), numpy, scipy 1.10.1, osqp 1.1.3 (runtime), toppra 0.6.8 (offline only), pytest (dev only), CARLA 0.9.12 + ROAR_PY.

**Spec:** the user's request in this session (2026-11-03): "hybrid system, MPC with long looking distance, hand-tuned racing line for the waypoints, optimize velocity with TOPP-RA, use the maximum power and speed the simulated car can give." Competition rules: `~/.claude/projects/-home-mt-labpc-Documents-roar-racing/memory/roar-competition-rules.md`.

## Global Constraints

- Only `ROAR_Competition/competition_code/submission.py` plus new files may change. `competition_runner.py` and `infrastructure.py` stay byte-identical to upstream `augcog/ROAR_Competition` `main`.
- `submission.py` must not call `receive_observation()` or `world.step()`. Read sensors only with `get_last_observation()` / `get_last_gym_observation()`.
- Score = 3 laps, sim time = ticks × 0.05 s, viewer on. A collision with impulse > 100 teleports the car to the start, resets progress and keeps the clock running. Timeout 5000 s.
- Runtime dependencies are limited to numpy, scipy and osqp, listed in `ROAR_Competition/requirements.txt`. toppra and pytest are dev-only and are never imported by `submission.py` or its runtime modules.
- All code runs under `/home/mt-labpc/Documents/roar_racing/.venv` (Python 3.8).
- The racing line comes from `github.com/micahc123/roar2026` (no licence file). Credit it in `ROAR_Competition/README.md`, and ask the user to check with the organizers or the author before the Dec 11 2026 submission.
- Every speed or grip limit lives in `competition_code/data/vehicle_params.json` as a tunable knob. Nothing hard-codes them.
- Work on branch `hybrid-mpc` in `ROAR_Competition`. Commit after each task.

## Review Focus

1. **Crash respawn:** the car is teleported to the start mid-race. The controller must relocalize globally and restart `v_ref` from 0, not drive off searching a ±50-point window (this is the known DNF bug in winning repos). Test: `test_track.py::test_relocalize_after_teleport`, `test_submission_logic.py::test_respawn_resets_progress`.
2. **Lap wrap:** `s` must unwrap across the line's start/finish seam, or `v_ref` jumps back to the lap-1 launch profile. Test: `test_track.py::test_progress_unwraps_across_seam`.
3. **Steering sign:** in ROAR, positive steer turns right (yaw decreases). A sign error crashes on the first corner. Test: `test_vehicle.py::test_left_curvature_gives_negative_steer`.
4. **Curvature noise:** raw 1 m-spaced points give spiky κ, which drives TOPP-RA to near-zero speeds. Test: `test_track.py::test_curvature_smooth_and_line_preserved`.
5. **Infeasible MPC:** if OSQP fails or hits its time limit, the car must not get a NaN or no command. Test: `test_mpc.py::test_fallback_on_solver_failure`.

---

## File Map

All paths are relative to `ROAR_Competition/`.

| File | Responsibility |
|---|---|
| `competition_code/track.py` | Racing line geometry: arc length, smoothed curvature, projection, global relocalization, progress unwrapping |
| `competition_code/vehicle.py` | Load `vehicle_params.json`; map curvature → steer and accel → throttle/brake; `a_drive(v)` |
| `competition_code/mpc.py` | LTV Frenet MPC on OSQP with fallback |
| `competition_code/submission.py` | Glue: `initialize()` loads data and builds objects; `step()` runs localize → MPC → actuate |
| `competition_code/data/racing_line.npz` | `xy` (5865×2), exported widened line |
| `competition_code/data/vehicle_params.json` | Identified + tunable limits |
| `competition_code/data/speed_profile.npz` | `s` and `v` arrays over 3 laps from TOPP-RA |
| `tools/export_line.py` | Parse the reference line literal via `ast` (no import or exec of the ref repo) and apply the shipped widening |
| `tools/sysid_run.py` | Runs the official runner with a logging `SysIdSolution`, writes `logs/sysid.csv` |
| `tools/fit_sysid.py` | `logs/sysid.csv` → `data/vehicle_params.json` |
| `tools/build_profile.py` | TOPP-RA → `data/speed_profile.npz`, prints predicted lap times |
| `tools/eval.sh` | N full races: start server, run, parse result, kill server, append to `results.csv` |
| `tests/test_*.py` | One file per runtime module |

---

### Task 0: Branch, dependencies, evaluation harness

**Files:** Create `tools/eval.sh`, `requirements.txt`; modify `.gitignore` (add `logs/`, `results.csv`).

**Interfaces:**
- Produces: `tools/eval.sh <runs> <label>`. Per run it appends a `results.csv` row `git_sha,label,run,sim_time_s,collisions,status`, where status ∈ {`finished`,`failed`,`crashed_client`,`server_hang`}.

- [ ] **Step 1:** `git -C ROAR_Competition checkout -b hybrid-mpc`. `uv pip install osqp==1.1.3 scipy==1.10.1 toppra==0.6.8 pytest`. Write `requirements.txt` with only `osqp==1.1.3` and `scipy==1.10.1`.
- [ ] **Step 2:** Write `tools/eval.sh`. Per run it:
  - kills any wineserver (`WINEPREFIX=~/tools/pfx/pfx ~/tools/GE-Proton9-27/files/bin/wineserver -k`) and starts `../run_monza.sh` in the background;
  - waits for port 2000 (fail as `server_hang` after 120 s), then sleeps 5 s;
  - runs `python -u ../demo.py` from the repo root with a wall timeout of 1800 s;
  - reads `Solution finished in X seconds` → `sim_time_s` and counts `major collision` lines;
  - kills the server.

  Restarting the server every run is required: after a run the server either exits or hangs in sync mode.
- [ ] **Step 3:** Verify with `tools/eval.sh 1 baseline` on the unmodified starter. Expected row: `finished` with `sim_time_s` = 1658.90 (±0.1) and 0 collisions. Determinism check.
- [ ] **Step 4:** Commit `chore: eval harness and runtime requirements`.

### Task 1: Export the racing line

**Files:** Create `tools/export_line.py`, `competition_code/data/racing_line.npz`, `tests/test_export_line.py`.

**Interfaces:**
- Produces: `racing_line.npz` with key `xy`, shape (5865, 2), float64, in the same order and index base as the reference `WaypointLine.line_locations`.
- `export_line.py <path/to/ref/WaypointLine.py> <out.npz>`

- [ ] **Step 1: Test** `test_export_line.py::test_shape_and_widening`:
  - `xy.shape == (5865, 2)`;
  - the last row equals the literal's last pair `(-281.61450497288973, 393.17536918350305)` (index 5864 is outside every widen window);
  - the per-point shift from the raw literal at index 1353 (the middle of window `1313:1393:+0.9`) has norm within 0.01 of 0.9;
  - at index 1313 the shift is < 1e-9.
- [ ] **Step 2:** Run `pytest tests/test_export_line.py`. Expect FAIL (no file).
- [ ] **Step 3: Implement `export_line.py`.**
  - Read the file with `ast.parse`, find the `WP_TO_FOLLOW` string assignment and parse its `x, y` lines.
  - Apply the shipped default widen specs exactly as the reference `apply_widen` does: raised-cosine `w = 0.5*(1-cos(2πk/span))`, shift along the left normal of the central-difference tangent of the *unshifted* line, and specs applied cumulatively to the copy.
  - Specs: `1313:1393:0.9, 1393:1473:-0.9, 883:963:-0.5, 963:1043:0.5, 4036:4116:0.5, 4116:4196:-0.5, 1688:1768:0.3, 1768:1848:-0.3`.
  - Never `import` the reference module.
- [ ] **Step 4:** Run `python -I tools/export_line.py ../ref/roar2026/competition_code/WaypointLine.py competition_code/data/racing_line.npz`, then pytest. Expect PASS.
- [ ] **Step 5:** Commit `feat: export hand-tuned racing line`.

### Task 2: Track geometry

**Files:** Create `competition_code/track.py`, `tests/test_track.py`.

**Interfaces:**
- Produces `class Track` with:
  - `Track(xy: np.ndarray, ds: float = 1.0)`: closed loop, resampled uniformly at `ds` along a periodic smoothing spline.
  - Attributes: `length: float`; `s: (M,)`; `xy: (M,2)`; `psi: (M,)` tangent heading; `kappa: (M,)` signed curvature, left-turn positive.
  - `project(p: np.ndarray, hint_idx: Optional[int]) -> Tuple[int, float, float]` returns `(idx, s, e_y)`. `e_y` is positive when the car is left of the line. With `hint_idx`, search ±40 m; with `None`, search globally.
  - `heading_error(idx: int, yaw: float) -> float`, wrapped to (−π, π].
  - `class Progress(track)`: `update(p, yaw) -> (s_total, e_y, e_psi, idx)`.
    - `s_total` is distance since (re)start, unwrapped across laps.
    - Uses a global search on the first call, or when the position jumps > 20 m in one tick (teleport), which also resets `s_total = 0`.
    - `start_s` = the line `s` where the car first appears.

- [ ] **Step 1: Tests** (`test_track.py`)
  - `test_circle_curvature`: a synthetic circle of radius 50 (2000 points) gives `kappa` ≈ +0.02 ±2% (CCW) everywhere and `length` ≈ 2π·50 ±0.5%.
  - `test_curvature_smooth_and_line_preserved`: on the real line,
    - the maximum distance from the original points to the resampled curve is < 0.10 m;
    - `max|kappa|` < 0.06 (the tightest corner is R≈20.8 m → 0.048);
    - `max|Δkappa|` per 1 m < 0.005.
  - `test_project_on_line`: each of 50 random line points projects to within 0.5 m of its own `s`, with `|e_y|` < 0.05. A point offset 2 m to the left normal gives `e_y` = +2.0 ±0.05.
  - `test_progress_unwraps_across_seam`: drive `Progress` along 1.5 laps of line points; `s_total` rises monotonically to ≈1.5·length ±2 m.
  - `test_relocalize_after_teleport`: after 300 m of driving, feed the start point. `s_total` resets to 0 ±2 m and `idx` is correct.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement with `scipy.interpolate.splprep(per=1)`. Pick the smoothing factor `s` to pass both curvature tests; analytic κ comes from `splev` derivatives. Vectorize the windowed nearest-point search with numpy.
- [ ] **Step 4:** Run. Expect PASS.
- [ ] **Step 5:** Commit `feat: track geometry with relocalization`.

### Task 3: System identification (in the simulator)

**Files:** Create `tools/sysid_run.py`, `tools/fit_sysid.py`, `competition_code/data/vehicle_params.json`, `tests/test_fit_sysid.py`.

**Interfaces:**
- `sysid_run.py` copies `demo.py`'s timeout patch, replaces `competition_runner.RoarCompetitionSolution` with `SysIdSolution` before calling `competition_runner.main()`, and writes `logs/sysid.csv` with columns `t,x,y,yaw,v,throttle,brake,steer,yaw_rate`.
  - Steering: pure pursuit on `Track` with a 15 m lookahead.
  - Speed logic: full throttle while the maximum `|kappa|` over the next 250 m is < 0.004 (straights); else full brake until v ≤ 25 m/s, then hold 25 m/s. This gives full-throttle data up to top speed and full-brake data from high speed.
  - Over one lap, add a ±0.05 steer square-wave dither on straights only while v < 40 m/s.
  - Stop after one lap.
- `vehicle_params.json` keys, with fitted defaults:
  - **Fitted from data:**
    - `a_drive_lines`: list of `[alpha, beta]` with `a_drive(v) = min_k(alpha_k − beta_k·v²)`. It's a concave upper envelope fitted to the full-throttle samples, with at most 4 lines.
    - `a_brake`: positive m/s², the 10th-percentile decel at brake = 1, v > 30.
    - `v_max`: max observed v.
    - `steer_gain_v`, `steer_gain_k`: a table of `K(v)` where `yaw_rate = −K(v)·v·steer`.
  - **Knobs, not fitted:**
    - `a_lat_sections`: `[[s_start_m, a_lat], ...]` along the line. Start at μ·g with the winners' per-section μ: 3.05, 2.97, 3.35, 3.36, 3.14, 2.98, 3.30, 2.85, 2.98, 2.10. Map their maneuverable-waypoint section starts `[2611, 322, 557, 739, 1158, 1317, 1516, 1881, 1944, 2359]` to line `s` by projecting `maneuverable_waypoints[i]`.
    - `speed_scale` 0.85
    - `brake_scale` 1.0
    - `mpc`: `{N: 60, dt: 0.05, q_ey: 10, q_epsi: 50, q_v: 0.5, r_dkappa: 2e5, r_da: 0.01, ey_max: 1.0}`

- [ ] **Step 1: Test** `test_fit_sysid.py::test_fit_recovers_synthetic`. Generate synthetic samples from known `a_drive = min(9.0, 12.0 − 0.0025 v²)`, `a_brake = 17`, `K(v) = 0.12/(1+0.0004 v²)` with noise σ = 0.2. The fit must recover each to within 5% at v ∈ {10, 40, 70} and preserve `K`'s sign.
- [ ] **Step 2:** Run. Expect FAIL. Implement `fit_sysid.py` (numpy least squares; acceleration from a centered difference of v). Run. Expect PASS.
- [ ] **Step 3:** Run `tools/sysid_run.py` (server via the `eval.sh` start logic), then `fit_sysid.py logs/sysid.csv competition_code/data/vehicle_params.json`. Expected sanity:
  - `v_max` within 69–73 m/s (≈257 km/h);
  - `a_brake` within 12–35;
  - `K(v)` positive under the `yaw_rate = −K·v·steer` convention.
  
  If any is outside its range, stop and report the values rather than tuning around them.
- [ ] **Step 4:** Commit `feat: system identification and vehicle params`.

### Task 4: Vehicle actuation maps

**Files:** Create `competition_code/vehicle.py`, `tests/test_vehicle.py`.

**Interfaces:**
- `class Vehicle(params: dict)` with:
  - `a_drive(v) -> float`
  - `kappa_max(v) -> float`: `min(|K(v)|·1.0, a_lat_max_global/v²)`, using v ≥ 1
  - `steer_for(kappa, v) -> float`: `−kappa/K(v)` clipped to [−1, 1]
  - `pedals_for(a_cmd, v) -> (throttle, brake)`. Throttle is `a_cmd/a_drive(v)`, set to 1.0 when ≥ 0.95. Brake is `−a_cmd/(a_brake·brake_scale)`. Both are clipped to [0, 1] and never both > 0.
  - `a_lat(s_line) -> float`: section lookup.

- [ ] **Step 1: Tests**
  - `test_left_curvature_gives_negative_steer`: `steer_for(+0.01, 30) < 0`.
  - `test_pedals_exclusive_and_saturate`: `pedals_for(+100, 30) == (1.0, 0.0)`, `pedals_for(−100, 30) == (0.0, 1.0)`, and `pedals_for(0.96·a_drive(30), 30)[0] == 1.0`.
  - `test_a_lat_section_lookup` checks boundary s values.
- [ ] **Step 2:** Run (FAIL), implement, run (PASS).
- [ ] **Step 3:** Commit `feat: vehicle actuation maps`.

### Task 5: TOPP-RA speed profile (offline)

**Files:** Create `tools/build_profile.py`, `competition_code/data/speed_profile.npz`, `tests/test_speed_profile.py`.

**Interfaces:**
- `build_profile(track: Track, veh: Vehicle, start_s: float, laps: int = 3, grid: float = 2.0) -> Tuple[np.ndarray, np.ndarray]` returns `(s, v)` with `s` the distance from the start (0 … laps·length).
- `load_profile(path) -> Callable[[float], float]`: `v_ref(s_total)`, linear interpolation, scaled by `speed_scale`. It lives in `vehicle.py` so the runtime never imports toppra.
- **TOPP-RA formulation** (1-DOF path `q = s`, identity spline `toppra.SplineInterpolator(ss, ss[:,None])`), using `toppra.constraint.SecondOrderConstraint`:
  - `inv_dyn(q, qd, qdd) = [qdd, qd², kappa(q)·qd²]`
  - rows of `F(q)·w ≤ g(q)`:
    - `[1, beta_k, 0] ≤ alpha_k` for each drive line;
    - `[−1, 0, 0] ≤ a_brake·brake_scale`;
    - `[0, 0, ±1] ≤ a_lat(q)`;
    - `[0, 1, 0] ≤ v_max²`.
  - Start `sd = 0`, end free. Solver: `toppra.algorithm.TOPPRA(..., parametrizer="ParametrizeConstAccel")`. Read `v = sqrt(K)` on the grid from `compute_parameterization` (`sd` vector).
  - `kappa` and `a_lat` are indexed by the line `s` = `(start_s + q) mod length`.

- [ ] **Step 1: Tests** (use a pure-numpy forward–backward pass written *in the test file* as the oracle)
  - `test_straight_hits_vmax`: on a 3 km straight, v reaches `v_max` ±1% and `dv²/ds ≤ 2·a_drive(v)+tol` everywhere.
  - `test_circle_lat_limit`: on an R = 50 circle, v ≤ sqrt(a_lat·50)·1.001 everywhere.
  - `test_matches_oracle`: on the real track, the TOPP-RA v is within 2% of the oracle at 95% of grid points.
  - `test_predicted_flying_lap_plausible`: lap-2 time Σ(ds/v) is within 95–125 s (winners' measured flying lap: 104.65 s).
- [ ] **Step 2:** Run (FAIL), implement, run (PASS).
- [ ] **Step 3:** Run `tools/build_profile.py`. It writes the npz and prints per-lap predicted times and the top 5 slowest corners (line s, v). Record the printout in the commit message.
- [ ] **Step 4:** Commit `feat: TOPP-RA speed profile`.

### Task 6: LTV MPC

**Files:** Create `competition_code/mpc.py`, `tests/test_mpc.py`.

**Interfaces:**
- `class FrenetMPC(veh: Vehicle, cfg: dict)` with `solve(e_y, e_psi, v, s_total, kappa_ref_fn, v_ref_fn, u_prev) -> (kappa_cmd, a_cmd, ok: bool)`.
- **Model:** states `[e_y, e_psi, v]`, inputs `[kappa, a]`. Linearized about `e = 0`, `v = v_k`, with `v_k` = the previous solution's predicted speeds (first call: `v_ref` along the predicted s).
  - `e_y⁺ = e_y + dt·v_k·e_psi`
  - `e_psi⁺ = e_psi + dt·v_k·(kappa − kappa_ref(s_k))`
  - `v⁺ = v + dt·a`
  - `s_k` is integrated from `v_k`.
- **Cost:** `Σ q_ey·e_y² + q_epsi·e_psi² + q_v·(v − v_ref(s_k))² + r_dkappa·Δkappa² + r_da·Δa²`.
- **Constraints:**
  - `|kappa| ≤ kappa_max(v_k)`
  - `−a_brake·brake_scale ≤ a ≤ a_drive(v_k)`
  - `|e_y| ≤ ey_max` (soft, via a slack with weight 1e6)
- **One-tick delay:** before building the QP, propagate the measured state one step with `u_prev`.
- **Solver:** OSQP, warm-started, `time_limit = 0.03`, `eps_abs = 1e-4`, `polish = False`. Update the matrices in place and build the sparsity once.
- **Fallback:** if the status isn't solved or the output is non-finite, return `ok = False` with a pure-pursuit curvature `2·sin(α)/L_d` (`L_d = max(10, 0.6·v)`) and `a = clip(k·(v_ref − v))`.

- [ ] **Step 1: Tests**
  - `test_converges_on_straight`: closed-loop on the exact linear model with a 1-tick delay, starting from `e_y = 1.0` at 40 m/s → `|e_y|` < 0.1 within 2 s with no overshoot > 0.2.
  - `test_respects_input_bounds`: over 200 random states, every output satisfies the kappa/a bounds within 1e-3.
  - `test_brakes_before_corner`: a straight followed by an R = 40 circle with `v_ref` from `build_profile`'s oracle. Starting at 60 m/s, the first-step `a_cmd` is negative once the corner is within the stopping distance.
  - `test_fallback_on_solver_failure`: patch the OSQP status to failed → `ok is False` and a finite output.
  - `test_solve_time`: median < 15 ms over 200 solves.
- [ ] **Step 2:** Run (FAIL), implement, run (PASS).
- [ ] **Step 3:** Commit `feat: LTV Frenet MPC with fallback`.

### Task 7: Integrate into `submission.py`

**Files:** Modify `competition_code/submission.py` (replace the body, keeping the class name and constructor signature); create `tests/test_submission_logic.py`.

**Interfaces:**
- `initialize()`:
  - loads `data/*.npz/json` via paths relative to `__file__`;
  - builds `Track`, `Vehicle`, `Progress`, `FrenetMPC`, `v_ref`;
  - all heavy work happens here (no sim time passes during `initialize`).
- `step()`:
  - reads location / rpy / velocity via `get_last_gym_observation()`;
  - `Progress.update` → `mpc.solve` → `veh.steer_for` / `veh.pedals_for`;
  - applies the control dict with `hand_brake 0, reverse 0, target_gear 0`;
  - logs one CSV line per tick to `logs/run.csv` only when env `ROAR_LOG=1`.

- [ ] **Step 1: Test** `test_respawn_resets_progress`. With mocked sensors, step a solution 300 m along the line, then teleport it to the start: `s_total` ≈ 0 and the next `v_ref` ≈ `v_ref(0)`.
- [ ] **Step 2:** Run (FAIL), implement, run (PASS).
- [ ] **Step 3:** Sim run: `tools/eval.sh 1 hybrid-s085` (speed_scale 0.85). Expected: `finished`, 0 collisions, time < 420 s.
  - If it collides, run `ROAR_LOG=1` and plot `e_y` versus s at the crash point.
  - Fix tracking (MPC weights or the steer gain) before raising speed.
- [ ] **Step 4:** Commit `feat: hybrid MPC + TOPP-RA submission`.

### Task 8: Push to the limit (tuning)

**Files:** Modify `competition_code/data/vehicle_params.json` only.

- [ ] **Step 1:** Raise `speed_scale` 0.85 → 1.0 in steps of 0.05, rebuilding nothing. Use 2 runs per step and keep a step only with 0 collisions in both.
- [ ] **Step 2:** Raise `a_lat` per section, one section at a time, in +3% steps; rebuild the profile after each change. Stop a section at the first collision and back off one step. (Winners found a global +2% → DNF; sections must be probed alone.)
- [ ] **Step 3:** Raise `brake_scale` in +10% steps the same way. The winners' effective braking was about 32 m/s² against a measured 16–19.
- [ ] **Step 4:** Confirm with 5 runs: all `finished`, 0 collisions; report the median and worst time. Pass gate: < 330 s. Stretch: < 319.65 s.
- [ ] **Step 5:** Commit the params with the 5-run results in the message.

### Task 9: Package the submission

**Files:** Modify `ROAR_Competition/README.md`; confirm `requirements.txt`.

- [ ] **Step 1:** README: run instructions (`pip install -r requirements.txt`, then the unchanged runner), a one-paragraph method summary, and credit for the racing line (`micahc123/roar2026`, built on `MightyMark3/ROAR_1_S25`).
- [ ] **Step 2:** Clean-env check: run `diff` on `competition_runner.py` and `infrastructure.py` against `upstream/main` (expect empty), then `tools/eval.sh 1 final-clean` from a fresh clone of the branch into a temp dir.
- [ ] **Step 3:** Commit. Ask the user before merging to `main` or pushing.

---

## Known risks (decide at the gates, don't pre-build)

- **A model-based profile can be slower than hand tuning.** The Summer 2026 3rd-place team measured +10 s/lap. Task 8's per-section knobs are the mitigation. If the Task 7 gate time is > 360 s, compare the logged `v` against the TOPP-RA `v_ref` per section before tuning anything else.
- **Combined grip (braking while turning) isn't modelled.** Friction-circle facets go in only if crashes cluster at corner entries under braking.
- **OSQP time limit + warm start:** if runs stop being bit-identical (the winners saw 0.05–0.3 s jitter), drop `time_limit` and use a fixed `max_iter`.
