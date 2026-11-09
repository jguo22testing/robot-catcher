# robot-catcher

Ball catching in general room geometries via learned ball trajectories (Team 4.5).
This is the stage-0 scaffold: a NumPy point-ball bounce simulator + throw/dataset
generator, and a MuJoCo scene of a KUKA iiwa 14 with a cup that catches using
ground-truth (oracle) ball positions.

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # creates .venv with mujoco, numpy, matplotlib, pytest
uv run pytest      # sanity tests for the ball sim and throw generator
```

## Layout

| File | What it does |
|---|---|
| `catcher/room.py` | `Panel` (bounded two-sided rectangle) + `Material` (restitution, friction). `make_room(stage, rng)` builds the procedural room for stages 1–4. Point-cloud sampling and a flat `(P, 13)` panel encoding for the model. |
| `catcher/ball_sim.py` | Event-driven point-ball sim. Flight is exact ballistic motion; bounces are found as quadratic roots, so there's no time step and no tunnelling. Bounce: `v_n' = -e v_n`, `v_t' = (1 - friction) v_t`. |
| `catcher/throws.py` | Catch zone (reach shell around the shoulder), rejection-sampled throws per stage (stage 1: 0 bounces, 2: exactly 1, 3: 1–3, 4: 1–4), Gaussian observation noise, and `generate_dataset` → fixed-shape arrays. |
| `catcher/mujoco_scene.py` | iiwa14 + cup + room panels built with `MjSpec`; computed-torque joint control driven by either IK + joint-space cubic (`ik`) or differential IK (`diffik`). IK is damped least squares on MuJoCo Jacobians with the cup-axis cone constraint (random restarts on failure). The ball is a mocap body replayed from the NumPy sim. Offscreen rendering helpers. |
| `models/kuka_iiwa_14/` | iiwa14 model vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). |
| `scripts/simulate_ball.py` | Plot a few throws (3D + side view, bounces, catch points). |
| `scripts/generate_dataset.py` | Write a stage dataset to `data/*.npz`. |
| `scripts/run_catch.py` | Simulate one catch and save a GIF, replay it in the MuJoCo viewer (`--viewer`), or `--eval N` for a headless catch rate. `--planner` (`mean`, `mean-replan` or `hedge`) plans on a synthetic belief instead of the ground truth. |

## Bounce Lab (web UI)

A local app for building rooms, running the NumPy simulator, and inspecting and
saving the throws. It adds no dependencies beyond the project's own.

```bash
uv run python webapp/server.py    # then open http://127.0.0.1:8000 (--port to change)
```

1. Pick **Custom room** (edit the room size, surface materials, and up to 20
   flat boards) or a **Stage 1-4** preset (a new random room per throw, as in the
   stage datasets; **Edit this room** copies one into the editor).
2. Choose sampled throws (with a bounce range) or one manual throw, then **Run simulation**.
3. Play back or scrub each throw, and overlay all of them for a custom room.
4. **Save** writes exactly the displayed throws to `data/web/<name>.npz`.

Saved files have the arrays listed under "Dataset arrays", plus `stage` (0 for a
custom room) and `config_json` (the full request, for reproducing the run). A manual
throw that never reaches the catch zone has `catch_index` and `bounces_before_catch` of -1.

| File | What it does |
|---|---|
| `webapp/server.py` | Standard-library HTTP server: static files plus a JSON API, on localhost only. |
| `webapp/sim_api.py` | Converts UI JSON to and from `Room`/`Panel`, validates it, runs `catcher`, builds datasets. All geometry math stays on the Python side. |
| `webapp/frontend/` | Plain HTML/CSS/JS with Three.js (vendored, so it works offline). |

Limits: surfaces are flat rectangles (no curved geometry yet), and the ball ignores
the robot, so it passes through the base and catch zone. The arm is only in the MuJoCo sim.

## Usage

```bash
uv run python scripts/simulate_ball.py --stage 3 --n 3 --seed 2 --out outputs/stage3.png
uv run python scripts/generate_dataset.py --stage 2 --n 1000 --seed 0
uv run python scripts/run_catch.py --stage 2 --seed 3                # -> outputs/catch_stage2_seed3.gif
uv run mjpython scripts/run_catch.py --stage 2 --seed 3 --viewer     # interactive viewer (mjpython on macOS)
uv run python scripts/run_catch.py --stage 4 --eval 50 --controller diffik
```

## Trajectory model

The architecture and experiments are specified in `trajectory_model_plan.md`. The code is in
`catcher/trajectory/` (`config`, `generate_data`, `dataset`, `windows`, `model`, `train`, `evaluate`).

```bash
uv run python scripts/generate_trajectories.py                                  # data/stage1: no-bounce throws, section 2
uv run python scripts/plot_trajectories.py                                      # E0 plot of random throws
uv run python scripts/train_trajectory.py --split overfit --steps 20000 --name e1
uv run python scripts/train_trajectory.py --split train --steps 20000 --name e2  # add --per-room N, --noise 0.005
uv run python scripts/animate_prediction.py --run e2 --split test_rooms --n 3   # GIFs of predictions over time
```

Training logs to Weights & Biases (project `robot-catcher`) if you've run `uv run wandb login`,
and otherwise saves the log locally for a later `wandb sync`. Pass `--wandb disabled` to turn it off.
Results go to `outputs/trajectory/<name>/`: `model.pt` when finished, `latest.pt` at every evaluation,
and `metrics.json` with the per-horizon numbers.

## Uncertainty-aware planning

The planners in `catcher/planning.py` act on a Gaussian belief over the ball's future
positions (`catcher/belief.py`) instead of the ground truth. The trajectory model isn't
wired in yet, so `SyntheticPredictor` stands in for it: it returns a per-axis mean and
std whose std grows with the prediction horizon and with each bounce ahead, and the
mean carries a matching random error, so the belief is calibrated. `GaussianBelief.sample`
draws whole trajectories from it (exact Gaussian marginals at every time, smooth in time).

| Planner | What it does |
|---|---|
| `MeanPlanner` | Observes for `t_plan`, then commits once to the earliest reachable belief mean. |
| `HedgePlanner` | Fixes the catch time at `t_plan`, then re-plans every 50 ms on the sharper belief until 0.1 s before the catch. Each re-plan samples trajectories from the belief, keeps those inside the 90% ellipsoid at the catch time, solves the catch IK for each, and steers toward the joint configuration that minimizes the worst-case joint-space time to any of them (the per-joint midrange). Each segment is a cubic that starts from the reference's current position and velocity, so the arm never stops between re-plans. `hedge=False` steers toward the mean's configuration instead (an ablation). |

```bash
uv run python scripts/compare_planners.py --stage 2 --n 60    # oracle vs mean vs mean-replan vs hedge
uv run python scripts/run_catch.py --stage 2 --seed 3 --planner hedge   # GIF of one belief-based catch
```

Catch rates with the default synthetic belief (std 5 mm + 6 cm/s of horizon + 3 cm per bounce ahead,
planning at t = 0.1 s):

| Planner | Stage 1 (40) | Stage 2 (60) | Stage 3 (40) |
|---|---|---|---|
| oracle (ground truth) | 39 | 48 | 32 |
| mean (plans once) | 8 | 5 | 1 |
| mean-replan | 37 | 30 | 16 |
| hedge | 37 | 29 | 18 |

Re-planning is what matters: planning once leaves 5-15 cm of belief std at the catch, more than
the 5.25 cm the cup allows. Hedging on top of re-planning is a wash so far (+2 on stage 3, -1 on
stage 2). The remaining misses are timing, not tracking: the controller follows the reference to
within a few mm, but the catch time is the earliest one the mean can reach, so a late correction
can't finish in time. Choosing a later catch time that covers the whole 90% ellipsoid was tried
and did worse (18/60 on stage 2): before the bounce the ellipsoid is wide, so it pushed catches
late into the throw.

Catches are always scored against the true ball. `run_catch(planner=...)` takes any
`(kin, throw, cone_deg) -> (catch index, joint reference)` callable.

## Conventions

- World frame: iiwa base at the origin, z up, thrower on the +x side of the room.
- The iiwa14 joint velocity/torque limits come from the KUKA spec sheet
  (`VELOCITY_LIMITS`, `TORQUE_LIMITS`); the Menagerie model doesn't include them.
  Arm contacts are disabled since the ball isn't a MuJoCo body yet.
- The default room is x ∈ [-1.5, 4.5], y ∈ [-3, 3], z ∈ [0, 3] (stage 4 randomizes it).
- Observations at 100 Hz over 3 s (300 samples), noise σ = 5 mm by default.
- A catch counts if, at the catch time, the ball is within the cup radius of the
  cup center and the cup axis is within 30° of the incoming ball direction (−v).

## Dataset arrays (`generate_dataset`)

`times (T,)`, `pos / vel / obs (N, T, 3)`, `panels (N, P, 13)` + `panel_mask (N, P)`
(center, normal, u-axis, half extents, restitution, friction), `cloud (N, 1024, 3)`
wall point cloud, `catch_index (N,)`, `bounces_before_catch (N,)`,
`bounce_times (N, B)` (padded with inf), `p0 / v0 (N, 3)`.

## Current oracle catch rates (ground-truth target, `ik` controller, 40 throws)

| Stage | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Caught | 39/40 | 33/40 | 32/40 | 33/40 |

Misses are mostly throws that reach the workspace faster than the arm can
reorient (often balls coming off the back wall, from behind the robot).
`diffik` toward a fixed pose drops further on bounce stages (25/40 on stage 2):
straight-line Cartesian motion is slower and stalls when the arm needs to flip.
