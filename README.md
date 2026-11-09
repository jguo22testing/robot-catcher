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
| `catcher/room.py` | `Panel` (bounded two-sided rectangle) + `Material` (restitution, friction). `make_room(stage, rng, with_pillar)` builds the procedural room for stages 1–4, optionally with a concrete pillar just in front of the robot. Point-cloud sampling and a flat `(P, 13)` panel encoding for the model. |
| `catcher/ball_sim.py` | Event-driven point-ball sim. Flight is exact ballistic motion; bounces are found as quadratic roots, so there's no time step and no tunnelling. Bounce: `v_n' = -e v_n`, `v_t' = (1 - friction) v_t`. |
| `catcher/throws.py` | Catch zone (reach shell around the shoulder), rejection-sampled throws per stage (stage 1: 0 bounces, 2: exactly 1, 3: 1–3, 4: 1–4), Gaussian observation noise, and `generate_dataset` → fixed-shape arrays. |
| `catcher/mujoco_scene.py` | iiwa14 + cup + room panels built with `MjSpec`; computed-torque joint control driven by either IK + joint-space cubic (`ik`) or differential IK (`diffik`). IK is damped least squares on MuJoCo Jacobians with the cup-axis cone constraint (random restarts on failure), and rejects poses and joint paths that hit the room. The ball is a free MuJoCo body: replayed from the NumPy sim until it comes within 35 cm of the arm, then simulated, so it can land in the cup, bounce out, or hit the arm. Offscreen rendering helpers. |
| `catcher/visibility.py` | Camera behind and above the robot; line-of-sight test against boards and pillar faces. |
| `models/kuka_iiwa_14/` | iiwa14 model vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0). |
| `scripts/simulate_ball.py` | Plot a few throws (3D + side view, bounces, catch points). |
| `scripts/generate_dataset.py` | Write a stage dataset to `data/*.npz`. |
| `scripts/run_catch.py` | Simulate one catch and save a GIF, replay it in the MuJoCo viewer (`--viewer`), or `--eval N` for a headless catch rate. `--planner` (`plan-once`, `mean-only` or `uncertainty-aware`) plans on a belief instead of the ground truth; `--predictor`, `--pillar` and `--occlusion` as in `compare_planners.py`. |

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
the robot, so it passes through the base and catch zone. The arm is only in the MuJoCo sim,
where the ball is simulated once it comes near the arm.

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
| plan-once (`MeanPlanner`) | Observes for `t_plan`, then commits once to the earliest reachable belief mean. |
| uncertainty-aware (`HedgePlanner`) | Fixes the catch time at `t_plan`, then re-plans every 50 ms on the sharper belief until 0.1 s before the catch. Each re-plan samples trajectories from the belief, keeps those inside the 90% ellipsoid at the catch time, solves the catch IK for each, and steers toward the joint configuration that minimizes the worst-case joint-space time to any of them (the per-joint midrange). Each segment is a cubic that starts from the reference's current position and velocity, so the arm never stops between re-plans. mean-only (`HedgePlanner(hedge=False)`) runs the same schedule but steers toward the mean's configuration, ignoring the uncertainty. |

```bash
uv run python scripts/compare_planners.py --stage 2 --n 60    # oracle vs plan-once vs mean-only vs uncertainty-aware
uv run python scripts/run_catch.py --stage 2 --seed 3 --planner uncertainty-aware   # GIF of one belief-based catch
```

Two predictors stand in for the trajectory model:

| Predictor | What it is |
|---|---|
| `SyntheticPredictor` (`--predictor gaussian`) | One Gaussian: std 5 mm + 6 cm per second of horizon + 3 cm per bounce ahead. |
| `ParticlePredictor` (`--predictor particle`) | 96 simulated balls from a noisy estimate of the ball's state at its last sighting (5 mm, 6 cm/s), each with slightly perturbed bounce materials. It can split into several outcomes (e.g. the ball clips a pillar edge or misses it); `ParticleBelief.modes` groups particles by the surfaces they bounce off. Planners use the most likely outcome's path, not the overall mean, which can lie between outcomes. |

`--pillar` adds a concrete pillar 0.75–1.4 m in front of the robot. `--occlusion` stops the
belief from sharpening while the pillar or a board hides the ball from a camera behind the robot.
With several outcomes, the uncertainty-aware planner scores waiting between them ("cover") against
committing to one and takes the more likely catch. Every planner re-chooses the catch time
if the predicted ball leaves the catch zone at that time or moves 20 cm from where it was aimed.

Results (60 throws, balls caught and kept in the cup):

| Setup | oracle | plan-once | mean-only | uncertainty-aware |
|---|---|---|---|---|
| Stage 2, gaussian | 44 | 8 | 28 | 28 |
| Stage 2 + pillar, gaussian | 49 | 16 | 31 | 31 |
| Stage 2 + pillar + occlusion, gaussian | 49 | 16 | 30 | 30 |
| Stage 2 + pillar, particle | 49 | 29 | 37 | 35 |
| Stage 2 + pillar + occlusion, particle | 49 | 29 | 37 | 35 |
| Stage 3 + pillar + occlusion, particle | 40 | 17 | 33 | 30 |

- Re-planning is still what matters. Using the uncertainty doesn't help: uncertainty-aware
  ties with mean-only on the Gaussian belief and loses 2–3 catches on the particle belief.
- With the particle belief, "cover" never scored strictly better than committing to one
  outcome; it only won ties. Ties now go to the most likely outcome.
- Occlusion costs at most one catch: when the pillar hides the ball it is mostly before
  the bounce, while the belief is wide anyway.
- The particle and Gaussian rows aren't a planner comparison: the particle predictor knows
  the room exactly and is sharper after the bounce, which is why every planner does better with it.
- The arm never touches the room in any of these runs.

### Blocking pillar: going left or right

`--pillar blocking` puts a thin pillar straight ahead of the robot, inside its reach (center
0.48–0.55 m out, half-width 7–11 cm), and throws lobs past one side of it, 11–25 cm from its
face (`sample_throw_beside_pillar`). The room gets a 7 m ceiling so the lobs can last
1.6–2.1 s. The arm waits tucked behind the pillar (`Q_TUCKED`) and has to reach around it:
about 2 s to either side, and about 3.9 s to switch sides (joints 1, 3 and 5 each turn ~2 rad).
So it has to commit to a side early, and a wrong guess can't be undone. Paths are IK goals plus
straight joint-space moves, with waypoints when those are blocked (`Kinematics.route`: swing
the base first, or go back through the tucked pose). With a blocking pillar the
uncertainty-aware planner chooses left, right or wait each re-plan (`_side_target`).

Results (stage 1 rooms + blocking pillar, particle belief, 40 throws):

| Velocity error of the estimate | oracle | plan-once | mean-only | uncertainty-aware |
|---|---|---|---|---|
| 0.06 m/s | 37 | 5 | 26 | 24 |
| 0.15 m/s | 37 | 5 | 23 | 18 |
| 0.3 m/s | 37 | 0 | 9 | 5 |

The commitment is forced immediately: reaching a side takes about as long as the lob is in the
air, so there is no slack to wait for information. Committing to the more likely side at once
is then the best policy, and that is what mean-only does. The uncertainty-aware planner's only
extra option is to wait, which here can only cost time; its forecast of when the side becomes
known is also optimistic, so it still waits in 9–17% of re-plans. It could pay off where there
is slack, e.g. with faster time-scaling of the moves (they are rest-to-rest cubics today).

`docs/pillar_catch.html` (open it in a browser) shows a mean-only catch of a ball bouncing
off the pillar, with these results.

Catches are always scored against the true ball. `run_catch(planner=...)` takes any
`(kin, throw, cone_deg) -> (catch index, joint reference)` callable.

## Conventions

- World frame: iiwa base at the origin, z up, thrower on the +x side of the room.
- The iiwa14 joint velocity/torque limits come from the KUKA spec sheet
  (`VELOCITY_LIMITS`, `TORQUE_LIMITS`); the Menagerie model doesn't include them.
- Contacts in MuJoCo: arm–room, ball–room and ball–arm (including the cup); the arm never
  collides with itself. The cup is 1 cm thick with a stiff, overdamped contact (a padded
  cup): with MuJoCo's default soft contact a 7 m/s ball sinks straight through it.
- The default room is x ∈ [-1.5, 4.5], y ∈ [-3, 3], z ∈ [0, 3] (stage 4 randomizes it).
- Observations at 100 Hz over 3 s (300 samples), noise σ = 5 mm by default.
- A catch counts if the simulated ball is still inside the cup at the end of the run
  (0.5 s after the catch time). The older geometric check is kept as `geometric_success`:
  at the catch time the ball is within the cup radius of the cup center and the cup axis
  is within 30° of the incoming ball direction (−v).

## Dataset arrays (`generate_dataset`)

`times (T,)`, `pos / vel / obs (N, T, 3)`, `panels (N, P, 13)` + `panel_mask (N, P)`
(center, normal, u-axis, half extents, restitution, friction), `cloud (N, 1024, 3)`
wall point cloud, `catch_index (N,)`, `bounces_before_catch (N,)`,
`bounce_times (N, B)` (padded with inf), `p0 / v0 (N, 3)`.

## Current oracle catch rates (ground-truth target, `ik` controller, 40 throws)

A catch is the simulated ball staying in the cup; the old geometric check is in brackets.

| Stage | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Caught | 40/40 (39) | 30/40 (31) | 18/40 (18) | 21/40 (25) |

Misses are mostly throws that reach the workspace faster than the arm can
reorient, and on stages 3–4 balls that are still rising when they reach the robot:
the cup would have to face down to meet them, so the planner only aims at descending
balls. `diffik` toward a fixed pose catches 26/40 on stage 2. The evaluation scripts skip
rooms where a random board passes through the robot's ready pose.
