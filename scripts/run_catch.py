"""Simulate the iiwa catching a NumPy-simulated ball in MuJoCo.

Render one throw to a GIF:
    python scripts/run_catch.py --stage 2 --seed 3 [--out outputs/catch.gif]

Replay it in the interactive viewer (macOS needs `mjpython` instead of `python`):
    mjpython scripts/run_catch.py --stage 2 --seed 3 --viewer

Headless batch evaluation (oracle catch rate):
    python scripts/run_catch.py --stage 2 --eval 50 [--controller diffik]

Plan on a noisy belief instead of the ground truth (see catcher.planning), optionally with
a pillar in front of the robot, occlusion, and the multimodal particle belief:
    python scripts/run_catch.py --stage 2 --seed 3 --planner uncertainty-aware [--pillar] [--occlusion] [--predictor particle]
"""

import argparse
import time
from pathlib import Path

import numpy as np

from catcher.mujoco_scene import (N_ARM, add_path_to_scene, build_model, default_camera, ready_pose_clear,
                                  render_frames, run_catch, set_ball)
from catcher.planning import HedgePlanner, MeanPlanner, particle_predictor_for, synthetic_predictor_for
from catcher.throws import sample_stage_throw

PLANNERS = {"oracle": lambda pf: None,
            "plan-once": lambda pf: MeanPlanner(pf),
            "mean-only": lambda pf: HedgePlanner(pf, hedge=False),
            "uncertainty-aware": lambda pf: HedgePlanner(pf)}


def save_gif(frames, path: Path, fps: float):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)


def replay_in_viewer(model, result, path_points):
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = default_camera()
        viewer.cam.lookat[:], viewer.cam.distance = cam.lookat, cam.distance
        viewer.cam.azimuth, viewer.cam.elevation = cam.azimuth, cam.elevation
        while viewer.is_running():
            start = time.time()
            for t, q, p in zip(result.times, result.q, result.ball):
                if not viewer.is_running():
                    break
                data.qpos[:N_ARM] = q
                set_ball(data, p)
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                add_path_to_scene(viewer.user_scn, path_points[::3])
                viewer.sync()
                time.sleep(max(0.0, t - (time.time() - start)))
            time.sleep(1.0)  # pause, then loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--controller", choices=["ik", "diffik"], default="ik")
    ap.add_argument("--planner", choices=list(PLANNERS), default="oracle",
                    help="plan on the ground truth (oracle) or on a synthetic belief")
    ap.add_argument("--predictor", choices=["gaussian", "particle"], default="gaussian")
    ap.add_argument("--pillar", choices=["front", "blocking"], default=None,
                    help="front: pillar 0.75-1.4 m out; blocking: straight ahead, the arm must go left or right of it")
    ap.add_argument("--occlusion", action="store_true", help="the belief only sharpens while the ball is visible")
    ap.add_argument("--eval", type=int, default=0, help="run N throws headless and report the catch rate")
    ap.add_argument("--viewer", action="store_true", help="replay in the MuJoCo viewer (use mjpython on macOS)")
    ap.add_argument("--out", type=str, default=None, help="GIF path (default outputs/catch_stage{S}_seed{N}.gif)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    factory = particle_predictor_for if args.predictor == "particle" else synthetic_predictor_for
    planner = PLANNERS[args.planner](factory(0, args.occlusion))
    label = args.controller if planner is None else args.planner
    if args.eval:
        results = [run_catch(sample_stage_throw(args.stage, rng, pillar=args.pillar, room_ok=ready_pose_clear),
                             controller=args.controller,
                             planner=planner) for _ in range(args.eval)]
        errs = np.array([r.position_error for r in results])
        print(f"stage {args.stage} / {label}: caught {sum(r.success for r in results)}/{args.eval} "
              f"(geometric {sum(r.geometric_success for r in results)}), "
              f"median position error {100 * np.median(errs):.1f} cm, arm hit room {sum(r.arm_room_contact for r in results)}")
        return

    throw = sample_stage_throw(args.stage, rng, pillar=args.pillar, room_ok=ready_pose_clear)
    model = build_model(throw.room)
    r = run_catch(throw, model=model, controller=args.controller, planner=planner)
    print(f"bounces before catch: {throw.bounces_before_catch}, catch time {r.t_catch:.2f}s")
    print(f"cup-ball error {100 * r.position_error:.1f} cm, axis angle {r.axis_angle_deg:.0f} deg, "
          f"ball {'stayed in the cup -> CAUGHT' if r.success else 'not in the cup at the end -> MISSED'}"
          f"{', arm hit the room' if r.arm_room_contact else ''}")
    path_points = throw.pos[throw.times <= r.times[-1]]
    if args.viewer:
        replay_in_viewer(model, r, path_points)
        return
    suffix = (f"_pillar-{args.pillar}" if args.pillar else "") + ("" if planner is None else f"_{args.planner}")
    out = Path(args.out or f"outputs/catch_stage{args.stage}_seed{args.seed}{suffix}.gif")
    fps = 1.0 / np.diff(r.times).mean()
    save_gif(render_frames(model, r, path_points), out, fps)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
