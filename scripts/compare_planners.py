"""Compare catch planners on identical throws and identical prediction errors.

oracle: plans on the ground-truth ball.
mean:   plans once on the mean of the belief (stand-in for the trajectory model).
mean-replan: re-plans every 50 ms on the sharper belief, steering toward the mean's catch
        configuration (same schedule as hedge, without the hedging; isolates what hedging adds).
hedge:  re-plans every 50 ms, steering toward the configuration that minimizes the worst-case
        joint-space time to any catch configuration in the belief's 90% region. With a
        multimodal belief it chooses between covering every mode and committing to one.

Beliefs: --predictor gaussian (SyntheticPredictor: one Gaussian, std growing with horizon and
bounces) or particle (ParticlePredictor: simulated particles, can split into several modes).
--pillar adds a pillar in front of the robot; --occlusion stops the belief from sharpening
while the pillar or a board hides the ball from the camera.

A catch counts if the simulated ball is still in the cup at the end ("caught"); "geometric"
is the older check (cup within tolerance of the true ball at the catch time, no contact).
"""

import argparse
from collections import defaultdict

import numpy as np

from catcher.mujoco_scene import build_model, ready_pose_clear, run_catch
from catcher.planning import HedgePlanner, MeanPlanner, particle_predictor_for, synthetic_predictor_for
from catcher.throws import sample_stage_throw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=40, help="number of throws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predictor", choices=["gaussian", "particle"], default="gaussian")
    ap.add_argument("--pillar", action="store_true", help="add a pillar just in front of the robot")
    ap.add_argument("--occlusion", action="store_true", help="the belief only sharpens while the ball is visible")
    ap.add_argument("--sigma0", type=float, default=0.005, help="gaussian: belief std at zero horizon (m)")
    ap.add_argument("--growth", type=float, default=0.06, help="gaussian: belief std growth per second of horizon (m/s)")
    ap.add_argument("--bounce-sigma", type=float, default=0.03, help="gaussian: extra belief std per bounce ahead (m)")
    ap.add_argument("--t-plan", type=float, default=0.1, help="seconds of ball observed before planning")
    args = ap.parse_args()

    if args.predictor == "gaussian":
        predictor_for = synthetic_predictor_for(args.seed, args.occlusion, sigma0=args.sigma0, growth=args.growth,
                                                bounce_sigma=args.bounce_sigma)
        belief = (f"gaussian belief, std = {args.sigma0} + {args.growth} m/s * horizon"
                  f" + {args.bounce_sigma} m/bounce")
    else:
        predictor_for = particle_predictor_for(args.seed, args.occlusion)
        belief = "particle belief"
    planners = {"oracle": None,
                "mean": MeanPlanner(predictor_for, args.t_plan),
                "mean-replan": HedgePlanner(predictor_for, args.t_plan, hedge=False),
                "hedge": HedgePlanner(predictor_for, args.t_plan)}
    rng = np.random.default_rng(args.seed)
    caught = {name: [] for name in planners}
    geometric = {name: [] for name in planners}
    contact = {name: [] for name in planners}
    by_bounces = {name: defaultdict(list) for name in planners}
    errors = {name: [] for name in planners}
    hedge_choices = defaultdict(int)
    reselected = {name: 0 for name in planners}
    for _ in range(args.n):
        throw = sample_stage_throw(args.stage, rng, with_pillar=args.pillar, room_ok=ready_pose_clear)
        model = build_model(throw.room)
        for name, planner in planners.items():
            r = run_catch(throw, model=model, planner=planner)
            caught[name].append(r.success)
            geometric[name].append(r.geometric_success)
            contact[name].append(r.arm_room_contact)
            by_bounces[name][throw.bounces_before_catch].append(r.success)
            errors[name].append(r.position_error)
            if isinstance(planner, HedgePlanner) and planner.last is not None:
                reselected[name] += planner.last.reselections > 0
                if name == "hedge":
                    for c in planner.last.choices:
                        hedge_choices[c] += 1
                planner.last = None

    extras = ", ".join(x for x, on in (("pillar", args.pillar), ("occlusion", args.occlusion)) if on)
    print(f"stage {args.stage}{' + ' + extras if extras else ''}, {args.n} throws, {belief}, planning at t = {args.t_plan} s")
    for name in planners:
        per_bounce = ", ".join(f"{b}b: {sum(v)}/{len(v)}" for b, v in sorted(by_bounces[name].items()))
        print(f"{name:>11}: caught {sum(caught[name]):2d}/{args.n} (geometric {sum(geometric[name]):2d}), "
              f"median error {100 * np.median(errors[name]):4.1f} cm, arm hit room {sum(contact[name])}  ({per_bounce})"
              + (f", catch time re-chosen in {reselected[name]}" if reselected[name] else ""))
    if any(c not in ("single",) for c in hedge_choices):
        total = sum(hedge_choices.values())
        print("hedge re-plans by choice: " + ", ".join(f"{c} {v / total:.0%}" for c, v in sorted(hedge_choices.items())))


if __name__ == "__main__":
    main()
