"""Compare catch planners on identical throws and identical synthetic prediction errors.

oracle: plans on the ground-truth ball.
mean:   plans on the mean of a Gaussian belief (synthetic stand-in for the trajectory model).
mean-replan: re-plans every 50 ms on the sharper belief, steering toward the mean's catch
        configuration (same schedule as hedge, without the hedging; isolates what hedging adds).
hedge:  re-plans every 50 ms, steering toward the configuration that minimizes the worst-case
        joint-space time to any catch configuration in the belief's 90% ellipsoid.
"""

import argparse
from collections import defaultdict

import numpy as np

from catcher.mujoco_scene import build_model, run_catch
from catcher.planning import HedgePlanner, MeanPlanner, synthetic_predictor_for
from catcher.throws import sample_stage_throw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=40, help="number of throws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma0", type=float, default=0.005, help="belief std at zero horizon (m)")
    ap.add_argument("--growth", type=float, default=0.06, help="belief std growth per second of horizon (m/s)")
    ap.add_argument("--bounce-sigma", type=float, default=0.03, help="extra belief std per bounce ahead (m)")
    ap.add_argument("--t-plan", type=float, default=0.1, help="seconds of ball observed before planning")
    args = ap.parse_args()

    predictor_for = synthetic_predictor_for(args.seed, sigma0=args.sigma0, growth=args.growth,
                                            bounce_sigma=args.bounce_sigma)
    planners = {"oracle": None,
                "mean": MeanPlanner(predictor_for, args.t_plan),
                "mean-replan": HedgePlanner(predictor_for, args.t_plan, hedge=False),
                "hedge": HedgePlanner(predictor_for, args.t_plan)}
    rng = np.random.default_rng(args.seed)
    caught = {name: [] for name in planners}
    by_bounces = {name: defaultdict(list) for name in planners}
    errors = {name: [] for name in planners}
    for _ in range(args.n):
        throw = sample_stage_throw(args.stage, rng)
        model = build_model(throw.room)
        for name, planner in planners.items():
            r = run_catch(throw, model=model, planner=planner)
            caught[name].append(r.success)
            by_bounces[name][throw.bounces_before_catch].append(r.success)
            errors[name].append(r.position_error)

    print(f"stage {args.stage}, {args.n} throws, belief std = {args.sigma0} + {args.growth} m/s * horizon"
          f" + {args.bounce_sigma} m/bounce, planning at t = {args.t_plan} s")
    for name in planners:
        per_bounce = ", ".join(f"{b}b: {sum(v)}/{len(v)}" for b, v in sorted(by_bounces[name].items()))
        print(f"{name:>11}: caught {sum(caught[name])}/{args.n}, "
              f"median error {100 * np.median(errors[name]):.1f} cm  ({per_bounce})")


if __name__ == "__main__":
    main()
