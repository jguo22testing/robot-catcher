"""Catch planners that act on a Gaussian belief over the ball's future (catcher.belief).

Both planners are callables (kin, throw, cone_deg) -> (catch index, joint reference),
the interface `run_catch(planner=...)` expects. They only see the belief, never the
ground truth; the catch is scored against the true ball.

MeanPlanner:   commit once to the earliest reachable belief mean.
HedgePlanner:  fix the catch time once, then re-plan every `replan_dt` on the sharper
               belief, each time steering toward a joint configuration that is quick to
               reach from any catch configuration in the 90% ellipsoid. `hedge=False`
               steers toward the mean's configuration instead (the re-planning ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .belief import GaussianBelief, SyntheticPredictor
from .mujoco_scene import Q_READY, VELOCITY_LIMITS, CubicTrajectory, Kinematics, plan_ik_catch, solve_catch_ik
from .throws import Throw

Predictor = Callable[[float], GaussianBelief]


@dataclass
class HermiteTrajectory:
    """Joint-space cubic from (q0, v0) to (q1, 0) over T seconds; holds q1 afterwards."""

    q0: np.ndarray
    v0: np.ndarray
    q1: np.ndarray
    T: float

    def __call__(self, t: float):
        if t >= self.T:
            return self.q1, np.zeros_like(self.q1), np.zeros_like(self.q1)
        s, T = max(t, 0.0) / self.T, self.T
        dq, m = self.q1 - self.q0, self.v0 * T
        q = self.q0 + m * (s**3 - 2 * s**2 + s) + dq * (3 * s**2 - 2 * s**3)
        v = (m * (3 * s**2 - 4 * s + 1) + dq * (6 * s - 6 * s**2)) / T
        a = (m * (6 * s - 4) + dq * (6 - 12 * s)) / T**2
        return q, v, a


class SegmentedReference:
    """Joint reference made of trajectory segments, each starting at its own absolute time.

    Holds the first segment's start pose until then; each segment runs until the next starts.
    """

    def __init__(self, segments: list[tuple[float, Callable]]):
        self.segments = sorted(segments, key=lambda s: s[0])

    def __call__(self, t: float):
        started = [s for s in self.segments if s[0] <= t]
        if not started:
            q0 = self.segments[0][1].q0
            return q0, np.zeros_like(q0), np.zeros_like(q0)
        t0, traj = started[-1]
        return traj(t - t0)


def min_move_time(q0: np.ndarray, q1: np.ndarray) -> float:
    """Fastest rest-to-rest cubic between two configurations (CubicTrajectory's duration)."""
    return CubicTrajectory(q0, q1).T


@dataclass
class MeanPlanner:
    """Observe until t_plan, then go to the earliest reachable belief mean."""

    predictor_for: Callable[[Throw], Predictor]
    t_plan: float = 0.1

    def __call__(self, kin: Kinematics, throw: Throw, cone_deg: float):
        belief = self.predictor_for(throw)(self.t_plan)
        k, traj = plan_ik_catch(kin, throw, cone_deg, belief=belief, t_start=self.t_plan)
        return k, SegmentedReference([(self.t_plan, traj)])


@dataclass
class HedgePlan:
    """What the last HedgePlanner call decided at t_plan (for logging and tests)."""

    q_hedge: np.ndarray
    n_samples: int  # sampled catch configurations the first hedge covers (incl. the mean's)
    worst_case_time: float  # slowest joint-space time from q_hedge to any of them
    mean_worst_case_time: float  # same, if the arm had aimed at the mean's configuration
    n_replans: int


@dataclass
class HedgePlanner:
    """Uncertainty-aware receding-horizon planner.

    1. At t_plan, pick the catch time t_c: the earliest reachable belief mean, as
       MeanPlanner does. It stays fixed from here on.
    2. Every replan_dt until t_c - final_lead, take the current (sharper) belief and
       sample trajectories from it; keep those whose position at t_c lies inside the
       90% ellipsoid. Solve the catch IK for the mean and every kept sample. The
       configuration minimizing the worst-case joint-space time
       max_i max_j |q_i - q|_j / vmax_j is the per-joint midrange of those solutions
       (the problem separates by joint).
    3. Steer there with a cubic that starts from the reference's current position and
       velocity and arrives at rest at t_c - arrive_lead (or as soon as the velocity
       limits allow). As the belief collapses the hedge converges to the catch pose.
    """

    predictor_for: Callable[[Throw], Predictor]
    t_plan: float = 0.1
    n_trajectories: int = 64
    seed: int = 0
    hedge: bool = True  # False: steer toward the mean's configuration (isolates the value of hedging)
    replan_dt: float = 0.05
    final_lead: float = 0.1  # last re-plan this long before the catch
    arrive_lead: float = 0.03  # arrive this long before the catch, to let tracking settle
    last: HedgePlan | None = field(default=None, init=False)

    def _target(self, kin, belief, k, q_guess, cone_deg, rng):
        """Hedge (or mean) configuration for catching at index k under belief.
        Returns (target, solutions)."""
        q_mean, ok = solve_catch_ik(kin, belief.mean[k], belief.vel[k], q_guess, cone_deg)
        if not ok:
            q_mean = q_guess
        if not self.hedge:
            return q_mean, q_mean[None]
        solutions = [q_mean]
        for p in belief.ellipsoid_samples(k, self.n_trajectories, rng):
            q, ok = solve_catch_ik(kin, p, belief.vel[k], q_mean, cone_deg)
            if ok:
                solutions.append(q)
        solutions = np.array(solutions)
        target = np.clip((solutions.min(0) + solutions.max(0)) / 2, kin.lo, kin.hi)
        if kin.collides(target):  # the midrange can land in the room even when every solution is clear
            target = q_mean
        return target, solutions

    @staticmethod
    def _segment(kin, q_now, v_now, target, time_left):
        """Cubic from the current reference state to target, or None if its path hits the room."""
        segment = HermiteTrajectory(q_now.copy(), v_now.copy(), target, max(time_left, min_move_time(q_now, target)))
        return None if kin.path_collides(segment, segment.T) else segment

    def __call__(self, kin: Kinematics, throw: Throw, cone_deg: float):
        predictor = self.predictor_for(throw)
        belief = predictor(self.t_plan)
        k, to_mean = plan_ik_catch(kin, throw, cone_deg, belief=belief, t_start=self.t_plan)
        if np.array_equal(to_mean.q1, to_mean.q0):  # nothing reachable: hold
            return k, SegmentedReference([(self.t_plan, to_mean)])
        rng = np.random.default_rng(self.seed)
        t_c = float(throw.times[k])

        q_guess = to_mean.q1
        segments: list[tuple[float, HermiteTrajectory]] = []
        q_now, v_now = Q_READY.copy(), np.zeros_like(Q_READY)
        replan_times = np.arange(self.t_plan, max(t_c - self.final_lead, self.t_plan) + 1e-9, self.replan_dt)
        for i, t_r in enumerate(replan_times):
            if segments:  # the reference's state at t_r, where the new segment starts
                t0, traj = segments[-1]
                q_now, v_now, _ = traj(t_r - t0)
            target, solutions = self._target(kin, predictor(t_r), k, q_guess, cone_deg, rng)
            if i == 0:
                worst = lambda q: float(np.max(np.abs(solutions - q) / VELOCITY_LIMITS))
                q_mean0 = solutions[0]
                self.last = HedgePlan(target, len(solutions), worst(target), worst(q_mean0), len(replan_times))
            q_guess = solutions[0]
            segment = self._segment(kin, q_now, v_now, target, t_c - self.arrive_lead - t_r)
            if segment is None:
                segment = self._segment(kin, q_now, v_now, solutions[0], t_c - self.arrive_lead - t_r)
            if segment is None:
                if segments:  # keep following the previous, collision-free segment
                    continue
                return k, SegmentedReference([(self.t_plan, CubicTrajectory(Q_READY, Q_READY))])
            segments.append((t_r, segment))
        return k, SegmentedReference(segments)


def synthetic_predictor_for(seed: int = 0, **kwargs) -> Callable[[Throw], SyntheticPredictor]:
    """predictor_for factory: a SyntheticPredictor per throw, seeded from the throw so that
    every planner compared on the same throw sees the same prediction errors."""
    def make(throw: Throw) -> SyntheticPredictor:
        throw_seed = int(abs(throw.p0).sum() * 1e6) % (2**31) + seed
        return SyntheticPredictor(throw, seed=throw_seed, **kwargs)
    return make
