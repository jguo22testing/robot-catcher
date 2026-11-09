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

from .belief import GaussianBelief, ParticleBelief, ParticlePredictor, SyntheticPredictor
from .mujoco_scene import Q_READY, VELOCITY_LIMITS, CubicTrajectory, Kinematics, plan_ik_catch, solve_catch_ik
from .throws import Throw, in_catch_zone
from .visibility import visible

Predictor = Callable[[float], GaussianBelief | ParticleBelief]


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
    choices: list[str] = field(default_factory=list)  # per re-plan: "single", "cover" or "mode<i>"
    reselections: int = 0  # times the catch time was chosen again


@dataclass
class HedgePlanner:
    """Uncertainty-aware receding-horizon planner.

    1. At t_plan, pick the catch time t_c: the earliest reachable point on the belief's
       planning path, as MeanPlanner does. It stays fixed unless a later belief moves that
       path out of the catch zone at t_c, or more than `reselect_dist` from where it was
       aimed; then t_c is chosen again from the arm's current reference.
    2. Every replan_dt until t_c - final_lead, take the current (sharper) belief and
       sample trajectories from it; keep those whose position at t_c lies inside the
       90% ellipsoid. Solve the catch IK for the mean and every kept sample. The
       configuration minimizing the worst-case joint-space time
       max_i max_j |q_i - q|_j / vmax_j is the per-joint midrange of those solutions
       (the problem separates by joint).
    3. Steer there with a cubic that starts from the reference's current position and
       velocity and arrives at rest at t_c - arrive_lead (or as soon as the velocity
       limits allow). As the belief collapses the hedge converges to the catch pose.

    When the belief has several modes (e.g. the ball either clips a pillar edge or
    misses it), step 2 scores a few targets: the midrange over every mode's catch
    configurations ("cover") and each mode's own midrange ("mode<i>"). The score is the
    probability-weighted fraction of catch configurations reachable in time, assuming
    the arm reaches the target (or waits there) until the modes split and the belief
    tells it which one is real, then moves straight to the catch. It picks the best;
    ties go to the most likely mode.
    """

    predictor_for: Callable[[Throw], Predictor]
    t_plan: float = 0.1
    n_trajectories: int = 64
    seed: int = 0
    hedge: bool = True  # False: steer toward the mean's configuration (isolates the value of hedging)
    replan_dt: float = 0.05
    final_lead: float = 0.1  # last re-plan this long before the catch
    arrive_lead: float = 0.03  # arrive this long before the catch, to let tracking settle
    observe_lag: float = 0.05  # after the modes split, time until the belief shows which is real
    reselect_dist: float = 0.2  # re-choose the catch time if the planned catch point moves this far
    last: HedgePlan | None = field(default=None, init=False)

    @staticmethod
    def _midrange(kin, solutions):
        return np.clip((solutions.min(0) + solutions.max(0)) / 2, kin.lo, kin.hi)

    def _mode_solutions(self, kin, mode, q_guess, cone_deg):
        """Catch configurations for a mode's mean (first, if solvable) and its sampled points."""
        q_center, ok = solve_catch_ik(kin, mode.mean, mode.vel, q_guess, cone_deg)
        seed = q_center if ok else q_guess
        solutions = [q_center] if ok else []
        for p in mode.points:
            q, ok_p = solve_catch_ik(kin, p, mode.vel, seed, cone_deg)
            if ok_p:
                solutions.append(q)
        return np.array(solutions).reshape(-1, len(q_guess)), ok

    def _target(self, kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c):
        """Hedge (or mean) configuration for catching at index k under belief.
        Returns (target, solutions, choice); solutions[0] is the most likely catch configuration."""
        if not self.hedge:
            q_mean, ok = solve_catch_ik(kin, belief.plan_mean[k], belief.plan_vel[k], q_guess, cone_deg)
            q_mean = q_mean if ok else q_guess
            return q_mean, q_mean[None], "mean"
        modes = belief.modes(k, self.n_trajectories, rng)
        if len(modes) == 1:
            solutions, ok = self._mode_solutions(kin, modes[0], q_guess, cone_deg)
            if not ok:  # keep the mean's configuration first, as the fallback target
                solutions = np.vstack([q_guess, solutions])
            target = self._midrange(kin, solutions)
            if kin.collides(target):  # the midrange can land in the room even when every solution is clear
                target = solutions[0]
            return target, solutions, "single"

        per_mode = [self._mode_solutions(kin, m, q_guess, cone_deg)[0] for m in modes]
        nonempty = [s for s in per_mode if len(s)]
        if not nonempty:
            return q_guess, q_guess[None], "none"
        everything = np.vstack(nonempty)
        candidates = [("cover", self._midrange(kin, everything))]
        candidates += [(f"mode{i}", self._midrange(kin, s)) for i, s in enumerate(per_mode) if len(s)]
        t_known = max(t_r, min(m.split_time for m in modes) + self.observe_lag)
        deadline = t_c - self.arrive_lead

        def p_catch(q):
            t_free = max(t_known, t_r + min_move_time(q_now, q))  # at q, and the real mode is known
            return sum(m.weight * np.mean([t_free + min_move_time(q, qi) <= deadline for qi in s])
                       for m, s in zip(modes, per_mode) if len(s))

        # Ties go to the most likely mode ("mode0" first), not to "cover": when nothing is
        # reachable in time, a pose between the modes is the worst place to wait.
        order = {name: rank for rank, name in enumerate([n for n, _ in candidates[1:]] + ["cover"])}
        scored = [(p_catch(q), -order[name], name, q) for name, q in candidates if not kin.collides(q)]
        likely_first = np.vstack([nonempty[0], everything])
        if not scored:
            return nonempty[0][0], likely_first, "mode0"
        _, _, name, target = max(scored, key=lambda c: c[:2])
        return target, likely_first, name

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
        aim = belief.plan_mean[k]

        q_guess = to_mean.q1
        segments: list[tuple[float, HermiteTrajectory]] = []
        q_now, v_now = Q_READY.copy(), np.zeros_like(Q_READY)
        t_r, i = self.t_plan, 0
        while i == 0 or t_r <= t_c - self.final_lead + 1e-9:
            if segments:  # the reference's state at t_r, where the new segment starts
                t0, traj = segments[-1]
                q_now, v_now, _ = traj(t_r - t0)
            belief = predictor(t_r)
            if i > 0 and (not in_catch_zone(belief.plan_mean[k])
                          or np.linalg.norm(belief.plan_mean[k] - aim) > self.reselect_dist):
                k_new, to_new = plan_ik_catch(kin, throw, cone_deg, belief=belief, q_start=q_now, t_start=t_r)
                if not np.array_equal(to_new.q1, to_new.q0):
                    k, t_c, q_guess = k_new, float(throw.times[k_new]), to_new.q1
                    self.last.reselections += 1
            aim = belief.plan_mean[k]
            target, solutions, choice = self._target(kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c)
            if i == 0:
                worst = lambda q: float(np.max(np.abs(solutions - q) / VELOCITY_LIMITS))
                q_mean0 = solutions[0]
                self.last = HedgePlan(target, len(solutions), worst(target), worst(q_mean0), 0)
            self.last.choices.append(choice)
            self.last.n_replans += 1
            t_r, i = t_r + self.replan_dt, i + 1
            q_guess = solutions[0]
            t_seg = t_r - self.replan_dt  # this re-plan's time (t_r already points at the next one)
            segment = self._segment(kin, q_now, v_now, target, t_c - self.arrive_lead - t_seg)
            if segment is None:
                segment = self._segment(kin, q_now, v_now, solutions[0], t_c - self.arrive_lead - t_seg)
            if segment is None:
                if segments:  # keep following the previous, collision-free segment
                    continue
                return k, SegmentedReference([(self.t_plan, CubicTrajectory(Q_READY, Q_READY))])
            segments.append((t_seg, segment))
        return k, SegmentedReference(segments)


def synthetic_predictor_for(seed: int = 0, occlusion: bool = False, **kwargs) -> Callable[[Throw], SyntheticPredictor]:
    """predictor_for factory: a SyntheticPredictor per throw, seeded from the throw so that
    every planner compared on the same throw sees the same prediction errors. With
    occlusion, the belief only sharpens while the camera can see the ball."""
    def make(throw: Throw) -> SyntheticPredictor:
        throw_seed = int(abs(throw.p0).sum() * 1e6) % (2**31) + seed
        seen = visible(throw.room, throw.pos) if occlusion else None
        return SyntheticPredictor(throw, seed=throw_seed, seen=seen, **kwargs)
    return make


def particle_predictor_for(seed: int = 0, occlusion: bool = False, **kwargs) -> Callable[[Throw], ParticlePredictor]:
    """predictor_for factory for the multimodal ParticlePredictor, seeded from the throw like
    synthetic_predictor_for. The last throw's predictor (and its cached beliefs) is reused,
    so planners compared on the same throw share it."""
    last: dict = {}

    def make(throw: Throw) -> ParticlePredictor:
        if last.get("throw") is not throw:
            throw_seed = int(abs(throw.p0).sum() * 1e6) % (2**31) + seed
            seen = visible(throw.room, throw.pos) if occlusion else None
            last.update(throw=throw, predictor=ParticlePredictor(throw, seed=throw_seed, seen=seen, **kwargs))
        return last["predictor"]
    return make
