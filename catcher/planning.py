"""Catch planners that act on a Gaussian belief over the ball's future (catcher.belief).

Both planners are callables (kin, throw, cone_deg) -> (catch index, joint reference),
the interface `run_catch(planner=...)` expects. They only see the belief, never the
ground truth; the catch is scored against the true ball.

MeanPlanner:    commit once, at t_plan, to the earliest reachable belief mean (plan-once).
ReplanPlanner:  re-plan every `replan_dt` on the sharper belief. use_uncertainty=False
                steers toward the mean's catch pose (mean-only); True scores candidate
                targets against scenarios drawn from the belief (uncertainty-aware).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .belief import GaussianBelief, ParticleBelief, ParticlePredictor, SyntheticPredictor
from .mujoco_scene import VELOCITY_LIMITS, CubicTrajectory, Kinematics, plan_ik_catch, solve_catch_ik
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
class ReplanLog:
    """What the last ReplanPlanner call did (for logging and tests)."""

    n_replans: int = 0
    choices: list[str] = field(default_factory=list)  # per re-plan, the candidate it steered toward
    reselections: int = 0  # times the catch time was chosen again


def _move_time(a: np.ndarray, b: np.ndarray) -> float:
    """Rest-to-rest joint-space move time (as CubicTrajectory, without its 50 ms floor)."""
    return 1.5 * float(np.max(np.abs(b - a) / VELOCITY_LIMITS))


@dataclass
class ReplanPlanner:
    """Receding-horizon catch planner, "mean-only" or "uncertainty-aware".

    1. At t_plan, pick the catch time t_c: the earliest reachable point on the belief's
       planning path, as MeanPlanner does. It stays fixed unless a later belief moves that
       path out of the catch zone at t_c, or more than `reselect_dist` from where it was
       aimed; then t_c is chosen again from the arm's current reference.
    2. Every replan_dt until t_c - final_lead, choose a target pose on the current belief
       and steer there with a cubic that starts from the reference's current position and
       velocity and arrives at rest at t_c - arrive_lead (or as soon as the velocity limits
       allow). Blocked moves follow Kinematics.route's waypoints.

    Mean-only (use_uncertainty=False) targets the catch pose for the belief's planning path.

    Uncertainty-aware (use_uncertainty=True) is a scenario-based contingency planner. It
    draws n_scenarios possible ball positions at t_c from the belief (every mode, weighted)
    and solves each one's catch pose. Candidate targets: the mean's pose; the center pose of
    each group of scenarios (a mode, or a side of a blocking pillar); the joint-space
    midrange of all scenario poses ("cover"); and waiting (the ready pose with a blocking
    pillar, otherwise stopping where the arm is). It steers toward the candidate under
    which the most scenario weight can still be caught, scoring each scenario as:

    - Information: the belief's spread shrinks to zero at t_c. A scenario whose catch point
      is m spreads from where the candidate aims becomes distinguishable at
      t_res = t_c - (t_c - t_r) * m / resolve_sigmas (now, if m >= resolve_sigmas): big
      corrections are known early, small ones late.
    - Feasibility: the arm heads for the candidate until t_res, then goes straight to the
      scenario's catch pose; caught if it gets there by the deadline. A scenario within
      cup_tolerance of the candidate's aim point needs no correction.
    - Waiting gives up at least one re-plan cycle (it acts on the next belief at the
      earliest), so a planner that keeps expecting news can't wait forever.
    """

    predictor_for: Callable[[Throw], Predictor]
    t_plan: float = 0.1
    use_uncertainty: bool = True
    n_scenarios: int = 64
    seed: int = 0
    replan_dt: float = 0.05
    final_lead: float = 0.1  # last re-plan this long before the catch
    arrive_lead: float = 0.03  # arrive this long before the catch, to let tracking settle
    reselect_dist: float = 0.2  # re-choose the catch time if the planned catch point moves this far
    resolve_sigmas: float = 2.5  # a scenario is distinguishable this many spreads from the aim (90% in 3D)
    cup_tolerance: float = 0.03  # catch-point error the cup absorbs without a correction
    last: ReplanLog | None = field(default=None, init=False)

    @staticmethod
    def _mean_pose(kin, belief, k, q_guess, cone_deg):
        q, ok = solve_catch_ik(kin, belief.plan_mean[k], belief.plan_vel[k], q_guess, cone_deg)
        if not ok and kin.pillar is not None:  # the mean may have moved to the other side
            q, ok = solve_catch_ik(kin, belief.plan_mean[k], belief.plan_vel[k], kin.q_ready, cone_deg)
        return q if ok else q_guess

    def _target(self, kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c):
        """Target pose for catching at index k. Returns (target, next IK seed, choice)."""
        q_mean = self._mean_pose(kin, belief, k, q_guess, cone_deg)
        if not self.use_uncertainty:
            return q_mean, q_mean, "mean"
        return self._scenario_target(kin, belief, k, q_guess, q_mean, cone_deg, rng, q_now, t_r, t_c)

    def _scenario_target(self, kin, belief, k, q_guess, q_mean, cone_deg, rng, q_now, t_r, t_c):
        modes = belief.modes(k, self.n_scenarios, rng)
        modes = [m for m in modes if len(m.points)]
        if not modes:
            return q_mean, q_mean, "mean"
        p = np.vstack([m.points for m in modes])
        v = np.vstack([np.tile(m.vel, (len(m.points), 1)) for m in modes])
        w = np.concatenate([np.full(len(m.points), m.weight / len(m.points)) for m in modes])
        if kin.pillar is not None:  # group by side of the pillar
            groups = np.where(p[:, 1] >= kin.pillar[0][1], 1, -1)
            names = {1: "left", -1: "right"}
        else:  # group by mode
            groups = np.concatenate([np.full(len(m.points), i) for i, m in enumerate(modes)])
            names = {i: f"mode{i}" for i in range(len(modes))}
        # Balls outside the catch zone (e.g. bounced back off the pillar) are misses whatever the
        # arm does: they lower every candidate's score equally, so they are left out.
        catchable = in_catch_zone(p)
        if not catchable.any():
            return q_mean, q_mean, "mean"
        p, v, w, groups = p[catchable], v[catchable], w[catchable], groups[catchable]

        # Each group's center pose, then each scenario's catch pose seeded from its group's.
        centers = {}
        for g in np.unique(groups):
            on = groups == g
            p_c, v_c = np.average(p[on], axis=0, weights=w[on]), np.average(v[on], axis=0, weights=w[on])
            same_side = kin.pillar is None or kin.side(q_guess) == g
            q_c, ok = solve_catch_ik(kin, p_c, v_c, q_guess if same_side else kin.q_ready, cone_deg)
            if not ok and kin.pillar is not None:
                q_c, ok = solve_catch_ik(kin, p_c, v_c, kin.q_ready, cone_deg)
            if ok:
                centers[int(g)] = (q_c, float(w[on].sum()))
        poses, solved = np.zeros((len(p), len(q_guess))), np.zeros(len(p), dtype=bool)
        for i in range(len(p)):
            seed = centers[int(groups[i])][0] if int(groups[i]) in centers else q_mean
            poses[i], solved[i] = solve_catch_ik(kin, p[i], v[i], seed, cone_deg)

        candidates = [("mean", q_mean, False)]
        if len(centers) > 1 or kin.pillar is not None:
            for g, (q_c, _) in sorted(centers.items(), key=lambda c: -c[1][1]):  # likelier group first
                candidates.append((names[g], q_c, False))
        if solved.sum() > 1:
            cover = np.clip((poses[solved].min(0) + poses[solved].max(0)) / 2, kin.lo, kin.hi)
            if not kin.collides(cover):
                candidates.append(("cover", cover, False))
        candidates.append(("wait", kin.q_ready, True) if kin.pillar is not None else ("hold", q_now, True))

        mu = np.average(p, axis=0, weights=w)
        sigma = np.maximum(np.sqrt(np.average((p - mu) ** 2, axis=0, weights=w)), 0.005)
        t_k, deadline = float(belief.times[k]), t_c - self.arrive_lead

        def score(q, waiting):
            aim = mu if waiting else kin(q)[0]
            t_go = t_r + (self.replan_dt if waiting else 0.0)
            T_to = _move_time(q_now, q)
            m = np.linalg.norm((p - aim) / sigma, axis=1)
            t_res = np.where(m >= self.resolve_sigmas, t_r, t_k - (t_k - t_r) * m / self.resolve_sigmas)
            t_res = np.maximum(t_res, t_go)
            total = 0.0
            for i in np.flatnonzero(solved):
                if not waiting and np.linalg.norm(p[i] - aim) < self.cup_tolerance:
                    ok = t_r + T_to <= deadline
                else:
                    f = 1.0 if T_to <= 1e-9 else min(1.0, (t_res[i] - t_r) / T_to)
                    ok = t_res[i] + _move_time(q_now + f * (q - q_now), poses[i]) <= deadline
                total += w[i] * ok
            return total

        scored = [(score(q, waiting), -i, name, q, waiting) for i, (name, q, waiting) in enumerate(candidates)]
        _, _, name, target, waiting = max(scored, key=lambda c: c[:2])
        return target, (q_mean if waiting else target), name

    @staticmethod
    def _segment(kin, q_now, v_now, target, time_left):
        """Cubic from the current reference state to target, or None if its path hits the room."""
        segment = HermiteTrajectory(q_now.copy(), v_now.copy(), target, max(time_left, min_move_time(q_now, target)))
        return None if kin.path_collides(segment, segment.T) else segment

    def __call__(self, kin: Kinematics, throw: Throw, cone_deg: float):
        predictor = self.predictor_for(throw)
        self.last = ReplanLog()
        t_start = self.t_plan
        belief = predictor(t_start)
        k, to_mean = plan_ik_catch(kin, throw, cone_deg, belief=belief, t_start=t_start)
        # With a blocking pillar, an early belief can put every likely outcome out of reach
        # (e.g. most predicted balls hit the pillar): wait in the ready pose and try again.
        while kin.pillar is not None and np.array_equal(to_mean.q1, to_mean.q0) and t_start < throw.times[k] - 0.3:
            t_start += self.replan_dt
            belief = predictor(t_start)
            k, to_mean = plan_ik_catch(kin, throw, cone_deg, belief=belief, t_start=t_start)
        if np.array_equal(to_mean.q1, to_mean.q0):  # nothing reachable: hold
            self.last = None
            return k, SegmentedReference([(self.t_plan, to_mean)])
        rng = np.random.default_rng(self.seed)
        t_c = float(throw.times[k])
        aim = belief.plan_mean[k]

        q_guess = to_mean.q1
        segments: list[tuple[float, HermiteTrajectory]] = []
        q_now, v_now = kin.q_ready.copy(), np.zeros_like(kin.q_ready)
        t_r, first = t_start, True
        while first or t_r <= t_c - self.final_lead + 1e-9:
            if segments:  # the reference's state at t_r, where the new segment starts
                t0, traj = segments[-1]
                q_now, v_now, _ = traj(t_r - t0)
            belief = predictor(t_r)
            if not first and (not in_catch_zone(belief.plan_mean[k])
                              or np.linalg.norm(belief.plan_mean[k] - aim) > self.reselect_dist):
                k_new, to_new = plan_ik_catch(kin, throw, cone_deg, belief=belief, q_start=q_now, t_start=t_r)
                if not np.array_equal(to_new.q1, to_new.q0):
                    k, t_c, q_guess = k_new, float(throw.times[k_new]), to_new.q1
                    self.last.reselections += 1
            aim = belief.plan_mean[k]
            target, q_guess, choice = self._target(kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c)
            self.last.choices.append(choice)
            self.last.n_replans += 1
            t_seg, t_r, first = t_r, t_r + self.replan_dt, False
            segment = self._segment(kin, q_now, v_now, target, t_c - self.arrive_lead - t_seg)
            if segment is None:
                segment = self._segment(kin, q_now, v_now, q_guess, t_c - self.arrive_lead - t_seg)
            if segment is None and kin.pillar is not None:  # blocked: head for the route's next waypoint
                waypoints = kin.route(q_now, target) or kin.route(q_now, q_guess)
                if waypoints is not None and len(waypoints) > 1:
                    segment = self._segment(kin, q_now, v_now, waypoints[0], 0.0)
            if segment is None:
                if segments:  # keep following the previous, collision-free segment
                    continue
                return k, SegmentedReference([(self.t_plan, CubicTrajectory(kin.q_ready, kin.q_ready))])
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
