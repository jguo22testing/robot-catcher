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

    With a blocking pillar (Kinematics.pillar) the choice is left, right or wait (the
    tucked ready pose), see _side_target: the pose between the sides is inside the
    pillar, and switching sides means backing out around it.
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
            if not ok and kin.pillar is not None:  # the mean may have moved to the other side
                q_mean, ok = solve_catch_ik(kin, belief.plan_mean[k], belief.plan_vel[k], kin.q_ready, cone_deg)
            q_mean = q_mean if ok else q_guess
            return q_mean, q_mean[None], "mean"
        if kin.pillar is not None:
            return self._side_target(kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c)
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
    def _route_time(kin, a, b):
        """Time estimate for moving a -> b, without collision checks: the straight joint-space
        move, which is what most moves are (Kinematics.route adds waypoints only when it is
        blocked). Crossing to the other side of a blocking pillar is slow anyway: the catch
        poses on the two sides differ by ~2 rad in each of joints 1, 3 and 5."""
        return min_move_time(a, b)

    def _side_target(self, kin, belief, k, q_guess, cone_deg, rng, q_now, t_r, t_c):
        """Left, right or wait, for a blocking pillar.

        Sampled ball positions at index k (every mode's, weighted by the mode) are split
        by the side of the pillar they are on; a side's probability is its share of the
        weight, and samples whose catch IK fails count as misses. Candidates: wait at the
        ready pose, or each side's catch pose for the weighted mean of its samples (a
        minimax pose over the samples would jump between re-plans as they are redrawn,
        and the arm would chase it). The side is forecast to become known when the belief's
        spread, shrinking linearly to zero at the catch, no longer reaches across the
        pillar's center line. Each candidate is scored by the probability of reaching the
        real catch configuration in time: committing reaches its own side directly and the
        other side only from where the arm is once the side is known; waiting leaves at the
        next re-plan at the earliest. Ties go to the likelier side.
        """
        center, _ = kin.pillar
        modes = belief.modes(k, self.n_trajectories, rng)
        points = np.vstack([m.points for m in modes if len(m.points)])
        weights = np.concatenate([np.full(len(m.points), m.weight / len(m.points)) for m in modes if len(m.points)])
        vels = np.vstack([np.tile(m.vel, (len(m.points), 1)) for m in modes if len(m.points)])
        # Only balls inside the catch zone can be caught; the rest (e.g. ones that bounced back
        # off the pillar) keep their weight as certain misses but don't pull the side targets.
        catchable = in_catch_zone(points)
        if not catchable.any():
            return q_guess, q_guess[None], "none"
        points, weights, vels = points[catchable], weights[catchable], vels[catchable]
        sides = np.where(points[:, 1] >= center[1], 1, -1)
        groups, centers = {}, {}
        for s, name in ((1, "left"), (-1, "right")):
            on_side = sides == s
            if not on_side.any():
                groups[name] = (np.zeros((0, len(q_guess))), np.zeros(0), 0.0)
                continue
            p_c = np.average(points[on_side], axis=0, weights=weights[on_side])
            v_c = np.average(vels[on_side], axis=0, weights=weights[on_side])
            q_c, ok = solve_catch_ik(kin, p_c, v_c, q_guess if kin.side(q_guess) == s else kin.q_ready, cone_deg)
            if not ok:
                q_c, ok = solve_catch_ik(kin, p_c, v_c, kin.q_ready, cone_deg)
            seed = q_c if ok else kin.q_ready
            if ok:
                centers[name] = q_c
            sols, ws = [], []
            for p, v, w in zip(points[on_side], vels[on_side], weights[on_side]):
                q, ok_p = solve_catch_ik(kin, p, v, seed, cone_deg)
                if ok_p:
                    sols.append(q); ws.append(w)
            groups[name] = (np.array(sols).reshape(-1, len(q_guess)), np.array(ws), weights[on_side].sum())
        if not any(len(g[0]) for g in groups.values()):
            return q_guess, q_guess[None], "none"

        # When will the side be known?
        mean_y = np.average(points[:, 1], weights=weights)
        spread = np.sqrt(np.average((points[:, 1] - mean_y) ** 2, weights=weights))
        d = abs(mean_y - center[1])
        t_k = float(belief.times[k])
        t_known = t_r if spread < 1e-6 else t_k - (t_k - t_r) * min(1.0, d / (1.64 * spread))
        deadline = t_c - self.arrive_lead

        T = lambda a, b: self._route_time(kin, a, b)

        def arm_at(q_target, t):
            """Where the arm is at time t heading from q_now to q_target (joint-space line)."""
            f = 1.0 if t <= t_r else min(1.0, (t - t_r) / max(T(q_now, q_target), 1e-6))
            return q_now + f * (q_target - q_now)

        def p_commit(side_name, q_side):
            """Head for one side now. Its catch poses are reached directly (corrections within a
            side are cheap); the other side's only from where the arm is once the side is known."""
            score = 0.0
            for name, (sols, ws, _) in groups.items():
                if not len(sols):
                    continue
                if name == side_name:
                    ok = [t_r + T(q_now, qi) <= deadline for qi in sols]
                else:
                    q_then = arm_at(q_side, t_known)
                    ok = [max(t_known, t_r) + T(q_then, qi) <= deadline for qi in sols]
                score += float(np.sum(ws * np.array(ok)))
            return score

        def p_wait():
            """Stay put: leave no earlier than the next re-plan (a waiting planner always loses
            at least that cycle, or it would wait forever), or when the side is known."""
            t_go = max(t_known, t_r + self.replan_dt)
            q_then = arm_at(kin.q_ready, t_go)
            return sum(float(np.sum(ws * np.array([t_go + T(q_then, qi) <= deadline for qi in sols])))
                       for sols, ws, _ in groups.values() if len(sols))

        candidates = [(name, centers[name]) for name, g in sorted(groups.items(), key=lambda g: -g[1][2])
                      if name in centers]  # likelier side first
        scored = [(p_commit(name, q), -i, name, q) for i, (name, q) in enumerate(candidates)]
        scored.append((p_wait(), -len(candidates), "wait", kin.q_ready))
        _, _, name, target = max(scored, key=lambda c: c[:2])
        likely = candidates[0][1] if candidates else kin.q_ready  # no side reachable: wait
        chosen = target if name != "wait" else likely
        return target, np.vstack([chosen, likely]), name

    @staticmethod
    def _segment(kin, q_now, v_now, target, time_left):
        """Cubic from the current reference state to target, or None if its path hits the room."""
        segment = HermiteTrajectory(q_now.copy(), v_now.copy(), target, max(time_left, min_move_time(q_now, target)))
        return None if kin.path_collides(segment, segment.T) else segment

    def __call__(self, kin: Kinematics, throw: Throw, cone_deg: float):
        predictor = self.predictor_for(throw)
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
            return k, SegmentedReference([(self.t_plan, to_mean)])
        rng = np.random.default_rng(self.seed)
        t_c = float(throw.times[k])
        aim = belief.plan_mean[k]

        q_guess = to_mean.q1
        segments: list[tuple[float, HermiteTrajectory]] = []
        q_now, v_now = kin.q_ready.copy(), np.zeros_like(kin.q_ready)
        t_r, i = t_start, 0
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
            if segment is None and kin.pillar is not None:  # blocked: head for the route's next waypoint
                waypoints = kin.route(q_now, target) or kin.route(q_now, solutions[0])
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
