"""Beliefs over the ball's future positions, and synthetic stand-ins for the trajectory model.

The learned predictor (catcher.trajectory) outputs a mean and per-axis variance per
future step. `GaussianBelief` holds exactly that and can draw whole trajectories from
it. `SyntheticPredictor` produces such beliefs for a ground-truth throw, with a
variance that grows with the prediction horizon and with each bounce, so the planners
can be built and compared before the model is wired in.

A single Gaussian can't say "the ball either clips the pillar edge or misses it".
`ParticlePredictor` can: it simulates a cloud of possible balls from a noisy state
estimate, and `ParticleBelief.modes` groups them by which surfaces they bounce off.

Planners read a belief through `plan_mean`, `plan_vel` (T, 3), the path they choose the
catch time on, and `modes(k, n, rng)`, a list of `Mode`s at time index k. A GaussianBelief
has exactly one mode and plans on its mean. A ParticleBelief plans on its most likely
path: the overall mean of a split belief can lie between the outcomes, far from either.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from .ball_sim import simulate
from .room import Room, blocking_pillar_footprint
from .throws import Throw, in_catch_zone
from .visibility import last_seen_time

CHI2_3D_90 = 6.25  # 0.9 quantile of the chi-squared distribution with 3 degrees of freedom


@dataclass
class Mode:
    """One way the throw can go, at a given time index k."""

    weight: float  # probability of this mode
    mean: np.ndarray  # (3,) mean ball position at k
    vel: np.ndarray  # (3,) mean ball velocity at k
    points: np.ndarray  # (m, 3) sampled positions at k inside the mode's 90% ellipsoid
    split_time: float = np.inf  # when this mode's path parts from the others' (inf: only mode)


@dataclass
class GaussianBelief:
    """Independent per-axis Gaussian over ball position at each time, as seen at `t_now`.

    Times up to t_now are already observed, so their std is 0 and their mean is the
    ball position. `vel` is the mean ball velocity, used only for the cup-axis cone.
    """

    times: np.ndarray  # (T,)
    mean: np.ndarray  # (T, 3)
    std: np.ndarray  # (T, 3)
    vel: np.ndarray  # (T, 3)
    t_now: float
    corr_time: float = 0.3  # seconds over which sampled errors stay correlated

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n trajectories (n, T, 3).

        Each time's marginal is exactly N(mean, std^2). The standardized error is a
        unit-variance AR(1) process in time (the discrete Ornstein-Uhlenbeck process),
        so sampled trajectories are smooth rather than jittering from step to step.
        """
        T = len(self.times)
        rho = np.exp(-np.diff(self.times, prepend=self.times[0]) / self.corr_time)
        w = np.empty((n, T, 3))
        w[:, 0] = rng.normal(size=(n, 3))
        for k in range(1, T):
            w[:, k] = rho[k] * w[:, k - 1] + np.sqrt(1.0 - rho[k] ** 2) * rng.normal(size=(n, 3))
        return self.mean + self.std * w

    def mahalanobis_sq(self, k: int, points: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of points (..., 3) from the belief at time index k."""
        return np.sum(((points - self.mean[k]) / np.maximum(self.std[k], 1e-9)) ** 2, axis=-1)

    def ellipsoid_samples(self, k: int, n: int, rng: np.random.Generator,
                          chi2: float = CHI2_3D_90) -> np.ndarray:
        """Positions at time index k of n sampled trajectories, keeping those inside the
        confidence ellipsoid (chi2 = 6.25 is the 90% one). Returns (m, 3), m <= n."""
        points = self.sample(n, rng)[:, k]
        return points[self.mahalanobis_sq(k, points) <= chi2]

    def modes(self, k: int, n: int, rng: np.random.Generator) -> list[Mode]:
        return [Mode(1.0, self.mean[k], self.vel[k], self.ellipsoid_samples(k, n, rng))]

    @property
    def plan_mean(self) -> np.ndarray:
        return self.mean

    @property
    def plan_vel(self) -> np.ndarray:
        return self.vel


class SyntheticPredictor:
    """Stands in for the trajectory model on one throw.

    The predicted mean is the true position plus a fixed random error, and the stated
    std is the error's true std, so the belief is calibrated: the truth lies in the
    predicted 90% ellipsoid 90% of the time. The std at time t seen from t_now is

        axis_scale * (sigma0 + growth * (t - t_now) + bounce_sigma * bounces in (t_now, t])

    The error realization is drawn once, so asking again at a later t_now gives the
    same mean errors with a tighter std: the belief sharpens as the ball is observed.

    `seen` (T,) marks the samples where the camera sees the ball (catcher.visibility).
    While the ball is hidden the belief stays the one from its last sighting.
    """

    def __init__(self, throw: Throw, sigma0: float = 0.005, growth: float = 0.06, bounce_sigma: float = 0.03,
                 corr_time: float = 0.3, vel_sigma_ratio: float = 2.0, seed: int = 0,
                 seen: np.ndarray | None = None):
        self.throw = throw
        self.sigma0, self.growth, self.bounce_sigma = sigma0, growth, bounce_sigma
        self.corr_time, self.vel_sigma_ratio = corr_time, vel_sigma_ratio
        rng = np.random.default_rng(seed)
        self.axis_scale = rng.uniform(0.6, 1.4, size=3)
        # One realized standardized error trajectory, shared by every call.
        err = GaussianBelief(throw.times, np.zeros_like(throw.pos), np.ones_like(throw.pos), throw.vel, 0.0,
                             corr_time)
        self.error = err.sample(1, rng)[0]
        self.vel_error = rng.normal(size=throw.vel.shape)
        self.bounce_times = np.array([b.t for b in throw.traj.bounces])
        self.seen = np.ones(len(throw.times), dtype=bool) if seen is None else seen

    def __call__(self, t_now: float) -> GaussianBelief:
        t_now = last_seen_time(self.throw.times, self.seen, t_now)
        t = self.throw.times
        horizon = np.maximum(t - t_now, 0.0)
        bounces = np.array([np.sum((self.bounce_times > t_now) & (self.bounce_times <= ti)) for ti in t])
        scale = self.sigma0 + self.growth * horizon + self.bounce_sigma * bounces
        std = np.where((t > t_now)[:, None], scale[:, None] * self.axis_scale, 0.0)
        return GaussianBelief(t, self.throw.pos + std * self.error, std,
                              self.throw.vel + self.vel_sigma_ratio * std * self.vel_error, t_now, self.corr_time)


@dataclass
class ParticleBelief:
    """A cloud of possible ball trajectories, as seen at `t_now`.

    Times up to t_now are observed (every particle is at the true position). After
    that, each particle carries its own bounce history, which `modes` groups by. With a
    blocking pillar (`pillar_y`, its center line) particles are also grouped by the side
    of it they are on: passing left and passing right bounce off the same panels (none),
    and their average would be inside the pillar.
    """

    times: np.ndarray  # (T,)
    particles: np.ndarray  # (M, T, 3)
    particle_vel: np.ndarray  # (M, T, 3)
    bounces: list[list[tuple[float, int]]]  # per particle: (time, panel index) of each future bounce
    t_now: float
    min_weight: float = 0.05  # modes holding fewer particles than this fraction are dropped
    pillar_y: float | None = None  # center line of a blocking pillar, if any

    def _side(self, i: int, k: int) -> int:
        if self.pillar_y is None:
            return 0
        return 1 if self.particles[i, k, 1] >= self.pillar_y else -1

    def __post_init__(self):
        self.mean = self.particles.mean(0)
        self.vel = self.particle_vel.mean(0)
        self.std = self.particles.std(0)
        # Most likely path: particles that reach the catch zone, grouped by the panels they
        # bounce off before entering it (and the side of a blocking pillar they enter on);
        # plan on the largest group's mean. Balls that never reach the zone can't be caught.
        zone = in_catch_zone(self.particles) & (self.times > self.t_now)
        groups, missing = defaultdict(list), []
        for i in range(len(self.particles)):
            entry = np.flatnonzero(zone[i])
            if not len(entry):
                missing.append(i)
                continue
            t_entry = self.times[entry[0]]
            panels = tuple(panel for t, panel in self.bounces[i] if t <= t_entry)
            groups[(panels, self._side(i, entry[0]))].append(i)
        likely = max(groups.values(), key=len) if groups else missing
        self.plan_mean = self.particles[likely].mean(0)
        self.plan_vel = self.particle_vel[likely].mean(0)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.particles[rng.integers(len(self.particles), size=n)]

    def mahalanobis_sq(self, k: int, points: np.ndarray) -> np.ndarray:
        return np.sum(((points - self.mean[k]) / np.maximum(self.std[k], 1e-9)) ** 2, axis=-1)

    def ellipsoid_samples(self, k: int, n: int, rng: np.random.Generator, chi2: float = CHI2_3D_90) -> np.ndarray:
        points = self.sample(n, rng)[:, k]
        return points[self.mahalanobis_sq(k, points) <= chi2]

    def signature(self, i: int, k: int) -> tuple[tuple[int, ...], int]:
        """Panels particle i bounces off between t_now and times[k], in order, and the side
        of a blocking pillar it is on at k (0 without one)."""
        return tuple(panel for t, panel in self.bounces[i] if t <= self.times[k]), self._side(i, k)

    def modes(self, k: int, n: int | None = None, rng: np.random.Generator | None = None) -> list[Mode]:
        """Particles grouped by the panels they bounce off before times[k], most likely first.
        Each mode keeps up to n of its particles inside its own 90% ellipsoid."""
        groups = defaultdict(list)
        for i in range(len(self.particles)):
            groups[self.signature(i, k)].append(i)
        kept = {sig: np.array(idx) for sig, idx in groups.items() if len(idx) >= self.min_weight * len(self.particles)}
        total = sum(len(idx) for idx in kept.values())
        modes = []
        for sig, idx in sorted(kept.items(), key=lambda kv: -len(kv[1])):
            pts = self.particles[idx, k]
            mean, std = pts.mean(0), np.maximum(pts.std(0), 1e-4)
            inside = pts[np.sum(((pts - mean) / std) ** 2, axis=-1) <= CHI2_3D_90]
            if n is not None and len(inside) > n:
                inside = inside[rng.choice(len(inside), n, replace=False)]
            split = min((self._split_time(sig, idx, other, kept[other]) for other in kept if other != sig),
                        default=np.inf)
            modes.append(Mode(len(idx) / total, mean, self.particle_vel[idx, k].mean(0), inside, split))
        return modes

    def _split_time(self, sig_a, idx_a, sig_b, idx_b) -> float:
        """When two modes part: the (median) time of the first bounce where their panel
        sequences differ, or of the extra bounce one of them has. Modes that differ only in
        the side of a blocking pillar have no such bounce (inf); the planner handles those
        with its own side forecast."""
        (sig_a, _), (sig_b, _) = sig_a, sig_b
        c = 0
        while c < min(len(sig_a), len(sig_b)) and sig_a[c] == sig_b[c]:
            c += 1
        times = [np.median([self.bounces[i][c][0] for i in idx]) for sig, idx in ((sig_a, idx_a), (sig_b, idx_b))
                 if len(sig) > c]
        return float(min(times)) if times else np.inf


class ParticlePredictor:
    """Physics-based stand-in for the trajectory model that can represent several outcomes.

    At each t_now it estimates the ball's state at its last sighting with a fixed error
    (sigma_p, sigma_v), spreads n particles around that estimate with the same spread,
    perturbs each particle's bounce materials by material_sigma, and simulates them
    through the room. The estimation error is drawn the same way as the particle spread,
    so the true ball is statistically one more particle: the belief is calibrated. The
    errors are drawn once per throw (one per flight segment), so beliefs at different
    t_now agree with each other the way a tracker's would.
    """

    def __init__(self, throw: Throw, n: int = 96, sigma_p: float = 0.005, sigma_v: float = 0.06,
                 material_sigma: float = 0.05, seed: int = 0, seen: np.ndarray | None = None):
        self.throw, self.n, self.sigma_p, self.sigma_v = throw, n, sigma_p, sigma_v
        rng = np.random.default_rng(seed)
        n_segments = len(throw.traj.bounces) + 1
        self.estimate_error = rng.normal(size=(n_segments, 6))
        self.spread = rng.normal(size=(n, 6))
        self.rooms = [self._perturbed(throw.room, rng, material_sigma) for _ in range(n)]
        self.bounce_times = np.array([b.t for b in throw.traj.bounces])
        self.seen = np.ones(len(throw.times), dtype=bool) if seen is None else seen
        footprint = blocking_pillar_footprint(throw.room)
        self.pillar_y = None if footprint is None else float(footprint[0][1])
        self._cache: dict[float, ParticleBelief] = {}

    @staticmethod
    def _perturbed(room: Room, rng: np.random.Generator, sigma: float) -> Room:
        panels = []
        for p in room.panels:
            m = p.material
            e = float(np.clip(m.restitution + sigma * rng.normal(), 0.05, 0.98))
            f = float(np.clip(m.friction + sigma * rng.normal(), 0.0, 0.9))
            panels.append(replace(p, material=replace(m, restitution=e, friction=f)))
        return Room(panels, room.lo, room.hi)

    def __call__(self, t_now: float) -> ParticleBelief:
        t_obs = last_seen_time(self.throw.times, self.seen, t_now)
        if t_obs not in self._cache:
            self._cache[t_obs] = self._belief(t_obs)
        return self._cache[t_obs]

    def _belief(self, t_obs: float) -> ParticleBelief:
        times = self.throw.times
        segment = int(np.sum(self.bounce_times <= t_obs))
        (p,), (v,) = self.throw.traj.sample(np.array([t_obs]))
        scale = np.r_[np.full(3, self.sigma_p), np.full(3, self.sigma_v)]
        estimate = np.r_[p, v] + scale * self.estimate_error[segment]
        starts = estimate + scale * self.spread
        future = times > t_obs
        pos = np.tile(self.throw.pos, (self.n, 1, 1))
        vel = np.tile(self.throw.vel, (self.n, 1, 1))
        bounces = []
        for i, (room, x) in enumerate(zip(self.rooms, starts)):
            traj = simulate(room, x[:3], x[3:], t_end=float(times[-1] - t_obs) + 1e-6)
            pos[i, future], vel[i, future] = traj.sample(times[future] - t_obs)
            bounces.append([(b.t + t_obs, b.panel_index) for b in traj.bounces])
        return ParticleBelief(times, pos, vel, bounces, t_obs, pillar_y=self.pillar_y)
