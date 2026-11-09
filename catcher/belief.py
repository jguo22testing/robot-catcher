"""Gaussian belief over the ball's future positions, and a synthetic stand-in for the trajectory model.

The learned predictor (catcher.trajectory) outputs a mean and per-axis variance per
future step. `GaussianBelief` holds exactly that and can draw whole trajectories from
it. `SyntheticPredictor` produces such beliefs for a ground-truth throw, with a
variance that grows with the prediction horizon and with each bounce, so the planners
can be built and compared before the model is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .throws import Throw
from .visibility import last_seen_time

CHI2_3D_90 = 6.25  # 0.9 quantile of the chi-squared distribution with 3 degrees of freedom


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
