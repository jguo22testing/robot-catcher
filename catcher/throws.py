"""Throw generation, noisy observations, and dataset building."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball_sim import GRAVITY, BallTrajectory, simulate
from .room import Room, make_room

# Catch zone: a spherical shell around the iiwa shoulder that the cup can reach
# with some margin, cut off near the floor.
SHOULDER = np.array([0.0, 0.0, 0.36])
REACH_MIN, REACH_MAX = 0.35, 0.70
CATCH_Z_MIN = 0.25

# Region the ball is thrown from (far side of the room from the robot).
THROW_LO = np.array([2.5, -1.5, 0.8])
THROW_HI = np.array([3.8, 1.5, 1.8])
MAX_THROW_SPEED = 12.0

OBS_RATE_HZ = 100.0


def in_catch_zone(p: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(p - SHOULDER, axis=-1)
    return (r >= REACH_MIN) & (r <= REACH_MAX) & (p[..., 2] >= CATCH_Z_MIN) & (p[..., 0] >= 0.0)


def first_catchable_index(pos: np.ndarray) -> int | None:
    idx = np.flatnonzero(in_catch_zone(pos))
    return int(idx[0]) if len(idx) else None


@dataclass
class Throw:
    room: Room
    p0: np.ndarray
    v0: np.ndarray
    traj: BallTrajectory
    times: np.ndarray
    pos: np.ndarray  # ground truth (T, 3)
    vel: np.ndarray  # ground truth (T, 3)
    catch_index: int  # first sample in the catch zone
    bounces_before_catch: int


def _velocity_to_hit(p0: np.ndarray, target: np.ndarray, flight_time: float) -> np.ndarray:
    return (target - p0 - 0.5 * GRAVITY * flight_time**2) / flight_time


def _random_catch_point(rng: np.random.Generator) -> np.ndarray:
    while True:
        p = SHOULDER + rng.uniform(-REACH_MAX, REACH_MAX, size=3)
        if in_catch_zone(p):
            return p


def sample_throw(room: Room, rng: np.random.Generator, min_bounces: int = 0, max_bounces: int = 0,
                 t_end: float = 3.0, max_tries: int = 5000,
                 throw_lo: np.ndarray = THROW_LO, throw_hi: np.ndarray = THROW_HI) -> Throw:
    """Rejection-sample a throw whose ball enters the catch zone after
    between min_bounces and max_bounces bounces.

    With min_bounces == 0 we aim directly at the catch zone; otherwise we aim at
    a random point on a random non-floor panel and let physics decide.
    """
    times = np.arange(0.0, t_end, 1.0 / OBS_RATE_HZ)
    aim_panels = [p for p in room.panels if p.name not in ("floor", "ceiling", "wall_x+")]
    for _ in range(max_tries):
        p0 = rng.uniform(throw_lo, throw_hi)
        if min_bounces == 0:
            target = _random_catch_point(rng)
            # Lobs: the iiwa needs ~0.5-1 s to reposition, so faster direct throws are uncatchable.
            flight_time = rng.uniform(0.8, 1.4)
        else:
            panel = aim_panels[rng.integers(len(aim_panels))]
            ab = rng.uniform(-0.9, 0.9, size=2)
            target = (panel.center + ab[0] * panel.half_extents[0] * panel.u
                      + ab[1] * panel.half_extents[1] * panel.v)
            flight_time = rng.uniform(0.3, 0.9)
        v0 = _velocity_to_hit(p0, target, flight_time)
        if np.linalg.norm(v0) > MAX_THROW_SPEED:
            continue
        traj = simulate(room, p0, v0, t_end=t_end)
        pos, vel = traj.sample(times)
        ci = first_catchable_index(pos)
        if ci is None:
            continue
        nb = traj.num_bounces_before(times[ci])
        if min_bounces <= nb <= max_bounces:
            return Throw(room, p0, v0, traj, times, pos, vel, ci, nb)
    raise RuntimeError("could not sample a valid throw; loosen the constraints")


STAGE_BOUNCES = {1: (0, 0), 2: (1, 1), 3: (1, 3), 4: (1, 4)}


def sample_stage_throw(stage: int, rng: np.random.Generator, t_end: float = 3.0,
                       with_pillar: bool = False, room_ok=None) -> Throw:
    """A throw for a stage. room_ok (Room -> bool) rejects rooms, e.g. ones whose boards pass
    through the robot (mujoco_scene.ready_pose_clear); datasets don't use it."""
    lo, hi = STAGE_BOUNCES[stage]
    for _ in range(50):  # resample the room if it makes valid throws too rare
        room = make_room(stage, rng, with_pillar)
        if room_ok is not None and not room_ok(room):
            continue
        try:
            return sample_throw(room, rng, lo, hi, t_end=t_end, max_tries=2000)
        except RuntimeError:
            continue
    raise RuntimeError(f"stage {stage}: could not find a valid room/throw")


def observe(pos: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Noisy position observations (Gaussian, iid per axis)."""
    return pos + rng.normal(0.0, noise_std, size=pos.shape)


def generate_dataset(stage: int, n: int, seed: int = 0, t_end: float = 3.0, noise_std: float = 0.005,
                     max_panels: int = 10, n_cloud_points: int = 1024) -> dict[str, np.ndarray]:
    """Fixed-shape arrays ready for np.savez / a PyTorch Dataset.

    times (T,), pos/vel/obs (N, T, 3), panels (N, P, 13) + panel_mask (N, P),
    cloud (N, n_cloud_points, 3), catch_index (N,), bounces_before_catch (N,),
    bounce_times (N, B) padded with inf, p0/v0 (N, 3).
    """
    rng = np.random.default_rng(seed)
    throws = [sample_stage_throw(stage, rng, t_end) for _ in range(n)]
    return {"stage": np.array(stage),
            **throws_to_arrays(throws, rng, noise_std, max_panels, n_cloud_points)}


def throws_to_arrays(throws: list[Throw], rng: np.random.Generator, noise_std: float = 0.005,
                     max_panels: int = 10, n_cloud_points: int = 1024) -> dict[str, np.ndarray]:
    """Stack already-sampled throws into the arrays described in generate_dataset."""
    n = len(throws)
    max_b = max(1, max(len(t.traj.bounces) for t in throws))
    bounce_times = np.full((n, max_b), np.inf)
    panels = np.zeros((n, max_panels, 13))
    mask = np.zeros((n, max_panels), dtype=bool)
    for i, th in enumerate(throws):
        bt = [b.t for b in th.traj.bounces]
        bounce_times[i, :len(bt)] = bt
        panels[i], mask[i] = th.room.panel_array(max_panels)
    pos = np.stack([t.pos for t in throws])
    return {
        "times": throws[0].times,
        "pos": pos,
        "vel": np.stack([t.vel for t in throws]),
        "obs": observe(pos, noise_std, rng),
        "noise_std": np.array(noise_std),
        "p0": np.stack([t.p0 for t in throws]),
        "v0": np.stack([t.v0 for t in throws]),
        "panels": panels,
        "panel_mask": mask,
        "cloud": np.stack([t.room.sample_point_cloud(n_cloud_points, rng) for t in throws]),
        "catch_index": np.array([t.catch_index for t in throws]),
        "bounces_before_catch": np.array([t.bounces_before_catch for t in throws]),
        "bounce_times": bounce_times,
    }
