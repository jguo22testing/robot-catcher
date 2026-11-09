"""Camera line of sight to the ball.

One camera behind and above the robot looks toward the thrower. The ball is visible
when the segment from the camera to the ball center crosses no free-standing panel
(boards, pillar faces). The enclosing walls, floor and ceiling can't come between
the camera and a ball inside the room. The arm itself is not treated as an occluder.
"""

from __future__ import annotations

import numpy as np

from .room import Panel, Room

CAMERA_POS = np.array([-0.3, 0.0, 1.4])
_ENCLOSURE = ("floor", "ceiling", "wall_")


def occluders(room: Room) -> list[Panel]:
    return [p for p in room.panels if not p.name.startswith(_ENCLOSURE)]


def visible(room: Room, points: np.ndarray, camera: np.ndarray = CAMERA_POS) -> np.ndarray:
    """(N,) bool: whether each point in (N, 3) is in the camera's line of sight."""
    points = np.atleast_2d(points)
    seen = np.ones(len(points), dtype=bool)
    d = points - camera  # segment camera -> point, parameter s in (0, 1)
    for panel in occluders(room):
        denom = d @ panel.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (panel.center - camera) @ panel.normal / denom
        crossing = (np.abs(denom) > 1e-12) & (s > 1e-9) & (s < 1 - 1e-6)
        if not crossing.any():
            continue
        hit = camera + s[crossing, None] * d[crossing]
        rel = hit - panel.center
        inside = (np.abs(rel @ panel.u) <= panel.half_extents[0]) & (np.abs(rel @ panel.v) <= panel.half_extents[1])
        idx = np.flatnonzero(crossing)[inside]
        seen[idx] = False
    return seen


def last_seen_time(times: np.ndarray, seen: np.ndarray, t_now: float) -> float:
    """Latest sample time <= t_now at which the ball was visible (0 if never)."""
    idx = np.flatnonzero(seen & (times <= t_now + 1e-9))
    return float(times[idx[-1]]) if len(idx) else 0.0
