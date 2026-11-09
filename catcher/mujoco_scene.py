"""MuJoCo scene: KUKA iiwa14 with a cup end-effector inside a room of panels.

MuJoCo simulates the arm: a joint reference (IK + cubic, or differential IK)
tracked by a computed-torque controller. The ball is a free MuJoCo body. Its flight
through the room comes from the NumPy simulator (catcher.ball_sim), which is the
ground truth the predictors work from: run_catch replays it until the ball comes
near the arm, then hands it to MuJoCo, so contacts with the cup, the arm and the
room near the robot are simulated. A catch counts if the ball is still in the cup
at the end of the run.

Contacts: arm-room, ball-room and ball-arm (including the cup). The arm never
collides with itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .room import Room
from .throws import Throw, in_catch_zone

IIWA_XML = Path(__file__).resolve().parent.parent / "models" / "kuka_iiwa_14" / "iiwa14.xml"
CUP_RADIUS = 0.07
CUP_DEPTH = 0.08
FLANGE_Z = 0.045  # link 7 flange (Menagerie "attachment_site")
BALL_RADIUS = 0.035  # MuJoCo ball radius; the NumPy simulator treats the ball as a point
BALL_MASS = 0.057  # tennis ball
RELEASE_DIST = 0.35  # hand the ball to MuJoCo once it is this close to any arm link origin
MIN_DESCENT = 0.0  # catch only where the ball descends: v_z / |v| below -MIN_DESCENT
PANEL_THICKNESS = 0.02

# KUKA LBR iiwa 14 R820 spec-sheet limits (the Menagerie model doesn't include them).
VELOCITY_LIMITS = np.radians([85, 85, 100, 75, 130, 135, 135])
TORQUE_LIMITS = np.array([320, 320, 176, 176, 110, 40, 40], dtype=float)

# "Ready" posture: cup centered in the catch zone, opening tilted up toward the
# thrower (+x). Incoming balls mostly descend, so this minimizes reorientation.
Q_READY = np.array([0.0, 0.3, 0.0, -1.7, 0.0, -1.2, 0.0])

_NOCONTACT = dict(contype=0, conaffinity=0)
# Collision bits: geoms a and b touch if (a.contype & b.conaffinity) or (b.contype & a.conaffinity).
_ROOM, _ARM, _BALL = 1, 2, 4
_ARM_CONTACT = dict(contype=_ARM, conaffinity=0)  # arm touches room and ball, never itself
_ROOM_CONTACT = dict(contype=_ROOM, conaffinity=_ARM | _BALL)
_BALL_CONTACT = dict(contype=_BALL, conaffinity=_ROOM | _ARM)
N_ARM = 7  # arm joints come first in qpos/qvel; the ball's free joint follows
# Stiff, overdamped contact for the ball and cup, i.e. a padded cup. MuJoCo's default
# (timeconst 0.02 s) is so soft that a 7 m/s ball sinks through the cup bottom; 0.002 s
# is the stiffest it allows at the 1 ms time step. With damping ratio 1 the ball still
# rebounds about 1 m/s off the (compliant) arm and climbs out; 2 stops that.
_STIFF = dict(solref=[0.002, 2.0])
CUP_WALL = 0.01  # cup bottom and wall thickness


def _add_cup(body: mujoco.MjsBody, n_walls: int = 12):
    """Open cup made of a bottom disc and a ring of box walls, both CUP_WALL thick, with
    inner radius CUP_RADIUS and inner depth CUP_DEPTH. Opening along link 7's +z."""
    green = [0.1, 0.6, 0.3, 1.0]
    z0 = FLANGE_Z + 0.005  # inner bottom surface
    body.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[CUP_RADIUS + CUP_WALL, CUP_WALL / 2, 0], pos=[0, 0, z0 - CUP_WALL / 2], rgba=green,
                  mass=0.05, **_ARM_CONTACT, **_STIFF)
    r_wall = CUP_RADIUS + CUP_WALL / 2
    wall_w = 2 * (CUP_RADIUS + CUP_WALL) * np.tan(np.pi / n_walls) + 0.004
    for i in range(n_walls):
        a = 2 * np.pi * i / n_walls
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUP_WALL / 2, wall_w / 2, CUP_DEPTH / 2],
                      pos=[r_wall * np.cos(a), r_wall * np.sin(a), z0 + CUP_DEPTH / 2],
                      quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=[*green[:3], 0.9],
                      mass=0.01, **_ARM_CONTACT, **_STIFF)
    body.add_site(name="cup_center", pos=[0, 0, z0 + CUP_DEPTH / 2])


def _add_room(spec: mujoco.MjSpec, room: Room):
    for panel in room.panels:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, panel.rotation().flatten())
        spec.worldbody.add_geom(name=panel.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[*panel.half_extents, PANEL_THICKNESS / 2], pos=panel.center,
                                quat=quat, rgba=panel.material.rgba, **_ROOM_CONTACT)


def build_model(room: Room | None = None) -> mujoco.MjModel:
    """iiwa14 (torque-controlled, no actuators) + cup, optional room, and a free ball."""
    spec = mujoco.MjSpec.from_file(str(IIWA_XML))
    for a in list(spec.actuators):  # drop Menagerie's position servos; we apply torques directly
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = 1e-3
    for g in spec.geoms:  # Menagerie collision spheres: arm bits (visual meshes keep contype 0)
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = _ARM, 0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 720
    _add_cup(spec.body("link7"))
    if room is not None:
        _add_room(spec, room)
    for light in list(spec.lights):  # Menagerie's tracking spotlight; replaced by room lighting
        spec.delete(light)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.worldbody.add_light(pos=[1.5, 0, 6], dir=[-0.2, 0.1, -1], diffuse=[0.5, 0.5, 0.5],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True)
    ball = spec.worldbody.add_body(name="ball", pos=[0, 0, -5])
    ball.add_freejoint(name="ball")
    ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=BALL_MASS,
                  rgba=[1.0, 0.45, 0.0, 1.0], **_BALL_CONTACT, **_STIFF)
    return spec.compile()


ARM_BODIES = [f"link{i}" for i in range(1, 8)]


def ready_pose_clear(room: Room) -> bool:
    """Whether the arm in its ready pose is clear of the room. Random boards are placed
    without regard to the robot and sometimes pass through it."""
    return not Kinematics(build_model(room)).collides(Q_READY)


def set_ball(data: mujoco.MjData, pos: np.ndarray, vel: np.ndarray | None = None):
    """Place the free ball (no rotation) and optionally set its linear velocity."""
    data.qpos[N_ARM:N_ARM + 3] = pos
    data.qpos[N_ARM + 3:N_ARM + 7] = [1, 0, 0, 0]
    data.qvel[N_ARM:N_ARM + 6] = 0.0
    if vel is not None:
        data.qvel[N_ARM:N_ARM + 3] = vel


def ball_in_cup(kin: "Kinematics", q: np.ndarray, p_ball: np.ndarray) -> bool:
    """Whether the ball center lies inside the cup volume (rim included, up to half a ball above it)."""
    center, axis, _ = kin(q)
    d = p_ball - center
    axial = d.dot(axis)
    radial = np.linalg.norm(d - axial * axis)
    return radial < CUP_RADIUS and -CUP_DEPTH / 2 < axial < CUP_DEPTH / 2 + BALL_RADIUS / 2


class Kinematics:
    """Cup-center forward kinematics and Jacobians on a scratch MjData."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        set_ball(self.data, np.array([0.0, 0.0, -5.0]))  # parked out of the way
        self.site = model.site("cup_center").id
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.lo, self.hi = model.jnt_range[:N_ARM].T  # arm joint limits
        arm_ids = [model.body(name).id for name in ARM_BODIES]
        self.arm_geoms = np.isin(model.geom_bodyid, arm_ids) & (model.geom_contype > 0)
        self.ball_geom = model.geom("ball").id

    def __call__(self, q: np.ndarray):
        """Returns cup center position, cup axis (opening direction), and the 6x7 Jacobian."""
        self.data.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
        R = self.data.site_xmat[self.site].reshape(3, 3)
        J = np.vstack([self.jacp[:, :N_ARM], self.jacr[:, :N_ARM]])
        return self.data.site_xpos[self.site].copy(), R[:, 2].copy(), J

    def collides(self, q: np.ndarray) -> bool:
        """Whether the arm or cup at q penetrates the room (walls, boards, pillar)."""
        self.data.qpos[:N_ARM] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        for c in self.data.contact[:self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.ball_geom not in (g1, g2) and self.arm_geoms[g1] != self.arm_geoms[g2]:
                return True
        return False

    def path_collides(self, traj, duration: float, n: int = 12) -> bool:
        """Whether a joint trajectory (called with time from its start) collides at any of
        n evenly spaced times over [0, duration]."""
        return any(self.collides(traj(t)[0]) for t in np.linspace(0.0, duration, n))


def _cone_error(axis: np.ndarray, target_axis: np.ndarray, cone_rad: float):
    """Rotation vector (perpendicular to the cup axis) that brings axis inside the cone, and the angle."""
    angle = np.arccos(np.clip(axis.dot(target_axis), -1, 1))
    c = np.cross(axis, target_axis)
    if angle <= cone_rad or np.linalg.norm(c) < 1e-9:
        return np.zeros(3), angle
    return c / np.linalg.norm(c) * (angle - cone_rad), angle


def _task_step(J: np.ndarray, err: np.ndarray, axis: np.ndarray, q: np.ndarray, q_nominal: np.ndarray,
               damping: float = 0.05, centering: float = 0.1) -> np.ndarray:
    """Damped least squares for position + the 2 rotations that tilt the cup axis
    (spin about the axis is free), with null-space pull toward q_nominal."""
    B = np.linalg.svd(axis[None, :])[2][1:]  # 2x3 basis orthogonal to the axis
    Jt = np.vstack([J[:3], B @ J[3:]])
    et = np.concatenate([err[:3], B @ err[3:]])
    Jpinv = Jt.T @ np.linalg.inv(Jt @ Jt.T + damping**2 * np.eye(5))
    return Jpinv @ et + (np.eye(len(q)) - Jpinv @ Jt) @ (centering * (q_nominal - q))


def _solve_ik_from(kin: Kinematics, p_ball, target_axis, q_init, q_nominal, cone, iters, pos_tol):
    lo, hi = kin.lo, kin.hi
    q = q_init.copy()
    for _ in range(iters):
        x, axis, J = kin(q)
        rot_err, angle = _cone_error(axis, target_axis, cone)
        pos_err = p_ball - x
        if np.linalg.norm(pos_err) < pos_tol and angle <= cone + 1e-3:
            return q, True
        q = np.clip(q + _task_step(J, np.concatenate([pos_err, rot_err]), axis, q, q_nominal), lo, hi)
    return q, False


def solve_catch_ik(kin: Kinematics, p_ball: np.ndarray, v_ball: np.ndarray, q_guess: np.ndarray,
                   cone_half_angle_deg: float, restarts: int = 0, iters: int = 200, pos_tol: float = 0.005):
    """Joint config putting the cup center at the ball with its axis inside the
    cone around -v_ball, staying close to q_guess. Returns (q, success).

    The damped-least-squares IK is local; if it fails from q_guess, retry from
    `restarts` random perturbations and keep the solution quickest to reach.
    A configuration where the arm or cup penetrates the room counts as a failure.
    """
    target_axis = -v_ball / np.linalg.norm(v_ball)
    cone = np.radians(cone_half_angle_deg)
    q, ok = _solve_ik_from(kin, p_ball, target_axis, q_guess, q_guess, cone, iters, pos_tol)
    ok = ok and not kin.collides(q)
    if ok or not restarts:
        return q, ok
    lo, hi = kin.lo, kin.hi
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        q_init = np.clip(q_guess + rng.normal(0.0, 0.8, len(q_guess)), lo, hi)
        q, ok = _solve_ik_from(kin, p_ball, target_axis, q_init, q_init, cone, iters, pos_tol)
        if ok and not kin.collides(q):
            move_time = np.max(np.abs(q - q_guess) / VELOCITY_LIMITS)
            if best is None or move_time < best[0]:
                best = (move_time, q)
    return (best[1], True) if best else (q, False)


@dataclass
class CubicTrajectory:
    """Rest-to-rest joint-space cubic, as fast as the velocity limits allow."""

    q0: np.ndarray
    q1: np.ndarray

    def __post_init__(self):
        self.T = max(1.5 * np.max(np.abs(self.q1 - self.q0) / VELOCITY_LIMITS), 0.05)

    def __call__(self, t: float):
        s = np.clip(t / self.T, 0.0, 1.0)
        dq = self.q1 - self.q0
        moving = t < self.T
        return (self.q0 + dq * (3 * s**2 - 2 * s**3),
                dq * (6 * s - 6 * s**2) / self.T * moving,
                dq * (6 - 12 * s) / self.T**2 * moving)


class DiffIKReference:
    """Differential IK toward a fixed catch target, re-solved every `dt`.

    Commands a cup-center velocity toward the target and an axis rotation toward
    the target direction, maps it through damped least squares, scales to respect
    joint velocity limits, and integrates the joint reference.
    """

    def __init__(self, kin: Kinematics, p_target: np.ndarray, axis_target: np.ndarray, dt: float,
                 gain: float = 20.0, max_speed: float = 3.0):
        self.kin, self.p, self.a, self.dt = kin, p_target, axis_target, dt
        self.gain, self.max_speed = gain, max_speed
        self.lo, self.hi = kin.lo, kin.hi
        self.q, self.v, self.t_next = Q_READY.copy(), np.zeros(len(Q_READY)), 0.0

    def __call__(self, t: float):
        while t >= self.t_next:
            x, axis, J = self.kin(self.q)
            V = self.gain * (self.p - x)
            speed = np.linalg.norm(V)
            if speed > self.max_speed:
                V *= self.max_speed / speed
            w, _ = _cone_error(axis, self.a, 0.0)
            v = _task_step(J, np.concatenate([V, self.gain * w]), axis, self.q, Q_READY)
            v *= min(1.0, 1.0 / np.max(np.abs(v) / VELOCITY_LIMITS))
            self.q = np.clip(self.q + v * self.dt, self.lo, self.hi)
            self.v = v
            self.t_next += self.dt
        return self.q, self.v, np.zeros_like(self.q)


def plan_ik_catch(kin: Kinematics, throw: Throw, cone_half_angle_deg: float,
                  stride: int = 2, max_candidates: int = 40, belief=None,
                  q_start: np.ndarray = Q_READY, t_start: float = 0.0, min_descent: float = MIN_DESCENT):
    """Catch plan: the earliest in-workspace ball sample the arm can reach in time
    (or the least-late one if none), skipping targets whose joint path hits the room.
    Only points where the ball is coming down (v_z / |v| < -min_descent) count: the cup
    faces the incoming ball, so a rising ball would need an upside-down cup, and the ball
    falls back out of it. Returns (index, joint trajectory).

    By default it plans on the ground truth (the oracle). With a `belief` (see
    catcher.belief) it aims at the belief's planning path (`plan_mean`) instead.
    The motion starts from rest at q_start at time t_start; the returned trajectory
    runs on time relative to t_start.
    """
    pos, vel = (throw.pos, throw.vel) if belief is None else (belief.plan_mean, belief.plan_vel)
    best = None
    descending = vel[:, 2] < -min_descent * np.linalg.norm(vel, axis=1)
    candidates = np.flatnonzero(in_catch_zone(pos) & descending & (throw.times > t_start))[::stride][:max_candidates]
    # Cheap pass from the start pose first; random restarts only if nothing was reachable.
    for restarts in (0, 8):
        for k in candidates[::1 if restarts == 0 else 3]:
            q, ok = solve_catch_ik(kin, pos[k], vel[k], q_start, cone_half_angle_deg, restarts)
            if not ok:
                continue
            traj = CubicTrajectory(q_start, q)
            if kin.path_collides(traj, traj.T):
                continue
            slack = throw.times[k] - t_start - traj.T
            if best is None or slack > best[0]:
                best = (slack, int(k), traj)
            if slack >= 0.05:
                break
        if best is not None:
            break
    if best is None:  # nothing reachable: hold the start pose (counts as a miss)
        k = oracle_catch_index(throw) if belief is None else int(candidates[0]) if len(candidates) else throw.catch_index
        return k, CubicTrajectory(q_start, q_start)
    return best[1], best[2]


def oracle_catch_index(throw: Throw, min_time: float = 0.35) -> int:
    """Proposal's mean planner on ground truth: earliest in-workspace sample after a reaction time."""
    ok = np.flatnonzero(in_catch_zone(throw.pos) & (throw.times >= min_time))
    return int(ok[0]) if len(ok) else throw.catch_index


@dataclass
class CatchResult:
    t_catch: float
    p_ball: np.ndarray
    p_cup_center: np.ndarray
    position_error: float
    axis_angle_deg: float  # angle between cup opening axis and -v_ball
    success: bool  # the ball is still inside the cup at the end of the run (simulated contact)
    geometric_success: bool  # at t_catch the cup was within tolerance of the true ball (no contact)
    times: np.ndarray
    q: np.ndarray  # (T, 7) logged joint positions
    ball: np.ndarray  # (T, 3) ball position at the logged times (replayed, then simulated)
    t_release: float  # when the ball was handed to MuJoCo (inf if never)
    arm_room_contact: bool  # the arm touched the room at some point


def run_catch(throw: Throw, model: mujoco.MjModel | None = None, sim_time: float | None = None,
              controller: str = "ik", ik_dt: float = 5e-3, log_dt: float = 1 / 60,
              cone_half_angle_deg: float = 30.0, kp: float = 400.0, planner=None) -> CatchResult:
    """Simulate the arm moving to a catch target and score the catch against the true ball.

    controller="ik":     IK (with the cone constraint) once, then a joint-space cubic.
    controller="diffik": differential IK toward a fixed catch pose, every ik_dt.
    Both plan on the oracle (ground-truth) ball and are tracked by a computed-torque
    controller on the simulated iiwa.

    planner overrides the controller choice: a callable (kin, throw, cone_deg) ->
    (catch index, joint reference), e.g. the belief-based ones in catcher.planning.
    """
    model = model or build_model(throw.room)
    kin = Kinematics(model)
    # Aim slightly inside the success cone to leave room for tracking error.
    ik_cone_deg = cone_half_angle_deg - 5.0
    if planner is not None:
        k, reference = planner(kin, throw, ik_cone_deg)
    elif controller == "ik":
        k, reference = plan_ik_catch(kin, throw, ik_cone_deg)
    elif controller == "diffik":
        k = oracle_catch_index(throw)
    else:
        raise ValueError(f"unknown controller {controller!r}")
    t_c = float(throw.times[k])
    p_ball, v_ball = throw.pos[k], throw.vel[k]
    if planner is None and controller == "diffik":
        reference = DiffIKReference(kin, p_ball, -v_ball / np.linalg.norm(v_ball), ik_dt)
    sim_time = sim_time if sim_time is not None else min(t_c + 0.5, float(throw.times[-1]))

    data = mujoco.MjData(model)
    data.qpos[:N_ARM] = Q_READY
    dt = model.opt.timestep
    step_times = np.arange(0.0, sim_time + dt, dt)
    replay_pos, replay_vel = throw.traj.sample(step_times)
    set_ball(data, replay_pos[0], replay_vel[0])
    mujoco.mj_forward(model, data)
    arm_ids = [model.body(name).id for name in ARM_BODIES]
    arm_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, arm_ids)))
    ball_geom = model.geom("ball").id
    kd = 2 * np.sqrt(kp)
    Mqdd = np.zeros(model.nv)
    qdd = np.zeros(model.nv)
    log_t, log_q, log_ball = [], [], []
    t_release, arm_room_contact, i = np.inf, False, 0
    while data.time < sim_time:
        released = np.isfinite(t_release)
        if not released:  # replay the true flight until the ball comes near the arm
            i = min(int(round(data.time / dt)), len(step_times) - 1)
            set_ball(data, replay_pos[i], replay_vel[i])
            if np.min(np.linalg.norm(data.xpos[arm_ids] - replay_pos[i], axis=1)) < RELEASE_DIST:
                t_release = data.time
        if not log_t or data.time >= log_t[-1] + log_dt - 1e-9:
            log_t.append(data.time)
            log_q.append(data.qpos[:N_ARM].copy())
            log_ball.append(data.qpos[N_ARM:N_ARM + 3].copy())
        mujoco.mj_step1(model, data)  # computes M, bias forces and contacts at the current state
        for c in data.contact[:data.ncon]:
            g = {int(c.geom1), int(c.geom2)}
            if ball_geom not in g and len(g & arm_geoms) == 1:
                arm_room_contact = True
        q_ref, v_ref, a_ref = reference(data.time)
        qdd[:N_ARM] = a_ref + kp * (q_ref - data.qpos[:N_ARM]) + kd * (v_ref - data.qvel[:N_ARM])
        mujoco.mj_mulM(model, data, Mqdd, qdd)
        data.qfrc_applied[:N_ARM] = np.clip(Mqdd[:N_ARM] + data.qfrc_bias[:N_ARM], -TORQUE_LIMITS, TORQUE_LIMITS)
        mujoco.mj_step2(model, data)
    times, q, ball = np.array(log_t), np.array(log_q), np.array(log_ball)

    # Geometric score at t_c, using the logged arm configuration closest in time.
    p_cup_center, cup_axis, _ = kin(q[np.argmin(abs(times - t_c))])
    err = float(np.linalg.norm(p_ball - p_cup_center))
    angle = float(np.degrees(np.arccos(np.clip(cup_axis.dot(-v_ball / np.linalg.norm(v_ball)), -1, 1))))
    geometric = err < CUP_RADIUS - BALL_RADIUS / 2 and angle < cone_half_angle_deg
    # Physical score: the ball is in the cup at the end.
    caught = ball_in_cup(kin, data.qpos[:N_ARM].copy(), data.qpos[N_ARM:N_ARM + 3].copy())
    return CatchResult(t_c, p_ball, p_cup_center, err, angle, bool(caught), bool(geometric), times, q, ball,
                       float(t_release), arm_room_contact)


def default_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.9, 0.0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 3.6, -140.0, -18.0
    return cam


def add_path_to_scene(scene: mujoco.MjvScene, points: np.ndarray, rgba=(1.0, 0.45, 0.0, 0.5),
                      width: float = 0.006):
    """Draw a polyline (e.g. the ball path) as capsules into a scene's user geoms."""
    for a, b in zip(points[:-1], points[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1


def render_frames(model: mujoco.MjModel, result: CatchResult, path_points: np.ndarray,
                  height: int = 480, width: int = 640) -> list[np.ndarray]:
    """Offscreen-render the logged arm motion and ball."""
    data = mujoco.MjData(model)
    cam = default_camera()
    frames = []
    with mujoco.Renderer(model, height, width) as renderer:
        for q, p in zip(result.q, result.ball):
            data.qpos[:N_ARM] = q
            set_ball(data, p)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            add_path_to_scene(renderer.scene, path_points[::3])
            frames.append(renderer.render().copy())
    return frames
