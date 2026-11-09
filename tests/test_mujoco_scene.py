import numpy as np

from catcher.ball_sim import simulate
from catcher.mujoco_scene import Q_READY, CubicTrajectory, Kinematics, build_model, run_catch, solve_catch_ik
from catcher.room import box_room
from catcher.throws import Throw, sample_stage_throw


def test_ik_reaches_target_inside_cone():
    kin = Kinematics(build_model())
    p, v = np.array([0.5, 0.1, 0.7]), np.array([-3.0, 0.0, -4.0])
    q, ok = solve_catch_ik(kin, p, v, Q_READY, cone_half_angle_deg=25)
    assert ok
    x, axis, _ = kin(q)
    assert np.linalg.norm(x - p) < 0.005
    assert np.degrees(np.arccos(axis.dot(-v / np.linalg.norm(v)))) <= 25.1


def test_stage1_oracle_catches():
    rng = np.random.default_rng(0)
    results = [run_catch(sample_stage_throw(1, rng)) for _ in range(5)]
    assert all(r.success for r in results)


def _hold(kin, throw, cone_deg):
    """Planner that never moves the arm."""
    return throw.catch_index, CubicTrajectory(Q_READY, Q_READY)


def test_ball_follows_the_true_path_until_released():
    throw = sample_stage_throw(2, np.random.default_rng(1))
    r = run_catch(throw)
    before = r.times < r.t_release
    assert before.sum() > 10 and r.t_release < r.t_catch
    truth, _ = throw.traj.sample(r.times[before])
    assert np.allclose(r.ball[before], truth, atol=1e-9)


def test_an_arm_that_does_not_move_does_not_catch():
    throw = sample_stage_throw(1, np.random.default_rng(0))
    r = run_catch(throw, planner=_hold)
    assert not r.success and not r.arm_room_contact


def test_ball_dropped_into_a_still_cup_stays_in():
    room = box_room()
    kin = Kinematics(build_model(room))
    center, axis, _ = kin(Q_READY)
    p0 = center + np.array([0.0, 0.0, 0.15])
    traj = simulate(room, p0, np.zeros(3), t_end=1.5)
    times = np.arange(0.0, 1.5, 0.01)
    pos, vel = traj.sample(times)
    k = int(np.argmin(np.linalg.norm(pos - center, axis=1)))
    throw = Throw(room, p0, np.zeros(3), traj, times, pos, vel, k, 0)
    r = run_catch(throw, planner=_hold, sim_time=1.4)
    assert r.success
