import numpy as np

from catcher.ball_sim import simulate
from catcher.mujoco_scene import (Q_READY, Q_TUCKED, CubicTrajectory, Kinematics, build_model, plan_ik_catch, ready_pose_clear,
                                  run_catch, solve_catch_ik)
from catcher.room import box_room, pillar, random_pillar
from catcher.throws import Throw, pillar_footprint, sample_stage_throw


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


def test_random_pillars_clear_the_ready_pose():
    rng = np.random.default_rng(0)
    for _ in range(200):
        room = box_room()
        room.panels += random_pillar(rng, room)
        assert not Kinematics(build_model(room)).collides(Q_READY)


def test_collision_check_sees_the_pillar_and_the_floor():
    room = box_room()
    room.panels += pillar((0.6, 0.0), 0.1, room)
    assert Kinematics(build_model(room)).collides(Q_READY)
    folded = Q_READY.copy()
    folded[1], folded[3] = 1.9, 0.0
    assert Kinematics(build_model(box_room())).collides(folded)


def test_catch_ik_avoids_the_pillar():
    room = box_room()
    room.panels += pillar((0.95, 0.0), 0.15, room)
    kin = Kinematics(build_model(room))
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.array([rng.uniform(0.3, 0.75), rng.uniform(-0.5, 0.5), rng.uniform(0.4, 0.9)])
        q, ok = solve_catch_ik(kin, p, np.array([-4.0, 0.0, -3.0]), Q_READY, 25, restarts=4)
        if ok:
            assert not kin.collides(q)


def test_planner_only_aims_at_descending_balls():
    rng = np.random.default_rng(0)
    for _ in range(10):
        throw = sample_stage_throw(3, rng)
        kin = Kinematics(build_model(throw.room))
        k, traj = plan_ik_catch(kin, throw, 25)
        if not np.array_equal(traj.q0, traj.q1):
            assert throw.vel[k, 2] < 0


def test_rooms_with_a_board_through_the_robot_can_be_skipped():
    room = box_room()
    assert ready_pose_clear(room)
    room.panels += pillar((0.25, 0.0), 0.15, room)  # through both the ready and the tucked pose
    assert not ready_pose_clear(room)


def _blocking_room():
    room = box_room(hi=(4.5, 3.0, 7.0))
    room.panels += pillar((0.52, 0.0), 0.09, room)
    return room


def test_blocking_pillar_switches_to_the_tucked_pose_and_knows_the_sides():
    kin = Kinematics(build_model(_blocking_room()))
    assert kin.pillar is not None and np.array_equal(kin.q_ready, Q_TUCKED)
    assert not kin.collides(kin.q_ready)
    assert kin.side(kin.q_ready) == 0
    assert Kinematics(build_model(box_room())).pillar is None


def test_routes_go_around_the_blocking_pillar_and_switching_sides_is_slow():
    kin = Kinematics(build_model(_blocking_room()))
    v = np.array([-3.0, 0.0, -5.0])
    left, ok_l = solve_catch_ik(kin, np.array([0.55, 0.28, 0.6]), v, kin.q_ready, 25)
    right, ok_r = solve_catch_ik(kin, np.array([0.55, -0.28, 0.6]), v, kin.q_ready, 25)
    assert ok_l and ok_r and kin.side(left) == 1 and kin.side(right) == -1
    # Switching sides costs far more than committing: the arm has to unwind around the pillar.
    switch = CubicTrajectory(left, right).T
    commit = max(CubicTrajectory(kin.q_ready, left).T, CubicTrajectory(kin.q_ready, right).T)
    assert switch > 1.8 * commit and switch > 2.1  # longer than any lob past the pillar is in the air
    for a, b in ((kin.q_ready, left), (kin.q_ready, right), (left, right)):
        waypoints = kin.route(a, b)
        assert waypoints is not None and np.array_equal(waypoints[-1], b)
        legs = zip([a] + waypoints[:-1], waypoints)
        assert all(kin.leg_clear(p, q) for p, q in legs)


def test_throws_pass_beside_the_blocking_pillar_on_both_sides():
    rng = np.random.default_rng(0)
    sides = set()
    for _ in range(10):
        throw = sample_stage_throw(1, rng, pillar="blocking")
        center, half = pillar_footprint(throw.room)
        y = throw.pos[throw.catch_index, 1] - center[1]
        assert abs(y) > half and throw.bounces_before_catch == 0
        sides.add(np.sign(y))
    assert sides == {-1.0, 1.0}
