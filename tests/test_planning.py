import numpy as np

from catcher.ball_sim import simulate
from catcher.belief import CHI2_3D_90, GaussianBelief, ParticlePredictor, SyntheticPredictor
from catcher.mujoco_scene import Q_READY, CubicTrajectory, Kinematics, build_model, run_catch
from catcher.planning import (HedgePlanner, HermiteTrajectory, MeanPlanner, SegmentedReference,
                              particle_predictor_for, synthetic_predictor_for)
from catcher.room import box_room, pillar
from catcher.throws import Throw, sample_stage_throw


def _flat_belief(std=0.1, n_steps=300, corr_time=0.3):
    times = np.arange(n_steps) * 0.01
    return GaussianBelief(times, np.tile([1.0, -2.0, 0.5], (n_steps, 1)), np.full((n_steps, 3), std),
                          np.zeros((n_steps, 3)), 0.0, corr_time)


def test_belief_samples_have_gaussian_marginals_and_smooth_correlation():
    belief = _flat_belief()
    samples = belief.sample(4000, np.random.default_rng(0))
    for k in (0, 50, 299):
        err = samples[:, k] - belief.mean[k]
        assert np.allclose(err.mean(0), 0, atol=0.01)
        assert np.allclose(err.std(0), 0.1, rtol=0.05)
    # Neighbouring steps are nearly identical; far-apart ones decorrelate.
    near = np.corrcoef(samples[:, 100, 0], samples[:, 101, 0])[0, 1]
    far = np.corrcoef(samples[:, 100, 0], samples[:, 250, 0])[0, 1]
    assert near > 0.95 and abs(far) < 0.1


def test_observed_part_of_the_belief_is_certain():
    rng = np.random.default_rng(0)
    throw = sample_stage_throw(1, rng)
    belief = SyntheticPredictor(throw)(0.5)
    past = throw.times <= 0.5
    assert np.all(belief.std[past] == 0) and np.array_equal(belief.mean[past], throw.pos[past])
    assert np.all(belief.std[~past] > 0)
    samples = belief.sample(5, rng)
    assert np.allclose(samples[:, past], throw.pos[past])


def test_synthetic_belief_is_calibrated():
    # The truth must fall in the 90% ellipsoid about 90% of the time, over prediction errors.
    throw = sample_stage_throw(2, np.random.default_rng(1))
    k = throw.catch_index
    inside = [SyntheticPredictor(throw, seed=s)(0.1).mahalanobis_sq(k, throw.pos[k]) <= CHI2_3D_90
              for s in range(600)]
    assert abs(np.mean(inside) - 0.9) < 0.05


def test_belief_sharpens_but_keeps_its_error_realization():
    throw = sample_stage_throw(2, np.random.default_rng(1))
    predictor = SyntheticPredictor(throw, seed=3)
    early, late = predictor(0.1), predictor(0.6)
    k = throw.catch_index
    assert np.all(late.std[k] < early.std[k])
    z_early = (early.mean[k] - throw.pos[k]) / early.std[k]
    z_late = (late.mean[k] - throw.pos[k]) / late.std[k]
    assert np.allclose(z_early, z_late)  # same standardized error, smaller in meters


def test_ellipsoid_samples_lie_inside_the_ellipsoid():
    belief = _flat_belief()
    points = belief.ellipsoid_samples(100, 500, np.random.default_rng(0))
    assert 0.8 * 500 < len(points) <= 500
    assert np.all(belief.mahalanobis_sq(100, points) <= CHI2_3D_90)


def test_segmented_reference_holds_then_moves_then_holds():
    q1, q2 = Q_READY + 0.2, Q_READY + 0.1
    ref = SegmentedReference([(0.5, CubicTrajectory(Q_READY, q1)), (2.0, CubicTrajectory(q1, q2))])
    assert np.array_equal(ref(0.0)[0], Q_READY) and not ref(0.0)[1].any()
    assert not np.allclose(ref(0.6)[0], Q_READY)
    assert np.allclose(ref(1.9)[0], q1) and np.allclose(ref(10.0)[0], q2)


def test_hedge_is_never_slower_in_the_worst_case_than_going_to_the_mean():
    rng = np.random.default_rng(0)
    planner = HedgePlanner(synthetic_predictor_for(0))
    checked = 0
    for _ in range(6):
        throw = sample_stage_throw(1, rng)
        planner_model = build_model(throw.room)
        run_catch(throw, model=planner_model, planner=planner)
        if planner.last is not None:
            assert planner.last.worst_case_time <= planner.last.mean_worst_case_time + 1e-9
            assert planner.last.n_samples > 1
            checked += 1
            planner.last = None
    assert checked >= 3


def test_planners_without_uncertainty_match_the_oracle_on_stage1():
    rng = np.random.default_rng(0)
    exact = synthetic_predictor_for(0, sigma0=0.0, growth=0.0, bounce_sigma=0.0)
    for _ in range(5):
        throw = sample_stage_throw(1, rng)
        model = build_model(throw.room)
        for planner in (MeanPlanner(exact), HedgePlanner(exact)):
            assert run_catch(throw, model=model, planner=planner).success


def test_hermite_segment_starts_from_the_given_state_and_ends_at_rest():
    q0, v0, q1 = Q_READY, np.full(7, 0.5), Q_READY + 0.3
    traj = HermiteTrajectory(q0, v0, q1, 0.4)
    q, v, _ = traj(0.0)
    assert np.allclose(q, q0) and np.allclose(v, v0)
    q, v, _ = traj(0.4 - 1e-9)
    assert np.allclose(q, q1, atol=1e-6) and np.allclose(v, 0, atol=1e-6)
    assert np.allclose(traj(1.0)[0], q1)


def test_replanned_reference_is_continuous():
    rng = np.random.default_rng(2)
    throw = sample_stage_throw(2, rng)
    kin = Kinematics(build_model(throw.room))
    _, ref = HedgePlanner(synthetic_predictor_for(0))(kin, throw, 25.0)
    assert len(ref.segments) > 2
    for t0, _ in ref.segments[1:]:
        before, after = ref(t0 - 1e-7), ref(t0)
        assert np.allclose(before[0], after[0], atol=1e-4) and np.allclose(before[1], after[1], atol=1e-2)


def _edge_throw(y0):
    """A ball flying in -x at height 1.2 m toward a pillar whose +y edge is at y = 0.15."""
    room = box_room()
    room.panels += pillar((2.0, 0.0), 0.15, room)
    p0, v0 = np.array([3.5, y0, 1.2]), np.array([-6.0, 0.0, 1.0])
    traj = simulate(room, p0, v0, t_end=1.0)
    times = np.arange(0.0, 1.0, 0.01)
    pos, vel = traj.sample(times)
    return Throw(room, p0, v0, traj, times, pos, vel, len(times) - 1, len(traj.bounces))


def test_particle_belief_splits_at_a_pillar_edge():
    throw = _edge_throw(0.15)  # straight at the edge
    modes = ParticlePredictor(throw, seed=0)(0.0).modes(70, 16, np.random.default_rng(0))
    assert len(modes) == 2 and min(m.weight for m in modes) > 0.2
    assert abs(sum(m.weight for m in modes) - 1) < 1e-9
    assert all(0.2 < m.split_time < 0.35 for m in modes)  # the ball reaches the pillar at ~0.23 s
    assert np.linalg.norm(modes[0].mean - modes[1].mean) > 0.5


def test_particle_belief_has_one_mode_far_from_any_edge():
    modes = ParticlePredictor(_edge_throw(0.6), seed=0)(0.0).modes(70, 16, np.random.default_rng(0))
    assert len(modes) == 1 and modes[0].weight == 1.0 and np.isinf(modes[0].split_time)


def test_particle_belief_is_calibrated():
    throw = sample_stage_throw(1, np.random.default_rng(1))
    k = throw.catch_index
    inside = []
    for s in range(150):
        belief = ParticlePredictor(throw, n=64, seed=s)(0.1)  # stage 1: no bounce, a single mode
        inside.append(belief.mahalanobis_sq(k, throw.pos[k]) <= CHI2_3D_90)
    assert abs(np.mean(inside) - 0.9) < 0.07


def test_hedge_scores_cover_or_commit_on_a_two_mode_belief():
    throw = sample_stage_throw(2, np.random.default_rng(0), pillar="front")
    planner = HedgePlanner(particle_predictor_for(0))
    run_catch(throw, planner=planner)
    assert planner.last is not None and len(planner.last.choices) == planner.last.n_replans
    assert set(planner.last.choices) <= {"single", "cover", "none"} | {f"mode{i}" for i in range(10)}


def test_particle_belief_plans_on_one_outcome_not_the_average():
    belief = ParticlePredictor(_edge_throw(0.15), seed=0)(0.0)
    modes = belief.modes(70)
    assert len(modes) == 2
    gap = np.linalg.norm(modes[0].mean - modes[1].mean)
    to_nearest_mode = min(np.linalg.norm(belief.plan_mean[70] - m.mean) for m in modes)
    assert to_nearest_mode < 0.1 * gap
    assert min(np.linalg.norm(belief.mean[70] - m.mean) for m in modes) > 0.2 * gap


def test_uncertainty_aware_chooses_left_right_or_wait_at_a_blocking_pillar():
    rng = np.random.default_rng(0)
    seen = set()
    planned = 0
    for _ in range(6):
        throw = sample_stage_throw(1, rng, pillar="blocking")
        planner = HedgePlanner(particle_predictor_for(0, sigma_v=0.2))
        run_catch(throw, planner=planner)
        if planner.last is not None:  # None: nothing reachable at t_plan, the arm holds
            planned += 1
            seen |= set(planner.last.choices)
    assert planned >= 3
    assert seen <= {"left", "right", "wait", "none"} and seen & {"left", "right"}
