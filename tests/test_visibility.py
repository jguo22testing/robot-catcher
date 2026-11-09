import numpy as np
from pytest import approx

from catcher.belief import SyntheticPredictor
from catcher.room import box_room, pillar
from catcher.throws import sample_stage_throw
from catcher.visibility import CAMERA_POS, last_seen_time, visible


def test_pillar_hides_what_is_behind_it_and_walls_hide_nothing():
    room = box_room()
    room.panels += pillar((1.5, 0.0), 0.2, room)
    behind = np.array([3.0, 0.0, CAMERA_POS[2]])  # straight behind the pillar, at camera height
    beside = np.array([3.0, 2.0, CAMERA_POS[2]])
    in_front = np.array([0.8, 0.0, CAMERA_POS[2]])
    assert list(visible(room, np.array([behind, beside, in_front]))) == [False, True, True]
    assert visible(box_room(), behind[None]).all()


def test_last_seen_time():
    times = np.arange(10) * 0.1
    seen = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    assert last_seen_time(times, seen, 0.45) == approx(0.2)
    assert last_seen_time(times, seen, 0.75) == approx(0.7)
    assert last_seen_time(times, np.zeros(10, dtype=bool), 0.5) == 0.0


def test_belief_stops_sharpening_while_the_ball_is_hidden():
    throw = sample_stage_throw(2, np.random.default_rng(1))
    seen = throw.times < 0.3  # hidden from 0.3 s on
    hidden = SyntheticPredictor(throw, seed=3, seen=seen)
    open_view = SyntheticPredictor(throw, seed=3)
    k = throw.catch_index
    assert np.allclose(hidden(0.6).std[k], open_view(0.29).std[k], rtol=0.05)
    assert np.all(open_view(0.6).std[k] < hidden(0.6).std[k])
