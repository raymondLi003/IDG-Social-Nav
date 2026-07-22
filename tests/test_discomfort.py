"""Tests for the graded social discomfort field (discomfort.py)."""

import numpy as np
import pytest

from idg_social_nav.core import LEFT, RIGHT, Gesture
from idg_social_nav.discomfort import DiscomfortParams, discomfort_field, has_line_of_sight
from idg_social_nav.scenarios import make_scenario


def _open_room():
    return make_scenario(
        "frontal_approach", {"ped_start_col": 7, "ped_delay": 0}).walls


def _doorway():
    return make_scenario("narrow_doorway", {"ped_delay": 0}).walls


class TestIntensities:
    def test_default_profile(self):
        field = discomfort_field(
            _open_room(), [((3, 5), LEFT, Gesture.NONE)], DiscomfortParams())
        assert field[3, 5] == pytest.approx(1.0)                # pedestrian cell
        assert field[3, 4] == pytest.approx(2 / 3, abs=1e-5)    # front 1
        assert field[3, 3] == pytest.approx(1 / 3, abs=1e-5)    # front 2
        assert field[3, 6] == pytest.approx(0.0)                # behind 1
        assert field[2, 5] == pytest.approx(1 / 3, abs=1e-5)    # side 1
        assert field[4, 5] == pytest.approx(1 / 3, abs=1e-5)    # side 1

    def test_bounded_and_typed(self):
        field = discomfort_field(
            _open_room(), [((3, 5), LEFT, Gesture.STOP)], DiscomfortParams())
        assert field.dtype == np.float32
        assert float(field.min()) >= 0.0
        assert float(field.max()) <= 1.0

    def test_walls_carry_no_field(self):
        walls = _open_room()
        field = discomfort_field(
            walls, [((3, 5), LEFT, Gesture.NONE)], DiscomfortParams())
        assert not field[walls == 1].any()


class TestGestureModulation:
    def test_stop_extends_and_go_shrinks_front(self):
        walls = _open_room()
        params = DiscomfortParams()
        ped = ((3, 5), LEFT)
        f_none = discomfort_field(walls, [(*ped, Gesture.NONE)], params)
        f_stop = discomfort_field(walls, [(*ped, Gesture.STOP)], params)
        f_go = discomfort_field(walls, [(*ped, Gesture.GO)], params)

        # frontal ordering at the cell directly in front of the pedestrian
        assert f_stop[3, 4] > f_none[3, 4] > f_go[3, 4]
        assert f_stop[3, 4] == pytest.approx(1 - 1 / 6, abs=1e-5)
        assert f_none[3, 4] == pytest.approx(2 / 3, abs=1e-5)
        assert f_go[3, 4] == pytest.approx(1 / 3, abs=1e-5)

        # gestures modulate only the frontal semi-axis
        assert f_stop[2, 5] == pytest.approx(f_none[2, 5], abs=1e-5)
        assert f_go[2, 5] == pytest.approx(f_none[2, 5], abs=1e-5)
        assert f_stop[3, 5] == f_go[3, 5] == pytest.approx(1.0)


class TestLineOfSight:
    def test_wall_blocks_field(self):
        walls = _doorway()
        # pedestrian at (2, 6) facing LEFT is behind the doorway wall 
        # so (2, 4) is not in line of sight
        assert walls[2, 5] == 1
        assert not has_line_of_sight(walls, (2, 6), (2, 4))
        field = discomfort_field(
            walls, [((2, 6), LEFT, Gesture.NONE)], DiscomfortParams())
        assert field[2, 4] == 0.0

    def test_field_passes_through_gap(self):
        walls = _doorway()
        assert has_line_of_sight(walls, (3, 6), (3, 4))
        field = discomfort_field(
            walls, [((3, 6), LEFT, Gesture.NONE)], DiscomfortParams())
        assert field[3, 4] == pytest.approx(1 / 3, abs=1e-5)

    def test_endpoints_never_block(self):
        walls = _doorway()
        assert has_line_of_sight(walls, (3, 4), (3, 6))
        assert has_line_of_sight(walls, (3, 5), (3, 6))


class TestMultiPedestrian:
    def test_max_combine(self):
        walls = _open_room()
        params = DiscomfortParams()
        p1 = ((3, 3), RIGHT, Gesture.NONE)
        p2 = ((3, 7), LEFT, Gesture.NONE)
        f1 = discomfort_field(walls, [p1], params)
        f2 = discomfort_field(walls, [p2], params)
        combined = discomfort_field(walls, [p1, p2], params)
        np.testing.assert_allclose(combined, np.maximum(f1, f2))
        # both frontal zones reach (3, 5) at 1/3: max, never a sum
        assert combined[3, 5] == pytest.approx(1 / 3, abs=1e-5)

    def test_no_pedestrians_is_zero(self):
        field = discomfort_field(_open_room(), [], DiscomfortParams())
        assert not field.any()
