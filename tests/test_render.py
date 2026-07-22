"""Tests for render."""

import os

import numpy as np

from idg_social_nav.env import SocialNavEnv
from idg_social_nav.render import render_frame


def _make_env(gesture: str = "STOP") -> SocialNavEnv:
    env = SocialNavEnv(
        scenario="frontal_gesture",
        variant={"gesture": gesture, "ped_start_col": 7, "ped_delay": 0},
        randomize_variant=False,
    )
    env.reset()
    return env


class TestRenderFrame:
    def test_shape_and_dtype(self):
        env = _make_env()
        frame = render_frame(env)
        h, w = env.walls.shape
        assert frame.shape == (h * 48, w * 48, 3)
        assert frame.dtype == np.uint8

    def test_tile_size(self):
        env = _make_env()
        frame = render_frame(env, tile_size=16)
        h, w = env.walls.shape
        assert frame.shape == (h * 16, w * 16, 3)

    def test_deterministic_between_calls(self):
        env = _make_env()
        assert np.array_equal(render_frame(env), render_frame(env))

    def test_deterministic_across_identical_envs(self):
        assert np.array_equal(render_frame(_make_env()), render_frame(_make_env()))

    def test_gesture_changes_pixels(self):
        stop_frame = render_frame(_make_env("STOP"))
        go_frame = render_frame(_make_env("GO"))
        assert stop_frame.shape == go_frame.shape
        assert not np.array_equal(stop_frame, go_frame)

    def test_annotate_off_still_renders(self):
        env = _make_env()
        frame = render_frame(env, annotate=False)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3

    def test_headless_safe(self):
        # the module forces a video driver before pygame initializes,
        # so this test should pass even in a headless CI environment
        assert os.environ.get("SDL_VIDEODRIVER") is not None
        frame = _make_env().render_frame()  # env's lazy import path
        assert frame.shape[-1] == 3
