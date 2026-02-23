from pathlib import Path
import uuid

import numpy as np
import pytest
from PIL import Image, ImageSequence

from fractals.array import ArrayFractal, DEFAULT_INITIAL_POLYGON


@pytest.fixture
def eq_even():
    return np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def eq_odd():
    return np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7],
            [0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.3],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def fast_limits(monkeypatch):
    monkeypatch.setattr(
        ArrayFractal,
        "calculate_limits",
        lambda self: np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    )


def test_from_imaginary_configures_polygon_mode(fast_limits):
    fractal = ArrayFractal.from_imaginary(
        [0 + 0j, 1 + 0j],
        [lambda z: z, lambda z: 0.5 * z + 0.5],
    )
    assert fractal.segment_plot is True
    assert fractal.requires_initial_polygon is False
    np.testing.assert_allclose(fractal.initial_polygon, np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_infer_requires_initial_polygon_and_default_line(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even)
    assert fractal.requires_initial_polygon is True
    np.testing.assert_allclose(fractal.initial_polygon, DEFAULT_INITIAL_POLYGON)
    np.testing.assert_allclose(fractal.S, DEFAULT_INITIAL_POLYGON)


def test_set_initial_polygon_updates_segment_size_when_required(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even)
    custom = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    fractal.set_initial_polygon(custom)
    assert fractal._segment_size == 3
    np.testing.assert_allclose(fractal.S, custom)


def test_set_initial_polygon_validates_shape(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even)
    with pytest.raises(ValueError):
        fractal.set_initial_polygon(np.array([0.0, 1.0]))


def test_create_functions_explicit_probabilities(fast_limits, eq_odd):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_odd)
    assert len(fractal.trans_list) == 2
    np.testing.assert_allclose(fractal.prob_list, np.array([0.7, 0.3]))


def test_create_functions_computes_probabilities_for_even_eq(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    assert len(fractal.prob_list) == 2
    np.testing.assert_allclose(sum(fractal.prob_list), 1.0)


def test_calculate_probabilities_handles_zero_determinant():
    trans_list = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    ]
    probs = ArrayFractal.calculate_probabilities(trans_list)
    assert probs[0] > 0
    np.testing.assert_allclose(sum(probs), 1.0)


def test_apply_works_for_vector_and_matrix():
    trans = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, -1.0]], dtype=np.float64)
    point = np.array([1.0, 2.0], dtype=np.float64)
    out1 = ArrayFractal.apply(point, trans)
    np.testing.assert_allclose(out1, np.array([3.0, 3.0]))

    points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    out = np.empty_like(points)
    out2 = ArrayFractal.apply(points, trans, out=out)
    assert out2 is out
    np.testing.assert_allclose(out2, np.array([[1.0, -1.0], [3.0, 1.0]]))


def test_deterministic_iterate_uses_initial_polygon_when_needed(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even, segment_plot=True)
    fractal.iterate(1)
    assert len(fractal.S) == 4
    assert np.isnan(fractal._plot_list[0]).any()


def test_deterministic_iterate_with_existing_polygon_seed(fast_limits, eq_even):
    s0 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    fractal = ArrayFractal(s0, eq_even, requires_initial_polygon=False)
    fractal.deterministic_iterate(1)
    assert len(fractal.S) == len(s0) * len(fractal.trans_list)


def test_divided_iterate_splits_plot_blocks(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even, segment_plot=True)
    fractal.divided_iterate(1)
    assert len(fractal._plot_list) == len(fractal.trans_list)
    assert all(block.shape[1] == 2 for block in fractal._plot_list)


def test_random_apply_and_random_iterate(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    point, idx = fractal.random_apply(np.array([0.0, 0.0]))
    assert point.shape == (2,)
    assert 0 <= idx < len(fractal.trans_list)

    fractal.random_iterate(10)
    assert fractal.S.shape == (10, 2)
    assert len(fractal.trans_used) == 10


def test_reset_restores_initial_state(fast_limits, eq_even):
    s0 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    fractal = ArrayFractal(s0, eq_even)
    fractal.iterate(1)
    fractal.reset()
    np.testing.assert_allclose(fractal.S, s0)
    assert len(fractal._plot_list) == 1


def test_translate_translate_in_place_and_scale(fast_limits, eq_even):
    s0 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    fractal = ArrayFractal(s0, eq_even, requires_initial_polygon=False)
    fractal.translate(1 + 0j, 0.0)
    assert len(fractal._plot_list) == 2
    np.testing.assert_allclose(fractal._plot_list[1], s0 + np.array([1.0, 0.0]))

    fractal.translate_in_place(np.array([0.0, 1.0]), 0.0)
    np.testing.assert_allclose(fractal._plot_list[0], s0 + np.array([0.0, 1.0]))

    fractal.scale(2.0)
    np.testing.assert_allclose(fractal._plot_list[0], (s0 + np.array([0.0, 1.0])) * 2.0)


def test_offset_vector_validation():
    with pytest.raises(ValueError):
        ArrayFractal._offset_vector(np.array([[1.0, 2.0, 3.0]]))


def test_collect_plot_points_and_make_image(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even, segment_plot=True)
    fractal.iterate(1)
    points = fractal._collect_plot_points()
    assert points.shape[1] == 2

    image = fractal.make_image(resolution=(64, 64))
    assert isinstance(image, Image.Image)
    assert image.size == (64, 64)


def test_plot_returns_image_and_autoscales(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    fractal.iterate(1)
    image = fractal.plot(autoscale=True, resolution=(32, 32))
    assert isinstance(image, Image.Image)
    assert image.size == (32, 32)


def test_save_image_writes_file(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    fractal.random_iterate(5_000)
    out = Path(f"array_test_{uuid.uuid4().hex}.png")
    try:
        fractal.save_image(str(out), resolution=(48, 48))
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        out.unlink(missing_ok=True)
        out.with_suffix(".svg").unlink(missing_ok=True)


def test_save_image_requires_chaos_game_iteration(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    fractal.iterate(1)
    with pytest.raises(RuntimeError):
        fractal.save_image("should_not_exist.png", resolution=(32, 32))


def test_save_gif_writes_file(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even, segment_plot=True)
    fractal.iterate(1)
    out = Path(f"{type(fractal).__name__}_2.gif")
    try:
        fractal.save_gif(
            2,
            duration=20,
            resolution=(64, 64),
            deterministic_png_resolution=(320, 240),
        )
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        out.unlink(missing_ok=True)


def test_save_svg_writes_file_for_segmented_deterministic(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even, segment_plot=True)
    fractal.iterate(3)
    svg = Path(f"array_test_{uuid.uuid4().hex}.svg")
    try:
        fractal.save_svg(str(svg), resolution=(200, 200))
        assert svg.exists()
        data = svg.read_text(encoding="utf-8")
        assert "<polyline" in data
        assert "stroke-width=" in data
    finally:
        svg.unlink(missing_ok=True)


def test_svg_stroke_width_scales_down_with_iterations(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_even, segment_plot=True)
    w0 = fractal._svg_stroke_width((800, 800))
    fractal.iterate(8)
    w8 = fractal._svg_stroke_width((800, 800))
    assert w8 < w0


def test_array_deterministic_gif_has_visible_content_and_frame_delta(fast_limits, eq_even):
    eq_dragon = np.array(
        [
            [0.5, -0.5, 0.5, 0.5, 0.0, 0.0],
            [-0.5, -0.5, 0.5, -0.5, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    fractal = ArrayFractal(np.array([[0.0, 0.0]]), eq_dragon, segment_plot=True)
    out = Path(f"{type(fractal).__name__}_4.gif")
    try:
        fractal.save_gif(
            4,
            duration=15,
            resolution=(256, 256),
            deterministic_png_resolution=(800, 600),
        )
        assert out.exists()
        with Image.open(out) as im:
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(im)]
        assert len(frames) >= 2
        arrays = [np.array(frame) for frame in frames]
        non_empty = any(np.count_nonzero(arr[:, :, :3] < 250) > 0 for arr in arrays)
        assert non_empty
        assert any(np.any(arrays[i] != arrays[i + 1]) for i in range(len(arrays) - 1))
    finally:
        out.unlink(missing_ok=True)


def test_array_random_gif_has_visible_content_and_frame_delta(fast_limits, eq_odd):
    eq_random = np.array(
        [
            [0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85],
            [0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07],
            [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07],
            [0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01],
        ],
        dtype=np.float64,
    )
    fractal = ArrayFractal(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        eq_random,
        segment_plot=False,
        requires_initial_polygon=False,
    )
    fractal.random_iterate(250)
    out = Path(f"{type(fractal).__name__}_4.gif")
    try:
        fractal.save_gif(4, duration=15, resolution=(256, 256))
        assert out.exists()
        with Image.open(out) as im:
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(im)]
        assert len(frames) >= 2
        arrays = [np.array(frame) for frame in frames]
        non_empty = any(np.count_nonzero(arr[:, :, 3] > 0) > 0 for arr in arrays)
        assert non_empty
        assert any(np.any(arrays[i] != arrays[i + 1]) for i in range(len(arrays) - 1))
    finally:
        out.unlink(missing_ok=True)


def test_get_plot_fig_ax_properties_are_available(fast_limits, eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    assert fractal.plot_fig is None
    assert fractal.plot_ax is None


def test_calculate_limits_returns_bounding_box(eq_even):
    fractal = ArrayFractal(np.array([[0.0, 0.0], [1.0, 0.0]]), eq_even)
    limits = fractal.calculate_limits()
    assert limits.shape == (4,)
    assert limits[0] <= limits[1]
    assert limits[2] <= limits[3]
