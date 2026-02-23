import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fractals.core import BinaryTree, DragonFractal, Fractal
from fractals.presets import (
    DurerPentagon,
    Flowsnake,
    FudgeFlake,
    GoldenDragon,
    GoldenFlake,
    HeighwayDragon,
    KochFlake,
    Kochawave,
    LevyC,
    LevyTapestryInside,
    LevyTapestryOutside,
    Pentadendrite,
    Pentigree,
    Terdragon,
    TwinDragon,
    Z2Dragon,
    Z2Levy,
)


@pytest.mark.parametrize(
    "cls",
    [
        HeighwayDragon,
        TwinDragon,
        GoldenDragon,
        Terdragon,
        FudgeFlake,
        LevyC,
        LevyTapestryOutside,
        LevyTapestryInside,
        KochFlake,
        Kochawave,
        Pentadendrite,
        Pentigree,
        DurerPentagon,
        Z2Dragon,
        Z2Levy,
        Flowsnake,
        GoldenFlake,
    ],
)
def test_presets_construct_and_iterate(cls):
    fractal = cls()
    fractal.iterate(1)
    assert len(fractal.S) > 0


def test_fractal_base_methods():
    funcs = [lambda z: z, lambda z: z + 1]
    f = Fractal([0 + 0j, 1 + 0j], funcs)

    f.iterate(1)
    assert len(f.S) == 4
    assert len(f._plot_list) == 1

    f.divided_iterate(1)
    assert len(f._plot_list) == 2

    original = [arr.copy() if hasattr(arr, "copy") else arr for arr in f._plot_list]
    f.translate(1 + 0j, 0.0)
    assert len(f._plot_list) == 3
    assert np.isclose(f._plot_list[-1][0], original[0][0] + 1)

    f.translate_in_place(0 + 1j, 0.0)
    assert np.isclose(f._plot_list[0][0], original[0][0] + 1j)

    f.scale(2.0)
    assert np.isclose(f._plot_list[0][0], (original[0][0] + 1j) * 2)

    limits = f.calculate_limits()
    assert len(limits) == 4
    assert limits[0] <= limits[1]
    assert limits[2] <= limits[3]

    f.reset()
    np.testing.assert_allclose(f.S, np.array([0 + 0j, 1 + 0j]))


def test_presets_compute_limits_on_init():
    fractal = HeighwayDragon()
    assert hasattr(fractal, "limits")
    assert len(fractal.limits) == 4


def test_dragon_segment_inserts_nan_breaks():
    d = DragonFractal([0, 1], [lambda z: z, lambda z: z + 1])
    seg = d.segment([0, 1, 2, 3])
    assert np.isnan(seg[2])
    assert np.isnan(seg[-1])


def test_binary_tree_iterate_and_translate():
    tree = BinaryTree(0.5, 0.2)
    tree.iterate(1)
    assert tree.iterations == 2
    before = len(tree._plot_list)
    tree.translate(0, 0)
    assert len(tree._plot_list) == before + tree.iterations + 1
