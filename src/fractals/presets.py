"""Named fractal presets built on top of core abstractions."""

from .core import (
    BinaryTree,
    DragonFractal,
    Fractal,
    IFS_function,
    PENTAGON,
    PHI,
    S0_TWIN,
    S0i,
    cmath,
    math,
    np,
)


class HeighwayDragon(DragonFractal):
    """[Heighway Dragon](https://larryriddle.agnesscott.org/ifs/heighway/heighway.htm)"""

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["dragon"])


class TwinDragon(DragonFractal):
    """[Twin Dragon](https://larryriddle.agnesscott.org/ifs/heighway/twindragon.htm)"""

    def __init__(self):
        super().__init__(S0_TWIN, IFS_function["twin_dragon"])


class GoldenDragon(DragonFractal):
    """[Golden Dragon](https://larryriddle.agnesscott.org/ifs/heighway/goldenDragon.htm)"""

    def __init__(self):
        super().__init__(S0i, IFS_function["golden_dragon"])


class Terdragon(Fractal):
    """[Terdragon](https://larryriddle.agnesscott.org/ifs/heighway/terdragon.htm)"""

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["terdragon"])


class FudgeFlake(Terdragon):
    """[Fudgeflake](https://larryriddle.agnesscott.org/ifs/heighway/fudgeflake.htm)"""

    def tile(self):
        self.translate(0, math.pi / 3)
        self.translate(1, 2 * math.pi / 3)


class LevyC(Fractal):
    """[Levy C Curve](https://larryriddle.agnesscott.org/ifs/levy/levy.htm)"""

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["levy_c"])


class LevyTapestryOutside(LevyC):
    """[Levy Tapestry](https://larryriddle.agnesscott.org/ifs/levy/tapestryOutside.htm)"""

    def tile(self):
        self.translate(1, math.pi)


class LevyTapestryInside(LevyC):
    """[Levy Tapestry](https://larryriddle.agnesscott.org/ifs/levy/tapestryInside.htm)"""

    def tile(self):
        translations = [(-1j, math.pi * 0.5), (1, -math.pi * 0.5), (1 - 1j, math.pi)]
        for off, theta in translations:
            self.translate(off, theta)


class KochFlake(Fractal):
    """
    [Koch Flake](https://larryriddle.agnesscott.org/ifs/kcurve/kcurve.htm)

    Note: this is constructed as a koch curve, then tiled.
    """

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["koch_flake"])

    def tile(self):
        translations = [
            (cmath.rect(-1, 2 * math.pi / 3), 2 * math.pi / 3),
            (1, -2 * math.pi / 3),
        ]
        for off, theta in translations:
            self.translate(off, theta)


class Kochawave(Fractal):
    """Kochawave Curve"""

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["kochawave"])

    # def tile(self):
    #     translations = [
    #         (cmath.rect(1, math.pi / 3), -2 * math.pi / 3),
    #         (1, 2 * math.pi / 3),
    #     ]
    #     for off, theta in translations:
    #         self.translate(off, theta)


class Pentadendrite(Fractal):
    """[Pentadendrite](https://larryriddle.agnesscott.org/ifs/pentaden/penta.htm)"""

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["pentadendrite"])

    def tile(self):
        translations = zip(PENTAGON[:4], np.arange(72, 361, 72) * math.pi / 180)
        for offset, angle in translations:
            self.translate(offset, angle)


class Pentigree(Fractal):
    """[Pentigree](https://larryriddle.agnesscott.org/ifs/pentaden/pentigree.htm)"""

    def __init__(self):
        super().__init__(S0i, IFS_function["pentigree"])

    def tile(self):
        translations = zip(PENTAGON[:4], np.arange(72, 361, 72) * math.pi / 180)
        for offset, angle in translations:
            self.translate(offset, angle)


class DurerPentagon(Fractal):
    """A different implementation of durer's pentagon"""
    def __init__(self):
        super().__init__(S0i, IFS_function["durer_pentagon"])

    def tile(self):
        translations = zip(PENTAGON[:4], np.arange(72, 361, 72) * math.pi / 180)
        for offset, angle in translations:
            self.translate(offset, angle)


class Z2Dragon(DragonFractal):
    def __init__(self):
        super().__init__([0, 1], IFS_function["z2_golden_dragon"])


class Z2Levy(DragonFractal):
    def __init__(self):
        super().__init__(S0i, IFS_function["z2_levy"])


class Flowsnake(DragonFractal):
    """[Flowsnake](https://larryriddle.agnesscott.org/ifs/ksnow/flowsnake.htm) inheriting from :class:`~fractal.DragonFractal`"""

    def __init__(self):
        super().__init__(S0i, IFS_function["flowsnake"])


class GoldenFlake(BinaryTree):
    """[GoldenFlake](https://larryriddle.agnesscott.org/ifs/pentagon/Durer.htm)"""

    def __init__(self):
        super().__init__(1 / PHI, 0.8 * math.pi)

    def tile(self):
        for angle in np.linspace(0, 2 * math.pi, 6):
            self.translate(0, angle)

    def iterate(self, i):
        for _ in range(i):
            self._plot_list.clear()
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
                self._plot_list.append(S)
                self.iterations += 1
            self.S = S
