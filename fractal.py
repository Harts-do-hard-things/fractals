# -*- coding: utf-8 -*-
"""A module for viewing fractals

Basic usage:

```python
heighway = HeighwayDragon()
heighway.iterate(15) # More than 15 iterations can cause crashing
heighway.plot()
```

The fractals available are:
  - [HeighwayDragon](index.md#fractal.HeighwayDragon)
  - [TwinDragon](index.md#fractal.TwinDragon)
  - [GoldenDragon](index.md#fractal.GoldenDragon)
  - [Terdragon](index.md#fractal.Terdragon)
  - [FudgeFlake](index.md#fractal.FudgeFlake)
  - [LevyC](index.md#fractal.LevyC)
  - [LevyTapestryOutside](index.md#fractal.LevyTapestryOutside)
  - [LevyTapestryInside](index.md#fractal.LevyTapestryInside)
  - [KochFlake](index.md#fractal.KochFlake)
  - [Pentadendrite](index.md#fractal.Pentadendrite)
  - [Pentigree](index.md#fractal.Pentigree)
  - [Flowsnake](index.md#fractal.Flowsnake)
  - [GoldenFlake](index.md#fractal.GoldenFlake)
"""


# N.B. I used segment in the DragonFractal code
# All fractals requiring segmentation should use DragonFractal as parent


import cmath
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import animatplot as amp

try:
    import gif
except ModuleNotFoundError:
    gif = None


class Fractal:
    """A class used to draw and calculate Fractals.

    Fractals can be generated by using one of the premade classes, or by
    creating a new fractal with arbitrary starting points and functions.
    Note: The majority of funtion systems don't result in fractals.

    Once created, fractals are iterated using (index.md#Fractal.iterate)

    Based on research by
    [Larry Riddle](https://larryriddle.agnesscott.org/ifs/ifs.htm)

    Attributes
    ----------
    S : list[complex]

        Default S0 = (0 + 0j) to (1 + 0j)
        the current points of the fractal

    func_list : list[functions]

        the functions to be applied every iteration


    Methods
    -------
    iterate(i: int)
        Applies func_list to the current points, S, and updates S to match

    plot()
        Plots the points S using matplotlib

    save_gif(iterations: int, duration: int = 1000)
    saves a gif at '__name__ _iterations.gif' with a frame duration of duration
    milliseconds

    """

    limits = False

    def __init__(self, S0: list[complex], func_list: list[Callable]):
        """
        Parameters
        ----------
        S0 : list[complex]
            The initial points to iterate
        func_list : list[Callable]
            A list of funtions that determines the function system

        Returns
        -------
        Fractal

        """
        self._S0 = S0
        self.S = S0
        self.func_list = func_list
        self._plot_list = [S0]
        self.plot_handle = []

    def reset(self):
        self.S = self._S0
        self._plot_list = [self._S0]

    def iterate(self, i: int=1) -> None:
        """maps the functions to S, reassigning S

        if used with the gif package
        clears the plot list (ensures proper gif plotting)
        appends S back to plot list

        Parameters
        ----------
        i : int
            The number of iterations to advance from the current state

        Returns
        -------
        None

        """
        self._plot_list.clear()
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
            self.S = S
        self._plot_list.append(S)

    # Rotate and translate (in that order) & create a copy
    def translate(self, offset: complex, angle: float) -> None:
        """Translate and rotate the fractal in the complex plane

        Parameters
        ----------
        offset : complex
            The vector to use as offset
        angle : float
            The angle (in radians) to rotate

        Returns
        -------
        None

        """
        s_trans = [i * cmath.exp(angle * 1j) + offset
                   for i in self._plot_list[0]]
        self._plot_list.append(s_trans)

    def tile(self):
        pass

    def translate_in_place(self, offset: complex, angle: float) -> None:
        self._plot_list = [ [i * cmath.exp(angle * 1j) + offset for i in j] for j in self._plot_list]

    def scale(self, scale: float) -> None:
        self._plot_list = [ [i * scale for i in j] for j in self._plot_list]

    def plot(self, autoscale=True):
        """Plots the fractal for human viewing"""
        self.tile()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_handle = [
            ax.plot(np.real(s), np.imag(s), color="tab:blue") for s in self._plot_list
            # (ax.plot(
            #     np.real(s)[:len(s)//len(self.func_list)],
            #     np.imag(s)[:len(s)//len(self.func_list)], color="tab:blue"),
            # ax.plot(
            #     np.real(s)[len(s)//len(self.func_list):],
            #     np.imag(s)[len(s)//len(self.func_list):], color="tab:red"))
            # for s in self._plot_list
        ]
        if self.limits and not autoscale:
            ax.set_xlim(self.limits[:2])
            ax.set_ylim(self.limits[2:])
        ax.set_aspect("equal")
        ax.axis("off")
        plt.show()

    # Not intended for Call except through save_gif method
    if gif:

        @gif.frame
        def _gif_plot(self) -> None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if self.limits:
                ax.set_xlim(self.limits[:2])
                ax.set_ylim(self.limits[2:])
            else:
                ax.set_aspect("equal")
            ax.axis("off")
            for s in self._plot_list:
                ax.plot(np.real(s), np.imag(s), color="tab:blue")

        def save_gif(self, iterations: int, duration: int = 1000) -> None:
            """
            Create a gif with a specific duration in milliseconds

            Parameters
            ----------
            iterations : int
                the number of iterations to include
            duration : int, optional
                The time in milliseconds for the gif to run.
                The default is 1000.

            Returns
            -------
            None

            """
            self.tile()
            frames = [self._gif_plot()]
            for _ in range(iterations - 1):
                self.iterate(1)
                self.tile()
                frame = self._gif_plot()
                frames.append(frame)
            gif.save(
                frames,
                f"{type(self).__name__}_{iterations}.gif",
                duration
            )


class DragonFractal(Fractal):
    """A class used to draw and calculate Fractals
    that require nans inbetween segments

    Based loosely on research by
    [Larry Riddle](https://larryriddle.agnesscott.org/ifs/ifs.htm)
    """

    def segment(self, points: list[complex]) -> list:
        len_ = len(self._S0)
        lines = []
        for i in range(len(points) - len_ + 1):
            if i % len_ == 0:
                lines.extend(points[i:i + len_])
                lines.append(np.nan)
        return lines

    def plot(self, autoscale=True):
        self._plot_list = [self.segment(s) for s in self._plot_list]
        super().plot(autoscale)

    if gif:
        def save_gif(self, iterations: int, duration=1000):
            self.tile()
            frames = [self._gif_plot()]
            for _ in range(iterations - 1):
                self.iterate(1)
                self._plot_list = [self.segment(s) for s in self._plot_list]
                self.tile()
                frame = self._gif_plot()
                frames.append(frame)
            gif.save(
                frames,
                "{0}_{1}.gif".format(type(self).__name__, iterations),
                duration
            )


class BinaryTree(DragonFractal):
    """
    Creates Binary Trees Based off of
    [Larry Riddle's Webpage](https://larryriddle.agnesscott.org/ifs/pythagorean/symbinarytree.htm )

    See Documentation for [DragonFractal](index.md#fractal.DragonFractal)
    for implementation details
    and attached resource for creation Details
    """
    def __init__(self, B_r: float, theta: float):
        super().__init__(
            S0=[0, 1j],
            func_list=[
                lambda z: B_r * z * cmath.rect(1, theta) + 1j,
                lambda z: B_r * z * cmath.rect(1, -theta) + 1j,
            ],
        )
        self.iterations = 0

    def iterate(self, i: int):
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
                self._plot_list.append(S)
                self.iterations += 1
            self.S = S

    def translate(self, offset: complex, angle: float):
        for j in range(self.iterations + 1):
            # print(j)
            s_trans = [i * cmath.exp(angle * 1j) + offset
                       for i in self._plot_list[j]]
            self._plot_list.append(s_trans)


class _AnimFractal(Fractal):
    """An experimental class
    TODO: Finish and document"""
    def create_block(self, X: list, Y: list) -> list:
        X.append(np.real(self.S))
        Y.append(np.imag(self.S))
        return X, Y

    def animate(self, i: int):
        x = [np.real(self._S0)]
        y = [np.imag(self._S0)]
        for _ in range(i):
            self.iterate(1)
            x, y = self.create_block(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Test")
        plt.axis(self.limits)
        timeline = amp.Timeline(range(i + 1), units="iter", fps=2)
        block = amp.blocks.Line(
            np.array(x, dtype=object), np.array(y, dtype=object), ax=ax
        )
        anim = amp.Animation([block], timeline)
        # anim.controls()
        anim.timeline_slider()
        # anim.save("levy2heighway", writer='ffmpeg', fps=timeline.fps)
        plt.show()


class _AnimateOverIter(DragonFractal):
    """Experimental Class
    TODO: Finish and document
    """
    # limits = -0.6, 1.6, -1.06, 0.308
    def reset(self):
        self.S = self._S0
        self._plot_list = [self._S0]

    def iterate(self, i: int, t: float) -> None:
        self._plot_list.clear()
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S, [t] * len(self.S))))
            self.S = S
        self._plot_list.append(S)

    def animate(self):
        x = []
        y = []
        time = np.linspace(0, 1, 50)
        for t in time:
            self.iterate(15, t)
            x.append(np.real(self.segment(self.S)))
            y.append(np.imag(self.segment(self.S)))
            self.reset()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Test")
        ax.set_aspect("equal")
        ax.set_ylim([-0.35, 1.75])
        ax.set_xlim([-0.76, 2.5])
        timeline = amp.Timeline(time, fps=10)
        block = amp.blocks.Line(x, y)
        anim = amp.Animation([block], timeline)
        anim.controls()
        anim.save_gif(f"{type(self).__name__}_15")
        plt.show()


# GLOBALS
S0i = [0, 1]

# Vars used in golden dragon
PHI = (1 + math.sqrt(5)) * 0.5
R = (1 / PHI) ** (1 / PHI)
A = math.acos((1 + R ** 2 - R ** 4) * 0.5 / R)
B = math.acos((1 - R ** 2 + R ** 4) * 0.5 / R ** 2)

# Vars Used in terdragon
LAMBDA = 0.5 - 1j * 0.5 / math.sqrt(3)
LAMBDACONJ = 0.5 + 1j * 0.5 / math.sqrt(3)

# Vars used in twindragon
S0_TWIN = [0, 1, 1 - 1j]

# Vars used in Pentigree
P_r = (3 - math.sqrt(5)) * 0.5
P_a1 = cmath.rect(1, 0.2 * math.pi)
P_a2 = cmath.rect(1, 0.6 * math.pi)
P_na1 = cmath.rect(1, -0.2 * math.pi)
P_na2 = cmath.rect(1, -0.6 * math.pi)
P_angle = [P_a1, P_a2, P_na1, P_na2, P_na1, P_a1]
P_offset = np.cumsum([0, *P_angle[:-1]])


def _make_func(DRPR, vector, offset):
    return lambda z: DRPR * (vector * z + offset)


# Vars used in pentadendrite
Dr = math.sqrt((6 - math.sqrt(5)) / 31)
RA = 0.20627323391771757578747269015392
PA = 1 * math.cos(RA) + 1j * math.sin(RA)
SA = math.pi * 0.4
BA = cmath.rect(1, RA + SA)
CA = cmath.rect(1, RA - SA)
DA = cmath.rect(1, RA - 2 * SA)
STAR = np.exp(1j * np.arange(0, 361, 72) * math.pi / 180)
PENTAGON = np.cumsum(STAR)
VVectors = [PA, BA, PA, DA, CA, PA]
VOffset = np.cumsum([0, *VVectors[:-1]])

# Vars used in Flowsnake
F_A = math.asin(math.sqrt(3) / math.sqrt(7) * 0.5)
F_A1 = F_A - 2 * math.pi / 3
F_r = math.sqrt(7) ** -1
F_angles = [F_A1, F_A, F_A, F_A1, F_A, F_A + 2 * math.pi / 3, F_A]
F_vectors = [cmath.rect(1, PHI) for PHI in F_angles]

# trans_matrix = [0]
IFS_function = {}
# Plain Ole' Dragon Curve
IFS_function["flowsnake"] = [
    lambda z: F_r * (z * F_vectors[0] - F_vectors[0]),
    lambda z: F_r * (z * F_vectors[1] - F_vectors[0]),
    lambda z: F_r * (z * F_vectors[2] + F_vectors[1] - F_vectors[0]),
    lambda z: F_r * (z * F_vectors[3] + 2 * F_vectors[2] - F_vectors[0]),
    lambda z: F_r * (z * F_vectors[4] + F_vectors[3] + F_vectors[2] - F_vectors[0]),
    lambda z: F_r
    * (z * F_vectors[5] - F_vectors[5] + F_vectors[3] + F_vectors[2] - F_vectors[0]),
    lambda z: F_r
    * (z * F_vectors[6] - F_vectors[5] + F_vectors[3] + F_vectors[2] - F_vectors[0]),
]

IFS_function["dragon"] = [
    lambda z: 0.5 * (1 + 1j) * z,
    lambda z: 1 - 0.5 * (1 - 1j) * z,
]

IFS_function["twin_dragon"] = [
    lambda z: 0.5 * (1 + 1j) * z,
    lambda z: 1 - 0.5 * (1 + 1j) * z,
]

IFS_function["golden_dragon"] = [
    lambda z: R * z * cmath.exp(A * 1j),
    lambda z: R ** 2 * z * cmath.exp((math.pi - B) * 1j) + 1,
]
# z2 dragon curve
IFS_function["z2_dragon"] = [
    lambda z: 0.5 * (1 + 1j) * z,
    lambda z: -(1 - 0.5 * (1 - 1j) * z),
    lambda z: 1 - 0.5 * (1 - 1j) * z,
    lambda z: -(0.5 * (1 + 1j) * z),
]

IFS_function["levy_c"] = [
    lambda z: 0.5 * (1 - 1j) * z,
    lambda z: 1 + 0.5 * (1 + 1j) * (z - 1),
]

IFS_function["z2_levy"] = [
    lambda z: 0.5 * (1 - 1j) * z,
    lambda z: -(1 + 0.5 * (1 + 1j) * (z - 1)),
    lambda z: 1 + 0.5 * (1 + 1j) * (z - 1),
    lambda z: -(0.5 * (1 - 1j) * z),
]

IFS_function["terdragon"] = [
    lambda z: LAMBDA * z,
    lambda z: 1j / math.sqrt(3) * z + LAMBDA,
    lambda z: LAMBDA * z + LAMBDACONJ,
]

IFS_function["z2_golden_dragon"] = [
    lambda z: R * z * cmath.exp(A * 1j),
    lambda z: R * z * cmath.exp((A - math.pi) * 1j),
    lambda z: R ** 2 * z * cmath.exp((math.pi - B) * 1j) + 1,
    lambda z: R ** 2 * z * cmath.exp(-B * 1j) - 1,
]

IFS_function["pentigree"] = [
    _make_func(P_r, vector, offset) for vector, offset in zip(P_angle, P_offset)
]

IFS_function["pentadendrite"] = [
    _make_func(Dr, vector, offset) for vector, offset in zip(VVectors, VOffset)
]

# Vars used in koch snowflake
K0 = 0 + 1j
Ka = 0.5 + 0.5 * 1j * math.sqrt(3)
Kna = 0.5 - 0.5 * 1j * math.sqrt(3)
KR = 1 / 3
K_vectors = [1, Ka, Kna, 1]

IFS_function["koch_flake"] = [
    lambda z: KR * (z),
    lambda z: KR * (Ka * z + 1),
    lambda z: KR * (Kna * z + 1 + Ka),
    lambda z: KR * (z + 2),
]

IFS_function["kochawave"] = [
    lambda z: KR * (z),
    lambda z: KR + 1/math.sqrt(3) * cmath.exp(1j*cmath.pi/6)*z,
    lambda z: KR + 1/math.sqrt(3) * cmath.exp(1j*cmath.pi/6) + KR * cmath.exp(-1j*2/3*cmath.pi)*z,
    lambda z: KR * (z + 2)]

DRPR = 1/(2 + 1/PHI)

IFS_function["durer_pentagon"] = [
    lambda z: DRPR*z,
    lambda z: DRPR*(1 + z*cmath.exp(2j*cmath.pi/5)),
    lambda z: DRPR*(1 + cmath.exp(2j*cmath.pi/5) + cmath.exp(-2j*cmath.pi/5)*z),
    lambda z: DRPR*(1 + 1/PHI + z)]

R1 = math.sqrt(1/PHI**2 + 1 - 2/PHI*math.cos(3*math.pi/5))
theta1 = math.asin(math.sin(3*math.pi/5)/R1)


class HeighwayDragon(DragonFractal):
    """[Heighway Dragon](https://larryriddle.agnesscott.org/ifs/heighway/heighway.htm)"""
    limits = (-0.407, 1.24, -0.382, 0.714)

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["dragon"])


class TwinDragon(DragonFractal):
    """[Twin Dragon](https://larryriddle.agnesscott.org/ifs/heighway/twindragon.htm)"""

    limits = (-0.4, 1.4, -0.75, 0.75)

    def __init__(self):
        super().__init__(S0_TWIN, IFS_function["twin_dragon"])


class GoldenDragon(DragonFractal):
    """[Golden Dragon](https://larryriddle.agnesscott.org/ifs/heighway/goldenDragon.htm)"""

    limits = (-0.317, 1.16, -0.243, 0.616)

    def __init__(self):
        super().__init__(S0i, IFS_function["golden_dragon"])


class Terdragon(Fractal):
    """[Terdragon](https://larryriddle.agnesscott.org/ifs/heighway/terdragon.htm)"""

    limits = (-0.12, 1.12, -0.357, 0.357)

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["terdragon"])


class FudgeFlake(Terdragon):
    """[Fudgeflake](https://larryriddle.agnesscott.org/ifs/heighway/fudgeflake.htm)"""

    limits = -0.55, 1.6, -0.4, 1.04

    def tile(self):
        self.translate(0, math.pi / 3)
        self.translate(1, 2 * math.pi / 3)


class LevyC(Fractal):
    """[Levy C Curve](https://larryriddle.agnesscott.org/ifs/levy/levy.htm)"""

    limits = -0.6, 1.6, -1.06, 0.308

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["levy_c"])


class LevyTapestryOutside(LevyC):
    """[Levy Tapestry](https://larryriddle.agnesscott.org/ifs/levy/tapestryOutside.htm)"""

    limits = -1.1, 2.1, -1.08, 1.08

    def tile(self):
        self.translate(1, math.pi)


class LevyTapestryInside(LevyC):
    """[Levy Tapestry](https://larryriddle.agnesscott.org/ifs/levy/tapestryInside.htm)"""

    limits = -1.2, 2.2, -1.6, 0.6

    def tile(self):
        translations = [(-1j, math.pi * 0.5), (1, -math.pi * 0.5), (1 - 1j, math.pi)]
        for off, theta in translations:
            self.translate(off, theta)


class KochFlake(Fractal):
    """
    [Koch Flake](https://larryriddle.agnesscott.org/ifs/kcurve/kcurve.htm)

    Note: this is constructed as a koch curve, then tiled.
    """

    limits = -0.5, 1.5, -0.924, 0.346

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

    def tile(self):
        translations = [
            (cmath.rect(1, math.pi / 3), -2 * math.pi / 3),
            (1, 2 * math.pi / 3),
        ]
        for off, theta in translations:
            self.translate(off, theta)


class Pentadendrite(Fractal):
    """[Pentadendrite](https://larryriddle.agnesscott.org/ifs/pentaden/penta.htm)"""

    limits = 0.85, 1.85, -0.152, 1.622

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["pentadendrite"])

    def tile(self):
        translations = zip(PENTAGON[:4], np.arange(72, 361, 72) * math.pi / 180)
        for offset, angle in translations:
            self.translate(offset, angle)


class Pentigree(Fractal):
    """[Pentigree](https://larryriddle.agnesscott.org/ifs/pentaden/pentigree.htm)"""
    limits = -0.4, 1.3, -0.312, 0.8

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
    limits = [-1, 2, -0.4, 1]

    def __init__(self):
        super().__init__(S0i, IFS_function["flowsnake"])


class GoldenFlake(BinaryTree):
    """[GoldenFlake](https://larryriddle.agnesscott.org/ifs/pentagon/Durer.htm)"""
    limits = -1.64, 1.64, -1.09, 1.09

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

if __name__ == "__main__":
    dragon = LevyC()
    dragon.iterate()
    dragon.plot(autoscale=True)
