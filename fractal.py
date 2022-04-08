# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:14:22 2016

@author: Harts
"""


# N.B. I used flatten and segment in the DragonFractal code
# Under the @staticmethod decorator (now works for any arbitrary starting line)
# All fractals requiring segmentation should use DragonFractal as parent


import cmath
import math

import matplotlib.pyplot as plt
import numpy as np
import animatplot as amp
import datashader as ds
import pandas as pd
import colorcet as cc

try:
    import gif
except ModuleNotFoundError:
    gif = None

from collections.abc import Callable
from typing import List


class Fractal:
    limits = False

    def __init__(self, S0: List[complex], func_list: List[Callable]):
        self.S0 = S0
        self.S = S0
        self.func_list = func_list
        self.plot_list = [S0]

    def iterate(self, i: int) -> None:
        self.plot_list.clear()
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
            self.S = S
        self.plot_list.append(S)

    # Rotate and translate (in that order) & create a copy
    def translate(self, offset: complex, angle: float) -> None:
        s_trans = [i * cmath.exp(angle * 1j) + offset for i in self.plot_list[0]]
        self.plot_list.append(s_trans)

    def tile(self):
        pass
        # Designed to be overridden, Not sure of the Syntax

    def plot(self, autoscale=True):
        self.tile()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_handle = [
            ax.plot(np.real(s), np.imag(s), color="tab:blue") for s in self.plot_list
        ]
        if self.limits and not autoscale:
            ax.set_xlim(self.limits[:2])
            ax.set_ylim(self.limits[2:])
        ax.set_aspect("equal")
        plt.show()

    # Not intended for Call except through save_gif method
    if gif:

        @gif.frame
        def gif_plot(self) -> None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if self.limits:
                ax.set_xlim(self.limits[:2])
                ax.set_ylim(self.limits[2:])
            else:
                ax.set_aspect("equal")
            for s in self.plot_list:
                ax.plot(np.real(s), np.imag(s), color="tab:blue")

        def save_gif(self, iterations: int, duration=1000) -> None:
            self.tile()
            frames = [self.gif_plot()]
            for _ in range(iterations - 1):
                self.iterate(1)
                self.tile()
                frame = self.gif_plot()
                frames.append(frame)
            gif.save(
                frames, "{0}_{1}.gif".format(type(self).__name__, iterations), duration
            )


class DragonFractal(Fractal):
    @staticmethod
    def flatten(iter_):
        return [item for sublist in iter_ for item in sublist]

    def segment(self, points: list) -> list:
        len_ = len(self.S0)
        # lines = [[[points[j].real, points[j].imag] for j in range(i,i+len_)]
        #          for i in range(len(points)-len_+1) if i % len_ == 0]
        # return lines
        lines = [
            [*points[i : i + len_], np.nan]
            for i in range(len(points) - len_ + 1)
            if i % len_ == 0
        ]
        s_ = self.flatten(lines)
        return s_

    def plot(self, autoscale=True):
        self.plot_list = [self.segment(s) for s in self.plot_list]
        super().plot(autoscale)

    if gif:

        def save_gif(self, iterations: int, duration=1000):
            self.tile()
            frames = [self.gif_plot()]
            for _ in range(iterations - 1):
                self.iterate(1)
                self.plot_list = [self.segment(s) for s in self.plot_list]
                self.tile()
                frame = self.gif_plot()
                frames.append(frame)
            gif.save(
                frames, "{0}_{1}.gif".format(type(self).__name__, iterations), duration
            )


class BinaryTree(DragonFractal):
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
                self.plot_list.append(S)
                self.iterations += 1
            self.S = S

    def translate(self, offset: complex, angle: float):
        for j in range(self.iterations + 1):
            # print(j)
            s_trans = [i * cmath.exp(angle * 1j) + offset for i in self.plot_list[j]]
            self.plot_list.append(s_trans)


class AnimFractal(Fractal):
    def create_block(self, X: list, Y: list) -> list:
        X.append(self.real(self.S))
        Y.append(self.imag(self.S))
        return X, Y

    def animate(self, i: int):
        x = [np.real(self.S0)]
        y = [np.imag(self.S0)]
        for _ in range(i):
            self.iterate(1)
            x, y = self.create_block(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Test")
        # ax.set_ylim(self.limits[2:])
        # ax.set_xlim(self.limits[:2])
        # ax.axis = self.limits
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


class AnimateOverIter(DragonFractal):
    # limits = -0.6, 1.6, -1.06, 0.308
    def reset(self):
        self.S = self.S0
        self.plot_list = [self.S0]

    def iterate(self, i: int, t: float) -> None:
        self.plot_list.clear()
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S, [t] * len(self.S))))
            self.S = S
        self.plot_list.append(S)

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
phi = (1 + math.sqrt(5)) * 0.5
r = (1 / phi) ** (1 / phi)
A = math.acos((1 + r ** 2 - r ** 4) * 0.5 / r)
B = math.acos((1 - r ** 2 + r ** 4) * 0.5 / r ** 2)

# Vars Used in terdragon
lamda = 0.5 - 1j * 0.5 / math.sqrt(3)
lamdaconj = 0.5 + 1j * 0.5 / math.sqrt(3)

# Vars used in twindragon
S0_twin = [0, 1, 1 - 1j]

# Vars used in Pentigree
P_r = (3 - math.sqrt(5)) * 0.5
P_a1 = cmath.rect(1, 0.2 * math.pi)
P_a2 = cmath.rect(1, 0.6 * math.pi)
P_na1 = cmath.rect(1, -0.2 * math.pi)
P_na2 = cmath.rect(1, -0.6 * math.pi)
P_angle = [P_a1, P_a2, P_na1, P_na2, P_na1, P_a1]
P_offset = np.cumsum([0, *P_angle[:-1]])


def make_func(scale, vector, offset):
    return lambda z: scale * (vector * z + offset)


# Vars used in pentadendrite
Dr = math.sqrt((6 - math.sqrt(5)) / 31)
RA = 0.20627323391771757578747269015392
PA = 1 * math.cos(RA) + 1j * math.sin(RA)
SA = math.pi * 0.4
BA = cmath.rect(1, RA + SA)
CA = cmath.rect(1, RA - SA)
DA = cmath.rect(1, RA - 2 * SA)
star = np.exp(1j * np.arange(0, 361, 72) * math.pi / 180)
pentagon = np.cumsum(star)
VVectors = [PA, BA, PA, DA, CA, PA]
VOffset = np.cumsum([0, *VVectors[:-1]])

# Vars used in Flowsnake
F_A = math.asin(math.sqrt(3) / math.sqrt(7) * 0.5)
F_A1 = F_A - 2 * math.pi / 3
F_r = math.sqrt(7) ** -1
F_angles = [F_A1, F_A, F_A, F_A1, F_A, F_A + 2 * math.pi / 3, F_A]
F_vectors = [cmath.rect(1, phi) for phi in F_angles]

# trans_matrix = [0]
IFS_function = dict()
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
    lambda z: r * z * cmath.exp(A * 1j),
    lambda z: r ** 2 * z * cmath.exp((math.pi - B) * 1j) + 1,
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
    lambda z: lamda * z,
    lambda z: 1j / math.sqrt(3) * z + lamda,
    lambda z: lamda * z + lamdaconj,
]

IFS_function["z2_golden_dragon"] = [
    lambda z: r * z * cmath.exp(A * 1j),
    lambda z: r * z * cmath.exp((A - math.pi) * 1j),
    lambda z: r ** 2 * z * cmath.exp((math.pi - B) * 1j) + 1,
    lambda z: r ** 2 * z * cmath.exp(-B * 1j) - 1,
]

IFS_function["pentigree"] = [
    make_func(P_r, vector, offset) for vector, offset in zip(P_angle, P_offset)
]

IFS_function["pentadendrite"] = [
    make_func(Dr, vector, offset) for vector, offset in zip(VVectors, VOffset)
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


class HeighwayDragon(DragonFractal):
    limits = (-0.407, 1.24, -0.382, 0.714)

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["dragon"])


class TwinDragon(DragonFractal):
    limits = (-0.4, 1.4, -0.75, 0.75)

    def __init__(self):
        super().__init__(S0_twin, IFS_function["twin_dragon"])


class GoldenDragon(DragonFractal):
    limits = (-0.317, 1.16, -0.243, 0.616)

    def __init__(self):
        super().__init__(S0i, IFS_function["golden_dragon"])


class Terdragon(Fractal):
    limits = (-0.12, 1.12, -0.357, 0.357)

    def __init__(self):
        super().__init__(S0i, func_list=IFS_function["terdragon"])


class FudgeFlake(Terdragon):
    limits = -0.55, 1.6, -0.4, 1.04

    def tile(self):
        self.translate(0, math.pi / 3)
        self.translate(1, 2 * math.pi / 3)


class LevyC(Fractal):
    limits = -0.6, 1.6, -1.06, 0.308

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["levy_c"])


class LevyTapestryOutside(LevyC):
    limits = -1.1, 2.1, -1.08, 1.08

    def tile(self):
        self.translate(1, math.pi)


class LevyTapestryInside(LevyC):
    limits = -1.2, 2.2, -1.6, 0.6

    def tile(self):
        translations = [(-1j, math.pi * 0.5), (1, -math.pi * 0.5), (1 - 1j, math.pi)]
        for off, theta in translations:
            self.translate(off, theta)


class KochFlake(Fractal):
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


class Pentadendrite(Fractal):
    limits = 0.85, 1.85, -0.152, 1.622

    def __init__(self):
        super().__init__(S0=[0, 1], func_list=IFS_function["pentadendrite"])

    def tile(self):
        translations = zip(pentagon[:4], np.arange(72, 361, 72) * math.pi / 180)
        for offset, angle in translations:
            self.translate(offset, angle)


class Pentigree(Fractal):
    limits = -0.4, 1.3, -0.312, 0.8

    def __init__(self):
        super().__init__(S0i, IFS_function["pentigree"])


class Z2Dragon(DragonFractal):
    def __init__(self):
        super().__init__([0, 1], IFS_function["z2_golden_dragon"])


class Z2Levy(DragonFractal):
    def __init__(self):
        super().__init__(S0i, IFS_function["z2_levy"])


class Flowsnake(DragonFractal):
    limits = [-1, 2, -0.4, 1]

    def __init__(self):
        super().__init__(S0i, IFS_function["flowsnake"])


class GoldenFlake(BinaryTree):
    limits = -1.64, 1.64, -1.09, 1.09

    def __init__(self):
        super().__init__(1 / phi, 0.8 * math.pi)

    def tile(self):
        for angle in np.linspace(0, 2 * math.pi, 6):
            self.translate(0, angle)

    def iterate(self, i):
        for _ in range(i):
            self.plot_list.clear()
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
                self.plot_list.append(S)
                self.iterations += 1
            self.S = S


class TestLevy(LevyC, AnimFractal):
    pass


if __name__ == "__main__":
    levy_c = Flowsnake()
    levy_c.iterate(8)
    df = pd.DataFrame({"real" :np.real(levy_c.S), "imag": np.imag(levy_c.S)})
    agg = ds.Canvas(plot_width = 1100, plot_height = 684, x_range = levy_c.limits[:2], y_range = levy_c.limits[2:]).points(df, "real", "imag")
    ds.tf.set_background(ds.tf.shade(agg, cmap=cc.palette['kbc']), "black")
    # dragon = HeighwayDragon()
    # dragon.save_gif(12)
    # dragon.iterate(20)
    # dragon.plot()
    # levy_c = LevyC()
    # levy_c.iterate(10)
    # pentadendrite = Pentadendrite()
    # pentadendrite.iterate(7)
    # pentadendrite.plot()
    # pentadendrite.save_gif(3)
    # twindragon = TwinDragon()
    # twindragon.iterate(2)
    # twindragon.plot()
    # golden_dragon = GoldenDragon()
    # golden_dragon.iterate(18)
    # golden_dragon.plot()
    # koch_snowflake = KochFlake()
    # koch_snowflake.iterate(3)
    # koch_snowflake.plot()
    # koch_snowflake.save_gif(7)
    # snake = Flowsnake()
    # snake.iterate(4)
    # snake.plot()
    # snake.save_gif(5)
    # pent = Pentigree()
    # pent.iterate(6)
    # pent.plot()
    # t_levy = TestLevy()
    # t_levy.animate(17)
    # tree.save_gif(7)
    # tree.iterate(10)
    # tree.plot()
    # test = AnimateOverIter(S0i, func_list=[
    #     lambda z,t: z*.5*(1+1j),
    #     lambda z,t: z*cmath.rect(math.sqrt(2)*.5, -math.pi*.25 + math.pi*t)+(.5 + .5*t)+(.5 - .5*t)*1j])
    # test.animate()
    pass
