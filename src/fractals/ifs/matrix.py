# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:27:32 2021

@author: cub525
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

COLORS = [(np.array(
    colors.to_rgb(colors.TABLEAU_COLORS[i])) * 255 // 1).astype("uint8")
    for i in ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']]


class FunctionSystem:
    @staticmethod
    def apply(data: np.ndarray, trans_matrix: np.ndarray):
        ndata = np.add(data.dot(trans_matrix[:2, :2].T), trans_matrix[:, 2])
        return ndata

    def plot(self, func=lambda S: S):
        fig = plt.figure()
        # fig.patch.set_facecolor("#000000")
        ax = fig.add_subplot(111)
        ax.axis('off')
        points = func(self.S)
        ax.set_aspect("equal")
        ax.plot(
            points[:, 0], points[:, 1], linestyle='',
            marker=',', color='tab:red')
        plt.show()

    def __repr__(self):
        return f"{type(self).__name__}()"


class DeterministicFunctionSystem(FunctionSystem):
    def __init__(self, S0, trans_list):
        self.S0 = S0
        self.S = np.array(S0)
        self.trans_list = trans_list
        self.iters = 0

    def iterate(self, i: int):
        for _ in range(i):
            self.iters += 1
            S = []
            for t in self.trans_list:
                S.extend(self.apply(self.S, t))
            self.S = np.array(S)

    def plot(self):
        super().plot(self.segment)


class FunctionSystemRandom(FunctionSystem):
    def __init__(self, size=1_000_000, run_prob=False):
        self.ifs_to_arrays(run_prob)
        self.S = []
        self.trans_used = []
        self.limits = self.calculate_limits()

    def reset(self):
        self.S.clear()
        self.trans_used.clear()

    def calculate_limits(self):
        self.iterate(10_000)
        mins = np.min(self.S, axis=0)
        maxs = np.max(self.S, axis=0)
        expected_diff = max(maxs - mins) * 1.05
        diff = maxs - mins

        maxmin = np.array(
            (mins - (expected_diff - diff) / 2,
             maxs + (expected_diff - diff) / 2))
        self.reset()
        return maxmin.flatten('F')

    def calculate_probabilities(self):
        det_list = [abs(np.linalg.det(a[:2, :2]))
                    if abs(np.linalg.det(a[:2, :2])) != 0 else .003
                    for a in self.trans_list]
        normalize = [a / sum(det_list) for a in det_list]
        return normalize

    def get_xlim(self):
        return self.limits[:2]

    def get_ylim(self):
        return self.limits[2:]

    xlim = property(get_xlim)
    ylim = property(get_ylim)

    def apply(self, point):
        index = np.random.choice(len(self.prob_list), p=self.prob_list)
        final_point = super().apply(point, self.trans_list[index])
        return final_point, index

    def iterate(self, n):
        point = self.eq[0][4:6]
        for _ in range(1_000):
            point, _ = self.apply(point)

        for _ in range(n):
            point, index = self.apply(point)
            self.S.append(point)
            self.trans_used.append(index)

    def make_image(self, resolution=(1080, 1080)):
        pixels = np.uint8(
            [[(0, 0, 0, 0) for _ in range(resolution[0])]
             for _ in range(resolution[1])])
        for point, index in zip(self.S, self.trans_used):
            x, y = point
            res = min(resolution)
            pixelx = int((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])*res)
            pixely = int((self.ylim[1] - y) / (self.ylim[1] - self.ylim[0])*res)
            pixels[pixely, pixelx] = np.append(COLORS[index], 255)
            # pixels[pixely, pixelx] = np.append(COLORS[0], 255)
        return Image.fromarray(pixels, "RGBA")

    def ifs_to_arrays(self, run_prob):
        if len(self.eq[0]) % 2 == 1:
            i = (len(self.eq[0]) - 1) // 3
        else:
            i = len(self.eq[0]) // 3
            run_prob = True
        self.trans_list = [
            np.append(e[: 2 * i].reshape((2, -1)), e[-1 - i: -1]
                      .reshape(-1, 1), 1)
            for e in np.array(self.eq)
        ]
        self.prob_list = self.calculate_probabilities() if run_prob else self.eq[:, -1]

    def plot(self):
        super().plot(np.array)
