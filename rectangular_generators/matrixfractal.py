# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:27:32 2021

@author: cub525
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_scatter_density


class FunctionSystem:
    @staticmethod
    def apply(data: np.ndarray, trans_matrix: np.ndarray):
        ndata = np.add(data.dot(trans_matrix[:2, :2].T), trans_matrix[:, 2])
        return ndata

    def plot(self, func=lambda S: S):
        fig = plt.figure()
        fig.patch.set_facecolor("#440154")
        ax = fig.add_subplot(111, projection="scatter_density")
        ax.axis('off')
        points = func(self.S)
        ax.set_aspect("equal")
        ax.scatter_density(points[:, 0], points[:, 1])
        # ax.plot(points[:, 0], points[:, 1], linestyle='', marker=',')
        plt.show()

    def __repr__(self):
        return f"{type(self).__name__}()"


class IFSystemDet(FunctionSystem):
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

    def segment(self, points: list) -> list:
        # len_ = len(self.S0)
        # s = points.size // (2 * len_)
        # npoints = np.append(
        #     points.reshape(-1, 2 * len_), [[np.nan, np.nan]] * s, 1
        # ).reshape(-1, 2)
        # return npoints
        return points


class IFSystemRand(FunctionSystem):
    def __init__(self, run_prob=True):
        self.fs_to_arrays(run_prob)
        self.S = []

    def calculate_prob(self):
        det_list = [abs(np.linalg.det(a[:2, :2]))
                    if abs(np.linalg.det(a[:2, :2])) != 0 else .003
                    for a in self.trans_list]
        normalize = [a / sum(det_list) for a in det_list]
        return normalize

    def apply(self, point):
        index = np.random.choice(len(self.prob_list), p=self.prob_list)
        final_point = super().apply(point, self.trans_list[index])
        return final_point, index

    def iterate(self, n):
        point = self.eq[0][4:6]
        colors = []
        # for _ in range(50):
        #     point, _ = self.apply(point)

        for _ in range(n):
            point, index = self.apply(point)
            colors.append('C' + str(index))
            self.S.append(point)

    def fs_to_arrays(self, run_prob):
        i = (len(self.eq[0]) - 1) // 3
        self.trans_list = [
            np.append(e[: 2 * i].reshape((2, -1)), e[-1 - i: -1]
                      .reshape(-1, 1), 1)
            for e in np.array(self.eq)
        ]
        if run_prob:
            self.prob_list = self.calculate_prob()
        else:
            self.prob_list = self.eq[:, -1]

    def plot(self):
        super().plot(np.array)
