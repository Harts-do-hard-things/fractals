# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:27:32 2021

@author: cub525
"""

import numpy as np
import matplotlib.pyplot as plt

class FunctionSystem:
    @staticmethod
    def apply(data: np.ndarray, trans_matrix: np.ndarray):
        ndata = np.add(data.dot(trans_matrix[:2, :2].T), trans_matrix[:, 2])
        return ndata

    def plot(self, func = lambda S: S):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        points = func(self.S)
        ax.set_aspect("equal")
        ax.plot(points[:, 0], points[:, 1], linestyle='', marker=',')
        plt.show()

class DeterministicSystem(FunctionSystem):
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
        len_ = len(self.S0)
        s = points.size // (2 * len_)
        npoints = np.append(
            points.reshape(-1, 2 * len_), [[np.nan, np.nan]] * s, 1
        ).reshape(-1, 2)
        return npoints


class IFSystemRand(FunctionSystem):
    def __init__(self, *args):
        self.fs_to_arrays()
        self.S = []

    def calculate_prob(self):
        det_list = [abs(np.linalg.det(a[:2,:2]))
                    if abs(np.linalg.det(a[:2,:2])) != 0 else .003
                    for a in self.trans_list]
        normalize = [a/sum(det_list) for a in det_list]
        return normalize

    def apply(self, point):
            index = np.random.choice(len(self.prob_list), p=self.prob_list)
            final_point = super().apply(point, self.trans_list[index])
            return final_point, index

    def iterate(self, n):
        point = np.array([0,0])
        colors = []
        for _ in range(50):
            point, _ = self.apply(point)

        for _ in range(n):
            point, index = self.apply(point)
            colors.append('C' + str(index))
            self.S.append(point)


    def fs_to_arrays(self):
        i = (len(self.eq[0]) - 1) // 3
        self.trans_list = [
            np.append(e[: 2 * i].reshape((2, -1)), e[-1 - i : -1].reshape(-1, 1), 1)
            for e in np.array(self.eq)
        ]
        self.prob_list = self.calculate_prob()

    def plot(self):
        super().plot(np.array)


