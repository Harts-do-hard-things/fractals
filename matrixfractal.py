# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:27:32 2021

@author: cub525
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_scatter_density

from typing import List
from tabulate import tabulate
from pprint import pprint
import ifslex
import time


path = r"fractint.ifs"

LEVY_1 = np.array([[0.5, -0.5, 0], [0.5, 0.5, 0]])
LEVY_2 = np.array([[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]])
S0I = [[0, 0], [1, 0]]


class FunctionSystem:
    def __init__(self, S0: List[float], trans_list: np.ndarray):
        self.S0 = S0
        self.S = np.array(S0)
        self.trans_list = trans_list

    def iterate(self, i: int):
        for _ in range(i):
            if self.S.size > 6e6:
                print("3 Million Points is Enough")
                return
            S = []
            for t in self.trans_list:
                S.extend(self.apply(self.S, t))
            self.S = np.array(S)

    @staticmethod
    def apply(data: np.ndarray, trans_matrix: np.ndarray):
        ndata = np.add(data.dot(trans_matrix[:2, :2].T), trans_matrix[:, 2])
        return ndata

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        points = self.segment(self.S)
        ax.set_aspect("equal")
        ax.plot(points[:, 0], points[:, 1], linestyle='', marker=',')
        plt.show()

    def segment(self, points: list) -> list:
        len_ = len(self.S0)
        s = points.size // (2 * len_)
        npoints = np.append(
            points.reshape(-1, 2 * len_), [[np.nan, np.nan]] * s, 1
        ).reshape(-1, 2)

        return npoints


class IFSystem(FunctionSystem):
    def __init__(self):
        self.S0 = S0I
        self.S = np.array(S0I)
        self.fs_to_arrays()

    def calculate_prob(self):
        det_list = [abs(np.linalg.det(a[:2,:2]))
                    if abs(np.linalg.det(a[:2,:2])) != 0 else .003
                    for a in self.trans_list]
        normalize = [a/sum(det_list) for a in det_list]
        print(normalize)
        return normalize

    def fs_to_arrays(self):
        i = (len(self.eq[0]) - 1) // 3
        # print(self.eq[0][-1-i:-1])
        self.trans_list = [
            np.append(e[: 2 * i].reshape((2, -1)), e[-1 - i : -1].reshape(-1, 1), 1)
            for e in np.array(self.eq)
        ]
        # self.prob_list = self.eq[:,-1].flatten()
        self.prob_list = self.calculate_prob()
        # pprint(self.trans_list)

    def random_iterate(self):
        point = np.array([0,0])
        self.S1 = []
        colors = []

        def iterate(self, initial_point):
            index = np.random.choice(len(self.prob_list), p=self.prob_list)
            final_point = self.apply(initial_point, self.trans_list[index])
            return final_point, index

        for _ in range(50):
            point, _ = iterate(self, point)

        for _ in range(2000000):
            point, index = iterate(self, point)
            colors.append('C' + str(index))
            self.S1.append(point)

        S1 = np.array(self.S1)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='scatter_density')
        ax.set_aspect("equal")
        self.plot_handle = ax.scatter_density(
            S1[:, 0], S1[:, 1])#, c = colors, marker='o', s=(72./fig.dpi)**2)
        fig.show()



if __name__ == "__main__":
    my_ifslist = ifslex.interpret_file(path)
    pprint(my_ifslist.__dict__)
    h = my_ifslist.IFS_fern()
    h.random_iterate()
    # h.iterate(15)
    # h.plot()
    # tabulate(my_ifslist.__dict__, headers='keys')
    # h = my_ifslist.4x4Cross([
    #     [0,0],
    #     [1,0], [1,1], [0,1], [0,0]])
    # h.iterate(8)
    # # h.segment(h.S)
    # h.plot()
    # l = IFSystem([[0,0],[1,0]], [LEVY_1, LEVY_2])
    # l.iterate(10)
    pass
