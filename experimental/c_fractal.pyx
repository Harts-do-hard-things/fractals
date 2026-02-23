# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:21:21 2021

@author: Emmett
"""

# from timeit import timeit
# import fractal


cdef class Fractal:
    cdef:
        complex S0[2]
        list S

    def __init__(self, S0):
        self.S0 = S0
        self.S = S0

    @staticmethod
    cdef complex f1(complex z):
        return 0.5 * (1 - 1j) * z

    @staticmethod
    cdef complex f2(complex z):
        return 1 + 0.5 * (1 + 1j) * (z - 1),

    # func_list = [&f1, &f2]

    def iterate(self, int i):
        for _ in range(i):
            S = []
            for func in self.func_list:
                S.extend(list(map(func, self.S)))
            self.S = S

    def reset(self):
        self.S = self.S0

    def test(self, int i):
        self.iterate(i)
        self.reset()
