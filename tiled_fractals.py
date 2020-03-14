# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:21:45 2019

@author: EOL
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import fractal as fr


def calc_angles(c):
    alpha = math.pi*.25
    beta = math.pi*.25
    gamma = np.pi-alpha-beta
    return alpha, beta, gamma

def calc_point(alpha, beta, c):
    x = (c*np.tan(beta) )/( np.tan(alpha)+np.tan(beta) )
    y = x * np.tan(alpha)
    return (x,y)

def get_triangle(c):
    alpha, beta, _ = calc_angles(c)
    x,y = calc_point(alpha, beta, c)
    return [(0,0), (c,0), (x,y)]

c = 2

fig, ax = plt.subplots()
ax.set_aspect("equal")

dreieck = plt.Polygon(get_triangle(c))
ax.add_patch(dreieck)
ax.relim()
ax.autoscale_view()
plt.show()