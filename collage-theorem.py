# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:47:46 2021

@author: Emmett
"""

import numpy as np

from scipy.special import expit
from matrixfractal import IFSystemRand
import matplotlib.pyplot as plt
# data = np.array(Image.open("Test_Fractal.bmp").convert("L")).reshape([-1,1])




def training_set():
    tr = []
    while len(tr) < 4:
        if v_trans(t := np.random.uniform(-1,1,[2,3])):
            tr.append(t)
    tr = np.array(tr)
    print(tr)
        

def v_trans(t):
    return 1 > sum(t[:,0]) and 1 > sum(t[:,1]) and np.sum(t[:,:2]) < 1 + np.linalg.det(t[:2,:2])**2

class RandomIFSystem(IFSystemRand):
    def __init__(self):
        tr = []
        while len(tr) < 4:
            if v_trans(t := np.random.uniform(-1,1,[2,3])):
                tr.append(t)
        self.trans_list = np.array(tr)
        self.S = []
        self.prob_list = self.calculate_prob()
        self.iterate(65536)
        self.im_data = self.get_data()

    def get_data(self):
        points = np.array(self.S)
        fig = plt.figure(figsize=(2,2), dpi=128)
        ax = fig.add_subplot()
        ax.axis('off')
        ax.plot(points[:, 0], points[:, 1], linestyle='', marker=',', color='k')
        plt.show()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(256,256,3)[:,:,0]
        return data
        
        
t = RandomIFSystem()

# def train_dat(self):
#         point = np.array([0,0])
#         self.S1 = []
        
#         for _ in range(50):
#             point, _ = self.r_iterate(point)
        
#         for _ in range(65536):
#             point, _ = self.r_iterate(point)
#             self.S1.append(point)
#         S1 = np.array(self.S1)
#         fig = plt.figure(figsize=(1,1), dpi=256)
#         ax = fig.add_subplot(111)
#         ax.axis('off')
#         ax.plot(S1[:, 0], S1[:,1], color='k', marker=',', linestyle='')#, s=(72./fig.dpi)**2)
#         fig.savefig("test.png", format='png')
#         fig.show()