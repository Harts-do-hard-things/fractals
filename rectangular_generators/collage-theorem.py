# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:47:46 2021

@author: Emmett
"""

import numpy as np

from scipy.special import expit
from matrixfractal import FunctionSystemRandom
import matplotlib.pyplot as plt
import json
# def training_set():
#     tr = []
#     while len(tr) < 4:
#         if v_trans(t := np.random.uniform(-1,1,[2,3])):
#             tr.append(t)
#     tr = np.array(tr)
#     print(tr)

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
        self.iterate(2**16)
        self.im_data = self.get_data()

    def get_data(self):
        points = np.array(self.S)
        fig = plt.figure(figsize=(2,2), dpi=128)
        ax = fig.add_subplot()
        ax.axis('off')
        ax.plot(points[:, 0], points[:, 1], linestyle='', marker=',', color='k')
        plt.show()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(-1, 3)[:,0]/255
        plt.close()
        return data

    def jsonize(self):
        return {"yhat": list(self.trans_list.reshape((-1))), "input": list(self.im_data)}

def create_training_data(n: int, file = 'NNData/training_data.json'):
    test = input("Will overwrite previous data, continue? ([y]/n) ")
    if test in 'yY ':
        with open(file, 'w') as f:
            for i in range(n):
                json.dump(RandomIFSystem().jsonize(),f)
                f.write("\n")
    else:
        print("Canceled")
        return
            

class Layer:
    def __init__(self, size: int, to_size: int, data = False):
        self.data = data if data else np.empty((size,))
        self.weights = np.random.random((to_size, size))
        self.bias = np.random.random((to_size))
        self.out = np.empty((to_size))

    def calc_data(self) -> np.ndarray:
        self.out = expit(self.weights.dot(self.data) + self.bias)
        return self.out

    def back_prop():
        pass

    def __repr__(self):
        return f"Layer({self.data.size}, {self.out.size})"

class CollageNetwork:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) 
         for i in range(len(layer_sizes) - 1)]

    def propogate(self) -> np.ndarray:
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].data = self.layers[i].calc_data()
        return self.layers[-1].out

    def train(self):
        with open("NNData/training_data1.json", "r") as f:
            for line in f:
                sample = json.load(line)
        
# print("Test")

# t = CollageNetwork([2**16, 32, 32, 24])
# t.train()
# print(a := t.propogate())

