# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:04:34 2021

@author: Emmett
"""


from ifsread import ifs
# from timeit import timeit

if __name__ == "__main__":
    print("Available rectangular generators:")
    for fractal in ifs:
        print(fractal)
    h = ifs.IFS_Heighway_Dragon(run_prob=False)
    h.iterate(100_000)
    # h.plot()
    image = h.make_image()
    # image.save("emmett.png")