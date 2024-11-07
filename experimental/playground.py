# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:04:34 2021

@author: Emmett
"""

from pprint import pprint
import ifsread as ifs
from timeit import timeit

if __name__ == "__main__":
    path = r"ifs_data/fractint.ifs"
    my_ifslist = ifs.interpret_file(path)
    pprint(my_ifslist.__dict__)
    h = my_ifslist.IFS_crystal(run_prob=False)
    h.iterate(50000)
    h.plot()

    pass