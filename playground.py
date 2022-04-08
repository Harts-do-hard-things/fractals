# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:04:34 2021

@author: Emmett
"""

from pprint import pprint
import numpy as np
import ifsread as ifs
import datashader as ds
import pandas as pd
import colorcet as cc




if __name__ == "__main__":
    path = r"ifs_data\fractint.ifs"
    my_ifslist = ifs.interpret_file(path)
    pprint(my_ifslist.__dict__)
    h = my_ifslist.IFS_dragon(run_prob = False)
    h.iterate(100_000)
    S = np.array(h.S)
    df = pd.DataFrame({"x" : S[:, 0], "y": S[:, 1]})
    agg = ds.Canvas().points(df, "x", "y")
    out = ds.tf.set_background(ds.tf.shade(agg, cmap=cc.fire), "black")
    pass
# cc.palette['kbc']