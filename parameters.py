# -*- coding: utf-8 -*
'''
Set some global parameters such as rate map and autocorrelation size.
'''
import numpy as np


def get_ratemap_edges():
    ex = np.linspace(-11, 11, 45)
    ey = np.linspace(-11, 11, 45)
    assert sum(ex - ey) == 0
    return [ey, ex]


def get_acmap_shifts():
    ey, ex = get_ratemap_edges()
    dva_per_bin = ey[1] - ey[0]
    shifts = np.arange(-(len(ex) - 1), len(ex))
    return shifts, shifts * dva_per_bin
