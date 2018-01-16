# -*- coding: utf-8 -*-
import glob

from os.path import join

import numpy as np

from numpy import *

try:
    from pylab import *
except:
    pass

from scipy.io import loadmat

import ellipsemodels as em
import gness
import gscore
import tools


def compare(acmap, model, style='solid', offset=[44, 44], lw=0.25):
    '''
    Plot a comparison between model and auto correlation.
    '''
    gness.pprediction(*model)
    imshow(acmap, interpolation='nearest', aspect='auto')
    f1, f2 = em.model2scheitel(*model)
    em.plot_ellipse(f1, f2, offset=offset, lw=lw)
    xlim([0, acmap.shape[0]])
    ylim([0, acmap.shape[0]])


def combine_units(units, spikes_everywhere=False):
    trjs = [unit['trajectory'] for unit in units]
    ctrjs = hstack(trjs)
    if spikes_everywhere:
        ctrjs['sx'] = ctrjs['x']
        ctrjs['sy'] = ctrjs['y']
    return {'trajectory': ctrjs}


def plot_chist(model, style='solid'):
    extend, angle, scale, sigma, rotangle = model
    binr = sigma * 1.5
    theta, r = gscore.model2coord(*model)
    index = ((extend - binr) <= r) & (r < (extend + binr))
    contour(index, 1, linestyles=style)


@tools.Memoize
def get_cell(cell):
    for tag, data in get_data(filter=cell[0], cache=False):
        if tag == cell:
            return data
    raise RuntimeError('Couldn\'t find cell')


def list_data(filter=None, path=None):
    if path is None:
        path = get_path()
    files = glob.glob(join(path, '*.mat'))
    if filter is not None:
        files = [f for f in files if filter in f]
    for f in files:
        yield f


def get_datum(filename, cache=False):
    if cache:
        tag, unit = cache_mat_file(filename)
    else:
        a = loadmat(filename)['u']
        tag, unit = (a['datafile'][0][0][0], a[
                     'unit_name'][0][0][0]), simplify(a)
    return tag, unit


def get_data(filter=None, cache=False, path=None):
    '''
    Load all data into lists.
    '''
    if path is None:
        path = get_path()
    files = glob.glob(join(path, '*.mat'))
    if filter is not None:
        files = [f for f in files if filter in f]
    cnt = 0
    for f in files:
        if cache:
            tag, unit = cache_mat_file(f)
        else:
            a = loadmat(f)['u']
            tag, unit = (a['datafile'][0][0][0], a[
                         'unit_name'][0][0][0]), simplify(a)
        yield tag, unit
        cnt += 1
        if not cache:
            del a
    return


def simplify(a):
    return dict((k, array(a[k][0][0])) for k in a.dtype.fields.keys())


@tools.Memoize
def cache_mat_file(filename):
    a = loadmat(filename)['u']
    tag, unit = (a['datafile'][0][0][0], a['unit_name'][0][0][0]), simplify(a)
    return tag, unit


# Next two functions taken from David Wolever (stackoverflow.com)
# http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle
