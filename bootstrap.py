# -*- coding: utf-8 -*-
'''
This module implements bootstrapping of grid score parameters.
'''
from numpy import *
import numpy as np
try:
    from pylab import *
except:
    pass
import rate_map
import gridscores

import sys


def bootstrap_unit(unit, dur, sigma, func=gridscores.bestfit_gscore, return_rm=False):
    cycles = divide_in_cycles(unit['trajectory'])
    rm = shuffled_ratemap(cycles, dur, sigma)
    acmap = rate_map.xcorr2(rm, rm)
    if acmap.std() == 0:
        return np.nan, acmap, [np.nan, np.nan, np.nan, np.nan, np.nan]
    try:
        gscore_s, model, _, _, _ = func(acmap)
    except (ValueError,) as e:
        print 'ValueError', e
        return np.nan, acmap, [np.nan, np.nan, np.nan, np.nan, np.nan]
    except (Exception,) as e:
        print 'Exception occured in bootstrap:bootstrap_unit'
        import cPickle
        cPickle.dump(acmap, open('err.pickle', 'w'))
        raise e
    if return_rm:
        return gscore_s, acmap, rm
    return gscore_s, acmap, model


def gscore_unit(unit, sigma, func=gridscores.bestfit_gscore):
    rm = rate_map.gauss_rmap(
        *rate_map.rates(*rate_map.rates_from_unit(unit)), sigma=sigma)
    acmapo = rate_map.xcorr2(rm, rm)
    try:
        gscore_o, model, _, _, _ = func(acmapo)
    except (Exception,) as e:
        print 'Exception occured in bootstrap:gscore_unit'
        import cPickle
        import traceback
        s = traceback.format_exc()
        cPickle.dump((s, acmapo), open('err.pickle', 'w'))
        raise e

    return gscore_o, acmapo, model


def divide_in_cycles(trajectory):
    low = 0
    cycles = []
    for k in range(len(trajectory['x'][0]) - 1):
        ccur, cnext = trajectory[0, k][
            'counter'], trajectory[0, k + 1]['counter']
        ccur = ccur[~isnan(ccur)]
        if ccur[-1] > cnext[0]:
            cycles.append((low, k))
            low = k
    cycles = [trajectory[0, low:high] for low, high in cycles]
    return cycles


def shuffled_ratemap(cycles, dur, sigma):
    '''
    Compute a shuffled ratemap from cycles of a recording day.
    '''
    scycles = [shuffle_cycle(concat_unit(c), dur) for c in cycles]
    scycles = concat_cycles(scycles)
    g = rate_map.gauss_rmap(*rate_map.rates(
        scycles['x'], scycles['y'], scycles['sx'], scycles['sy']), sigma=sigma)
    return g


def shuffle_cycle(cycle, dur):
    '''
    Shuffle blocks of spikes for one cycle.

    cycle is a dictionary that represents a trajectory with
    spikes. x,y locations of spikes are encoded in cycle['sx'] and
    cycle['sy']. x,y locations of trajectory are encoded in
    cycle['x'], cycle['y'].
    '''
    # Get a spike vector
    spikes = (in1d(cycle['x'], cycle['sx']) & in1d(cycle['y'], cycle['sy']))
    assert sum(spikes) == len(cycle['sx'])
    assert dur < len(cycle['x'])
    # Shuffle segments of dur length.
    num_segments = len(cycle['x']) / dur
    indices = arange(num_segments)
    np.random.shuffle(indices)
    shuffled_spikes = spikes.copy()
    for i, start in enumerate(indices):
        start_index = start * dur
        shuffled_spikes[i * dur:i * dur +
                        dur] = spikes[start_index:start_index + dur]
    # Take care of the part at the end
    shuffled_spikes = roll(shuffled_spikes, np.random.randint(dur))

    sx = cycle['x'][shuffled_spikes]
    sy = cycle['y'][shuffled_spikes]
    cycle = cycle.copy()
    cycle['sx'] = sx
    cycle['sy'] = sy
    return cycle


def concat_unit(trajectory, var=['x', 'y', 'sx', 'sy']):
    N = max(trajectory[var[0]].shape)
    res = dict()
    for v in var:
        Y = hstack([trajectory[v][i].flatten()
                    for i in range(N)])
        res[v] = Y
    return res


def concat_cycles(cycles, var=['x', 'y', 'sx', 'sy']):
    N = len(cycles)
    res = dict()
    for v in var:
        Y = hstack([cycles[i][v].flatten()
                    for i in range(N)])
        res[v] = Y
    return res


def plot_cycles(cycles, idx=None, plot_cycle=1, sigma=2.5):
    pcycles = cycles
    if len(cycles) > 1:
        if idx is None:
            idx = [slice(None)]
        cycles = [concat_cycles(cycles[i]) for i in idx]

    for i, c in enumerate(cycles):

        rm = rate_map.gauss_rmap(
            *rate_map.rates(*rate_map.rates_from_trj(c)))
        subplot(1, len(cycles), i + 1)
        r = concat_unit(pcycles[plot_cycle])
        x, y = r['x'], r['y']
        sx, sy = r['sx'], r['sy']

        plot(x, y, 'r-')
        left, right = -11 * 24 + 400, 11 * 24 + 400
        top, bottom = -11 * 24 + 300, 11 * 24 + 300
        imshow(rm, extent=[left, right, top, bottom], interpolation='none')
        plot(sx, sy, 'k.')
        xticks([])
        yticks([])
