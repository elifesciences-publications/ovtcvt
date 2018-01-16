# -*- coding: utf-8 -*
'''
Estimate a 2D rate map from samples
'''
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import parameters


def compute_all_firing_rates():
    import utils
    firing_rates = {}
    for tag, unit in utils.get_data(filter='cvt'):
        X, Y, sX, sY = rates_from_unit(unit)
        r, ts, s = rates(X, Y, sX, sY)
        firing_rates[tag] = (s.sum() / ts.sum(), len(sX) /
                             (len(X) / 1000.), r.max())
    return firing_rates


def rates_from_unit(unit):
    # Convert data to pixels
    N = unit['trajectory']['x'].shape[1]
    Y = np.concatenate([unit['trajectory']['y'][0, i]
                        for i in range(N)]).flatten()
    X = np.concatenate([unit['trajectory']['x'][0, i]
                        for i in range(N)]).flatten()
    sY = np.concatenate([
        unit['trajectory']['sy'][0, i]
        for i in range(N)
        if sum(unit['trajectory']['sy'][0, i].shape) > 0]).flatten()
    sX = np.concatenate([
        unit['trajectory']['sx'][0, i]
        for i in range(N)
        if sum(unit['trajectory']['sx'][0, i].shape) > 0]).flatten()
    return X, Y, sX, sY


def rates_from_trj(unit):
    # Convert data to pixels
    N = unit['x'].shape[0]
    Y = np.concatenate([unit['y'][i]
                        for i in range(N)]).flatten()
    X = np.concatenate([unit['x'][i]
                        for i in range(N)]).flatten()
    sY = np.concatenate([
        unit['sy'][i]
        for i in range(N)
        if sum(unit['sy'][i].shape) > 0]).flatten()
    sX = np.concatenate([
        unit['sx'][i]
        for i in range(N)
        if sum(unit['sx'][i].shape) > 0]).flatten()
    return X, Y, sX, sY


def rates_over_time(unit):
    N = unit['trajectory']['x'].shape[1]
    sample = []
    cnt = 0
    offset = 0
    for trial in range(N):
        Y = np.around(unit['trajectory']['y'][0, trial].flatten(), 4)
        X = np.around(unit['trajectory']['x'][0, trial].flatten(), 4)
        sY = np.around(unit['trajectory']['sy'][0, trial].flatten(), 4)
        sX = np.around(unit['trajectory']['sx'][0, trial].flatten(), 4)
        counter = unit['trajectory']['scounter'][0, trial].flatten()
        if len(sX) == 0:
            continue

        if counter[0] < cnt:
            print('ping')
            offset += cnt
            print(offset)
        cnt = counter[0]

        for sx, sy in zip(sX, sY):
            idselect = (Y == sy) & (X == sx)
            index = np.where(idselect)[0]
            if (len(index) == 0) or (len(index) > 1):
                print(index)
                print(sx, sy)
                print(X, Y)
                raise RuntimeError('')
            sample.append(index + counter[0] + offset)
    return np.array(sample)


def psth(unit):
    N = unit['trajectory']['x'].shape[1]
    trials = []
    for trial in range(N):
        t = []
        Y = np.around(unit['trajectory']['y'][0, trial].flatten(), 4)
        X = np.around(unit['trajectory']['x'][0, trial].flatten(), 4)
        sY = np.around(unit['trajectory']['sy'][0, trial].flatten(), 4)
        sX = np.around(unit['trajectory']['sx'][0, trial].flatten(), 4)
        counter = unit['trajectory']['scounter'][0, trial].flatten()
        for sx, sy in zip(sX, sY):
            idselect = (Y == sy) & (X == sx)
            index = np.where(idselect)[0]
            if (len(index) == 0) or (len(index) > 1):
                print(index)
                print(sx, sy)
                print(X, Y)
                raise RuntimeError('')
            t.append(index)
        trials.append(t)
    return trials


def degrates(X, Y, sX, sY, Hz=1000):
    return rates((X * 24) + 400, (Y * 24) + 300,
                 (sX * 24) + 400, (sY * 24) + 300, Hz)


def rates(X, Y, sX, sY, Hz=1000):
    edges = parameters.get_ratemap_edges()
    samples = np.array((-1 * (Y - 300) / 24.0, (X - 400) / 24.0)).T
    time_spent = np.histogramdd(samples, edges)[0]
    time_spent = (time_spent) / Hz
    samples = np.array((-1 * (sY - 300) / 24.0, (sX - 400) / 24.0)).T
    spikes = np.histogramdd(samples, edges)[0]
    rate_map = spikes / time_spent
    rate_map[np.isnan(rate_map)] = 0
    return rate_map, time_spent, spikes


def gauss_rmap(rate_estimate, time_spent, spikes, sigma=2.5, mode='constant'):
    '''
    Calculates a gaussian rate map.
    '''
    ts = gaussian_filter(time_spent, sigma, mode=mode, cval=0.0)
    sp = gaussian_filter(spikes, sigma, mode=mode, cval=0.0)
    rmap = sp / ts
    return rmap


def make_fdm(x, y, xedges=np.linspace(-1000, 1000, 250),
             yedges=np.linspace(-1000, 1000, 250),
             sigma=1, mode='constant'):
    fdm = np.histogramdd([x, y], [xedges, yedges])[0]
    fdm = gaussian_filter(fdm, sigma, mode=mode, cval=0.0)
    fdm /= fdm.sum()
    return fdm


def xcorr2(a, b, nth=50):
    '''
    Calculate 2D Auto correlation
    '''
    Nshifts = a.shape[0]
    shifts = np.arange(-Nshifts + 2, Nshifts - 1)
    xcorr = np.ones((a.shape[0] * 2 + 1, a.shape[1] * 2 + 1)) * np.nan
    for x in shifts:
        for y in shifts:
            a2 = a.copy()
            idx0, idxe = 0, a2.shape[0]
            idy0, idye = 0, a2.shape[1]

            ix0 = max(0, idx0 + x)
            ixe = min(idxe, idxe + x)
            ixx0 = max(0, idx0 - x)
            ixxe = min(idxe, idxe - x)

            iy0 = max(0, idy0 + y)
            iye = min(idye, idye + y)
            iyy0 = max(0, idy0 - y)
            iyye = min(idye, idye - y)
            l1 = a[ix0:ixe, iy0:iye:].flatten()
            if l1.shape[0] <= nth:
                continue
            l2 = b[ixx0:ixxe, iyy0:iyye].flatten()
            xcorr[y + Nshifts, x + Nshifts] = np.corrcoef(l1, l2)[0, 1]
    return xcorr


def spike_intervals(trajectory):
    N = len(trajectory['x'][0])
    N = trajectory['x'].shape[1]
    X, Y, spikes = [], [], []
    for n in range(N):
        x, y = trajectory['x'][0][n], trajectory['y'][0][n]
        sx, sy = trajectory['sx'][0][n], trajectory['sy'][0][n]
        s = (np.in1d(x, sx) & np.in1d(y, sy))
        s = np.where(s > 0)[0]
        if len(s) > 1:
            s = s[1:] - s[0]
        else:
            continue
        X += [x]
        Y += [y]
        spikes += [s]
    return np.array(X), np.array(Y), np.array(spikes)
