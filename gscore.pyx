# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
import parameters
from scipy import nanmean


def gscore_model_list(acmap, model_collection, nbins=72, filter=None):
    scores = np.empty(model_collection.shape[0])
    circ_cors = []
    for i, model in §enumerate(model_collection):
        # Compute circular histogram.
        bin_centers, circ_values, width = circhist(acmap, model, nbins=nbins)
        shifts, circ_cor = circcorr(circ_values, width)
        gridscore = _gscore(shifts, circ_cor)
        scores[i] = gridscore
        circ_cors.append(circ_cor)
    scores, circ_cors = np.array(scores), np.array(circ_cors)
    return scores, circ_cors, shifts


def gscore_from_model_list(acmap, model_collection, nbins=72, filter=None):
    scores, circ_cors, shifts = gscore_model_list(
        acmap, model_collection, nbins)
    id_nan = np.isnan(scores)
    scores = scores[~id_nan]
    circ_cors = circ_cors[~id_nan]
    if len(scores) == 0:
        bin_centers, circ_values, width = circhist(
            acmap, [10, 0, 2, 2, 2], nbins=nbins)
        shifts, circ_cor = circcorr(circ_values, width)
        return (0, np.array([np.nan] * 5), shifts * np.nan,
                circ_cor * np.nan, model_collection)
    id_max = np.argmax(scores)
    model_collection = model_collection[~id_nan, :]
    if filter is not None:
        filter = filter[~id_nan, :]
        return (scores[id_max],  model_collection[id_max, :], shifts,
                circ_cors[id_max], model_collection, filter[id_max, :])
    return (scores[id_max],  model_collection[id_max, :], shifts,
            circ_cors[id_max], model_collection)


def _gscore(shifts, circ_cor):
    oncenter = [[55, 65], [115, 125]]
    offcenter = [[25, 35], [85, 95], [145, 155]]
    mins_offcenter = []
    for low, high in offcenter:
        idx = (low <= shifts) & (shifts <= high)
        mins_offcenter.append(min(circ_cor[idx]))
    max_oncenter = []
    for low, high in oncenter:
        idx = (low <= shifts) & (shifts <= high)
        max_oncenter.append(max(circ_cor[idx]))
    return min(max_oncenter) - max(mins_offcenter)


def circhist(acmap, model, nbins=12, binsize=0.5):
    '''
    Compute a circular hisogram.

    Input:
        acmap :  2d autocorrelation matrix
        model :  model vector
        nbins :  the number of bins to use for the
            circular histogram.
        binsize : Extend in degrees of each entry in the rate map that
            generated the autocorrelation function.
    '''
    assert acmap.shape[0] == len(parameters.get_acmap_shifts()[0])
    extend, angle, scale, sigma, rotangle = model
    assert angle >= 0
    binr = sigma * 1.5
    edges = np.linspace(0, 360, nbins + 1)
    edges = np.mod(edges + angle, 360)
    clow, chigh = extend - binr, extend + binr
    width = abs(np.diff(edges)).min()
    centers = np.cumsum([width] * nbins) - (width / 2.0)
    theta, r = model2coord(model[0], model[1], model[2], model[3], model[4])
    res = []
    for low, high in zip(edges[:-1], edges[1:]):
        index = (low <= theta) & (theta <= high)
        if high < low:
            index = ~ ((high <= theta) & (theta <= low))
        index = index & ((clow) <= r) & (r <= (chigh))
        res.append(nanmean(acmap[index]))
    return centers, np.array(res), width


cpdef model2coord(double extend, double angle, double scale,
                  double sigma, double rotangle):
    cdef np.ndarray[np.int64_t, ndim = 2] X, Y
    cdef np.ndarray[np.float64_t, ndim = 2] theta, XA, YA, r
    cdef np.ndarray[np.int64_t] sh = parameters.get_acmap_shifts()[0]
    cdef double pi = np.pi
    Y, X = np.meshgrid(sh, sh, indexing='xy')
    theta = np.mod(-angle + ((np.arctan2(Y, X) * 180 / pi) + 180), 360.0)
    rotangle = -rotangle * np.pi / 180.0
    XA = X * np.cos(rotangle) - Y * np.sin(rotangle)
    YA = X * np.sin(rotangle) + Y * np.cos(rotangle)
    r = ((XA / scale)**2 + YA**2)**.5
    return theta, r


cpdef circcorr(np.ndarray[np.float64_t] r, int binwidth):
    '''
    Compute a circular correclation.
    '''
    cdef np.ndarray[np.float64_t] res = np.empty(r.shape[0])
    for l in range(r.shape[0]):
        res[l] = np.corrcoef(r, np.roll(r, l))[0, 1]
    #res = [np.corrcoef(r,np.roll(r,l))[0,1] for l in range(len(r))]
    cdef np.ndarray[np.float64_t] shifts = binwidth * np.ones(r.shape[0])
    return np.cumsum(shifts), res
