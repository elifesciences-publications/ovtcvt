# -*- coding: utf-8 -*-
import cPickle
import glob
import socket

from os.path import join

from numpy import *
import numpy as np

try:
   from pylab import *
except:
   pass

from itertools import product
from scipy import nanmean
from scipy.io import loadmat
from scipy.stats import norm
from skimage.feature import peak_local_max
from skimage.morphology import watershed

import ellipsemodels as em
import gness
import gscore
import parameters
import tools

from itertools import combinations
from scipy.misc import comb

# Default parameters and book keeping
def_extends = np.arange(10, 40, 5)
def_angle = np.arange(0, 60, 5)
def_scale = [0.5, .75, 1, 1.25, 1.5]
def_theta = np.arange(0, 90, 10)
def_sigma = np.linspace(2, 4, 5)

models = None
indices = None
dimensions = None


def peak_outline(acmap, peak, sump=50):
    '''
    Finds and returns points associated with peaks. The points follow
    the outline of the maximum. The returned points are 0 centered.
    '''
    markers = acmap.copy() * 0
    markers[tuple(peak)] = 1
    maxval = acmap[tuple(peak)]
    test_vals = np.unique(acmap[(acmap > 0) & (acmap < maxval)].round(2))
    for t in test_vals:
        markers[acmap < t] = 2
        p = watershed(-1 * acmap, markers)
        if np.sum(p == 1) <= sump:
            break
    X, Y = np.meshgrid(parameters.get_acmap_shifts()[0],
                       parameters.get_acmap_shifts()[0])

    return Y[p == 1][::2], X[p == 1][::2]


def check_acmap(acmap, peaks=None, jitter=False):
    acmap[np.isnan(acmap)] = 0
    if acmap.std() == 0:
        raise ValueError('STD of acmap has to be larger than 0')
    if peaks is None:
        peaks = get_bestfit_peaks(acmap, jitter=False)[
            ::-1] - acmap.shape[0] / 2.0
        if len(peaks) < 3:
            raise ValueError
    return peaks


def constrained_distance_models(acmap, models=None, peaks=None):
    if peaks is None:
        peaks = get_bestfit_peaks(acmap, jitter=False, group=False)
        # Peaks are Non centered (y,x)
        if len(peaks) < 4:
            raise ValueError
    if models is None:
        ps = [np.array(peak_outline(acmap, p, sump=75)).T for p in peaks]
        peaks = np.array([np.mean(p, 0) for p in ps])
        scheitel, models = generate_2point_models(ps, flipxy=True)
    else:
        scheitel, models = models[0]
    dists = gness.peak_distances(scheitel, np.array(peaks).astype(float))
    rr = gness.peaks2triple(dists, len(peaks))
    return models, peaks, dists, rr


def constrained_distance_gscore(acmap, models=None, peaks=None):
    try:
        models, peaks, dists, rr = constrained_distance_models(
            acmap, models, peaks)
    except ValueError:
        gs, m, s, c, ml = gscore.gscore_from_model_list(
            acmap, np.array([[10, 0, 0, 2, 0]]))
        return 0, m * nan, s * nan, c * nan, ml * nan

    model_list = []
    for rs in rr.T:
        idsmall = rs < 15
        if np.sum(idsmall) == 0:
            idsmall = rs <= min(rs)
        model_list.extend(models[idsmall])
    model_list = np.array(list(set([tuple(m) for m in model_list])))
    if len(model_list) < 1:
        gs, m, s, c, ml = gscore.gscore_from_model_list(
            acmap, np.array([[10, 0, 0, 2, 0]]))
        return 0, m * nan, s * nan, c * nan, ml * nan
    return gscore.gscore_from_model_list(acmap, model_list)


def bestfit_gscore(acmap, percent_models=0.1):
    '''
    Compute a gridscore that sorts possible models by their
    distance to peaks on the map.
    '''
    try:
        models, peaks, dists, rr = constrained_distance_models(
            acmap)
    except ValueError:
        gs, m, s, c, ml = gscore.gscore_from_model_list(
            acmap, np.array([[10, 0, 0, 2, 0]]))
        return 0, m * nan, s * nan, c * nan, ml * nan

    model_list = []
    for peak_comb in range(rr.shape[1]):
        dists = rr[:, peak_comb]
        idx = argsort(dists)[:int(len(dists) * percent_models)]
        for i in idx:
            model_list.append(models[i])
    model_list = array(list(set([tuple(m) for m in model_list])))
    return gscore.gscore_from_model_list(acmap, array(model_list))


def get_bestfit_peaks(acmap, jitter=False, group=False):
    peaks = get_peaks(acmap)
    if len(peaks) == []:
        return ValueError
    d = ((peaks - 89 / 2.0)**2).sum(1)**.5
    peaks = peaks[d < acmap.shape[0] / 2.0, :]
    # Now jitter the peaks around, they are never perfect.
    ls = []
    for p in peaks:
        if acmap[tuple(p)] < 0:
            continue
        ls.append(p)
    peaks = array(ls)
    if jitter:
        x = arange(-2, 3)
        X, Y = meshgrid(x, x)
        D = (X**2 + Y**2)**.5
        D = (D < 2.5).astype(bool)
        jitter = np.vstack((X[D].flat, Y[D].flat))
        if group:
            peaks = np.vstack([[p - jitter.T] for p in peaks])
        else:
            peaks = np.vstack([p - jitter.T for p in peaks])

    return peaks


def peaks2triple(dists, peaks):
    results = np.ones((dists.shape[0], comb(
        len(peaks), 3, exact=True))) * np.nan
    for m, (i, j, k) in enumerate(combinations(arange(len(peaks)), 3)):
        for l in range(dists.shape[0]):
            results[l, m] = dists[l, i] + dists[l, j] + dists[l, k]
    return results

import itertools


def generate_2point_models(peaks, scheitelform=True, flipxy=True):
    '''
    Generate ellipses that go through pairs of points.
    Peaks is supposed to be a list  that contains a mx2 array for each peak.
    Each mx2 array contains y,x points associated with one peak.
    '''
    scheitel_models = []
    models = []
    for i, p1 in enumerate(peaks):
        for j, p2 in enumerate(peaks[i + 1:]):
            for point1, point2 in itertools.product(p1, p2):
                if flipxy:
                    point1, point2 = point1[::-1], point2[::-1]
                f1, f2, s1, s2, shift, extend, scale, rotation = em.toparameters(
                    point1, point2)

                #assert (all(point1[::-1] == f1)) or all((point1[::-1] == f2))

                if scale < 0.5 or scale > 1.5:
                    continue
                extend, shift, = round(extend, 1), round(
                    normalize_angle(shift), 0)
                scale, rotation, = round(scale, 2), round(
                    normalize_angle(rotation), 0)

                scheitel_models.append((f1[0], f1[1], f2[0], f2[1], shift))
                models.append((extend, shift, scale, 2.0, rotation))
    return vstack(scheitel_models), vstack(models)


def generate_models(acmap, peaks, scheitelform=True):
    '''
    Generate a set of candidate ellipses by iterating
    through a grid.

    peaks needs to be 0-centered.
    '''
    w = acmap.shape[0]
    points = array([(y, x) for x in arange(1, w / 2.0, 1)
                    for y in arange((-w / 2) + 1, w / 2, 1)])
    models = []
    fs = []
    for i, p1 in enumerate(peaks):
        dp1 = (p1**2).sum()**.5
        for j, p2 in enumerate(points[i + 1:]):
            dp2 = (p2**2).sum()**.5
            if dp2 < 1.5 or dp2 > w / 2.0:
                continue
            f1, f2, s1, s2, shift, extend, scale, rotation = em.toparameters(
                p1, p2)
            assert (all(p1 == f1)) or all((p1 == f2))
            if scale < 0.5 or scale > 1.5:
                continue
            if scheitelform:
                models.append((f1[0], f1[1], f2[0], f2[1], shift))
            else:
                models.append(
                    (extend, normalize_angle(shift), scale, 2.0, normalize_angle(rotation)))
            fs.append((f1, f2))
    return vstack(models)


def twopeak_gscore(acmap):
    '''
    Compute a grid score where the model is constrained
    to go through two of the local maxima
    '''
    acmap[isnan(acmap)] = 0
    if acmap.std() == 0:
        raise ValueError('STD of acmap has to be larger than 0')
    peaks = get_peaks(acmap, min_distance=2.5)
    if len(peaks) == 1:
        return constrained_gscore(acmap)
    peaks = peaks[:, ::-1] - acmap.shape[0] / 2.0
    models = []
    num_peaks = peaks.shape[0]
    for i in range(num_peaks):
        for j in range(i + 1, num_peaks):
            f1, f2 = peaks[i, :], peaks[j, :]
            f1, f2, s1, s2, shift, extend, scale, rotation = em.toparameters(
                f1, f2)
            # shift puts one of the points onto one of the grid peaks
            # so we have four parameters set, only sigma needs to be tested
            if np.linalg.norm(s1) < 2.5 or np.linalg.norm(s2) < 2.5:
                continue
            if extend * scale < 10:
                continue
            if extend * scale > (acmap.shape[0] / 2):
                continue
            if scale < 0.5 or scale > 1.5:
                continue
            for sigma in def_sigma:
                m1, m2 = model_from_two_peaks(f1, f2, sigma)
                models.append([m1])
                models.append([m2])
    if len(models) == 0:
        # No valid models found.
        gs, model, shifts, cc, mc = gscore.gscore_from_model_list(acmap,
                                                                  np.array([[10, 0, 0, 2, 0]]))
        return 0, model * np.nan, shifts * np.nan, cc * np.nan, mc * np.nan
    models = np.concatenate(models)
    model_collection, op = gness.fit(acmap, models)
    return gscore.gscore_from_model_list(acmap, model_collection)


def constrained_gscore(acmap):
    '''
    Compute a grid score where the model is constrained
    to go through one of the local maxima.
    '''
    peaks = get_peaks(acmap)
    try:
        models = []
        for x, y in peaks:
            if not ((x == acmap.shape[0] / 2) and (y == acmap.shape[1] / 2)):
                m = models_from_peak(x, y, def_sigma)
                if m is not None:
                    models.append(m)
        models = np.concatenate(models)
    except ValueError as e:
        print e
        print 'No Peaks found, using unconstrained gridscore'
        gs, model, shifts, cc, mc = gscore.gscore_from_model_list(
            acmap,
            np.array([[10, 0, 0, 2, 0]]))
        return 0, model * np.nan, shifts * np.nan, cc * np.nan, mc * np.nan
    model_collection, op = gness.fit(acmap, models)
    return gscore.gscore_from_model_list(acmap, model_collection)


def unconstrained_gscore(acmap):
    '''
    Compute a gridness score from a 2D auto correlation.
    '''
    assert acmap.shape[0] == acmap.shape[1]
    if sum(isnan(acmap)) > 0:
        acmap[isnan(acmap)] = 0
    assert acmap.std() > 0
    # Compute gridness model
    model_collection, op = gness.fit(acmap, models)
    return gscore.gscore_from_model_list(acmap, model_collection)


def get_peaks(acmap, min_distance=5):
    '''
    Get a list of local maxima. If none can be found return
    global maximum (exluding the center).
    '''
    assert acmap.shape[0] == acmap.shape[1]
    if sum(isnan(acmap)) > 0:
        acmap[isnan(acmap)] = 0
    assert acmap.std() > 0
    peaks = peak_local_max(acmap,
                           min_distance=min_distance, exclude_border=False)
    peaks = peaks[peaks[:, 1] > (acmap.shape[0] / 2.0), :].astype(float)
    peaks[:, 1] -= acmap.shape[0] / 2.0
    # Remove center peak
    pdist = sum((peaks - [acmap.shape[0] / 2.0, 0])**2, 1)**.5
    peaks = peaks[pdist > 2.5, :]
    if len(peaks) == 0:
        # If no peak can be found, take maximum of acmap, exluding center
        ac = acmap.copy()
        X, Y = meshgrid(parameters.get_acmap_shifts()[0],
                        parameters.get_acmap_shifts()[0])
        dist = ((X**2 + Y**2)**.5)
        id_dist = (dist < 10) & (dist < 40)
        ac[id_dist] = 0
        idx = argmax(ac)
        peaks = array([unravel_index(idx, ac.shape)])
    else:
        peaks[:, 1] += acmap.shape[1] / 2.0

    return peaks


def normalize_angle(alpha):
    '''
    Normalize all alphas in model to be in [0...360]
    and to be in degrees.
    '''
    alpha = alpha * 180 / np.pi
    if alpha < 0:
        alpha = 360 + alpha
    return alpha


def models_from_peak(x, y, sigmas):
    '''
    Computes some models that go through the (x,y) coordinate.
    '''
    rot = 90 + (np.arctan2(y - 44, x - 44) * 180 / pi)
    extend = np.linalg.norm([x - 44, y - 44])
    models = []
    for angle, sigma, scale in product(def_angle, sigmas, def_scale):
        model = [extend, angle, scale, sigma, rot]
        if valid(model)[0]:
            models.append(model)
    if len(models) == 0:
        return None
        #raise RuntimeError('Could not find any suitable model')
    return models


def model_from_two_peaks(f1, f2, sigma=2.0):
    f1, f2, s1, s2, shift, extend, scale, rotation = em.toparameters(f1, f2)
    # shift puts one of the points onto one of the grid peaks
    # so we have four parameters set, only sigma needs to be tested
    m1 = (extend, normalize_angle(shift), scale, sigma,
          normalize_angle(rotation))
    f1r, f2r, s1, s2, shift, extend, scale, rotation = em.toparameters(f2, f1)
    if all(f2r == f1) and all(f1r == f2):
        shift = -shift
    else:
        shift = (np.pi / 2.0) + shift
    m2 = (extend, normalize_angle(shift), scale, sigma,
          normalize_angle(rotation))
    return m1, m2


def gscore_from_model(acmap, model):
    '''
    Compute a grid score from a single model.
    '''
    return gscore.gscore_from_model_list(acmap, array([model]))


def valid(model):
    extend, angle, scale, sigma, theta = model
    angles = (angle + np.array([0, 60, 120, 180,
                                240, 300])) * float(np.pi) / 180.0
    xa = -scale * extend * np.sin(angles)
    ya = extend * np.cos(angles)
    theta = theta * np.pi / 180.0
    xaa = +xa * np.cos(theta) - ya * np.sin(theta)
    yaa = +xa * np.sin(theta) + ya * np.cos(theta)
    distances = []
    max_distances = []
    for x, y in zip(xaa, yaa):
        if x < 0:
            continue
        distance = (x**2 + y**2)**.5
        box = parameters.get_acmap_shifts()[0][-1]
        if y < 0:
            box = box * -1
        slope = y / x
        p = min(box / slope, box * slope)
        max_dist = (box**2 + p**2)**.5
        distances.append(distance)
        max_distances.append(max_dist)
    distances, max_distances = array(distances), array(max_distances)
    return all(distances < max_distances), distances, max_distances


@tools.Memoize
def valid_parameters(extends, angle, scale, theta, sigma, cutoff=49):
    print 'Calculating valid models'
    valid = []
    dimensions = (len(extends), len(angle), len(scale), len(sigma), len(theta))
    indices = []
    for cnt, (e, sig, sc, t, a) in enumerate(product(extends, sigma, scale,
                                                     theta, angle)):
        angles = (a +
                  np.arange(360)) * float(np.pi) / 180.0
        xa = -sc * e * np.sin(angles)
        ya = e * np.cos(angles)
        if (any(xa < -cutoff) or any(xa > cutoff) or
                any(ya < -cutoff) or any(ya > cutoff)):
            continue
        if e * sc < 15:   # Ellipsis has to be big enough in center
            continue
        valid.append(array((e, a, sc, sig, t)))
        indices.append(cnt)
    return array(valid), array(indices), dimensions

if models is None:
    import os
    if os.path.exists('valid_model_cache.pickle'):
        models, indices, dimensions = cPickle.load(
            open('valid_model_cache.pickle'))
        print "Using cached model parameters"
    else:
        models, indices, dimensions = valid_parameters(
            def_extends, def_angle, def_scale, def_theta, def_sigma)
        cPickle.dump((models, indices, dimensions),
                     open('valid_model_cache.pickle', 'w'))

gscore_fcts = {
    'constrained_gscore': constrained_gscore,
    'unconstrained_gscore': unconstrained_gscore,
    'bestfit_gscore': bestfit_gscore,
    'constrained_distance_gscore': constrained_distance_gscore,
    'twopeak_gscore': twopeak_gscore}
