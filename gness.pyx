from __future__ import division
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

cimport numpy as np
cimport cython

from scipy.ndimage import zoom
import ellipsemodels as em
import parameters

from scipy.misc import comb
from itertools import combinations

DTYPE = np.float
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t

# cython: profile=True

cdef extern from "math.h":
    double exp(double theta)
    double sqrt(double theta)
    double log(double theta)
    double pow(double theta, int p)
    double sin(double x)
    double cos(double x)


@cython.boundscheck(False)
def peak_distances(np.ndarray[np.double_t, ndim=2] models,
                   np.ndarray[np.double_t, ndim=2] peaks):
    '''
    Models needs to be in Scheitelpunktform.
    Peaks need to be 0-centered (x,y).
    Scheitelpunktform also is 0 centered (x,y)
    '''
    cdef np.ndarray[np.double_t, ndim = 2] dists = np.ones((models.shape[0],
                                                            peaks.shape[0]))
    cdef int i, j
    cdef float s1x, s1y, s2x, s2y
    for i in range(models.shape[0]):
        s1x = models[i, 0]
        s1y = models[i, 1]
        s2x = models[i, 2]
        s2y = models[i, 3]
        for j in range(peaks.shape[0]):
            dists[i, j] = clp(s1x, s1y, s2x, s2y, peaks[j, 0], peaks[j, 1])
    return dists


def pclp(s1x, s1y, s2x, s2y, px, py):
    return clp(s1x, s1y, s2x, s2y, px, py)


@cython.boundscheck(False)
cdef float clp(float s1x, float s1y, float s2x, float s2y, float px, float py):
    """
    Find closest point on ellipse (s1,s2) relative to point p.
    """
    cdef np.ndarray[np.double_t] theta = np.linspace(0, 2 * np.pi, 2 * 360)
    cdef int i
    cdef np.ndarray[np.double_t] d = np.zeros((theta.shape[0],), dtype=np.float)
    cdef double xp, yp
    cdef int idx
    for i in range(theta.shape[0]):
        xp = s1x * cos(theta[i]) + s2x * sin(theta[i])
        yp = s1y * cos(theta[i]) + s2y * sin(theta[i])
        d[i] = sqrt(pow(xp - px, 2) + pow(yp - py, 2))
    idx = np.argmin(d)
    return d[idx]


@cython.boundscheck(False)
def peaks2triple(np.ndarray[np.double_t, ndim=2] dists,
                 int num_peaks):
    '''
    Return summed distance to each peak triplet.
    '''
    cdef np.ndarray[np.double_t, ndim = 2] results = np.ones(
        (dists.shape[0], comb(num_peaks, 3, exact=True)))
    cdef int m, i, j, k, l
    for m, (i, j, k) in enumerate(combinations(range(num_peaks), 3)):
        for l in range(dists.shape[0]):
            results[l, m] = dists[l, i]**2 + dists[l, j]**2 + dists[l, k]**2
    return results


def fit(np.ndarray[np.float64_t, ndim=2] acmap,
        np.ndarray[np.float64_t, ndim=2] models):
    '''
    Fit a mdoel to the auto correlation map.
    '''
    if (np.isnan(acmap)).sum() > 0:
        acmap[np.isnan(acmap)] = 0
    # Scaling Down image, need to upscale extend in model later
    if acmap.shape[0] != parameters.get_acmap_shifts()[0].shape[0]:
        raise RuntimeError('acmap has incorrect size')
    pre_size = acmap.shape[0]

    # Size image down such that it has usable size
    scale_factor = (1 + parameters.get_ratemap_edges()
                    [0].shape[0]) / float(acmap.shape[0])

    cdef np.ndarray[np.double_t, ndim = 2] acmapo
    acmap = zoom(acmap, scale_factor)
    w = acmap.shape[0]

    acmap = acmap[: int(w / 2), :]
    cdef np.ndarray[np.float64_t, ndim = 1] opfunc
    opfunc = np.empty(len(models))
    cdef Py_ssize_t i
    cdef np.ndarray q, q2
    for i in range(models.shape[0]):
        # extend, angle, scale, sigma, sigma, theta
        q = prediction(models[i, 0], models[i, 1], models[i, 2],
                       models[i, 3], models[i, 3], models[i, 4])

        q2 = prediction(models[i, 0], models[i, 1] + 30, models[i, 2],
                        models[i, 3], models[i, 3], models[i, 4])
        q = q - (q2 / 2.0)
        opfunc[i] = rmse(acmap, q)
    id_min = np.unique(find_mins(opfunc, len(opfunc) / 5))
    model = models[id_min.astype(int), :]
    return model, opfunc


def find_mins(np.ndarray op, int l):
    cdef np.ndarray v = np.nan * np.ones(len(op) - l)
    cdef np.ndarray ids = np.nan * np.ones(len(op) - l)
    for k in range(len(op) - l):
        v[k] = op[k:k + l].min()
        ids[k] = k + np.argmin(op[k:k + l])
    return ids


def pprediction(double extend, double angle, double scale=1,
                double sigmax=5, theta=0, full=True):
    q1 = prediction(extend, angle, scale, sigmax, sigmax, theta, full)
    q2 = prediction(extend, angle + 30, scale, sigmax, sigmax, theta, full)
    return q1 - (q2 / 2.0)


cdef np.ndarray prediction(double extend, double angle, double scale,
                           double sigmax, double sigmay, double theta, int full=False):
    '''
    Compute the prediction of a grid cell model given.
    '''
    cdef int dim = parameters.get_ratemap_edges()[0].shape[0]
    # np.mgrid[-50:50:101j]
    cdef np.ndarray[np.float64_t] c1 = np.arange(-dim, dim + 1, 2, dtype=np.float64)
    cdef np.ndarray[np.float64_t] c0 = np.arange(-dim, 0, 2, dtype=np.float64)
    if full:
        c0 = np.arange(-dim + 1, dim, 1, dtype=np.float64)
        c1 = np.arange(-dim + 1, dim, 1, dtype=np.float64)
    cdef np.ndarray[np.float64_t] angles = (angle +
                                            np.array([0, 60, 120, 180, 240, 300])) * float(np.pi) / 180.0
    cdef np.ndarray[np.float64_t] xa = -scale * extend * np.sin(angles)
    cdef np.ndarray[np.float64_t] ya = extend * np.cos(angles)
    theta = theta * np.pi / 180.0
    cdef np.ndarray xaa = +xa * np.cos(theta) - ya * np.sin(theta)
    cdef np.ndarray yaa = +xa * np.sin(theta) + ya * np.cos(theta)
    xa = xaa
    ya = yaa
    cdef np.ndarray[np.float64_t, ndim = 2] acc = 0 * np.ones((len(c0), len(c1)))
    cdef int i
    cdef double x, y
    cdef np.ndarray[np.float64_t, ndim = 1] xc
    cdef np.ndarray[np.float64_t, ndim = 1] yc
    for i in range(xa.shape[0]):
        x = xa[i]
        y = ya[i]
        xc = normpdf_fast(c0, x, sigmax)
        yc = normpdf_fast(c1, y, sigmay)
        acc = acc + np.dot(xc[:, np.newaxis], yc[np.newaxis, :])
    acc = acc / acc.max()
    return acc


def prmse(q, p):
    return rmse(q, p)


cdef double rmse(np.ndarray[np.float64_t, ndim=2] q,
                 np.ndarray[np.float64_t, ndim=2] p):
    return ((q - p)**2).sum() / (q.shape[0] * q.shape[1])


def normpdf(np.ndarray[np.float64_t] x, double mu, double sigma):
    cdef np.ndarray[np.float64_t, ndim = 1] y, u
    u = (x - mu) / abs(sigma)
    y = np.exp(-u * u / 2.0) / (np.sqrt(2.0 * np.pi) * abs(sigma))
    return y


def pnormpdf_fast(x, mu, sigma):
    return normpdf_fast(x, mu, sigma)


cdef np.ndarray normpdf_fast(np.ndarray[np.float64_t, ndim=1] x,
                             double mu, double sigma):
    cdef np.ndarray[np.float64_t, ndim = 1] y = np.empty(x.shape[0],
                                                         dtype=np.float64)
    cdef double normalization = sqrt(2.0 * 3.141592653589793) * sigma
    cdef double u
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        u = (x[i] - mu) / sigma
        y[i] = exp(-u * u / 2.0) / normalization
    return y
