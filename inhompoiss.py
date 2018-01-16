import numpy as np
import gness
import rate_map
from scipy.stats import expon
import parameters
try:
    from pylab import *
except:
    pass
import gridscores


def gscore_from_unit(
        cycles=2, rate=1.5, signal_ratio=.5,
        distance=10, sigma=1, rmap=None, **kwargs):
    if rmap is None:
        rot = np.random.randint(90)
        rmap = make_rate_map(distance, sigma, rot=rot)
    g, acmap, rmap, X, Y, rf, rn, sX, sY, samples = simulate_unit(
        cycles, rate,
        signal_ratio, rmap=rmap)
    gscore, model, s, cc, mc = gridscores.bestfit_gscore(acmap)
    res = {'cycles': cycles, 'rate': rate, 'signal_ratio': signal_ratio,
           'distance': distance, 'sigma': sigma,
           'gmap': g, 'acmap': acmap, 'rmap': rmap, 'gscore': gscore}
    return res


def plot_samples(samples, y=1, end=None, h=1, color='k', alpha=1):
    for s in samples:
        if end is not None and s > end:
            break
        plot([s, s], [y + 0, y + h], color=color, alpha=alpha)


def simulate_unit(
        cycles, rate=3, signal_ratio=1, rmap=None, noise=None,
        selected_cycles=None):
    assert (signal_ratio >= 0) and (signal_ratio <= 1)
    if noise is None:
        noise = expon.rvs
    signalrate = rate * signal_ratio
    noiserate = rate * (1 - signal_ratio)
    X, Y = [], []
    if not selected_cycles:
        for n in range(cycles):
            x, y = get_trj(np.random.randint(90))
            X.append(x)
            Y.append(y)
    else:
        for n in selected_cycles:
            x, y = get_trj(n)
            X.append(x)
            Y.append(y)

    X, Y = np.hstack(X), np.hstack(Y)
    if rmap is None:
        rmap = make_rate_map(15, 1)
    rmap = rmap / rmap.max()

    SX, SY, ratefct = get_rates(X, Y, rmap)
    # Scale the rates such that it matches the expected rate
    ratefct = ratefct / ratefct.sum()
    ratefct = ratefct * len(ratefct) * (signalrate / 120.0)
    # Add noise, noise parameter sets mean # of noise spikes
    # per second
    expnoise = noise(size=len(ratefct))
    expnoise = expnoise / expnoise.sum()
    expnoise = expnoise * len(expnoise) * (noiserate / 120.0)

    samples = sample(ratefct + expnoise)
    sX, sY = samples_to_space(X, Y, samples)
    g = rate_map.gauss_rmap(*rate_map.degrates(X, Y, sX, sY, 120))
    g[np.isnan(g)] = 0
    acmap = rate_map.xcorr2(g, g)
    return g, acmap, rmap, X, Y, ratefct, expnoise, sX, sY, samples


def plots(g, acmap, pp, X, Y):
    subplot(2, 2, 1)
    imshow(g)
    plot(X / 2.0, Y / 2.0, 'k')
    ylim([0, len(g) - 1])
    xlim([0, len(g) - 1])
    subplot(2, 2, 2)
    imshow(pp)
    ylim([0, len(pp) - 1])
    xlim([0, len(pp) - 1])
    plot(X, Y, 'k')
    subplot(2, 2, 3)
    imshow(acmap)


def sample(lambda_t):
    '''
    Generates samples from an inhomogenous poisson process.
    lambda_t describes the firing rate as a function of time.
    Each index into lambda_t is treated as one unit of time.
    If each index describes a ms, the firing rates are encoded
    in spikes per ms.
    '''
    T = 0
    i = 0
    lu = (lambda_t * 0) + lambda_t.max()
    res = [0]
    while T < len(lambda_t) - 1:
        u1 = np.random.rand()
        T = T - (1 / lu[int(T)]) * (np.log(u1))
        u2 = np.random.rand()
        if int(T) > (len(lambda_t) - 1):
            return np.array(res)
        if u2 <= lambda_t[int(T)] / lu[int(T)]:
            i = i + 1
            res.append(T)
    return np.array(res)


def samples_to_space(X, Y, samples):
    return X[samples.astype(int)], Y[samples.astype(int)]


def get_trj(num):
    '''
    Open a trajectory file and return X and Y coordinates
    of the trajectory.
    '''
    t = open('trjs/%dhm.trj' % num)
    t.readline()
    c = np.array([float(l) for l in t])
    X = c[0::2]
    Y = c[1::2]
    return X, Y


def get_rates(X, Y, rate_map, binsize=0.25):
    '''
    Assign firing rates to coordinates.
    Expects X and Y to be in degree of visual angle.
    '''
    ex, ey = parameters.get_ratemap_edges()
    Xi = np.digitize(X, ex)
    Yi = np.digitize(Y, ey)
    return X, Y, flipud(rate_map)[Yi - 1, Xi - 1]


def make_rate_map(distance=10, sigma=1, rot=0, scale_x=1.0,
                  x_shift=0, y_shift=0):
    '''
    Make a rate map for a fake grid cell.

    Parameters are interpreted as being given in degree.
    Binsize indicates how many degrees are covered by one bin.

    '''
    ex, ey = parameters.get_ratemap_edges()
    binsize = ex[1] - ex[0]
    rot = rot * np.pi / 180.0
    scale_x = scale_x * ((.75 * (distance**2)) / ((0.5 * distance)**2))**.5
    # Screen is -25:25
    points = []
    dimensions = ex[0] - 2 * distance, ex[-1] + 2 * distance, 0, distance
    acc_dims = ex[0], ex[-1], int((ex[-1] - ex[0]) * (1 / binsize))
    for ix in centerspace(*dimensions):
        for iy in centerspace(*dimensions):
            points.append((ix * scale_x, iy))
    for ix in centerspace(*dimensions):
        for iy in centerspace(*dimensions):
            points.append((ix * scale_x + scale_x * distance / 2.0,
                           iy + distance / 2.0))
    points = set(points)
    points = [(np.cos(rot) * x - np.sin(rot) * y,
               np.sin(rot) * x + np.cos(rot) * y) for x, y in points]
    c0 = np.linspace(*acc_dims)
    c1 = np.linspace(*acc_dims)
    acc = 0 * np.ones((len(c0), len(c1)))
    sigmax = sigmay = sigma
    for x, y in points:
        xc = gness.pnormpdf_fast(c0, x + x_shift, sigmax)
        yc = gness.pnormpdf_fast(c1, y + y_shift, sigmay)
        acc = acc + np.dot(xc[:, np.newaxis], yc[np.newaxis, :])
    return acc


def centerspace(minv, maxv, center, spacing, exclusive=True):
    if not exclusive and (mod(minv, spacing) == 0 or mod(maxv, spacing) == 0):
        minv = minv - spacing
        maxv = maxv + spacing
    vals = arange(center, minv, -
                  spacing)[-1:0:-1], arange(center, maxv, spacing)
    return np.concatenate(vals)
