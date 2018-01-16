
'''
This module computes ellipses from two points. The ellipses
are centered around the origin. See https://de.wikipedia.org/wiki/Ellipse
for more info, especially the section "Ellipse als affines Bild des
Einheitskreises"

'''
import numpy as np
from pylab import *


def tzero(f1, f2):
    '''
    Compute t0 parameter of ellipse defined by f1 and f2:
    cot(2*t0) = f1^2 * f2^2 / 2*f1*f2
    ~ roughly means how far you have turn along the ellipse to get
    to the major axis of the ellipse.
    '''
    d = np.linalg.norm(f1)**2 - np.linalg.norm(f2)**2
    n = 2 * np.dot(f1, f2)
    if d == 0:
        return inf
    return d / n


def scheitel(f1, f2):
    '''
    Compute the intersections of the ellipse with its major and minor axis
    Returns:
        s1 : intersection of major axis is at +- s1
        s2 : intersection of minor axis is at +-s2
    '''
    shift = acot(tzero(f1, f2)) / 2
    if np.dot(f1, f2) == 0:
        shift = 0
    s1 = (np.array(f1) * np.cos(shift) +
          np.array(f2) * np.sin(shift))
    s2 = (np.array(f1) * np.cos(shift + np.pi / 2.) +
          np.array(f2) * np.sin(shift + np.pi / 2))
    return s1, s2, shift


def scheitel2parameters(s1, s2):
    rotation = np.arctan2(s1[0], s1[1])
    if rotation > np.pi / 2 or rotation < -np.pi / 2:
        rotation = -np.arctan2(s1[1], s1[0])
        s1, s2 = s2, s1
    extend = np.linalg.norm(s1)
    scale = np.linalg.norm(s2) / extend
    return extend, scale, rotation - (pi / 2.0)


def toparameters(f1, f2):
    '''
    Convert ellipse (f1,f2) to flat form.
    '''
    s1, s2, shift = scheitel(f1, f2)
    rotation = np.arctan2(s1[0], s1[1])
    if rotation > np.pi / 2 or rotation < -np.pi / 2:
        rotation = -np.arctan2(s1[1], s1[0])
        s1, s2 = s2, s1
        f1, f2 = f2, f1
    extend = np.linalg.norm(s1)
    scale = np.linalg.norm(s2) / extend
    return f1, f2, s1, s2, shift, extend, scale, rotation - (pi / 2.0)


def model2scheitel(
        extend, angle, scale,
        sigma, rotangle):

    # sh = parameters.get_acmap_shifts()[0]
    X, Y = array([0, extend]), array([extend * scale, 0])
    # theta = np.mod(-angle+((np.arctan2(Y,X)*180/pi)+180),360.0)
    rotangle = -rotangle * np.pi / 180.0
    XA = X * np.cos(rotangle) - Y * np.sin(rotangle)
    YA = X * np.sin(rotangle) + Y * np.cos(rotangle)
    r = ((XA / scale)**2 + YA**2)**.5
    return (XA[0], YA[0]), (XA[1], YA[1])


def acot(x):
    '''
    Dinverse cotangent
    '''
    return np.arctan(1. / x)


def point(angle, f1, f2, shift=0):
    angle = np.asarray(angle)
    try:
        return (np.array(f1)[:, np.newaxis] * np.cos(angle + shift)[np.newaxis, :] +
                np.array(f2)[:, np.newaxis] * np.sin(angle + shift)[np.newaxis, :])
    except IndexError:
        angle = np.array([angle])
        return (np.array(f1)[:, np.newaxis] * np.cos(angle + shift)[np.newaxis, :] +
                np.array(f2)[:, np.newaxis] * np.sin(angle + shift)[np.newaxis, :])


def compare(f1, f2, offset=0):
    s1, s2, shift = scheitel(f1, f2)
    l = point(np.linspace(0, 4 * np.pi, 1000), s1, s2, shift)
    plot(l[0, :] + offset, l[1, :] + offset, 'k', lw=0.5)
    for (x, y), c in zip([f1, f2, s1, s2], ['g', 'g', 'r', 'r']):
        plot([-x + offset, x + offset], [-y + offset, y + offset], c, lw=0.5)
    axis('equal')


def plot_ellipse(f1, f2, offset=[0, 0], lw=0.5):
    s1, s2, shift = scheitel(f1, f2)
    l = point(np.linspace(0, 4 * np.pi, 1000), s1, s2, shift)
    plot(l[0, :] + offset[0], l[1, :] + offset[1], 'k', lw=lw)


import gness


def test_random_points():
    f1 = [np.random.randint(69) - 69 / 2, np.random.randint(69) - 69 / 2]
    f2 = [np.random.randint(69) - 69 / 2, np.random.randint(69) - 69 / 2]

    s1, s2, shift = scheitel(f1, f2)
    extend, scale, rot = scheitel2parameters(s1, s2)
    p = gness.pprediction(extend, shift * 180 / pi, scale, 2, rot * 180 / pi)
    imshow(p, interpolation='nearest')
    compare(f1, f2)
    xlim([0, 89])
    ylim([0, 89])
    return f1, f2, s1, s2, shift


def rotatepoint(p, angle):
    return array((p[0] * cos(angle) - p[1] * sin(angle),
                  p[0] * sin(angle) + p[1] * cos(angle)))


def to_canonical(s1, s2):
    extend, scale, rotation = scheitel2parameters(s1, s2)
    return rotatepoint(s1, rotation), rotatepoint(s2, rotation), rotation


def intersect(s1, s2, p):
    '''
    Computes the intersection of the line (0,0)->p with ellipse
    s1,s2.

    Analytical solution taken from:
    http://gieseanw.wordpress.com/2013/07/19/an-analytic-solution-for-ellipse-and-line-intersection/

    Ellipse (s1,s2) is rotated into canonical form to make it work.
    '''
    s1, s2, rotation = to_canonical(s1, s2)
    pr = rotatepoint(p, rotation)
    a, b = np.linalg.norm(s1), np.linalg.norm(s2)
    if b > a:
        a, b = b, a
    m, c = pr[1] / pr[0], 0
    asq, bsq, csq, msq = a**2, b**2, c**2, m**2
    D = sqrt(((bsq - csq) / (bsq + asq * msq)) +
             ((asq * csq * msq) / (bsq * asq + msq)**2))
    P = -(2 * asq * c * m) / (2 * (bsq + asq * msq))
    ippos = P + a * D, m * (P + a * D) + c
    ineg = P - a * D, m * (P - a * D) + c
    ippos = rotatepoint(ippos, -rotation)
    ineg = rotatepoint(ineg, -rotation)
    return ippos, ineg


def find_closest_point(s1, s2, p):
    theta = np.linspace(0, 2 * pi, 1000)
    d = dist_to_point(s1, s2, p, theta)
    idx = argmin(d)
    closest_point = point([theta[idx]], s1, s2)
    return d[idx], theta[idx], closest_point


def dist_to_point(s1, s2, p, theta):
    p = asarray(p)
    ep = point(theta, s1, s2)
    d = ((p[:, np.newaxis] - ep)**2).sum(0)
    return d**.5


def test_intersect(f1=None, f2=None, p=None):
    if f1 is None or f2 is None:
        f1 = array((28, 46))
        f2 = array((37, 61))

    s1, s2, shift = scheitel(f1, f2)
    if p is None:
        p = array((50, 50))
    compare(s1, s2)
    plot(array([p[0], 0]), array([p[1], 0]), 'ko-')
    i, ineg = intersect(s1, s2, p)
    plot(array([i[0], 0]), array([i[1], 0]), 'go-')
    plot(array([ineg[0], 0]), array([ineg[1], 0]), 'go-')
    i2, ineg = intersect(s1, s2, s1)
    plot(array([i2[0], 0]), array([i2[1], 0]), 'bd--')
    return s1, s2, p
