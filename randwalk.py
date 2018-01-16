import numpy as np
from pylab import normpdf
from scipy.ndimage import gaussian_filter, rotate
import pylab as pp
from pylab import *
from matplotlib import animation


class PathWalker(object):

    def __init__(self, tour, speed, start=(0, 0)):
        self.tour = self.align_tour(tour, start)
        self.tour.append(start)
        self.cur_target = tour[1]
        self.speed = speed
        self.x, self.y = [start[0]], [start[1]]
        self.i = 1
        self.update_angle()

    def align_tour(self, tour, start=(0, 0)):
        '''
        Find starting location and make this the first entry
        in the tour. 
        '''
        idx = [j for j, i in enumerate(tour) if i == start][0]
        tour = tour[idx:] + tour[:idx]
        return tour

    def next(self):
        if self.update():
            self.i += 1
            self.cur_target = self.tour[self.i]
            self.update_angle()

        self.x.append(self.x[-1] + cos(self.alpha) * self.scale)
        self.y.append(self.y[-1] + sin(self.alpha) * self.scale)

    def update_angle(self):
        x0, y0 = self.tour[self.i - 1][0], self.tour[self.i - 1][1]
        x1, y1 = self.tour[self.i][0], self.tour[self.i][1]

        self.alpha = np.arctan2(y1 - y0, x1 - x0)
        h = ((x1 - x0)**2 + (y1 - y0)**2)
        self.scale = self.speed

    def update(self):
        d = ((self.tour[self.i][0] - self.x[-1])**2 +
             (self.tour[self.i][1] - self.y[-1])**2)**.5
        if d < 1 * self.speed:
            self.x[-1] = self.tour[self.i][0]
            self.y[-1] = self.tour[self.i][1]
            return True
        else:
            return False


def make_trajectory(tour, out,  speed=4, smoothing=60):
    '''
    '''
    w = PathWalker(tour, speed / 120.0)
    while True:
        try:
            w.next()
        except IndexError:
            break

    x = gaussian_filter(w.x, smoothing)
    y = gaussian_filter(w.y, smoothing)
    f = open(out, 'w')
    f.write('%d\n' % len(x))
    for xx, yy in zip(x, y):
        f.write('%f\n' % xx)
        f.write('%f\n' % yy)
    return x, y


class GridWalker(object):

    def __init__(self):
        self.x, self.y = [0], [0]
        self.tx, self.ty = [], []
        self.yy, self.xx = np.mgrid[-10:10:21j, -10:10:21j]
        self.targets = list(zip(self.xx.flatten(), self.yy.flatten()))
        np.random.shuffle(self.targets)
        self.pick_target()

    def pick_target(self):
        if len(self.targets) == 0:
            raise StopIteration

        x, y = self.targets.pop()
        self.tx.append(x)
        self.ty.append(y)

    def check_targets(self):
        tlist = []
        for x, y in self.targets:
            d = (((x - self.x[-1])**2 + (y - self.y[-1])**2)**.5)
            if d > .5:
                tlist.append((x, y))
        self.targets = tlist

    def next(self):
        speed = 2.0 / 120.0
        angle = np.arctan2(self.ty[-1] - self.y[-1], self.tx[-1] - self.x[-1])
        angle = reshift((angle * 180 / np.pi))
        x, y = (speed * np.cos(angle * np.pi / 180.0),
                speed * np.sin(angle * np.pi / 180.0))
        self.x.append(x + self.x[-1])
        self.y.append(y + self.y[-1])
        dist = ((self.tx[-1] - self.x[-1])**2 +
                (self.ty[-1] - self.y[-1])**2)**.5
        if dist < 1.0:
            self.pick_target()
        self.check_targets()


class Walker(object):

    def __init__(self, start, bias):
        self.x, self.y = [], []
        self.tx, self.ty = [], []
        self.x.append(start[0])
        self.y.append(start[1])
        self.bias = bias
        self.bias[self.y[-1], self.x[-1]] += 150
        self.step_size = 5
        z = normpdf(np.linspace(-100, 100, 100), 0, 5)
        t1 = np.dot(z[:, np.newaxis], z[:, np.newaxis].T)
        z = normpdf(np.linspace(-100, 100, 100), 0, 10)
        t2 = np.dot(z[:, np.newaxis], z[:, np.newaxis].T)
        self.step = (t2 / t2.max()) - (t1 / t1.max())
        self.step = self.step - self.step.min()
        self.step = self.step / self.step.sum()
        self.t = None
        self.pick_target()

    def pick_target(self):
        d1, d2 = self.bias.shape
        m = gaussian_filter(self.bias, 15, mode='constant')
        m = pdf(m)
        m = 1 - m
        mask(m)
        m = pdf(m)
        # if len(self.tx)>1:
        #    y,x = np.mgrid[0:d1, 0:d2]
        #    angle = np.arctan2(self.ty[-1]-self.ty[-2], self.tx[-1]-self.tx[-2])
        #    t = np.arctan2(x-self.tx[-1], y-self.ty[-1])*180/np.pi
        #    t = reshift(t-90-angle*180/np.pi)
        #    m = (t>0)*m
        #    self.t = t
        x, y = sample2d(pdf(m))
        self.tx.append(x)
        self.ty.append(y)

    def p_next(self):
        d = self.step.shape[0] / 2
        cutout = self.bias[self.y[-1] - d:self.y[-1] +
                           d, self.x[-1] - d:self.x[-1] + d]
        idzero = cutout < 0
        self.k = cutout.copy()
        cutout = gaussian_filter(cutout, 15, mode='constant')
        cutout = pdf(cutout)
        cutout = 1 - cutout
        cutout = cutout - cutout.min()
        cutout[idzero] = 0
        cutout = cutout / cutout.sum()
        cutout[idzero] = 0
        target = self.p_target()
        t = cutout * self.step * target
        t = pdf(t)
        tmp = np.zeros(self.bias.shape)
        tmp[self.y[-1] - d:self.y[-1] + d, self.x[-1] - d:self.x[-1] + d] += t
        self.t = t
        dist = ((self.tx[-1] - self.x[-1])**2 +
                (self.ty[-1] - self.y[-1])**2)**.5
        if dist < 30:
            self.pick_target()
        t[idzero] = 0
        return t

    def p_target(self):
        x, y = np.mgrid[-100:100:100j, -100:100:100j]
        id0 = y < 0
        t = normpdf(x, 0, abs(y) / 2.5 + 1)
        t[id0] = 0
        angle = -1 * np.arctan2(self.ty[-1] -
                                self.y[-1], self.tx[-1] - self.x[-1])
        t = rotate(t, reshift((angle * 180 / np.pi)))
        dt = self.step.shape[0]
        d0 = t.shape[0]
        t = t[(d0 / 2) - (dt / 2):(d0 / 2) + (dt / 2),
              (d0 / 2) - (dt / 2):(d0 / 2) + (dt / 2)]
        t = t / t.sum()
        return t

    def next(self):
        d = self.step.shape[0] / 2
        t = self.p_next()
        x, y = sample2d(t)
        x, y = x - d + self.x[-1], y - d + self.y[-1]
        k = np.ones(self.bias.shape)
        mask(k)
        if k[y, x] < 0:
            1 / 0
        self.bias[y, x] += 150
        self.x.append(x)
        self.y.append(y)
        self.tx.append(self.tx[-1])
        self.ty.append(self.ty[-1])


def pdf(x):
    x -= x.min()
    return x / x.sum()


def sample2d(distpdf):
    d1, d2 = distpdf.shape
    cdall = distpdf.cumsum(1)
    cdlast = (cdall[:, d2 - 1]).cumsum()
    cdlast = cdlast / cdlast.max()
    r1 = np.random.rand()
    y = np.where(cdlast > r1)[0][0]
    cdrow = cdall[y, :]
    cdrow = cdrow / cdrow.max()
    r2 = np.random.rand()
    x = np.where(cdrow > r2)[0][0]
    return x, y


def check_coverage():
    x, y, tx, ty = [], [], [], []
    b = bias()
    ion()
    figure()
    subplot(1, 2, 1)
    xlim((200, 800))
    ylim((200, 800))
    for k in range(6):
        sx = np.random.randint(300, 900)
        sy = np.random.randint(300, 500)
        w = Walker((sx, sy), b.copy())
        for r in range(500):
            w.next()
            if r % 2 == 0:
                subplot(1, 2, 1)
                plot(w.x, w.y, 'b')
                print w.x[-10:]
                plot(w.x[-1], w.y[-1], 'ro')
                plot(w.tx, w.ty, 'ko')
                draw()
                subplot(1, 2, 2)
                imshow(w.p_next())
                draw()
                # waitforbuttonpress()
        x.append(gaussian_filter(w.x, 10))
        y.append(gaussian_filter(w.y, 10))
        tx.append(copy(w.tx))
        ty.append(copy(w.ty))
        b = w.bias.copy()
    return x, y, tx, ty, w


def reshift(I):
    # Output -180 to +180
    if type(I) == list:
        I = np.array(I)

    if type(I) == np.ndarray:
        while((I > 180).sum() > 0 or (I < -180).sum() > 0):
            I[I > 180] = I[I > 180] - 360
            I[I < -180] = I[I < -180] + 360

    if (type(I) == int or type(I) == np.float64 or
            type(I) == float or type(I) == np.float):
        while (I > 180 or I < -180):
            if I > 180:
                I -= 360
            if I < -180:
                I += 360
    return I


def mask(x):
    x[:200, :] = -10
    x[800:, :] = -10
    x[:, :200] = -10
    x[:, 1000:] = -10


def bias():
    b = np.ones((1000, 1200))
    mask(b)
    return b

xx = []
yy = []
tx = []
ty = []
ax = None


def setup_fig(x, y, txx, tyy):
    global xx, yy, ax, tx, ty
    xx, yy = x, y
    tx, ty = txx, tyy
    f = pp.figure()
    ax = pp.axes(xlim=(-12, 12), ylim=(-12, 12))
    return f


def animate(i):
    try:
        ax.cla()
        a = ax.plot(xx[:i], yy[:i], 'b')
        #c = ax.plot(tx[cnt][:i%l], ty[cnt][:i%l], 'ko')
        ax.set_ylim(-12, 12)
        ax.set_xlim(-12, 12)
        b = ax.plot(xx[i], yy[i], 'ro')
        return a + b
    except IndexError:
        print cnt, i % l, i
        raise StopIteration


def make_movie(x, y, tx, ty):
    f = setup_fig(x, y, tx, ty)
    ani = animation.FuncAnimation(f, animate,
                                  interval=1 / 60.0, frames=len(x), blit=True, repeat=True)
    return ani
    # ani.save('t.mp4',fps=120)# extra_args=['-vcodec', 'libx264'])
