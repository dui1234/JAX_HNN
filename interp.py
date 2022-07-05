from functools import partial
import jax.numpy as np
import jax
from jax import vmap


_first_axis_last = lambda x: np.moveaxis(x, 0, -1)
_last_axis_first = lambda x: np.moveaxis(x, -1, 0)


def _shuffle_output(generator):
    def newgenerator(xp):
        func = generator(xp)

        def newfunc(yp, x):
            yp = _first_axis_last(yp)
            return _last_axis_first(func(yp, x))

        return newfunc

    return newgenerator


def _linear(xp):
    def func(yp, x):
        x = np.array(x)
        j = np.searchsorted(xp, x)  # find index where xs should be inserted to maintain order
        j = np.clip(j, 1, len(xp) - 1)
        x0, x1 = xp[j - 1], xp[j]  # bracketing interval
        t = (x - x0) / (x1 - x0)  # weight
        yp = yp
        y = (1 - t) * yp[..., j - 1] + t * yp[..., j]  # linear interpolation
        return y

    return func


linear = _linear
vlinear = _shuffle_output(_linear)


def _cubic(xp):
    h00 = lambda t: (1 + 2 * t) * (1 - t) ** 2
    h10 = lambda t: t * (1 - t) ** 2
    h01 = lambda t: t**2 * (3 - 2 * t)
    h11 = lambda t: t**2 * (t - 1)

    def slope(yp):
        mi = 0.5 * (np.roll(yp, -1, axis=-1) - yp) / (np.roll(xp, -1) - xp)
        mi += 0.5 * (yp - np.roll(yp, 1, axis=-1)) / (xp - np.roll(xp, 1))
        mi = mi.at[..., 0].set((yp[..., 1] - yp[..., 0]) / (xp[1] - xp[0]))
        mi = mi.at[..., -1].set((yp[..., -1] - yp[..., -2]) / (xp[-1] - xp[-2]))
        return mi

    def func(yp, x):
        x = np.array(x)
        j = np.searchsorted(xp, x)
        j = np.clip(j, 1, len(xp) - 1)
        x0, x1 = xp[j - 1], xp[j]
        t = (x - x0) / (x1 - x0)
        mi = slope(yp)
        y = (
            h00(t) * yp[..., j - 1]
            + h10(t) * (x1 - x0) * mi[..., j - 1]
            + h01(t) * yp[..., j]
            + h11(t) * (x1 - x0) * mi[..., j]
        )
        return y

    return func


cubic = _cubic
vcubic = _shuffle_output(_cubic)


def _lagrange(xp):
    def scan_fun(carry, x):
        i, xs = carry
        return (i + 1, np.roll(xs, -1)), np.prod((x - xs[1:]) / (xs[0] - xs[1:]))

    def scan_vfun(carry, x):
        i, xs = carry
        return (i + 1, np.roll(xs, -1)), np.prod((x[:, None] - xs[1:]) / (xs[0] - xs[1:]), axis=1)

    def fun(yp, x):
        x = np.array(x)
        if x.ndim == 0:
            _, lp = jax.lax.scan(scan_fun, (0, xp), np.broadcast_to(x, xp.shape))
        else:
            _, lp = jax.lax.scan(scan_vfun, (0, xp), np.broadcast_to(x, xp.shape + x.shape))

        return np.dot(yp, lp)

    return fun


lagrange = _lagrange
vlagrange = _shuffle_output(_lagrange)


def _lagrange_mapping(gen):
    def mgen(xp):
        a, b = xp[0], xp[-1]
        S = lambda z: (a - b) * 2 * np.cos(np.pi * (z - a) / (b - a)) + (a + b) / 2
        func = gen(S(xp))

        def mfunc(yp, x):
            return func(yp, S(x))

        return mfunc

    return mgen


lagrangeX = _lagrange_mapping(lagrange)
vlagrangeX = _lagrange_mapping(vlagrange)
