"""Vectorised interval arithmetic with outward rounding (for the Krawczyk door test).

An Interval holds arrays `lo`, `hi` of one shape; every operation returns an
interval guaranteed to contain the exact result for EVERY choice of inputs in
the operand intervals. Floating-point error is covered by widening each
result outward:

  +, -, *  correctly rounded in IEEE 754 (error <= 1/2 ulp): one ulp outward
           (np.nextafter) is enough;
  sums     of n terms: by gamma_n * sum|terms| (Higham, Thm 4.4), gamma_n =
           n u / (1 - n u), applied once to the whole sum;
  sin, cos, sqrt: libm is not guaranteed correctly rounded, so these widen
           by 4 ulp of the magnitude plus 1e-300 -- far above the <= 1 ulp
           error of glibc's implementations.

Only what the Krawczyk test needs is implemented. Point (float) operands
are accepted wherever an interval is.
"""

from __future__ import annotations

import numpy as np

_U = np.finfo(float).eps / 2.0          # unit roundoff
_TINY = 1e-300


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _wide(lo, hi, k: float = 8.0):
    """Widen by k unit roundoffs of the magnitude (transcendentals)."""
    m = np.maximum(np.abs(lo), np.abs(hi))
    return lo - k * _U * m - _TINY, hi + k * _U * m + _TINY


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo.copy() if hi is None else np.asarray(hi, dtype=float)

    # -- construction / shape ----------------------------------------------

    @staticmethod
    def of(x) -> "Interval":
        return x if isinstance(x, Interval) else Interval(x)

    @staticmethod
    def around(mid, rad) -> "Interval":
        mid, rad = np.asarray(mid, dtype=float), np.asarray(rad, dtype=float)
        return Interval(_down(mid - rad), _up(mid + rad))

    @property
    def shape(self):
        return self.lo.shape

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self):
        return _up(0.5 * (self.hi - self.lo))

    def mag(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def contains(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (self.lo <= x) & (x <= self.hi)

    def __getitem__(self, k) -> "Interval":
        return Interval(self.lo[k], self.hi[k])

    def __setitem__(self, k, v):
        v = Interval.of(v)
        self.lo[k] = v.lo
        self.hi[k] = v.hi

    def __repr__(self):
        return f"Interval({self.lo!r}, {self.hi!r})"

    # -- arithmetic --------------------------------------------------------

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __add__(self, o):
        o = Interval.of(o)
        return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __sub__(self, o):
        o = Interval.of(o)
        return Interval(_down(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, o):
        return Interval.of(o) - self

    def __mul__(self, o):
        o = Interval.of(o)
        p = np.stack(np.broadcast_arrays(self.lo * o.lo, self.lo * o.hi,
                                         self.hi * o.lo, self.hi * o.hi))
        return Interval(_down(p.min(axis=0)), _up(p.max(axis=0)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Interval.of(o)
        bad = (o.lo <= 0.0) & (o.hi >= 0.0)
        inv_lo = np.where(bad, -np.inf, _down(1.0 / np.where(bad, 1.0, o.hi)))
        inv_hi = np.where(bad, np.inf, _up(1.0 / np.where(bad, 1.0, o.lo)))
        r = self * Interval(inv_lo, inv_hi)
        # 0 * inf is nan: an unbounded quotient
        r.lo = np.where(np.isnan(r.lo), -np.inf, r.lo)
        r.hi = np.where(np.isnan(r.hi), np.inf, r.hi)
        return r

    def sqr(self):
        a, b = np.abs(self.lo), np.abs(self.hi)
        zero = (self.lo <= 0.0) & (self.hi >= 0.0)
        lo = np.where(zero, 0.0, _down(np.minimum(a, b) ** 2))
        return Interval(lo, _up(np.maximum(a, b) ** 2))

    def sqrt(self):
        lo, hi = _wide(np.sqrt(np.maximum(self.lo, 0.0)), np.sqrt(np.maximum(self.hi, 0.0)))
        return Interval(np.maximum(lo, 0.0), hi)

    def sin(self):
        return _sin(self.lo, self.hi)

    def cos(self):
        # cos x = sin(x + pi/2); the shift is a rounded sum, so widen it by
        # a few ulps of its magnitude.
        e = 4.0 * _U * (np.maximum(np.abs(self.lo), np.abs(self.hi)) + 2.0)
        return _sin(self.lo + np.pi / 2.0 - e, self.hi + np.pi / 2.0 + e)

    def abs(self):
        zero = (self.lo <= 0.0) & (self.hi >= 0.0)
        a, b = np.abs(self.lo), np.abs(self.hi)
        return Interval(np.where(zero, 0.0, np.minimum(a, b)), np.maximum(a, b))


def _sin(lo, hi) -> Interval:
    """sin over [lo, hi]: the endpoint values, extended to +-1 wherever the
    interval contains a crest (pi/2 + 2k pi) or a trough (-pi/2 + 2k pi)."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    a, b = np.sin(lo), np.sin(hi)
    rlo, rhi = _wide(np.minimum(a, b), np.maximum(a, b))
    k_max = np.ceil((lo - np.pi / 2.0) / (2.0 * np.pi))
    has_max = np.pi / 2.0 + 2.0 * np.pi * k_max <= hi
    k_min = np.ceil((lo + np.pi / 2.0) / (2.0 * np.pi))
    has_min = -np.pi / 2.0 + 2.0 * np.pi * k_min <= hi
    full = (hi - lo) >= 2.0 * np.pi
    rhi = np.where(has_max | full, 1.0, np.minimum(rhi, 1.0))
    rlo = np.where(has_min | full, -1.0, np.maximum(rlo, -1.0))
    return Interval(rlo, rhi)


def isum(x: Interval, axis: int) -> Interval:
    """Sum along an axis, with the summation error bounded once."""
    n = x.lo.shape[axis]
    g = 1.01 * n * _U / (1.0 - n * _U)
    slo, shi = x.lo.sum(axis=axis), x.hi.sum(axis=axis)
    m = np.maximum(np.abs(x.lo), np.abs(x.hi)).sum(axis=axis)
    return Interval(slo - g * m - _TINY, shi + g * m + _TINY)


def imatmul(a, b) -> Interval:
    """(..., n, k) @ (..., k, m) for intervals or points."""
    a, b = Interval.of(a), Interval.of(b)
    prod = Interval(a.lo[..., :, :, None], a.hi[..., :, :, None]) * \
        Interval(b.lo[..., None, :, :], b.hi[..., None, :, :])
    return isum(prod, axis=-2)


def imatvec(a, v) -> Interval:
    """(..., n, k) @ (..., k)."""
    a, v = Interval.of(a), Interval.of(v)
    prod = a * Interval(v.lo[..., None, :], v.hi[..., None, :])
    return isum(prod, axis=-1)


def idot(a, b) -> Interval:
    a, b = Interval.of(a), Interval.of(b)
    return isum(a * b, axis=-1)


def icross(a, b) -> Interval:
    a, b = Interval.of(a), Interval.of(b)

    def c(i):
        return a[..., i]

    def d(i):
        return b[..., i]
    x = c(1) * d(2) - c(2) * d(1)
    y = c(2) * d(0) - c(0) * d(2)
    z = c(0) * d(1) - c(1) * d(0)
    return Interval(np.stack([x.lo, y.lo, z.lo], axis=-1),
                    np.stack([x.hi, y.hi, z.hi], axis=-1))


def istack(parts, axis=-1) -> Interval:
    parts = [Interval.of(p) for p in parts]
    return Interval(np.stack(np.broadcast_arrays(*[p.lo for p in parts]), axis=axis),
                    np.stack(np.broadcast_arrays(*[p.hi for p in parts]), axis=axis))
