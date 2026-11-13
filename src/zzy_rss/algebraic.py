"""The algebraic route to the doors: polynomial systems + homotopy continuation.

Finding doors with sigma needs a guaranteed speed limit L to be complete. A
polynomial system needs none: homotopy continuation tracks one path from
EVERY root of a start system whose root count bounds the target's (Bezout),
so for a generic complex gamma ("gamma trick") every isolated solution of
the target is the end of some path, with probability one. No starting guess,
no sampling, no Lipschitz bound.

THE SYSTEMS, in posture variables (S-R-S arm, affine task, slice v fixed):

  q2 door: the upper arm lies on the joint-1 axis (joints 1 and 3 coaxial),
           i.e. the elbow is on the shoulder's vertical line.
             unknowns  x = (u, Ex, Ey, Ez)
             Ex - Sx = 0,  Ey - Sy = 0                       (elbow on the axis)
             |E - S|^2 - d_se^2 = 0                           (upper arm length)
             |E - W(u, v)|^2 - d_ew^2 = 0                     (forearm length)
           total degree 1*1*2*2 = 4 paths.
  q6 door: the forearm lies on the tool axis (joints 5 and 7 coaxial).
             unknowns  x = (u, lam, Ex, Ey, Ez)
             E - W(u, v) + lam t = 0                         (3 eqs, forearm along t)
             lam^2 - d_ew^2 = 0,  |E - S|^2 - d_se^2 = 0
           total degree 1*1*1*2*2 = 4 paths.

Real solutions with u in [0, 1] are the door points of the slice; the arm
angle theta follows from the elbow (SrsChart's convention). E = S - d_se e_z
(q2 = pi) and lam = -d_ew (q6 = pi) are the other spellings of the same
coaxiality and are reported separately.

What is general here and what is not: the homotopy solver is general (any
square polynomial system). Writing the door conditions as "two joint axes
coaxial" in POSTURE variables used the S-R-S structure; that is why the
systems have 4 paths. In joint variables (c_i = cos q_i, s_i = sin q_i,
c_i^2 + s_i^2 = 1) the same conditions are generic for any serial arm, but
the Bezout count is in the thousands to millions -- a job for a dedicated
solver (HomotopyContinuation.jl, Bertini), not this numpy one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .chart import SrsChart


# --------------------------------------------------------------------------
# a small total-degree homotopy solver (complex, gamma trick)
# --------------------------------------------------------------------------


@dataclass
class HomotopyResult:
    solutions: np.ndarray          # (k, n) complex endpoints that converged
    paths: int
    diverged: int
    failed: int


def solve_total_degree(F, J, degrees, rng=None, tol=1e-12, max_steps=20000) -> HomotopyResult:
    """All isolated solutions of F(x) = 0 (square, polynomial), by tracking
    H(x, t) = (1 - t) gamma G(x) + t F(x) from every root of
    G_i(x) = x_i^{d_i} - 1. F, J: callables on complex vectors."""
    rng = np.random.default_rng(0) if rng is None else rng
    deg = np.asarray(degrees, dtype=int)
    n = len(deg)
    gamma = np.exp(2j * np.pi * rng.uniform())

    def G(x):
        return x ** deg - 1.0

    def GJ(x):
        return np.diag(deg * x ** (deg - 1))

    def H(x, t):
        return (1 - t) * gamma * G(x) + t * F(x)

    def Hx(x, t):
        return (1 - t) * gamma * GJ(x) + t * J(x)

    def Ht(x):
        return F(x) - gamma * G(x)

    def xdot(x, t):
        return -np.linalg.solve(Hx(x, t), Ht(x))

    starts = [np.array(c) for c in product(*[np.exp(2j * np.pi * np.arange(d) / d) for d in deg])]
    sols, diverged, failed = [], 0, 0
    for x0 in starts:
        x, t, dt = x0.astype(complex), 0.0, 0.02
        ok = True
        for _ in range(max_steps):
            if t >= 1.0:
                break
            h = min(dt, 1.0 - t)
            try:   # RK4 predictor
                k1 = xdot(x, t)
                k2 = xdot(x + 0.5 * h * k1, t + 0.5 * h)
                k3 = xdot(x + 0.5 * h * k2, t + 0.5 * h)
                k4 = xdot(x + h * k3, t + h)
            except np.linalg.LinAlgError:
                dt *= 0.5
                continue
            y = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            conv = False
            for _c in range(4):   # Newton corrector at t + h
                try:
                    d = np.linalg.solve(Hx(y, t + h), H(y, t + h))
                except np.linalg.LinAlgError:
                    break
                y = y - d
                if np.linalg.norm(d) < 1e-10 * (1 + np.linalg.norm(y)):
                    conv = True
                    break
            if conv:
                x, t = y, t + h
                dt = min(2.0 * dt, 0.1)
            else:
                dt *= 0.5
                if dt < 1e-14:
                    ok = False
                    break
            if np.linalg.norm(x) > 1e8:
                break
        if np.linalg.norm(x) > 1e8:
            diverged += 1
            continue
        if not ok or t < 1.0:
            failed += 1
            continue
        for _ in range(8):   # polish on F
            try:
                x = x - np.linalg.solve(J(x), F(x))
            except np.linalg.LinAlgError:
                break
        if np.linalg.norm(F(x)) < 1e-8:
            sols.append(x)
        else:
            failed += 1
    return HomotopyResult(np.asarray(sols).reshape(-1, n), len(starts), diverged, failed)


# --------------------------------------------------------------------------
# the door systems of an S-R-S chart, slice by slice
# --------------------------------------------------------------------------


@dataclass
class SliceDoors:
    v: float
    points: list                   # dicts: kind, u, theta, spelling, residual
    paths: int
    failed: int


def _arm_angle(chart: SrsChart, W: np.ndarray, E: np.ndarray) -> float:
    S = chart.S
    u = (W - S) / np.linalg.norm(W - S)
    v0 = np.array([0.0, 0.0, 1.0]) - u[2] * u
    v0 /= np.linalg.norm(v0)
    w0 = np.cross(u, v0)
    e = E - S - np.dot(E - S, u) * u
    return float(np.arctan2(np.dot(e, w0), np.dot(e, v0)))


def doors_on_slice(chart: SrsChart, v: float, rng=None, imag_tol=1e-8) -> SliceDoors:
    """Every q2 and q6 door point of the slice v, by homotopy continuation."""
    S = chart.S.astype(complex)
    A = chart.task.A.astype(complex)
    w0 = chart.w0.astype(complex)
    t = chart.t.astype(complex)
    dse, dew = chart.d_se, chart.d_ew

    def W(u):
        return w0 + A[:, 0] * u + A[:, 1] * v

    # q2: x = (u, Ex, Ey, Ez)
    def F2(x):
        u, E = x[0], x[1:]
        return np.array([E[0] - S[0], E[1] - S[1], np.sum((E - S) ** 2) - dse ** 2,
                         np.sum((E - W(u)) ** 2) - dew ** 2])

    def J2(x):
        u, E = x[0], x[1:]
        J = np.zeros((4, 4), dtype=complex)
        J[0, 1] = 1.0
        J[1, 2] = 1.0
        J[2, 1:] = 2 * (E - S)
        J[3, 0] = -2 * np.dot(E - W(u), A[:, 0])
        J[3, 1:] = 2 * (E - W(u))
        return J

    # q6: x = (u, lam, Ex, Ey, Ez)
    def F6(x):
        u, lam, E = x[0], x[1], x[2:]
        r = E - W(u) + lam * t
        return np.array([r[0], r[1], r[2], lam ** 2 - dew ** 2, np.sum((E - S) ** 2) - dse ** 2])

    def J6(x):
        u, lam, E = x[0], x[1], x[2:]
        J = np.zeros((5, 5), dtype=complex)
        J[0:3, 0] = -A[:, 0]
        J[0:3, 1] = t
        J[0:3, 2:] = np.eye(3)
        J[3, 1] = 2 * lam
        J[4, 2:] = 2 * (E - S)
        return J

    pts, paths, failed = [], 0, 0
    for kind, F, J, deg in (("q2", F2, J2, [1, 1, 2, 2]), ("q6", F6, J6, [1, 1, 1, 2, 2])):
        res = solve_total_degree(F, J, deg, rng=rng)
        paths += res.paths
        failed += res.failed
        for x in res.solutions:
            if np.max(np.abs(x.imag)) > imag_tol:
                continue
            x = x.real
            u = float(x[0])
            if not (0.0 <= u <= 1.0):
                continue
            E = x[1:] if kind == "q2" else x[2:]
            Wr = (w0 + A[:, 0] * u + A[:, 1] * v).real
            if kind == "q2":
                spelling = "q2=0" if E[2] > chart.S[2] else "q2=pi"
            else:
                spelling = "q6=0" if x[1] > 0 else "q6=pi"
            pts.append(dict(kind=kind, u=u, theta=_arm_angle(chart, Wr, E),
                            spelling=spelling,
                            residual=float(np.linalg.norm(F(x.astype(complex))))))
    return SliceDoors(v=v, points=pts, paths=paths, failed=failed)
