"""The algebraic route in JOINT variables: general for any serial revolute chain.

algebraic.py wrote the door conditions in posture variables (the elbow),
which used the S-R-S structure. Here nothing does:

  unknowns  x = (c_1..c_n, s_1..s_n, u)        c_i = cos q_i, s_i = sin q_i
  equations c_i^2 + s_i^2 - 1 = 0                     n
            tool_position(c, s) - (p0 + u A) = 0       3   (the task: a weld slice)
            M[0,2], M[1,2], M[1,0] = 0, M = R_t^T R    3   (tool orientation)
            s_j = 0                                    1   (the door: label joint j is 0)
            a c_k + b s_k = 0                          1   (fixes the free spin of the
                                                            neighbour pair at the door)

for an n = 7 chain: 15 x 15. The forward kinematics come from krawczyk.SerialChain
(joint axes and offsets), i.e. from a URDF. Its total-degree Bezout number is
in the millions, so no start system is used. Instead (the method of
HomotopyContinuation.jl):

  1. MONODROMY. Pick random complex joint values q* with s_j = 0 and DEFINE a
     parameter instance P0 for which q* is a solution (target pose := FK(q*)).
     Walk the parameters around random loops P0 -> P1 -> P2 -> P0; tracking the
     known solutions permutes them and reveals new ones. Stop when several
     loops in a row add nothing: the solution set of the generic instance.
  2. PARAMETER HOMOTOPY. Track every generic solution from P0 to each real
     instance (a weld slice), through a random complex detour. Every isolated
     solution of the real instance is the limit of a tracked path (with
     probability one); some paths go to infinity or to spurious solutions,
     which are filtered.

Completeness of step 1 is probabilistic (the stopping rule); a "trace test"
would certify it. Nothing here uses a door formula, the arm's IK, or an L.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .krawczyk import SerialChain

_AX = {"x": 0, "y": 1, "z": 2}


def _rot(axis: str, c, s):
    z, o = 0.0 * c, 1.0 + 0.0 * c
    if axis == "z":
        return np.array([[c, -s, z], [s, c, z], [z, z, o]])
    if axis == "y":
        return np.array([[c, z, s], [z, o, z], [-s, z, c]])
    return np.array([[o, z, z], [z, c, -s], [z, s, c]])


def fk_cs(chain: SerialChain, c, s):
    """Tool position and rotation from (c, s) -- polynomial in them."""
    R = np.eye(3, dtype=complex)
    p = chain.base.astype(complex)
    for i, ax in enumerate(chain.axes):
        R = R @ _rot(ax, c[i], s[i])
        if i in chain.after:
            p = p + R @ chain.after[i]
    return p, R


@dataclass
class DoorSystem:
    chain: SerialChain
    label: int                      # joint index whose sign labels the pages (s_j = 0)
    spin: int                       # joint index used to fix the null spin (a neighbour)

    @property
    def n(self) -> int:
        return len(self.chain.axes)

    def F(self, x, P):
        """P = dict(p0 (3), A (3), Rt (3,3), ab (2))."""
        n = self.n
        c, s, u = x[:n], x[n:2 * n], x[2 * n]
        T, R = fk_cs(self.chain, c, s)
        M = P["Rt"].T @ R
        out = list(c * c + s * s - 1.0)
        out += list(T - (P["p0"] + u * P["A"]))
        # Orientation: tool axis right (M[0,2] = M[1,2] = 0) and roll right
        # (M[1,0] = 0), M = R_t^T R. NOT the skew part of M: that also vanishes
        # for every half-turn about ANY axis, a 2-D continuum of spurious
        # solutions that captured every path on real targets. These three
        # leave only isolated half-turns about the frame axes (filtered later).
        out += [M[0, 2], M[1, 2], M[1, 0]]
        out += [s[self.label], P["ab"][0] * c[self.spin] + P["ab"][1] * s[self.spin]]
        return np.array(out, dtype=complex)

    def J(self, x, P, h=1e-7):
        """Central differences in the complex variables (F is a polynomial,
        so the truncation error is O(h^2) times its third derivative)."""
        f0 = self.F(x, P)
        Jm = np.zeros((len(f0), len(x)), dtype=complex)
        for k in range(len(x)):
            e = np.zeros(len(x), dtype=complex)
            e[k] = h
            Jm[:, k] = (self.F(x + e, P) - self.F(x - e, P)) / (2 * h)
        return Jm


# --------------------------------------------------------------------------
# parameter homotopy tracking
# --------------------------------------------------------------------------


def _lerp(P, Q, t):
    return {k: (1 - t) * P[k] + t * Q[k] for k in P}


LAST = {}


def track(sys: DoorSystem, x0, P, Q, max_steps=4000):
    """Track a solution of F(., P) to F(., Q) along the straight parameter
    segment. Returns the endpoint or None (LAST records how far it got)."""
    x, t, dt = np.asarray(x0, dtype=complex), 0.0, 0.05
    LAST.clear()
    for _ in range(max_steps):
        if t >= 1.0:
            break
        h = min(dt, 1.0 - t)

        def xdot(xx, tt):
            Pt = _lerp(P, Q, tt)
            dP = {k: Q[k] - P[k] for k in P}
            # dF/dt = F_p . dP: by a directional difference in the parameters.
            eps = 1e-7
            fp = (sys.F(xx, _lerp(Pt, {k: Pt[k] + eps * dP[k] for k in P}, 1.0))
                  - sys.F(xx, _lerp(Pt, {k: Pt[k] - eps * dP[k] for k in P}, 1.0))) / (2 * eps)
            return -np.linalg.solve(sys.J(xx, Pt), fp)
        try:
            k1 = xdot(x, t)
            k2 = xdot(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = xdot(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = xdot(x + h * k3, t + h)
        except np.linalg.LinAlgError:
            dt *= 0.5
            if dt < 1e-12:
                return None
            continue
        y = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        Pn = _lerp(P, Q, t + h)
        ok = False
        for _c in range(4):
            try:
                d = np.linalg.solve(sys.J(y, Pn), sys.F(y, Pn))
            except np.linalg.LinAlgError:
                break
            y = y - d
            if np.linalg.norm(d) < 1e-10 * (1 + np.linalg.norm(y)):
                ok = True
                break
        if ok:
            x, t = y, t + h
            dt = min(1.5 * dt, 0.1)
        else:
            dt *= 0.5
            if dt < 1e-12:
                return None
        LAST.update(t=t, x=x)
        if np.linalg.norm(x) > 1e7:
            return None
    if t < 1.0:
        return None
    for _ in range(6):
        try:
            x = x - np.linalg.solve(sys.J(x, Q), sys.F(x, Q))
        except np.linalg.LinAlgError:
            return None
    return x if np.linalg.norm(sys.F(x, Q)) < 1e-8 else None


def _rand_params(rng, scale=1.0):
    def cz(*shape):
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return dict(p0=cz(3), A=cz(3), Rt=np.eye(3) + 0.5 * cz(3, 3), ab=cz(2))


def _dedupe(sols, tol=1e-6):
    out = []
    for x in sols:
        if all(np.linalg.norm(x - y) > tol * (1 + np.linalg.norm(x)) for y in out):
            out.append(x)
    return out


@dataclass
class MonodromyResult:
    base: dict
    solutions: list
    loops: int
    history: list = field(default_factory=list)


def monodromy_solve(sys: DoorSystem, rng=None, stable_loops=5, max_loops=60,
                    verbose=False) -> MonodromyResult:
    rng = np.random.default_rng(1) if rng is None else rng
    n = sys.n
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    c, s = np.cos(z), np.sin(z)
    c[sys.label], s[sys.label] = 1.0, 0.0
    u = complex(rng.normal() + 1j * rng.normal())
    T, R = fk_cs(sys.chain, c, s)
    A = rng.normal(size=3) + 1j * rng.normal(size=3)
    ab = rng.normal(size=2) + 1j * rng.normal(size=2)
    # make the spin equation hold: choose b from a
    ab[1] = -ab[0] * c[sys.spin] / s[sys.spin]
    P0 = dict(p0=T - u * A, A=A, Rt=R, ab=ab)
    x0 = np.concatenate([c, s, [u]])
    assert np.linalg.norm(sys.F(x0, P0)) < 1e-9
    sols = [x0]
    calm, loops, hist = 0, 0, []
    while calm < stable_loops and loops < max_loops:
        loops += 1
        P1, P2 = _rand_params(rng), _rand_params(rng)
        found = []
        for x in sols:
            y = track(sys, x, P0, P1)
            y = track(sys, y, P1, P2) if y is not None else None
            y = track(sys, y, P2, P0) if y is not None else None
            if y is not None:
                found.append(y)
        before = len(sols)
        sols = _dedupe(sols + found)
        hist.append(len(sols))
        calm = calm + 1 if len(sols) == before else 0
        if verbose:
            print(f"[monodromy] loop {loops}: {len(sols)} solutions")
    return MonodromyResult(base=P0, solutions=sols, loops=loops, history=hist)


def solve_instance(sys: DoorSystem, mono: MonodromyResult, P_real: dict, rng=None) -> list:
    """Every solution of the real instance: track all generic solutions from
    the base through a random complex detour."""
    rng = np.random.default_rng(2) if rng is None else rng
    Pmid = _rand_params(rng)
    out = []
    for x in mono.solutions:
        y = track(sys, x, mono.base, Pmid)
        y = track(sys, y, Pmid, P_real) if y is not None else None
        if y is not None:
            out.append(y)
    return _dedupe(out)


def task_instance(surf, v: float, ab=(0.37, 0.81)) -> dict:
    """The real parameter instance of a weld-patch slice v (any fixed ab with
    a, b != 0 picks one spin root; the other is q_k + pi, deduped below)."""
    return dict(p0=(surf.p0 + v * surf.dv).astype(complex), A=surf.du.astype(complex),
                Rt=surf.rot.astype(complex), ab=np.asarray(ab, dtype=complex))


def real_doors(sys: DoorSystem, sols, P: dict, imag_tol=1e-6, rot_tol=1e-6) -> list:
    """Door configurations among the endpoints of a real instance: real,
    u in (0, 1), the tool rotation really equal to R_t (drops the isolated
    half-turn solutions of the three orientation equations), q_label = 0 (not
    pi). Returns (u, q) pairs, sorted by u. One door POINT of the page has
    several joint spellings here (the other sign flips, and the two spin
    roots q_k, q_k + pi): group them by (u, arm angle) to compare with a chart."""
    n = sys.n
    Rt = np.asarray(P["Rt"]).real
    out = []
    for x in sols:
        if np.max(np.abs(x.imag)) > imag_tol:
            continue
        x = x.real
        c, s, u = x[:n], x[n:2 * n], float(x[2 * n])
        if not (0.0 < u < 1.0) or c[sys.label] < 0:
            continue
        _, R = fk_cs(sys.chain, c, s)
        if np.linalg.norm(Rt.T @ R.real - np.eye(3)) > rot_tol:
            continue
        q = np.arctan2(s, c)
        if any(abs(u - v) < 1e-7 and np.all(np.abs(np.angle(np.exp(1j * (q - p)))) < 1e-6)
               for v, p in out):
            continue
        out.append((u, q))
    return sorted(out, key=lambda t: t[0])
