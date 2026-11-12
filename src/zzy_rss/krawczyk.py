"""Method C: certify "no door in this cell" with the Krawczyk test (docs/recursive_proof_design.md §10.4: sound, but only for small cells).

THE THEOREM. A page of Y+ is a chart: page coordinates z = (x, theta) plus a
sign label s fix one configuration q = IK_s(z). A door is a place where the
label of the configuration changes while the robot moves continuously -- for
the S-R-S arm, where a labelled joint (q2, q4, q6) passes through 0. Write the
IK as a square system

        F(q, z) = 0,   F = (tool position - p(z),             3
                            orientation error vs R_target,      3
                            elbow off the theta direction)      1

If, for a joint box Q around q_c = IK_s(z_c), the Krawczyk operator

        K = q_c - Y F(q_c, Z) + (I - Y J_q(Q, Z)) (Q - q_c),   Y ~ J_q(q_c, z_c)^-1

lies strictly inside Q, then (Krawczyk 1969; Moore; Rump) for EVERY z in the
cell Z there is exactly one solution q(z) in Q, it lies in K, and every
matrix in J_q(Q, Z) is nonsingular -- the solutions form one smooth branch.
If moreover no labelled joint can be 0 in K, then the label is constant on
that branch, so the branch IS the page's IK over the whole cell (the IK with
a given label is unique) and the page has no door in the cell. That is
exactly the certificate the proof needs, and it uses only:

  * FORWARD kinematics of a serial chain (joint axes and link offsets), in
    interval arithmetic -- the same code for any arm;
  * the definition of the extra parameter (here the arm angle theta, through
    the target wrist), and which joints carry the labels.

No IK formula, no door formula, no hand-derived bound. Near a door the test
fails (J_q is singular there), the cell is split, and the surviving cells
cluster on the doors -- interior.find_doors does the rest unchanged.

Cells with no IK at the centre are cleared only if the reach annulus
certifiably excludes the whole cell (no configuration, no door); cells off
the task box are cleared as before (invariant 37).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arm import SrsArm
from .chart import I_COORD, I_Q2, I_Q4, I_Q6, BallTest, SrsChart
from .interval import Interval, icross, idot, imatmul, imatvec, istack

_G7 = 1.01 * 7 * np.finfo(float).eps / 2.0 / (1.0 - 7 * np.finfo(float).eps / 2.0)

# --------------------------------------------------------------------------
# a serial chain, in intervals
# --------------------------------------------------------------------------

_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass
class SerialChain:
    """Revolute chain: joint i rotates about local axis `axes[i]`; after it,
    `after[i]` (local vector) is translated. `marks[name]` = the joint after
    whose translation a named point sits. `labels` = joints whose sign is
    the branch label."""

    base: np.ndarray
    axes: list
    after: dict
    marks: dict
    labels: tuple

    @classmethod
    def srs(cls, arm: SrsArm) -> "SerialChain":
        p = arm.p
        return cls(base=np.asarray(arm.shoulder, dtype=float),
                   axes=["z", "y", "z", "y", "z", "y", "z"],
                   after={1: np.array([0.0, 0.0, p.d_se]), 3: np.array([0.0, 0.0, p.d_ew]),
                          6: np.array([0.0, 0.0, p.d_wt])},
                   marks={"elbow": 1, "wrist": 3, "tool": 6},
                   labels=(1, 3, 5))


def _rot(axis: str, q: Interval) -> Interval:
    c, s = q.cos(), q.sin()
    z, o = np.zeros_like(c.lo), np.ones_like(c.lo)
    Z, O = Interval(z), Interval(o)
    if axis == "z":
        rows = [[c, -s, Z], [s, c, Z], [Z, Z, O]]
    elif axis == "y":
        rows = [[c, Z, s], [Z, O, Z], [-s, Z, c]]
    else:
        rows = [[O, Z, Z], [Z, c, -s], [Z, s, c]]
    return istack([istack(r, axis=-1) for r in rows], axis=-2)


def interval_fk(chain: SerialChain, Q: Interval) -> dict:
    """Q (N, n_joints) -> world joint axes (N, n, 3), joint origins (N, n, 3),
    the final rotation (N, 3, 3) and the marked points (N, 3)."""
    N = Q.shape[0]
    R = Interval(np.broadcast_to(np.eye(3), (N, 3, 3)).copy())
    p = Interval(np.broadcast_to(chain.base, (N, 3)).copy())
    axes, origins, marks = [], [], {}
    for i, ax in enumerate(chain.axes):
        origins.append(p)
        axes.append(R[:, :, _AXIS[ax]])
        R = imatmul(R, _rot(ax, Q[:, i]))
        if i in chain.after:
            p = p + imatvec(R, np.broadcast_to(chain.after[i], (N, 3)))
        for name, j in chain.marks.items():
            if j == i:
                marks[name] = p
    return dict(axes=istack(axes, axis=1), origins=istack(origins, axis=1), R=R, **marks)


# --------------------------------------------------------------------------
# the IK system F(q, z) = 0 of an S-R-S chart on an affine task
# --------------------------------------------------------------------------


def _vee_skew(M: Interval) -> Interval:
    """vee((M - M^T) / 2) for (..., 3, 3)."""
    a = (M[..., 2, 1] - M[..., 1, 2]) * 0.5
    b = (M[..., 0, 2] - M[..., 2, 0]) * 0.5
    c = (M[..., 1, 0] - M[..., 0, 1]) * 0.5
    return istack([a, b, c], axis=-1)


class SrsSystem:
    """F(q, z) and J_q(q, z) for the S-R-S chart of an affine task."""

    def __init__(self, chart: SrsChart):
        self.chart = chart
        self.chain = SerialChain.srs(chart.arm)
        self.Rt = np.asarray(chart.task.rot, dtype=float)
        self.A = np.asarray(chart.task.A, dtype=float)
        self.p0 = np.asarray(chart.task.p0, dtype=float)
        self.S = np.asarray(chart.S, dtype=float)

    def targets(self, X: Interval, TH: Interval) -> tuple:
        """Tool target p(x) and the normal n(x, theta) of the theta half-plane."""
        N = X.shape[0]
        pt = imatvec(np.broadcast_to(self.A, (N,) + self.A.shape), X) + self.p0
        W = pt - self.chart.d_wt * self.chart.t
        x = W - self.S
        L = (x[:, 0].sqr() + x[:, 1].sqr() + x[:, 2].sqr()).sqrt()
        u = istack([x[:, 0] / L, x[:, 1] / L, x[:, 2] / L], axis=-1)
        v = istack([-(u[:, 2] * u[:, 0]), -(u[:, 2] * u[:, 1]), 1.0 - u[:, 2].sqr()], axis=-1)
        vn = (v[:, 0].sqr() + v[:, 1].sqr() + v[:, 2].sqr()).sqrt()
        v0 = istack([v[:, 0] / vn, v[:, 1] / vn, v[:, 2] / vn], axis=-1)
        w0 = icross(u, v0)
        s, c = TH.sin(), TH.cos()
        n = istack([-(s * v0[:, k]) + c * w0[:, k] for k in range(3)], axis=-1)
        return pt, n, L

    def residual_and_jacobian(self, Q: Interval, X: Interval, TH: Interval,
                              want_f: bool = True) -> tuple:
        """F(Q, Z) (N, 7) and J_q(Q, Z) (N, 7, 7), both as intervals."""
        fk = interval_fk(self.chain, Q)
        pt, n, _ = self.targets(X, TH)
        T, E, R7 = fk["tool"], fk["elbow"], fk["R"]
        N = Q.shape[0]
        RtT = np.broadcast_to(self.Rt.T, (N, 3, 3))
        F = None
        if want_f:
            Fp = T - pt
            FR = _vee_skew(imatmul(RtT, R7))
            Fth = idot(E - self.S, n)
            F = istack([Fp[:, 0], Fp[:, 1], Fp[:, 2], FR[:, 0], FR[:, 1], FR[:, 2], Fth], axis=-1)
        a, o = fk["axes"], fk["origins"]                      # (N, 7, 3)
        nj = a.shape[1]
        dT = icross(a, Interval(T.lo[:, None, :], T.hi[:, None, :]) - o)   # (N, 7, 3)
        # d(R_t^T R7)/dq_i = R_t^T [a_i]x R7: the columns of [a_i]x R7 are a_i x col_k.
        cols = Interval(np.swapaxes(R7.lo, 1, 2)[:, None, :, :],
                        np.swapaxes(R7.hi, 1, 2)[:, None, :, :])          # (N, 1, 3k, 3)
        ax_ = Interval(a.lo[:, :, None, :], a.hi[:, :, None, :])          # (N, 7, 1, 3)
        AxR = icross(ax_, cols)                                             # (N, 7, k, 3)
        AxR = Interval(np.swapaxes(AxR.lo, -1, -2), np.swapaxes(AxR.hi, -1, -2))  # (N,7,3,k)
        dR = _vee_skew(imatmul(np.broadcast_to(self.Rt.T, (N, nj, 3, 3)), AxR))  # (N, 7, 3)
        el = self.chain.marks["elbow"]
        Eb = Interval(E.lo[:, None, :], E.hi[:, None, :])
        dE = icross(a, Eb - o)                                              # (N, 7, 3)
        nb = Interval(n.lo[:, None, :], n.hi[:, None, :])
        dth = idot(dE, nb)                                                  # (N, 7)
        mask = np.arange(nj) <= el
        dth = Interval(np.where(mask, dth.lo, 0.0), np.where(mask, dth.hi, 0.0))
        J = istack([dT[:, :, 0], dT[:, :, 1], dT[:, :, 2],
                    dR[:, :, 0], dR[:, :, 1], dR[:, :, 2], dth], axis=1)     # (N, 7, 7)
        return F, J, fk


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------


def _mid_rad(I: Interval) -> tuple:
    m = 0.5 * (I.lo + I.hi)
    r = np.nextafter(np.maximum(I.hi - m, m - I.lo), np.inf)
    return m, r


@dataclass
class KrawczykResult:
    ok: np.ndarray            # (N,) certified: unique smooth branch, labels fixed
    K_mid: np.ndarray         # (N, 7) enclosure of q(z) for all z in the cell
    K_rad: np.ndarray         # (N, 7)
    why: np.ndarray           # (N,) object: reason of failure ("" if ok)
    stats: dict = field(default_factory=dict)


class KrawczykCertifier:
    """Drop-in for SrsChart.ball_test / clear_many in the door search and the
    facet test, built on the Krawczyk test (module docstring)."""

    def __init__(self, chart: SrsChart, signs=(1, 1, 1), tries: int = 10):
        self.chart = chart
        self.scale = chart.scale
        self.signs = tuple(signs)
        self.system = SrsSystem(chart)
        self.tries = tries

    # -- the core ----------------------------------------------------------

    def krawczyk(self, C: np.ndarray, H: np.ndarray) -> KrawczykResult:
        """Run the test on N cells C +- H (unscaled page coordinates)."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        N, n = C.shape[0], C.shape[1] - 1
        ch = self.chart
        ok = np.zeros(N, dtype=bool)
        why = np.full(N, "", dtype=object)
        Km = np.full((N, 7), np.nan)
        Kr = np.full((N, 7), np.inf)
        qc = np.full((N, 7), np.nan)
        pos = ch.task.position(C[:, :n])
        for i in range(N):
            r = ch.arm.ik_branch(pos[i], ch.task.rot, float(C[i, n]), self.signs)
            if r.ok:
                qc[i] = r.q
        have = np.all(np.isfinite(qc), axis=1)
        why[~have] = "no IK at the centre"
        idx = np.flatnonzero(have)
        if not len(idx):
            return KrawczykResult(ok, Km, Kr, why)
        Xc = Interval(C[idx, :n])
        THc = Interval(C[idx, n])
        Xz = Interval.around(C[idx, :n], H[idx, :n])
        THz = Interval.around(C[idx, n], H[idx, n])
        q0 = Interval(qc[idx])
        F0, _, _ = self.system.residual_and_jacobian(q0, Xz, THz)        # F(q_c, Z)
        _, Jc, _ = self.system.residual_and_jacobian(q0, Xc, THc, want_f=False)
        Jm, _ = _mid_rad(Jc)
        Fm, Fr = _mid_rad(F0)
        good = np.all(np.isfinite(Fm) & np.isfinite(Fr), axis=1) & np.all(np.isfinite(Jm), axis=(1, 2))
        Y = np.full_like(Jm, np.nan)
        for k in np.flatnonzero(good):
            try:
                Y[k] = np.linalg.inv(Jm[k])
            except np.linalg.LinAlgError:
                good[k] = False
        good &= np.all(np.isfinite(Y), axis=(1, 2))
        aY = np.abs(Y)
        # b = |Y F_mid| + |Y| F_rad, plus the rounding of the products.
        YF = np.einsum("nij,nj->ni", Y, Fm)
        b = (np.abs(YF) + np.einsum("nij,nj->ni", aY, Fr)
             + 2.0 * _G7 * np.einsum("nij,nj->ni", aY, np.abs(Fm)))
        rad = 1.25 * b + 1e-12
        done = ~good
        why[idx[~good]] = "residual or Jacobian undefined on the cell (Sigma_coord?)"
        for _t in range(self.tries):
            todo = np.flatnonzero(~done)
            if not len(todo):
                break
            Q = Interval.around(qc[idx[todo]], rad[todo])
            _, J, _ = self.system.residual_and_jacobian(Q, Xz[todo], THz[todo], want_f=False)
            Jm2, Jr2 = _mid_rad(J)
            fin = np.all(np.isfinite(Jm2) & np.isfinite(Jr2), axis=(1, 2))
            Yt, aYt = Y[todo], aY[todo]
            M = np.eye(7)[None] - np.einsum("nij,njk->nik", Yt, Jm2)
            Mr = (np.einsum("nij,njk->nik", aYt, Jr2)
                  + 2.0 * _G7 * np.einsum("nij,njk->nik", aYt, np.abs(Jm2)) + 1e-300)
            contr = np.einsum("nij,nj->ni", np.abs(M) + Mr, rad[todo])
            tot = (b[todo] + contr) * (1.0 + 1e-12) + 1e-300
            inside = fin & np.all(tot < rad[todo], axis=1)
            for j, k in enumerate(todo):
                if inside[j]:
                    done[k] = True
                    Km[idx[k]] = qc[idx[k]]
                    Kr[idx[k]] = tot[j]
            # epsilon-inflation (Rump): the next box takes the shape of K.
            nxt = np.flatnonzero(~inside)
            grow = np.where(np.isfinite(tot[nxt]), 1.25 * tot[nxt] + 1e-12, np.inf)
            rad[todo[nxt]] = grow
            done[todo[nxt[~np.all(np.isfinite(grow), axis=1)]]] = True
        cert = np.isfinite(Kr).all(axis=1)
        lab = list(self.system.chain.labels)
        zero = cert[:, None] & (np.abs(Km[:, lab]) <= Kr[:, lab])
        ok = cert & ~zero.any(axis=1)
        for i in np.flatnonzero(cert & ~ok):
            js = [f"q{lab[k] + 1}" for k in np.flatnonzero(zero[i])]
            why[i] = f"label joint(s) {', '.join(js)} may be 0"
        for i in np.flatnonzero(have & ~cert):
            if not why[i]:
                why[i] = "Krawczyk did not contract (Jacobian near-singular on the cell)"
        return KrawczykResult(ok, Km, Kr, why, stats=dict(cells=N, certified=int(ok.sum())))

    # -- the interface the search and the facet test use ---------------------

    def box_test(self, C: np.ndarray, H: np.ndarray) -> BallTest:
        C = np.atleast_2d(np.asarray(C, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        ch = self.chart
        N, n = C.shape[0], C.shape[1] - 1
        R = np.linalg.norm(H * self.scale, axis=1)
        # Off the task box (in x): no state of the task, no door (invariant 37).
        off = np.any((C[:, :n] + H[:, :n] < ch.lo[:n]) | (C[:, :n] - H[:, :n] > ch.hi[:n]),
                     axis=1)
        # No configuration anywhere in the cell: the reach annulus excludes it.
        Xz = Interval.around(C[:, :n], H[:, :n])
        _, _, L = self.system.targets(Xz, Interval.around(C[:, n], H[:, n]))
        no_ik = (L.lo > ch.reach_max) | (L.hi < ch.reach_min)
        res = self.krawczyk(C, H)
        clear = off | no_ik | res.ok
        sig = ch.sigma(C)
        may = np.zeros((N, 4), dtype=bool)
        for i in np.flatnonzero(~clear):
            w = str(res.why[i])
            if "q2" in w:
                may[i, I_Q2] = True
            if "q4" in w:
                may[i, I_Q4] = True
            if "q6" in w:
                may[i, I_Q6] = True
            if "Sigma_coord" in w:
                may[i, I_COORD] = True
            if not may[i].any():
                if w.startswith("no IK"):
                    may[i, I_Q4] = True
                else:
                    # Did not contract: name the labelled joint closest to 0
                    # (evidence only -- it labels the region, it decides nothing).
                    s = np.nan_to_num(sig[i, [I_Q2, I_Q6]], nan=np.inf)
                    may[i, [I_Q2, I_Q6][int(np.argmin(s))]] = True
        return BallTest(clear=clear, may_vanish=may, sigma=sig,
                        lip=np.full((N, 4), np.nan), off_task=off, no_ik=no_ik)

    def ball_test(self, C, R) -> BallTest:
        C = np.atleast_2d(np.asarray(C, dtype=float))
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
        return self.box_test(C, R[:, None] / self.scale[None, :])

    def clear_many(self, C, R) -> np.ndarray:
        return self.ball_test(C, R).clear

    # -- collision, from the same enclosure -----------------------------------

    def collision_certified(self, scene, C: np.ndarray, R) -> np.ndarray:
        """Every posture in the ball collides: the Krawczyk enclosure K of
        q(z) over the ball, pushed through interval FK, boxes every link end;
        a link's distance to an obstacle moves no more than its ends."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
        H = R[:, None] / self.scale[None, :]
        res = self.krawczyk(C, H)
        out = np.zeros(len(C), dtype=bool)
        idx = np.flatnonzero(np.isfinite(res.K_rad).all(axis=1))
        if not len(idx):
            return out
        fk = interval_fk(self.system.chain, Interval.around(res.K_mid[idx], res.K_rad[idx]))
        for j, i in enumerate(idx):
            pts = {}
            disp = 0.0
            for name in ("elbow", "wrist", "tool"):
                P = fk[name][j]
                m = 0.5 * (P.lo + P.hi)
                pts[name] = m
                disp = max(disp, float(np.linalg.norm(np.maximum(P.hi - m, m - P.lo))))
            frames = dict(shoulder=self.chart.S, **pts)
            pen = scene.check(frames).penetration
            out[i] = pen > disp * (1.0 + 1e-9) + 1e-12
        return out
