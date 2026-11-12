"""The chart of Y+ and its singularity function sigma (docs/recursive_proof_design.md).

[IROS] plans in Y+ = Y x X: task coordinates x (the weld line's y, or the
patch's (u, v)) plus the extra parameters X = (theta, s2, s4, s6). Each sign
triple is a PAGE with coordinates z = (x, theta). The pages are glued where the
chart breaks down -- the DOORS -- and a robot path in Q crosses pages there.

This module answers ONE question for the proof, without any door list:

    could the chart break down anywhere in the ball B(c, rho)?

through a vector-valued, distance-like function sigma whose zeros are exactly
the chart's singular set, and a GUARANTEED Lipschitz bound of every component
over the ball. A ball is clear of doors when sigma_i(c) > L_i * rho for every
component i. Facet checks (test ii) and the interior zero search (test iii)
are built on that one predicate.

sigma is the singularity of the CHART, not of the task: at q2 = 0 and q6 = 0
the 6x7 Jacobian is full rank (two axes coincide, the other six span), so a
task-Jacobian test would miss every q2 and q6 door. For the S-R-S arm the
chart breaks down on four sets:

    q2     |q2| = angle(e_z, upper-arm direction)       radians
    q6     |q6| = angle(forearm direction, tool axis)    radians
    q4     |reach slack| = |min(reach_max - L, L - reach_min)|, L = |W - S|
           (the fold; |q4| itself behaves like sqrt(distance) there and has
           no Lipschitz bound, the slack does)            metres
    coord  horizontal distance of W from the shoulder's z axis (the arm-angle
           reference plane is undefined there)            metres

sigma is the same on all eight pages: the shoulder, elbow, wrist and tool are
placed identically on every sign branch (arm.py), and |q2|, |q6| are angles
between those physical directions. A door is a PHYSICAL place.

THE BOUNDS. Everything is in SCALED page coordinates (the facet checker's
units). The task here is AFFINE with a fixed tool orientation -- W(x) =
w0 + A x -- which covers every weld line and flat patch in this project:

  * L = |W - S| and the horizontal distance are 1-Lipschitz in W, so their
    bounds are the exact operator norms G = ||A D_x^-1||_2 and
    G_c = ||P A D_x^-1||_2 (P drops z).
  * The elbow E(W, theta) = S + h u + r (cos theta v0 + sin theta w0) (the
    same formula as SrsArm.ik_branch). Its derivative is bounded over the
    ball from the ball's worst case of L (via the reach slack), of the
    elbow-circle radius r (via |h| at the ends of the L interval -- h is
    convex or monotone in L, so its largest |h| is at an end), and of the
    angle between S-W and z (via the horizontal distance). Each term blows
    up exactly at one of the singular sets, so the bound is finite iff the
    ball avoids the fold and Sigma_coord.
  * |q2| is 1-Lipschitz in the upper-arm direction (geodesic metric), and
    that direction moves at |dE| / d_se; |q6| likewise with W - E and d_ew.

These are analytic inequalities, not estimates; tests/test_chart.py checks
them against random pairs (a violation would be a bug in the algebra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import SIGN_BRANCHES, SrsArm

SIGMA_NAMES = ("q2", "q6", "q4", "coord")
I_Q2, I_Q6, I_Q4, I_COORD = range(4)


# --------------------------------------------------------------------------
# the task: an affine wrist map with a fixed tool orientation
# --------------------------------------------------------------------------


@dataclass
class AffineTask:
    """Tool position p(x) = p0 + A_p x on x in [0, 1]^n, fixed orientation rot.

    The wrist centre is then W(x) = w0 + A x with A = A_p and
    w0 = p0 - d_wt * tool_axis."""

    p0: np.ndarray
    A: np.ndarray              # (3, n)
    rot: np.ndarray            # (3, 3)

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def tool_axis(self) -> np.ndarray:
        return self.rot[:, 2]

    def position(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self.p0[None, :] + X @ self.A.T

    def wrist(self, X: np.ndarray, d_wt: float) -> np.ndarray:
        return self.position(X) - d_wt * self.tool_axis[None, :]

    @classmethod
    def from_line(cls, line) -> "AffineTask":
        """weld.WeldLine: x = (y,)."""
        return cls(p0=np.asarray(line.p0, dtype=float),
                   A=np.asarray(line.direction, dtype=float).reshape(3, 1),
                   rot=np.asarray(line.rot, dtype=float))

    @classmethod
    def from_surface(cls, surf) -> "AffineTask":
        """weld.WeldSurface: x = (u, v)."""
        return cls(p0=np.asarray(surf.p0, dtype=float),
                   A=np.column_stack([surf.du, surf.dv]).astype(float),
                   rot=np.asarray(surf.rot, dtype=float))


# --------------------------------------------------------------------------
# the chart
# --------------------------------------------------------------------------


@dataclass
class BallTest:
    """Result of SrsChart.ball_test for N balls."""

    clear: np.ndarray          # (N,) no zero of any sigma component in the ball
    may_vanish: np.ndarray     # (N, 4) per component: could not be excluded
    sigma: np.ndarray          # (N, 4) sigma at the centres (nan: undefined)
    lip: np.ndarray            # (N, 4) Lipschitz bounds over the balls (inf: none)
    off_task: np.ndarray       # (N,) the ball misses the task box (no doors there)
    no_ik: np.ndarray          # (N,) no configuration anywhere in the ball

    def reasons(self, i: int) -> str | None:
        if self.clear[i]:
            return None
        names = [SIGMA_NAMES[k] for k in range(4) if self.may_vanish[i, k]]
        return "near sigma = 0 (" + ", ".join(names) + ")"


class SrsChart:
    """Y+ chart of an S-R-S arm on an affine task with fixed orientation.

    Page coordinates z = (x_1 .. x_n, theta); `scale` maps them to the scaled
    units the walls and facets live in (Page3Params.scale at d = 3)."""

    def __init__(self, arm: SrsArm, task: AffineTask, scale, theta_range=(-np.pi, np.pi),
                 bound: str = "frame"):
        """bound: "frame" (default, the arm-angle frame's rotation bounded
        exactly) or "triangle" (the first version, every term separately)."""
        if bound not in ("frame", "triangle"):
            raise ValueError(f"unknown bound {bound!r}")
        self.bound = bound
        self.arm = arm
        self.task = task
        self.scale = np.asarray(scale, dtype=float)
        n = task.n
        if self.scale.shape != (n + 1,):
            raise ValueError(f"scale must have {n + 1} entries for an n = {n} task")
        p = arm.p
        self.d_se, self.d_ew, self.d_wt = float(p.d_se), float(p.d_ew), float(p.d_wt)
        self.reach_max = self.d_se + self.d_ew
        self.reach_min = abs(self.d_se - self.d_ew)
        self.S = np.asarray(arm.shoulder, dtype=float)
        self.t = task.tool_axis / np.linalg.norm(task.tool_axis)
        self.w0 = task.p0 - self.d_wt * self.t
        self.lo = np.r_[np.zeros(n), theta_range[0]]
        self.hi = np.r_[np.ones(n), theta_range[1]]
        Ax = task.A / self.scale[None, :n]                      # dW / d(scaled x)
        self.G = float(np.linalg.norm(Ax, 2))                   # |W| Lipschitz
        P = np.diag([1.0, 1.0, 0.0])
        self.G_c = float(np.linalg.norm(P @ Ax, 2))             # horizontal distance
        self.s_th = float(self.scale[n])
        self.pages = SIGN_BRANCHES

    @property
    def dim(self) -> int:
        return self.task.n + 1

    @classmethod
    def for_surface(cls, arm: SrsArm, surf, p3) -> "SrsChart":
        return cls(arm, AffineTask.from_surface(surf), p3.scale,
                   (p3.theta_min, p3.theta_max))

    @classmethod
    def for_line(cls, arm: SrsArm, line, scale=(1.0, 1.0 / np.pi)) -> "SrsChart":
        return cls(arm, AffineTask.from_line(line), scale)

    # -- geometry ----------------------------------------------------------

    def wrist(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return self.w0[None, :] + Z[:, :-1] @ self.task.A.T

    def elbow(self, Z: np.ndarray) -> tuple:
        """(E, ok): the elbow of every state, as SrsArm.ik_branch places it
        (the world-z reference; ok False where there is no IK or the
        reference is degenerate)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        W = self.wrist(Z)
        x = W - self.S[None, :]
        L = np.linalg.norm(x, axis=1)
        u = x / np.maximum(L, 1e-300)[:, None]
        h = (self.d_se ** 2 - self.d_ew ** 2 + L ** 2) / (2.0 * np.maximum(L, 1e-300))
        r = np.sqrt(np.maximum(0.0, self.d_se ** 2 - h ** 2))
        nz = np.array([0.0, 0.0, 1.0])[None, :] - u[:, 2:3] * u
        sn = np.linalg.norm(nz, axis=1)
        v0 = nz / np.maximum(sn, 1e-300)[:, None]
        w0 = np.cross(u, v0)
        th = Z[:, -1]
        E = self.S[None, :] + h[:, None] * u + r[:, None] * (
            np.cos(th)[:, None] * v0 + np.sin(th)[:, None] * w0)
        ok = (L >= self.reach_min) & (L <= self.reach_max) & (sn > 1e-12) & (L > 1e-12)
        return E, ok

    def sigma(self, Z: np.ndarray) -> np.ndarray:
        """(N, 4) sigma at N page states: |q2|, |q6| (nan where there is no
        configuration), |reach slack|, axis distance."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        W = self.wrist(Z)
        x = W - self.S[None, :]
        L = np.linalg.norm(x, axis=1)
        slack = np.minimum(self.reach_max - L, L - self.reach_min)
        E, ok = self.elbow(Z)
        du = (E - self.S[None, :]) / self.d_se
        df = (W - E) / self.d_ew
        s2 = np.arccos(np.clip(du[:, 2], -1.0, 1.0))
        s6 = np.arccos(np.clip(df @ self.t, -1.0, 1.0))
        out = np.empty((len(Z), 4))
        out[:, I_Q2] = np.where(ok, s2, np.nan)
        out[:, I_Q6] = np.where(ok, s6, np.nan)
        out[:, I_Q4] = np.abs(slack)
        out[:, I_COORD] = np.linalg.norm(x[:, :2], axis=1)
        return out

    def reach_slack(self, Z: np.ndarray) -> np.ndarray:
        W = self.wrist(Z)
        L = np.linalg.norm(W - self.S[None, :], axis=1)
        return np.minimum(self.reach_max - L, L - self.reach_min)

    # -- the bounds --------------------------------------------------------

    def lipschitz(self, C: np.ndarray, R) -> np.ndarray:
        """(N, 4) guaranteed Lipschitz bounds of each sigma component over the
        balls B(C_i, R_i) (scaled radius), in sigma units per scaled unit.
        inf where the ball reaches the fold or Sigma_coord (no bound exists)."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
        G, Gc = self.G, self.G_c
        W = self.wrist(C)
        x = W - self.S[None, :]
        Lc = np.linalg.norm(x, axis=1)
        slack = np.minimum(self.reach_max - Lc, Lc - self.reach_min)
        hc = np.linalg.norm(x[:, :2], axis=1)
        out = np.full((len(C), 4), np.inf)
        out[:, I_Q4] = G
        out[:, I_COORD] = Gc

        # The ball must stay strictly inside the annulus and off the axis.
        ok = (slack - G * R > 0.0) & (hc - Gc * R > 0.0)
        if not ok.any():
            return out
        Rk = R[ok]
        L_lo = Lc[ok] - G * Rk
        L_hi = Lc[ok] + G * Rk
        c = self.d_se ** 2 - self.d_ew ** 2

        def h_of(L):
            return (c + L ** 2) / (2.0 * L)

        # |h| is largest at an end of [L_lo, L_hi] (h is convex for c > 0 and
        # increasing for c <= 0), so r = sqrt(d_se^2 - h^2) is smallest there.
        hmax = np.maximum(np.abs(h_of(L_lo)), np.abs(h_of(L_hi)))
        r_min_sq = self.d_se ** 2 - hmax ** 2
        good = r_min_sq > 0.0
        r_min = np.sqrt(np.where(good, r_min_sq, 1.0))
        hp = 0.5 + abs(c) / (2.0 * L_lo ** 2)                  # max |dh/dL|
        sin_min = (hc[ok] - Gc * Rk) / L_hi                    # angle(S-W, z)
        if self.bound == "triangle":
            # Findings of the first version: every term by its own triangle
            # inequality. Sound, ~5-6x the true derivative (median).
            rp = hmax * hp / r_min
            d_u = 1.0 / L_lo
            d_v0 = 2.0 / (L_lo * sin_min)
            d_w0 = d_u + d_v0
            r_max = np.full(len(Rk), min(self.d_se, self.d_ew))
            e_w = hp + hmax * d_u + rp + r_max * np.sqrt(d_v0 ** 2 + d_w0 ** 2)
            e_f = 1.0 + e_w
        else:
            # "frame" (default): the same quantities, combined exactly.
            # Split a wrist step dW = p u + q (q _|_ u). Then dL = p,
            # du = q / L, and with rho = cos(theta) v0 + sin(theta) w0:
            #   dE = (h' u + r' rho) p + (h / L) q + r * dRho,
            # where (u, v0, w0) is the arm-angle frame. Writing du = a v0 + b w0
            # gives dv0 = -a u - b cot(alpha) w0 and dw0 = -b u + b cot(alpha) v0,
            # so |dRho|^2 = (a cos + b sin)^2 + b^2 cot^2(alpha)
            # <= |du|^2 / sin^2(alpha). With r' = -h h' / r and r^2 + h^2 = d_se^2,
            # |h' u + r' rho| = |h'| d_se / r. Cauchy-Schwarz over (p, |q|):
            #   |dE / dW| <= sqrt((h'_max d_se / r_min)^2
            #                    + ((|h|_max + r_max / sin(alpha)_min) / L_lo)^2).
            # The forearm W - E = (L - h) u - r rho gives the same form with
            # 1 - h' and |L - h|.
            hmin = np.minimum(np.abs(h_of(L_lo)), np.abs(h_of(L_hi)))
            if c > 0.0:
                Ls = np.sqrt(c)
                inside = (L_lo <= Ls) & (Ls <= L_hi)
                hmin = np.where(inside, np.minimum(hmin, abs(h_of(Ls))), hmin)
            else:
                hmin = np.where(h_of(L_lo) * h_of(L_hi) <= 0.0, 0.0, hmin)
            r_max = np.sqrt(np.maximum(0.0, self.d_se ** 2 - hmin ** 2))
            r_max = np.minimum(r_max, min(self.d_se, self.d_ew))
            rp = hp * self.d_se / r_min                         # |(h', r')|
            e_w = np.sqrt(rp ** 2 + ((hmax + r_max / sin_min) / L_lo) ** 2)
            # h' is monotone in L (h'' = c / L^3 keeps its sign).
            hp_lo = 0.5 - c / (2.0 * L_lo ** 2)
            hp_hi = 0.5 - c / (2.0 * L_hi ** 2)
            one_m = np.maximum(np.abs(1.0 - hp_lo), np.abs(1.0 - hp_hi))
            rp_f = hmax * hp / r_min
            g_lo = np.abs((L_lo ** 2 - c) / (2.0 * L_lo))       # |L - h|, extreme
            g_hi = np.abs((L_hi ** 2 - c) / (2.0 * L_hi))       # at an end of the
            e_f = np.sqrt(one_m ** 2 + rp_f ** 2                 # interval
                          + ((np.maximum(g_lo, g_hi) + r_max / sin_min) / L_lo) ** 2)
        m_e = np.sqrt((e_w * G) ** 2 + (r_max / self.s_th) ** 2)
        m_f = np.sqrt((e_f * G) ** 2 + (r_max / self.s_th) ** 2)
        L2 = np.where(good, m_e / self.d_se, np.inf)
        L6 = np.where(good, m_f / self.d_ew, np.inf)
        idx = np.flatnonzero(ok)
        out[idx, I_Q2] = L2
        out[idx, I_Q6] = L6
        return out

    # -- the predicate the proof uses --------------------------------------

    def off_task(self, C: np.ndarray, R) -> np.ndarray:
        """The ball misses the task box in the x axes (theta is an angle, not
        a task bound). Off the task the tool is not on the weld: no state
        there is a state of the task, so it has no doors (invariant 37)."""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
        n = self.task.n
        cs = C[:, :n] * self.scale[:n]
        lo, hi = self.lo[:n] * self.scale[:n], self.hi[:n] * self.scale[:n]
        gap = np.maximum(0.0, np.maximum(lo[None, :] - cs, cs - hi[None, :]))
        return np.linalg.norm(gap, axis=1) > R

    def ball_test(self, C: np.ndarray, R) -> BallTest:
        """Can any sigma component vanish in B(C_i, R_i)? (scaled radius)"""
        C = np.atleast_2d(np.asarray(C, dtype=float))
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),)).astype(float)
        sig = self.sigma(C)
        lip = self.lipschitz(C, R)
        slack = self.reach_slack(C)
        off = self.off_task(C, R)
        no_ik = (slack < 0.0) & (-slack > self.G * R)
        with np.errstate(invalid="ignore"):
            may = ~(sig > lip * R[:, None])        # nan or inf -> may vanish
        # With no IK anywhere in the ball, q2 / q6 have no states to vanish at.
        may[no_ik] = False
        may[off] = False
        clear = ~may.any(axis=1)
        return BallTest(clear=clear, may_vanish=may, sigma=sig, lip=lip,
                        off_task=off, no_ik=no_ik)

    def clear_many(self, C: np.ndarray, R) -> np.ndarray:
        """FacetOracle.sigma_many: True where the ball is certified clear."""
        return self.ball_test(C, R).clear

    def facet_sigma(self, c: np.ndarray, r: float) -> str | None:
        """FacetOracle.sigma: None if the facet's ball is certified clear."""
        t = self.ball_test(np.asarray(c, dtype=float)[None, :], np.array([float(r)]))
        return t.reasons(0)

    def facet_sigma_many(self, C: np.ndarray, R) -> list:
        t = self.ball_test(C, R)
        return [t.reasons(i) for i in range(len(t.clear))]


# --------------------------------------------------------------------------
# certified collision over a ball (posture facts are page-independent)
# --------------------------------------------------------------------------


def link_speed(chart: SrsChart, C: np.ndarray, R) -> np.ndarray:
    """Guaranteed bound on how far any point of the arm's links can move per
    scaled unit within B(C_i, R_i): the shoulder is fixed, the elbow moves at
    most M_E = L_q2 * d_se (the chart's elbow bound), the wrist and the tool
    at most G (theta does not move them; the orientation is fixed). Points
    on a link segment move no more than its ends. inf where the elbow bound
    does not exist (the ball reaches the fold or Sigma_coord)."""
    lip = chart.lipschitz(C, R)
    return np.maximum(lip[:, I_Q2] * chart.d_se, chart.G)


def collision_certified(chart: SrsChart, scene, C: np.ndarray, R) -> np.ndarray:
    """True where EVERY posture in B(C_i, R_i) collides.

    scene.check's penetration is a sampled LOWER bound on the true overlap
    (its distance samples can only overestimate the segment-box distance), and
    a link's distance to a box changes no faster than its points move. So
    pen(c) > link_speed * R puts every posture of the ball in collision. A
    posture is the same on all eight pages (they differ only in joint
    spelling), so this certificate holds on every page -- for rotationally
    symmetric link geometry only (source.geometry_is_page_dependent)."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
    E, ok = chart.elbow(C)
    W = chart.wrist(C)
    T = W + chart.d_wt * chart.t[None, :]
    speed = link_speed(chart, C, R)
    out = np.zeros(len(C), dtype=bool)
    for i in np.flatnonzero(ok & np.isfinite(speed)):
        frames = dict(shoulder=chart.S, elbow=E[i], wrist=W[i], tool=T[i])
        pen = scene.check(frames).penetration
        out[i] = pen > speed[i] * R[i]
    return out


# --------------------------------------------------------------------------
# the general singularity function (detection only: no certified bound yet)
# --------------------------------------------------------------------------


def arm_angle(arm: SrsArm, q: np.ndarray) -> float:
    """The arm angle theta of a configuration, as SrsArm.ik_branch defines it
    (elbow position about the shoulder-wrist axis, from the world-z plane)."""
    f = arm.frames(q)
    S, E, W = f["shoulder"], f["elbow"], f["wrist"]
    u = (W - S) / np.linalg.norm(W - S)
    v0 = np.array([0.0, 0.0, 1.0]) - u[2] * u
    v0 /= np.linalg.norm(v0)
    w0 = np.cross(u, v0)
    e = E - S - np.dot(E - S, u) * u
    return float(np.arctan2(np.dot(e, w0), np.dot(e, v0)))


def augmented_jacobian(arm: SrsArm, q: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """J_psi = d(pose, theta)/dq, 7 x 7: the 6 x 7 task Jacobian plus the
    arm angle's gradient."""
    J = np.zeros((7, 7))
    J[:6] = arm.jacobian(q)
    for i in range(7):
        d = np.zeros(7)
        d[i] = h
        a, b = arm_angle(arm, q + d), arm_angle(arm, q - d)
        J[6, i] = np.angle(np.exp(1j * (a - b))) / (2.0 * h)
    return J


def general_sigma(chart: SrsChart, Z: np.ndarray, signs=(1, 1, 1)) -> np.ndarray:
    """sigma_general(z) = smallest singular value of J_psi at q(z).

    By the inverse function theorem the IK is locally unique and smooth
    wherever J_psi is invertible, so no branch label can change there: every
    door lies where sigma_general = 0 or where theta(q) is undefined
    (Sigma_coord). It needs no knowledge of WHICH joints make doors. It has
    no guaranteed bound yet, so it can DETECT doors but cannot certify that a
    region has none; the certified search uses SrsChart.sigma.
    nan where there is no IK."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    out = np.full(len(Z), np.nan)
    pos = chart.task.position(Z[:, :-1])
    for i, z in enumerate(Z):
        r = chart.arm.ik_branch(pos[i], chart.task.rot, float(z[-1]), signs)
        if r.ok:
            out[i] = float(np.linalg.svd(augmented_jacobian(chart.arm, r.q),
                                         compute_uv=False)[-1])
    return out
