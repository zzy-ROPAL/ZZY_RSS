"""Sigma on a 3-D page (u, v, theta): door CURVES and their tubes (findings 42).

On a weld line a door is a point of the (y, theta) page; on a weld PATCH it is
a curve of the (u, v, theta) page. The geometry carries over unchanged:

  q2 = 0   pins the elbow at E* = S + d_se e_z, so |W(u, v) - E*| = d_ew
  q6 = 0   pins the elbow at W - d_ew t, so |W(u, v) - (S + d_ew t)| = d_se

and on a FLAT patch with a fixed tool orientation W(u, v) is affine, i.e. it
sweeps a plane. A sphere cut by a plane is a circle, so each door is a circle
in the wrist plane, parameterised exactly -- no scan, no tolerance. theta along
it follows from the elbow position, exactly as on the line.

The other two singular sets are checked, not covered:

  q4 fold     |W - S| = reach_max or reach_min. If the patch's |W - S| range
              stays strictly inside the annulus there is no fold anywhere on
              it; otherwise the module REFUSES (a fold is a surface, not a
              curve, and has no tube here yet).
  Sigma_coord W on the shoulder's z axis (the arm-angle reference breaks).
              Same rule: the patch must stay clear of the axis or we refuse.

Sigma_kin is not built, as on the default 2-D roadmap (with_kin=False).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arm import SrsArm
from .doors import DOOR_INTERIOR, SigmaDisc, _arm_angle_of_elbow, certify_fiber
from .facets import SigmaTube
from .params import Config, Page3Params
from .scene import Scene
from .weld import WeldSurface

DOOR_JOINT = {0: 2, 2: 6}           # sign index -> joint number (q2, q6)


@dataclass
class DoorCurve:
    """One connected run of a door circle inside the patch window."""

    sign_index: int
    pts: np.ndarray                 # (n, 3) (u, v, theta), ordered along the curve
    closed: bool                    # the whole circle lies in the window
    tube: SigmaTube | None = None
    certified_blocked: bool = False
    unknown: bool = True
    reason: str = "not certified"
    cert: dict = field(default_factory=dict)

    @property
    def joint(self) -> int:
        return DOOR_JOINT[self.sign_index]


# --------------------------------------------------------------------------
# the patch's own singularities: refuse rather than cover
# --------------------------------------------------------------------------


def _min_convex_quadratic_on_patch(w0, a, b, c) -> float:
    """min over (u, v) in [0,1]^2 of |w0 + u a + v b - c|. Exact: the
    unconstrained minimiser if it is inside, else the best of the four edges
    (each a 1-D clamped projection)."""
    A = np.column_stack([a, b])
    x, *_ = np.linalg.lstsq(A, c - w0, rcond=None)
    if np.all((x >= 0.0) & (x <= 1.0)):
        return float(np.linalg.norm(w0 + A @ x - c))
    best = np.inf
    for p, d in ((w0, a), (w0 + b, a), (w0, b), (w0 + a, b)):
        t = np.clip(np.dot(c - p, d) / np.dot(d, d), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p + t * d - c)))
    return best


def patch_singularities(cfg: Config, arm: SrsArm, surf: WeldSurface) -> dict:
    """Is the patch clear of the q4 fold and of Sigma_coord? Exact bounds."""
    p = arm.p
    w0 = surf.wrist_center(0.0, 0.0, p.d_wt)
    a, b = surf.du, surf.dv
    S = arm.shoulder
    corners = [w0, w0 + a, w0 + b, w0 + a + b]
    r_max = max(float(np.linalg.norm(c - S)) for c in corners)   # convex: max at a corner
    r_min = _min_convex_quadratic_on_patch(w0, a, b, S)
    # Distance of the patch from the shoulder's vertical axis: the same
    # problem projected onto the horizontal plane.
    P = np.diag([1.0, 1.0, 0.0])
    axis_dist = _min_convex_quadratic_on_patch(P @ w0, P @ a, P @ b, P @ S)
    fold = not (p.reach_min < r_min and r_max < p.reach_max)
    coord = axis_dist < 1e-3
    reason = []
    if fold:
        reason.append(f"q4 fold reaches the patch (|W-S| in [{r_min:.4f}, "
                      f"{r_max:.4f}], annulus [{p.reach_min:.4f}, {p.reach_max:.4f}])")
    if coord:
        reason.append(f"wrist passes {axis_dist * 1e3:.2f} mm from the shoulder "
                      f"axis (Sigma_coord)")
    return dict(ok=not (fold or coord), fold=fold, coord=coord,
                w_range=(r_min, r_max), axis_dist=axis_dist,
                reason="; ".join(reason) or "patch clear of the q4 fold and Sigma_coord")


# --------------------------------------------------------------------------
# door curves: sphere x plane = circle
# --------------------------------------------------------------------------


def _uv_of(surf: WeldSurface, d_wt: float, P: np.ndarray) -> np.ndarray:
    """(u, v) of wrist-plane points P (n, 3)."""
    w0 = surf.wrist_center(0.0, 0.0, d_wt)
    A = np.column_stack([surf.du, surf.dv])
    x, *_ = np.linalg.lstsq(A, (P - w0).T, rcond=None)
    return x.T


def door_curves(cfg: Config, arm: SrsArm, surf: WeldSurface,
                p3: Page3Params) -> list[DoorCurve]:
    """Every q2 and q6 door curve meeting the patch window, with its tube."""
    p = arm.p
    e_z = np.array([0.0, 0.0, 1.0])
    t = surf.tool_axis
    S = arm.shoulder
    w0 = surf.wrist_center(0.0, 0.0, p.d_wt)
    e1 = surf.du / np.linalg.norm(surf.du)
    e2 = surf.dv - np.dot(surf.dv, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    n = np.cross(e1, e2)
    ru, rv = float(p3.tube_radius[0]), float(p3.tube_radius[1])
    out = []
    for idx, (C, rho) in ((0, (S + p.d_se * e_z, p.d_ew)),
                          (2, (S + p.d_ew * t, p.d_se))):
        h = float(np.dot(C - w0, n))
        if rho * rho <= h * h:
            continue                                    # sphere misses the plane
        Cp = C - h * n
        rc = float(np.sqrt(rho * rho - h * h))
        phi = np.linspace(0.0, 2.0 * np.pi, int(p3.door_samples), endpoint=False)
        P = Cp + rc * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        uv = _uv_of(surf, p.d_wt, P)
        # The window: the patch plus one tube radius, so a curve that grazes
        # the edge is still covered where it enters.
        inside = ((uv[:, 0] >= -ru) & (uv[:, 0] <= 1.0 + ru)
                  & (uv[:, 1] >= -rv) & (uv[:, 1] <= 1.0 + rv))
        if not inside.any():
            continue
        theta = np.full(len(P), np.nan)
        for i in np.flatnonzero(inside):
            u, v = float(uv[i, 0]), float(uv[i, 1])
            elbow = C if idx == 0 else (surf.wrist_center(u, v, p.d_wt) - p.d_ew * t)
            th = _arm_angle_of_elbow(arm, surf.line(v), u, elbow)
            if th is not None:
                theta[i] = th
        ok = inside & np.isfinite(theta)
        for run, closed in _runs(ok, theta):
            pts = np.column_stack([uv[run], theta[run]])
            out.append(DoorCurve(sign_index=idx, pts=pts, closed=closed))
    for c in out:
        poly = np.vstack([c.pts, c.pts[:1]]) if c.closed else c.pts
        c.tube = SigmaTube(polyline=poly, radii=np.asarray(p3.tube_radius, dtype=float),
                           kind=f"q{c.joint} door curve")
    return out


def _runs(mask: np.ndarray, theta: np.ndarray) -> list:
    """Contiguous runs of `mask` around the circle, also split where theta
    jumps across the +/-pi cut. Returns [(index array, closed), ...]."""
    n = len(mask)
    if mask.all():
        jumps = np.flatnonzero(np.abs(np.diff(np.r_[theta, theta[0]])) > np.pi / 2)
        if len(jumps) == 0:
            return [(np.arange(n), True)]
        start = (jumps[0] + 1) % n
        order = np.r_[np.arange(start, n), np.arange(0, start)]
        mask_r = np.ones(n, dtype=bool)
    else:
        start = int(np.flatnonzero(~mask)[0])
        order = np.r_[np.arange(start, n), np.arange(0, start)]
        mask_r = mask[order]
    runs, cur = [], []
    for k, i in enumerate(order):
        if not mask_r[k]:
            if cur:
                runs.append(np.array(cur))
            cur = []
            continue
        if cur and abs(theta[i] - theta[cur[-1]]) > np.pi / 2:
            runs.append(np.array(cur))
            cur = []
        cur.append(int(i))
    if cur:
        runs.append(np.array(cur))
    return [(r, False) for r in runs if len(r) >= 2]


# --------------------------------------------------------------------------
# certification along a curve
# --------------------------------------------------------------------------


def _blocking_margin(cert, penetration: float, limit_excess: float = 0.0) -> tuple:
    """(kind, margin > 0) for a certified-blocked fiber, or (None, 0).

    Joint-limit blockers give a margin in radians (how far the interval falls
    short). A collision blocker gives one in metres: the whole fiber is ONE
    posture (null motion), so its workspace penetration depth is the margin,
    and scene.check's value is a sampled LOWER bound on it -- the safe side.
    """
    if not cert.exact or cert.crossable:
        return None, 0.0
    if cert.empty and str(cert.blocker).startswith("collision"):
        return "collision", float(penetration)
    if cert.empty and str(cert.blocker).startswith("joint-limit:q"):
        # A joint that does not move along the fiber is out of range: the
        # margin is how far out it is.
        return f"constant-{cert.blocker.split(':')[1]}-out", float(limit_excess)
    if cert.empty:
        if np.isfinite(cert.lo) and np.isfinite(cert.hi) and cert.lo > cert.hi:
            return "no-admissible", float(cert.lo - cert.hi)
        return None, 0.0
    if not cert.at_zero:
        return "own-spelling-out", float(max(cert.lo, -cert.hi))
    return "short-of-pi", float(-cert.margin)


def certify_curve(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                  curve: DoorCurve, signs: tuple, p3: Page3Params) -> DoorCurve:
    """Is the WHOLE curve impassable? Only a positive answer discharges it.

    At each sampled point the 2-D closed form applies unchanged (the fiber is
    still 1-D on a surface task -- certify_fiber checks that itself). Between
    samples nothing is evaluated, so the curve is discharged only when

      * every sample is certified blocked by the SAME joint-limit mechanism,
        so one continuous margin (radians) describes them all, and
      * the smallest margin exceeds cert_lipschitz_safety x the largest
        change of the margin between neighbouring samples,

    i.e. the margin cannot reach zero between samples unless it varies far
    faster there than anywhere it was measured. That is a Lipschitz guard,
    not interval arithmetic (TODO 5): it is evidence, and it is labelled so.
    "Same mechanism" means one unit: joint-limit margins are radians, a
    collision margin is the penetration depth in metres. Anything else -- a
    crossable sample, an inexact one, mixed mechanisms -- leaves the curve
    LIVE (invariant 30).
    """
    idx = list(range(0, len(curve.pts), max(1, int(p3.cert_stride))))
    if idx[-1] != len(curve.pts) - 1:
        idx.append(len(curve.pts) - 1)
    kinds, margins = set(), []
    for i in idx:
        u, v, th = (float(x) for x in curve.pts[i])
        disc = SigmaDisc(kind=DOOR_INTERIOR, y=u, theta=th, radius=0.01,
                         vanishing=frozenset({curve.sign_index}),
                         exact_y=u, exact_theta=th)
        line = surf.line(v)
        cert = certify_fiber(cfg, arm, line, scene, disc, signs)
        if cert.crossable or not cert.exact:
            curve.certified_blocked, curve.unknown = False, not cert.crossable
            curve.reason = (f"sample at (u={u:.4f}, v={v:.4f}): {cert.reason}")
            curve.cert = dict(n_samples=len(idx), first_live=i)
            return curve
        pen = excess = 0.0
        if cert.empty and (str(cert.blocker).startswith("collision")
                           or str(cert.blocker).startswith("joint-limit:q")):
            r = arm.ik_branch(*line.pose(u), th, signs)
            if r.ok and str(cert.blocker).startswith("collision"):
                pen = scene.check(arm.frames(r.q)).penetration
            elif r.ok:
                j = int(cert.blocker.split(":q")[1]) - 1
                excess = float(max(arm.q_min[j] - r.q[j], r.q[j] - arm.q_max[j], 0.0))
        kind, m = _blocking_margin(cert, pen, excess)
        kinds.add(kind)
        margins.append(m)
    margins = np.asarray(margins)
    jump = float(np.max(np.abs(np.diff(margins)))) if len(margins) > 1 else 0.0
    unit = "mm" if kinds == {"collision"} else "deg"
    conv = (lambda x: 1e3 * x) if unit == "mm" else np.degrees
    curve.cert = dict(n_samples=len(idx), kinds=sorted(str(k) for k in kinds),
                      unit=unit, min_margin=float(conv(margins.min())),
                      max_jump=float(conv(jump)))
    if len(kinds) != 1 or None in kinds:
        curve.certified_blocked, curve.unknown = False, True
        curve.reason = (f"blocked at every sample, but by {sorted(str(k) for k in kinds)}"
                        f" -- no single margin to guard between samples; stays live")
        return curve
    if margins.min() <= p3.cert_lipschitz_safety * jump:
        curve.certified_blocked, curve.unknown = False, True
        curve.reason = (f"blocked at every sample, but the margin "
                        f"{conv(margins.min()):.4f} {unit} is within "
                        f"{p3.cert_lipschitz_safety:g} x the largest step "
                        f"{conv(jump):.4f} {unit}; stays live")
        return curve
    curve.certified_blocked, curve.unknown = True, False
    curve.reason = (f"blocked at all {len(idx)} samples ({kinds.pop()}), margin >= "
                    f"{conv(margins.min()):.4f} {unit} vs largest step "
                    f"{conv(jump):.4f} {unit} -> discharged (Lipschitz-guarded)")
    return curve
