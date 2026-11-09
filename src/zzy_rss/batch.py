"""Batched validity: IK + joint limits + collision for many states at once.

`classify` (prm.py) answers one state at a time in ~200 us of Python, and
every stage calls it: roadmap samples, edge checks, near-door states, Alg 2
projections, Alg 3 facet centres, the depth march. This module computes the
SAME answer for N states with numpy, following arm.ik_branch, arm.within_limits
and Scene.check line by line -- same formulas, same branches, same thresholds.

Equality is TESTED, not assumed (tests/test_batch.py compares the verdicts
with classify on tens of thousands of states). Off-axis hub geometry is not
modelled here; with it switched on, the batch path falls back to the scalar
code so the answer never changes silently.
"""

from __future__ import annotations

import numpy as np

from .arm import INVALID_COLLISION, INVALID_JOINT_LIMIT, INVALID_NO_IK, VALID, SrsArm
from .params import Config
from .scene import Scene


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1),
                     np.stack([-s, z, c], -1)], -2)


def _norm(x):
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def ik_batch(arm: SrsArm, pos: np.ndarray, rot: np.ndarray, psi: np.ndarray,
             signs: tuple, coord_axis_tol: float = 1e-6) -> tuple:
    """arm.ik_branch for N states. pos (N, 3); rot (3, 3) shared or (N, 3, 3);
    psi (N,). Returns (ok (N,), q (N, 7)); q rows are undefined where not ok."""
    p = arm.p
    s2, s4, s6 = signs
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    n = len(pos)
    rot = np.asarray(rot, dtype=float)
    rotn = np.broadcast_to(rot, (n, 3, 3)) if rot.ndim == 2 else rot
    psi = np.asarray(psi, dtype=float).reshape(n)
    S = arm.shoulder

    wrist = pos - p.d_wt * rotn[:, :, 2]
    x_sw = wrist - S
    length = _norm(x_sw)
    ok = length >= 1e-12
    L = np.where(ok, length, 1.0)
    cos_q4 = (L ** 2 - p.d_se ** 2 - p.d_ew ** 2) / (2.0 * p.d_se * p.d_ew)
    ok &= (cos_q4 <= 1.0) & (cos_q4 >= -1.0)
    q4 = s4 * np.arccos(np.clip(cos_q4, -1.0, 1.0))

    u = x_sw / L[:, None]
    h = (p.d_se ** 2 - p.d_ew ** 2 + L ** 2) / (2.0 * L)
    radius = np.sqrt(np.maximum(0.0, p.d_se ** 2 - h ** 2))
    near_axis = np.abs(u[:, 2]) > 1.0 - coord_axis_tol
    ref = np.where(near_axis[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    v0 = ref - np.einsum("ij,ij->i", ref, u)[:, None] * u
    n_v0 = _norm(v0)
    ok &= n_v0 >= 1e-12
    v0 = v0 / np.where(n_v0 >= 1e-12, n_v0, 1.0)[:, None]
    w0 = np.cross(u, v0)
    elbow = S + h[:, None] * u + radius[:, None] * (np.cos(psi)[:, None] * v0
                                                    + np.sin(psi)[:, None] * w0)

    d_u = elbow - S
    n_du = _norm(d_u)
    ok &= n_du >= 1e-12
    d_u = d_u / np.where(n_du >= 1e-12, n_du, 1.0)[:, None]
    q2 = s2 * np.arccos(np.clip(d_u[:, 2], -1.0, 1.0))
    q1 = (np.arctan2(d_u[:, 1], d_u[:, 0]) if s2 >= 0
          else np.arctan2(-d_u[:, 1], -d_u[:, 0]))

    d_f = wrist - elbow
    n_df = _norm(d_f)
    ok &= n_df >= 1e-12
    d_f = d_f / np.where(n_df >= 1e-12, n_df, 1.0)[:, None]
    cr = np.cross(d_u, d_f)
    n_cr = _norm(cr)
    small = n_cr < 1e-9
    n_target = s4 * (cr / np.where(small, 1.0, n_cr)[:, None])
    r02 = _rot_z(q1) @ _rot_y(q2)
    m = np.einsum("nji,nj->ni", r02, n_target)          # r02.T @ n_target
    q3 = np.where(small, 0.0, np.arctan2(-m[:, 0], m[:, 1]))

    r04 = r02 @ _rot_z(q3) @ _rot_y(q4)
    r47 = np.einsum("nji,njk->nik", r04, rotn)            # r04.T @ rot
    c6 = np.clip(r47[:, 2, 2], -1.0, 1.0)
    q6 = s6 * np.arccos(c6)
    if s6 >= 0:
        q5 = np.arctan2(r47[:, 1, 2], r47[:, 0, 2])
        q7 = np.arctan2(r47[:, 2, 1], -r47[:, 2, 0])
    else:
        q5 = np.arctan2(-r47[:, 1, 2], -r47[:, 0, 2])
        q7 = np.arctan2(-r47[:, 2, 1], r47[:, 2, 0])
    q = np.stack([q1, q2, q3, q4, q5, q6, q7], axis=1)
    return ok, q


def frames_batch(arm: SrsArm, q: np.ndarray) -> tuple:
    """Shoulder, elbow, wrist and tool positions for N joint vectors (arm.frames)."""
    p = arm.p
    r02 = _rot_z(q[:, 0]) @ _rot_y(q[:, 1])
    r04 = r02 @ _rot_z(q[:, 2]) @ _rot_y(q[:, 3])
    r07 = r04 @ _rot_z(q[:, 4]) @ _rot_y(q[:, 5]) @ _rot_z(q[:, 6])
    elbow = arm.shoulder + p.d_se * r02[:, :, 2]
    wrist = elbow + p.d_ew * r04[:, :, 2]
    tool = wrist + p.d_wt * r07[:, :, 2]
    return elbow, wrist, tool


def collision_batch(arm: SrsArm, scene: Scene, q: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Scene.check(...).in_collision for N joint vectors: same capsules, same
    n_coarse sample points per axis, same clearance test."""
    n = len(q)
    out = np.zeros(n, dtype=bool)
    if not scene.boxes:
        return out
    p = scene.arm_params
    radii = np.array([p.r_upper_arm, p.r_forearm, p.r_tool])
    ts = np.linspace(0.0, 1.0, scene.n_coarse)[None, None, :, None]
    for s in range(0, n, chunk):
        elbow, wrist, tool = frames_batch(arm, q[s:s + chunk])
        m = len(elbow)
        shoulder = np.broadcast_to(arm.shoulder, (m, 3))
        a = np.stack([shoulder, elbow, wrist], axis=1)          # (m, 3, 3)
        b = np.stack([elbow, wrist, tool], axis=1)
        pts = a[:, :, None, :] + ts * (b - a)[:, :, None, :]     # (m, caps, n_s, 3)
        hit = np.zeros(m, dtype=bool)
        for center, half, _label in scene.boxes:
            delta = np.abs(pts - center) - half
            dist = np.sqrt(np.sum(np.maximum(delta, 0.0) ** 2, axis=3))
            clear = dist.min(axis=2) - radii                      # (m, caps)
            hit |= (clear < 0.0).any(axis=1)
        out[s:s + chunk] = hit
    return out


def classify_batch(cfg: Config, arm: SrsArm, scene: Scene, pos: np.ndarray,
                   rot: np.ndarray, psi: np.ndarray, signs: tuple) -> tuple:
    """classify for N states given their tool poses and arm angles.

    Returns (valid (N,), tag (N,), q (N, 7))."""
    p = arm.p
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    n = len(pos)
    tag = np.full(n, VALID, dtype=int)
    if getattr(p, "use_off_axis_upper_arm", False) or getattr(p, "use_off_axis_elbow", False):
        # Hub geometry is not modelled in batch form: fall back, exactly.
        rotn = np.broadcast_to(rot, (n, 3, 3)) if np.ndim(rot) == 2 else rot
        q = np.zeros((n, 7))
        for i in range(n):
            r = arm.ik_branch(pos[i], rotn[i], float(psi[i]), signs,
                              coord_axis_tol=cfg.pagemap.coord_axis_tol,
                              coord_radius_tol=cfg.pagemap.coord_radius_tol)
            if not r.ok:
                tag[i] = INVALID_NO_IK
                continue
            q[i] = r.q
            if not arm.within_limits(r.q):
                tag[i] = INVALID_JOINT_LIMIT
            elif scene.check(arm.frames(r.q)).in_collision:
                tag[i] = INVALID_COLLISION
        return tag == VALID, tag, q
    ok, q = ik_batch(arm, pos, rot, psi, signs, coord_axis_tol=cfg.pagemap.coord_axis_tol)
    tag[~ok] = INVALID_NO_IK
    lim = np.all((q >= arm.q_min) & (q <= arm.q_max), axis=1)
    tag[ok & ~lim] = INVALID_JOINT_LIMIT
    rest = ok & lim
    if rest.any():
        col = collision_batch(arm, scene, q[rest])
        idx = np.flatnonzero(rest)
        tag[idx[col]] = INVALID_COLLISION
    return tag == VALID, tag, q


def classify_line_batch(cfg: Config, arm: SrsArm, weld, scene: Scene, pts: np.ndarray,
                        signs: tuple) -> tuple:
    """classify for N (y, theta) states on a WeldLine."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    pos = weld.p0 + pts[:, :1] * weld.direction
    return classify_batch(cfg, arm, scene, pos, weld.rot, pts[:, 1], signs)


def classify_surface_batch(cfg: Config, arm: SrsArm, surf, scene: Scene, pts: np.ndarray,
                           signs: tuple) -> tuple:
    """classify for N (u, v, theta) states on a WeldSurface."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    pos = surf.p0 + pts[:, :1] * surf.du + pts[:, 1:2] * surf.dv
    return classify_batch(cfg, arm, scene, pos, surf.rot, pts[:, 2], signs)


def classify_all_pages_line_batch(cfg: Config, arm: SrsArm, weld, scene: Scene,
                                  pts: np.ndarray) -> dict:
    """source.classify_all_pages for N (y, theta) states: {signs: valid (N,)}.

    The same mechanism (invariant 19): IK existence is sign-free, collision is
    the posture's (computed once, on the reference branch), only the joint
    limits depend on the page -- here read off each branch's own IK."""
    from .arm import SIGN_BRANCHES
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    pos = weld.p0 + pts[:, :1] * weld.direction
    ref = SIGN_BRANCHES[0]
    ok, q = ik_batch(arm, pos, weld.rot, pts[:, 1], ref,
                     coord_axis_tol=cfg.pagemap.coord_axis_tol)
    hit = np.zeros(len(pts), dtype=bool)
    if ok.any():
        hit[ok] = collision_batch(arm, scene, q[ok])
    out = {}
    for sg in SIGN_BRANCHES:
        okb, qb = (ok, q) if sg == ref else ik_batch(
            arm, pos, weld.rot, pts[:, 1], sg, coord_axis_tol=cfg.pagemap.coord_axis_tol)
        lim = np.all((qb >= arm.q_min) & (qb <= arm.q_max), axis=1)
        out[sg] = ok & okb & lim & ~hit
    return out


def pd_march_batch(points: np.ndarray, valid_seeds: np.ndarray, valid_many,
                   scale: np.ndarray, step: float, attempts: int, bisect_iters: int,
                   caps: np.ndarray) -> np.ndarray:
    """wall.pd_march for N points at once (same answer, see the note below).

    Each point marches toward its `attempts` nearest seeds, `step` at a time,
    up to min(seed distance, nearest-seed distance, cap); the first valid
    sample is bisected. The scalar version shortens later rays as it goes
    (only a hit closer than the best so far can matter); this one marches
    every ray to the INITIAL bound. Hits beyond the shortened bound are
    farther than the best already found, so the minimum -- the answer -- is
    the same. Returns pd (N,), capped as pd_march does."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.zeros(0)
    scale = np.asarray(scale, dtype=float)
    k = min(int(attempts), len(valid_seeds))
    d, idx = cKDTree(valid_seeds * scale).query(points * scale, k=k)
    d = np.asarray(d).reshape(n, k)
    idx = np.asarray(idx).reshape(n, k)
    best = d[:, 0].copy()
    caps = np.asarray(caps, dtype=float).reshape(n)
    rays = []                                   # (point i, seed, length, n_steps)
    for i in range(n):
        if d[i, 0] <= 0.0:
            best[i] = 0.0
            continue
        for a in range(k):
            length = float(d[i, a])
            m = int(np.ceil(min(length, best[i], caps[i]) / step))
            if m > 0:
                rays.append((i, valid_seeds[idx[i, a]], length, m))
    if rays:
        # every march sample of every ray, classified in one call
        owner, tvals, xs = [], [], []
        for r, (i, seed, length, m) in enumerate(rays):
            for s in range(1, m + 1):
                t = min(1.0, s * step / length)
                owner.append(r)
                tvals.append(t)
                xs.append(points[i] + t * (seed - points[i]))
        ok = valid_many(np.asarray(xs))
        owner = np.asarray(owner)
        tvals = np.asarray(tvals)
        lo = np.zeros(len(rays))
        hi = np.full(len(rays), np.nan)
        for j in range(len(owner)):             # first valid sample per ray
            r = owner[j]
            if not np.isnan(hi[r]):
                continue
            if ok[j]:
                hi[r] = tvals[j]
            else:
                lo[r] = tvals[j]
        live = np.flatnonzero(~np.isnan(hi))
        if len(live):
            P = np.array([points[rays[r][0]] for r in live])
            Sd = np.array([rays[r][1] for r in live])
            lo_l, hi_l = lo[live], hi[live]
            for _ in range(int(bisect_iters)):
                mid = 0.5 * (lo_l + hi_l)
                okm = valid_many(P + mid[:, None] * (Sd - P))
                hi_l = np.where(okm, mid, hi_l)
                lo_l = np.where(okm, lo_l, mid)
            for r, h in zip(live, hi_l):
                i, _seed, length, _m = rays[r]
                best[i] = min(best[i], h * length)
    return np.where(best > caps, caps, best)
