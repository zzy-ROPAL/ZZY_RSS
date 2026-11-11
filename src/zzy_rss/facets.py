"""IJRR Algorithm 3 on simplices of any dimension (findings 41).

A facet of the wall is a (d-1)-simplex: a segment at d = 2, a triangle at
d = 3. Algorithm 3 accepts a facet when the ball that encloses it lies inside
C_obs, which it shows by

    Sigma guard  ->  enclosing ball (c, r)  ->  centre invalid?  ->  pd(c) >= r?

and, failing the last test, subdivides the facet and tries the pieces. None of
that depends on the robot, so it lives here; everything that does -- which
points are valid, how deep an invalid one is, where the doors are -- comes in
through a FacetOracle.

Geometry is done in SCALED coordinates (one unit means the same on every
axis), so the enclosing ball is a ball there; facets and centres are handed to
the oracle in UNSCALED page coordinates.

Soundness only needs SOME enclosing ball. The minimal one is used because a
smaller r passes more facets at depth 0; subdivision halves r per level for
both segments (1 -> 2) and triangles (1 -> 4 midpoint split), so
facet_max_depth means the same radius reduction in both dimensions. The cost
does not: a triangle that passes only at depth k pays up to 4^k leaf checks,
against 2^k for a segment (invariant 13 applies twice over).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import budget as _budget


@dataclass
class FacetResult:
    ok: bool
    reason: str = ""
    depth_used: float = 0.0
    radius: float = 0.0
    recursions: int = 0


@dataclass
class FacetOracle:
    """Everything Algorithm 3 needs to know about the page.

    guard(verts)  -> None, or the reason this facet may not exist at all
                     (it touches a door). Checked FIRST and at every level.
    virtual(c)    -> True if c is in the virtual invalid band (Fig. 5).
    valid(c)      -> True if c is a VALID configuration.
    depth(c, r)   -> (pd, how): penetration depth of an invalid c, in SCALED
                     units, so it compares directly with the ball radius r.
                     Only pd >= r matters, so an oracle may stop searching
                     once it knows that and return any value >= r.
    """

    scale: np.ndarray
    guard: Callable[[np.ndarray], str | None]
    virtual: Callable[[np.ndarray], bool]
    valid: Callable[[np.ndarray], bool]
    depth: Callable[[np.ndarray], tuple]
    # Optional batched forms (batch.py): valid_many(C) -> bool (N,), and
    # depth_many(C, R) -> pd (N,) with the same pd >= r verdicts as depth.
    valid_many: Callable | None = None
    depth_many: Callable | None = None
    # Test (ii) of the recursive proof (docs/recursive_proof_design.md):
    # sigma_many(C, R) -> bool (N,), True where the ball B(C_i, R_i) (scaled
    # radius) is CERTIFIED free of zeros of the chart singularity function
    # (chart.SrsChart.ball_test). With it set, a facet passes only if it is
    # covered by sigma-clear pieces down to sigma_max_depth subdivisions.
    # This replaces the door-list guard: it needs no door list.
    sigma_many: Callable | None = None
    sigma_max_depth: int = 8


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def enclosing_ball(verts: np.ndarray) -> tuple:
    """Minimum enclosing ball of a segment or triangle (any ambient d).

    Segment: its midpoint, half its length. Triangle: if it is right or
    obtuse, the midpoint of the longest edge (that ball already holds the
    third vertex); otherwise the circumcentre. Higher simplices fall back to
    the centroid with the farthest-vertex radius, which encloses but is not
    minimal.
    """
    v = np.asarray(verts, dtype=float)
    k = len(v) - 1
    if k == 0:
        return v[0].copy(), 0.0
    if k == 1:
        c = 0.5 * (v[0] + v[1])
        return c, 0.5 * float(np.linalg.norm(v[1] - v[0]))
    if k == 2:
        a, b, c = v
        e = [(np.dot(b - c, b - c), 0), (np.dot(c - a, c - a), 1), (np.dot(a - b, a - b), 2)]
        e.sort()
        (_, _), (_, _), (l_max, i_max) = e
        # Right or obtuse at the vertex opposite the longest edge?
        others = [x for x, _ in e[:2]]
        if l_max >= others[0] + others[1]:
            p, q = [(b, c), (c, a), (a, b)][i_max]
            m = 0.5 * (p + q)
            return m, 0.5 * float(np.sqrt(l_max))
        # Circumcentre in the triangle's own plane: a + s*u + t*w with
        # [u.u u.w; u.w w.w] [s t]^T = 0.5 [u.u w.w]^T.
        u, w = b - a, c - a
        uu, ww, uw = float(np.dot(u, u)), float(np.dot(w, w)), float(np.dot(u, w))
        det = uu * ww - uw * uw
        s = 0.5 * (uu * ww - ww * uw) / det
        t = 0.5 * (ww * uu - uu * uw) / det
        cc = a + s * u + t * w
        return cc, float(np.linalg.norm(cc - a))
    cen = v.mean(axis=0)
    return cen, float(np.max(np.linalg.norm(v - cen, axis=1)))


def subdivide(verts: np.ndarray) -> list:
    """Split a facet into pieces of half the size.

    Segment -> 2 halves. Triangle -> 4 by edge midpoints (each child is the
    parent scaled by 1/2, so its enclosing ball has exactly half the radius).
    """
    v = np.asarray(verts, dtype=float)
    k = len(v) - 1
    if k == 1:
        m = 0.5 * (v[0] + v[1])
        return [np.array([v[0], m]), np.array([m, v[1]])]
    if k == 2:
        a, b, c = v
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        return [np.array([a, ab, ca]), np.array([ab, b, bc]),
                np.array([ca, bc, c]), np.array([ab, bc, ca])]
    raise NotImplementedError(f"subdivision of a {k}-simplex")


# --------------------------------------------------------------------------
# Algorithm 3
# --------------------------------------------------------------------------


def sigma_clear_simplex(oracle: FacetOracle, verts: np.ndarray) -> tuple:
    """Test (ii): is the facet covered by sigma-clear pieces?

    Its own subdivision, separate from Algorithm 3's: it evaluates only the
    chart's sigma and its bound (no collision), so it can afford to go deep,
    and a facet is clear exactly when the union of its clear pieces is the
    facet. Level by level, batched. Returns (ok, reason, depth reached)."""
    scale = np.asarray(oracle.scale, dtype=float)
    pieces = [np.asarray(verts, dtype=float)]
    for depth in range(int(oracle.sigma_max_depth) + 1):
        balls = [enclosing_ball(p * scale) for p in pieces]
        C = np.array([cs / scale for cs, _ in balls])
        R = np.array([r for _, r in balls])
        clear = np.asarray(oracle.sigma_many(C, R), dtype=bool)
        bad = [p for p, ok in zip(pieces, clear) if not ok]
        if not bad:
            return True, "", depth
        if depth == int(oracle.sigma_max_depth):
            return False, (f"near sigma = 0: {len(bad)} piece(s) not certified clear "
                           f"at sigma depth {depth}"), depth
        pieces = [q for p in bad for q in subdivide(p)]
    return False, "unreachable", -1


def check_simplex(oracle: FacetOracle, verts: np.ndarray, max_depth: int,
                  depth: int = 0, sigma_done: bool = False) -> FacetResult:
    """IJRR Algorithm 3 on one facet, with the Sigma guard checked FIRST.

    The guard must precede the ball test: a facet that touches a door lets a
    path hop pages inside the door and bypass the wall, and no depth of C_obs
    around it changes that. With oracle.sigma_many set, test (ii) -- the
    certified sigma margin -- runs once, on the whole facet, before anything
    else (its pieces are then clear too, so the recursion below skips it).
    """
    verts = np.asarray(verts, dtype=float)
    if oracle.sigma_many is not None and depth == 0 and not sigma_done:
        ok, why, _ = sigma_clear_simplex(oracle, verts)
        if not ok:
            return FacetResult(False, why, recursions=0)
    why = oracle.guard(verts)
    if why:
        return FacetResult(False, why, recursions=depth)

    scale = np.asarray(oracle.scale, dtype=float)
    cs, radius = enclosing_ball(verts * scale)
    centre = cs / scale

    if oracle.virtual(centre):
        return FacetResult(True, "virtual boundary band", radius=radius,
                           recursions=depth)
    if oracle.valid(centre):
        return FacetResult(False, "facet centre is VALID", radius=radius,
                           recursions=depth)

    pd, how = oracle.depth(centre, radius)
    if pd >= radius:
        return FacetResult(True, how, depth_used=pd, radius=radius,
                           recursions=depth)
    if depth >= max_depth:
        return FacetResult(False, f"depth limit; pd={pd:.4g} < r={radius:.4g}",
                           depth_used=pd, radius=radius, recursions=depth)

    res = None
    for child in subdivide(verts):
        res = check_simplex(oracle, child, max_depth, depth + 1)
        if not res.ok:
            return res
    return res


# --------------------------------------------------------------------------
# Sigma guard at d = 3: a door is a CURVE, its neighbourhood a tube
# --------------------------------------------------------------------------


def _closest_on_triangle(p, a, b, c):
    """Closest point to p on triangle abc (Ericson, Real-Time Collision
    Detection, 5.1.5). Any ambient dimension."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def segment_segment_distance(p0, p1, q0, q1) -> float:
    """Exact distance between segments p0p1 and q0q1 (Ericson 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    eps = 1e-300
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def segment_triangle_distance(p0, p1, tri) -> float:
    """Exact distance between a segment and a triangle in 3-D.

    Zero if the segment pierces the triangle; otherwise the minimum lies on
    the boundary of one of them: an endpoint against the triangle, or a
    triangle edge against the segment.
    """
    a, b, c = tri
    n = np.cross(b - a, c - a)
    da, db = np.dot(p0 - a, n), np.dot(p1 - a, n)
    if da * db <= 0.0 and abs(da - db) > 1e-300:
        x = p0 + (da / (da - db)) * (p1 - p0)
        if np.linalg.norm(_closest_on_triangle(x, a, b, c) - x) <= 1e-12 * (
                1.0 + np.linalg.norm(x)):
            return 0.0
    best = min(np.linalg.norm(_closest_on_triangle(p0, a, b, c) - p0),
               np.linalg.norm(_closest_on_triangle(p1, a, b, c) - p1))
    for u, v in ((a, b), (b, c), (c, a)):
        best = min(best, segment_segment_distance(p0, p1, u, v))
    return float(best)


@dataclass
class SigmaTube:
    """The neighbourhood of a door CURVE: every point within an ellipsoidal
    radius of the polyline. Elliptical for the same reason the d = 2 discs
    are (invariant 6): the page axes differ in units, so `radii` is per
    axis, in unscaled page units."""

    polyline: np.ndarray        # (n, d) unscaled
    radii: np.ndarray           # (d,) unscaled
    kind: str = "door"


def triangle_hits_tube(tri: np.ndarray, tube: SigmaTube) -> bool:
    """Does a triangle enter the tube? Exact: map the ellipsoidal tube to a
    unit-radius round one, then compare segment-triangle distances with 1."""
    sc = 1.0 / np.maximum(np.asarray(tube.radii, dtype=float), 1e-12)
    t = np.asarray(tri, dtype=float) * sc
    pl = np.asarray(tube.polyline, dtype=float) * sc
    if len(pl) == 1:
        p = pl[0]
        return bool(np.linalg.norm(_closest_on_triangle(p, *t) - p) <= 1.0)
    # Cull first, vectorised: a segment further than r + 1 from the centre of
    # the triangle's enclosing ball cannot come within 1 of the triangle.
    # Only the survivors get the exact test, so the answer is still exact.
    c, r = enclosing_ball(t)
    p0, p1 = pl[:-1], pl[1:]
    d = p1 - p0
    dd = np.einsum("ij,ij->i", d, d)
    s = np.clip(np.einsum("ij,ij->i", c - p0, d) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    near = np.linalg.norm(p0 + s[:, None] * d - c, axis=1) <= r + 1.0
    for i in np.flatnonzero(near):
        if segment_triangle_distance(p0[i], p1[i], t) <= 1.0:
            return True
    return False


def tube_guard(tubes: list) -> Callable[[np.ndarray], str | None]:
    """A FacetOracle.guard for triangles: reject any facet touching a tube."""
    def guard(verts: np.ndarray) -> str | None:
        if len(verts) != 3:
            raise ValueError("tube_guard is for triangles")
        for tb in tubes:
            if triangle_hits_tube(verts, tb):
                return f"crosses Sigma tube ({tb.kind})"
        return None
    return guard


def check_many(oracle: FacetOracle, simplices: list, max_depth: int) -> list:
    """check_simplex over many facets, with depth 0 done in ONE batch.

    Most facets pass at depth 0 (findings 19: 66 of 73), so their whole check
    is one guard, one ball, one validity test and one depth march -- which is
    exactly what batches. Every facet that does not PASS at depth 0 (a guard
    hit, a valid centre, pd < r) goes through check_simplex unchanged, so
    failures, their reasons and every subdivision are the scalar path's.
    Without valid_many on the oracle this is a plain loop.

    Test (ii) (oracle.sigma_many) runs first for every facet: one batched
    call for all facets at depth 0, then sigma_clear_simplex for the few
    that are not clear at once."""
    sig_fail: dict = {}
    if oracle.sigma_many is not None and simplices:
        scale = np.asarray(oracle.scale, dtype=float)
        balls = [enclosing_ball(np.asarray(v, dtype=float) * scale) for v in simplices]
        C = np.array([cs / scale for cs, _ in balls])
        R = np.array([r for _, r in balls])
        clear0 = np.asarray(oracle.sigma_many(C, R), dtype=bool)
        for i in np.flatnonzero(~clear0):
            ok, why, _ = sigma_clear_simplex(oracle, np.asarray(simplices[i], dtype=float))
            if not ok:
                sig_fail[int(i)] = FacetResult(False, why)
    if oracle.valid_many is None or not simplices:
        out = []
        for i, v in enumerate(simplices):
            _budget.check_time()
            out.append(sig_fail[i] if i in sig_fail else
                       check_simplex(oracle, v, max_depth, sigma_done=True))
        return out
    scale = np.asarray(oracle.scale, dtype=float)
    n = len(simplices)
    out: list = [None] * n
    centres, radii, todo = [], [], []
    for i, v in enumerate(simplices):
        v = np.asarray(v, dtype=float)
        if i in sig_fail:
            out[i] = sig_fail[i]
            continue
        if oracle.guard(v):
            out[i] = check_simplex(oracle, v, max_depth, sigma_done=True)
            continue
        cs, r = enclosing_ball(v * scale)
        c = cs / scale
        if oracle.virtual(c):
            out[i] = FacetResult(True, "virtual boundary band", radius=r)
            continue
        centres.append(c)
        radii.append(r)
        todo.append(i)
    if todo:
        C, R = np.asarray(centres), np.asarray(radii)
        ok = oracle.valid_many(C)
        cand = np.flatnonzero(~ok)
        pd = np.full(len(todo), -np.inf)
        if len(cand):
            if oracle.depth_many is not None:
                pd[cand] = oracle.depth_many(C[cand], R[cand])
            else:
                pd[cand] = [oracle.depth(C[j], R[j])[0] for j in cand]
        for j, i in enumerate(todo):
            if not ok[j] and pd[j] >= R[j]:
                out[i] = FacetResult(True, "IJRR march (batched)",
                                     depth_used=float(pd[j]), radius=float(R[j]))
            else:
                _budget.check_time()
                out[i] = check_simplex(oracle, np.asarray(simplices[i], dtype=float),
                                       max_depth, sigma_done=True)
    return out
