"""Test (iii) of check(A, B): the doors INSIDE a closed wall (docs/recursive_proof_design.md).

A facet check only sees the wall's SURFACE. A door strictly inside the wall
touches no facet, so no facet test can see it -- yet the robot leaves the
page through it. Without a closed-form door list, the only way to know is to
search the enclosed region for zeros of the chart singularity function sigma.

The search is branch and bound over axis-aligned cells of the page box:

    cell outside the wall                    -> discard
    sigma_i(c) > L_i * rho for every i       -> discard (certified: no zero)
    cell small enough (rho <= tol)           -> CANDIDATE
    else                                     -> split into 2^d children

rho is the cell's half-diagonal in scaled units, so the cell lies in the ball
B(c, rho) the chart's bound is about. Candidates are clustered into connected
groups; each group is reported as ONE door region. Nothing is ever discarded
without a certificate, so every zero of sigma inside the wall lies in some
reported region: the conservative direction. A spurious region only costs
the recursion work or an INCONCLUSIVE; a missed one would be a false
INFEASIBLE, and that is what this search exists to rule out.

Sign changes are no use here: on a page sigma touches zero without crossing
it (invariant 31), and a door is a point at d = 2 and a curve at d = 3.
Only sigma is evaluated, never collision, so the search is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .chart import SIGMA_NAMES, SrsChart
from .prm import UnionFind

IN, OUT, CROSS = 1, 0, 2


# --------------------------------------------------------------------------
# where a cell is relative to the wall
# --------------------------------------------------------------------------


def _closest_on_triangles(P, A, B, C):
    """Vectorised Ericson 5.1.5: closest points on triangles (A_i, B_i, C_i)
    to points P_i. All (m, d)."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) > 1e-300, den, 1e-300)
    out = A + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]   # face region
    done = np.zeros(len(P), dtype=bool)

    def put(mask, val):
        nonlocal out, done
        m = mask & ~done
        out = np.where(m[:, None], val, out)
        done |= m

    put((d1 <= 0) & (d2 <= 0), A)
    put((d3 >= 0) & (d4 <= d3), B)
    t = d1 / np.where(np.abs(d1 - d3) > 1e-300, d1 - d3, 1e-300)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + t[:, None] * ab)
    put((d6 >= 0) & (d5 <= d6), C)
    t = d2 / np.where(np.abs(d2 - d6) > 1e-300, d2 - d6, 1e-300)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + t[:, None] * ac)
    e1, e2 = d4 - d3, d5 - d6
    t = e1 / np.where(np.abs(e1 + e2) > 1e-300, e1 + e2, 1e-300)
    put((va <= 0) & (e1 >= 0) & (e2 >= 0), B + t[:, None] * (C - B))
    return out


def _closest_on_segments(P, A, B):
    d = B - A
    dd = np.einsum("ij,ij->i", d, d)
    t = np.clip(np.einsum("ij,ij->i", P - A, d) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    return A + t[:, None] * d


def polytope_region(poly, scale) -> Callable:
    """region(C, R) -> codes IN / OUT / CROSS for cells with centres C
    (unscaled) and scaled radii R, against a closed wall `poly`
    (wall.Polytope; d = 2 loops or d = 3 triangle mesh).

    CROSS means the wall passes within R of the centre; otherwise the centre's
    side decides for the whole ball. Distances are exact (closest point on
    each nearby facet), so IN / OUT are never wrong; CROSS may be returned
    for a ball that only nearly touches, which costs a split, not soundness.
    """
    from .tangential import inside

    scale = np.asarray(scale, dtype=float)
    if poly.dim == 3:
        V = np.asarray(poly.vertices, dtype=float) * scale
        T = np.asarray(poly.simplices, dtype=int)
        FA, FB, FC = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        k = 2
    else:
        segs = [(np.asarray(lp[i]), np.asarray(lp[i + 1]))
                for lp in poly.loops for i in range(len(lp) - 1)]
        FA = np.array([a for a, _ in segs], dtype=float) * scale
        FB = np.array([b for _, b in segs], dtype=float) * scale
        FC = None
        V = np.vstack([FA, FB])
        T = np.arange(2 * len(FA)).reshape(2, -1).T
        k = 1
    if FC is None:
        cen = 0.5 * (FA + FB)
        rad = 0.5 * np.linalg.norm(FB - FA, axis=1)
    else:
        cen = (FA + FB + FC) / 3.0
        rad = np.max(np.stack([np.linalg.norm(X - cen, axis=1) for X in (FA, FB, FC)]),
                     axis=0)
    tree = cKDTree(cen)
    rmax = float(rad.max()) if len(rad) else 0.0

    def region(C, R):
        Cs = np.atleast_2d(np.asarray(C, dtype=float)) * scale
        R = np.broadcast_to(np.asarray(R, dtype=float), (len(Cs),))
        code = np.full(len(Cs), OUT, dtype=int)
        cross = np.zeros(len(Cs), dtype=bool)
        for i, (p, r) in enumerate(zip(Cs, R)):
            js = tree.query_ball_point(p, r + rmax)
            if not js:
                continue
            js = np.asarray(js, dtype=int)
            js = js[np.linalg.norm(cen[js] - p, axis=1) <= r + rad[js]]
            if not len(js):
                continue
            Pm = np.repeat(p[None, :], len(js), axis=0)
            if FC is None:
                q = _closest_on_segments(Pm, FA[js], FB[js])
            else:
                q = _closest_on_triangles(Pm, FA[js], FB[js], FC[js])
            if float(np.min(np.linalg.norm(q - Pm, axis=1))) <= r:
                cross[i] = True
        rest = ~cross
        if rest.any():
            if k == 2:
                ins = inside(Cs[rest], V, T, 2)
            else:
                ins = inside(Cs[rest], V, T, 1)
            code[np.flatnonzero(rest)] = np.where(ins, IN, OUT)
        code[cross] = CROSS
        return code

    return region


def whole_box(C, R):
    """region() for "everything": every cell counts as inside."""
    return np.full(len(np.atleast_2d(C)), IN, dtype=int)


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------


@dataclass
class DoorRegion:
    """One connected group of cells where sigma could not be excluded.

    Every zero of sigma in the searched region lies in the union of the
    cells of some DoorRegion. `kinds` names the sigma components that could
    not be excluded somewhere in it."""

    kinds: tuple
    centres: np.ndarray            # (m, d) unscaled
    half: np.ndarray               # (m, d) unscaled half-widths
    on_wall: bool                  # some cell is within reach of the wall
    sigma_min: np.ndarray          # (4,) smallest sigma seen at its centres

    @property
    def lo(self) -> np.ndarray:
        return (self.centres - self.half).min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return (self.centres + self.half).max(axis=0)

    def contains(self, P, slack: float = 1e-9) -> np.ndarray:
        """Which points lie in the union of the cells (unscaled)."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        inn = np.zeros(len(P), dtype=bool)
        for c, h in zip(self.centres, self.half):
            inn |= np.all(np.abs(P - c[None, :]) <= h[None, :] + slack, axis=1)
        return inn

    def label(self) -> str:
        return "+".join(self.kinds)


@dataclass
class SigmaSearch:
    doors: list = field(default_factory=list)       # DoorRegion
    complete: bool = True                            # False: max_cells hit
    stats: dict = field(default_factory=dict)


def find_doors(chart: SrsChart, region: Callable = whole_box, tol: float = 0.01,
               coarse: float = 0.25, max_cells: int = 2_000_000,
               lo=None, hi=None) -> SigmaSearch:
    """Certified search for zeros of sigma in `region` within the page box.

    tol, coarse: scaled half-diagonal at which a cell becomes a candidate,
    and the starting cell size (scaled edge length). Returns complete=False
    -- the search did NOT finish, so nothing may be concluded -- if more than
    max_cells cells would have to be evaluated."""
    scale = chart.scale
    d = chart.dim
    lo = np.asarray(chart.lo if lo is None else lo, dtype=float)
    hi = np.asarray(chart.hi if hi is None else hi, dtype=float)
    span = (hi - lo) * scale
    n0 = np.maximum(1, np.ceil(span / coarse).astype(int))
    h0 = (hi - lo) / n0 / 2.0
    axes = [lo[a] + h0[a] * (2 * np.arange(n0[a]) + 1) for a in range(d)]
    C = np.array(list(product(*axes)), dtype=float)
    H = np.repeat(h0[None, :], len(C), axis=0)
    signs = np.array(list(product((-1.0, 1.0), repeat=d)))

    cand_c, cand_h, cand_wall, cand_sig, cand_may = [], [], [], [], []
    evaluated, level, per_level = 0, 0, []
    while len(C):
        evaluated += len(C)
        if evaluated > max_cells:
            return SigmaSearch(complete=False, stats=dict(
                evaluated=evaluated, levels=per_level,
                reason=f"more than {max_cells} cells; search not finished"))
        R = np.linalg.norm(H * scale, axis=1)
        code = region(C, R)
        keep = code != OUT
        C, H, R, code = C[keep], H[keep], R[keep], code[keep]
        t = chart.ball_test(C, R)
        keep = ~t.clear
        per_level.append(dict(level=level, cells=int(len(keep)), open=int(keep.sum())))
        C, H, R, code = C[keep], H[keep], R[keep], code[keep]
        sig, may = t.sigma[keep], t.may_vanish[keep]
        small = R <= tol
        if small.any():
            cand_c.append(C[small])
            cand_h.append(H[small])
            cand_wall.append(code[small] == CROSS)
            cand_sig.append(sig[small])
            cand_may.append(may[small])
        C, H = C[~small], H[~small]
        if not len(C):
            break
        Hc = H / 2.0
        C = (C[:, None, :] + signs[None, :, :] * Hc[:, None, :]).reshape(-1, d)
        H = np.repeat(Hc, len(signs), axis=0)
        level += 1

    stats = dict(evaluated=evaluated, levels=per_level)
    if not cand_c:
        return SigmaSearch(doors=[], complete=True, stats=stats)
    Cc, Hc = np.vstack(cand_c), np.vstack(cand_h)
    Wc, Sc, Mc = np.concatenate(cand_wall), np.vstack(cand_sig), np.vstack(cand_may)
    doors = _cluster(chart, Cc, Hc, Wc, Sc, Mc)
    stats["candidates"] = int(len(Cc))
    return SigmaSearch(doors=doors, complete=True, stats=stats)


def _cluster(chart, C, H, W, S, M) -> list:
    """Connected groups of touching cells. theta wraps: cells on either side
    of the +/-pi cut touch when the page spans the whole circle."""
    scale = chart.scale
    n = len(C)
    uf = UnionFind(n)
    Cs = C * scale
    reach = 2.0 * float(np.max(np.linalg.norm(H * scale, axis=1))) + 1e-9
    period = None
    if np.isclose(chart.hi[-1] - chart.lo[-1], 2.0 * np.pi):
        period = 2.0 * np.pi

    def link(Cq, shift):
        tree = cKDTree(Cs)
        for i, js in enumerate(tree.query_ball_point(Cq * scale, reach)):
            for j in js:
                if i == j and shift == 0.0:
                    continue
                gap = np.abs(C[j] - Cq[i]) - (H[i] + H[j])
                if np.all(gap <= 1e-9):
                    uf.union(i, j)

    link(C, 0.0)
    if period is not None:
        Cp = C.copy()
        Cp[:, -1] += period
        link(Cp, period)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    out = []
    for idx in groups.values():
        idx = np.asarray(idx)
        kinds = tuple(SIGMA_NAMES[k] for k in range(4) if M[idx, k].any())
        with np.errstate(invalid="ignore"):
            smin = np.array([np.nanmin(S[idx, k]) if np.isfinite(S[idx, k]).any()
                             else np.nan for k in range(4)])
        out.append(DoorRegion(kinds=kinds, centres=C[idx], half=H[idx],
                              on_wall=bool(W[idx].any()), sigma_min=smin))
    out.sort(key=lambda r: (r.kinds, float(r.lo[0])))
    return out
