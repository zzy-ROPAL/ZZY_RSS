"""IJRR 4.3 -- reconstruct the wall with the tangential Delaunay complex.

No grid is evaluated anywhere in this module. The wall is an implicit
hypersurface M = {f = 0} of intrinsic dimension k = d - 1 in a d-dimensional
page, and everything below is built from POINTS ON M:

  1. sample_manifold_net   an (epsilon, eta)-net of M (IJRR Definition 2),
                           grown from seed points by projecting onto M and
                           stepping along the tangent space. Spacing >= eta
                           (sparse), and every frontier point keeps stepping
                           until no gap wider than ~2 eta is left (dense).
  2. tangential_complex    Boissonnat & Ghosh's tangential Delaunay complex:
                           for each sample p, the star of p in the WEIGHTED
                           Delaunay triangulation of its neighbours projected
                           onto the tangent space T_p. The weight of a
                           neighbour q is minus its squared height above T_p,
                           which makes the triangulation inside T_p exactly
                           the ambient Delaunay complex restricted to T_p.
                           A simplex is kept only if it is in the star of
                           EVERY one of its vertices (consistency).
  3. fix by weights        inconsistent simplices are removed the way the
                           theory does it -- by perturbing small point
                           weights and recomputing the affected stars.
  4. verify                closed (every (k-1)-face shared by exactly two
                           k-simplices, every vertex link a single cycle for
                           k = 2), consistently oriented, and, given labelled
                           points, on the same side of the complex as the
                           sign of f says.

Nothing here has to be trusted for soundness. The infeasibility claim rests on
(i) the complex being closed, (ii) start and goal on opposite sides, and
(iii) Algorithm 3 accepting every facet. Steps 4 and Algorithm 3 check those
directly, so a bad reconstruction can only cost a verdict (INCONCLUSIVE), never
manufacture a false INFEASIBLE.

Why exact tangents: the wall is an RBF-SVM, so the normal at a sample is the
closed-form gradient of f, not a PCA estimate from neighbours. Estimating
tangent spaces is the generic TDC's weakest step; here it is exact.

Coordinates: everything in this module is in SCALED page coordinates (the
SVM's own input space), so one distance means the same thing on every axis.
The `field` callable maps (n, d) scaled points to (f (n,), grad (n, d)).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

Field = Callable[[np.ndarray], tuple]


# --------------------------------------------------------------------------
# Projection onto M (IJRR Eq. 2), batched
# --------------------------------------------------------------------------


def project_batch(field: Field, pts: np.ndarray, iters: int = 40,
                  tol: float = 1.0e-10, accept: float = 1.0e-7) -> tuple:
    """Newton onto f = 0 for many points at once.

    z <- z - f(z) grad / |grad|^2, i.e. the same step as wall.project_to_wall,
    vectorised. Returns (points, ok-mask). A point whose gradient vanishes or
    that does not reach |f| <= accept is marked not ok and must be ignored:
    near a critical point of f there is no tangent space to step along.
    """
    z = np.array(pts, dtype=float, copy=True)
    if z.size == 0:
        return z.reshape(0, pts.shape[1] if pts.ndim == 2 else 0), np.zeros(0, bool)
    live = np.ones(len(z), dtype=bool)
    for _ in range(iters):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        f, g = field(z[idx])
        gg = np.einsum("ij,ij->i", g, g)
        bad = ~np.isfinite(gg) | (gg < 1e-18) | ~np.isfinite(f)
        done = np.abs(f) < tol
        step = np.where(bad | done, 0.0, f / np.where(bad, 1.0, gg))
        z[idx] -= step[:, None] * g
        live[idx[bad | done]] = False
    f, g = field(z)
    gg = np.einsum("ij,ij->i", g, g)
    ok = (np.abs(f) <= accept) & np.isfinite(gg) & (gg > 1e-18) & np.all(np.isfinite(z), axis=1)
    return z, ok


def unit_normals(field: Field, pts: np.ndarray) -> np.ndarray:
    _, g = field(pts)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.where(n > 0, n, 1.0)


def tangent_bases(normals: np.ndarray) -> np.ndarray:
    """(n, d) unit normals -> (n, d, k) orthonormal tangent bases.

    Householder reflection taking e_1 to the normal; its remaining columns
    span the orthogonal complement. Vectorised, any d.
    """
    n, d = normals.shape
    e1 = np.zeros(d)
    e1[0] = 1.0
    s = np.where(normals[:, :1] >= 0, 1.0, -1.0)
    v = normals + s * e1                       # never small: |v| >= 1
    vv = np.einsum("ij,ij->i", v, v)
    H = np.eye(d)[None, :, :] - 2.0 * v[:, :, None] * v[:, None, :] / vv[:, None, None]
    return H[:, :, 1:]


# --------------------------------------------------------------------------
# Step 1 -- an (epsilon, eta)-net of the manifold, grown by tangent stepping
# --------------------------------------------------------------------------


def _greedy_net(cands: np.ndarray, existing: cKDTree | None, r: float) -> np.ndarray:
    """Indices of a maximal subset of `cands` with pairwise spacing >= r that
    also keeps >= r from every point already in `existing`."""
    if len(cands) == 0:
        return np.zeros(0, dtype=int)
    keep_mask = np.ones(len(cands), dtype=bool)
    if existing is not None and existing.n > 0:
        d, _ = existing.query(cands, k=1)
        keep_mask &= d >= r
    idx = np.flatnonzero(keep_mask)
    if idx.size <= 1:
        return idx
    tree = cKDTree(cands[idx])
    pairs = tree.query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return idx
    nbrs: dict = {}
    for a, b in pairs:
        nbrs.setdefault(int(a), []).append(int(b))
        nbrs.setdefault(int(b), []).append(int(a))
    taken = np.zeros(idx.size, dtype=bool)
    blocked = np.zeros(idx.size, dtype=bool)
    for i in range(idx.size):
        if blocked[i]:
            continue
        taken[i] = True
        for j in nbrs.get(i, ()):
            blocked[j] = True
    return idx[taken]


def sample_manifold_net(
    field: Field, seeds: np.ndarray, eta: float, lo: np.ndarray, hi: np.ndarray,
    step: float = 1.0, accept: float = 0.8, max_points: int = 20000,
    max_rounds: int = 100000, rng: np.random.Generator | None = None,
) -> tuple:
    """Grow an (epsilon, eta)-net of M from seed points. Returns (pts, stats).

    Seeds are projected onto M; a greedy net with spacing accept*eta is kept;
    then every newly accepted point proposes neighbours one step*eta away
    along its tangent space (two directions for a curve, six for a surface),
    which are projected back onto M and accepted if they keep the spacing.
    The loop stops when a round accepts nothing: no point of the net has an
    empty tangent neighbourhood left, which is the density half of the net.

    Only the part of M reachable from a seed is sampled. That is deliberate:
    the proof object is the reconstructed complex, not the zero set, and the
    sidedness check in `reconstruct` catches a missed component that matters
    (one that separates two labelled points).
    """
    rng = rng or np.random.default_rng(0)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = len(lo)
    k = d - 1
    r_acc = accept * eta

    def in_box(z):
        return np.all((z >= lo) & (z <= hi), axis=1)

    z, ok = project_batch(field, np.asarray(seeds, dtype=float).reshape(-1, d))
    z = z[ok & in_box(z)]
    sel = _greedy_net(z, None, r_acc)
    net = z[sel]
    frontier = net.copy()
    rounds = 0
    n_proposed = 0
    capped = False
    while len(frontier) and rounds < max_rounds:
        rounds += 1
        normals = unit_normals(field, frontier)
        T = tangent_bases(normals)                         # (m, d, k)
        if k == 1:
            dirs = np.stack([T[:, :, 0], -T[:, :, 0]], axis=1)   # (m, 2, d)
        else:
            # Six directions in the tangent plane with a random phase per
            # point, so no axis-aligned pattern is imposed on the net. For
            # k > 2 a random set of 4k unit directions plays the same role.
            nd = 6 if k == 2 else 4 * k
            if k == 2:
                ph = rng.uniform(0.0, np.pi / 3.0, size=len(frontier))
                ang = ph[:, None] + np.arange(nd)[None, :] * (np.pi / 3.0)
                coef = np.stack([np.cos(ang), np.sin(ang)], axis=2)  # (m, nd, 2)
            else:
                coef = rng.normal(size=(len(frontier), nd, k))
                coef /= np.linalg.norm(coef, axis=2, keepdims=True)
            dirs = np.einsum("mdk,mnk->mnd", T, coef)
        props = (frontier[:, None, :] + step * eta * dirs).reshape(-1, d)
        n_proposed += len(props)
        zp, okp = project_batch(field, props)
        zp = zp[okp & in_box(zp)]
        sel = _greedy_net(zp, cKDTree(net), r_acc)
        frontier = zp[sel]
        if len(frontier):
            net = np.vstack([net, frontier])
        if len(net) >= max_points:
            capped = True
            break
    return net, dict(n=len(net), rounds=rounds, proposed=n_proposed, capped=capped)


# --------------------------------------------------------------------------
# Step 2 -- the tangential Delaunay complex
# --------------------------------------------------------------------------


def _star(i: int, pts: np.ndarray, normals: np.ndarray, bases: np.ndarray,
          nbr: np.ndarray, omega: np.ndarray) -> list:
    """Star of point i in the weighted Delaunay triangulation of its
    neighbours projected onto T_i. Returns k-simplices as sorted tuples of
    GLOBAL indices, each containing i.

    Power distance from u in T_i to a projected neighbour q:
        |u - u_q|^2 + h_q^2 - omega_q^2
    with h_q the height of q above T_i. On T_i this equals the ambient power
    distance, so the regular triangulation computed here is the ambient
    (weighted) Delaunay complex restricted to T_i -- the definition of the
    tangential complex. It is computed as the lower convex hull of the lifted
    points (u_q, |u_q|^2 + h_q^2 - omega_q^2).
    """
    loc = nbr[i]
    k = bases.shape[2]
    rel = pts[loc] - pts[i]
    u = rel @ bases[i]                                  # (K, k)
    h = rel @ normals[i]
    lift = np.einsum("ij,ij->i", u, u) + h * h - omega[loc] ** 2
    P = np.column_stack([u, lift])
    if len(P) < k + 2:
        return []
    try:
        hull = ConvexHull(P, qhull_options="Qt QbB")
    except (QhullError, ValueError):
        try:
            hull = ConvexHull(P, qhull_options="QJ QbB")
        except (QhullError, ValueError):
            return []
    out = []
    lower = hull.equations[:, -2] < -1e-12            # outward normal points down
    for simp, is_low in zip(hull.simplices, lower):
        if not is_low or 0 not in simp:               # loc[0] is i itself
            continue
        out.append(tuple(sorted(int(loc[s]) for s in simp)))
    return out


@dataclass
class TangentialComplex:
    points: np.ndarray                      # (n, d) scaled
    normals: np.ndarray                     # (n, d) unit, pointing to f > 0
    simplices: np.ndarray                   # (m, k+1) consistent, sorted
    n_inconsistent: int
    fix_rounds: int
    stars: list = dc_field(default_factory=list, repr=False)
    inconsistent: list = dc_field(default_factory=list, repr=False)


def tangential_complex(
    pts: np.ndarray, normals: np.ndarray, k_neighbors: int = 0,
    fix_rounds: int = 20, weight_frac: float = 0.3,
    rng: np.random.Generator | None = None,
) -> TangentialComplex:
    """Build the TDC of `pts` with the given (exact) unit normals.

    Inconsistencies are removed the way Boissonnat & Ghosh do it: give the
    points involved small random weights (omega_q <= weight_frac * the
    distance to q's nearest neighbour) and recompute the stars they affect,
    for up to `fix_rounds` rounds. Whatever stays inconsistent is dropped,
    and the closure check then decides whether what is left is a valid
    proof object.
    """
    rng = rng or np.random.default_rng(0)
    n, d = pts.shape
    k = d - 1
    if n < k + 2:
        return TangentialComplex(pts, normals, np.zeros((0, k + 1), int), 0, 0)
    K = k_neighbors or (10 if k == 1 else 24)
    K = min(K, n)
    tree = cKDTree(pts)
    dist, nbr = tree.query(pts, k=K)
    nbr = np.asarray(nbr).reshape(n, K)
    nn = np.asarray(dist).reshape(n, K)[:, 1] if K > 1 else np.full(n, 1.0)
    bases = tangent_bases(normals)
    omega = np.zeros(n)

    # Which stars a point's weight can change: every point that lists it as
    # a neighbour.
    users: list = [[] for _ in range(n)]
    for i in range(n):
        for j in nbr[i]:
            users[int(j)].append(i)

    stars = [_star(i, pts, normals, bases, nbr, omega) for i in range(n)]

    def tally():
        cnt: dict = {}
        for s in stars:
            for simp in s:
                cnt[simp] = cnt.get(simp, 0) + 1
        good = [s for s, c in cnt.items() if c == k + 1]
        bad = [s for s, c in cnt.items() if c != k + 1]
        return good, bad

    good, bad = tally()
    rounds = 0
    while bad and rounds < fix_rounds:
        rounds += 1
        involved = sorted({v for s in bad for v in s})
        omega[involved] = weight_frac * nn[involved] * rng.uniform(0.0, 1.0, len(involved))
        affected = sorted({u for v in involved for u in users[v]} | set(involved))
        for i in affected:
            stars[i] = _star(i, pts, normals, bases, nbr, omega)
        good, bad = tally()

    simp = (np.asarray(sorted(good), dtype=int).reshape(-1, k + 1)
            if good else np.zeros((0, k + 1), dtype=int))
    return TangentialComplex(pts, normals, simp, len(bad), rounds, stars,
                             inconsistent=list(bad))


# --------------------------------------------------------------------------
# Step 4 -- verification: closed, oriented, on the right side
# --------------------------------------------------------------------------


def closure_report(simplices: np.ndarray, k: int) -> tuple:
    """Is the complex a closed k-manifold? Returns (ok, note)."""
    if len(simplices) == 0:
        return False, "empty complex"
    if k == 1:
        deg = np.bincount(simplices.ravel())
        used = deg[deg > 0]
        if np.any(used != 2):
            n1 = int(np.sum(used == 1))
            n3 = int(np.sum(used >= 3))
            return False, f"not closed: {n1} curve end(s), {n3} branch vertex(es)"
        return True, ""
    if k == 2:
        edges: dict = {}
        for t in simplices:
            a, b, c = (int(x) for x in t)
            for e in ((a, b), (b, c), (a, c)):
                e = (min(e), max(e))
                edges[e] = edges.get(e, 0) + 1
        bad = [e for e, c in edges.items() if c != 2]
        if bad:
            n1 = sum(1 for e in bad if edges[e] == 1)
            return False, (f"not watertight: {n1} boundary edge(s), "
                           f"{len(bad) - n1} non-manifold edge(s)")
        # Vertex links must be single cycles, or two sheets touch at a point
        # (a pinch) and "inside" stops meaning anything.
        link: dict = {}
        for t in simplices:
            a, b, c = (int(x) for x in t)
            for v, e in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
                link.setdefault(v, []).append(e)
        for v, es in link.items():
            adj: dict = {}
            for x, y in es:
                adj.setdefault(x, []).append(y)
                adj.setdefault(y, []).append(x)
            start = es[0][0]
            seen = {start}
            stack = [start]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if len(seen) != len(adj):
                return False, f"pinched vertex {v}: its link is not one cycle"
        return True, ""
    return False, f"closure check not implemented for k={k}"


def extract_loops(edges: np.ndarray) -> list:
    """Closed 1-manifold (every vertex degree 2) -> ordered vertex cycles,
    each with its first vertex repeated at the end."""
    adj: dict = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen = set()
    loops = []
    for s in adj:
        if s in seen:
            continue
        cyc = [s]
        seen.add(s)
        prev, cur = s, adj[s][0]
        while cur != s:
            cyc.append(cur)
            seen.add(cur)
            a, b = adj[cur]
            prev, cur = cur, (b if a == prev else a)
        cyc.append(s)
        loops.append(np.asarray(cyc, dtype=int))
    return loops


def orient_triangles(pts: np.ndarray, tris: np.ndarray, normals: np.ndarray) -> tuple:
    """Orient each triangle so its normal points to f < 0 (outside), and check
    the orientation is coherent: every edge must be traversed once in each
    direction by its two triangles. Returns (oriented, ok, note)."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    nrm = np.cross(b - a, c - a)
    grad = normals[tris].mean(axis=1)                  # points to f > 0
    s = np.einsum("ij,ij->i", nrm, -grad)
    if np.any(np.abs(s) < 1e-15):
        return tris, False, "degenerate triangle (zero area or edge-on to the gradient)"
    out = tris.copy()
    flip = s < 0
    out[flip] = out[flip][:, [0, 2, 1]]
    directed: dict = {}
    for t in out:
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (int(e[0]), int(e[1]))
            directed[e] = directed.get(e, 0) + 1
    for (x, y), cnt in directed.items():
        if cnt != 1 or directed.get((y, x), 0) != 1:
            return out, False, ("incoherent orientation: a triangle folds "
                                "against the gradient of f")
    return out, True, ""


def inside(query: np.ndarray, pts: np.ndarray, simplices: np.ndarray,
           k: int) -> np.ndarray:
    """Which query points the closed complex encloses.

    k = 1: even-odd ray casting over every edge (works for any number of
    nested loops). k = 2: generalised winding number over the coherently
    oriented triangles (Van Oosterom & Strackee solid angles), rounded.
    """
    q = np.atleast_2d(np.asarray(query, dtype=float))
    if k == 1:
        A = pts[simplices[:, 0]]
        B = pts[simplices[:, 1]]
        ty = q[:, 1][:, None]
        tx = q[:, 0][:, None]
        a1, b1 = A[:, 1][None, :], B[:, 1][None, :]
        straddle = (a1 > ty) != (b1 > ty)
        den = np.where(straddle, b1 - a1, 1.0)
        xint = A[:, 0][None, :] + (ty - a1) * (B[:, 0] - A[:, 0])[None, :] / den
        hits = straddle & (xint > tx)
        return (np.sum(hits, axis=1) % 2) == 1
    if k == 2:
        wn = np.zeros(len(q))
        for chunk in range(0, len(simplices), 4096):
            t = simplices[chunk:chunk + 4096]
            A = pts[t[:, 0]][None, :, :] - q[:, None, :]
            B = pts[t[:, 1]][None, :, :] - q[:, None, :]
            C = pts[t[:, 2]][None, :, :] - q[:, None, :]
            la, lb, lc = (np.linalg.norm(X, axis=2) for X in (A, B, C))
            det = np.einsum("qtd,qtd->qt", A, np.cross(B, C))
            den = (la * lb * lc + np.einsum("qtd,qtd->qt", A, B) * lc
                   + np.einsum("qtd,qtd->qt", B, C) * la
                   + np.einsum("qtd,qtd->qt", C, A) * lb)
            wn += np.sum(2.0 * np.arctan2(det, den), axis=1)
        return np.rint(wn / (4.0 * np.pi)).astype(int) % 2 == 1
    raise NotImplementedError(f"inside test for k={k}")


# --------------------------------------------------------------------------
# The whole of IJRR 4.3
# --------------------------------------------------------------------------


@dataclass
class Reconstruction:
    points: np.ndarray                      # (n, d) scaled, net vertices
    simplices: np.ndarray                   # (m, k+1) facets, oriented if k=2
    loops: list                             # k = 1 only: closed index cycles
    closed: bool
    note: str = ""
    stats: dict = dc_field(default_factory=dict)


def reconstruct(
    field: Field, seeds: np.ndarray, eta: float, lo: np.ndarray, hi: np.ndarray,
    side_points: np.ndarray | None = None, step: float = 1.0, accept: float = 0.8,
    k_neighbors: int = 0, fix_rounds: int = 20, weight_frac: float = 0.3,
    side_tol: float = 1.0, max_points: int = 20000,
    rng: np.random.Generator | None = None, refine_rounds: int = 6,
) -> Reconstruction:
    """Net -> TDC -> closed? -> oriented? -> right side of every labelled point?

    `side_points` (scaled) are points whose side must agree with the sign of
    f -- in practice the SVM's training data, so the check asks "does the
    complex separate the two classes the way the wall does?". A point is
    checked only if it sits more than side_tol*eta
    from M (|f| / |grad f|), because a facet is a chord and can legitimately
    cut off a sliver of the curved zero set next to it.
    """
    rng = rng or np.random.default_rng(0)
    lo = np.asarray(lo, dtype=float)
    d = len(lo)
    k = d - 1
    net, nstats = sample_manifold_net(field, seeds, eta, lo, hi, step=step,
                                      accept=accept, max_points=max_points, rng=rng)
    stats = dict(net=nstats)
    empty = Reconstruction(net, np.zeros((0, k + 1), int), [], False, stats=stats)
    if nstats["capped"]:
        empty.note = f"net hit max_points={max_points}"
        return empty
    if len(net) < k + 2:
        empty.note = "no manifold points reached from the seeds"
        return empty
    normals = unit_normals(field, net)
    tc = tangential_complex(net, normals, k_neighbors=k_neighbors,
                            fix_rounds=fix_rounds, weight_frac=weight_frac, rng=rng)
    simp = tc.simplices
    ok, note = closure_report(simp, k)
    # LOCAL refinement (findings 42). TDC is consistent only where the
    # sample is dense relative to the manifold's local feature size, and a
    # high-gamma wall has sharp spots. Halving eta everywhere costs 2^k times
    # the points; adding them only where stars disagree costs a handful: the
    # edge midpoints of every inconsistent simplex, projected onto M, are
    # inserted and the complex is rebuilt. Only runs when closure FAILED with
    # inconsistencies left, so a reconstruction that already closes is
    # untouched.
    n_refined = 0
    for _ in range(int(refine_rounds)):
        if ok or not tc.inconsistent:
            break
        mids = []
        for simp_bad in tc.inconsistent:
            vs = net[list(simp_bad)]
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    mids.append(0.5 * (vs[i] + vs[j]))
        z, okp = project_batch(field, np.asarray(mids))
        z = z[okp & np.all((z >= lo) & (z <= np.asarray(hi, dtype=float)), axis=1)]
        sel = _greedy_net(z, cKDTree(net), 0.25 * accept * eta)
        if len(sel) == 0:
            break
        net = np.vstack([net, z[sel]])
        n_refined += len(sel)
        normals = unit_normals(field, net)
        tc = tangential_complex(net, normals, k_neighbors=k_neighbors,
                                fix_rounds=fix_rounds, weight_frac=weight_frac, rng=rng)
        simp = tc.simplices
        ok, note = closure_report(simp, k)
    stats.update(n_simplices=len(tc.simplices), n_inconsistent=tc.n_inconsistent,
                 fix_rounds=tc.fix_rounds, refined_points=n_refined)
    rec = Reconstruction(net, simp, [], ok, note, stats)
    if not ok:
        if tc.n_inconsistent:
            rec.note += f" ({tc.n_inconsistent} inconsistent simplex(es) left)"
        return rec
    if k == 1:
        rec.loops = extract_loops(simp)
    elif k == 2:
        simp, ok, note = orient_triangles(net, simp, normals)
        rec.simplices = simp
        if not ok:
            rec.closed, rec.note = False, note
            return rec
    stats["euler"] = _euler(simp, k)

    if side_points is not None and len(side_points):
        lp = np.asarray(side_points, dtype=float)
        f, g = field(lp)
        dist = np.abs(f) / np.maximum(np.linalg.norm(g, axis=1), 1e-300)
        far = dist > side_tol * eta
        ins = inside(lp[far], net, simp, k)
        want = f[far] > 0
        wrong = int(np.sum(ins != want))
        stats.update(side_checked=int(far.sum()), side_wrong=wrong)
        if wrong:
            rec.closed = False
            rec.note = (f"sidedness: {wrong} of {int(far.sum())} labelled points "
                        f"lie on the wrong side of the complex (a piece of the "
                        f"wall was not reconstructed)")
    return rec


def _euler(simp: np.ndarray, k: int) -> int:
    v = len(np.unique(simp))
    if k == 1:
        return v - len(simp)
    edges = {tuple(sorted((int(t[i]), int(t[j]))))
             for t in simp for i, j in ((0, 1), (1, 2), (0, 2))}
    return v - len(edges) + len(simp)
