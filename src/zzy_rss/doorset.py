"""The doors of a task, found from the chart's sigma alone (docs/recursive_proof_design.md).

Two products, with two different jobs:

  REGIONS   interior.find_doors over the whole page box: unions of cells that
            together contain EVERY zero of sigma, by certificate. These are
            what the proof reasons with -- a door "inside a wall" means a
            region with a cell the wall does not certifiably exclude. Since
            sigma is the same on all eight pages (a door is a physical
            place), one search serves every page.
  CURVES    points ON the doors, found by minimising sigma_k^2 from the
            region's cells and chained into polylines. These are what the
            ROADMAP needs: crossing states either side of a door point
            (flood3.door_samples), the door posture of a crossing leg, the
            posture check that may discharge a door, and training samples
            that steer the SVM. They are evidence, not certificates: a
            missed piece of curve can only cost a crossing or a discharge,
            because the regions, not the curves, decide what is inside a
            wall -- except for discharge, which is guarded below.

No closed form is used anywhere here. doors3.door_curves (the sphere-plane
circle) appears only in tests, as an independent cross-check.

Only the q2 and q6 components produce crossable point/curve doors. A region
whose cells could not exclude the q4 fold or Sigma_coord is kept and marked
UNSUPPORTED: a wall that has one inside is not a proof (check.py refuses it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .chart import I_Q2, I_Q6, SrsChart
from .doors3 import DoorCurve
from .facets import SigmaTube
from .interior import DoorRegion, find_doors

SIGN_OF = {I_Q2: 0, I_Q6: 2}          # sigma component -> sign index it flips
COMP_OF = {0: I_Q2, 2: I_Q6}


@dataclass
class Door:
    """One certified door region and what was found on it."""

    index: int
    region: DoorRegion
    sign_index: int | None            # 0 (q2) / 2 (q6); None: unsupported kind
    curves: list = field(default_factory=list)      # doors3.DoorCurve
    covered: bool | None = None       # None: not yet checked (check_coverage)
    note: str = ""
    cover_far: float = float("nan")

    def check_coverage(self, chart: SrsChart, cover_tol: float = 0.004,
                       cover_slack: float = 0.06, max_cells: int = 3_000_000) -> bool:
        """Do the curves account for every place in the region where a zero
        could still be? A finer certified search (cover_tol) of the region's
        box must leave no candidate cell further than cover_slack (scaled)
        from a curve point. Only a covered door may be discharged by its
        posture: a discharge rests on the curve samples, so a piece of door
        the curves missed would otherwise be discharged unseen."""
        if self.covered is not None:
            return self.covered
        if not self.curves:
            self.covered = False
            return False
        reg = self.region
        fine = find_doors(chart, tol=cover_tol, lo=np.maximum(reg.lo, chart.lo),
                          hi=np.minimum(reg.hi, chart.hi), max_cells=max_cells)
        if not fine.complete:
            self.covered = False
            return False
        tree = cKDTree(self.points * chart.scale)
        far = 0.0
        for fr in fine.doors:
            dd, _ = tree.query(fr.centres * chart.scale, k=1)
            far = max(far, float(np.max(dd)))
        self.cover_far = far
        self.covered = far <= cover_slack
        return self.covered

    @property
    def supported(self) -> bool:
        return self.sign_index is not None

    @property
    def joint(self) -> int | None:
        return None if self.sign_index is None else (2 if self.sign_index == 0 else 6)

    @property
    def points(self) -> np.ndarray:
        if not self.curves:
            return np.zeros((0, len(self.region.centres[0])))
        return np.vstack([c.pts for c in self.curves])

    def label(self) -> str:
        return f"D{self.index}:{'q' + str(self.joint) if self.supported else self.region.label()}"


@dataclass
class DoorSet:
    doors: list
    complete: bool
    stats: dict = field(default_factory=dict)

    def curves(self) -> list:
        """Every curve of every supported door, with its door index."""
        return [(d.index, c) for d in self.doors for c in d.curves]


def refine_points(chart: SrsChart, region: DoorRegion, comp: int, spacing: float,
                  tol_sigma: float = 1e-7, iters: int = 40,
                  start_factor: float = 3.0) -> np.ndarray:
    """Points where sigma_comp = 0, by vectorised Newton steps on f = sigma^2.

    f is smooth at a door (an angle squared). Near a door curve its Hessian
    has rank d - 1 -- flat along the curve, curved across it -- so the
    pseudo-inverse Newton step z <- z - H^+ grad f lands on the curve even
    where the cone of sigma is very lopsided (old EXP 3: steep in theta,
    shallow in (u, v), where a plain gradient step zig-zags). Derivatives by
    central differences; a backtracking line search guards each step.
    Starts: the smallest-sigma cell of every bucket of side
    start_factor * spacing over the region."""
    scale = chart.scale
    d = chart.dim
    C = region.centres
    s0 = chart.sigma(C)[:, comp]
    ok = np.isfinite(s0)
    if not ok.any():
        return np.zeros((0, d))
    # One start per bucket of side start_spacing: the cell with the smallest
    # sigma in it. Buckets tile the whole region, so every stretch of a long
    # curve gets starts (a global quantile favoured where sigma is flattest).
    Cc, sc = C[ok], s0[ok]
    keys = np.floor(Cc * scale / (start_factor * spacing)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.lexsort((sc, inv))
    first = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
    Z = Cc[first] * scale                                 # scaled starts
    lo = (region.lo - 2.0 * region.half.max(axis=0)) * scale
    hi = (region.hi + 2.0 * region.half.max(axis=0)) * scale

    def sig(Zs):
        v = chart.sigma(Zs / scale)[:, comp]
        return np.where(np.isfinite(v), v, 1e3)

    h = 1e-4
    E = np.eye(d) * h
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    ts = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    for _ in range(iters):
        sv = sig(Z)
        act = sv > tol_sigma
        if not act.any():
            break
        Za = Z[act]
        n = len(Za)
        f0 = sv[act] ** 2
        offs = [E[i] for i in range(d)] + [-E[i] for i in range(d)]
        offs += [E[i] + E[j] for i, j in pairs] + [E[i] - E[j] for i, j in pairs]
        offs += [-E[i] + E[j] for i, j in pairs] + [-E[i] - E[j] for i, j in pairs]
        O = np.asarray(offs)
        F = sig((Za[:, None, :] + O[None]).reshape(-1, d)).reshape(n, len(O)) ** 2
        fp, fm = F[:, :d], F[:, d:2 * d]
        g = (fp - fm) / (2.0 * h)
        H = np.zeros((n, d, d))
        H[:, range(d), range(d)] = (fp - 2.0 * f0[:, None] + fm) / h ** 2
        m = len(pairs)
        base = 2 * d
        for q, (i, j) in enumerate(pairs):
            hij = (F[:, base + q] - F[:, base + m + q] - F[:, base + 2 * m + q]
                   + F[:, base + 3 * m + q]) / (4.0 * h * h)
            H[:, i, j] = H[:, j, i] = hij
        w, V = np.linalg.eigh(H)
        keep = w > 1e-8 * np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        step = -np.einsum("nij,nj,nkj,nk->ni", V, winv, V, g)
        # Fallback where the Newton step is not a descent direction.
        bad = np.einsum("ni,ni->n", step, g) >= 0.0
        if bad.any():
            gg = np.einsum("ij,ij->i", g[bad], g[bad])
            step[bad] = -(f0[bad] / np.maximum(gg, 1e-300))[:, None] * g[bad]
        trial = np.clip(Za[:, None, :] + ts[None, :, None] * step[:, None, :], lo, hi)
        st = sig(trial.reshape(-1, d)).reshape(n, len(ts))
        best = np.argmin(st, axis=1)
        better = st[np.arange(n), best] < sv[act]
        Za = np.where(better[:, None], trial[np.arange(n), best], Za)
        Z[act] = Za
        if not better.any():
            break
    sv = sig(Z)
    P = Z[sv < tol_sigma] / scale
    if not len(P):
        return np.zeros((0, d))
    keys = np.floor(P * scale / (0.5 * spacing)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return P[np.sort(first)]


def chain(points: np.ndarray, scale: np.ndarray, gap: float) -> list:
    """Order points into polylines: greedy nearest-neighbour from an end,
    split where the next point is further than `gap` (scaled)."""
    pts = np.asarray(points, dtype=float)
    left = set(range(len(pts)))
    out = []
    S = pts * scale
    while left:
        idx = np.fromiter(left, dtype=int)
        # Start at an extreme point of what is left (an end of a curve).
        c = S[idx].mean(axis=0)
        cur = int(idx[np.argmax(np.linalg.norm(S[idx] - c, axis=1))])
        # Walk to the far end first so the chain starts at an end.
        for _ in range(2):
            seq = [cur]
            rem = set(left) - {cur}
            while rem:
                r = np.fromiter(rem, dtype=int)
                d = np.linalg.norm(S[r] - S[seq[-1]], axis=1)
                j = int(np.argmin(d))
                if d[j] > gap:
                    break
                seq.append(int(r[j]))
                rem.discard(int(r[j]))
            cur = seq[-1]
        left -= set(seq)
        out.append(np.asarray(seq))
    return out


def find_doorset(chart: SrsChart, tube_radius, tol: float = 0.01, spacing: float = 0.01,
                 max_cells: int = 3_000_000) -> DoorSet:
    """Certified door regions of the whole page box, and the curves on them.
    Coverage (Door.check_coverage) is computed only when a discharge needs it."""
    scale = chart.scale
    res = find_doors(chart, tol=tol, max_cells=max_cells)
    doors = []
    if not res.complete:
        return DoorSet(doors=[], complete=False, stats=dict(search=res.stats))
    for k, reg in enumerate(res.doors):
        kinds = set(reg.kinds)
        comps = [c for c in (I_Q2, I_Q6) if ("q2" if c == I_Q2 else "q6") in kinds]
        if kinds - {"q2", "q6"} or len(comps) != 1:
            doors.append(Door(index=k, region=reg, sign_index=None,
                              note=f"unsupported sigma kinds {sorted(kinds)}"))
            continue
        comp = comps[0]
        P = refine_points(chart, reg, comp, spacing)
        curves = []
        for seq in chain(P, scale, gap=4.0 * spacing):
            if len(seq) < 2:
                continue
            pts = P[seq]
            closed = bool(np.linalg.norm((pts[0] - pts[-1]) * scale) <= 4.0 * spacing
                          and len(seq) > 8)
            c = DoorCurve(sign_index=SIGN_OF[comp], pts=pts, closed=closed)
            poly = np.vstack([pts, pts[:1]]) if closed else pts
            c.tube = SigmaTube(polyline=poly, radii=np.asarray(tube_radius, dtype=float),
                               kind=f"q{2 if comp == I_Q2 else 6} door curve (sigma)")
            curves.append(c)
        door = Door(index=k, region=reg, sign_index=SIGN_OF[comp], curves=curves,
                    note=(f"{len(curves)} curve(s), {len(P)} points" if curves else
                          "no point with sigma = 0 found in the region"))
        doors.append(door)
    return DoorSet(doors=doors, complete=True, stats=dict(search=res.stats))
