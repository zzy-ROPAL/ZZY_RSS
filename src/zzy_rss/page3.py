"""The 3-D page (u, v, theta) of a weld PATCH, and its proof (findings 42).

This is the [IROS] d = 3 case: a 2-D task (a flat weld patch) on a 7-DOF arm
leaves one arm angle free, so each sign branch is a 3-D page. Everything here
is grid-free and reuses the d-generic pieces:

    roadmap        uniform samples + k-NN edges validated at edge_resolution
    Algorithm 1    wall.train_wall with the page's 3-axis scale
    Algorithm 2    wall.sample_wall with this page's validity
    IJRR 4.3       tangential.reconstruct -> a closed, oriented triangle mesh
    Algorithm 3    facets.check_simplex on triangles, through a robot oracle
                   whose Sigma guard is the door TUBES of doors3
    Algorithm 4    the outer loop below (feed valid facet centres back, or
                   halve the net spacing)
    FEASIBLE       a roadmap path, densified and re-checked by
                   witness.verify_witness with (u, v) per sample

What it does NOT do yet is the cross-page flood. The verdict is therefore a
SINGLE-PAGE proof, and it is only issued when that is enough:

    INFEASIBLE  the wall is closed, every triangle is in C_obs, the start is
                inside, the whole goal slab u >= 1 - goal_tol is outside
                (exactly: the wall's u-range ends before it, and the enclosed
                volume lies in the wall's convex hull), and every LIVE door
                tube is outside -- so the robot can neither reach the goal on
                this page nor leave the page.
    FEASIBLE    a verified end-to-end path on this page.
    INCONCLUSIVE everything else, including "a live door is inside the wall"
                (following it needs the 3-D flood), a fold or Sigma_coord on
                the patch, or valid configurations at the theta cut.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import budget as _budget
from .arm import SrsArm
from .batch import classify_surface_batch, pd_march_batch
from .doors3 import certify_curve, door_curves, patch_singularities
from .facets import FacetOracle, tube_guard
from .params import Config, Page3Params
from .prm import UnionFind
from .scene import Scene
from .source import eta_separate
from .tangential import inside, reconstruct
from .wall import (CheckReport, Polytope, assign_label, check_polytope, pd_march,
                   sample_wall, train_wall)
from .weld import WeldSurface
from .witness import LEG_PAGE, Leg, WitnessPath, verify_witness

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"
INCONCLUSIVE = "INCONCLUSIVE"


def classify3(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
              pt, signs: tuple) -> bool:
    """Validity of one (u, v, theta) state: IK on this branch, limits, collision."""
    pos, rot = surf.pose(float(pt[0]), float(pt[1]))
    res = arm.ik_branch(pos, rot, float(pt[2]), signs,
                        coord_axis_tol=cfg.pagemap.coord_axis_tol,
                        coord_radius_tol=cfg.pagemap.coord_radius_tol)
    if not res.ok or not arm.within_limits(res.q):
        return False
    return not scene.check(arm.frames(res.q)).in_collision


# --------------------------------------------------------------------------
# the page: a roadmap with no cells
# --------------------------------------------------------------------------


class Page3:
    """One sign branch of the patch task, as a PRM in (u, v, theta)."""

    def __init__(self, cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                 signs: tuple, p3: Page3Params, start_theta: float | None,
                 n_samples: int | None = None, inject: np.ndarray | None = None):
        """`start_theta` None builds a page with no start posture (a partner
        page of the flood). `inject` are extra states (near-door points)
        made vertices when valid; `inject_index[i]` is the vertex of the i-th
        one, or -1 if it is invalid on this page."""
        self.cfg, self.arm, self.surf, self.scene = cfg, arm, surf, scene
        self.signs, self.p3 = signs, p3
        self.scale = np.asarray(p3.scale, dtype=float)
        self.lo = np.array([0.0, 0.0, p3.theta_min])
        self.hi = np.array([1.0, 1.0, p3.theta_max])
        head = []
        self.start = None
        if start_theta is not None:
            self.start = np.array([p3.start_u, p3.start_v, start_theta], dtype=float)
            if not self.valid_fn(self.start):
                raise ValueError(f"start pose {self.start} is not valid on page {signs}")
            head.append(self.start)
        # Injected states next, so their indices are stable and never thinned.
        inject = np.zeros((0, 3)) if inject is None else np.asarray(inject, dtype=float)
        self.inject_index = np.full(len(inject), -1, dtype=int)
        ok_inj = self.valid_many(inject) if len(inject) else np.zeros(0, dtype=bool)
        for i in np.flatnonzero(ok_inj):
            self.inject_index[i] = len(head)
            head.append(inject[i])
        # Experimental (findings 47, item 8): goal states as vertices, a hair
        # inside the goal slab (not on u = 1: invariant 18), so the slab is
        # found without relying on random samples landing in it.
        n_goal = int(getattr(p3, "goal_inject", 0))
        if n_goal > 0:
            grng = np.random.default_rng(p3.rng_seed + 7)
            g = grng.uniform(self.lo, self.hi, size=(n_goal, 3))
            g[:, 0] = 1.0 - 0.5 * float(p3.goal_tol)
            head.extend(list(g[self.valid_many(g)]))
        rng = np.random.default_rng(p3.rng_seed)
        n = int(n_samples or p3.n_samples)
        pts = rng.uniform(self.lo, self.hi, size=(n, 3))
        ok = self.valid_many(pts)
        self.valid = np.vstack([np.asarray(head).reshape(-1, 3), pts[ok]])
        self.invalid = pts[~ok]
        # Lazy PRM (findings 43): edges a witness found invalid between the
        # edge checker's samples are blacklisted and the graph rebuilt.
        self.blacklist: set = set()
        self._edge_cache: dict = {}
        # Components the current wall must enclose besides the focus one:
        # set by region_wall to the regions touching a seed-side door tube.
        self.inside: set = set()
        self._build_graph()
        self._focus = int(self.comp[0]) if self.start is not None else None

    def rebuild(self) -> None:
        """Recompute the forest and components without the blacklisted edges
        (cached edge checks make this cheap). Component ids change."""
        self._build_graph()
        self._focus = int(self.comp[0]) if self.start is not None else None

    def valid_fn(self, pt) -> bool:
        return classify3(self.cfg, self.arm, self.surf, self.scene, pt, self.signs)

    def valid_many(self, pts) -> np.ndarray:
        """classify3 for N states at once (batch.py; tag-for-tag equal to the
        scalar path, tests/test_batch.py). `planner.batch_classify=False`
        falls back to the scalar loop."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if not self.cfg.planner.batch_classify:
            return np.array([self.valid_fn(x) for x in pts], dtype=bool)
        return classify_surface_batch(self.cfg, self.arm, self.surf, self.scene,
                                      pts, self.signs)[0]

    def in_box(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return np.all((pts >= self.lo) & (pts <= self.hi), axis=1)

    def valid_task(self, pt) -> bool:
        """Validity AS A STATE OF THE TASK: outside the page box the tool is
        off the patch, so the state is virtual invalid (invariant 37) even
        where the extrapolated surface has an IK solution. Algorithm 3
        already reads the box this way (_box_band); Algorithm 2 must too
        (findings 45: projections at u < 0 or v outside [0, 1] were fed back
        as VALID seed points 0.05 from the virtual ring, gamma 21 -> > 100)."""
        return bool(self.in_box(pt)[0]) and self.valid_fn(pt)

    def valid_task_many(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        ok = self.in_box(pts)
        if ok.any():
            ok[ok] = self.valid_many(pts[ok])
        return ok

    def states_many(self, pts) -> tuple:
        """(valid (N,), q (N, 7)) -- `_state` for N states."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if not self.cfg.planner.batch_classify:
            out = [self._state(x) for x in pts]
            ok = np.array([o for o, _ in out], dtype=bool)
            q = np.array([qq if qq is not None else np.zeros(7) for _, qq in out])
            return ok, q.reshape(-1, 7)
        ok, _tag, q = classify_surface_batch(self.cfg, self.arm, self.surf, self.scene,
                                             pts, self.signs)
        return ok, q

    def _state(self, pt) -> tuple:
        """(valid, q) of one state -- classify3, keeping the joint vector."""
        cfg = self.cfg
        pos, rot = self.surf.pose(float(pt[0]), float(pt[1]))
        res = self.arm.ik_branch(pos, rot, float(pt[2]), self.signs,
                                 coord_axis_tol=cfg.pagemap.coord_axis_tol,
                                 coord_radius_tol=cfg.pagemap.coord_radius_tol)
        if not res.ok or not self.arm.within_limits(res.q):
            return False, None
        return (not self.scene.check(self.arm.frames(res.q)).in_collision), res.q

    def _edge(self, a: np.ndarray, b: np.ndarray) -> bool:
        """A straight page edge, checked to the WITNESS standard (findings 43).

        Samples every edge_resolution in page coordinates, then bisects any
        gap whose JOINT step exceeds witness.step_max. Near a door the joints
        move fast per unit of page distance, and a fixed page step can stride
        over a thin out-of-limits sliver: on EXP 4's patch, seven lazy-PRM
        rounds kept finding new such edges. A gap that does not shrink under
        bisection is a jump, not a path, and fails the edge."""
        step = self.cfg.planner.edge_resolution
        jmax = float(self.cfg.witness.step_max)
        m = max(2, int(np.ceil(float(np.linalg.norm((b - a) * self.scale)) / step)))
        ts = np.linspace(0.0, 1.0, m + 1)
        ok, qs = self.states_many(a[None, :] + ts[:, None] * (b - a)[None, :])
        if not ok.all():
            return False
        stack = [(ts[i], qs[i], ts[i + 1], qs[i + 1], 0) for i in range(m)]
        while stack:
            t0, q0, t1, q1, depth = stack.pop()
            if float(np.abs(q1 - q0).max()) <= jmax:
                continue
            if depth >= 14:
                return False
            tm = 0.5 * (t0 + t1)
            ok, qm = self._state(a + tm * (b - a))
            if not ok:
                return False
            stack.append((t0, q0, tm, qm, depth + 1))
            stack.append((tm, qm, t1, q1, depth + 1))
        return True

    def _build_graph(self) -> None:
        """k-NN candidates, shortest first, validated only when they would
        join two components (as RoadmapSource). The accepted edges are kept:
        they form a spanning forest, and a path along it is a validated path."""
        n = len(self.valid)
        sc = self.valid * self.scale
        tree = cKDTree(sc)
        k = min(int(self.p3.k_neighbors) + 1, n)
        d, nbr = tree.query(sc, k=k)
        cand = sorted({(float(d[i][c]), min(i, int(nbr[i][c])), max(i, int(nbr[i][c])))
                       for i in range(n) for c in range(1, k)})
        uf = UnionFind(n)
        self.adj = [[] for _ in range(n)]
        for _w, i, j in cand:
            if uf.find(i) == uf.find(j) or (i, j) in self.blacklist:
                continue
            if (i, j) not in self._edge_cache:
                self._edge_cache[(i, j)] = self._edge(self.valid[i], self.valid[j])
            if self._edge_cache[(i, j)]:
                uf.union(i, j)
                self.adj[i].append(j)
                self.adj[j].append(i)
        roots: dict = {}
        self.comp = np.array([roots.setdefault(uf.find(i), len(roots))
                              for i in range(n)])

    # -- the five questions, page-local -----------------------------------

    @property
    def start_comp(self) -> int:
        """The component a wall is currently being built for: the start's on
        the start page, whichever region the flood is on otherwise."""
        return self._focus

    def set_focus(self, comp: int | None) -> None:
        self._focus = None if comp is None else int(comp)

    def path(self, a: int, targets: np.ndarray) -> list | None:
        """Vertex indices from vertex a to the first vertex with targets[j]
        True, along the validated forest (BFS)."""
        prev = {a: None}
        queue = [a]
        for i in queue:
            if targets[i]:
                out = [i]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            for j in self.adj[i]:
                if j not in prev:
                    prev[j] = i
                    queue.append(j)
        return None

    def goal_mask(self) -> np.ndarray:
        return self.valid[:, 0] >= 1.0 - self.p3.goal_tol

    def reaches_goal(self) -> bool:
        return bool(np.any(self.goal_mask() & (self.comp == self.start_comp)))

    def focus_set(self) -> set:
        """The components the current wall must enclose: the focus (None for
        a region entered through a door that touches no sampled component)
        and the door pockets."""
        base = {int(self.start_comp)} if self.start_comp is not None else set()
        return base | set(int(c) for c in self.inside)

    def seed_points(self) -> np.ndarray:
        return self.valid[np.isin(self.comp, sorted(self.focus_set()))]

    def rest_points(self) -> np.ndarray:
        """IJRR's split (invariant 15): the components that reach the goal."""
        goal_comps = set(int(c) for c in self.comp[self.goal_mask()]) - self.focus_set()
        return self.valid[np.isin(self.comp, sorted(goal_comps))]

    def label_point(self, x: np.ndarray) -> int:
        """SDCL's rule: +1 if x connects to the start component, -1 if to a
        goal component, 0 if to neither (by validated edges to its nearest
        roadmap vertices)."""
        goal_comps = set(int(c) for c in self.comp[self.goal_mask()]) - {self.start_comp}
        k = min(int(self.p3.anchor_connect_tries), len(self.valid))
        _, js = cKDTree(self.valid * self.scale).query(x * self.scale, k=k)
        for j in np.atleast_1d(js):
            if self._edge(x, self.valid[int(j)]):
                c = int(self.comp[int(j)])
                if c == self.start_comp or c in self.inside:
                    return 1
                return -1 if c in goal_comps else 0
        return 0

    def nearest_valid_point(self, c: np.ndarray) -> np.ndarray | None:
        """A VALID state near c: march from c toward its nearest roadmap
        vertices and bisect at the first valid sample (as pd_march does, but
        returning the point)."""
        step = self.cfg.planner.edge_resolution
        d, js = cKDTree(self.valid * self.scale).query(
            c * self.scale, k=min(3, len(self.valid)))
        best, best_d = None, np.inf
        for dist, j in zip(np.atleast_1d(d), np.atleast_1d(js)):
            seed = self.valid[int(j)]
            n = int(np.ceil(min(float(dist), best_d) / step))
            lo_t, hit = 0.0, None
            for i in range(1, n + 1):
                t = min(1.0, i * step / max(float(dist), 1e-12))
                if self.valid_fn(c + t * (seed - c)):
                    hit = t
                    break
                lo_t = t
            if hit is None:
                continue
            for _ in range(12):
                mid = 0.5 * (lo_t + hit)
                if self.valid_fn(c + mid * (seed - c)):
                    hit = mid
                else:
                    lo_t = mid
            if hit * float(dist) < best_d:
                best, best_d = c + hit * (seed - c), hit * float(dist)
        return best

    def path_to_goal(self) -> list | None:
        """Vertex indices from the start to a goal vertex, along the forest."""
        return self.path(0, self.goal_mask())


def choose_start_theta(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                       signs: tuple, p3: Page3Params, n: int = 721) -> float | None:
    """The middle of the widest valid theta interval at (start_u, start_v).

    This only CHOOSES the start posture -- a 1-D scan of the arm angle at one
    point of the task -- and plays no part in any proof.
    """
    th = np.linspace(p3.theta_min, p3.theta_max, n)
    pts = np.column_stack([np.full(n, p3.start_u), np.full(n, p3.start_v), th])
    if cfg.planner.batch_classify:
        ok = classify_surface_batch(cfg, arm, surf, scene, pts, signs)[0]
    else:
        ok = np.array([classify3(cfg, arm, surf, scene, x, signs) for x in pts])
    best, run = None, []
    for i in range(n + 1):
        if i < n and ok[i]:
            run.append(i)
            continue
        if run and (best is None or len(run) > len(best)):
            best = run
        run = []
    return None if best is None else float(th[best[len(best) // 2]])


def theta_cut_check(page: Page3) -> dict:
    """The bounded-theta decision, checked: no valid state near the cut.

    The page is the rectangle theta in [theta_min, theta_max], but theta is an
    angle, so theta_min and theta_max are the SAME arm angle. Treating them as
    a wall the robot cannot cross is exact only if nothing valid touches the
    cut. Sampled: seam_samples random states within seam_eps of the cut,
    plus every roadmap vertex there.
    """
    p3 = page.p3
    rng = np.random.default_rng(p3.rng_seed + 1)
    m = int(p3.seam_samples)
    side = rng.integers(0, 2, m)
    th = np.where(side == 0, p3.theta_min + rng.uniform(0, p3.seam_eps, m),
                  p3.theta_max - rng.uniform(0, p3.seam_eps, m))
    pts = np.column_stack([rng.uniform(0, 1, m), rng.uniform(0, 1, m), th])
    hits = list(pts[page.valid_many(pts)])
    near = ((page.valid[:, 2] <= p3.theta_min + p3.seam_eps)
            | (page.valid[:, 2] >= p3.theta_max - p3.seam_eps))
    n_v = int(near.sum())
    ok = not hits and n_v == 0
    return dict(ok=ok, n_checked=m, n_valid=len(hits), n_roadmap_near=n_v,
                reason=("theta cut is clear" if ok else
                        f"{len(hits)} sampled + {n_v} roadmap valid state(s) within "
                        f"{p3.seam_eps} rad of the theta cut"))


# --------------------------------------------------------------------------
# FEASIBLE: a verified path
# --------------------------------------------------------------------------


def densify3(arm: SrsArm, surf: WeldSurface, wp, scale: np.ndarray, way: np.ndarray,
             signs_of) -> tuple:
    """IK at samples along a polyline of page states, refined until the JOINT
    step respects step_max -- the bound the verifier rests on.

    `signs_of(a_index, t)` gives the sign branch for the sample at parameter t
    of segment a_index: constant for a page leg; switching mid-segment for a
    straight-through door crossing. Returns (q, uv) or (None, None)."""
    qs, uv = [], []
    for si, (a, b) in enumerate(zip(way[:-1], way[1:])):
        n = max(2, int(np.ceil(float(np.linalg.norm((b - a) * scale)) / wp.page_step)))
        seg = None
        for _ in range(10):
            seg = []
            for t in np.linspace(0.0, 1.0, n + 1):
                p = a + t * (b - a)
                r = arm.ik_branch(*surf.pose(float(p[0]), float(p[1])), float(p[2]),
                                  signs_of(si, t))
                if not r.ok:
                    return None, None
                seg.append((r.q, p[:2].copy()))
            qq = np.asarray([x[0] for x in seg])
            if float(np.abs(np.diff(qq, axis=0)).max()) <= wp.step_max:
                break
            n *= 2
        for q_i, uv_i in seg[:-1]:
            qs.append(q_i)
            uv.append(uv_i)
    last = way[-1]
    r = arm.ik_branch(*surf.pose(float(last[0]), float(last[1])), float(last[2]),
                      signs_of(len(way) - 2, 1.0))
    if not r.ok:
        return None, None
    qs.append(r.q)
    uv.append(last[:2].copy())
    return np.asarray(qs), np.asarray(uv)


def witness_path3(page: Page3, idx: list) -> WitnessPath | None:
    """Densify a roadmap path on one page into a verifiable leg."""
    way = page.valid[idx]
    q, uv = densify3(page.arm, page.surf, page.cfg.witness, page.scale, way,
                     lambda _i, _t: page.signs)
    if q is None:
        return None
    return WitnessPath(legs=[Leg(kind=LEG_PAGE, page=page.signs, q=q, y=uv,
                                 waypoints=way)])


# --------------------------------------------------------------------------
# INFEASIBLE: the wall
# --------------------------------------------------------------------------


def _box_band(cfg: Config, page: Page3, c: np.ndarray) -> bool:
    """Outside the page box = virtual invalid (Page3Params.ring_offset has
    the argument for why that is exact on a patch task)."""
    return bool(np.any((c < page.lo) | (c > page.hi)))


def virtual_ring3(cfg: Config, page: Page3, n_per_face: int, rng) -> np.ndarray:
    """Rest-class samples on a box ring_offset (scaled) outside the page:
    they steer the SVM shut outside the page, nothing more."""
    r = float(page.p3.ring_offset) / page.scale
    lo, hi = page.lo - r, page.hi + r
    out = []
    for ax in range(3):
        for val in (lo[ax], hi[ax]):
            p = rng.uniform(lo, hi, size=(n_per_face, 3))
            p[:, ax] = val
            out.append(p)
    return np.vstack(out)


def clip_to_box(poly: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list:
    """The parts of a door polyline whose (u, v) lie on the patch, with the
    boundary crossings interpolated exactly (Liang-Barsky per segment).

    door_curves keeps samples up to one tube radius OUTSIDE the patch so an
    edge-grazing curve is still covered where it enters. Outside the patch
    the tool is off the task: the page there is virtual invalid (invariant
    37) and there is no door. Training samples placed there only pin the
    SVM against the virtual ring (findings 45: d = 0.09 between such a
    sample and the ring, gamma 52 for a wall the start region alone would
    not need), and a guard tube there forbids nothing a robot could do."""
    pieces, cur = [], []
    for a, b in zip(poly[:-1], poly[1:]):
        t0, t1 = 0.0, 1.0
        for ax in (0, 1):
            d = b[ax] - a[ax]
            for side, bound in ((-1.0, lo[ax]), (1.0, hi[ax])):
                num = side * (bound - a[ax])        # side*(x(t) - bound) <= 0
                den = side * d
                if abs(den) < 1e-15:
                    if num < 0:
                        t0, t1 = 1.0, 0.0
                    continue
                t = num / den
                if den > 0:
                    t1 = min(t1, t)
                else:
                    t0 = max(t0, t)
        if t0 > t1:
            if len(cur) >= 1:
                pieces.append(np.asarray(cur))
            cur = []
            continue
        p0, p1 = a + t0 * (b - a), a + t1 * (b - a)
        if cur and np.allclose(cur[-1], p0):
            cur.append(p1)
        else:
            if cur:
                pieces.append(np.asarray(cur))
            cur = [p0, p1]
        if t1 < 1.0:
            pieces.append(np.asarray(cur))
            cur = []
    if cur:
        pieces.append(np.asarray(cur))
    return [p for p in pieces if len(p) >= 2]


def in_box_tubes(page: "Page3", curves: list) -> list:
    """SigmaTubes around the on-patch parts of the curves (clip_to_box)."""
    from .facets import SigmaTube
    out = []
    for c in curves:
        for piece in clip_to_box(np.asarray(c.tube.polyline, dtype=float), page.lo, page.hi):
            out.append(SigmaTube(polyline=piece, radii=np.asarray(c.tube.radii, dtype=float),
                                 kind=c.tube.kind))
    return out


def tube_samples(curves: list, region_pts: np.ndarray, scale: np.ndarray,
                 tol: float, lo: np.ndarray | None = None,
                 hi: np.ndarray | None = None) -> tuple:
    """Training points that keep the wall off the door tubes (invariant 7).

    A curve that comes near the start region goes WHOLE into the seed class
    (the wall must enclose it), any other into the rest class. Per curve, not
    per point: a curve split between the classes would put the wall across
    its own tube."""
    seed, rest = [], []
    tree = cKDTree(region_pts * scale) if len(region_pts) else None
    for c in curves:
        r = np.asarray(c.tube.radii, dtype=float)
        base = c.pts
        if lo is not None:
            pieces = clip_to_box(np.asarray(c.tube.polyline, dtype=float), lo, hi)
            if not pieces:
                continue                         # never on the patch: no door
            base = np.vstack(pieces)
        pts = [base]
        for ax in range(3):
            for sgn in (-1.0, 1.0):
                off = np.zeros(3)
                off[ax] = sgn * r[ax]
                pts.append(base + off)
        pts = np.vstack(pts)
        if lo is not None:                       # (findings 45) on the patch only
            keep = np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), axis=1)
            pts = pts[keep]
        near = tree is not None and float(tree.query(c.pts * scale, k=1)[0].min()) <= max(
            tol, float(np.max(r * scale)))
        (seed if near else rest).append(pts)
    return (np.vstack(seed) if seed else np.zeros((0, 3)),
            np.vstack(rest) if rest else np.zeros((0, 3)))


def face_anchors(page: Page3, rng) -> tuple:
    """Valid states on the four task faces, for the TRAINING set only.

    Each is labelled by connectivity (SDCL's rule): the first of its nearest
    roadmap vertices that a validated straight edge reaches decides --
    start component -> seed, a goal component -> rest. One that reaches
    neither is dropped: a wrong label could only cost a wall (Algorithm 3
    rejects a wall that cuts valid space), but dropping costs nothing.
    """
    p3 = page.p3
    n = int(p3.anchors_per_face)
    if n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    faces = []
    for ax in (0, 1):
        for val in (0.0, 1.0):
            pts = rng.uniform(page.lo, page.hi, size=(n, 3))
            pts[:, ax] = val
            faces.append(pts)
    cand = np.vstack(faces)
    cand = cand[page.valid_many(cand)]
    if not len(cand):
        return np.zeros((0, 3)), np.zeros((0, 3))
    goal_comps = set(int(c) for c in page.comp[page.goal_mask()]) - {page.start_comp}
    tree = cKDTree(page.valid * page.scale)
    k = min(int(p3.anchor_connect_tries), len(page.valid))
    _, nbr = tree.query(cand * page.scale, k=k)
    seed, rest = [], []
    for x, js in zip(cand, np.atleast_2d(nbr).reshape(len(cand), k)):
        for j in js:
            if page._edge(x, page.valid[int(j)]):
                c = int(page.comp[int(j)])
                if c == page.start_comp:
                    seed.append(x)
                elif c in goal_comps:
                    rest.append(x)
                break
    return (np.asarray(seed).reshape(-1, 3), np.asarray(rest).reshape(-1, 3))


def robot_oracle3(cfg: Config, page: Page3, live_curves: list,
                  tubes: list | None = None) -> FacetOracle:
    """Algorithm 3's view of the 3-D page. Penetration depth by the capped
    march toward the nearest VALID roadmap samples of either class (closer
    valid points only tighten the bound). `tubes`, when given, is the guard
    (one SigmaTube per door piece); otherwise the on-patch parts of
    `live_curves`."""
    if tubes is None:
        tubes = in_box_tubes(page, live_curves) if live_curves else []
    guard = tube_guard(tubes) if tubes else (lambda v: None)
    valid_all = page.valid
    wp = cfg.wall

    def depth(c, need):
        return pd_march(c, valid_all, page.valid_fn, page.scale,
                        step=float(cfg.planner.edge_resolution),
                        attempts=int(wp.pd_seed_attempts),
                        bisect_iters=int(wp.pd_bisect_iters), cap=float(need))

    def depth_many(C, R):
        return pd_march_batch(C, valid_all, page.valid_many, page.scale,
                              step=float(cfg.planner.edge_resolution),
                              attempts=int(wp.pd_seed_attempts),
                              bisect_iters=int(wp.pd_bisect_iters), caps=R)

    return FacetOracle(scale=page.scale, guard=guard,
                       virtual=lambda c: _box_band(cfg, page, c),
                       valid=page.valid_fn, depth=depth,
                       valid_many=page.valid_many if cfg.planner.batch_classify else None,
                       depth_many=depth_many)


class _PageLabeller:
    """wall.sample_wall's labeller interface over Page3.label_point."""

    def __init__(self, page: "Page3"):
        self.page = page

    def label(self, m: np.ndarray) -> int:
        return self.page.label_point(np.asarray(m, dtype=float))


def goal_feedback3(poly: Polytope, u_limit: float, u_goal: float, lo: np.ndarray,
                   hi: np.ndarray, n_max: int = 60) -> np.ndarray:
    """Wall vertices at u >= u_limit, projected onto the goal slab's inner
    face (u_goal, v, theta), clipped to the patch: REST-class feedback for the
    goal-distance check (findings 47, item 4). They are goal states (or
    obstacle states on the goal face), so the label is not a guess."""
    v = np.asarray(poly.vertices, dtype=float)
    hot = v[v[:, 0] >= u_limit].copy()
    if not len(hot):
        return np.zeros((0, 3))
    hot[:, 0] = u_goal
    hot[:, 1:] = np.clip(hot[:, 1:], lo[1:], hi[1:])
    hot = np.unique(np.round(hot, 6), axis=0)
    if len(hot) > n_max:
        hot = hot[np.linspace(0, len(hot) - 1, n_max).astype(int)]
    return hot


def find_wall3(cfg: Config, page: Page3, seed: np.ndarray, rest: np.ndarray,
               inv: np.ndarray, oracle: FacetOracle, verbose: bool = True,
               goal_limit: float | None = None,
               goal_feed: float | None = None) -> tuple:
    """IJRR Algorithm 4 on the 3-D page. Returns (poly, report, wall, log).

    With goal_limit, a wall whose every triangle passed but whose u reaches
    goal_limit is not accepted: its offending vertices, projected onto the
    goal face (goal_feed), go into the rest class and the wall is retrained
    (findings 47). The caller still re-checks u_max on what comes back."""
    p3 = page.p3
    rng = np.random.default_rng(p3.rng_seed + 2)
    scale = page.scale
    # Topological labels and the feedback guard for everything fed back
    # (Page3Params.feedback_margin has the measurement).
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(
        cfg.wall, label_mode="topological", feedback_margin=float(p3.feedback_margin)))
    labeller = _PageLabeller(page)
    m = float(p3.window_margin) / scale          # unscaled, per axis
    lo_s, hi_s = (page.lo - m) * scale, (page.hi + m) * scale
    wall = train_wall(cfg, seed, rest, scale=scale)
    eta = float(p3.eta)
    poly, rep, log = Polytope(), CheckReport(False, 0, 0), []
    checked_wall = wall          # the wall `poly` was reconstructed from
    for it in range(1, int(p3.alg4_iters) + 1):
        _budget.check_time()
        t0 = time.time()
        wall, wall_pts, seed, rest, rounds = sample_wall(
            cfg, page.arm, None, page.scene, page.signs, wall, inv, seed, rest,
            verbose=False, valid_fn=page.valid_task, gamma_warm=p3.gamma_warm,
            valid_many=page.valid_task_many,
            labeller=labeller)
        seeds = np.vstack([rng.uniform(page.lo - m, page.hi + m, size=(400, 3)),
                           wall_pts, seed, rest]) * scale
        rec = reconstruct(wall.field_scaled, seeds, eta, lo_s, hi_s,
                          side_points=np.vstack([seed, rest]) * scale,
                          step=cfg.wall.tdc_step, accept=cfg.wall.tdc_accept,
                          k_neighbors=cfg.wall.tdc_k_neighbors,
                          fix_rounds=cfg.wall.tdc_fix_rounds,
                          weight_frac=cfg.wall.tdc_weight_frac,
                          side_tol=cfg.wall.tdc_side_tol,
                          max_points=cfg.wall.tdc_max_points, rng=rng)
        entry = dict(iter=it, gamma=round(wall.gamma, 2), alg2_rounds=rounds, eta=eta,
                     net=len(rec.points), triangles=len(rec.simplices),
                     closed=rec.closed, note=rec.note)
        if not rec.closed:
            log.append(entry)
            if verbose:
                print(f"[stage3-3d] alg4 iter {it}: wall not closed ({rec.note})")
            _budget.note_unclosed(rec.note)
            eta *= 0.5
            continue
        _budget.note_closed()
        poly = Polytope(closed=True, n_segments=len(rec.simplices), method="tdc",
                        vertices=rec.points / scale, simplices=rec.simplices,
                        stats=rec.stats)
        checked_wall = wall
        rep = check_polytope(cfg, None, None, None, page.signs, poly, [], None,
                             verbose=False, oracle=oracle)
        entry.update(facets=f"{rep.n_passed}/{rep.n_facets}", seconds=round(time.time() - t0, 1))
        log.append(entry)
        if verbose:
            print(f"[stage3-3d] alg4 iter {it}: gamma {wall.gamma:.2f}, "
                  f"{len(rec.simplices)} triangles, facets {rep.n_passed}/{rep.n_facets} "
                  f"({time.time() - t0:.1f} s)")
        if rep.ok and goal_limit is not None:
            u_hi = float(poly.vertices[:, 0].max())
            if u_hi >= goal_limit:
                feed_g = goal_feedback3(poly, goal_limit,
                                        goal_feed if goal_feed is not None else goal_limit,
                                        page.lo, page.hi)
                entry["goal_feedback"] = int(len(feed_g))
                entry["u_max"] = round(u_hi, 4)
                if verbose:
                    print(f"[stage3-3d] alg4 iter {it}: wall reaches u={u_hi:.4f} >= "
                          f"{goal_limit:.4f} -> {len(feed_g)} goal-face point(s) into the "
                          f"rest class, retrain")
                if len(feed_g):
                    rest = np.vstack([rest, feed_g])
                    wall = train_wall(cfg, seed, rest, scale=scale,
                                      gamma_start=(None if p3.gamma_warm is None
                                                   else p3.gamma_warm * wall.gamma))
                    continue
                # nothing to feed: report the wall as it is; the caller's
                # u_max check refuses it
                return poly, rep, checked_wall, log
        if rep.ok:
            return poly, rep, checked_wall, log
        # IJRR Algorithm 4: a facet that fails because the free space is
        # there must push the WALL, not be subdivided (invariant 11). A VALID
        # centre is fed back as it is; behind a depth-limit failure there is a
        # valid state inside the facet's ball, and THAT is fed back. Labels by
        # connectivity, nearest class when nothing connects.
        feed = []
        for c, why in rep.failure_points:
            if "VALID" in why:
                feed.append(np.asarray(c, dtype=float))
            elif "depth limit" in why:
                v = page.nearest_valid_point(np.asarray(c, dtype=float))
                if v is not None:
                    feed.append(v)
        added = 0
        for c in feed:
            lab = page.label_point(c) or assign_label(c, seed, rest, scale)
            other = rest if lab > 0 else seed
            if cfg.wall.feedback_margin > 0 and len(other) and float(np.min(
                    np.linalg.norm((other - c) * scale, axis=1))) < cfg.wall.feedback_margin:
                continue
            if lab > 0:
                seed = np.vstack([seed, c])
            else:
                rest = np.vstack([rest, c])
            added += 1
        if added:
            wall = train_wall(cfg, seed, rest, scale=scale,
                              gamma_start=(None if p3.gamma_warm is None
                                           else p3.gamma_warm * wall.gamma))
        else:
            eta *= 0.5
    # Report the wall that was CHECKED, so facets and gamma describe one object.
    return poly, rep, checked_wall, log


@dataclass
class DoorPiece:
    """One ON-PATCH piece of a door curve -- the unit the wall reasons about.

    A curve can leave the patch and come back, so it may have several
    pieces. Each is guarded on its own (its tube may not touch a triangle),
    hence lies wholly on one side of a verified wall, and needs its OWN probe:
    probing only the first piece of a curve would miss a second piece inside
    the wall (findings 47). Indices are page-independent -- the curves come
    from the patch geometry, not from the sign branch -- so (curve, piece)
    names the same door on both pages it joins."""

    curve: int
    piece: int
    joint: int
    sign_index: int
    polyline: np.ndarray
    radii: np.ndarray

    @property
    def key(self) -> tuple:
        return (int(self.curve), int(self.piece))

    def probe(self) -> np.ndarray:
        return np.asarray(self.polyline[len(self.polyline) // 2], dtype=float)

    def tube(self):
        from .facets import SigmaTube
        return SigmaTube(polyline=np.asarray(self.polyline, dtype=float),
                         radii=np.asarray(self.radii, dtype=float),
                         kind=f"q{self.joint} door piece")


def door_pieces(page: Page3, curves) -> list:
    """The on-patch pieces of `curves`, an iterable of (index, DoorCurve)."""
    out = []
    for ci, c in curves:
        polys = clip_to_box(np.asarray(c.tube.polyline, dtype=float), page.lo, page.hi)
        for j, poly in enumerate(polys):
            out.append(DoorPiece(curve=int(ci), piece=j, joint=int(c.joint),
                                 sign_index=int(c.sign_index), polyline=poly,
                                 radii=np.asarray(c.tube.radii, dtype=float)))
    return out


def piece_samples(page: Page3, pc: DoorPiece) -> np.ndarray:
    """Training points covering a WHOLE piece's tube: the polyline and its
    offsets by one tube radius along each axis, kept on the patch. The same
    set serves the seed class (a door the wall must enclose) and the rest
    class (a door it tries to keep out); a piece is never split between the
    classes, which would put the wall across its own tube."""
    base = np.asarray(pc.polyline, dtype=float)
    pts = [base]
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            off = np.zeros(3)
            off[ax] = sgn * pc.radii[ax]
            pts.append(base + off)
    pts = np.vstack(pts)
    keep = np.all((pts[:, :2] >= page.lo[:2]) & (pts[:, :2] <= page.hi[:2]), axis=1)
    return pts[keep]


def piece_vertices(page: Page3, pc: DoorPiece) -> np.ndarray:
    """Roadmap vertices inside the piece's tube -- a TRAINING / EVIDENCE hint.
    No verdict rests on this list: sampling can miss a component that
    touches the door between vertices."""
    if not len(page.valid):
        return np.zeros(0, dtype=int)
    r = np.asarray(pc.radii, dtype=float)
    tree = cKDTree(page.valid / r)
    hits = set()
    for js in tree.query_ball_point(np.asarray(pc.polyline, dtype=float) / r, 1.0):
        hits.update(int(j) for j in js)
    return np.array(sorted(hits), dtype=int)


def piece_near_vertices(page: Page3, pc: DoorPiece, comps: set, k: int = 6,
                        reach: float = 0.25) -> list:
    """Up to k vertices of `comps` nearest to the piece (scaled distance <=
    reach), nearest first -- candidate stepping stones for the evidence path.
    A hint: nothing is concluded from proximity itself."""
    if not len(page.valid) or not comps:
        return []
    mask = np.isin(page.comp, sorted(comps))
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    tree = cKDTree(np.asarray(pc.polyline, dtype=float) * page.scale)
    d, _ = tree.query(page.valid[idx] * page.scale, k=1)
    order = np.argsort(d)
    return [int(idx[i]) for i in order[:k] if d[i] <= reach]


def piece_reach_evidence(page: Page3, entry_vertex: int | None, pc: DoorPiece,
                         comps: set) -> bool:
    """A VERIFIED path from `entry_vertex` to a state INSIDE the piece's tube:
    the roadmap path to a nearby vertex of the region, then one straight step
    to a point half a tube radius from the door, all densified and checked by
    verify_witness (findings 47).

    True promotes the door to the seed class: a reachable state inside the
    tube means the wall cannot keep the tube outside without cutting free
    space, so trying would only burn a wall. False is not a proof of
    anything -- the door stays in the rest class, where the wall either
    excludes it (a proof) or fails (and the fallback promotes it).

    The step stops short of the door point on purpose: its IK spelling is an
    atan2(0, 0) artifact (invariant 42)."""
    if entry_vertex is None:
        return False
    poly = np.asarray(pc.polyline, dtype=float)
    r_min = float(np.min(pc.radii * page.scale))
    for v in piece_near_vertices(page, pc, comps):
        x0 = page.valid[v]
        j = int(np.argmin(np.linalg.norm((poly - x0) * page.scale, axis=1)))
        p = poly[j]
        gap = float(np.linalg.norm((x0 - p) * page.scale))
        if gap <= 1e-9:
            continue
        x = p + min(1.0, 0.5 * r_min / gap) * (x0 - p)
        if not page.valid_task(x):
            continue
        mask = np.zeros(len(page.valid), dtype=bool)
        mask[v] = True
        idx = page.path(int(entry_vertex), mask) if v != entry_vertex else [v]
        if idx is None:
            continue
        way = np.vstack([page.valid[idx], x])
        q, uv = densify3(page.arm, page.surf, page.cfg.witness, page.scale, way,
                         lambda _i, _t: page.signs)
        if q is None:
            continue
        path = WitnessPath(legs=[Leg(kind=LEG_PAGE, page=page.signs, q=q, y=uv,
                                     waypoints=way)])
        if verify_witness(page.cfg, page.arm, page.surf, page.scene, path).ok:
            return True
    return False


def region_wall(cfg: Config, page: Page3, pieces: list, entries: np.ndarray,
                seed_keys=(), verbose: bool = True,
                policy: str = "exclude_first") -> dict:
    """Build and check the wall around the page's FOCUS component.

    The training set is the d = 3 recipe of findings 42: the focus
    component (seed) vs the goal components (rest), eta-separated; face
    anchors; goal-face samples; the outside ring. Door pieces (findings 47):
    a piece in `seed_keys` (an entry door, or one promoted on evidence) goes
    WHOLE into the seed class; every other piece WHOLE into the rest class.
    policy "legacy" reproduces the old proximity rule instead.

    ok means a usable certificate for THIS region: closed, every triangle in
    C_obs and off every door tube, every entry state and every seed-side
    piece inside, and the whole goal slab outside (every vertex at
    u < 1 - goal_tol - goal_margin; u is linear on a triangle, so this is
    exact). `piece_inside[key]` says on which side each piece lies -- the
    guard keeps every tube wholly on one side, so its probe decides.
    """
    p3 = page.p3
    st: dict = {}
    rng = np.random.default_rng(p3.rng_seed + 3)
    seed_keys = set(seed_keys)
    entries = np.asarray(entries, dtype=float).reshape(-1, 3)
    page.inside = set()
    region0 = (page.valid[page.comp == page.start_comp] if page.start_comp is not None
               else np.zeros((0, 3)))
    goal_comps = set(int(c) for c in page.comp[page.goal_mask()]) - page.focus_set()
    if policy == "legacy" and len(region0):
        # The old rule: a piece near the region's samples is seed side.
        tree0 = cKDTree(region0 * page.scale)
        for pc in pieces:
            r = float(np.max(pc.radii * page.scale))
            if float(tree0.query(pc.polyline * page.scale, k=1)[0].min()) <= max(0.03, r):
                seed_keys.add(pc.key)
    seed_pieces = [pc for pc in pieces if pc.key in seed_keys]
    rest_pieces = [pc for pc in pieces if pc.key not in seed_keys]
    # Door pockets (findings 45). A seed-side piece must end up INSIDE the
    # wall. Any other region reaching into its tube -- on a q6 door, the
    # sector between the q7 band and the q5 sliver that meets the start
    # region only at the door point -- must then be inside too: the wall
    # cannot pass between the tube and states that are valid right up to it.
    # Enclosing a non-goal region only strengthens the certificate (invariant
    # 15); a goal region is never added, so a pocket that reaches the goal
    # makes the wall fail instead. (Training only: the proof is the probe.)
    if seed_pieces and len(page.valid):
        vtree = cKDTree(page.valid * page.scale)
        for pc in seed_pieces:
            r = float(np.max(pc.radii * page.scale))
            reach = max(0.08, 2.0 * r)
            for js in vtree.query_ball_point(pc.polyline * page.scale, reach):
                for j in js:
                    cj = int(page.comp[j])
                    if cj not in page.focus_set() and cj not in goal_comps:
                        page.inside.add(cj)
    st["door_pockets"] = len(page.inside)
    st["door_pieces"] = dict(seed=[pc.key for pc in seed_pieces],
                             rest=[pc.key for pc in rest_pieces])
    region = page.seed_points()
    seed = region
    if len(seed) > p3.n_train_max:
        seed = seed[rng.choice(len(seed), p3.n_train_max, replace=False)]
    seed = np.vstack([entries, seed])
    rest0 = page.rest_points()
    if len(rest0) > p3.n_train_max:
        rest0 = rest0[rng.choice(len(rest0), p3.n_train_max, replace=False)]
    if cfg.wall.eta_separation > 0.0 and len(seed):
        seed, rest0, n_cut = eta_separate(seed, rest0, cfg.wall.eta_separation, page.scale)
        st["eta_cut"] = int(n_cut)
    a_seed, a_rest = face_anchors(page, rng)
    st["face_anchors"] = [len(a_seed), len(a_rest)]
    if len(a_seed):
        seed = np.vstack([seed, a_seed])
    gface = rng.uniform(page.lo, page.hi, size=(int(p3.goal_face_samples), 3))
    gface[:, 0] = 1.0 - p3.goal_tol
    rest = np.vstack([rest0, a_rest, gface,
                      virtual_ring3(cfg, page, int(p3.ring_per_face), rng)])
    for pc in seed_pieces:
        seed = np.vstack([seed, piece_samples(page, pc)])
    for pc in rest_pieces:
        rest = np.vstack([rest, piece_samples(page, pc)])
    inv = page.invalid
    if len(inv) > p3.n_invalid_max:
        inv = inv[rng.choice(len(inv), p3.n_invalid_max, replace=False)]

    oracle = robot_oracle3(cfg, page, [], tubes=[pc.tube() for pc in pieces])
    label = (f"{''.join('+' if s > 0 else '-' for s in page.signs)}/"
             f"{page.start_comp if page.start_comp is not None else 'door'}")
    goal_limit = 1.0 - float(p3.goal_tol) - float(getattr(p3, "goal_margin", 0.0))
    try:
        with _budget.wall_attempt(cfg.budget, label):
            poly, rep, wall, log = find_wall3(cfg, page, seed, rest, inv, oracle, verbose,
                                              goal_limit=goal_limit,
                                              goal_feed=1.0 - float(p3.goal_tol))
    except _budget.BudgetExceeded as e:
        if e.scope != "wall":
            raise
        # A stopped wall is a FAILED wall: the region stays in the run, which
        # can therefore only end INCONCLUSIVE (findings 44).
        if verbose:
            print(f"[budget] {e.reason}")
        st["budget_stop"] = e.reason
        info = dict(closed=False, facets="0/0", ok=False,
                    gamma=None if e.gamma is None else round(float(e.gamma), 2))
        return dict(ok=False, poly=Polytope(), wall=info, stats=st, piece_inside={},
                    reason=e.reason)
    st["alg4"] = log
    info = dict(closed=poly.closed, facets=f"{rep.n_passed}/{rep.n_facets}",
                gamma=round(float(wall.gamma), 2), ok=bool(rep.ok and poly.closed),
                goal_limit=round(goal_limit, 4))
    out = dict(ok=False, poly=poly, wall=info, stats=st, piece_inside={}, reason="")
    if not info["ok"]:
        why = rep.failures[0][1] if rep.failures else (poly.note or "not closed")
        out["reason"] = f"no verified wall ({info['facets']} facets; {why})"
        return out
    verts, simp = poly.vertices * page.scale, poly.simplices
    if len(entries) and not np.all(inside(entries * page.scale, verts, simp, 2)):
        info["ok"] = False
        out["reason"] = "verified wall does not enclose every entry state"
        return out
    u_max = float(poly.vertices[:, 0].max())
    st["wall_u_max"] = u_max
    info["u_max"] = round(u_max, 4)
    if u_max >= goal_limit:
        info["ok"] = False
        out["reason"] = (f"wall reaches u={u_max:.4f} >= {goal_limit:.4f}: the goal slab "
                         f"is not provably outside")
        return out
    side = {pc.key: bool(inside(pc.probe()[None, :] * page.scale, verts, simp, 2)[0])
            for pc in pieces}
    bad = [k for k in seed_keys if k in side and not side[k]]
    if bad:
        info["ok"] = False
        out["reason"] = (f"door piece(s) {bad} that must be inside the wall are outside")
        return out
    out["piece_inside"] = side
    out["ok"] = True
    return out


def wall_for_region(cfg: Config, page: Page3, pieces: list, entries: np.ndarray,
                    entry_keys=(), entry_vertex: int | None = None,
                    verbose: bool = True) -> dict:
    """region_wall with the door policy of findings 47 around it.

    Every live door piece starts in the REST class: the wall tries to keep
    it out, since an excluded door is a page the flood never visits. A piece
    moves to the SEED class only on evidence --

      1. a VERIFIED in-page path from the entry to a vertex in its tube
         (checked before the first wall: a reachable door cannot be excluded,
         so trying would only burn a wall), or
      2. the wall failed with it excluded, and the roadmap connects the
         region to its tube (door_promotion_rounds retries).

    Distance to the region decides nothing: a door beside the start can be
    sealed off by an obstacle. Promoting a door that is in fact unreachable
    costs walls, never soundness. Entry pieces are always seed side.
    """
    fp = cfg.flood
    policy = getattr(fp, "door_policy", "exclude_first")
    rounds = int(getattr(fp, "door_promotion_rounds", 1))
    seed_keys = set(entry_keys)
    promoted: dict = {}
    page.inside = set()          # pockets of an earlier region are not ours
    comps = page.focus_set()
    if policy != "legacy":
        for pc in pieces:
            if pc.key in seed_keys:
                continue
            if piece_reach_evidence(page, entry_vertex, pc, comps):
                promoted[pc.key] = "verified path into the door tube"
                seed_keys.add(pc.key)
    attempts = []
    w = None
    for attempt in range(rounds + 1):
        w = region_wall(cfg, page, pieces, entries, seed_keys, verbose, policy)
        attempts.append(dict(attempt=attempt, seed=sorted(seed_keys), ok=w["ok"],
                             facets=w["wall"].get("facets"), reason=w["reason"][:120]))
        if w["ok"] or policy == "legacy" or attempt == rounds or "budget_stop" in w["stats"]:
            break
        # The fallback hint: the region's roadmap reaches next to the tube
        # (Page3Params-free reach of the door-pocket rule), and the wall
        # could not be built with the piece outside. Unverified -- following
        # an unreachable door costs a wall, never soundness.
        new = [pc for pc in pieces if pc.key not in seed_keys
               and piece_near_vertices(page, pc, page.focus_set(), k=1,
                                       reach=max(0.08, 2.0 * float(np.max(pc.radii * page.scale))))]
        if not new:
            break
        for pc in new:
            promoted[pc.key] = "could not be excluded (roadmap-connected)"
            seed_keys.add(pc.key)
        if verbose:
            print(f"[doors-3d] wall failed with {len(new)} roadmap-connected door "
                  f"piece(s) excluded -> promoting them to the seed class, retrying")
    w["promoted"] = promoted
    w["seed_keys"] = sorted(seed_keys)
    w["wall"]["seed_doors"] = [dict(door=list(k), why=v) for k, v in promoted.items()]
    w["wall"]["entry_doors"] = [list(k) for k in sorted(set(entry_keys))]
    w["wall"]["attempts"] = len(attempts)
    w["stats"]["door_attempts"] = attempts
    return w


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


@dataclass
class Result3:
    verdict: str
    reason: str
    stats: dict = field(default_factory=dict)
    doors: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    witness: dict | None = None
    poly: Polytope | None = None
    # "task": the verdict is about the weld task. "page": about the task
    # WITHOUT a page change (prove3(doors_as_exits=True)) -- a live door
    # inside the wall is then an exit the question excludes, not a gap.
    scope: str = "task"


def prove3(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
           signs: tuple, p3: Page3Params, start_theta: float | None = None,
           verbose: bool = True, doors_as_exits: bool = False) -> Result3:
    """Single-page decision on a weld patch: FEASIBLE / INFEASIBLE / INCONCLUSIVE.

    doors_as_exits=True asks the narrower question "can the goal be reached
    on THIS page, without a page change?" (findings 45; result scope
    "page"). The certificate is the same wall -- closed, every triangle in
    C_obs, start inside, goal slab outside -- but live door curves may lie
    inside it: they are exactly the page changes the question excludes. The
    Sigma guard still keeps every tube wholly on one side, so a path that
    stays on the page cannot slip out through a door either.

    Runs under cfg.budget (findings 44): a run that exhausts its budget is
    INCONCLUSIVE "stopped by budget: ...", never a verdict."""
    scope = "page" if doors_as_exits else "task"
    try:
        with _budget.run_budget(cfg.budget, "prove3"):
            res = _prove3(cfg, arm, surf, scene, signs, p3, start_theta, verbose,
                          doors_as_exits)
    except _budget.BudgetExceeded as e:
        if verbose:
            print(f"[budget] {e.reason}")
        res = Result3(INCONCLUSIVE, e.reason, {"budget_stop": e.reason})
    res.scope = scope
    return res


def _prove3(cfg, arm, surf, scene, signs, p3, start_theta, verbose,
            doors_as_exits=False) -> Result3:
    t0 = time.time()
    st: dict = {}
    sing = patch_singularities(cfg, arm, surf)
    st["patch"] = sing["reason"]
    if not sing["ok"]:
        return Result3(INCONCLUSIVE, f"unsupported singular set: {sing['reason']}", st)

    if start_theta is None:
        start_theta = choose_start_theta(cfg, arm, surf, scene, signs, p3)
        if start_theta is None:
            return Result3(INCONCLUSIVE, "no valid start posture at the start point", st)
    st["start"] = [p3.start_u, p3.start_v, float(start_theta)]
    page = Page3(cfg, arm, surf, scene, signs, p3, start_theta)
    st.update(roadmap_valid=len(page.valid), roadmap_invalid=len(page.invalid),
              start_component=int((page.comp == page.start_comp).sum()),
              roadmap_s=round(time.time() - t0, 1))
    if verbose:
        print(f"[page3] {signs}: {len(page.valid)} valid / {len(page.invalid)} invalid "
              f"samples, start component {st['start_component']} "
              f"({st['roadmap_s']} s)")

    cut = theta_cut_check(page)
    st["theta_cut"] = cut["reason"]
    if not cut["ok"]:
        return Result3(INCONCLUSIVE, f"bounded theta not justified: {cut['reason']}", st)

    if page.reaches_goal():
        path = witness_path3(page, page.path_to_goal())
        if path is None:
            return Result3(INCONCLUSIVE, "goal reached on the roadmap, but the path "
                           "could not be densified", st)
        rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start,
                             goal_y=1.0)
        if verbose:
            print(f"[witness-3d] {rep.reason}")
        if not rep.ok:
            return Result3(INCONCLUSIVE, "goal reached on the roadmap, but the path "
                           "failed verification", st, witness=rep.as_dict())
        st["seconds"] = round(time.time() - t0, 1)
        return Result3(FEASIBLE, "a verified path from the start to the goal edge "
                       "exists on this page", st, witness=rep.as_dict())

    curves = door_curves(cfg, arm, surf, p3)
    for c in curves:
        certify_curve(cfg, arm, surf, scene, c, signs, p3)
    live = [c for c in curves if not c.certified_blocked]
    doors = [dict(joint=f"q{c.joint}", n=len(c.pts),
                  u_range=[float(c.pts[:, 0].min()), float(c.pts[:, 0].max())],
                  v_range=[float(c.pts[:, 1].min()), float(c.pts[:, 1].max())],
                  live=not c.certified_blocked, reason=c.reason) for c in curves]
    if verbose:
        for d in doors:
            print(f"[doors-3d] {d['joint']} curve u in [{d['u_range'][0]:.3f}, "
                  f"{d['u_range'][1]:.3f}]: {'LIVE' if d['live'] else 'discharged'} "
                  f"-- {d['reason']}")

    pieces = door_pieces(page, enumerate(live))
    w = wall_for_region(cfg, page, pieces, page.start[None, :], entry_keys=(),
                        entry_vertex=0, verbose=verbose)
    st.update(w["stats"])
    st["seconds"] = round(time.time() - t0, 1)
    if not w["ok"]:
        return Result3(INCONCLUSIVE, w["reason"], st, doors, [w["wall"]], poly=w["poly"])
    # Every piece is probed (findings 47) -- not only each curve's first one.
    inside_pcs = [pc for pc in pieces if w["piece_inside"].get(pc.key)]
    st["live_inside"] = [f"q{pc.joint}#{pc.curve}.{pc.piece}" for pc in inside_pcs]
    if doors_as_exits:
        return Result3(INFEASIBLE, "on this page alone: closed wall, every triangle in "
                       "C_obs, start inside, goal slab outside"
                       + (f"; the only exits are {len(inside_pcs)} live door piece(s) "
                          f"inside the wall ({', '.join(st['live_inside'])}), i.e. page "
                          f"changes" if inside_pcs else ""),
                       st, doors, [w["wall"]], poly=w["poly"])
    if inside_pcs:
        return Result3(INCONCLUSIVE, f"live q{inside_pcs[0].joint} door piece is INSIDE "
                       f"the wall: following it needs the cross-page flood "
                       f"(flood3.run_flood3)", st, doors, [w["wall"]], poly=w["poly"])
    return Result3(INFEASIBLE, "closed wall, every triangle in C_obs, start inside, goal "
                   "slab and every live door outside", st, doors, [w["wall"]], poly=w["poly"])
