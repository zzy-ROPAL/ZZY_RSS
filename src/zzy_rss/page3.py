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

from .arm import SrsArm
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
                 signs: tuple, p3: Page3Params, start_theta: float,
                 n_samples: int | None = None):
        self.cfg, self.arm, self.surf, self.scene = cfg, arm, surf, scene
        self.signs, self.p3 = signs, p3
        self.scale = np.asarray(p3.scale, dtype=float)
        self.lo = np.array([0.0, 0.0, p3.theta_min])
        self.hi = np.array([1.0, 1.0, p3.theta_max])
        self.start = np.array([p3.start_u, p3.start_v, start_theta], dtype=float)
        if not self.valid_fn(self.start):
            raise ValueError(f"start pose {self.start} is not valid on page {signs}")
        rng = np.random.default_rng(p3.rng_seed)
        n = int(n_samples or p3.n_samples)
        pts = rng.uniform(self.lo, self.hi, size=(n, 3))
        ok = np.array([self.valid_fn(x) for x in pts])
        # The start goes first so its index is 0 and it is never thinned.
        self.valid = np.vstack([self.start[None, :], pts[ok]])
        self.invalid = pts[~ok]
        self._build_graph()

    def valid_fn(self, pt) -> bool:
        return classify3(self.cfg, self.arm, self.surf, self.scene, pt, self.signs)

    def _edge(self, a: np.ndarray, b: np.ndarray) -> bool:
        step = self.cfg.planner.edge_resolution
        m = max(2, int(np.ceil(float(np.linalg.norm((b - a) * self.scale)) / step)))
        return all(self.valid_fn(a + t * (b - a)) for t in np.linspace(0.0, 1.0, m + 1))

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
            if uf.find(i) == uf.find(j):
                continue
            if self._edge(self.valid[i], self.valid[j]):
                uf.union(i, j)
                self.adj[i].append(j)
                self.adj[j].append(i)
        roots: dict = {}
        self.comp = np.array([roots.setdefault(uf.find(i), len(roots))
                              for i in range(n)])

    # -- the five questions, page-local -----------------------------------

    @property
    def start_comp(self) -> int:
        return int(self.comp[0])

    def goal_mask(self) -> np.ndarray:
        return self.valid[:, 0] >= 1.0 - self.p3.goal_tol

    def reaches_goal(self) -> bool:
        return bool(np.any(self.goal_mask() & (self.comp == self.start_comp)))

    def seed_points(self) -> np.ndarray:
        return self.valid[self.comp == self.start_comp]

    def rest_points(self) -> np.ndarray:
        """IJRR's split (invariant 15): the components that reach the goal."""
        goal_comps = set(int(c) for c in self.comp[self.goal_mask()]) - {self.start_comp}
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
                return 1 if c == self.start_comp else (-1 if c in goal_comps else 0)
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
        goal = self.goal_mask()
        prev = {0: None}
        queue = [0]
        for i in queue:
            if goal[i]:
                out = [i]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            for j in self.adj[i]:
                if j not in prev:
                    prev[j] = i
                    queue.append(j)
        return None


def choose_start_theta(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                       signs: tuple, p3: Page3Params, n: int = 721) -> float | None:
    """The middle of the widest valid theta interval at (start_u, start_v).

    This only CHOOSES the start posture -- a 1-D scan of the arm angle at one
    point of the task -- and plays no part in any proof.
    """
    th = np.linspace(p3.theta_min, p3.theta_max, n)
    ok = np.array([classify3(cfg, arm, surf, scene, (p3.start_u, p3.start_v, t), signs)
                   for t in th])
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
    hits = [x for x in pts if page.valid_fn(x)]
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


def witness_path3(page: Page3, idx: list) -> WitnessPath | None:
    """Densify a roadmap path and solve IK at every sample. Refined until the
    JOINT step respects step_max -- the bound the verifier rests on."""
    arm, surf, wp = page.arm, page.surf, page.cfg.witness
    way = page.valid[idx]
    qs, uv = [], []
    for a, b in zip(way[:-1], way[1:]):
        n = max(2, int(np.ceil(float(np.linalg.norm((b - a) * page.scale)) / wp.page_step)))
        for _ in range(8):
            seg = []
            for t in np.linspace(0.0, 1.0, n + 1):
                p = a + t * (b - a)
                r = arm.ik_branch(*surf.pose(float(p[0]), float(p[1])), float(p[2]),
                                  page.signs)
                if not r.ok:
                    return None
                seg.append((r.q, p[:2].copy()))
            qq = np.asarray([s[0] for s in seg])
            if float(np.abs(np.diff(qq, axis=0)).max()) <= wp.step_max:
                break
            n *= 2
        for q_i, uv_i in seg[:-1]:
            qs.append(q_i)
            uv.append(uv_i)
    r = arm.ik_branch(*surf.pose(float(way[-1][0]), float(way[-1][1])), float(way[-1][2]),
                      page.signs)
    if not r.ok:
        return None
    qs.append(r.q)
    uv.append(way[-1][:2].copy())
    return WitnessPath(legs=[Leg(kind=LEG_PAGE, page=page.signs, q=np.asarray(qs),
                                 y=np.asarray(uv), waypoints=way)])


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


def tube_samples(curves: list, region_pts: np.ndarray, scale: np.ndarray,
                 tol: float) -> tuple:
    """Training points that keep the wall off the door tubes (invariant 7).

    A curve that comes near the start region goes WHOLE into the seed class
    (the wall must enclose it), any other into the rest class. Per curve, not
    per point: a curve split between the classes would put the wall across
    its own tube."""
    seed, rest = [], []
    tree = cKDTree(region_pts * scale) if len(region_pts) else None
    for c in curves:
        r = np.asarray(c.tube.radii, dtype=float)
        pts = [c.pts]
        for ax in range(3):
            for sgn in (-1.0, 1.0):
                off = np.zeros(3)
                off[ax] = sgn * r[ax]
                pts.append(c.pts + off)
        pts = np.vstack(pts)
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
    ok = np.array([page.valid_fn(x) for x in cand])
    cand = cand[ok]
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


def robot_oracle3(cfg: Config, page: Page3, live_curves: list) -> FacetOracle:
    """Algorithm 3's view of the 3-D page. Penetration depth by the capped
    march toward the nearest VALID roadmap samples of either class (closer
    valid points only tighten the bound)."""
    guard = tube_guard([c.tube for c in live_curves]) if live_curves else (lambda v: None)
    valid_all = page.valid
    wp = cfg.wall

    def depth(c, need):
        return pd_march(c, valid_all, page.valid_fn, page.scale,
                        step=float(cfg.planner.edge_resolution),
                        attempts=int(wp.pd_seed_attempts),
                        bisect_iters=int(wp.pd_bisect_iters), cap=float(need))

    return FacetOracle(scale=page.scale, guard=guard,
                       virtual=lambda c: _box_band(cfg, page, c),
                       valid=page.valid_fn, depth=depth)


class _PageLabeller:
    """wall.sample_wall's labeller interface over Page3.label_point."""

    def __init__(self, page: "Page3"):
        self.page = page

    def label(self, m: np.ndarray) -> int:
        return self.page.label_point(np.asarray(m, dtype=float))


def find_wall3(cfg: Config, page: Page3, seed: np.ndarray, rest: np.ndarray,
               inv: np.ndarray, oracle: FacetOracle, verbose: bool = True) -> tuple:
    """IJRR Algorithm 4 on the 3-D page. Returns (poly, report, wall, log)."""
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
        t0 = time.time()
        wall, wall_pts, seed, rest, rounds = sample_wall(
            cfg, page.arm, None, page.scene, page.signs, wall, inv, seed, rest,
            verbose=False, valid_fn=page.valid_fn, gamma_warm=p3.gamma_warm,
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
            eta *= 0.5
            continue
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


def prove3(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
           signs: tuple, p3: Page3Params, start_theta: float | None = None,
           verbose: bool = True) -> Result3:
    """Single-page decision on a weld patch: FEASIBLE / INFEASIBLE / INCONCLUSIVE."""
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

    rng = np.random.default_rng(p3.rng_seed + 3)
    seed = page.seed_points()
    if len(seed) > p3.n_train_max:
        seed = seed[np.r_[0, 1 + rng.choice(len(seed) - 1, p3.n_train_max - 1, replace=False)]]
    rest0 = page.rest_points()
    if len(rest0) > p3.n_train_max:
        rest0 = rest0[rng.choice(len(rest0), p3.n_train_max, replace=False)]
    if cfg.wall.eta_separation > 0.0:
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
    s_extra, r_extra = tube_samples(live, page.seed_points(), page.scale, 0.03)
    if len(s_extra):
        seed = np.vstack([seed, s_extra])
    if len(r_extra):
        rest = np.vstack([rest, r_extra])
    inv = page.invalid
    if len(inv) > p3.n_invalid_max:
        inv = inv[rng.choice(len(inv), p3.n_invalid_max, replace=False)]

    oracle = robot_oracle3(cfg, page, live)
    poly, rep, wall, log = find_wall3(cfg, page, seed, rest, inv, oracle, verbose)
    st["alg4"] = log
    st["seconds"] = round(time.time() - t0, 1)
    walls = [dict(closed=poly.closed, facets=f"{rep.n_passed}/{rep.n_facets}",
                  gamma=round(float(wall.gamma), 2), ok=bool(rep.ok and poly.closed))]
    if not (rep.ok and poly.closed):
        why = rep.failures[0][1] if rep.failures else (poly.note or "not closed")
        return Result3(INCONCLUSIVE, f"no verified wall ({walls[0]['facets']} facets; "
                       f"{why})", st, doors, walls, poly=poly)

    verts, simp = poly.vertices * page.scale, poly.simplices
    if not inside(page.start[None, :] * page.scale, verts, simp, 2)[0]:
        return Result3(INCONCLUSIVE, "verified wall does not enclose the start",
                       st, doors, walls, poly=poly)
    u_max = float(poly.vertices[:, 0].max())
    st["wall_u_max"] = u_max
    if u_max >= 1.0 - p3.goal_tol:
        return Result3(INCONCLUSIVE, f"wall reaches u={u_max:.4f}: the goal slab is not "
                       f"provably outside", st, doors, walls, poly=poly)
    for c in live:
        if inside(c.pts[:1] * page.scale, verts, simp, 2)[0]:
            return Result3(INCONCLUSIVE, f"live q{c.joint} door curve is INSIDE the wall: "
                           f"following it needs the 3-D cross-page flood",
                           st, doors, walls, poly=poly)
    return Result3(INFEASIBLE, "closed wall, every triangle in C_obs, start inside, goal "
                   "slab and every live door outside", st, doors, walls, poly=poly)
