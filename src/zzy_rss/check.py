"""check(A, B): one attempt at cutting A from B (docs/recursive_proof_design.md, step 2b).

The world is the glued Y+ of a weld PATCH: eight pages (u, v, theta), joined
at the doors. A and B are SETS:

    StartSet(signs)          the start posture, on its page
    GoalSet()                every state with u >= 1 - goal_tol, on every page
    DoorNode(door, pages)    the door region `door` as the place where the
                             two pages {P, flip(P)} meet; the robot standing
                             in it may continue on either page

check(A, B) builds, on EVERY page A lives on, a wall that encloses A and
excludes B, and returns one of

    CONNECTED   A and B are in one roadmap component on one of A's pages: no
                wall can separate them there. (Not a verdict: feasibility is
                decided by a VERIFIED path elsewhere, never by this.)
    WALLED      every wall is closed and verified, A is inside and B outside
                on every page; `doors` lists the live door regions that are
                not certifiably outside some wall -- the split points.
    FAILED      anything else.

The three tests of the design:

  (i)   every facet in C_obs                       IJRR Algorithm 3
  (ii)  every facet away from LIVE doors           FacetOracle.sigma_many:
        the facet's ball is certified free of zeros of sigma, or misses every
        cell of every live door region. Every zero of sigma on the page lies
        in some region's cells (interior.find_doors over the whole box), so
        missing the live cells means missing every live door; a door
        discharged by its posture (collision, or a constant joint out of
        range) may be touched.
  (iii) doors inside the wall                      a door region with ANY cell
        the wall does not certifiably exclude (interior.polytope_region) is
        treated as inside. Regions, not the sampled curves, decide this.

No door list is used: the regions come from the chart's sigma, the curves
(for crossing states, posture checks and training) from minimising it
(doorset.py).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import budget as _budget
from .arm import SrsArm
from .chart import SrsChart
from .doors3 import certify_curve
from .doorset import DoorSet, find_doorset
from .facets import FacetOracle
from .flood3 import door_samples, flip, page_name
from .interior import OUT, polytope_region
from .page3 import (Page3, choose_start_theta, clip_to_box, robot_oracle3,
                    theta_cut_check, tube_samples, virtual_ring3)
from .params import Config, Page3Params
from .scene import Scene
from .source import eta_separate
from .tangential import inside, reconstruct
from .wall import CheckReport, Polytope, assign_label, check_polytope, sample_wall, train_wall
from .weld import WeldSurface

CONNECTED, WALLED, FAILED = "CONNECTED", "WALLED", "FAILED"


# --------------------------------------------------------------------------
# sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StartSet:
    signs: tuple

    def label(self) -> str:
        return f"start@{page_name(self.signs)}"


@dataclass(frozen=True)
class GoalSet:
    def label(self) -> str:
        return "goal"


@dataclass(frozen=True)
class DoorNode:
    door: int
    pages: frozenset

    def label(self) -> str:
        return f"D{self.door}[{'|'.join(page_name(p) for p in sorted(self.pages))}]"


# --------------------------------------------------------------------------
# the world: pages, doors, crossing states
# --------------------------------------------------------------------------


class World:
    """Everything check() and the recursion share for one task."""

    def __init__(self, cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                 start_signs: tuple, p3: Page3Params, start_theta: float | None = None,
                 verbose: bool = True):
        self.cfg, self.arm, self.surf, self.scene = cfg, arm, surf, scene
        self.p3, self.start_signs, self.verbose = p3, tuple(start_signs), verbose
        self.chart = SrsChart.for_surface(arm, surf, p3)
        self.scale = self.chart.scale
        t = time.time()
        self.doorset: DoorSet = find_doorset(self.chart, p3.tube_radius)
        self.door_seconds = round(time.time() - t, 1)
        if verbose:
            print(f"[doors] sigma search: {len(self.doorset.doors)} door region(s), "
                  f"complete={self.doorset.complete} ({self.door_seconds} s)")
            for d in self.doorset.doors:
                print(f"[doors]   {d.label()}: {d.note}")
        self.curves, self.curve_door = [], []
        for d in self.doorset.doors:
            for c in d.curves:
                self.curves.append(c)
                self.curve_door.append(d.index)
        self.curve_door = np.asarray(self.curve_door, dtype=int)
        self.ds = door_samples(self.curves, self.scale, float(p3.cross_eps), int(p3.cert_stride))
        self.sample_door = (self.curve_door[[k[0] for k in self.ds.key]]
                            if len(self.ds.key) else np.zeros(0, dtype=int))
        if start_theta is None:
            start_theta = choose_start_theta(cfg, arm, surf, scene, self.start_signs, p3)
        self.start_theta = start_theta
        self.pages: dict = {}
        self._discharge: dict = {}
        self._live_index: dict = {}
        self.walls_built = 0

    # -- pages -------------------------------------------------------------

    def page(self, signs: tuple) -> Page3:
        signs = tuple(signs)
        if signs not in self.pages:
            t = time.time()
            pg = Page3(self.cfg, self.arm, self.surf, self.scene, signs, self.p3,
                       self.start_theta if signs == self.start_signs else None,
                       inject=self.ds.pts)
            cut = theta_cut_check(pg)
            pg.cut_ok, pg.cut_reason = cut["ok"], cut["reason"]
            self.pages[signs] = pg
            if self.verbose:
                print(f"[world] page {page_name(signs)}: {len(pg.valid)} valid vertices, "
                      f"{int((pg.inject_index >= 0).sum())}/{len(self.ds.pts)} crossing "
                      f"states valid ({time.time() - t:.1f} s)")
        return self.pages[signs]

    # -- doors -------------------------------------------------------------

    def door(self, k: int):
        return self.doorset.doors[k]

    def discharged(self, signs: tuple, k: int) -> tuple:
        """(blocked, reason) of door k as seen from page `signs`. Only a
        posture fact discharges (invariant 41), only on a door whose curves
        are certified to account for its whole region (Door.check_coverage),
        and only if EVERY curve of it is blocked."""
        key = (tuple(signs), k)
        if key in self._discharge:
            return self._discharge[key]
        d = self.door(k)
        if not d.supported or not d.curves:
            out = (False, "unsupported or no curve: live")
        else:
            why = []
            blocked = True
            for c in d.curves:
                cc = type(c)(sign_index=c.sign_index, pts=c.pts, closed=c.closed, tube=c.tube)
                certify_curve(self.cfg, self.arm, self.surf, self.scene, cc, tuple(signs), self.p3)
                why.append(cc.reason)
                blocked &= bool(cc.certified_blocked)
            if blocked and not d.check_coverage(self.chart):
                blocked = False
                why.append(f"curves do not cover the region (farthest fine cell "
                           f"{d.cover_far:.3f}): stays live")
            out = (blocked, "; ".join(why))
        self._discharge[key] = out
        return out

    def live_doors(self, signs: tuple) -> list:
        return [d.index for d in self.doorset.doors if not self.discharged(signs, d.index)[0]]

    def live_cells(self, signs: tuple):
        """(KD-tree, centres_scaled, half_scaled) over every cell of every
        live door region, for test (ii)."""
        signs = tuple(signs)
        if signs not in self._live_index:
            C, H = [], []
            for k in self.live_doors(signs):
                r = self.door(k).region
                C.append(r.centres * self.scale)
                H.append(r.half * self.scale)
            if C:
                C, H = np.vstack(C), np.vstack(H)
                self._live_index[signs] = (cKDTree(C), C, H)
            else:
                self._live_index[signs] = (None, np.zeros((0, 3)), np.zeros((0, 3)))
        return self._live_index[signs]

    def sigma_clear(self, signs: tuple):
        """FacetOracle.sigma_many for page `signs` (test ii)."""
        tree, Cl, Hl = self.live_cells(signs)
        hmax = float(np.max(np.linalg.norm(Hl, axis=1))) if len(Hl) else 0.0
        period = 2.0 * np.pi * self.scale[-1]

        def clear(C, R):
            C = np.atleast_2d(np.asarray(C, dtype=float))
            R = np.broadcast_to(np.asarray(R, dtype=float), (len(C),))
            ok = self.chart.ball_test(C, R).clear
            if tree is None:
                return np.ones(len(C), dtype=bool)
            for i in np.flatnonzero(~ok):
                hit = False
                for sh in (0.0, period, -period):
                    c = C[i] * self.scale
                    c = c + np.array([0.0, 0.0, sh])
                    js = tree.query_ball_point(c, float(R[i]) + hmax)
                    if js:
                        js = np.asarray(js)
                        gap = np.maximum(0.0, np.abs(Cl[js] - c) - Hl[js])
                        if np.any(np.linalg.norm(gap, axis=1) <= R[i]):
                            hit = True
                            break
                ok[i] = not hit
            return ok
        return clear

    # -- set membership on a page -----------------------------------------

    def states(self, S, signs: tuple) -> np.ndarray:
        """Vertex indices of set S on page `signs` (empty if S is not there)."""
        page = self.page(signs)
        if isinstance(S, StartSet):
            return np.array([0]) if (tuple(signs) == S.signs and page.start is not None) \
                else np.zeros(0, dtype=int)
        if isinstance(S, GoalSet):
            return np.flatnonzero(page.goal_mask())
        if isinstance(S, DoorNode):
            if tuple(signs) not in S.pages:
                return np.zeros(0, dtype=int)
            n = np.flatnonzero(self.sample_door == S.door)
            v = page.inject_index[n]
            return np.unique(v[v >= 0])
        raise TypeError(S)

    def pages_of(self, S) -> list:
        if isinstance(S, StartSet):
            return [S.signs]
        if isinstance(S, DoorNode):
            return sorted(S.pages)
        raise TypeError("only a start or a door can be the enclosed set A")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


@dataclass
class PageWall:
    signs: tuple
    ok: bool
    reason: str = ""
    poly: Polytope | None = None
    info: dict = field(default_factory=dict)
    doors_inside: list = field(default_factory=list)      # door indices


@dataclass
class CheckResult:
    kind: str
    reason: str = ""
    walls: list = field(default_factory=list)              # PageWall
    doors: list = field(default_factory=list)              # DoorNode, the split points
    connected_on: tuple | None = None


def check(world: World, A, B, verbose: bool = True) -> CheckResult:
    pages = world.pages_of(A)
    # CONNECTED first: no wall can cut what the roadmap already joins.
    for signs in pages:
        page = world.page(signs)
        if not page.cut_ok:
            return CheckResult(FAILED, f"bounded theta not justified on {page_name(signs)}: "
                                       f"{page.cut_reason}")
        a = world.states(A, signs)
        b = world.states(B, signs)
        if len(a) and len(b) and np.intersect1d(page.comp[a], page.comp[b]).size:
            return CheckResult(CONNECTED, f"{A.label()} and {B.label()} share a roadmap "
                                          f"component on {page_name(signs)}",
                               connected_on=tuple(signs))
    walls, doors = [], []
    for signs in pages:
        w = page_wall(world, signs, A, B, verbose=verbose)
        walls.append(w)
        if not w.ok:
            return CheckResult(FAILED, f"{page_name(signs)}: {w.reason}", walls=walls)
        for k in w.doors_inside:
            dn = DoorNode(door=k, pages=frozenset({tuple(signs),
                                                   flip(tuple(signs), world.door(k).sign_index)}))
            if dn not in doors:
                doors.append(dn)
    return CheckResult(WALLED, "verified wall(s): A inside, B outside", walls=walls,
                       doors=doors)


# --------------------------------------------------------------------------
# one page's wall
# --------------------------------------------------------------------------


class _SetLabeller:
    """SDCL's connectivity rule with explicit seed / rest components."""

    def __init__(self, page: Page3, seed_comps: set, rest_comps: set):
        self.page, self.S, self.R = page, seed_comps, rest_comps
        self.tree = cKDTree(page.valid * page.scale)

    def label(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float)
        k = min(int(self.page.p3.anchor_connect_tries), len(self.page.valid))
        _, js = self.tree.query(x * self.page.scale, k=k)
        for j in np.atleast_1d(js):
            if self.page._edge(x, self.page.valid[int(j)]):
                c = int(self.page.comp[int(j)])
                return 1 if c in self.S else (-1 if c in self.R else 0)
        return 0


def _face_anchors(page: Page3, lab: _SetLabeller, rng) -> tuple:
    """Valid states on the four task faces, labelled by connectivity."""
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
    seed, rest = [], []
    for x in cand:
        lb = lab.label(x)
        if lb > 0:
            seed.append(x)
        elif lb < 0:
            rest.append(x)
    return np.asarray(seed).reshape(-1, 3), np.asarray(rest).reshape(-1, 3)


def page_wall(world: World, signs: tuple, A, B, verbose: bool = True) -> PageWall:
    """The wall on page `signs` around A's components, excluding B; verified."""
    cfg, p3 = world.cfg, world.p3
    page = world.page(signs)
    scale = page.scale
    rng = np.random.default_rng(p3.rng_seed + 3)
    st: dict = {}

    a_v = world.states(A, signs)
    b_v = world.states(B, signs)
    S = set(int(c) for c in page.comp[a_v])
    R = set(int(c) for c in page.comp[b_v]) - S
    # The goal is excluded on every page when it is the target; a door
    # target only on its own pages (elsewhere it is just another door).
    live = [k for k in world.live_doors(signs)]
    a_door = A.door if isinstance(A, DoorNode) else None
    b_door = B.door if isinstance(B, DoorNode) and tuple(signs) in B.pages else None

    # Which door CURVES the wall must enclose (the training aim; verification
    # decides): every curve of A's own door, and every other live curve that
    # comes near A's components (today's per-curve rule, page3.tube_samples).
    # Per curve, not per region: a region can hold two curves (a loose bound
    # merges them), one beside the start and one beside the goal.
    region0 = page.valid[np.isin(page.comp, sorted(S))] if S else np.zeros((0, 3))
    tree0 = cKDTree(region0 * scale) if len(region0) else None
    vtree = cKDTree(page.valid * scale)
    curves_in, curves_out = [], []
    for k in live:
        for c in world.door(k).curves:
            if k == a_door:
                curves_in.append(c)
                continue
            if k == b_door:
                curves_out.append(c)
                continue
            pieces = clip_to_box(np.asarray(c.tube.polyline, dtype=float), page.lo, page.hi)
            if not pieces:
                continue
            near = False
            if tree0 is not None:
                P = np.vstack(pieces) * scale
                r = float(np.max(np.asarray(c.tube.radii) * scale))
                near = float(tree0.query(P, k=1)[0].min()) <= max(0.03, r)
            (curves_in if near else curves_out).append(c)
    # Door pockets (invariant 53): non-target components reaching into the
    # tube of an enclosed curve go inside too.
    for c in curves_in:
        pieces = clip_to_box(np.asarray(c.tube.polyline, dtype=float), page.lo, page.hi)
        if not pieces:
            continue
        P = np.vstack(pieces) * scale
        r = float(np.max(np.asarray(c.tube.radii) * scale))
        for js in vtree.query_ball_point(P, max(0.08, 2.0 * r)):
            for j in js:
                cj = int(page.comp[j])
                if cj not in R:
                    S.add(cj)
    st["seed_components"] = len(S)
    st["curves_in"] = len(curves_in)
    lab = _SetLabeller(page, S, R)

    # ---- training set (the wall's AIM; verification decides) -------------
    seed = page.valid[np.isin(page.comp, sorted(S))] if S else np.zeros((0, 3))
    if len(seed) > p3.n_train_max:
        seed = seed[rng.choice(len(seed), p3.n_train_max, replace=False)]
    seed = np.vstack([page.valid[a_v], seed]) if len(a_v) else seed
    rest0 = page.valid[np.isin(page.comp, sorted(R))] if R else np.zeros((0, 3))
    if len(rest0) > p3.n_train_max:
        rest0 = rest0[rng.choice(len(rest0), p3.n_train_max, replace=False)]
    if cfg.wall.eta_separation > 0.0 and len(seed) and len(rest0):
        seed, rest0, _ = eta_separate(seed, rest0, cfg.wall.eta_separation, scale)
    a_seed, a_rest = _face_anchors(page, lab, rng)
    rest_parts = [rest0, a_rest, virtual_ring3(cfg, page, int(p3.ring_per_face), rng)]
    if isinstance(B, GoalSet):
        gface = rng.uniform(page.lo, page.hi, size=(int(p3.goal_face_samples), 3))
        gface[:, 0] = 1.0 - p3.goal_tol
        rest_parts.append(gface)
    seed_parts = [seed, a_seed]
    # Door samples: enclosed curves -> seed; the target door and every other
    # live curve -> rest (outside is never wrong for them).
    if curves_in:
        s_in, r_in = tube_samples(curves_in, np.zeros((0, 3)), scale, 0.0, page.lo, page.hi)
        seed_parts.append(np.vstack([s_in, r_in]))
    if curves_out:
        s_out, r_out = tube_samples(curves_out, np.zeros((0, 3)), scale, 0.0, page.lo, page.hi)
        rest_parts.append(np.vstack([s_out, r_out]))
    seed = np.vstack([x for x in seed_parts if len(x)])
    rest = np.vstack([x for x in rest_parts if len(x)])
    inv = page.invalid
    if len(inv) > p3.n_invalid_max:
        inv = inv[rng.choice(len(inv), p3.n_invalid_max, replace=False)]

    oracle = dataclasses.replace(robot_oracle3(cfg, page, []),
                                 sigma_many=world.sigma_clear(signs),
                                 sigma_max_depth=int(p3.sigma_max_depth))
    label = f"{page_name(signs)}:{A.label()}|{B.label()}"
    world.walls_built += 1
    try:
        with _budget.wall_attempt(cfg.budget, label):
            poly, rep, wall, log = _find_wall(world, page, seed, rest, inv, oracle, lab, verbose)
    except _budget.BudgetExceeded as e:
        if e.scope != "wall":
            raise
        return PageWall(tuple(signs), False, e.reason, info=dict(budget_stop=e.reason))
    st["alg4"] = log
    info = dict(closed=poly.closed, facets=f"{rep.n_passed}/{rep.n_facets}",
                gamma=round(float(wall.gamma), 2), stats=st)
    if not (poly.closed and rep.ok):
        why = rep.failures[0][1] if rep.failures else (poly.note or "not closed")
        return PageWall(tuple(signs), False, f"no verified wall ({info['facets']}; {why})",
                        poly=poly, info=info)
    return _verify(world, signs, A, B, poly, info)


def _verify(world: World, signs: tuple, A, B, poly: Polytope, info: dict) -> PageWall:
    """A inside, B outside, and the doors the wall does not exclude."""
    page = world.page(signs)
    scale = page.scale
    verts, simp = poly.vertices * scale, poly.simplices
    region = polytope_region(poly, scale)

    def cells_code(k):
        r = world.door(k).region
        return region(r.centres, np.linalg.norm(r.half * scale, axis=1))

    a_v = world.states(A, signs)
    if isinstance(A, StartSet):
        if not bool(inside(page.start[None, :] * scale, verts, simp, 2)[0]):
            return PageWall(tuple(signs), False, "the start is not inside the wall",
                            poly=poly, info=info)
    else:
        code = cells_code(A.door)
        if not np.all(code == 1):
            n = int(np.sum(code != 1))
            return PageWall(tuple(signs), False, f"{n} cell(s) of {A.label()} not certified "
                                                 f"inside the wall", poly=poly, info=info)
    if len(a_v) and not np.all(inside(page.valid[a_v] * scale, verts, simp, 2)):
        return PageWall(tuple(signs), False, "a state of A is outside the wall",
                        poly=poly, info=info)
    u_max = float(poly.vertices[:, 0].max())
    info["u_max"] = round(u_max, 4)
    if isinstance(B, GoalSet) and u_max >= 1.0 - world.p3.goal_tol:
        return PageWall(tuple(signs), False, f"wall reaches u={u_max:.4f}: the goal slab is "
                                             f"not provably outside", poly=poly, info=info)
    if isinstance(B, DoorNode) and tuple(signs) in B.pages:
        code = cells_code(B.door)
        if np.any(code != OUT):
            return PageWall(tuple(signs), False, f"{B.label()} is not certified outside the "
                                                 f"wall", poly=poly, info=info)
    doors_in = []
    for d in world.doorset.doors:
        if isinstance(A, DoorNode) and d.index == A.door:
            continue
        code = cells_code(d.index)
        if np.all(code == OUT):
            continue
        if not d.supported:
            return PageWall(tuple(signs), False, f"unsupported singular set {d.label()} "
                                                 f"inside the wall", poly=poly, info=info)
        blocked, _ = world.discharged(signs, d.index)
        if blocked:
            continue
        doors_in.append(d.index)
    info["doors_inside"] = [world.door(k).label() for k in doors_in]
    return PageWall(tuple(signs), True, "verified", poly=poly, info=info, doors_inside=doors_in)


def _find_wall(world: World, page: Page3, seed, rest, inv, oracle: FacetOracle,
               lab: _SetLabeller, verbose: bool) -> tuple:
    """IJRR Algorithm 4 on one page (page3.find_wall3, with the set labeller
    and one more failure route: a facet near a live door feeds the door's
    nearest curve points back, labelled by connectivity or nearest class)."""
    cfg, p3 = world.cfg, world.p3
    rng = np.random.default_rng(p3.rng_seed + 2)
    scale = page.scale
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(
        cfg.wall, label_mode="topological", feedback_margin=float(p3.feedback_margin)))
    m = float(p3.window_margin) / scale
    lo_s, hi_s = (page.lo - m) * scale, (page.hi + m) * scale
    wall = train_wall(cfg, seed, rest, scale=scale)
    eta = float(p3.eta)
    poly, rep, log = Polytope(), CheckReport(False, 0, 0), []
    checked = wall
    door_pts = np.vstack([c.pts for c in world.curves]) if world.curves else np.zeros((0, 3))
    dtree = cKDTree(door_pts * scale) if len(door_pts) else None
    for it in range(1, int(p3.alg4_iters) + 1):
        _budget.check_time()
        t0 = time.time()
        wall, wall_pts, seed, rest, rounds = sample_wall(
            cfg, page.arm, None, page.scene, page.signs, wall, inv, seed, rest,
            verbose=False, valid_fn=page.valid_task, gamma_warm=p3.gamma_warm,
            valid_many=page.valid_task_many, labeller=lab)
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
        entry = dict(iter=it, gamma=round(wall.gamma, 2), triangles=len(rec.simplices),
                     closed=rec.closed)
        if not rec.closed:
            log.append(entry)
            if verbose:
                print(f"[check] alg4 iter {it}: wall not closed ({rec.note})")
            _budget.note_unclosed(rec.note)
            eta *= 0.5
            continue
        _budget.note_closed()
        poly = Polytope(closed=True, n_segments=len(rec.simplices), method="tdc",
                        vertices=rec.points / scale, simplices=rec.simplices, stats=rec.stats)
        checked = wall
        rep = check_polytope(cfg, None, None, None, page.signs, poly, [], None,
                             verbose=False, oracle=oracle)
        n_sig = sum(1 for _, why in rep.failures if "sigma" in why)
        entry.update(facets=f"{rep.n_passed}/{rep.n_facets}", sigma_fail=n_sig,
                     seconds=round(time.time() - t0, 1))
        log.append(entry)
        if verbose:
            print(f"[check] alg4 iter {it}: gamma {wall.gamma:.2f}, {len(rec.simplices)} "
                  f"triangles, facets {rep.n_passed}/{rep.n_facets} ({n_sig} near a live "
                  f"door) ({time.time() - t0:.1f} s)")
        if rep.ok:
            return poly, rep, checked, log
        feed = []
        for c, why in rep.failure_points:
            c = np.asarray(c, dtype=float)
            if "VALID" in why:
                feed.append(c)
            elif "depth limit" in why:
                v = page.nearest_valid_point(c)
                if v is not None:
                    feed.append(v)
            elif "sigma" in why and dtree is not None:
                for j in dtree.query_ball_point(c * scale, 0.1):
                    feed.append(door_pts[j])
        added = 0
        for c in feed:
            lb = lab.label(c) or assign_label(c, seed, rest, scale)
            other = rest if lb > 0 else seed
            if cfg.wall.feedback_margin > 0 and len(other) and float(np.min(
                    np.linalg.norm((other - c) * scale, axis=1))) < cfg.wall.feedback_margin:
                continue
            if lb > 0:
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
    return poly, rep, checked, log
