"""Where the proof gets its five answers from -- a grid, or a roadmap.

`run_proof` asks a page exactly five things:

    regions()          the connected components of the valid set
    seed_pool(k)       the start class
    rest_pool(k, mode) the class to be excluded
    obstacle_points()  Algorithm 2's projection seeds
    regions_under(..)  which component a door's anchor lands in, over there

Four of those have an adjudicator downstream: answer them wrong and the wall
is asked to separate two points that are actually connected, a facet centre
lands in the valid set, Algorithm 3 rejects it, and the result is
INCONCLUSIVE. `regions_under` had none -- a component never enqueued is never
looked at again -- which made it the one place where a sampling error could
become a false proof.

GridSource is the behaviour that shipped: cells of a PageMap.
RoadmapSource replaces it with a PRM and no grid at all.

Why the roadmap is HARDER, not easier
-------------------------------------
A grid is a uniform lattice, so two points of opposite classes are never
closer than one cell. That is an artefact: it OVERSTATES the separation.
Measured on EXP 3's start page, the two classes' nearest pair sits at

    grid     0.07502        ->  gamma ~ 1/d^2 ~ 178
    roadmap  0.03839        ->  gamma ~ 679

and the roadmap figure is the true one -- it is the obstacle's actual
thickness in page coordinates. Algorithm 1 raises gamma until the training
accuracy is 1, so a truer measurement means a narrower kernel, and at 400 the
accuracy is still 0.9956 while at 1500 the zero set fragments and the wall
will not close.

`eta_separate` is the fix, and it is IJRR's own Definition 2 (an (eps, eta)-
net) applied across the two classes rather than within one. See its docstring
for why dropping points is sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .arm import SrsArm
from .params import Config
from .pagemap import PageMap
from .prm import UnionFind, classify
from .scene import Scene
from .weld import WeldLine


@dataclass
class Region:
    """One page-restricted connected component of the valid set."""

    page: str
    index: int
    n_cells: int
    y_min: float
    y_max: float
    reaches_start: bool
    reaches_goal: bool


# --------------------------------------------------------------------------
# eta-separation
# --------------------------------------------------------------------------


def eta_separate(seed: np.ndarray, rest: np.ndarray, eta: float,
                 scale: np.ndarray) -> tuple:
    """Drop BOTH members of every cross-class pair closer than eta.

    This is what restores the grid's one useful property without a grid: after
    it, no training point of one class lies within eta of the other class, so
    the kernel width Algorithm 1 has to reach is bounded by 1/eta^2 instead of
    by however close two random samples happened to fall.

    Dropping points is SOUND, and it is worth being precise about why.
    Definition 1 asks for three things -- closed, inside C_obs, and
    f(q_start)*f(q_goal) < 0. None of them mentions the training set. Training
    points are the means of FINDING a wall; the certificate is about the wall.
    So a wall built from a thinned set is a proof exactly when it satisfies
    (I), (II) and (III), and Algorithm 3 checks (II) against the true obstacle
    region either way. Thinning can cost us a wall; it cannot forge one.

    Both members go, not one. The place where the classes come closest is the
    neck of the obstacle, which is exactly where the wall has to pass. Keeping
    one side there pins the wall against that side and forces the huge gamma;
    keeping neither lets it through the neck smoothly, and Algorithm 3 is
    still the thing that decides whether where it went is inside C_obs.
    """
    if len(seed) == 0 or len(rest) == 0 or eta <= 0.0:
        return seed, rest, 0
    s = seed * scale
    r = rest * scale
    # (n_seed, n_rest) -- these sets are thinned to hundreds, so the dense
    # pairwise distance is cheaper than building a tree.
    d = np.linalg.norm(s[:, None, :] - r[None, :, :], axis=2)
    bad_s = (d < eta).any(axis=1)
    bad_r = (d < eta).any(axis=0)
    n = int(bad_s.sum() + bad_r.sum())
    return seed[~bad_s], rest[~bad_r], n


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------


class GridSource:
    """The five answers read off a PageMap's cells -- the original behaviour."""

    kind = "grid"

    def __init__(self, cfg: Config, pm: PageMap, labels: np.ndarray,
                 regions: list):
        self.cfg, self.pm = cfg, pm
        self.lab, self._regions = labels, regions
        self.signs = pm.signs
        self.name = pm.name

    def regions(self) -> list:
        return self._regions

    def seed_pool(self, k: int) -> np.ndarray:
        idx = np.argwhere(self.lab == k)
        return np.column_stack([self.pm.y_grid[idx[:, 0]],
                                self.pm.theta_grid[idx[:, 1]]])

    def rest_pool(self, k: int, mode: str = "goal_only") -> np.ndarray:
        mask = self.pm.valid_mask & (self.lab != k)
        if mode == "goal_only" and self._regions:
            goal_k = [r.index for r in self._regions if r.reaches_goal]
            if goal_k:
                gm = np.zeros_like(self.lab, dtype=bool)
                for g in goal_k:
                    gm |= (self.lab == g)
                mask = self.pm.valid_mask & gm & (self.lab != k)
        idx = np.argwhere(mask)
        if idx.size == 0:
            return np.zeros((0, 2))
        return np.column_stack([self.pm.y_grid[idx[:, 0]],
                                self.pm.theta_grid[idx[:, 1]]])

    def obstacle_points(self) -> np.ndarray:
        idx = np.argwhere(~self.pm.valid_mask)
        return np.column_stack([self.pm.y_grid[idx[:, 0]],
                                self.pm.theta_grid[idx[:, 1]]])

    def tol(self) -> float:
        return float(np.hypot(self.pm.y_grid[1] - self.pm.y_grid[0],
                              self.pm.theta_grid[1] - self.pm.theta_grid[0]))

    def regions_under(self, y: float, theta: float, radius: float) -> list:
        """Every component under the door's footprint. No adjudicator."""
        i = int(np.argmin(np.abs(self.pm.y_grid - y)))
        j = int(np.argmin(np.abs(self.pm.theta_grid - theta)))
        rad = max(1, int(np.ceil(radius / (self.pm.y_grid[1] - self.pm.y_grid[0]))))
        sub = self.lab[max(0, i - rad): i + rad + 1,
                       max(0, j - rad): j + rad + 1]
        return sorted(set(int(v) for v in np.unique(sub) if v > 0))

    def entry_point(self, k: int) -> tuple | None:
        pts = self.seed_pool(k)
        if not len(pts):
            return None
        return tuple(pts[int(np.argmin(pts[:, 0]))])


# --------------------------------------------------------------------------
# Roadmap
# --------------------------------------------------------------------------


class RoadmapSource:
    """The same five answers from a PRM. No cells anywhere.

    Door anchors are injected as nodes at construction. That is what retires
    `regions_under`'s hole: the anchor is a real vertex of the roadmap, so
    "which component is under the door" stops being a footprint guess over a
    label array and becomes a lookup of the component the anchor actually
    joined. An anchor that connects to nothing becomes its own singleton
    component, which is the conservative answer -- the flood then explores
    from it rather than silently skipping it.
    """

    kind = "roadmap"

    def __init__(self, cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                 signs: tuple, name: str, n_samples: int | None = None,
                 anchors: list | None = None, rng=None,
                 start_yt: tuple | None = None):
        self.cfg, self.arm, self.weld, self.scene = cfg, arm, weld, scene
        self.signs, self.name = signs, name
        wp = cfg.witness
        rng = rng or np.random.default_rng(cfg.planner.rng_seed)
        pm = cfg.pagemap
        n = n_samples or cfg.planner.n_samples
        self.scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])

        pts, inv = [], []
        # Injected first so their indices are stable and they are never
        # thinned away: the start posture, then every door anchor.
        for extra in ([start_yt] if start_yt is not None else []) + list(anchors or []):
            e = np.asarray(extra, dtype=float)
            ok, _ = classify(cfg, arm, weld, scene, float(e[0]), float(e[1]), signs)
            (pts if ok else inv).append(e)
        self.n_injected = len(pts)

        ys = rng.uniform(pm.y_min, pm.y_max, n)
        ths = rng.uniform(pm.theta_min, pm.theta_max, n)
        for y, th in zip(ys, ths):
            ok, _ = classify(cfg, arm, weld, scene, float(y), float(th), signs)
            (pts if ok else inv).append(np.array([y, th]))

        self.valid = np.asarray(pts, dtype=float).reshape(-1, 2)
        self.invalid = np.asarray(inv, dtype=float).reshape(-1, 2)
        self._tree = None
        self._tol = None
        self.comp = self._components()
        self._regions = self._build_regions()

    # -- construction ------------------------------------------------------

    def _components(self) -> np.ndarray:
        """Page-restricted connectivity: k-NN edges validated by local planning."""
        n = self.valid.shape[0]
        if n == 0:
            return np.zeros(0, dtype=int)
        kk = self.cfg.planner.k_neighbors
        sc = self.valid * self.scale
        self._tree = cKDTree(sc)
        uf = UnionFind(n)
        _d, nbr = self._tree.query(sc, k=min(kk + 1, n))
        for i in range(n):
            for j in np.atleast_1d(nbr[i])[1:]:
                j = int(j)
                if uf.find(i) == uf.find(j):
                    continue
                if self._edge(self.valid[i], self.valid[j]):
                    uf.union(i, j)
        roots, comp = {}, np.zeros(n, dtype=int)
        for i in range(n):
            r = uf.find(i)
            roots.setdefault(r, len(roots) + 1)     # 1-based, like ndimage
            comp[i] = roots[r]
        return comp

    def _edge(self, a: np.ndarray, b: np.ndarray) -> bool:
        step = self.cfg.planner.edge_resolution
        dist = float(np.linalg.norm((b - a) * self.scale))
        m = max(2, int(np.ceil(dist / step)))
        for t in np.linspace(0.0, 1.0, m + 1):
            p = a + t * (b - a)
            ok, _ = classify(self.cfg, self.arm, self.weld, self.scene,
                             float(p[0]), float(p[1]), self.signs)
            if not ok:
                return False
        return True

    def _build_regions(self) -> list:
        out = []
        for k in sorted(set(int(c) for c in self.comp)):
            ys = self.valid[self.comp == k][:, 0]
            out.append(Region(
                page=self.name, index=k, n_cells=int((self.comp == k).sum()),
                y_min=float(ys.min()), y_max=float(ys.max()),
                reaches_start=bool(ys.min() <= self.cfg.flood.start_y
                                   + self.cfg.flood.start_y_tol),
                reaches_goal=bool(ys.max() >= self.cfg.flood.goal_y
                                  - self.cfg.flood.goal_y_tol),
            ))
        return out

    # -- the five answers --------------------------------------------------

    def regions(self) -> list:
        return self._regions

    def seed_pool(self, k: int) -> np.ndarray:
        return self.valid[self.comp == k]

    def rest_pool(self, k: int, mode: str = "goal_only") -> np.ndarray:
        if mode == "goal_only" and self._regions:
            goal_k = [r.index for r in self._regions if r.reaches_goal]
            if goal_k:
                m = np.isin(self.comp, goal_k) & (self.comp != k)
                if m.any():
                    return self.valid[m]
        return self.valid[self.comp != k]

    def obstacle_points(self) -> np.ndarray:
        return self.invalid

    def tol(self) -> float:
        """Characteristic sample spacing -- the roadmap's analogue of a cell.

        Median nearest-neighbour distance. Used only to decide which doors are
        near enough to this page to consider, never as a proof quantity.
        """
        if len(self.valid) < 2:
            return float(self.cfg.wall.net_d_min)
        if self._tol is None:
            d, _ = self._tree.query(self.valid * self.scale, k=2)
            self._tol = float(np.median(d[:, 1]))
        return self._tol

    def regions_under(self, y: float, theta: float, radius: float) -> list:
        """Which component the anchor joined. It is a real vertex, so this is
        a lookup, not a guess over a label array."""
        if not len(self.valid):
            return []
        dist, j = self._tree.query(np.array([y, theta]) * self.scale, k=1)
        if float(dist) <= max(radius, self.tol()):
            return [int(self.comp[int(j)])]
        return []

    def entry_point(self, k: int) -> tuple | None:
        pts = self.seed_pool(k)
        if not len(pts):
            return None
        return tuple(pts[int(np.argmin(pts[:, 0]))])


# --------------------------------------------------------------------------
# Labelling Algorithm 4's fed-back points, without a grid
# --------------------------------------------------------------------------


class RoadmapLabeller:
    """Class a fed-back point by the roadmap component it CONNECTS to.

    SDCL's addToTraining -> disjoint-set rule, in the setting it was written
    for. The grid version (`wall.PageLabeller`) consults a label array; here
    the roadmap is the primary object and its components are the classes.

    +1 start side, -1 goal side, 0 when nothing in reach connects, which the
    caller resolves with the nearest-class fallback.
    """

    def __init__(self, cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                 source: RoadmapSource, k: int, n_try: int = 6):
        self.cfg, self.arm, self.weld, self.scene = cfg, arm, weld, scene
        self.src, self.k, self.n_try = source, int(k), int(n_try)
        self.scale = source.scale
        goal_k = [r.index for r in source.regions() if r.reaches_goal]
        self.in_goal = np.isin(source.comp, goal_k)
        self.n_topological = self.n_fallback = self.n_virtual = 0

    def label(self, m) -> int:
        from .wall import in_virtual_invalid
        mm = np.asarray(m, dtype=float).reshape(1, 2)
        if in_virtual_invalid(self.cfg, mm)[0]:
            self.n_virtual += 1
            return -1
        verts = self.src.valid
        if not len(verts):
            self.n_fallback += 1
            return 0
        _d, idx = self.src._tree.query(mm[0] * self.scale,
                                       k=min(self.n_try, len(verts)))
        for j in np.atleast_1d(idx):
            j = int(j)
            if self.src._edge(np.asarray(m, dtype=float), verts[j]):
                self.n_topological += 1
                return -1 if self.in_goal[j] else 1
        self.n_fallback += 1
        return 0


# --------------------------------------------------------------------------


def grid_sources(cfg: Config, pages: dict) -> dict:
    """Wrap a dict of PageMaps as GridSources (the pre-existing behaviour)."""
    from .flood import label_regions
    out = {}
    for name, pm in pages.items():
        lab, regs = label_regions(cfg, pm)
        out[name] = GridSource(cfg, pm, lab, regs)
    return out


def roadmap_sources(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                    pages: dict, sigma: dict | None = None,
                    n_samples: int | None = None,
                    start_yt: tuple | None = None, start_page: str | None = None,
                    verbose: bool = False) -> dict:
    """One RoadmapSource per page, with every door anchor injected as a node."""
    out = {}
    rng = np.random.default_rng(cfg.planner.rng_seed)
    for name, pm in pages.items():
        anchors = [d.anchor for d in (sigma or {}).get(name, [])
                   if d.vanishing]
        src = RoadmapSource(
            cfg, arm, weld, scene, pm.signs, name, n_samples=n_samples,
            anchors=anchors, rng=rng,
            start_yt=start_yt if name == start_page else None)
        out[name] = src
        if verbose:
            print(f"[roadmap] page {name}: {len(src.valid):5d} valid / "
                  f"{len(src.invalid):5d} invalid, {len(src.regions())} "
                  f"component(s), {src.n_injected} injected, "
                  f"spacing {src.tol():.5f}")
    return out


def bootstrap_roadmap(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                      page_names=None, n_samples: int | None = None,
                      start_yt: tuple | None = None,
                      start_page: str | None = None,
                      with_kin: bool = False, verbose: bool = False) -> tuple:
    """Build sources AND Sigma with no PageMap anywhere. Returns (sources, sigma).

    Two passes, because the two depend on each other: doors are found from
    roadmap edges, and door anchors are injected as roadmap vertices so that
    `regions_under` is a lookup rather than a guess.

      pass 1   sample each page, no anchors        -> a roadmap to look for
                                                      sign changes along
      pass 2   rebuild with every anchor injected  -> anchors are real
                                                      vertices in real
                                                      components

    Pass 1's roadmap is used only to FIND doors; pass 2's is what the proof
    runs on. The extra sampling is the price of not having a grid to scan.
    """
    from .arm import SIGN_BRANCHES
    from .doors import build_sigma_roadmap
    from .pagemap import page_name

    names = list(page_names) if page_names is not None else \
        [page_name(s_) for s_ in SIGN_BRANCHES]
    signs_of = {page_name(s_): s_ for s_ in SIGN_BRANCHES}

    rng = np.random.default_rng(cfg.planner.rng_seed)
    scout = {}
    for name in names:
        scout[name] = RoadmapSource(
            cfg, arm, weld, scene, signs_of[name], name, n_samples=n_samples,
            anchors=None, rng=rng,
            start_yt=start_yt if name == start_page else None)
    sigma = build_sigma_roadmap(cfg, arm, weld, scout, with_kin=with_kin,
                                verbose=verbose)

    rng = np.random.default_rng(cfg.planner.rng_seed)
    sources = {}
    for name in names:
        src = RoadmapSource(
            cfg, arm, weld, scene, signs_of[name], name, n_samples=n_samples,
            anchors=[d.anchor for d in sigma.get(name, []) if d.vanishing],
            rng=rng, start_yt=start_yt if name == start_page else None)
        sources[name] = src
        if verbose:
            print(f"[roadmap] page {name}: {len(src.valid):5d} valid / "
                  f"{len(src.invalid):5d} invalid, {len(src.regions())} "
                  f"component(s), {src.n_injected} injected, "
                  f"spacing {src.tol():.5f}")
    return sources, sigma
