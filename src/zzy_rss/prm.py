"""PRM in the augmented coordinate space Y+, as the sample source for Stage 3.

IJRR runs the planner and the proof construction in parallel, and the proof
side consumes the planner's samples: valid points carrying their graph
component (to form the two training classes) and invalid points (as the seeds
that project onto the wall). [IROS]'s planner discards invalid samples, so this
implementation retains them -- that instrumentation is a prerequisite for
IJRR Algorithm 2.

State is (y, theta, s2, s4, s6): continuous coordinates plus the discrete page
label, exactly [IROS] Eq. 7. Edges are only drawn WITHIN a page here; door
crossings are handled by the Stage 4 flood fill, and the per-page training
labels must come from page-restricted connectivity (see wall.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arm import SIGN_BRANCHES, SrsArm
from .params import Config
from .pagemap import page_name
from .scene import Scene
from .weld import WeldLine


@dataclass
class PageSamples:
    """Planner output restricted to one page."""

    signs: tuple
    valid: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    invalid: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    invalid_tag: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    component: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def name(self) -> str:
        return page_name(self.signs)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def classify(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    y: float, theta: float, signs: tuple,
) -> tuple[bool, int]:
    """Validity of one Y+ state, with the invalidity tag on failure."""
    from .arm import INVALID_COLLISION, INVALID_JOINT_LIMIT, INVALID_NO_IK, VALID

    pos, rot = weld.pose(y)
    res = arm.ik_branch(pos, rot, theta, signs,
                        coord_axis_tol=cfg.pagemap.coord_axis_tol,
                        coord_radius_tol=cfg.pagemap.coord_radius_tol)
    if not res.ok:
        return False, INVALID_NO_IK
    if not arm.within_limits(res.q):
        return False, INVALID_JOINT_LIMIT
    if scene.check(arm.frames(res.q)).in_collision:
        return False, INVALID_COLLISION
    return True, VALID


def build_prm(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, verbose: bool = True,
) -> dict:
    """Sample Y+ uniformly per page, connect within pages, label components."""
    pp = cfg.planner
    pm = cfg.pagemap
    rng = np.random.default_rng(pp.rng_seed)
    out = {}

    for signs in SIGN_BRANCHES:
        name = page_name(signs)
        ys = rng.uniform(pm.y_min, pm.y_max, pp.n_samples)
        ths = rng.uniform(pm.theta_min, pm.theta_max, pp.n_samples)

        valid_pts, invalid_pts, invalid_tags = [], [], []
        for y, th in zip(ys, ths):
            ok, tag = classify(cfg, arm, weld, scene, float(y), float(th), signs)
            if ok:
                valid_pts.append((y, th))
            else:
                invalid_pts.append((y, th))
                invalid_tags.append(tag)

        valid = np.asarray(valid_pts, dtype=float).reshape(-1, 2)
        invalid = np.asarray(invalid_pts, dtype=float).reshape(-1, 2)
        comp = _components(cfg, arm, weld, scene, valid, signs)
        out[name] = PageSamples(
            signs=signs, valid=valid, invalid=invalid,
            invalid_tag=np.asarray(invalid_tags, dtype=np.int8), component=comp,
        )
        if verbose:
            n_comp = len(set(comp.tolist())) if comp.size else 0
            print(f"[prm]   page {name}  valid {valid.shape[0]:5d}  "
                  f"invalid {invalid.shape[0]:5d}  components {n_comp}")
    return out


def _components(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    valid: np.ndarray, signs: tuple,
) -> np.ndarray:
    """Page-restricted connectivity: k-NN edges validated by local planning."""
    n = valid.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    pp = cfg.planner
    scale = np.array([1.0, 1.0 / np.pi])
    scaled = valid * scale
    uf = UnionFind(n)

    for i in range(n):
        d = np.linalg.norm(scaled - scaled[i], axis=1)
        order = np.argsort(d)[1 : pp.k_neighbors + 1]
        for j in order:
            if uf.find(i) == uf.find(int(j)):
                continue
            if _edge_valid(cfg, arm, weld, scene, valid[i], valid[j], signs):
                uf.union(i, int(j))

    roots = {}
    comp = np.zeros(n, dtype=int)
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        comp[i] = roots[r]
    return comp


def _edge_valid(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    a: np.ndarray, b: np.ndarray, signs: tuple,
) -> bool:
    """Straight-line local plan in page coordinates, resolution-checked."""
    step = cfg.planner.edge_resolution
    dist = float(np.linalg.norm((b - a) * np.array([1.0, 1.0 / np.pi])))
    n_steps = max(2, int(np.ceil(dist / step)))
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        p = a + t * (b - a)
        ok, _ = classify(cfg, arm, weld, scene, float(p[0]), float(p[1]), signs)
        if not ok:
            return False
    return True
