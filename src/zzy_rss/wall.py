"""Stage 3 -- the RBF-SVM wall, following IJRR Algorithms 1-3 step for step.

Mapping of IJRR objects onto one page of Y+:

  IJRR                               here
  ---------------------------------  -----------------------------------------
  state space C                      one page, coords (y, theta), d = 2
  C_free / C_obs                     Y+_valid / Y+_invalid, invalidity tagged
  joint limits as virtual obstacles   page-boundary bands (Fig. 5 trick)
  P_goal / P_rest                    seed-connected vs not, PAGE-RESTRICTED
  learned manifold f(q) = 0          wall f(y, theta) = 0, RBF-SVM
  P_obs                              invalid Y+ samples on this page
  polytope (the actual proof)         closed polyline (d=2) or triangle mesh (d=3)
  facet                              segment (d=2) or triangle (d=3)
  CPD (Eq. 3)                        Y+ penetration depth, dispatched by tag

Two IJRR structural choices are preserved deliberately:

  * Both training classes are VALID points (IJRR 4.1). This learns a surface
    that lives INSIDE the invalid region, not a valid/invalid boundary.
  * Reconstruction is what VALIDATES closedness (IJRR 4.3). It is the
    tangential Delaunay complex of an (epsilon, eta)-net of points ON the wall
    (tangential.py), built without any grid and in any dimension: a closed
    polyline at d = 2, a closed oriented triangle mesh at d = 3. The old grid
    contour tracer survives only as WallParams.reconstruction = "contour".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

from .arm import INVALID_COLLISION, INVALID_JOINT_LIMIT, INVALID_NO_IK, SrsArm
from . import budget as _budget
from .facets import FacetOracle, FacetResult, check_simplex  # noqa: F401
from .params import Config
from .prm import classify
from .scene import Scene
from .weld import WeldLine

PD_IJRR = "ijrr_local_opt"     # strict IJRR Eq. 3, lifted to Y+
PD_LIPSCHITZ = "lipschitz"     # hardened: exact workspace PD + Lipschitz transfer

LABEL_TOPOLOGICAL = "topological"   # SDCL rule: connectivity decides the class
LABEL_METRIC = "metric"             # the nearest-class heuristic it replaces


# --------------------------------------------------------------------------
# IJRR 4.1 -- Algorithm 1, Train-Manifold
# --------------------------------------------------------------------------


@dataclass
class WallModel:
    """A trained RBF-SVM wall in scaled page coordinates."""

    svm: SVC
    gamma: float
    scale: np.ndarray
    n_seed: int
    n_rest: int
    accuracy: float
    gamma_capped: bool = False
    projection: str = "newton"
    newton_iters: int = 30
    newton_tol: float = 1.0e-10

    def __post_init__(self):
        # Cache the RBF expansion. sklearn's decision_function carries heavy
        # per-call validation, and SLSQP calls it ~180 times per projection;
        # with 900 projections per Algorithm 2 pass that was 57 s of the 58 s
        # an Algorithm 4 iteration took, for 9 support vectors' worth of
        # arithmetic. Evaluating the sum directly in numpy removes it.
        object.__setattr__(self, "_sv", np.asarray(self.svm.support_vectors_,
                                                   dtype=float))
        object.__setattr__(self, "_dual", np.asarray(self.svm.dual_coef_[0],
                                                     dtype=float))
        object.__setattr__(self, "_b", float(self.svm.intercept_[0]))
        object.__setattr__(self, "_gamma", float(self.svm._gamma))

    def decision(self, pts: np.ndarray) -> np.ndarray:
        """Same value as svm.decision_function, computed directly."""
        x = np.atleast_2d(np.asarray(pts, dtype=float)) * self.scale
        d2 = np.sum((x[:, None, :] - self._sv[None, :, :]) ** 2, axis=2)
        return np.exp(-self._gamma * d2) @ self._dual + self._b

    def decision_grad(self, pt: np.ndarray) -> np.ndarray:
        """Gradient of `decision` at one point, in UNSCALED coordinates.

        Giving SLSQP the analytic constraint gradient removes its numerical
        differentiation, which was two extra evaluations per iteration.
        """
        x = np.asarray(pt, dtype=float) * self.scale
        diff = x[None, :] - self._sv
        w = np.exp(-self._gamma * np.sum(diff ** 2, axis=1)) * self._dual
        g = (-2.0 * self._gamma) * (w[:, None] * diff).sum(axis=0)
        return g * self.scale

    def field_scaled(self, x: np.ndarray, chunk: int = 2048) -> tuple:
        """(f, grad f) for many points given in SCALED coordinates.

        This is the wall as an implicit hypersurface in the SVM's own input
        space, which is where the tangential complex is built: one distance
        means the same thing along every axis there. Batched, any dimension.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        f = np.empty(len(x))
        g = np.empty_like(x)
        for s in range(0, len(x), chunk):
            xs = x[s:s + chunk]
            d2 = np.sum((xs[:, None, :] - self._sv[None, :, :]) ** 2, axis=2)
            e = np.exp(-self._gamma * d2) * self._dual          # (n, m)
            f[s:s + chunk] = e.sum(axis=1) + self._b
            g[s:s + chunk] = (-2.0 * self._gamma) * (
                xs * e.sum(axis=1)[:, None] - e @ self._sv)
        return f, g


def train_wall(cfg: Config, seed_pts: np.ndarray, rest_pts: np.ndarray,
               scale: np.ndarray | None = None,
               gamma_start: float | None = None) -> WallModel:
    """IJRR Algorithm 1: escalate gamma until the training data is fully separated.

    The loop deliberately over-fits -- complete separation is the requirement,
    not a pathology. It stops at the LARGEST effecting range (smallest gamma)
    that still separates, which is what keeps the zero set connected instead of
    fragmenting into Gaussian islands (IJRR Fig. 7).
    """
    wp = cfg.wall
    # The page's own axis scaling: (y, theta) by default, (u, v, theta) for a
    # surface task -- any dimension, as long as the points match it.
    scale = (np.array([wp.scale_y, wp.scale_theta]) if scale is None
             else np.asarray(scale, dtype=float))
    x = np.vstack([seed_pts, rest_pts]) * scale
    y = np.concatenate([np.ones(len(seed_pts)), -np.ones(len(rest_pts))])

    # gamma_start (opt-in, findings 42): a RETRAIN after a few points were
    # added may start the scan near the previous gamma instead of at
    # gamma_init. It can only miss a smaller separating gamma below the start,
    # i.e. cost kernel width -- never soundness, which rests on closedness and
    # Algorithm 3. The d = 2 pipeline never passes it.
    gamma = wp.gamma_init if gamma_start is None else max(wp.gamma_init, float(gamma_start))
    svm = None
    acc = 0.0
    capped = False

    def fit(g):
        _budget.check_gamma(g)              # no-op without an active budget
        s = SVC(kernel="rbf", gamma=g, C=wp.svm_C)
        s.fit(x, y)
        return s, float(s.score(x, y))

    if getattr(wp, "gamma_search", "linear") == "gallop":
        svm, acc = fit(gamma)
        lo = None                            # largest gamma known NOT to separate
        while acc < 1.0:
            if gamma >= wp.gamma_max:
                capped = True
                break
            lo = gamma
            gamma = min(wp.gamma_max, gamma * float(wp.gamma_gallop))
            svm, acc = fit(gamma)
        if not capped and lo is not None:
            while gamma - lo > wp.gamma_step:
                mid = 0.5 * (lo + gamma)
                s_mid, a_mid = fit(mid)
                if a_mid >= 1.0:
                    gamma, svm, acc = mid, s_mid, a_mid
                else:
                    lo = mid
    while getattr(wp, "gamma_search", "linear") != "gallop":
        _budget.check_gamma(gamma)          # no-op without an active budget
        svm = SVC(kernel="rbf", gamma=gamma, C=wp.svm_C)
        svm.fit(x, y)
        acc = float(svm.score(x, y))
        if acc >= 1.0:
            break
        if gamma + wp.gamma_step > wp.gamma_max:
            capped = True
            break
        gamma += wp.gamma_step
    if capped:
        # Accuracy < 1 means the wall does not separate its own training data,
        # so it is not a separator at all. Say so loudly; Algorithm 3 will
        # then report the facets it costs.
        print(f"[stage1] WARNING gamma hit gamma_max={wp.gamma_max:g} with "
              f"accuracy {acc:.4f} < 1: the wall does not separate its own "
              f"training data (findings 29)")
    elif gamma > getattr(wp, "gamma_warn", float("inf")):
        # Separated, but with a kernel narrow enough that IJRR Fig. 7 would
        # worry about the zero set breaking into islands. Closedness and the
        # facet checks are what actually decide, so this is a flag, not a
        # refusal.
        print(f"[stage1] note gamma={gamma:.1f} exceeds gamma_warn="
              f"{wp.gamma_warn:g} (accuracy {acc:.4f}); relying on closedness "
              f"and facet checks")
    return WallModel(svm=svm, gamma=gamma, scale=scale, gamma_capped=capped,
                     n_seed=len(seed_pts), n_rest=len(rest_pts), accuracy=acc,
                     projection=wp.projection, newton_iters=wp.newton_iters,
                     newton_tol=wp.newton_tol)


# --------------------------------------------------------------------------
# IJRR 4.1 -- the Fig. 5 virtual-boundary trick
# --------------------------------------------------------------------------


def sigma_disc_samples(discs: list, region_pts: np.ndarray, tol: float,
                       n_ring: int = 12) -> tuple:
    """Training points that stop the wall routing THROUGH a door.

    The SVM separates two classes of valid points and knows nothing about
    Sigma, so its zero set happily runs straight through a door disc -- in a
    first run on the real arm, 18 of 182 facets failed for exactly this. A
    facet may not cross a disc (a path would hop pages inside it and bypass
    the wall), and shrinking facets cannot fix that, because the wall itself
    is in the wrong place.

    A door belonging to this region must end up INSIDE the wall, since the
    flood fill follows it from here. So each such disc contributes its centre
    and a ring of points to the SEED class; discs belonging elsewhere go to
    the REST class. Returns (seed_extra, rest_extra).
    """
    seed_extra, rest_extra = [], []
    for d in discs:
        c = np.array([d.y, d.theta])
        pts = [c]
        for a in np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False):
            pts.append(c + d.radius * np.array([np.cos(a), np.sin(a)]))
        near = (len(region_pts) > 0
                and float(np.min(np.linalg.norm(region_pts - c, axis=1)))
                <= max(tol, d.radius))
        (seed_extra if near else rest_extra).extend(pts)
    return (np.asarray(seed_extra, dtype=float).reshape(-1, 2),
            np.asarray(rest_extra, dtype=float).reshape(-1, 2))


def virtual_boundary_samples(cfg: Config, n_per_edge: int = 120) -> np.ndarray:
    """Virtual-VALID ring outside the page, sampled as training data.

    IJRR treats configuration-space boundaries as a virtual obstacle region of
    positive thickness with a virtual free region enclosing it, and samples
    both. The outer ring goes into the class NOT holding the seed, which forces
    the wall to close through the virtual invalid band instead of running off
    the page.

    ONLY IN y. theta is an arm angle, so the page is a cylinder and has no
    theta boundary to close against; putting a ring there asks the wall to
    close across a seam that is not an edge. The y ends are real -- the weld
    genuinely starts and stops.
    """
    pm = cfg.pagemap
    eps = cfg.wall.virtual_band
    ring = getattr(cfg.wall, "virtual_ring", 2.0) * eps
    y0, y1 = pm.y_min - ring, pm.y_max + ring
    ts = np.linspace(pm.theta_min, pm.theta_max, n_per_edge)
    if not getattr(pm, "theta_periodic", True):
        t0, t1 = pm.theta_min - ring, pm.theta_max + ring
        ys = np.linspace(y0, y1, n_per_edge)
        return np.vstack([
            np.column_stack([ys, np.full_like(ys, t0)]),
            np.column_stack([ys, np.full_like(ys, t1)]),
            np.column_stack([np.full_like(ts, y0), ts]),
            np.column_stack([np.full_like(ts, y1), ts]),
        ])
    return np.vstack([
        np.column_stack([np.full_like(ts, y0), ts]),
        np.column_stack([np.full_like(ts, y1), ts]),
    ])


def edge_anchor_samples(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    n_scan: int = 181, edges: tuple | None = None,
) -> np.ndarray:
    """Exact VALID configurations ON the page's y edges, for the TRAINING set.

    The Fig. 5 ring above tells the wall that everything beyond the page is
    the OTHER class. It does not tell the wall where the seed region STOPS.
    Those are different facts, and only the first one is supplied by the ring:
    if no training point of the seed class sits at y = y_min, nothing stops the
    SVM from closing the wall INSIDE the page and slicing the valid start
    region off at its own edge.

    The grid pipeline never notices, because the y = 0 row of the grid is in
    the training data for free. A random roadmap practically never samples
    y = 0, which is exactly why every failed facet of the sampled pipeline's
    EXP 2 wall sat at y ~ 0 (findings 25): the wall had no idea where the left
    edge of the start region was, and cut into it.

    This returns the missing anchors: every arm angle at y = y_min and
    y = y_max whose configuration is VALID. They are scanned exactly rather
    than sampled, since the edge is a 1-D line and 181 samples cover it.

    These belong in the TRAINING data ONLY. Putting them in the ROADMAP is a
    different change and measured worse -- it moves graph vertices onto the
    boundary band itself, and the wall can then no longer close against the
    band (111/122 closed became 132/152 open; see invariant 18).
    """
    pm = cfg.pagemap
    if edges is None:
        edges = (pm.y_min, pm.y_max)
    out = []
    thetas = np.linspace(pm.theta_min, pm.theta_max, n_scan)
    for yv in edges:
        for t in thetas:
            ok, _tag = classify(cfg, arm, weld, scene, float(yv), float(t), signs)
            if ok:
                out.append((float(yv), float(t)))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def in_virtual_invalid(cfg: Config, pts: np.ndarray) -> np.ndarray:
    """The positive-thickness invalid band just outside the page boundary.

    It reaches out to virtual_band_outer * virtual_band, which is where the
    virtual free ring begins -- see WallParams.virtual_band_outer for why the
    two must be adjacent.
    """
    pm = cfg.pagemap
    eps = cfg.wall.virtual_band * getattr(cfg.wall, "virtual_band_outer", 1.0)
    y, t = pts[:, 0], pts[:, 1]
    if getattr(pm, "theta_periodic", True):
        outside = (y < pm.y_min) | (y > pm.y_max)
        far = (y < pm.y_min - eps) | (y > pm.y_max + eps)
        return outside & ~far
    outside = (y < pm.y_min) | (y > pm.y_max) | (t < pm.theta_min) | (t > pm.theta_max)
    far = ((y < pm.y_min - eps) | (y > pm.y_max + eps)
           | (t < pm.theta_min - eps) | (t > pm.theta_max + eps))
    return outside & ~far


# --------------------------------------------------------------------------
# IJRR 4.2 -- Algorithm 2, Sample-Manifold
# --------------------------------------------------------------------------


def project_to_wall(wall: WallModel, start: np.ndarray) -> np.ndarray | None:
    """IJRR Eq. 2: nearest point on the wall.

    Two modes, switchable by WallParams.projection.

      [proj:newton]  (default) Newton on the constraint using the analytic
                     gradient: z <- z - f(z) * grad / |grad|^2, started at the
                     seed. Because each step moves along the steepest
                     direction, the fixed point is the nearest wall point to
                     the seed, which is what Eq. 2 asks for.
      [proj:slsqp]   the general constrained solve. Same answer, but scipy's
                     per-call overhead is ~8 ms, and Algorithm 2 does ~900
                     projections per pass, so it dominated the whole pipeline.
    """
    if wall.scale is None:
        return None
    if getattr(wall, "projection", "newton") == "slsqp":
        return _project_slsqp(wall, start)

    z = np.asarray(start, dtype=float).copy()
    for _ in range(wall.newton_iters):
        f = float(wall.decision(z)[0])
        if abs(f) < wall.newton_tol:
            break
        g = wall.decision_grad(z)
        gg = float(np.dot(g, g))
        if not np.isfinite(gg) or gg < 1e-18:
            return None
        z = z - (f / gg) * g
        if not np.all(np.isfinite(z)):
            return None
    if abs(float(wall.decision(z)[0])) > 1e-5:
        return None
    return z


def _project_slsqp(wall: WallModel, start: np.ndarray) -> np.ndarray | None:
    def obj(z):
        return float(np.sum((z - start) ** 2))

    def grad(z):
        return 2.0 * (z - start)

    def con(z):
        if not np.all(np.isfinite(z)):
            return 1.0e3
        return float(wall.decision(np.asarray(z, dtype=float))[0])

    def con_jac(z):
        if not np.all(np.isfinite(z)):
            return np.zeros_like(z)
        return wall.decision_grad(z)

    res = minimize(obj, start.copy(), jac=grad,
                   constraints=[{"type": "eq", "fun": con, "jac": con_jac}],
                   method="SLSQP", options={"maxiter": 60, "ftol": 1e-10})
    if not np.all(np.isfinite(res.x)):
        return None
    if not res.success or abs(float(wall.decision(res.x)[0])) > 1e-5:
        return None
    return np.asarray(res.x, dtype=float)


# --------------------------------------------------------------------------
# Label assignment for fed-back points -- SDCL's rule, transplanted
# --------------------------------------------------------------------------


class PageLabeller:
    """Class a fed-back point by CONNECTIVITY to the start region, not distance.

    IJRR's released implementation (dyalab/SDCL, makeTrainingDataFromGraph)
    never measures a distance to assign a label. A manifold point that turns
    out valid is inserted into the roadmap by addToTraining, and on the next
    round its class is read off the disjoint set: goal component -> -1,
    otherwise +1. The label is a topological fact about the roadmap.

    A page carries the same structure already -- label_regions gives the
    connected components of the valid mask -- so the rule transplants
    directly: a point joins the start class iff it is CONNECTED through valid
    page space to the start region.

    What this replaces is "whichever class has a nearer training point wins",
    which is wrong exactly where it matters. Near the wall the two classes
    interleave, so the nearest sample is a poor proxy for which side a point
    is on, and a mislabelled point drags the retrained wall the wrong way.

    `_connects` is the si_->checkMotion analogue: a straight page segment all
    of whose samples must be valid. theta is periodic, so the segment takes
    the short way around the cylinder.
    """

    def __init__(self, cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                 signs: tuple, pm, lab: np.ndarray, k: int,
                 search_cells: int = 3, motion_samples: int = 8):
        self.cfg, self.arm, self.weld, self.scene = cfg, arm, weld, scene
        self.signs = signs
        self.y_grid = np.asarray(pm.y_grid, dtype=float)
        self.theta_grid = np.asarray(pm.theta_grid, dtype=float)
        self.lab = lab
        self.k = int(k)
        self.periodic = bool(getattr(cfg.pagemap, "theta_periodic", True))
        self.search_cells = int(search_cells)
        self.motion_samples = int(motion_samples)
        self.scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
        self.n_topological = 0
        self.n_fallback = 0
        self.n_virtual = 0

    # -- geometry on a cylinder -------------------------------------------

    def _dtheta(self, a: float, b: float) -> float:
        d = b - a
        if self.periodic:
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
        return float(d)

    def _cell_centre(self, i: int, j: int) -> np.ndarray:
        return np.array([self.y_grid[i], self.theta_grid[j]])

    def _connects(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Every sample strictly between a and b is valid (checkMotion)."""
        dy = float(b[0] - a[0])
        dt = self._dtheta(float(a[1]), float(b[1]))
        for t in np.linspace(0.0, 1.0, self.motion_samples + 1)[1:]:
            y = float(a[0]) + t * dy
            th = float(a[1]) + t * dt
            if self.periodic:
                th = (th + np.pi) % (2.0 * np.pi) - np.pi
            ok, _tag = classify(self.cfg, self.arm, self.weld, self.scene,
                                y, th, self.signs)
            if not ok:
                return False
        return True

    # -- the rule ----------------------------------------------------------

    def label(self, m: np.ndarray) -> int:
        """+1 start class, -1 rest, 0 undecided (caller falls back)."""
        # The virtual band is "outside" BY CONSTRUCTION (IJRR Fig. 5): it is a
        # fictitious obstacle that lets a wall close against the page edge, not
        # a part of the page. Connectivity says nothing about it -- a band
        # point at y < y_min sits on a valid continuation of the weld and so
        # reaches the start region, which would label it +1 and destroy the
        # very boundary it exists to provide. Measured: every disagreement
        # between this rule and the metric one on EXP 2 was a band point
        # (y in [-0.050, -0.030]), and labelling them +1 drove gamma from
        # 14.8 to 192.4. See findings 21.
        mm = np.asarray(m, dtype=float).reshape(1, 2)
        if in_virtual_invalid(self.cfg, mm)[0]:
            self.n_virtual += 1
            return -1
        ny, nt = self.lab.shape
        i0 = int(np.argmin(np.abs(self.y_grid - float(m[0]))))
        j0 = int(np.argmin(np.abs(
            [self._dtheta(float(m[1]), t) for t in self.theta_grid])))

        cand = []
        r = self.search_cells
        for di in range(-r, r + 1):
            i = i0 + di
            if not 0 <= i < ny:
                continue
            for dj in range(-r, r + 1):
                j = j0 + dj
                if self.periodic:
                    j %= nt
                elif not 0 <= j < nt:
                    continue
                if self.lab[i, j] <= 0:
                    continue
                c = self._cell_centre(i, j)
                d = float(np.hypot((c[0] - m[0]) * self.scale[0],
                                   self._dtheta(float(m[1]), float(c[1]))
                                   * self.scale[1]))
                cand.append((d, i, j))
        cand.sort()

        for _d, i, j in cand:
            if self._connects(m, self._cell_centre(i, j)):
                self.n_topological += 1
                return 1 if int(self.lab[i, j]) == self.k else -1

        self.n_fallback += 1
        return 0


def assign_label(m: np.ndarray, seed_pts: np.ndarray, rest_pts: np.ndarray,
                 scale: np.ndarray, labeller=None) -> int:
    """+1 seed / -1 rest: connectivity when available, nearest class otherwise."""
    if labeller is not None:
        lbl = labeller.label(m)
        if lbl != 0:
            return lbl
    d_seed = (float(np.min(np.linalg.norm((seed_pts - m) * scale, axis=1)))
              if len(seed_pts) else np.inf)
    d_rest = (float(np.min(np.linalg.norm((rest_pts - m) * scale, axis=1)))
              if len(rest_pts) else np.inf)
    return 1 if d_seed <= d_rest else -1


def sample_wall(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    wall: WallModel, invalid_pts: np.ndarray,
    seed_pts: np.ndarray, rest_pts: np.ndarray,
    max_rounds: int = 8, verbose: bool = True, labeller=None,
    valid_fn=None, gamma_warm: float | None = None, valid_many=None,
) -> tuple[WallModel, np.ndarray, np.ndarray, np.ndarray, int]:
    """IJRR Algorithm 2: project invalid seeds onto the wall and retrain if any
    projection lands in the valid region.

    A wall point found VALID means the wall is not inside the invalid region;
    it is added back into the training data (which is what pushes the wall in)
    and the SVM is retrained.
    """
    dim = int(np.asarray(invalid_pts).reshape(len(invalid_pts), -1).shape[1]) \
        if len(invalid_pts) else len(wall.scale)
    if valid_fn is None:
        def valid_fn(m):
            return classify(cfg, arm, weld, scene, float(m[0]), float(m[1]), signs)[0]
        if valid_many is None and cfg.planner.batch_classify and dim == 2:
            from .batch import classify_line_batch

            def valid_many(M):
                return classify_line_batch(cfg, arm, weld, scene, M, signs)[0]
    if valid_many is None:
        def valid_many(M):
            return np.array([bool(valid_fn(m)) for m in M], dtype=bool)
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        _budget.check_time()
        proj = [project_to_wall(wall, q) for q in invalid_pts]
        proj = np.asarray([m for m in proj if m is not None], dtype=float).reshape(-1, dim)
        okp = valid_many(proj) if len(proj) else np.zeros(0, dtype=bool)
        free_hits = list(proj[okp])
        wall_pts = proj[~okp]

        if not free_hits:
            if verbose:
                print(f"[stage3] alg2 converged after {rounds} round(s): "
                      f"{len(wall_pts)} wall points, all invalid")
            return wall, wall_pts, seed_pts, rest_pts, rounds

        # add-point (SDCL addToTraining): the point joins the class it is
        # CONNECTED to, with nearest-class as the fallback when nothing in
        # reach connects.
        #
        # These points sit ON the wall, i.e. in the corridor between the two
        # classes, so they are the ones most likely to contradict the class
        # they are not given -- and this loop runs up to max_rounds times per
        # Algorithm 4 iteration, so an inconsistency here compounds fastest.
        # A point closer than feedback_margin to the opposite class is dropped
        # rather than allowed to make the training data inseparable below the
        # fragmentation limit (findings 29).
        free_hits = np.asarray(free_hits, dtype=float)
        lb = labeller if cfg.wall.label_mode == LABEL_TOPOLOGICAL else None
        margin = float(cfg.wall.feedback_margin)
        n_added = n_skipped = 0
        for m in free_hits:
            lab_m = assign_label(m, seed_pts, rest_pts, wall.scale, lb)
            other = rest_pts if lab_m > 0 else seed_pts
            if len(other):
                d_other = float(np.min(np.linalg.norm(
                    (other - m) * wall.scale, axis=1)))
                if d_other < margin:
                    n_skipped += 1
                    continue
            if lab_m > 0:
                seed_pts = np.vstack([seed_pts, m])
            else:
                rest_pts = np.vstack([rest_pts, m])
            n_added += 1
        if not n_added:
            # Nothing can be learned from this round: every free hit would
            # contradict the data. Stop and let Algorithm 3 report the facets
            # this costs, instead of retraining into a fragmented wall.
            if verbose:
                print(f"[stage3] alg2 round {rounds}: all {n_skipped} free "
                      f"hit(s) contradict the training data -> stop "
                      f"retraining (gamma {wall.gamma:.2f})")
            return wall, wall_pts, seed_pts, rest_pts, rounds
        if verbose:
            print(f"[stage3] alg2 round {rounds}: {n_added} wall points landed "
                  f"VALID -> retrain (gamma was {wall.gamma:.2f})"
                  + (f", skipped {n_skipped} within the {margin:g} margin"
                     if n_skipped else ""))
        wall = train_wall(cfg, seed_pts, rest_pts, scale=wall.scale,
                          gamma_start=(None if gamma_warm is None
                                       else gamma_warm * wall.gamma))
    return wall, np.zeros((0, dim)), seed_pts, rest_pts, rounds


# --------------------------------------------------------------------------
# IJRR 4.3 -- reconstruction, which also validates closedness
# --------------------------------------------------------------------------


@dataclass
class Polytope:
    """The actual proof object.

    d = 2: closed polyline(s); `loops` holds each as an ordered (n_i, 2) array
    with its first vertex repeated, and its segments are the facets.
    d = 3 (tdc only): `vertices` + `simplices` form a closed, coherently
    oriented triangle mesh; the triangles are the facets and `loops` is empty.
    Coordinates are UNSCALED page coordinates in both cases.
    """

    loops: list = field(default_factory=list)   # list of (n_i, 2) arrays
    closed: bool = False
    n_segments: int = 0
    note: str = ""
    vertices: np.ndarray | None = None          # (n, d), tdc only
    simplices: np.ndarray | None = None         # (m, d) facet vertex indices
    method: str = "contour"
    stats: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        if self.vertices is not None:
            return int(self.vertices.shape[1])
        return 2


def reconstruct_wall(cfg: Config, wall: WallModel, grid_res: int = 281,
                     wall_pts: np.ndarray | None = None,
                     seed_pts: np.ndarray | None = None,
                     rest_pts: np.ndarray | None = None,
                     refine: int = 0) -> Polytope:
    """IJRR 4.3: turn the learned wall into a closed polytope, or fail.

    Dispatches on WallParams.reconstruction. "tdc" (default) builds the
    tangential Delaunay complex from points on the wall and never touches a
    grid; `grid_res` is ignored there and `refine` halves the net spacing
    that many times instead. "contour" is the legacy grid tracer.
    """
    if getattr(cfg.wall, "reconstruction", "contour") == "tdc":
        return reconstruct_wall_tdc(cfg, wall, wall_pts=wall_pts,
                                    seed_pts=seed_pts, rest_pts=rest_pts,
                                    refine=refine)
    return reconstruct_wall_contour(cfg, wall, grid_res=grid_res)


def reconstruct_wall_tdc(cfg: Config, wall: WallModel,
                         wall_pts: np.ndarray | None = None,
                         seed_pts: np.ndarray | None = None,
                         rest_pts: np.ndarray | None = None,
                         refine: int = 0, rng=None) -> Polytope:
    """The tangential Delaunay complex of the wall (IJRR 4.3), no grid.

    Samples: Algorithm 2's wall points, the projections of the training data,
    and the projections of uniform random points in the reconstruction window
    -- the window reaches through the virtual band, where Algorithm 2 has no
    invalid samples to project. From those seeds an (epsilon, eta)-net of the
    wall is grown along its tangent space (IJRR Definition 2), and the TDC of
    the net is the polytope.

    The window is the page widened by recon_margin * virtual_band on every
    side that has a virtual ring. With a periodic theta the window would need
    a periodic distance; the project fixes a bounded theta range instead, so
    a periodic page is refused here (closed=False -> INCONCLUSIVE) rather than
    reconstructed on a rectangle that is not the page.

    The closure test is the TDC's: every vertex in exactly two edges (d = 2),
    every edge in exactly two triangles with single-cycle vertex links and a
    coherent orientation (d = 3). On top of it, every training point further
    than tdc_side_tol * eta from the wall must lie on the side of the
    polytope that the sign of the SVM gives it; a missed piece of wall that
    separates the two classes fails here instead of reaching Algorithm 3.
    """
    from .tangential import reconstruct

    wp = cfg.wall
    pm = cfg.pagemap
    scale = np.asarray(wall.scale, dtype=float)
    d = len(scale)
    if getattr(pm, "theta_periodic", False):
        return Polytope(closed=False, method="tdc",
                        note="tdc: periodic theta is not supported; use a "
                             "bounded theta range (findings 39)")
    rng = rng or np.random.default_rng(cfg.planner.rng_seed)
    m = wp.recon_margin * wp.virtual_band
    lo = np.array([pm.y_min - m, pm.theta_min - m])[:d]
    hi = np.array([pm.y_max + m, pm.theta_max + m])[:d]

    seeds = [rng.uniform(lo, hi, size=(int(wp.tdc_seed_points), d))]
    for extra in (wall_pts, seed_pts, rest_pts):
        if extra is not None and len(extra):
            seeds.append(np.asarray(extra, dtype=float).reshape(-1, d))
    seeds = np.vstack(seeds) * scale
    side = [np.asarray(p, dtype=float).reshape(-1, d)
            for p in (seed_pts, rest_pts) if p is not None and len(p)]
    side = np.vstack(side) * scale if side else None

    eta = float(wp.net_d_min * wp.tdc_eta_factor) * (0.5 ** int(refine))
    rec = reconstruct(
        wall.field_scaled, seeds, eta, lo * scale, hi * scale,
        side_points=side, step=wp.tdc_step, accept=wp.tdc_accept,
        k_neighbors=wp.tdc_k_neighbors, fix_rounds=wp.tdc_fix_rounds,
        weight_frac=wp.tdc_weight_frac, side_tol=wp.tdc_side_tol,
        max_points=wp.tdc_max_points, rng=rng,
    )
    verts = rec.points / scale
    stats = dict(rec.stats, eta=eta)
    if d == 2:
        loops = [verts[c] for c in rec.loops] if rec.closed else []
        n_seg = sum(len(v) - 1 for v in loops)
    else:
        loops = []
        n_seg = int(len(rec.simplices)) if rec.closed else 0
    return Polytope(
        loops=loops, closed=bool(rec.closed and n_seg > 0), n_segments=n_seg,
        note=("tdc: " + rec.note) if rec.note else "",
        vertices=verts, simplices=rec.simplices, method="tdc", stats=stats,
    )


def reconstruct_wall_contour(cfg: Config, wall: WallModel,
                             grid_res: int = 281) -> Polytope:
    """LEGACY (grid). Trace the wall's zero level set into closed polylines.

    At d = 2 the manifold is 1-D, so IJRR's tangential Delaunay complex
    degenerates to curve reconstruction; contour tracing gives the ordered
    polyline directly. The closedness test is retained: a loop whose endpoints
    do not meet fails Definition 1 (I) and triggers a retrain, exactly as a
    failed triangulation does in IJRR.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = cfg.pagemap
    eps = cfg.wall.virtual_band
    # The traced domain must be STRICTLY WIDER than the virtual-valid ring at
    # 2*eps. With both at 2*eps the zero set ran along the domain edge and the
    # contour came back as an open arc with both endpoints at y = -0.060, so
    # the closure test failed and Definition 1 (I) was never established.
    m = cfg.wall.recon_margin * eps
    ys = np.linspace(pm.y_min - m, pm.y_max + m, grid_res)
    if getattr(pm, "theta_periodic", True):
        # Tile theta by one period either side. A wall that wraps the cylinder
        # is not a closed curve on the [-pi, pi] rectangle, so tracing it flat
        # returns an open arc ending on the theta edge; on the tiled grid it
        # traces as one continuous curve.
        span = pm.theta_max - pm.theta_min
        ts = np.linspace(pm.theta_min - span, pm.theta_max + span, 3 * grid_res // 2)
    else:
        ts = np.linspace(pm.theta_min - m, pm.theta_max + m, grid_res)
    yy, tt = np.meshgrid(ys, ts, indexing="ij")
    pts = np.column_stack([yy.ravel(), tt.ravel()])
    vals = wall.decision(pts).reshape(yy.shape)

    fig = plt.figure()
    cs = plt.contour(yy, tt, vals, levels=[cfg.wall.contour_level])
    loops, closed_all = [], True
    for path in cs.get_paths():
        v = path.vertices
        if len(v) < 4:
            continue
        gap = float(np.linalg.norm((v[0] - v[-1]) * wall.scale))
        if gap > cfg.wall.closure_tol:
            closed_all = False
        else:
            v = np.vstack([v, v[0]])
        loops.append(v)
    plt.close(fig)

    loops = [subsample_net(cfg, v, wall.scale) for v in loops]
    n_seg = sum(max(0, len(v) - 1) for v in loops)
    return Polytope(
        loops=loops, closed=closed_all and bool(loops), n_segments=n_seg,
        note="" if loops else "no contour found",
    )


def subsample_net(cfg: Config, verts: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """IJRR Definition 2: enforce an (epsilon, eta)-net via minimum spacing d_min."""
    d_min = cfg.wall.net_d_min
    keep = [0]
    for i in range(1, len(verts)):
        if np.linalg.norm((verts[i] - verts[keep[-1]]) * scale) >= d_min:
            keep.append(i)
    if keep[-1] != len(verts) - 1:
        keep.append(len(verts) - 1)
    return verts[keep]


# --------------------------------------------------------------------------
# IJRR 4.4 -- Algorithm 3, Check-Polytope, with the Sigma guard added
# --------------------------------------------------------------------------


def segment_hits_disc(a: np.ndarray, b: np.ndarray, disc) -> bool:
    """Does segment [a, b] enter the disc? Elliptical: y and theta differ in units."""
    sc = np.array([1.0 / max(disc.radius, 1e-12), 1.0 / max(disc.r_theta, 1e-12)])
    c = np.array([disc.y, disc.theta]) * sc
    aa, bb = a * sc, b * sc
    d = bb - aa
    denom = float(np.dot(d, d))
    t = 0.0 if denom < 1e-18 else float(np.clip(np.dot(c - aa, d) / denom, 0.0, 1.0))
    return bool(np.linalg.norm(aa + t * d - c) <= 1.0)


def y_plus_penetration(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    point: np.ndarray, valid_seeds: np.ndarray,
) -> tuple[float, str]:
    """Penetration depth of an invalid page point, dispatched by invalidity tag.

    Two modes, switchable:

      [pd:ijrr]       strict IJRR Eq. 3 lifted to Y+: minimise distance in page
                      coordinates to the nearest VALID point, by local
                      optimisation seeded from the nearest valid sample. This
                      is what the paper does; it can over-estimate.

      [pd:lipschitz]  exact workspace penetration depth from the capsule/box
                      test, transferred to page coordinates by a Lipschitz
                      bound on d(robot)/d(y, theta). Sound given a guaranteed
                      constant; the constant here is a finite-difference
                      estimate with a safety factor, not interval arithmetic.
    """
    y, th = float(point[0]), float(point[1])
    pos, rot = weld.pose(y)
    res = arm.ik_branch(pos, rot, th, signs)

    # no-IK: analytic. W(y) depends on y alone, so the no-IK set is a union of
    # y-bands and the page-coordinate distance to its edge is exact.
    if not res.ok:
        d = _no_ik_margin(cfg, arm, weld, y)
        return d, "no-IK (analytic)"

    if cfg.wall.pd_mode == PD_LIPSCHITZ:
        return _pd_lipschitz(cfg, arm, weld, scene, signs, point, res)
    return _pd_ijrr(cfg, arm, weld, scene, signs, point, valid_seeds)


def _no_ik_margin(cfg: Config, arm: SrsArm, weld: WeldLine, y: float) -> float:
    """Distance in y to the nearest y where IK exists (exact, bisection-free)."""
    p = arm.p
    w = weld.wrist_center(y, p.d_wt)
    L = float(np.linalg.norm(w - arm.shoulder))
    if L > p.reach_max:
        excess = L - p.reach_max
    elif L < p.reach_min:
        excess = p.reach_min - L
    else:
        return 0.0
    # |dL/dy| <= |p1 - p0|, so a step of excess / |p1-p0| in y is needed.
    rate = max(1e-9, weld.length)
    return excess / rate


def _pd_ijrr(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    point: np.ndarray, valid_seeds: np.ndarray,
) -> tuple[float, str]:
    """Strict IJRR Eq. 3, lifted to Y+: distance to the nearest VALID point.

    Seeded from the pd_seed_attempts nearest valid samples, as before. What
    changed is how the closer valid point is searched for (findings 40).

      [pd:march]  (default) walk from the point toward each seed at
                  edge_resolution, take the FIRST valid sample, and bisect
                  the last step down to pd_bisect_iters halvings. Every
                  answer is a real valid point, never further than its seed,
                  so the bound is never looser than the nearest-sample
                  fallback -- and usually much tighter.
      [pd:slsqp]  legacy. SLSQP with classify() as an inequality constraint.
                  classify() is a +/-1 step whose gradient is zero almost
                  everywhere, so SLSQP has nothing to follow: on EXP 2 all
                  722 of 722 solves hit maxiter=40 after 443-494 classify
                  calls each, and EVERY pd came from the fallback below.
                  That was 99.9% of run_proof (215 s of 215 s).

    Both are an UPPER bound on the true depth (a valid point closer than any
    found one may exist), which is the IJRR method's known weakness; the
    "lipschitz" pd_mode is the hardened alternative.
    """
    if len(valid_seeds) == 0:
        return 0.0, "no valid seed available"
    scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
    d = np.linalg.norm((valid_seeds - point) * scale, axis=1)
    order = np.argsort(d)
    best = float(d[order[0]])

    if getattr(cfg.wall, "pd_search", "march") == "slsqp":
        for k in order[: cfg.wall.pd_seed_attempts]:
            seed = valid_seeds[k]

            def obj(z):
                return float(np.sum(((z - point) * scale) ** 2))

            def con(z):
                ok, _ = classify(cfg, arm, weld, scene, float(z[0]), float(z[1]), signs)
                return 1.0 if ok else -1.0

            res = minimize(obj, seed.copy(), constraints=[{"type": "ineq", "fun": con}],
                           method="SLSQP", options={"maxiter": 40, "ftol": 1e-9})
            if res.success:
                cand = np.asarray(res.x)
                ok, _ = classify(cfg, arm, weld, scene, float(cand[0]), float(cand[1]), signs)
                if ok:
                    return float(np.linalg.norm((cand - point) * scale)), "IJRR local opt"
        return best, "IJRR local opt (fallback: nearest sample)"

    def valid(z):
        return classify(cfg, arm, weld, scene, float(z[0]), float(z[1]), signs)[0]

    return pd_march(point, valid_seeds, valid, scale,
                    step=float(cfg.planner.edge_resolution),
                    attempts=int(cfg.wall.pd_seed_attempts),
                    bisect_iters=int(cfg.wall.pd_bisect_iters))


def pd_march(point: np.ndarray, valid_seeds: np.ndarray, valid_fn, scale: np.ndarray,
             step: float, attempts: int, bisect_iters: int,
             cap: float = float("inf")) -> tuple[float, str]:
    """IJRR Eq. 3 by marching toward the nearest valid seeds (any dimension).

    Returns an upper bound on the distance from `point` to the valid set, in
    SCALED units: the nearest seed, tightened by walking from the point
    toward each of the `attempts` nearest seeds and bisecting at the first
    valid sample.

    `cap`: Algorithm 3 only asks whether pd >= r. Marching stops at the step
    that passes the cap, and the answer is min(found, nearest seed, cap) --
    so pd >= r comes out exactly when the uncapped march would say so, at a
    fraction of the cost (findings 42).
    """
    scale = np.asarray(scale, dtype=float)
    d = np.linalg.norm((valid_seeds - point) * scale, axis=1)
    order = np.argsort(d)
    best = float(d[order[0]])
    how = "IJRR (fallback: nearest sample)"
    for k in order[: attempts]:
        seed = valid_seeds[k]
        length = float(d[k])
        if length <= 0.0:
            return 0.0, "point is a valid seed"
        # Only a valid point closer than the best so far can tighten the bound.
        n = int(np.ceil(min(length, best, cap) / step))
        lo_t = 0.0                       # last parameter known invalid
        hit = None
        for i in range(1, n + 1):
            t = min(1.0, i * step / length)
            if valid_fn(point + t * (seed - point)):
                hit = t
                break
            lo_t = t
        if hit is None:
            continue
        hi_t = hit
        for _ in range(int(bisect_iters)):
            mid = 0.5 * (lo_t + hi_t)
            if valid_fn(point + mid * (seed - point)):
                hi_t = mid
            else:
                lo_t = mid
        cand = hi_t * length
        if cand < best:
            best, how = cand, "IJRR march to first valid"
    if best > cap:
        return float(cap), "IJRR march: no valid point within the cap"
    return best, how


def _pd_lipschitz(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    point: np.ndarray, res,
) -> tuple[float, str]:
    """Exact workspace penetration depth, transferred by a Lipschitz bound."""
    q = res.q
    margin = arm.limit_margin(q)
    if margin < 0.0:
        lq = _jac_norm_q(cfg, arm, weld, signs, point)
        return abs(margin) / max(1e-9, lq), "joint-limit (Lipschitz)"

    report = scene.check(arm.frames(q))
    if report.in_collision:
        lw = _jac_norm_workspace(cfg, arm, weld, signs, point)
        return report.penetration / max(1e-9, lw), "collision (workspace PD / Lipschitz)"
    return 0.0, "point is valid"


def _fd_jac(cfg, arm, weld, signs, point, fn):
    h = cfg.wall.lipschitz_fd_step
    cols = []
    for axis in (0, 1):
        d = np.zeros(2)
        d[axis] = h
        a = fn(point + d)
        b = fn(point - d)
        if a is None or b is None:
            return None
        cols.append((a - b) / (2.0 * h))
    return np.column_stack(cols)


def _jac_norm_q(cfg, arm, weld, signs, point) -> float:
    def fn(z):
        pos, rot = weld.pose(float(z[0]))
        r = arm.ik_branch(pos, rot, float(z[1]), signs)
        return r.q if r.ok else None

    j = _fd_jac(cfg, arm, weld, signs, point, fn)
    if j is None:
        return cfg.wall.lipschitz_fallback
    return float(np.linalg.norm(j, 2)) * cfg.wall.lipschitz_safety


def _jac_norm_workspace(cfg, arm, weld, signs, point) -> float:
    def fn(z):
        pos, rot = weld.pose(float(z[0]))
        r = arm.ik_branch(pos, rot, float(z[1]), signs)
        if not r.ok:
            return None
        f = arm.frames(r.q)
        return np.concatenate([f["shoulder"], f["elbow"], f["wrist"], f["tool"]])

    j = _fd_jac(cfg, arm, weld, signs, point, fn)
    if j is None:
        return cfg.wall.lipschitz_fallback
    return float(np.linalg.norm(j, 2)) * cfg.wall.lipschitz_safety


def robot_facet_oracle(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    discs: list, valid_seeds: np.ndarray,
) -> FacetOracle:
    """The (y, theta) page of one sign branch, as Algorithm 3 sees it."""
    def guard(verts: np.ndarray) -> str | None:
        a, b = verts
        for d in discs:
            if segment_hits_disc(a, b, d):
                return f"crosses Sigma disc ({d.kind})"
        return None

    def valid(c: np.ndarray) -> bool:
        ok, _tag = classify(cfg, arm, weld, scene, float(c[0]), float(c[1]), signs)
        return bool(ok)

    from .arm import INVALID_NO_IK
    from .batch import classify_line_batch, pd_march_batch
    scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])

    def valid_many(C):
        return classify_line_batch(cfg, arm, weld, scene, C, signs)[0]

    depth_many = None
    if not cfg.planner.batch_classify:
        valid_many = None
    elif (cfg.wall.pd_mode == PD_IJRR and getattr(cfg.wall, "pd_search", "march") == "march"
            and len(valid_seeds)):
        def depth_many(C, R):
            # Same as y_plus_penetration: the analytic margin where there is
            # no IK, the uncapped IJRR march everywhere else.
            tags = classify_line_batch(cfg, arm, weld, scene, C, signs)[1]
            pd = np.zeros(len(C))
            noik = tags == INVALID_NO_IK
            for j in np.flatnonzero(noik):
                pd[j] = y_plus_penetration(cfg, arm, weld, scene, signs, C[j],
                                           valid_seeds)[0]
            rest = np.flatnonzero(~noik)
            if len(rest):
                pd[rest] = pd_march_batch(
                    C[rest], valid_seeds, valid_many, scale,
                    step=float(cfg.planner.edge_resolution),
                    attempts=int(cfg.wall.pd_seed_attempts),
                    bisect_iters=int(cfg.wall.pd_bisect_iters),
                    caps=np.full(len(rest), np.inf))
            return pd

    return FacetOracle(
        scale=scale,
        guard=guard,
        virtual=lambda c: bool(in_virtual_invalid(cfg, c[None, :])[0]),
        valid=valid,
        depth=lambda c, _r: y_plus_penetration(cfg, arm, weld, scene, signs, c,
                                               valid_seeds),
        valid_many=valid_many,
        depth_many=depth_many,
    )


def check_facet(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    a: np.ndarray, b: np.ndarray, discs: list, valid_seeds: np.ndarray,
    depth: int = 0,
) -> FacetResult:
    """IJRR Algorithm 3 on one segment, with the Sigma guard checked FIRST.

    The Sigma guard is new and must precede the minimum-enclosing-ball test: a
    wall may not pass through a door, or a path hops pages inside the disc and
    bypasses the wall entirely. The recursion itself is facets.check_simplex,
    shared with the triangle facets of a d = 3 wall.
    """
    oracle = robot_facet_oracle(cfg, arm, weld, scene, signs, discs, valid_seeds)
    return check_simplex(oracle, np.array([a, b], dtype=float),
                         cfg.wall.facet_max_depth, depth)


@dataclass
class CheckReport:
    ok: bool
    n_facets: int
    n_passed: int
    failures: list = field(default_factory=list)
    failure_points: list = field(default_factory=list)
    max_recursion: int = 0


def check_polytope(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    poly: Polytope, discs: list, valid_seeds: np.ndarray, verbose: bool = True,
    oracle: FacetOracle | None = None,
) -> CheckReport:
    """IJRR Algorithm 3 over every facet of the polytope.

    d = 2: the polyline segments, checked on the robot's (y, theta) page.
    d >= 3: the triangles of poly.simplices, checked through `oracle`. The
    page model is still 2-D, so there is no robot oracle for d = 3 yet; with
    none given the check REFUSES -- an unchecked facet must never count as
    passed (invariant 34) -- and in particular a d = 3 oracle must bring its
    own Sigma guard (facets.tube_guard), since the disc guard is 2-D.
    """
    if poly.dim != 2:
        if oracle is None or poly.vertices is None or poly.simplices is None:
            if verbose:
                print(f"[stage3] alg3 refused: no d={poly.dim} validity oracle "
                      f"(the robot page model is 2-D)")
            return CheckReport(ok=False, n_facets=int(poly.n_segments), n_passed=0,
                               failures=[(-1, f"no validity oracle for d={poly.dim}")])
        return _check_simplices(cfg, poly, oracle, verbose)
    n_facets = 0
    n_passed = 0
    failures = []
    failure_points = []
    max_rec = 0
    from .facets import check_many
    oracle2 = robot_facet_oracle(cfg, arm, weld, scene, signs, discs, valid_seeds)
    segs = [(loop, i) for loop in poly.loops for i in range(len(loop) - 1)]
    results = iter(check_many(oracle2, [np.array([lp[i], lp[i + 1]]) for lp, i in segs],
                              cfg.wall.facet_max_depth))
    for loop in poly.loops:
        for i in range(len(loop) - 1):
            n_facets += 1
            r = next(results)
            max_rec = max(max_rec, r.recursions)
            if r.ok:
                n_passed += 1
            else:
                failures.append((i, r.reason))
                failure_points.append((0.5 * (loop[i] + loop[i + 1]), r.reason))
    ok = n_facets > 0 and n_passed == n_facets
    if verbose:
        print(f"[stage3] alg3 facets {n_passed}/{n_facets} passed, "
              f"max recursion {max_rec}")
        for i, why in failures[:5]:
            print(f"         facet {i} FAILED: {why}")
    return CheckReport(ok=ok, n_facets=n_facets, n_passed=n_passed,
                       failures=failures, failure_points=failure_points,
                       max_recursion=max_rec)


def _check_simplices(cfg: Config, poly: Polytope, oracle: FacetOracle,
                     verbose: bool) -> CheckReport:
    """Algorithm 3 over the triangles of a d >= 3 polytope."""
    from .facets import enclosing_ball

    n_passed, failures, failure_points, max_rec = 0, [], [], 0
    verts = np.asarray(poly.vertices, dtype=float)
    simp = np.asarray(poly.simplices, dtype=int)
    scale = np.asarray(oracle.scale, dtype=float)
    from .facets import check_many
    results = check_many(oracle, [verts[s_] for s_ in simp],
                         getattr(cfg.wall, "facet_max_depth_tri", 5))
    for i, (s_, r) in enumerate(zip(simp, results)):
        max_rec = max(max_rec, r.recursions)
        if r.ok:
            n_passed += 1
        else:
            failures.append((i, r.reason))
            c, _ = enclosing_ball(verts[s_] * scale)
            failure_points.append((c / scale, r.reason))
    ok = len(simp) > 0 and n_passed == len(simp)
    if verbose:
        print(f"[stage3] alg3 facets {n_passed}/{len(simp)} passed "
              f"(d={poly.dim} triangles), max recursion {max_rec}")
        for i, why in failures[:5]:
            print(f"         facet {i} FAILED: {why}")
    return CheckReport(ok=ok, n_facets=len(simp), n_passed=n_passed,
                       failures=failures, failure_points=failure_points,
                       max_recursion=max_rec)
