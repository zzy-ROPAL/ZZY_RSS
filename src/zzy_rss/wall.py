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
  polytope (the actual proof)         closed polyline per region
  facet                              polyline segment
  CPD (Eq. 3)                        Y+ penetration depth, dispatched by tag

Two IJRR structural choices are preserved deliberately:

  * Both training classes are VALID points (IJRR 4.1). This learns a surface
    that lives INSIDE the invalid region, not a valid/invalid boundary.
  * Reconstruction is what VALIDATES closedness (IJRR 4.3). At d = 2 the wall
    is 1-D so the tangential complex degenerates to contour tracing, which is
    where IJRR's >90% triangulation bottleneck disappears -- but the closedness
    check must survive, or Definition 1 (I) is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

from .arm import INVALID_COLLISION, INVALID_JOINT_LIMIT, INVALID_NO_IK, SrsArm
from .params import Config
from .prm import classify
from .scene import Scene
from .weld import WeldLine

PD_IJRR = "ijrr_local_opt"     # strict IJRR Eq. 3, lifted to Y+
PD_LIPSCHITZ = "lipschitz"     # hardened: exact workspace PD + Lipschitz transfer

LABEL_TOPOLOGICAL = "topological"   # SDCL rule: connectivity decides the class
LABEL_METRIC = "metric"             # the nearest-class heuristic it replaces


# --------------------------------------------------------------------------
# The page as a CYLINDER (invariant 5, and why it stopped being optional)
# --------------------------------------------------------------------------
#
# theta is an arm angle, so a page is a cylinder; the flat rectangle is exact
# only while no valid cell touches theta = +/-pi, i.e. while the joint limits
# cut the theta circle before the seam. EXP 1-4 measured that and it held, so
# PageMapParams.theta_periodic stayed False and every distance in this file
# could be a plain Euclidean one.
#
# It does NOT hold for EXP 5, and it cannot be made to hold. On the arc
# placement each page's forbidden theta window is a thin band, and the s4
# mirror shifts q3 and q5 by pi, so the band of a page and the band of its s4
# PARTNER sit pi apart. A single cut in theta cannot land inside both. Since
# the fold door makes the proof visit both pages, one of them always has valid
# cells at the cut, and a wall "closed" across the cut there is open around the
# back of the cylinder -- invariant 5's void proof, exactly.
#
# So the cylinder is now carried properly, in three places:
#
#   1. DISTANCE. Every scaled difference in theta goes through
#      page_delta/page_dist below, which takes the short way round.
#
#   2. THE LEARNER. An RBF-SVM on (y, theta) is not a function on the
#      cylinder: it does not know that theta and theta + 2 pi are the same
#      state. WallModel therefore embeds theta as (cos theta, sin theta) when
#      the page is periodic, which makes the decision function EXACTLY
#      periodic by construction rather than approximately so by tiling the
#      training data. That matters because the certificate's argument is
#      "f > 0 on the seed class, f < 0 on the rest class, f = 0 only inside
#      C_obs, therefore every path between them crosses C_obs" -- and that
#      argument needs f to BE a function on the space the path lives in.
#      The embedding radius is scale_theta = 1/pi, which makes the embedding
#      locally isometric to the old flat scaling (chord ~ d_theta / pi), so
#      gamma keeps its meaning and nothing has to be retuned.
#
#   3. RECONSTRUCTION. On a cylinder a separating closed curve may WRAP
#      instead of enclosing, and a wrapping curve traced on a rectangle comes
#      back as an open arc. reconstruct_wall now accepts a loop whose
#      endpoints agree modulo 2 pi and closes it through the seam.


def page_delta(a: np.ndarray, b: np.ndarray, periodic: bool) -> np.ndarray:
    """b - a in page coordinates, taking the short way round in theta."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if not periodic:
        return d
    d = d.copy()
    d[..., 1] = (d[..., 1] + np.pi) % (2.0 * np.pi) - np.pi
    return d


def page_dist(a: np.ndarray, b: np.ndarray, scale: np.ndarray,
              periodic: bool) -> np.ndarray:
    """Scaled page distance(s) between a and b, cylinder-aware."""
    d = page_delta(a, b, periodic)
    d = np.atleast_2d(d) * np.asarray(scale, dtype=float)
    return np.linalg.norm(d, axis=1)


def is_periodic(cfg: Config) -> bool:
    return bool(getattr(cfg.pagemap, "theta_periodic", True))


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
    # True -> theta is embedded as (cos theta, sin theta), so `decision` is
    # EXACTLY 2 pi periodic and is therefore a function on the cylinder page.
    periodic: bool = False

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

    def features(self, pts: np.ndarray) -> np.ndarray:
        """Page coordinates -> the space the SVM actually lives in.

        Flat page: (scale_y * y, scale_theta * theta), as before.

        Cylinder page: (scale_y * y, scale_theta * cos theta,
        scale_theta * sin theta). With scale_theta = 1/pi the embedded circle
        has circumference 2, the same length the flat theta axis had, so a
        small dtheta maps to a chord of scale_theta * dtheta -- identical to
        the flat scaling to first order. gamma therefore means what it meant
        before and nothing needs retuning; what changes is that theta and
        theta + 2 pi are now literally the same input.
        """
        x = np.atleast_2d(np.asarray(pts, dtype=float))
        if not self.periodic:
            return x * self.scale
        sy, st = float(self.scale[0]), float(self.scale[1])
        return np.column_stack([x[:, 0] * sy,
                                st * np.cos(x[:, 1]),
                                st * np.sin(x[:, 1])])

    def decision(self, pts: np.ndarray) -> np.ndarray:
        """Same value as svm.decision_function, computed directly."""
        x = self.features(pts)
        d2 = np.sum((x[:, None, :] - self._sv[None, :, :]) ** 2, axis=2)
        return np.exp(-self._gamma * d2) @ self._dual + self._b

    def decision_grad(self, pt: np.ndarray) -> np.ndarray:
        """Gradient of `decision` at one point, in PAGE coordinates (y, theta).

        Giving SLSQP (and Newton) the analytic constraint gradient removes
        numerical differentiation, which was two extra evaluations per
        iteration. On the cylinder the chain rule through the embedding is one
        2x3 Jacobian: dy -> scale_y, dtheta -> scale_theta * (-sin, cos).
        """
        x = self.features(pt)[0]
        diff = x[None, :] - self._sv
        w = np.exp(-self._gamma * np.sum(diff ** 2, axis=1)) * self._dual
        g = (-2.0 * self._gamma) * (w[:, None] * diff).sum(axis=0)
        if not self.periodic:
            return g * self.scale
        sy, st = float(self.scale[0]), float(self.scale[1])
        th = float(np.asarray(pt, dtype=float).reshape(-1)[1])
        return np.array([g[0] * sy,
                         st * (-np.sin(th) * g[1] + np.cos(th) * g[2])])


def train_wall(cfg: Config, seed_pts: np.ndarray, rest_pts: np.ndarray) -> WallModel:
    """IJRR Algorithm 1: escalate gamma until the training data is fully separated.

    The loop deliberately over-fits -- complete separation is the requirement,
    not a pathology. It stops at the LARGEST effecting range (smallest gamma)
    that still separates, which is what keeps the zero set connected instead of
    fragmenting into Gaussian islands (IJRR Fig. 7).
    """
    wp = cfg.wall
    scale = np.array([wp.scale_y, wp.scale_theta])
    periodic = is_periodic(cfg)
    pts = np.vstack([seed_pts, rest_pts])
    if periodic:
        sy, st = float(scale[0]), float(scale[1])
        x = np.column_stack([pts[:, 0] * sy, st * np.cos(pts[:, 1]),
                             st * np.sin(pts[:, 1])])
    else:
        x = pts * scale
    y = np.concatenate([np.ones(len(seed_pts)), -np.ones(len(rest_pts))])

    gamma = wp.gamma_init
    svm = None
    acc = 0.0
    capped = False
    while True:
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
                     newton_tol=wp.newton_tol, periodic=periodic)


# --------------------------------------------------------------------------
# IJRR 4.1 -- the Fig. 5 virtual-boundary trick
# --------------------------------------------------------------------------


def sigma_disc_samples(discs: list, region_pts: np.ndarray, tol: float,
                       n_ring: int = 12, periodic: bool = False,
                       n_ring_fold: int = 2, ) -> tuple:
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
    unit = np.array([1.0, 1.0])

    # A FOLD is a LINE door, and the whole line is ONE posture of the robot:
    # at full extension the elbow circle has radius 0, so every theta at that
    # y is the same arm in space. So the line cannot be partly the seed class
    # and partly the rest class -- if the robot can be anywhere on it, it can
    # be everywhere on it. Classing the discs one by one would split the line
    # wherever the page happens to be invalid, and the SVM would then try to
    # thread a wall ALONG the fold, every facet of which Algorithm 3 rejects
    # for crossing a disc. Grouping by the line's y and deciding once is both
    # the cheap fix and the correct statement.
    groups: dict = {}
    singles = []
    for d in discs:
        if getattr(d, "fold", False) and d.exact_y is not None:
            groups.setdefault(round(float(d.exact_y), 12), []).append(d)
        else:
            singles.append(d)

    def _near(c: np.ndarray, rad: float) -> bool:
        return (len(region_pts) > 0
                and float(np.min(page_dist(region_pts, c, unit, periodic)))
                <= max(tol, rad))

    for d in singles:
        c = np.array([d.y, d.theta])
        pts = [c]
        for a in np.linspace(0.0, 2.0 * np.pi, max(1, n_ring), endpoint=False):
            pts.append(c + d.radius * np.array([np.cos(a), np.sin(a)]))
        (seed_extra if _near(c, d.radius) else rest_extra).extend(pts)

    for _y, grp in groups.items():
        near = any(_near(np.array([d.y, d.theta]), d.radius) for d in grp)
        bucket = seed_extra if near else rest_extra
        for d in grp:
            c = np.array([d.y, d.theta])
            bucket.append(c)
            # A thin ring: ~190 discs sit shoulder to shoulder along one y, so
            # a 12-point ring each would add ~2500 near-duplicate points and no
            # information. Rings exist to stop a wall squeezing between a POINT
            # door and its surroundings; a line has no such gap.
            for a in np.linspace(0.0, 2.0 * np.pi, max(1, n_ring_fold),
                                 endpoint=False):
                bucket.append(c + d.radius * np.array([np.cos(a), np.sin(a)]))

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


def boundary_anchor_samples(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, pm,
    mask: np.ndarray, dens: int | None = None, n_bisect: int = 24,
) -> np.ndarray:
    """Exact VALID postures ON the true boundary of `mask`, for TRAINING.

    edge_anchor_samples above supplies the page's y EDGES, which is where the
    sampled pipeline's wall was cutting in (findings 32). This supplies the
    other missing fact: where the class stops in the INTERIOR of the page.

    The grid samples the page every dy / dtheta, so the true edge of the valid
    set can sit up to a full cell away from any training point and the SVM is
    free to place its zero set anywhere in that gap. Measured on the EXP 5a
    candidate: all 16 crossed facets sat 0.0002-0.062 rad inside the true
    valid edge near the theta seam -- 0.001 to 0.395 of ONE cell, every one
    within a cell, i.e. the wall was hugging the joint-limit band and
    undershooting by a sub-cell amount it had no way to see.

    An SVM is a max-margin classifier, so pinning the true edge of BOTH
    classes puts the zero set down the middle of the corridor between them
    rather than in the middle of the sampling gap. On that scene it took the
    crossings from 16 to 6 while keeping the wall closed and gamma at 6.3.

    For every 4-adjacent pair (cell in `mask`, cell outside it) the segment
    between the two is bisected on classify(), and the valid-side point kept.
    A neighbour that is VALID but merely in a different component is skipped:
    there is no validity edge to find there. theta is treated as periodic when
    the page is, so the seam is covered like any other column.

    TRAINING SET ONLY. Putting these in the roadmap is invariant 18's mistake.
    """
    if dens is None:
        dens = int(getattr(cfg.wall, "boundary_anchors", 0) or 0)
    if dens <= 0:
        return np.zeros((0, 2), dtype=float)
    per = is_periodic(cfg)
    ys, ts = pm.y_grid, pm.theta_grid
    signs = pm.signs
    ny, nt = mask.shape
    out: list[tuple[float, float]] = []
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(ny):
        for j in range(nt):
            if not mask[i, j]:
                continue
            for di, dj in neigh:
                i2 = i + di
                if i2 < 0 or i2 >= ny:
                    continue
                j2 = j + dj
                if j2 < 0 or j2 >= nt:
                    if not per:
                        continue
                    j2 %= nt
                if mask[i2, j2]:
                    continue
                ay, at = float(ys[i]), float(ts[j])
                by = float(ys[i2])
                dt = float(ts[j2]) - at
                if per:
                    dt -= 2.0 * np.pi * round(dt / (2.0 * np.pi))
                if classify(cfg, arm, weld, scene, by,
                            _wrap_theta(at + dt), signs)[0]:
                    continue
                lo, hi = 0.0, 1.0
                for _ in range(n_bisect):
                    m = 0.5 * (lo + hi)
                    if classify(cfg, arm, weld, scene, ay + m * (by - ay),
                                _wrap_theta(at + m * dt), signs)[0]:
                        lo = m
                    else:
                        hi = m
                for sub in range(dens):
                    f = (sub + 0.5) / dens if dens > 1 else 1.0
                    m = lo * f
                    out.append((ay + m * (by - ay), _wrap_theta(at + m * dt)))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _wrap_theta(t: float) -> float:
    return float(np.arctan2(np.sin(t), np.cos(t)))


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
                 scale: np.ndarray, labeller=None, periodic: bool = False) -> int:
    """+1 seed / -1 rest: connectivity when available, nearest class otherwise."""
    if labeller is not None:
        lbl = labeller.label(m)
        if lbl != 0:
            return lbl
    d_seed = (float(np.min(page_dist(seed_pts, m, scale, periodic)))
              if len(seed_pts) else np.inf)
    d_rest = (float(np.min(page_dist(rest_pts, m, scale, periodic)))
              if len(rest_pts) else np.inf)
    return 1 if d_seed <= d_rest else -1


def sample_wall(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    wall: WallModel, invalid_pts: np.ndarray,
    seed_pts: np.ndarray, rest_pts: np.ndarray,
    max_rounds: int = 8, verbose: bool = True, labeller=None,
) -> tuple[WallModel, np.ndarray, np.ndarray, np.ndarray, int]:
    """IJRR Algorithm 2: project invalid seeds onto the wall and retrain if any
    projection lands in the valid region.

    A wall point found VALID means the wall is not inside the invalid region;
    it is added back into the training data (which is what pushes the wall in)
    and the SVM is retrained.
    """
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        wall_pts = []
        free_hits = []
        for q in invalid_pts:
            m = project_to_wall(wall, q)
            if m is None:
                continue
            ok, _tag = classify(cfg, arm, weld, scene, float(m[0]), float(m[1]), signs)
            if ok:
                free_hits.append(m)
            else:
                wall_pts.append(m)
        wall_pts = np.asarray(wall_pts, dtype=float).reshape(-1, 2)

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
            lab_m = assign_label(m, seed_pts, rest_pts, wall.scale, lb,
                                 wall.periodic)
            other = rest_pts if lab_m > 0 else seed_pts
            if len(other):
                d_other = float(np.min(page_dist(other, m, wall.scale,
                                                 wall.periodic)))
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
        wall = train_wall(cfg, seed_pts, rest_pts)
    return wall, np.zeros((0, 2)), seed_pts, rest_pts, rounds


# --------------------------------------------------------------------------
# IJRR 4.3 -- reconstruction, which also validates closedness
# --------------------------------------------------------------------------


@dataclass
class Polytope:
    """The actual proof object: closed polyline(s) with their segments."""

    loops: list = field(default_factory=list)   # list of (n_i, 2) arrays
    closed: bool = False
    n_segments: int = 0
    note: str = ""
    # Per loop: "contractible" (bounds a disc on the page) or "wrapping"
    # (goes round the theta circle and closes through the seam). Only a
    # cylinder page can have the second kind, and on a cylinder a SEPARATING
    # closed curve may well be of that kind, so the distinction is recorded
    # rather than normalised away.
    loop_kinds: list = field(default_factory=list)


def _fold_theta(verts: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Shift a loop by a whole number of periods so its centroid sits in range."""
    span = t_max - t_min
    c = float(np.mean(verts[:, 1]))
    k = np.floor((c - t_min) / span)
    if k == 0:
        return verts
    out = verts.copy()
    out[:, 1] -= k * span
    return out


def _contour_subpaths(cs) -> list:
    """Every separate curve of a contour level, as its own vertex array.

    Matplotlib stores ONE compound Path per contour level, with the separate
    curves joined inside it and marked by MOVETO codes. Reading `.vertices`
    straight off it therefore concatenates unrelated curves, and the first and
    last vertex then belong to DIFFERENT curves -- which makes the closure test
    compare two unrelated points. It only ever worked because the flat-page
    walls of EXP 1-4 happen to be a single loop; the moment a wall is two
    wrapping loops (which is the natural shape on a cylinder) the test read one
    open arc with a 0.876 gap in y and reported the wall not closed.
    """
    out = []
    for path in cs.get_paths():
        v = np.asarray(path.vertices, dtype=float)
        codes = path.codes
        if codes is None:
            out.append(v)
            continue
        starts = list(np.flatnonzero(codes == 1)) or [0]
        starts.append(len(v))
        for a, b in zip(starts[:-1], starts[1:]):
            if b - a >= 2:
                out.append(v[a:b])
    return out


def _one_period(verts: np.ndarray, span: float) -> np.ndarray | None:
    """Cut one full period out of a curve that wraps the cylinder.

    A wrapping wall traced on a grid tiled over three periods comes back as a
    single arc running across the whole tiled range in theta. One period of it
    IS the closed curve on the cylinder: by exact periodicity of the decision
    function its two ends sit at the same y, so they are the same point of the
    page and no closing facet is needed.
    """
    t = verts[:, 1]
    if t[-1] < t[0]:
        verts = verts[::-1]
        t = verts[:, 1]
    target = float(t[0]) + span
    # Walk the curve rather than binary-searching it: theta along a traced
    # contour is only approximately monotone, and searchsorted on a sequence
    # that dips by a vertex returns nonsense.
    j = None
    for i in range(1, len(verts)):
        if t[i] >= target:
            j = i
            break
    if j is None or j < 2:
        return None
    t0, t1 = float(t[j - 1]), float(t[j])
    if abs(t1 - t0) < 1e-15:
        return None
    w = (target - t0) / (t1 - t0)
    end = verts[j - 1] + w * (verts[j] - verts[j - 1])
    return np.vstack([verts[:j], end[None, :]])


def reconstruct_wall(cfg: Config, wall: WallModel, grid_res: int = 281) -> Polytope:
    """Trace the wall's zero level set into ordered closed polylines.

    At d = 2 the manifold is 1-D, so IJRR's tangential Delaunay complex
    degenerates to curve reconstruction; contour tracing gives the ordered
    polyline directly. The closedness test is retained: a loop whose endpoints
    do not meet fails Definition 1 (I) and triggers a retrain, exactly as a
    failed triangulation does in IJRR.

    On a CYLINDER page "endpoints meet" has two cases, and both are genuine
    closed curves:

      contractible  the endpoints coincide outright, as on a flat page.
      wrapping      the curve goes once round the theta circle and its
                    endpoints differ by exactly one period. On a cylinder this
                    is the natural shape for a curve that separates the start
                    end of the weld from the goal end, so refusing it would
                    reject the very wall the geometry calls for.

    Both are detected on a grid tiled over three periods in theta, which is
    exact because WallModel.decision is exactly periodic there; tiling also
    means a contractible loop STRADDLING the seam is traced whole instead of
    arriving as two arcs, one at each edge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = cfg.pagemap
    periodic = is_periodic(cfg)
    eps = cfg.wall.virtual_band
    # The traced domain must be STRICTLY WIDER than the virtual-valid ring at
    # 2*eps. With both at 2*eps the zero set ran along the domain edge and the
    # contour came back as an open arc with both endpoints at y = -0.060, so
    # the closure test failed and Definition 1 (I) was never established.
    m = cfg.wall.recon_margin * eps
    ys = np.linspace(pm.y_min - m, pm.y_max + m, grid_res)
    span = pm.theta_max - pm.theta_min
    if periodic:
        ts = np.linspace(pm.theta_min - span, pm.theta_max + span,
                         3 * grid_res // 2)
    else:
        ts = np.linspace(pm.theta_min - m, pm.theta_max + m, grid_res)
    yy, tt = np.meshgrid(ys, ts, indexing="ij")
    pts = np.column_stack([yy.ravel(), tt.ravel()])
    vals = wall.decision(pts).reshape(yy.shape)

    fig = plt.figure()
    cs = plt.contour(yy, tt, vals, levels=[cfg.wall.contour_level])
    loops, kinds, closed_all = [], [], True
    seen_centroids: list = []
    for v in _contour_subpaths(cs):
        if len(v) < 4:
            continue
        plain = float(np.linalg.norm((v[0] - v[-1]) * wall.scale))
        if plain <= cfg.wall.closure_tol:
            kind = "contractible"
            if periodic:
                v = _fold_theta(v, pm.theta_min, pm.theta_max)
                c = (float(np.mean(v[:, 0])), float(np.mean(v[:, 1])))
                if any(abs(c[0] - q[0]) < 1e-9 and abs(c[1] - q[1]) < 1e-9
                       for q in seen_centroids):
                    continue        # a translate of a loop already taken
                seen_centroids.append(c)
            v = np.vstack([v, v[0]])
        elif periodic and abs(abs(v[-1, 1] - v[0, 1])) >= span - 1e-9:
            cut = _one_period(v, span)
            if cut is None:
                closed_all = False
                loops.append(v)
                kinds.append("open")
                continue
            # By exact periodicity the two ends are the SAME page point, so the
            # curve is already closed; verify that rather than assume it.
            gap = float(np.linalg.norm(
                page_delta(cut[0], cut[-1], True) * wall.scale))
            if gap > max(cfg.wall.closure_tol, 1e-6):
                closed_all = False
                kind = "open"
            else:
                kind = "wrapping"
                cut = cut[:-1]       # drop the duplicate of the first vertex
            v = _fold_theta(cut, pm.theta_min, pm.theta_max)
        else:
            closed_all = False
            kind = "open"
        loops.append(v)
        kinds.append(kind)
    plt.close(fig)

    loops = [subsample_net(cfg, v, wall.scale, periodic) for v in loops]
    n_seg = sum(max(0, len(v) - 1) + (1 if k == "wrapping" else 0)
                for v, k in zip(loops, kinds))
    return Polytope(
        loops=loops, closed=closed_all and bool(loops), n_segments=n_seg,
        note="" if loops else "no contour found", loop_kinds=kinds,
    )


def subsample_net(cfg: Config, verts: np.ndarray, scale: np.ndarray,
                  periodic: bool = False) -> np.ndarray:
    """IJRR Definition 2: enforce an (epsilon, eta)-net via minimum spacing d_min."""
    d_min = cfg.wall.net_d_min
    keep = [0]
    for i in range(1, len(verts)):
        if float(page_dist(verts[keep[-1]], verts[i], scale, periodic)[0]) >= d_min:
            keep.append(i)
    if keep[-1] != len(verts) - 1:
        keep.append(len(verts) - 1)
    return verts[keep]


# --------------------------------------------------------------------------
# IJRR 4.4 -- Algorithm 3, Check-Polytope, with the Sigma guard added
# --------------------------------------------------------------------------


@dataclass
class FacetResult:
    ok: bool
    reason: str = ""
    depth_used: float = 0.0
    radius: float = 0.0
    recursions: int = 0


def segment_hits_disc(a: np.ndarray, b: np.ndarray, disc,
                      periodic: bool = False) -> bool:
    """Does segment [a, b] enter the disc? Elliptical: y and theta differ in units.

    On a cylinder page the segment and the disc centre are compared through
    the short way round in theta, so a facet at theta = 3.0 is tested against
    a disc at theta = -3.0 as the 0.28 rad apart that they really are.
    """
    sc = np.array([1.0 / max(disc.radius, 1e-12), 1.0 / max(disc.r_theta, 1e-12)])
    centre = np.array([disc.y, disc.theta])
    aa = page_delta(centre, a, periodic) * sc
    bb = page_delta(centre, b, periodic) * sc
    d = bb - aa
    denom = float(np.dot(d, d))
    t = 0.0 if denom < 1e-18 else float(np.clip(np.dot(-aa, d) / denom, 0.0, 1.0))
    return bool(np.linalg.norm(aa + t * d) <= 1.0)


def pack_discs(discs: list) -> tuple | None:
    """Discs as flat arrays, so a facet can be tested against ALL of them at once.

    check_facet used to loop over the disc list in Python. That was fine while
    a page had a handful of point doors; a fold LINE needs ~190 discs to cover
    theta, and the facet check then costs facets x discs scalar calls --
    measured 845 208 calls and 15.52 s, 38% of the whole proof. The test
    itself is a closed-form point-segment distance, so it vectorises exactly:
    same arithmetic, same answers, one numpy call per facet.
    """
    if not discs:
        return None
    return (np.array([d.y for d in discs], dtype=float),
            np.array([d.theta for d in discs], dtype=float),
            np.array([max(d.radius, 1e-12) for d in discs], dtype=float),
            np.array([max(d.r_theta, 1e-12) for d in discs], dtype=float))


def segment_hits_any_disc(a: np.ndarray, b: np.ndarray, packed,
                          periodic: bool = False) -> int:
    """Index of the first disc the segment enters, or -1. Vectorised form of
    segment_hits_disc over the whole disc set."""
    if packed is None:
        return -1
    cy, ct, ry, rt = packed
    day = (a[0] - cy) / ry
    dby = (b[0] - cy) / ry
    dat = a[1] - ct
    dbt = b[1] - ct
    if periodic:
        dat = (dat + np.pi) % (2.0 * np.pi) - np.pi
        dbt = (dbt + np.pi) % (2.0 * np.pi) - np.pi
    dat = dat / rt
    dbt = dbt / rt
    dy, dt = dby - day, dbt - dat
    denom = dy * dy + dt * dt
    t = np.where(denom < 1e-18, 0.0,
                 np.clip(-(day * dy + dat * dt) / np.where(denom < 1e-18, 1.0,
                                                           denom), 0.0, 1.0))
    hit = np.hypot(day + t * dy, dat + t * dt) <= 1.0
    idx = np.flatnonzero(hit)
    return int(idx[0]) if len(idx) else -1


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
    """Strict IJRR Eq. 3, lifted to Y+."""
    if len(valid_seeds) == 0:
        return 0.0, "no valid seed available"
    scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
    periodic = is_periodic(cfg)
    d = page_dist(valid_seeds, point, scale, periodic)
    order = np.argsort(d)

    for k in order[: cfg.wall.pd_seed_attempts]:
        seed = valid_seeds[k]

        def obj(z):
            return float(np.sum((page_delta(point, z, periodic) * scale) ** 2))

        def con(z):
            ok, _ = classify(cfg, arm, weld, scene, float(z[0]), float(z[1]), signs)
            if ok:
                return 1.0
            return -1.0

        res = minimize(obj, seed.copy(), constraints=[{"type": "ineq", "fun": con}],
                       method="SLSQP", options={"maxiter": 40, "ftol": 1e-9})
        if res.success:
            cand = np.asarray(res.x)
            ok, _ = classify(cfg, arm, weld, scene, float(cand[0]), float(cand[1]), signs)
            if ok:
                return (float(page_dist(point, cand, scale, periodic)[0]),
                        "IJRR local opt")
    return float(d[order[0]]), "IJRR local opt (fallback: nearest sample)"


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


def check_facet(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    a: np.ndarray, b: np.ndarray, discs: list, valid_seeds: np.ndarray,
    depth: int = 0, packed=None,
) -> FacetResult:
    """IJRR Algorithm 3 on one segment, with the Sigma guard checked FIRST.

    The Sigma guard is new and must precede the minimum-enclosing-ball test: a
    wall may not pass through a door, or a path hops pages inside the disc and
    bypasses the wall entirely.
    """
    scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
    periodic = is_periodic(cfg)

    if packed is None:
        packed = pack_discs(discs)
    k = segment_hits_any_disc(a, b, packed, periodic)
    if k >= 0:
        return FacetResult(False, f"crosses Sigma disc ({discs[k].kind})",
                           recursions=depth)

    # The midpoint and the half-length take the short way round in theta, so a
    # facet that crosses the seam is measured as the short segment it is and
    # not as a sweep of the whole theta range.
    delta = page_delta(a, b, periodic)
    centre = a + 0.5 * delta
    radius = 0.5 * float(np.linalg.norm(delta * scale))

    # Virtual invalid band counts as invalid with positive thickness.
    if in_virtual_invalid(cfg, centre[None, :])[0]:
        return FacetResult(True, "virtual boundary band", radius=radius, recursions=depth)

    ok, _tag = classify(cfg, arm, weld, scene,
                        float(centre[0]), float(centre[1]), signs)
    if ok:
        return FacetResult(False, "facet centre is VALID", radius=radius, recursions=depth)

    pd, how = y_plus_penetration(cfg, arm, weld, scene, signs, centre, valid_seeds)
    if pd >= radius:
        return FacetResult(True, how, depth_used=pd, radius=radius, recursions=depth)

    if depth >= cfg.wall.facet_max_depth:
        return FacetResult(False, f"depth limit; pd={pd:.4g} < r={radius:.4g}",
                           depth_used=pd, radius=radius, recursions=depth)

    mid = centre
    left = check_facet(cfg, arm, weld, scene, signs, a, mid, discs, valid_seeds,
                       depth + 1, packed)
    if not left.ok:
        return left
    return check_facet(cfg, arm, weld, scene, signs, mid, b, discs, valid_seeds,
                       depth + 1, packed)


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
) -> CheckReport:
    """IJRR Algorithm 3 over every facet of the polytope."""
    n_facets = 0
    n_passed = 0
    failures = []
    failure_points = []
    max_rec = 0
    periodic = is_periodic(cfg)
    packed = pack_discs(discs)
    for li, loop in enumerate(poly.loops):
        kind = (poly.loop_kinds[li] if li < len(poly.loop_kinds)
                else "contractible")
        # A contractible loop already repeats its first vertex at the end. A
        # WRAPPING loop does not: its last vertex and its first are the same
        # page point only after going through the theta seam, so that closing
        # facet has to be checked explicitly or one segment of the wall goes
        # unverified -- and it is precisely the segment at the seam, which is
        # the one invariant 5 warns about.
        pairs = [(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]
        if kind == "wrapping" and len(loop) > 1:
            pairs.append((loop[-1], loop[0] + np.array([0.0, 2.0 * np.pi])))
        for aa, bb in pairs:
            n_facets += 1
            r = check_facet(cfg, arm, weld, scene, signs, aa, bb,
                            discs, valid_seeds, 0, packed)
            max_rec = max(max_rec, r.recursions)
            if r.ok:
                n_passed += 1
            else:
                failures.append((n_facets - 1, r.reason))
                failure_points.append(
                    (aa + 0.5 * page_delta(aa, bb, periodic), r.reason))
    ok = n_facets > 0 and n_passed == n_facets
    if verbose:
        print(f"[stage3] alg3 facets {n_passed}/{n_facets} passed, "
              f"max recursion {max_rec}")
        for i, why in failures[:5]:
            print(f"         facet {i} FAILED: {why}")
    return CheckReport(ok=ok, n_facets=n_facets, n_passed=n_passed,
                       failures=failures, failure_points=failure_points,
                       max_recursion=max_rec)
