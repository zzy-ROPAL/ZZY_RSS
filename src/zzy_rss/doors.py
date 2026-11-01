"""Stage 2 -- the singular set Sigma and the door graph.

Sigma = Sigma_sign U Sigma_coord U Sigma_kin.

Three detection mechanisms, because the three kinds of door are geometrically
different and a single sign-crossing scan would miss two of them:

  interior doors  q2 = 0, q6 = 0.  Both IK branches exist on either side and
                  coincide at the crossing, so these ARE found by a sign
                  crossing, refined by bisection. Isolated points.

  fold doors      q4 = 0.  The s4=+ and s4=- branches merge and the page
                  region simply ENDS; there is no sign crossing to scan for.
                  Found where the IK discriminant vanishes (|cos q4| -> 1).
                  Also Sigma_coord, since the elbow circle collapses and the
                  arm angle loses meaning, so the whole theta-line degenerates.

  coord / kin     arm-angle reference breakdown, and det(J J^T) -> 0.

Sigma_kin is RECORDED but never invalidates a cell: see SigmaPolicy. Admitting
singular passages is the conservative direction for an infeasibility proof.

Connection rule is conservative: a disc connects every page whose sign vector
differs only in signs that vanish somewhere inside that disc. A disc holding
both a q2 and a q6 door therefore joins four pages, not two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .arm import SIGN_BRANCHES, VALID, SrsArm
from .params import Config
from .pagemap import PageMap, page_name, q_for_branch
from .scene import Scene
from .weld import WeldLine

DOOR_INTERIOR = "interior"
DOOR_FOLD = "fold"
SIGMA_COORD = "coord"
SIGMA_KIN = "kin"


@dataclass
class SigmaDisc:
    """A covering disc in page coordinates. Too large is safe; missing one is not."""

    kind: str
    y: float
    theta: float
    radius: float          # in y units; theta uses radius_theta
    radius_theta: float = 0.0
    # Which sign indices (0 -> s2, 1 -> s4, 2 -> s6) vanish inside this disc.
    vanishing: frozenset = field(default_factory=frozenset)
    pages: tuple = ()
    detail: str = ""
    # EXACT analytic door location, when known. The disc centre may be merged
    # or grid-snapped, which is fine for covering but NOT for evaluating the
    # self-motion fiber: the counter-rotation of q1/q3 is a true self-motion
    # only where q2 = 0 exactly. Measured drift of the tool tip along the
    # fiber: 7.85e-14 mm anchored at the exact door, 0.65 mm anchored 6.3e-4
    # rad off it. Anchoring off the door scans configurations that have left
    # the weld line, so the fiber must always be anchored here when available.
    exact_y: float | None = None
    exact_theta: float | None = None
    # True when this disc covers a q4 FOLD. A fold is a different object from
    # a point door and is certified by a different (exact) routine, so the
    # distinction has to survive _merge_discs -- `kind` does not, because a
    # merge concatenates kinds into "fold+interior".
    fold: bool = False

    @property
    def anchor(self) -> tuple:
        """(y, theta) at which to evaluate the fiber: exact if known."""
        if self.exact_y is not None and self.exact_theta is not None:
            return self.exact_y, self.exact_theta
        return self.y, self.theta

    @property
    def r_theta(self) -> float:
        return self.radius_theta if self.radius_theta > 0.0 else self.radius

    def contains(self, y: float, theta: float) -> bool:
        """Elliptical, because y and theta are in different units.

        A circular disc using hypot(dy, dtheta) was 0.318 wide on a 51x51 grid
        -- a third of the page -- because dtheta (0.126) dwarfs dy (0.02).
        Walls could not route around it and 38 facets failed on that alone.
        The analytic door location is exact, so the cover only needs to be a
        few cells in each coordinate.
        """
        return (((y - self.y) / self.radius) ** 2
                + ((theta - self.theta) / self.r_theta) ** 2) <= 1.0


SIGN_INDEX = {0: "s2", 1: "s4", 2: "s6"}
JOINT_OF_SIGN = {0: 1, 1: 3, 2: 5}   # sign index -> joint index in q


def _cell_diag(pm: PageMap) -> float:
    dy = float(pm.y_grid[1] - pm.y_grid[0])
    dt = float(pm.theta_grid[1] - pm.theta_grid[0])
    return float(np.hypot(dy, dt))


def _disc_radius(cfg: Config, pm: PageMap) -> float:
    """Cover radius in the y coordinate."""
    dy = float(pm.y_grid[1] - pm.y_grid[0])
    return max(cfg.sigma.disc_radius_floor, cfg.sigma.disc_radius_scale * dy)


def _disc_radius_theta(cfg: Config, pm: PageMap) -> float:
    """Cover radius in theta. Separate because the units differ by ~6x."""
    dt = float(pm.theta_grid[1] - pm.theta_grid[0])
    return max(cfg.sigma.disc_radius_floor, cfg.sigma.disc_radius_scale * dt)


def _quadratic_roots_on_unit(a: float, b: float, c: float, tol: float = 1e-12) -> list:
    """Roots of a y^2 + b y + c = 0 that lie in [0, 1].

    Kept as the straight-seam cross-check only; the door finders now call
    _roots_on_unit, which does not assume the condition is a quadratic.
    tests/test_ik.py::test_door_rootfinder_matches_quadratic asserts the two
    agree on the straight seam.
    """
    out = []
    if abs(a) < tol:
        if abs(b) > tol:
            out = [-c / b]
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            r = np.sqrt(disc)
            out = [(-b - r) / (2.0 * a), (-b + r) / (2.0 * a)]
    return [float(y) for y in out if -1e-9 <= y <= 1.0 + 1e-9]


def _roots_on_unit(f, n_scan: int = 1201, merge_tol: float = 1e-9) -> list:
    """Every root of a scalar f on [0, 1], found numerically.

    The door conditions used to be solved in closed form as quadratics in y,
    which was only possible because a straight seam with a constant tool pose
    makes W(y) AFFINE. An arc seam breaks that (invariant 22, and it is the
    whole reason EXP 5 exists), so the conditions are no longer polynomial and
    are bracketed by a dense sign scan and then refined by Brent to the last
    representable bit. On the straight seam this reproduces the quadratic roots
    exactly, which is what the cross-check test asserts.

    A dense scan can in principle step over a pair of roots closer together
    than its spacing. The same caveat applied to nothing before -- a quadratic
    has at most two roots and they were found exactly -- so n_scan is an
    explicit parameter and the fold finder below additionally looks for
    TANGENCIES, which is precisely the double root a sign scan cannot see.
    """
    ys = np.linspace(0.0, 1.0, n_scan)
    vals = np.array([float(f(float(y))) for y in ys])
    out: list[float] = []
    for i in range(n_scan - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0.0:
            out.append(float(ys[i]))
        elif a * b < 0.0:
            out.append(float(brentq(f, ys[i], ys[i + 1],
                                    xtol=1e-15, rtol=8.9e-16)))
    if vals[-1] == 0.0:
        out.append(1.0)
    uniq: list[float] = []
    for y in sorted(out):
        if not uniq or abs(y - uniq[-1]) > merge_tol:
            uniq.append(y)
    return uniq


def _tangencies_on_unit(f, tol: float, n_scan: int = 1201) -> list:
    """Interior local extrema of f where |f| <= tol: the DOUBLE roots.

    A fold door needs |W(y) - S| = reach_max. When the seam's farthest point
    sits exactly at the arm's reach limit, that equation has a double root:
    |W - S| touches reach_max from below and comes back down, so both halves of
    the task stay reachable and the fold lands in the task's INTERIOR. There is
    no sign change, so _roots_on_unit cannot see it -- this is the finder for
    exactly that case, and it is the geometry docs/exp5_spec.md condition B
    asks for.

    The placement is solved so that the residual is at machine precision
    (measured 0.0 to 1.1e-16 over the candidates in docs/exp5_spec.md), and
    `tol` is SigmaParams.fold_tangency_tol. Its meaning in robot terms:
    a residual delta leaves the arm |q4| ~ sqrt(2 reach_max delta / (d_se d_ew))
    short of straight, i.e. delta = 1e-9 m is |q4| <= 3.5e-5 rad and an elbow
    circle of radius <= 18 um. Below that the elbow circle has collapsed as
    far as double precision can express.
    """
    ys = np.linspace(0.0, 1.0, n_scan)
    vals = np.array([float(f(float(y))) for y in ys])
    out: list[float] = []
    for i in range(1, n_scan - 1):
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        if not ((b >= a and b >= c) or (b <= a and b <= c)):
            continue
        lo, hi = float(ys[i - 1]), float(ys[i + 1])
        sign = 1.0 if (b >= a and b >= c) else -1.0
        r = minimize_scalar(lambda y: sign * -float(f(float(y))),
                            bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-14})
        y_ext = float(np.clip(r.x, 0.0, 1.0))
        if abs(float(f(y_ext))) <= tol and 1e-6 < y_ext < 1.0 - 1e-6:
            out.append(y_ext)
    uniq: list[float] = []
    for y in sorted(out):
        if not uniq or abs(y - uniq[-1]) > 1e-7:
            uniq.append(y)
    return uniq


def _arm_angle_of_elbow(
    arm: SrsArm, weld: WeldLine, y: float, elbow_target: np.ndarray,
) -> float | None:
    """Recover the arm angle theta that places the elbow at `elbow_target`."""
    p = arm.p
    pos, rot = weld.pose(y)
    e_z = np.array([0.0, 0.0, 1.0])
    wrist = pos - p.d_wt * (rot @ e_z)
    x_sw = wrist - arm.shoulder
    length = float(np.linalg.norm(x_sw))
    if length < 1e-12:
        return None
    u = x_sw / length
    h = (p.d_se ** 2 - p.d_ew ** 2 + length ** 2) / (2.0 * length)
    radius = float(np.sqrt(max(0.0, p.d_se ** 2 - h ** 2)))
    if radius < 1e-9:
        return None
    ref = e_z if abs(float(np.dot(u, e_z))) < 1.0 - 1e-6 else np.array([1.0, 0.0, 0.0])
    v0 = ref - float(np.dot(ref, u)) * u
    v0 = v0 / float(np.linalg.norm(v0))
    w0 = np.cross(u, v0)
    rel = elbow_target - arm.shoulder - h * u
    return float(np.arctan2(float(np.dot(rel, w0)), float(np.dot(rel, v0))))


def find_interior_doors_analytic(
    cfg: Config, arm: SrsArm, weld: WeldLine, pm: PageMap, sign_index: int,
) -> list[SigmaDisc]:
    """Exact interior doors from the geometric condition.

    Within a page the sign is FIXED by construction -- the IK returns
    q_j = s_j * arccos(.), which is never negative on an s_j=+ page -- so q_j
    can only TOUCH zero, never cross it. A sign-crossing scan therefore finds
    nothing. The door is located instead from its geometry:

      q2 = 0  pins the elbow at E* = S + d_se * e_z, so the wrist must satisfy
              |W(y) - E*| = d_ew.

      q6 = 0  aligns the forearm with the tool axis, so the elbow is pinned at
              E(y) = W(y) - d_ew * (R(y) e_z) and must satisfy |E(y) - S| = d_se.

    For a STRAIGHT seam with a constant tool pose both conditions are affine in
    y, hence quadratic, hence "at most two q2 doors" was a derived bound rather
    than a scan. An ARC seam is not affine (invariant 22), so the conditions are
    solved by _roots_on_unit instead: a dense sign scan refined by Brent to
    1e-15. On a straight seam that returns the quadratic's roots to the last
    bit, which test_door_rootfinder_matches_quadratic asserts; on an arc it is
    the only method available, and the "at most two" bound no longer holds.

    theta at the door follows by inverting the elbow-circle parameterisation.
    """
    p = arm.p
    e_z = np.array([0.0, 0.0, 1.0])
    radius = _disc_radius(cfg, pm)
    r_th = _disc_radius_theta(cfg, pm)

    if sign_index == 0:          # q2 door
        target = arm.shoulder + p.d_se * e_z     # E*
        joint = 2

        def cond(y: float) -> float:
            w = weld.wrist_center(y, p.d_wt)
            return float(np.linalg.norm(w - target)) - p.d_ew

        def elbow_at(y: float) -> np.ndarray:
            return target
    elif sign_index == 2:        # q6 door
        joint = 6

        def elbow_at(y: float) -> np.ndarray:
            _pos, rot = weld.pose(y)
            return weld.wrist_center(y, p.d_wt) - p.d_ew * (rot @ e_z)

        def cond(y: float) -> float:
            return float(np.linalg.norm(elbow_at(y) - arm.shoulder)) - p.d_se
    else:
        return []

    found = []
    for y in _roots_on_unit(cond):
        y = min(max(y, 0.0), 1.0)
        theta = _arm_angle_of_elbow(arm, weld, y, elbow_at(y))
        if theta is None:
            continue
        found.append(
            SigmaDisc(
                kind=DOOR_INTERIOR,
                y=y,
                theta=theta,
                radius=radius,
                radius_theta=r_th,
                vanishing=frozenset({sign_index}),
                exact_y=y,
                exact_theta=theta,
                detail=f"q{joint}=0 door (analytic) on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def find_interior_doors_numeric(
    cfg: Config, arm: SrsArm, weld: WeldLine, pm: PageMap, sign_index: int,
) -> list[SigmaDisc]:
    """Numerical cross-check: grid minima of |q_j| below a tolerance.

    Generalises to welds where the analytic condition is not a quadratic. Used
    to confirm the analytic finder rather than to replace it.
    """
    joint = JOINT_OF_SIGN[sign_index]
    field_map = {1: pm.q2, 3: pm.q4, 5: pm.q6}[joint]
    radius = _disc_radius(cfg, pm)
    r_th = _disc_radius_theta(cfg, pm)
    tol = cfg.sigma.touch_tol

    mag = np.abs(field_map)
    found = []
    n_y, n_t = mag.shape
    for i in range(n_y):
        for j in range(n_t):
            v = mag[i, j]
            if not np.isfinite(v) or v > tol:
                continue
            window = mag[max(0, i - 1): i + 2, max(0, j - 1): j + 2]
            if np.nanmin(window) < v - 1e-15:
                continue
            found.append(
                SigmaDisc(
                    kind=DOOR_INTERIOR,
                    y=float(pm.y_grid[i]),
                    theta=float(pm.theta_grid[j]),
                    radius=radius,
                radius_theta=r_th,
                    vanishing=frozenset({sign_index}),
                    detail=f"q{joint + 1}~0 local minimum on page {pm.name}",
                )
            )
    return _merge_discs(found, radius)


def find_interior_doors(
    cfg: Config, arm: SrsArm, weld: WeldLine, pm: PageMap, sign_index: int,
) -> list[SigmaDisc]:
    """Analytic doors, unioned with the numerical cross-check (conservative)."""
    discs = find_interior_doors_analytic(cfg, arm, weld, pm, sign_index)
    if cfg.sigma.use_numeric_crosscheck:
        discs = discs + find_interior_doors_numeric(cfg, arm, weld, pm, sign_index)
    return _merge_discs(discs, _disc_radius(cfg, pm))


def find_fold_doors(cfg: Config, pm: PageMap, arm=None, weld=None) -> list[SigmaDisc]:
    """q4 folds: where the IK discriminant vanishes, not where a sign crosses.

    |W(y) - S| = d_se + d_ew is full extension and |d_se - d_ew| is the fully
    folded pose; at either the s4 branches merge, so there is no sign crossing
    to scan for.

    A fold is simultaneously Sigma_coord: the elbow circle collapses, the arm
    angle loses meaning, and the WHOLE theta-line at that y degenerates to one
    posture. The disc cover therefore spans the full theta range there, which
    is what makes a fold a LINE door rather than a point door.

    TWO kinds of fold, and the difference is the whole of EXP 5:

      TRANSVERSAL  |W - S| - reach_max changes sign. The task leaves the
                   workspace: one side is reachable, the other is not. This is
                   the only fold a STRAIGHT seam can have, and it sits at the
                   edge of the reachable y interval (invariant 22).

      TANGENTIAL   |W - S| - reach_max touches zero from below: a DOUBLE root.
                   Both sides stay reachable and the fold lands in the task's
                   INTERIOR. No sign scan can find it, hence
                   _tangencies_on_unit. Only a non-affine W(y) -- an arc seam
                   or a tool pose that varies along the seam -- can produce
                   one, which is why weld.py grew a curve mode.
    """
    radius = _disc_radius(cfg, pm)
    r_th = _disc_radius_theta(cfg, pm)
    found: list[SigmaDisc] = []

    def _lay_line(y: float, what: str) -> None:
        """Cover the WHOLE theta line at this y: the fold is a line door.

        Spacing is the y radius so consecutive discs overlap in theta (whose
        radius is ~6x larger), giving a gap-free cover of the full theta range
        extended by r_theta at each end. On a cylinder page that is the entire
        theta circle, so no wall can cross the fold anywhere.
        """
        n = max(3, int(np.ceil((pm.theta_grid[-1] - pm.theta_grid[0]) / radius)))
        for theta in np.linspace(pm.theta_grid[0], pm.theta_grid[-1], n):
            found.append(
                SigmaDisc(
                    kind=DOOR_FOLD,
                    y=float(min(max(y, 0.0), 1.0)),
                    theta=float(theta),
                    radius=radius,
                    radius_theta=r_th,
                    vanishing=frozenset({1}),
                    fold=True,
                    # The fold's y is exact; theta is genuinely free there, so
                    # the anchor's theta is whichever column this disc covers.
                    exact_y=float(y),
                    exact_theta=float(theta),
                    detail=f"q4 fold ({what}) on page {pm.name}",
                )
            )

    if arm is not None and weld is not None:
        p = arm.p
        for r_eq, what in ((p.reach_max, "full extension"),
                           (p.reach_min, "full fold")):
            if r_eq <= 1e-9:
                continue

            def cond(y: float, r_eq=r_eq) -> float:
                w = weld.wrist_center(y, p.d_wt)
                return float(np.linalg.norm(w - arm.shoulder)) - r_eq

            for y in _roots_on_unit(cond):
                _lay_line(y, f"{what}, transversal")
            for y in _tangencies_on_unit(cond, cfg.sigma.fold_tangency_tol):
                _lay_line(y, f"{what}, TANGENTIAL -- interior line door")

    # Numerical backstop: cells the grid flags as collapsed.
    margin = cfg.sigma.fold_cos_margin
    near = (np.abs(pm.q4) < margin) | pm.near_full
    for i, j in np.argwhere(near):
        found.append(
            SigmaDisc(
                kind=DOOR_FOLD,
                y=float(pm.y_grid[i]),
                theta=float(pm.theta_grid[j]),
                radius=radius,
                radius_theta=r_th,
                vanishing=frozenset({1}),
                fold=True,
                detail=f"q4 fold (grid) on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def find_coord_discs(cfg: Config, pm: PageMap) -> list[SigmaDisc]:
    """Arm-angle reference breakdown: Sigma_coord."""
    radius = _disc_radius(cfg, pm)
    r_th = _disc_radius_theta(cfg, pm)
    found = []
    for i, j in np.argwhere(pm.near_axis):
        found.append(
            SigmaDisc(
                kind=SIGMA_COORD,
                y=float(pm.y_grid[i]),
                theta=float(pm.theta_grid[j]),
                radius=radius,
                radius_theta=r_th,
                vanishing=frozenset(),
                detail=f"arm-angle reference degenerate on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def find_kin_discs(cfg: Config, pm: PageMap) -> list[SigmaDisc]:
    """det(J J^T) zeros: Sigma_kin. RECORDED ONLY -- never invalidates a cell."""
    radius = _disc_radius(cfg, pm)
    r_th = _disc_radius_theta(cfg, pm)
    found = []
    m = pm.kin_measure
    mask = np.isfinite(m) & (np.abs(m) < cfg.sigma.kin_det_tol)
    for i, j in np.argwhere(mask):
        found.append(
            SigmaDisc(
                kind=SIGMA_KIN,
                y=float(pm.y_grid[i]),
                theta=float(pm.theta_grid[j]),
                radius=radius,
                radius_theta=r_th,
                vanishing=frozenset(),
                detail=f"det(J J^T) ~ 0 on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def _merge_discs(discs: list[SigmaDisc], radius: float) -> list[SigmaDisc]:
    """Collapse discs that overlap, unioning their vanishing-sign sets."""
    out: list[SigmaDisc] = []
    for d in discs:
        for k, e in enumerate(out):
            if ((d.y - e.y) / max(radius, 1e-12)) ** 2 + \
                    ((d.theta - e.theta) / max(radius, 1e-12)) ** 2 <= 1.0:
                out[k] = SigmaDisc(
                    kind=e.kind if e.kind == d.kind else f"{e.kind}+{d.kind}",
                    y=0.5 * (e.y + d.y),
                    theta=0.5 * (e.theta + d.theta),
                    radius=max(e.radius, d.radius),
                    radius_theta=max(e.r_theta, d.r_theta),
                    vanishing=e.vanishing | d.vanishing,
                    pages=e.pages,
                    detail=e.detail,
                    # keep the exact anchor from whichever disc carries one
                    exact_y=e.exact_y if e.exact_y is not None else d.exact_y,
                    exact_theta=(e.exact_theta if e.exact_theta is not None
                                 else d.exact_theta),
                    fold=bool(e.fold or d.fold),
                )
                break
        else:
            out.append(d)
    return out


def assign_pages(disc: SigmaDisc, page: tuple[int, int, int]) -> tuple:
    """Conservative connection rule.

    A disc connects every page whose sign vector differs from `page` only in
    signs that vanish somewhere inside that disc. With two vanishing signs in
    one disc this gives four pages, not two.
    """
    if not disc.vanishing:
        # Sigma_coord / Sigma_kin: connect conservatively to every page.
        return tuple(page_name(s) for s in SIGN_BRANCHES)
    pages = set()
    v = sorted(disc.vanishing)
    for mask in range(1 << len(v)):
        signs = list(page)
        for bit, idx in enumerate(v):
            if mask & (1 << bit):
                signs[idx] = -signs[idx]
        pages.add(page_name(tuple(signs)))
    return tuple(sorted(pages))


def build_sigma(
    cfg: Config,
    arm: SrsArm,
    weld: WeldLine,
    pages: dict,
    verbose: bool = True,
) -> dict:
    """Stage 2 for every page. Returns {page_name: [SigmaDisc, ...]}."""
    if cfg.sigma_policy.sigma_kin_invalidates:
        raise ValueError(
            "sigma_kin_invalidates=True is unsound for the headline claim: it "
            "would prove infeasibility of a narrower problem than 'no plan "
            "exists'. Refusing to run."
        )

    sigma = {}
    for name, pm in pages.items():
        discs: list[SigmaDisc] = []
        discs += find_interior_doors(cfg, arm, weld, pm, 0)   # q2 = 0
        discs += find_interior_doors(cfg, arm, weld, pm, 2)   # q6 = 0
        discs += find_fold_doors(cfg, pm, arm, weld)          # q4 fold
        discs += find_coord_discs(cfg, pm)
        discs += find_kin_discs(cfg, pm)
        discs = _merge_discs(discs, _disc_radius(cfg, pm))
        discs = [
            SigmaDisc(
                kind=d.kind, y=d.y, theta=d.theta, radius=d.radius,
                radius_theta=d.r_theta,
                vanishing=d.vanishing, pages=assign_pages(d, pm.signs),
                detail=d.detail, exact_y=d.exact_y, exact_theta=d.exact_theta,
                fold=d.fold,
            )
            for d in discs
        ]
        sigma[name] = discs
        if verbose:
            by_kind: dict = {}
            for d in discs:
                by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(by_kind.items())) or "none"
            print(f"[stage2] page {name}  Sigma discs: {len(discs):3d}  ({summary})")
    return sigma


# --------------------------------------------------------------------------
# Stage 5 refinement (OPTIONAL) -- fiber traversability
# --------------------------------------------------------------------------


@dataclass
class FiberReport:
    """Whether the 1-D self-motion fiber at a door can actually be traversed."""

    door_kind: str
    y: float
    theta: float
    traversable: bool
    blocked_fraction: float
    reason: str
    n_samples: int


def certify_fiber(
    cfg: Config,
    arm: SrsArm,
    weld: WeldLine,
    scene: Scene,
    disc: SigmaDisc,
    page: tuple[int, int, int],
) -> FiberReport | None:
    """Stage 5 only: try to show a door's self-motion fiber is wholly invalid.

    At a q2 = 0 door the fiber is the gimbal arc: q1 and q3 counter-rotate with
    q1 + q3 held, so the tool pose never moves -- the end-effector stays exactly
    on the weld line for the whole fiber. Only frame 2 rotates, so only joint
    limits on q1/q3 and off-axis upper-arm geometry can block it.

    Returns None when fiber certification is disabled, which is the default:
    assuming the door OPEN is conservative and never needed for soundness.
    """
    if not cfg.sigma_policy.enable_fiber_certification:
        return None

    anchor_y, anchor_theta = disc.anchor
    pos, rot = weld.pose(anchor_y)
    if disc.fold:
        cert = certify_fold_door(cfg, arm, weld, scene, disc,
                                 pages=disc.pages or None)
        return FiberReport(disc.kind, disc.y, disc.theta, bool(cert.open),
                           0.0 if cert.open else 1.0, cert.reason, 0)

    res = arm.ik_branch(pos, rot, anchor_theta, page)
    if not res.ok:
        return FiberReport(disc.kind, disc.y, disc.theta, True, 0.0,
                           "no IK at door; left open", 0)
    q0 = res.q

    if 0 in disc.vanishing:
        free_pair = (0, 2)       # q1 / q3 counter-rotation
    elif 2 in disc.vanishing:
        free_pair = (4, 6)       # q5 / q7 counter-rotation
    else:
        return FiberReport(disc.kind, disc.y, disc.theta, True, 0.0,
                           "fiber not characterised for this disc kind; left open", 0)

    i, k = free_pair
    n = cfg.sigma_policy.fiber_samples
    deltas = np.linspace(-np.pi, np.pi, n)
    n_blocked = 0
    for d in deltas:
        q = q0.copy()
        q[i] += d
        q[k] -= d
        if not arm.within_limits(q):
            n_blocked += 1
            continue
        if scene.check(arm.frames(q)).in_collision:
            n_blocked += 1
    frac = n_blocked / n
    traversable = frac < 1.0
    return FiberReport(
        door_kind=disc.kind,
        y=disc.y,
        theta=disc.theta,
        traversable=traversable,
        blocked_fraction=frac,
        reason=(
            "fiber wholly invalid -> door CERTIFIED BLOCKED"
            if not traversable
            else f"{frac * 100:.1f}% of fiber invalid -> door stays OPEN"
        ),
        n_samples=n,
    )


def door_fiber_validity(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    disc: SigmaDisc, page: tuple[int, int, int], n: int | None = None,
) -> dict:
    """Validity of a door's whole self-motion fiber, not just its representative.

    This is the correct reachability primitive for a door. The IK returns ONE
    representative of the fiber, and that representative can be invalid while
    other fiber points are valid: at a q2 = 0 door the IK picks a particular
    (q1, q3) split, which on an s2=+ page comes out as q3 = pi and can sit just
    outside a +/-170 degree limit, even though the fiber point a few degrees
    away is comfortably inside it.

    Every fiber point places the end-effector at the SAME point on the weld
    line, so this is never a question of leaving the constraint -- only of
    whether the fiber can be occupied and traversed.

    Returns the valid fraction and the valid delta-intervals.
    """
    # A FOLD (line) door is not decided by sampling at all: the whole theta
    # line is one posture, so certify_fold_door settles it with one collision
    # query and an exact interval intersection. Routing it here rather than at
    # every call site also fixes a soundness bug: the old `else` branch below
    # returned ok=False for any disc whose fiber it could not characterise,
    # and ok=False is read by live_discs and run_proof as "CERTIFIED BLOCKED"
    # -- so a fold door was silently treated as shut, the wall was allowed to
    # cross it and the flood never followed it. Invariant 2 says the
    # conservative default is OPEN, and that is now what an unknown fiber gets.
    if disc.fold:
        cert = certify_fold_door(cfg, arm, weld, scene, disc,
                                 pages=disc.pages or None)
        return {"ok": bool(cert.open), "reason": cert.reason,
                "valid_fraction": 1.0 if cert.open else 0.0,
                "intervals": [], "exact": cert.exact, "fold": True,
                "certificate": cert}

    n = n or cfg.sigma_policy.fiber_samples
    anchor_y, anchor_theta = disc.anchor
    pos, rot = weld.pose(anchor_y)
    res = arm.ik_branch(pos, rot, anchor_theta, page)
    if not res.ok:
        return {"ok": True, "reason": "no IK at the door; left OPEN "
                                      "(invariant 2)",
                "valid_fraction": 0.0, "intervals": []}

    if 0 in disc.vanishing:
        pair = (0, 2)
    elif 2 in disc.vanishing:
        pair = (4, 6)
    else:
        return {"ok": True,
                "reason": "fiber not characterised; left OPEN (invariant 2)",
                "valid_fraction": 0.0, "intervals": []}

    i, k = pair
    q0 = res.q
    deltas = np.linspace(-np.pi, np.pi, n)
    flags = np.zeros(n, dtype=bool)
    blockers = {}
    for idx, d in enumerate(deltas):
        q = q0.copy()
        q[i] += d
        q[k] -= d
        if not arm.within_limits(q):
            blockers["joint-limit"] = blockers.get("joint-limit", 0) + 1
            continue
        rep = scene.check(arm.frames(q))
        if rep.in_collision:
            blockers[f"collision:{rep.witness}"] = (
                blockers.get(f"collision:{rep.witness}", 0) + 1)
            continue
        flags[idx] = True

    intervals = []
    start = None
    for idx, f in enumerate(flags):
        if f and start is None:
            start = idx
        elif not f and start is not None:
            intervals.append((float(deltas[start]), float(deltas[idx - 1])))
            start = None
    if start is not None:
        intervals.append((float(deltas[start]), float(deltas[-1])))

    return {
        "ok": bool(flags.any()),
        "reason": ("fiber has valid points -> door OPEN" if flags.any()
                   else "fiber wholly invalid -> door CERTIFIED BLOCKED"),
        "valid_fraction": float(flags.mean()),
        "intervals": intervals,
        "blockers": blockers,
    }


# --------------------------------------------------------------------------
# Fold doors: EXACT certification (closes todo 5 for line doors)
# --------------------------------------------------------------------------
#
# A point door (q2 = 0 or q6 = 0) is certified by SAMPLING its self-motion
# fiber, and the fiber is a measure-zero set in Q, so a dense sample is
# evidence and not a certificate (todo 5). A fold door is different, and the
# difference is exact rather than a matter of degree:
#
#   * at q4 = 0 the elbow circle has radius 0, so the elbow sits ON the
#     shoulder-wrist axis and the arm's four points -- shoulder, elbow, wrist,
#     tool -- do not depend on theta at all. The WHOLE theta line is one
#     posture, so collision is ONE query, not a scan.
#
#   * q4 = 0 also makes the q3 and q5 axes collinear (roty(0) = I), so
#     (q3 + delta, q5 - delta) is a null motion: another whole family that
#     leaves every link exactly where it is. Collision is still that same one
#     query along it.
#
#   * what DOES vary along both families is the joint VALUES, so the only
#     remaining question is joint limits -- and limits are a box while the
#     family is a straight line in q, so the admissible set is computed by
#     INTERSECTING INTERVALS. Exactly, with no samples anywhere.
#
# So a fold door's state is decided by one collision query plus an exact
# interval intersection per page. That is the whole of todo 5 for line doors.


def _wrapped_admissible_deltas(
    value: float, coeff: float, lo: float, hi: float,
    d_lo: float = -np.pi, d_hi: float = np.pi,
) -> list:
    """{delta in [d_lo, d_hi] : wrap(value + coeff*delta) in [lo, hi]}, exactly.

    A joint angle is read modulo 2 pi, so the admissible set is a union of
    intervals rather than one interval: value + coeff*delta may land in
    [lo, hi] + 2 pi k for several k. Returned as a sorted list of disjoint
    closed intervals. No sampling: the endpoints are the preimages of lo and
    hi under an affine map.
    """
    if abs(coeff) < 1e-15:
        v = _wrap_angle(value)
        return [(d_lo, d_hi)] if lo <= v <= hi else []
    out = []
    k_max = int(np.ceil((abs(coeff) * (d_hi - d_lo) + 2.0 * np.pi) / (2.0 * np.pi))) + 2
    for k in range(-k_max, k_max + 1):
        a = (lo + 2.0 * np.pi * k - value) / coeff
        b = (hi + 2.0 * np.pi * k - value) / coeff
        a, b = min(a, b), max(a, b)
        a, b = max(a, d_lo), min(b, d_hi)
        if a <= b:
            out.append((float(a), float(b)))
    return _merge_intervals(out)


def _wrap_angle(a: float) -> float:
    return float(-((-a + np.pi) % (2.0 * np.pi) - np.pi))


def _merge_intervals(ivs: list) -> list:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1] + 1e-15:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersect_intervals(x: list, y: list) -> list:
    out = []
    i = j = 0
    while i < len(x) and j < len(y):
        a = max(x[i][0], y[j][0])
        b = min(x[i][1], y[j][1])
        if a <= b:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


@dataclass
class FoldCertificate:
    """The exact state of a q4 fold (line) door."""

    y: float
    open: bool
    exact: bool                    # False -> fell back to sampling, say so
    collision_free: bool
    witness: str
    posture_drift_m: float         # max link motion over theta AND the fiber
    pages: dict = field(default_factory=dict)   # page -> per-page verdict
    reason: str = ""


def certify_fold_door(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    disc: SigmaDisc, pages: tuple | None = None, n_drift: int = 181,
) -> FoldCertificate:
    """Decide a fold door EXACTLY: one collision query + interval intersection.

    `pages` lists the sign branches the door is supposed to join; the door is
    OPEN if the posture is collision-free and at least one of those pages can
    spell it within its joint limits. Per invariant 2 an undecidable door is
    left OPEN, so every fallback below errs that way.
    """
    p = arm.p
    y_star = disc.exact_y if disc.exact_y is not None else disc.y
    pos, rot = weld.pose(float(y_star))

    ref = SIGN_BRANCHES[0]
    res = arm.ik_branch(pos, rot, 0.0, ref,
                        coord_axis_tol=cfg.pagemap.coord_axis_tol,
                        coord_radius_tol=cfg.pagemap.coord_radius_tol)
    if not res.ok:
        return FoldCertificate(float(y_star), True, False, True, "", 0.0, {},
                               "no IK at the fold; left OPEN (invariant 2)")
    q0 = res.q

    # -- is the posture really independent of theta and of the q3/q5 fiber? --
    # Asserted here rather than assumed, because the exactness of everything
    # below rests on it. Off-axis geometry hangs on frame 2 / frame 3, whose
    # ORIENTATION does vary along the fiber even though the four points do not,
    # so with those switches on the collision query is no longer single and we
    # say the certificate is not exact.
    base = arm.frames(q0)
    ref_pts = np.concatenate([base["shoulder"], base["elbow"],
                              base["wrist"], base["tool"]])
    drift = 0.0
    for th in np.linspace(-np.pi, np.pi, n_drift):
        r = arm.ik_branch(pos, rot, float(th), ref)
        if not r.ok:
            continue
        f = arm.frames(r.q)
        pts = np.concatenate([f["shoulder"], f["elbow"], f["wrist"], f["tool"]])
        drift = max(drift, float(np.max(np.abs(pts - ref_pts))))
    for dlt in np.linspace(-np.pi, np.pi, n_drift):
        q = q0.copy()
        q[2] += dlt
        q[4] -= dlt
        f = arm.frames(q)
        pts = np.concatenate([f["shoulder"], f["elbow"], f["wrist"], f["tool"]])
        drift = max(drift, float(np.max(np.abs(pts - ref_pts))))

    exact = not (p.use_off_axis_upper_arm or p.use_off_axis_elbow)

    # -- ONE collision query ------------------------------------------------
    rep = scene.check(base)
    if rep.in_collision:
        return FoldCertificate(
            float(y_star), False, exact, False, rep.witness, drift, {},
            "the fold posture itself is in collision -> door CERTIFIED BLOCKED "
            "(one exact query: the whole theta line is this one posture)")

    # -- joint limits, per page, by exact interval intersection -------------
    want = pages if pages else tuple(page_name(s) for s in SIGN_BRANCHES)
    verdicts = {}
    any_open = False
    for name in want:
        signs = tuple(1 if c == "+" else -1 for c in name)
        q = q_for_branch(q0, ref, signs)
        # the q3/q5 null motion, in this page's own spelling
        coeffs = np.zeros(7)
        coeffs[2], coeffs[4] = 1.0, -1.0
        admissible = [(-np.pi, np.pi)]
        for j in range(7):
            admissible = _intersect_intervals(
                admissible,
                _wrapped_admissible_deltas(float(q[j]), float(coeffs[j]),
                                           float(arm.q_min[j]), float(arm.q_max[j])))
            if not admissible:
                break
        ok = bool(admissible)
        # The margin is a REPORTING number, scanned inside the admissible set;
        # the door's state is the exact non-emptiness above, not this.
        best_margin, best_delta = -np.inf, 0.0
        for a, b in admissible:
            for dlt in np.linspace(a, b, 41):
                qq = q.copy()
                qq[2] = _wrap_angle(qq[2] + dlt)
                qq[4] = _wrap_angle(qq[4] - dlt)
                m = arm.limit_margin(qq)
                if m > best_margin:
                    best_margin, best_delta = m, float(dlt)
        verdicts[name] = {
            "within_limits": ok,
            "delta_intervals": [[round(a, 9), round(b, 9)] for a, b in admissible],
            "best_margin_rad": (float(best_margin) if ok else None),
            "best_margin_deg": (float(np.degrees(best_margin)) if ok else None),
            "best_delta_rad": (best_delta if ok else None),
            "q_deg": [round(float(v), 4) for v in np.degrees(q)],
        }
        any_open = any_open or ok

    return FoldCertificate(
        y=float(y_star), open=bool(any_open), exact=exact,
        collision_free=True, witness=rep.witness, posture_drift_m=drift,
        pages=verdicts,
        reason=("fold posture collision-free and spellable within limits -> "
                "door OPEN (exact: one collision query + interval "
                "intersection)" if any_open else
                "fold posture collision-free but NO page can spell it within "
                "limits -> door CERTIFIED BLOCKED (exact)"),
    )
