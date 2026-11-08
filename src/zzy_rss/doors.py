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

from .arm import SIGN_BRANCHES, VALID, SrsArm
from .params import Config
from .pagemap import PageMap, page_name
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
    """Roots of a y^2 + b y + c = 0 that lie in [0, 1]."""
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
    radius: float | None = None, r_th: float | None = None,
    name: str | None = None,
) -> list[SigmaDisc]:
    """Exact interior doors from the geometric condition.

    Within a page the sign is FIXED by construction -- the IK returns
    q_j = s_j * arccos(.), which is never negative on an s_j=+ page -- so q_j
    can only TOUCH zero, never cross it. A sign-crossing scan therefore finds
    nothing. The door is located instead from its geometry:

      q2 = 0  pins the elbow at E* = S + d_se * e_z, so the wrist must satisfy
              |W(y) - E*| = d_ew. W(y) is affine in y, so this is a quadratic
              with at most two roots -- the "at most two q2 doors for a
              straight weld" bound, derived rather than scanned.

      q6 = 0  aligns the forearm with the tool axis, so the elbow is pinned at
              E(y) = W(y) - d_ew * (R e_z) and must satisfy |E(y) - S| = d_se.
              Again affine in y, hence at most two roots.

    theta at the door follows by inverting the elbow-circle parameterisation.
    """
    p = arm.p
    e_z = np.array([0.0, 0.0, 1.0])
    # pm is used ONLY for the cover radius and the page name -- the door
    # LOCATIONS are a quadratic. Passing them explicitly makes this finder
    # grid-free (findings 38b).
    radius = _disc_radius(cfg, pm) if radius is None else radius
    r_th = _disc_radius_theta(cfg, pm) if r_th is None else r_th
    name = pm.name if name is None else name
    p0 = np.asarray(cfg.weld.p0, dtype=float)
    p1 = np.asarray(cfg.weld.p1, dtype=float)
    rot = weld.rot
    tool_axis = rot @ e_z

    # Wrist centre is affine in y: W(y) = w0 + y * wd
    w0 = p0 - p.d_wt * tool_axis
    wd = p1 - p0

    if sign_index == 0:          # q2 door
        target = arm.shoulder + p.d_se * e_z     # E*
        base, radius_eq = w0, p.d_ew
        joint = 2
    elif sign_index == 2:        # q6 door
        target = arm.shoulder
        base, radius_eq = w0 - p.d_ew * tool_axis, p.d_se
        joint = 6
    else:
        return []

    delta = base - target
    a = float(np.dot(wd, wd))
    b = 2.0 * float(np.dot(delta, wd))
    c = float(np.dot(delta, delta)) - radius_eq ** 2
    roots = _quadratic_roots_on_unit(a, b, c)

    found = []
    for y in roots:
        y = min(max(y, 0.0), 1.0)
        if sign_index == 0:
            elbow_target = target
        else:
            elbow_target = base + y * wd
        theta = _arm_angle_of_elbow(arm, weld, y, elbow_target)
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
                detail=f"q{joint}=0 door (analytic) on page {name}",
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


def find_fold_doors(cfg: Config, pm: PageMap, arm=None, weld=None,
                    radius: float | None = None, r_th: float | None = None,
                    name: str | None = None, theta_range: tuple | None = None,
                    grid_backstop: bool = True) -> list[SigmaDisc]:
    """q4 folds: where the IK discriminant vanishes, not where a sign crosses.

    |W(y) - S| = d_se + d_ew is full extension and |d_se - d_ew| is the fully
    folded pose; at either the s4 branches merge and the page region ENDS, so
    there is no crossing to scan. Both conditions are quadratic in y.

    A fold is simultaneously Sigma_coord: the elbow circle collapses, the arm
    angle loses meaning, and the WHOLE theta-line at that y degenerates to one
    configuration. The disc cover therefore spans the full theta range there.
    """
    radius = _disc_radius(cfg, pm) if radius is None else radius
    r_th = _disc_radius_theta(cfg, pm) if r_th is None else r_th
    name = pm.name if name is None else name
    th_lo, th_hi = (theta_range if theta_range is not None
                    else (float(pm.theta_grid[0]), float(pm.theta_grid[-1])))
    found: list[SigmaDisc] = []

    if arm is not None and weld is not None:
        p = arm.p
        e_z = np.array([0.0, 0.0, 1.0])
        p0 = np.asarray(cfg.weld.p0, dtype=float)
        p1 = np.asarray(cfg.weld.p1, dtype=float)
        w0 = p0 - p.d_wt * (weld.rot @ e_z)
        wd = p1 - p0
        delta = w0 - arm.shoulder
        a = float(np.dot(wd, wd))
        b = 2.0 * float(np.dot(delta, wd))
        for r_eq, what in ((p.reach_max, "full extension"), (p.reach_min, "full fold")):
            if r_eq <= 1e-9:
                continue
            c = float(np.dot(delta, delta)) - r_eq ** 2
            for y in _quadratic_roots_on_unit(a, b, c):
                # theta degenerates: cover the whole theta line at this y.
                for theta in np.linspace(th_lo, th_hi,
                                         max(3, int(np.ceil(
                                             (th_hi - th_lo) / radius)))):
                    found.append(
                        SigmaDisc(
                            kind=DOOR_FOLD,
                            y=float(min(max(y, 0.0), 1.0)),
                            theta=float(theta),
                            radius=radius,
                radius_theta=r_th,
                            vanishing=frozenset({1}),
                            detail=f"q4 fold ({what}, analytic) on page {name}",
                        )
                    )

    # Numerical backstop: cells the grid flags as collapsed. The roadmap
    # build supplies its own (degenerate_discs_on_vertices) and switches this
    # off, because there is no grid to scan.
    if not grid_backstop or pm is None:
        return _merge_discs(found, radius)
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
                detail=f"q4 fold (grid) on page {name}",
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
    n = n or cfg.sigma_policy.fiber_samples
    anchor_y, anchor_theta = disc.anchor
    pos, rot = weld.pose(anchor_y)
    res = arm.ik_branch(pos, rot, anchor_theta, page)
    if not res.ok:
        return {"ok": False, "reason": "no IK at door", "valid_fraction": 0.0,
                "intervals": []}

    if 0 in disc.vanishing:
        pair = (0, 2)
    elif 2 in disc.vanishing:
        pair = (4, 6)
    else:
        return {"ok": False, "reason": "fiber not characterised", "valid_fraction": 0.0,
                "intervals": []}

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
# Exact fiber certification -- item #0
#
# The old question was "does ANY delta on the fiber work?", answered yes as
# soon as one sampled point was admissible. That is the wrong question. To
# change pages the robot must TRAVEL from this page's IK spelling (delta = 0)
# to the partner page's spelling (delta = +/- pi) without ever leaving the
# admissible set. Admissible points on both sides of a gap prove nothing --
# that is invariant 21 seen from the other end, where checking only the
# endpoint accepts a motion the joint cannot perform (mirror_q wraps q7 from
# -316.62 deg to +43.38 deg, inside the limits, but the joint cannot turn
# there).
#
# The answer is available in CLOSED FORM, with no sampling at all:
#
#   * the robot does not move along the fiber (gimbal null motion), so
#     COLLISION IS CONSTANT  ->  one query decides it;
#   * every joint is constant or LINEAR in delta, and joint limits are box
#     constraints, so each gives an interval and the admissible set is their
#     INTERSECTION.
#
# Measured against the sampled sweep this reproduces the interval exactly:
# v5b's [0.0199927, 3.1216] is [offset, pi - offset] with offset = the URDF's
# 0.02 rad safety margin, and EXP 3's q2 door misses delta = pi by the same
# 1.15 deg. 721 samples were recovering two linear inequalities.
# --------------------------------------------------------------------------


@dataclass
class FiberCertificate:
    """Exact admissible delta-interval on a door's self-motion fiber."""

    pair: tuple | None = None      # counter-rotating joint indices (i, k)
    lo: float = 0.0                # admissible delta interval [lo, hi]
    hi: float = 0.0
    empty: bool = True
    at_zero: bool = False          # is this page's own spelling admissible
    crossable: bool = False        # ONE interval holding both 0 and +/-pi
    direction: int = 0             # +1 traverse to +pi, -1 to -pi, 0 neither
    margin: float = 0.0            # rad of slack at the pi end; <0 = deficit
    exact: bool = False            # closed form applied (null motion held)
    drift_m: float = 0.0           # measured null-motion drift
    blocker: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "pair": list(self.pair) if self.pair else None,
            "interval": [self.lo, self.hi] if not self.empty else [],
            "empty": self.empty,
            "at_zero": self.at_zero,
            "crossable": self.crossable,
            "direction": self.direction,
            "margin_rad": self.margin,
            "margin_deg": float(np.degrees(self.margin)),
            "exact": self.exact,
            "drift_mm": self.drift_m * 1e3,
            "blocker": self.blocker,
            "reason": self.reason,
        }


def fiber_pair(disc: SigmaDisc) -> tuple | None:
    """The two collinear joints that counter-rotate along this door's fiber.

    A vanishing sign index names the joint that goes to zero (s2 -> q2,
    s4 -> q4, s6 -> q6, i.e. joint indices 1, 3, 5). At that zero the joint
    axes on either SIDE of it become collinear, so rotating them against each
    other moves nothing:

        q2 = 0  ->  (q1, q3)   axes 0 and 2
        q4 = 0  ->  (q3, q5)   axes 2 and 4     <- was missing before
        q6 = 0  ->  (q5, q7)   axes 4 and 6

    The q4 case used to fall through to "fiber not characterised", which
    returned ok=False -- and live_discs reads ok=False as CERTIFIED CLOSED,
    so a fold door nobody had evaluated let the wall pass straight through it.
    """
    if not disc.vanishing:
        return None
    idx = min(disc.vanishing)
    j = JOINT_OF_SIGN.get(idx)
    if j is None or j - 1 < 0 or j + 1 > 6:
        return None
    return (j - 1, j + 1)


def _capsule_geometry(arm: SrsArm, scene: Scene, q: np.ndarray) -> np.ndarray:
    """Flattened capsule endpoints, so the null-motion test sees ALL geometry.

    The skeleton alone is not enough. Along a q2 door's fiber frame 2 rotates
    (R02 = rot_z(q1 + delta)), and while that leaves every skeleton point
    fixed -- rot_z does not move the z axis, and R03 = rot_z(q1 + q3) is
    constant -- it DOES carry an off-axis upper-arm hub around with it. With
    that geometry switched on, collision is no longer constant along the
    fiber and the closed form does not apply. Measuring the capsules rather
    than assuming the null motion is what catches that.
    """
    caps = scene.capsules(arm.frames(q))
    return np.concatenate([np.concatenate([c[0], c[1]]) for c in caps])


def _null_motion_drift(cfg: Config, arm: SrsArm, scene: Scene,
                       q0: np.ndarray, pair: tuple) -> float:
    """Largest capsule-endpoint displacement over a few probe deltas."""
    i, k = pair
    base = _capsule_geometry(arm, scene, q0)
    worst = 0.0
    for d in (0.37, 1.1, -0.8, np.pi):
        q = q0.copy()
        q[i] += d
        q[k] -= d
        g = _capsule_geometry(arm, scene, q)
        worst = max(worst, float(np.abs(g - base).max()))
    return worst


def certify_fiber(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    disc: SigmaDisc, page: tuple[int, int, int],
) -> FiberCertificate:
    """Closed-form admissible delta-interval, and whether the door is crossable.

    Returns exact=False when the fiber is not a null motion; the caller must
    then fall back to the sampled sweep and may NOT treat the result as a
    certificate.
    """
    pair = fiber_pair(disc)
    if pair is None:
        return FiberCertificate(reason="not a sign door; no self-motion fiber")

    anchor_y, anchor_theta = disc.anchor
    pos, rot = weld.pose(anchor_y)
    res = arm.ik_branch(pos, rot, anchor_theta, page)
    if not res.ok:
        return FiberCertificate(pair=pair, reason="no IK at the door anchor")

    q0 = res.q
    i, k = pair

    # (0) REDUNDANCY GUARD. The closed form describes the 1-D counter-rotating
    # slice (q_i + delta, q_k - delta). That is the WHOLE fiber only when the
    # self-motion manifold is 1-dimensional, i.e. the task fixes the full pose
    # of a 7-DOF arm. Raising the TASK dimension is fine -- a 2-D surface task
    # ([IROS] d=3) still leaves exactly the arm angle free, so the page grows
    # to (u, v, theta) while the fiber stays 1-D. Raising the REDUNDANCY is
    # not: relaxing the torch's spin about its own axis, say, gives a 2-D
    # self-motion set, and then "no connected delta interval" does NOT prove
    # the pages cannot be joined -- a route may exist through the directions
    # this slice does not see.
    #
    # That failure is the dangerous direction: it would report
    # certified_blocked, live_discs would discharge the disc, the door would
    # be skipped, and the result could be a FALSE INFEASIBILITY PROOF. The
    # null-motion test below cannot catch it, because the slice is still a
    # null motion and its drift is still zero.
    #
    # Measured off the degeneracy, since ON it the Jacobian is rank-deficient
    # by construction (that is what makes it a door).
    q_gen = q0.copy()
    q_gen[(i + k) // 2] += 1e-3        # nudge the vanishing joint off zero
    n_self = int(q0.size - np.linalg.matrix_rank(arm.jacobian(q_gen), tol=1e-6))
    if n_self != 1:
        return FiberCertificate(
            pair=pair, exact=False,
            reason=(f"self-motion manifold is {n_self}-dimensional, so the "
                    f"counter-rotating slice is not the whole fiber; the "
                    f"closed form would not be conclusive"))

    # (1) Is this actually a null motion? Verify, never assume.
    drift = _null_motion_drift(cfg, arm, scene, q0, pair)
    if drift > cfg.crossing.null_motion_tol_m:
        return FiberCertificate(
            pair=pair, exact=False, drift_m=drift,
            reason=(f"fiber moves the robot ({drift * 1e3:.3f} mm): collision is "
                    f"not constant, closed form does not apply"))

    # (2) Collision is constant along the fiber -- ONE query decides it.
    rep = scene.check(arm.frames(q0))
    if rep.in_collision:
        return FiberCertificate(
            pair=pair, exact=True, drift_m=drift, empty=True,
            blocker=f"collision:{rep.witness}",
            reason="whole fiber in collision -> door CERTIFIED BLOCKED")

    # (3) Joint limits. Constant joints decide the fiber outright; the two
    #     moving joints are linear in delta, so each gives an interval.
    lo, hi = -np.inf, np.inf
    for j in range(len(q0)):
        if j == i or j == k:
            continue
        if not (arm.q_min[j] <= q0[j] <= arm.q_max[j]):
            return FiberCertificate(
                pair=pair, exact=True, drift_m=drift, empty=True,
                blocker=f"joint-limit:q{j + 1}",
                reason=(f"q{j + 1} is constant along the fiber and out of "
                        f"limits -> door CERTIFIED BLOCKED"))
    lo = max(lo, float(arm.q_min[i] - q0[i]), float(q0[k] - arm.q_max[k]))
    hi = min(hi, float(arm.q_max[i] - q0[i]), float(q0[k] - arm.q_min[k]))

    if lo > hi:
        return FiberCertificate(
            pair=pair, exact=True, drift_m=drift, empty=True,
            blocker="joint-limit", lo=lo, hi=hi,
            reason="no admissible delta -> door CERTIFIED BLOCKED")

    # (4) Crossing needs ONE connected interval holding this page's spelling
    #     (delta = 0) and the partner's (delta = +/- pi). Both directions
    #     reach the mirror; they differ by 2*pi on one joint, which is a
    #     different JOINT VALUE even though it is the same posture, so each
    #     is limit-checked on the values actually traversed.
    at_zero = bool(lo <= 0.0 <= hi)
    slack_pos = float(hi - np.pi)
    slack_neg = float(-np.pi - lo)
    margin = max(slack_pos, slack_neg)
    direction = 0
    if at_zero and slack_pos >= 0.0:
        direction = +1
    elif at_zero and slack_neg >= 0.0:
        direction = -1

    if not cfg.crossing.require_connected:
        # Old semantics: any admissible point counts. Kept only so the
        # regression can measure the difference.
        return FiberCertificate(
            pair=pair, lo=lo, hi=hi, empty=False, at_zero=at_zero,
            crossable=True, direction=direction, margin=margin, exact=True,
            drift_m=drift, reason="fiber has admissible points (legacy test)")

    if not at_zero:
        return FiberCertificate(
            pair=pair, lo=lo, hi=hi, empty=False, at_zero=False,
            margin=margin, exact=True, drift_m=drift, blocker="joint-limit",
            reason=(f"this page's own spelling (delta=0) is outside the "
                    f"admissible interval [{lo:.6f}, {hi:.6f}] -> NOT crossable"))

    if direction == 0:
        return FiberCertificate(
            pair=pair, lo=lo, hi=hi, empty=False, at_zero=True,
            margin=margin, exact=True, drift_m=drift, blocker="joint-limit",
            reason=(f"admissible interval [{lo:.6f}, {hi:.6f}] reaches neither "
                    f"+pi nor -pi; short by {abs(margin):.6f} rad "
                    f"({abs(np.degrees(margin)):.2f} deg) -> NOT crossable"))

    return FiberCertificate(
        pair=pair, lo=lo, hi=hi, empty=False, at_zero=True, crossable=True,
        direction=direction, margin=margin, exact=True, drift_m=drift,
        reason=(f"connected admissible interval [{lo:.6f}, {hi:.6f}] holds "
                f"delta=0 and delta={direction:+d}*pi with "
                f"{np.degrees(margin):.2f} deg to spare -> CROSSABLE"))


def partner_spelling_check(
    arm: SrsArm, weld: WeldLine, disc: SigmaDisc,
    page: tuple[int, int, int], partner: tuple[int, int, int],
    cert: FiberCertificate,
) -> dict:
    """Does the traversed endpoint agree with the partner page's own IK?

    They are the same POSTURE but need not be the same joint VALUES: the s2
    mirror is (q1 + pi, -q2, q3 + pi) while traversing to delta = +pi gives
    (q1 + pi, q3 - pi), which differs by 2*pi in q3. Both are legitimate
    spellings of one configuration; the traversed one is the one the robot
    actually performs, so limits are checked on THAT. This reports the
    agreement as a diagnostic, never as the limit test.
    """
    out = {"checked": False}
    if cert.pair is None or not cert.crossable:
        return out
    anchor_y, anchor_theta = disc.anchor
    pos, rot = weld.pose(anchor_y)
    here = arm.ik_branch(pos, rot, anchor_theta, page)
    there = arm.ik_branch(pos, rot, anchor_theta, partner)
    if not (here.ok and there.ok):
        return {"checked": True, "partner_ik": False}
    i, k = cert.pair
    q_end = here.q.copy()
    q_end[i] += cert.direction * np.pi
    q_end[k] -= cert.direction * np.pi
    diff = q_end - there.q
    wraps = np.round(diff / (2.0 * np.pi))
    residual = float(np.abs(diff - 2.0 * np.pi * wraps).max())
    return {
        "checked": True,
        "partner_ik": True,
        "matches_mod_2pi": bool(residual < 1e-9),
        "residual": residual,
        "wrapped_joints": [int(j) for j in np.nonzero(wraps)[0]],
        "q_end_deg": [round(float(v), 4) for v in np.degrees(q_end)],
        "partner_rep_deg": [round(float(v), 4) for v in np.degrees(there.q)],
        "traversed_within_limits": bool(arm.within_limits(q_end)),
    }


def door_passable(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    disc: SigmaDisc, page: tuple[int, int, int],
) -> dict:
    """Can the robot actually use this door to change pages?

    Three-valued, and the distinction is load-bearing:

      passable           -- it MIGHT be usable; follow it (conservative)
      certified_blocked  -- PROVED unusable; the disc may be discharged
      unknown            -- no decision; treat exactly like passable

    The old code collapsed the last two. `door_fiber_validity` returned
    ok=False both for "fiber wholly invalid" and for "no IK at door" / "fiber
    not characterised", and `live_discs` read ok=False as CERTIFIED CLOSED and
    discharged the disc -- letting the wall route straight through a door
    nobody had evaluated. Only a positive proof discharges anything here.
    """
    if cfg.crossing.exact:
        cert = certify_fiber(cfg, arm, weld, scene, disc, page)
        if cert.exact and not cert.crossable and (cert.empty or not cert.at_zero
                                                  or cert.direction == 0):
            return {"passable": False, "certified_blocked": True,
                    "unknown": False, "exact": True, "reason": cert.reason,
                    "certificate": cert.as_dict()}
        if cert.crossable:
            return {"passable": True, "certified_blocked": False,
                    "unknown": False, "exact": True, "reason": cert.reason,
                    "certificate": cert.as_dict()}
        # Not exact: the fiber moves the robot, or there is no IK here.
        return {"passable": True, "certified_blocked": False, "unknown": True,
                "exact": False, "reason": cert.reason or "undecided",
                "certificate": cert.as_dict()}

    fib = door_fiber_validity(cfg, arm, weld, scene, disc, page)
    characterised = fib["reason"].startswith("fiber has valid") or \
        fib["reason"].startswith("fiber wholly invalid")
    if not characterised:
        return {"passable": True, "certified_blocked": False, "unknown": True,
                "exact": False, "reason": fib["reason"], "certificate": fib}
    return {"passable": bool(fib["ok"]), "certified_blocked": not fib["ok"],
            "unknown": False, "exact": False, "reason": fib["reason"],
            "certificate": fib}


# --------------------------------------------------------------------------
# Grid-free Stage 2 -- findings 38b
#
# The analytic finders never needed the grid for anything but the disc radius
# and the page name: door LOCATIONS come from a quadratic. What did need it
# was the numerical cross-check behind them, and the Sigma_coord / Sigma_kin
# scans.
#
# Dropping the cross-check is NOT the conservative direction. Fewer doors
# means fewer escape routes followed, so a door that exists and is missed can
# turn an honest INCONCLUSIVE into a false INFEASIBLE. It has to be replaced,
# not deleted.
#
# The replacement is better suited to the job than the grid version was. A
# grid scan looked for local minima of |q_j| below a tolerance, which depends
# on the tolerance and misses a crossing that dips between two cells. A
# roadmap has EDGES, and an interior door is exactly a SIGN CHANGE of q_j
# across one -- so the detector becomes a sign test plus a bisection, with no
# tolerance in it at all. Fold doors keep the magnitude test, because at q4=0
# the branch ENDS rather than crossing (which is why they needed their own
# finder in the first place).
#
# What this costs: completeness is now probabilistic in the sampling, where
# the grid's was resolution-limited. Neither is exact. The analytic solve
# remains the primary finder in both.
# --------------------------------------------------------------------------


def _radius_from_spacing(cfg: Config, spacing: float) -> float:
    return max(cfg.sigma.disc_radius_floor, cfg.sigma.disc_radius_scale * spacing)


def _q_at(cfg: Config, arm: SrsArm, weld: WeldLine, y: float, theta: float,
          signs: tuple, joint: int):
    """One joint value at a page point, or None where the branch has no IK."""
    pos, rot = weld.pose(y)
    res = arm.ik_branch(pos, rot, theta, signs,
                        coord_axis_tol=cfg.pagemap.coord_axis_tol,
                        coord_radius_tol=cfg.pagemap.coord_radius_tol)
    return (float(res.q[joint]), res) if res.ok else (None, res)


def interior_doors_on_edges(
    cfg: Config, arm: SrsArm, weld: WeldLine, verts: np.ndarray, tree,
    signs: tuple, name: str, sign_index: int, spacing: float,
    k_neighbors: int = 8, n_bisect: int = 40,
) -> list[SigmaDisc]:
    """Backstop for q2/q6 doors: a SIGN CHANGE of q_j across a roadmap edge.

    No tolerance. Either the two endpoints disagree in sign or they do not,
    and when they disagree the zero is bracketed and bisection locates it to
    machine precision along that edge.
    """
    joint = JOINT_OF_SIGN[sign_index]
    radius = _radius_from_spacing(cfg, spacing)
    found: list[SigmaDisc] = []
    if len(verts) < 2:
        return found
    kk = min(k_neighbors + 1, len(verts))
    _d, nbr = tree.query(verts, k=kk)
    seen = set()
    for i in range(len(verts)):
        qi, _ = _q_at(cfg, arm, weld, float(verts[i][0]), float(verts[i][1]),
                      signs, joint)
        if qi is None:
            continue
        for j in np.atleast_1d(nbr[i])[1:]:
            j = int(j)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            qj, _ = _q_at(cfg, arm, weld, float(verts[j][0]), float(verts[j][1]),
                          signs, joint)
            if qj is None or np.sign(qi) == np.sign(qj):
                continue
            a, b = verts[i].copy(), verts[j].copy()
            qa = qi
            for _ in range(n_bisect):
                m = 0.5 * (a + b)
                qm, _ = _q_at(cfg, arm, weld, float(m[0]), float(m[1]),
                              signs, joint)
                if qm is None:
                    break
                if np.sign(qm) == np.sign(qa):
                    a, qa = m, qm
                else:
                    b = m
            m = 0.5 * (a + b)
            found.append(SigmaDisc(
                kind=DOOR_INTERIOR, y=float(m[0]), theta=float(m[1]),
                radius=radius, radius_theta=radius,
                vanishing=frozenset({sign_index}),
                detail=f"q{joint + 1} sign change on a roadmap edge, page {name}",
            ))
    return _merge_discs(found, radius)


def degenerate_discs_on_vertices(
    cfg: Config, arm: SrsArm, weld: WeldLine, verts: np.ndarray, signs: tuple,
    name: str, spacing: float, with_kin: bool = False,
) -> list[SigmaDisc]:
    """Fold backstop + Sigma_coord + Sigma_kin, evaluated at roadmap vertices.

    Folds keep a magnitude test because at q4 = 0 the branch ends rather than
    crossing, so there is no sign to watch.
    """
    radius = _radius_from_spacing(cfg, spacing)
    margin = cfg.sigma.fold_cos_margin
    found: list[SigmaDisc] = []
    for v in verts:
        y, th = float(v[0]), float(v[1])
        q4, res = _q_at(cfg, arm, weld, y, th, signs, 3)
        if res is None or not res.ok:
            continue
        if q4 is not None and (abs(q4) < margin or res.near_full_extension):
            found.append(SigmaDisc(
                kind=DOOR_FOLD, y=y, theta=th, radius=radius,
                radius_theta=radius, vanishing=frozenset({1}),
                detail=f"q4 fold at a roadmap vertex, page {name}"))
        if res.near_axis_align:
            found.append(SigmaDisc(
                kind=SIGMA_COORD, y=y, theta=th, radius=radius,
                radius_theta=radius,
                detail=f"arm-angle reference degenerate, page {name}"))
        if with_kin:
            m = arm.kin_singularity_measure(res.q)
            if np.isfinite(m) and abs(m) < cfg.sigma.kin_det_tol:
                found.append(SigmaDisc(
                    kind=SIGMA_KIN, y=y, theta=th, radius=radius,
                    radius_theta=radius,
                    detail=f"det(J J^T) ~ 0, page {name}"))
    return _merge_discs(found, radius)


def _analytic_free(cfg, arm, weld, signs, name, sign_index, spacing):
    r = _radius_from_spacing(cfg, spacing)
    return find_interior_doors_analytic(cfg, arm, weld, None, sign_index,
                                        radius=r, r_th=r, name=name)


def _fold_analytic_free(cfg, arm, weld, signs, name, spacing):
    r = _radius_from_spacing(cfg, spacing)
    return find_fold_doors(cfg, None, arm, weld, radius=r, r_th=r, name=name,
                           theta_range=(cfg.pagemap.theta_min,
                                        cfg.pagemap.theta_max),
                           grid_backstop=False)


def build_sigma_roadmap(
    cfg: Config, arm: SrsArm, weld: WeldLine, sources: dict,
    with_kin: bool = False, verbose: bool = True,
) -> dict:
    """Stage 2 with no PageMap anywhere. Same union as build_sigma.

    `sources` is {page name: RoadmapSource}; its vertices and KD-tree stand in
    for the grid's cells and their neighbours.
    """
    if cfg.sigma_policy.sigma_kin_invalidates:
        raise ValueError(
            "sigma_kin_invalidates=True is unsound for the headline claim: it "
            "would prove infeasibility of a narrower problem than 'no plan "
            "exists'. Refusing to run.")

    sigma = {}
    for name, src in sources.items():
        signs = src.signs
        spacing = src.tol()
        radius = _radius_from_spacing(cfg, spacing)
        discs: list[SigmaDisc] = []
        # Primary: the analytic solve. It never read the grid except for the
        # radius, so it transplants unchanged.
        for sign_index in (0, 2):
            discs += _analytic_free(cfg, arm, weld, signs, name, sign_index,
                                    spacing)
        discs += _fold_analytic_free(cfg, arm, weld, signs, name, spacing)
        # Backstops, now on the roadmap.
        if cfg.sigma.use_numeric_crosscheck:
            for sign_index in (0, 2):
                discs += interior_doors_on_edges(
                    cfg, arm, weld, src.valid, src._tree, signs, name,
                    sign_index, spacing,
                    k_neighbors=cfg.planner.k_neighbors)
        discs += degenerate_discs_on_vertices(
            cfg, arm, weld, src.valid, signs, name, spacing, with_kin=with_kin)

        discs = _merge_discs(discs, radius)
        discs = [
            SigmaDisc(
                kind=d.kind, y=d.y, theta=d.theta, radius=d.radius,
                radius_theta=d.r_theta, vanishing=d.vanishing,
                pages=assign_pages(d, signs), detail=d.detail,
                exact_y=d.exact_y, exact_theta=d.exact_theta,
            )
            for d in discs
        ]
        sigma[name] = discs
        if verbose:
            by_kind: dict = {}
            for d in discs:
                by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(by_kind.items())) \
                or "none"
            print(f"[stage2*] page {name}  Sigma discs: {len(discs):3d}  "
                  f"({summary})   [roadmap, spacing {spacing:.5f}]")
    return sigma
