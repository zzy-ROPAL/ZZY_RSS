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
    radius: float
    # Which sign indices (0 -> s2, 1 -> s4, 2 -> s6) vanish inside this disc.
    vanishing: frozenset = field(default_factory=frozenset)
    pages: tuple = ()
    detail: str = ""

    def contains(self, y: float, theta: float) -> bool:
        return (y - self.y) ** 2 + (theta - self.theta) ** 2 <= self.radius ** 2


SIGN_INDEX = {0: "s2", 1: "s4", 2: "s6"}
JOINT_OF_SIGN = {0: 1, 1: 3, 2: 5}   # sign index -> joint index in q


def _cell_diag(pm: PageMap) -> float:
    dy = float(pm.y_grid[1] - pm.y_grid[0])
    dt = float(pm.theta_grid[1] - pm.theta_grid[0])
    return float(np.hypot(dy, dt))


def _disc_radius(cfg: Config, pm: PageMap) -> float:
    return max(cfg.sigma.disc_radius_floor, cfg.sigma.disc_radius_scale * _cell_diag(pm))


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
    radius = _disc_radius(cfg, pm)
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
                vanishing=frozenset({sign_index}),
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
    folded pose; at either the s4 branches merge and the page region ENDS, so
    there is no crossing to scan. Both conditions are quadratic in y.

    A fold is simultaneously Sigma_coord: the elbow circle collapses, the arm
    angle loses meaning, and the WHOLE theta-line at that y degenerates to one
    configuration. The disc cover therefore spans the full theta range there.
    """
    radius = _disc_radius(cfg, pm)
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
                for theta in np.linspace(pm.theta_grid[0], pm.theta_grid[-1],
                                         max(3, int(np.ceil(
                                             (pm.theta_grid[-1] - pm.theta_grid[0])
                                             / radius)))):
                    found.append(
                        SigmaDisc(
                            kind=DOOR_FOLD,
                            y=float(min(max(y, 0.0), 1.0)),
                            theta=float(theta),
                            radius=radius,
                            vanishing=frozenset({1}),
                            detail=f"q4 fold ({what}, analytic) on page {pm.name}",
                        )
                    )

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
                vanishing=frozenset({1}),
                detail=f"q4 fold (grid) on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def find_coord_discs(cfg: Config, pm: PageMap) -> list[SigmaDisc]:
    """Arm-angle reference breakdown: Sigma_coord."""
    radius = _disc_radius(cfg, pm)
    found = []
    for i, j in np.argwhere(pm.near_axis):
        found.append(
            SigmaDisc(
                kind=SIGMA_COORD,
                y=float(pm.y_grid[i]),
                theta=float(pm.theta_grid[j]),
                radius=radius,
                vanishing=frozenset(),
                detail=f"arm-angle reference degenerate on page {pm.name}",
            )
        )
    return _merge_discs(found, radius)


def find_kin_discs(cfg: Config, pm: PageMap) -> list[SigmaDisc]:
    """det(J J^T) zeros: Sigma_kin. RECORDED ONLY -- never invalidates a cell."""
    radius = _disc_radius(cfg, pm)
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
            if (d.y - e.y) ** 2 + (d.theta - e.theta) ** 2 <= radius ** 2:
                out[k] = SigmaDisc(
                    kind=e.kind if e.kind == d.kind else f"{e.kind}+{d.kind}",
                    y=0.5 * (e.y + d.y),
                    theta=0.5 * (e.theta + d.theta),
                    radius=max(e.radius, d.radius),
                    vanishing=e.vanishing | d.vanishing,
                    pages=e.pages,
                    detail=e.detail,
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
                vanishing=d.vanishing, pages=assign_pages(d, pm.signs),
                detail=d.detail,
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

    pos, rot = weld.pose(disc.y)
    res = arm.ik_branch(pos, rot, disc.theta, page)
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
    pos, rot = weld.pose(disc.y)
    res = arm.ik_branch(pos, rot, disc.theta, page)
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
