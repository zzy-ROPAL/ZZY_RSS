"""Stage 4 -- flood fill over pages and doors, and the Stage 5 certificate.

This control layer has no IJRR analogue: IJRR has a single state space, while
here the planning space is a disjoint union of pages glued along Sigma.

Region labels come from connected components of the Stage 1 grid. That is a
ground-truth stand-in for IJRR's PRM graph components, taken deliberately so
the flood fill and certificate plumbing can be validated independently of the
learner; the SVM walls are then trained on valid points drawn from those same
components. Labels are PAGE-RESTRICTED: using connectivity that crosses doors
would merge regions joined only via other pages, and then no intra-page wall
exists to learn.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

from . import budget as _budget
from .arm import SrsArm
from .params import Config
from .pagemap import PageMap, page_name
from .scene import Scene
from .doors import SigmaDisc, door_fiber_validity, door_passable
from .source import (
    GridSource, Region, RoadmapLabeller, RoadmapSource, eta_separate,
    grid_sources, roadmap_sources,
)
from .wall import (
    LABEL_TOPOLOGICAL, PageLabeller, assign_label,
    sigma_disc_samples,
    CheckReport, Polytope, check_polytope, reconstruct_wall, sample_wall,
    train_wall, virtual_boundary_samples,
)
from .weld import WeldLine
from .witness import LEG_DOOR, WitnessPath, door_leg, plan_in_page, verify_witness

# Three-way verdict. The pair of proof systems -- a wall for INFEASIBLE, a
# path for FEASIBLE -- are each SOUND when they succeed and SILENT when they
# fail, so there must be somewhere to put the silence. Collapsing it into
# either verdict is how "failed to find a path" becomes "no path exists".
FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CertificateEntry:
    page: str
    region: int
    n_facets: int
    n_passed: int
    closed: bool
    gamma: float
    alg2_rounds: int
    ok: bool
    note: str = ""
    # The goal-distance check (findings 47): the wall's largest y, and the
    # limit it had to stay below (None = check off).
    y_max: float | None = None
    goal_limit: float | None = None
    # Doors: entered through (whole disc in the seed class), promoted to the
    # seed class on evidence, and kept in the rest class.
    entry_doors: list = field(default_factory=list)
    seed_doors: list = field(default_factory=list)
    excluded_doors: list = field(default_factory=list)
    attempts: list = field(default_factory=list)


@dataclass
class DoorDecision:
    """How one door was resolved, and by which of the four exits.

    The exit matters even when two of them take the same action. Exits 3 and 4
    both follow the door; only exit 3 may contribute to a FEASIBLE verdict,
    because only exit 3 produced a proof that the robot can get there and
    through. Following on exit 4 is safe for the infeasibility side -- it
    over-estimates reachability, so a sealed queue proves something stronger
    than required -- but a route through it can never be called feasible.
    """

    page: str
    region: int
    y: float
    theta: float
    kind: str
    exit: str                 # "1-wall1" | "2-wall2" | "3-witness" | "4-unknown"
                              # | "0-uncrossable"
    followed: bool
    crossing_verified: bool
    reason: str = ""
    to: str = ""


@dataclass
class ProofResult:
    proven: bool
    reason: str
    regions_visited: list = field(default_factory=list)
    certificate: list = field(default_factory=list)
    doors_followed: list = field(default_factory=list)
    goal_reached_via: str = ""
    # Three-way verdict. `proven` is kept as INFEASIBLE for callers that
    # predate this, but it can no longer be read as "not proven => feasible".
    verdict: str = INCONCLUSIVE
    witness_path: dict | None = None
    witness_report: dict | None = None
    door_decisions: list = field(default_factory=list)
    second_walls: list = field(default_factory=list)


def label_regions(cfg: Config, pm: PageMap) -> tuple[np.ndarray, list]:
    """Connected components of the valid mask, 4-connectivity.

    theta is an ARM ANGLE, so the page is a cylinder, not a rectangle: the
    columns at theta = -pi and theta = +pi are adjacent. Labelling the grid
    flat splits regions that are actually joined around the seam. Measured on
    the real LWA 4D at the q6-door placement, four of the eight pages changed
    region count once the seam was joined (4 -> 2, 2 -> 1, 4 -> 2, 2 -> 1),
    which changes how many walls the proof needs and which regions the flood
    fill visits.

    Set PageMapParams.theta_periodic False when joint limits bound theta, in
    which case the page really is a rectangle (see the handoff, Stage 0).
    """
    lab, n = ndimage.label(pm.valid_mask)
    if getattr(cfg.pagemap, "theta_periodic", True) and n > 1:
        lab, n = _join_theta_seam(lab, n)
    regions = []
    for k in range(1, n + 1):
        mask = lab == k
        idx = np.argwhere(mask)
        ys = pm.y_grid[idx[:, 0]]
        regions.append(
            Region(
                page=pm.name, index=k, n_cells=int(mask.sum()),
                y_min=float(ys.min()), y_max=float(ys.max()),
                reaches_start=bool(ys.min() <= cfg.flood.start_y + cfg.flood.start_y_tol),
                reaches_goal=bool(ys.max() >= cfg.flood.goal_y - cfg.flood.goal_y_tol),
            )
        )
    return lab, regions



def _join_theta_seam(lab: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Merge components that touch across the periodic theta boundary."""
    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(lab.shape[0]):
        a, b = int(lab[i, 0]), int(lab[i, -1])
        if a > 0 and b > 0:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    out = np.zeros_like(lab)
    remap: dict = {}
    for k in range(1, n + 1):
        r = find(k)
        if r not in remap:
            remap[r] = len(remap) + 1
        out[lab == k] = remap[r]
    return out, len(remap)


def region_points(pm: PageMap, lab: np.ndarray, k: int) -> np.ndarray:
    idx = np.argwhere(lab == k)
    return np.column_stack([pm.y_grid[idx[:, 0]], pm.theta_grid[idx[:, 1]]])


def other_valid_points(pm: PageMap, lab: np.ndarray, k: int,
                       regions: list | None = None,
                       mode: str = "all_valid") -> np.ndarray:
    """The 'rest' class: what the wall has to keep on the far side.

    mode "all_valid" takes every valid cell outside region k. mode "goal_only"
    takes only the cells of components that REACH THE GOAL, which is IJRR's
    actual split -- Li trains the start component against the goal component,
    not against every other vertex on the page.

    The difference matters when a small component sits next to region k and
    reaches neither start nor goal. Including it forces the wall to thread a
    hair between the two, which drives gamma up and, worse, puts facets in
    valid cells so Algorithm 3 rejects them. Enclosing it instead costs the
    certificate nothing: it does not reach the goal, so a closed wall lying in
    C_obs with region k AND that component inside still proves the goal
    unreachable. Measured on EXP 3's partner page, where a 16-cell neighbour
    of region 2 pushed the separating gamma from 2 to 10 (findings 29).

    Falls back to "all_valid" on a page where nothing reaches the goal, since
    the rest class would otherwise be only the virtual boundary ring.
    """
    mask = pm.valid_mask & (lab != k)
    if mode == "goal_only" and regions:
        goal_k = [r.index for r in regions if r.reaches_goal]
        if goal_k:
            gm = np.zeros_like(lab, dtype=bool)
            for g in goal_k:
                gm |= (lab == g)
            mask = pm.valid_mask & gm & (lab != k)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return np.zeros((0, 2))
    return np.column_stack([pm.y_grid[idx[:, 0]], pm.theta_grid[idx[:, 1]]])


def invalid_points(pm: PageMap) -> np.ndarray:
    idx = np.argwhere(~pm.valid_mask)
    return np.column_stack([pm.y_grid[idx[:, 0]], pm.theta_grid[idx[:, 1]]])


def thin(pts: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= n_max:
        return pts
    sel = rng.choice(len(pts), n_max, replace=False)
    return pts[sel]


def discs_in_region(discs: list, pts: np.ndarray, tol: float) -> list:
    """Sigma discs lying inside a region (centre close to one of its cells)."""
    out = []
    if len(pts) == 0:
        return out
    for d in discs:
        c = np.array([d.y, d.theta])
        if np.min(np.linalg.norm(pts - c, axis=1)) <= max(tol, d.radius):
            out.append(d)
    return out



def live_discs(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
               signs: tuple, discs: list, verbose: bool = False) -> tuple:
    """Split Sigma discs into those a wall must avoid and those it may cross.

    The rule "a wall may not cross a Sigma disc" exists because a door MIGHT
    be passable. A door whose whole self-motion fiber is invalid carries no
    connectivity at all, so its disc constrains nothing and the wall is free
    to cross it. Discharging those discs is exactly what lets an infeasibility
    proof close inside a single page.

    Only applies when fiber certification is switched on; with it off every
    door is assumed open, which is the conservative default.

    CAVEAT: the fiber test is a dense SAMPLE of a measure-zero set, not a
    certificate. A proof that relies on a discharged disc is conditional on
    it; a proof that never needs one is not.
    """
    if not (cfg.crossing.exact or cfg.sigma_policy.enable_fiber_certification):
        return list(discs), []
    live, discharged = [], []
    for d in discs:
        if not d.vanishing:
            live.append(d)
            continue
        verdict = door_passable(cfg, arm, weld, scene, d, signs)
        # ONLY a positive proof discharges. "unknown" -- no IK here, or a
        # fiber that moves the robot so the closed form does not apply --
        # leaves the disc live. The old code discharged on ok=False, which
        # covered "not characterised" too, so the wall was free to route
        # through a door nobody had evaluated.
        (discharged if verdict["certified_blocked"] else live).append(d)
        if verbose:
            ay, _ = d.anchor
            state = ("BLOCKED, disc discharged" if verdict["certified_blocked"]
                     else ("UNDECIDED, disc stays live" if verdict["unknown"]
                           else "CROSSABLE, wall must avoid it"))
            print(f"[stage5-fiber] door y={ay:.4f}: {state} -- {verdict['reason']}")
    return live, discharged


# --------------------------------------------------------------------------
# doors as geometric objects of the wall (findings 47)
# --------------------------------------------------------------------------


def same_door(a: SigmaDisc, b: SigmaDisc, tol: float = 1.0e-6) -> bool:
    """One physical door seen from two pages. The anchor is the exact,
    page-independent door point (the posture is the same on both pages), so
    the anchors coincide; the discs' covering radii may not."""
    (ay, at), (by, bt) = a.anchor, b.anchor
    return abs(ay - by) <= tol and abs(at - bt) <= tol


def door_key(d: SigmaDisc) -> tuple:
    ay, at = d.anchor
    return (round(float(ay), 6), round(float(at), 6))


def door_disc_samples(d: SigmaDisc, n_ring: int = 16) -> np.ndarray:
    """Training points covering a WHOLE door disc: the centre and two
    elliptical rings (on the boundary and 1.5x outside it). Elliptical
    because y and theta differ in units (invariant 6). Used for the seed
    class (a door the wall must enclose) and the rest class (a door it
    tries to exclude) alike: a disc split between the classes would put the
    wall across its own door, which the Sigma guard then rejects."""
    c = np.array([d.y, d.theta], dtype=float)
    out = [c]
    for s in (1.0, 1.5):
        for a in np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False):
            out.append(c + s * np.array([d.radius * np.cos(a), d.r_theta * np.sin(a)]))
    return np.asarray(out, dtype=float)


def touching_regions(src, d: SigmaDisc) -> list:
    """Components of `src` that touch door `d` -- a TRAINING HINT only.

    On a roadmap: the component the injected anchor joined (regions_under)
    plus every component with a vertex inside the disc. Sampling can miss a
    component that touches the door between samples; that is why no proof
    step relies on this list (findings 47): a followed door goes WHOLE into
    the partner page's seed class, and the verified wall must enclose it, so
    every component touching it is inside whether it was listed here or not.
    """
    out = set(int(k) for k in src.regions_under(d.y, d.theta, d.radius))
    if getattr(src, "kind", "grid") == "roadmap" and len(src.valid):
        for p, c in zip(src.valid, src.comp):
            if d.contains(float(p[0]), float(p[1])):
                out.add(int(c))
    return sorted(out)


def door_reach_evidence(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                        signs: tuple, entry, d: SigmaDisc):
    """A VERIFIED in-page path from `entry` to door `d`'s anchor, or None.

    This is the evidence that moves a door from the rest class to the seed
    class (door_policy "exclude_first"): a door the robot provably reaches
    cannot be walled off, so trying to exclude it would only burn a wall.
    None is NOT a proof of unreachability -- the PRM is probabilistically
    complete -- and the door simply stays in the rest class, where the wall
    either excludes it (a proof) or fails (and the fallback promotes it).
    """
    if entry is None:
        return None
    pleg = plan_in_page(cfg, arm, weld, scene, signs, entry, target_yt=d.anchor)
    if pleg is None:
        return None
    rep = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[pleg]))
    return pleg if rep.ok else None


def goal_line_samples(cfg: Config) -> np.ndarray:
    """Rest-class points ON the goal band's inner edge, uniform in theta,
    valid or not (FloodParams.goal_line_samples): the whole geometric goal,
    not just the components the roadmap happened to sample."""
    n = int(getattr(cfg.flood, "goal_line_samples", 0))
    if n <= 0 or not getattr(cfg.flood, "goal_check", True):
        return np.zeros((0, 2))
    pm = cfg.pagemap
    th = np.linspace(pm.theta_min, pm.theta_max, n)
    return np.column_stack([np.full(n, cfg.flood.goal_y - cfg.flood.goal_y_tol), th])


def wall_y_max(poly: Polytope) -> float:
    """Largest y over the wall. A d = 2 facet is a straight segment, so y is
    linear along it and the maximum over the whole wall is attained at a
    vertex: checking the vertices is EXACT, not a sample."""
    ys = [float(np.max(np.asarray(lp, dtype=float)[:, 0])) for lp in poly.loops if len(lp)]
    return max(ys) if ys else float("-inf")


def goal_feedback_points(poly: Polytope, y_limit: float, y_goal: float,
                         n_max: int = 24) -> np.ndarray:
    """The goal-distance check's feedback (findings 47, item 4): every wall
    vertex at y >= y_limit, projected onto the goal band's inner edge
    (y_goal, theta). These go into the REST class -- they ARE goal states
    (or obstacle states on the goal band), so labelling them is not a guess.
    """
    pts = []
    for lp in poly.loops:
        v = np.asarray(lp, dtype=float)
        hot = v[v[:, 0] >= y_limit]
        for th in hot[:, 1]:
            pts.append((y_goal, float(th)))
    if not pts:
        return np.zeros((0, 2))
    pts = np.unique(np.round(np.asarray(pts), 6), axis=0)
    if len(pts) > n_max:
        pts = pts[np.linspace(0, len(pts) - 1, n_max).astype(int)]
    return pts


def find_inf_proof_region(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    seed_pts: np.ndarray, rest_pts: np.ndarray, inv_pts: np.ndarray,
    discs: list, max_iters: int = 4, verbose: bool = True, labeller=None,
    goal_y_limit: float | None = None, goal_y_feed: float | None = None,
):
    """IJRR Algorithm 4: the loop that ties Algorithms 1-3 together.

    Algorithm 2 alone is not enough. It samples the wall only where INVALID
    seeds project onto it, so stretches of wall it never touches can still lie
    in the valid region -- observed directly in this implementation: Algorithm
    2 converged with 110 wall points all invalid, and Algorithm 3 then found
    22 of 61 facet centres VALID. IJRR's answer is this outer loop: a facet
    that fails because it may be in the free region is fed back into the
    training data, and the retrain is what pushes the remaining wall in.

    Failure routing follows IJRR Algorithm 4 lines 6-18:
      reconstruction not closed    -> densify, retry
      facet failed, centre valid   -> add it to the training data, retrain
      facet failed otherwise       -> shrink facets (generate-mpoints branch)
    plus the goal-distance check (findings 47), when goal_y_limit is given:
      every facet passed, but the wall reaches y >= goal_y_limit
                                   -> feed the offending vertices, projected
                                      onto the goal band (goal_y_feed), into
                                      the REST class and retrain
    The returned report is ok only when Algorithm 3 AND the goal check pass.
    """
    from .wall import train_wall

    wall = train_wall(cfg, seed_pts, rest_pts)
    if verbose:
        print(f"[stage3] alg1 gamma={wall.gamma:.2f} accuracy={wall.accuracy:.4f} "
              f"(seed {wall.n_seed}, rest {wall.n_rest})")

    poly = Polytope()
    rep = CheckReport(False, 0, 0)
    tc_ok = False
    alg2_total = 0

    for it in range(1, max_iters + 1):
        _budget.check_time()
        wall, wall_pts, seed_pts, rest_pts, rounds = sample_wall(
            cfg, arm, weld, scene, signs, wall, inv_pts, seed_pts, rest_pts,
            verbose=False, labeller=labeller,
        )
        alg2_total += rounds

        if not tc_ok:
            poly = reconstruct_wall(cfg, wall, wall_pts=wall_pts,
                                    seed_pts=seed_pts, rest_pts=rest_pts)
            tc_ok = poly.closed and bool(poly.loops)
        if not tc_ok:
            # Retry finer: a denser net for the TDC (half the spacing), a finer
            # grid for the legacy contour tracer.
            poly = reconstruct_wall(cfg, wall, grid_res=361, wall_pts=wall_pts,
                                    seed_pts=seed_pts, rest_pts=rest_pts,
                                    refine=1)
            tc_ok = poly.closed and bool(poly.loops)
            if not tc_ok:
                if verbose:
                    print(f"[stage3] alg4 iter {it}: wall not closed "
                          f"({poly.note or 'open loop'})")
                _budget.note_unclosed(poly.note or "open loop")
                continue
        _budget.note_closed()

        rep = check_polytope(cfg, arm, weld, scene, signs, poly, discs,
                             seed_pts, verbose=False)
        if verbose:
            print(f"[stage3] alg4 iter {it}: facets {rep.n_passed}/{rep.n_facets} "
                  f"passed (max recursion {rep.max_recursion})")
        if rep.ok and goal_y_limit is not None:
            y_hi = wall_y_max(poly)
            if y_hi >= goal_y_limit:
                why = (f"wall reaches y={y_hi:.4f} >= {goal_y_limit:.4f}: the goal "
                       f"band is not provably outside")
                rep = CheckReport(False, rep.n_facets, rep.n_passed,
                                  failures=[(-1, why)], max_recursion=rep.max_recursion)
                feed = goal_feedback_points(
                    poly, goal_y_limit,
                    goal_y_feed if goal_y_feed is not None else goal_y_limit)
                if len(feed):
                    rest_pts = np.vstack([rest_pts, feed])
                    wall = train_wall(cfg, seed_pts, rest_pts)
                    tc_ok = False
                if verbose:
                    print(f"[stage3] alg4 iter {it}: {why} -> fed {len(feed)} goal-band "
                          f"point(s) into the rest class, retrain (gamma={wall.gamma:.2f})")
                continue
        if rep.ok:
            return poly, rep, wall, it, alg2_total

        valid_centres = [c for c, why in rep.failure_points if "VALID" in why]
        if valid_centres:
            add = np.asarray(valid_centres, dtype=float).reshape(-1, 2)
            lb = labeller if cfg.wall.label_mode == LABEL_TOPOLOGICAL else None
            margin = float(cfg.wall.feedback_margin)
            n_fed = n_skipped = 0
            for m in add:
                lab_m = assign_label(m, seed_pts, rest_pts, wall.scale, lb)
                # Refuse a centre that contradicts the class it is NOT given:
                # adding it would make the training data inseparable below the
                # fragmentation limit, and Algorithm 1 would then either cap
                # out (accuracy < 1, so the wall no longer separates its own
                # data) or run gamma up until the zero set breaks into islands.
                # IJRR Algorithm 4 has a branch for a facet that cannot be
                # fixed by retraining -- shrink the facets -- so use it.
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
                n_fed += 1
            if n_fed:
                wall = train_wall(cfg, seed_pts, rest_pts)
                tc_ok = False
                if verbose:
                    print(f"[stage3] alg4 iter {it}: fed {n_fed} valid facet "
                          f"centre(s) back into training"
                          + (f", skipped {n_skipped} within the "
                             f"{margin:g} margin" if n_skipped else "")
                          + f" -> retrain (gamma={wall.gamma:.2f})")
                continue
            if verbose:
                print(f"[stage3] alg4 iter {it}: all {n_skipped} failed facet "
                      f"centre(s) contradict the training data -> shrinking "
                      f"facets instead of poisoning the fit")

        cfg = dataclasses.replace(
            cfg,
            wall=dataclasses.replace(
                cfg.wall, net_d_min=max(0.002, cfg.wall.net_d_min * 0.5)),
        )
        tc_ok = False
        if verbose:
            print(f"[stage3] alg4 iter {it}: shrinking facets "
                  f"(d_min -> {cfg.wall.net_d_min:.4f})")

    return poly, rep, wall, max_iters, alg2_total


def budgeted_alg4(cfg: Config, label: str, verbose: bool = True, **kw) -> tuple:
    """find_inf_proof_region under a wall budget (findings 44).

    A stopped attempt comes back as a FAILED wall -- open polytope, a check
    report with the stop as its only failure, the gamma it had reached -- so
    every caller treats it exactly like a wall that did not verify: wall 1's
    region stays in the certificate with ok=False (the run can only be
    INCONCLUSIVE), and a stopped second wall is "no separating wall", so the
    door is followed (exit 4)."""
    from types import SimpleNamespace
    try:
        with _budget.wall_attempt(cfg.budget, label):
            return find_inf_proof_region(cfg, verbose=verbose, **kw)
    except _budget.BudgetExceeded as e:
        if e.scope != "wall":
            raise
        if verbose:
            print(f"[budget] {e.reason}")
        g = float("nan") if e.gamma is None else float(e.gamma)
        return (Polytope(note=e.reason), CheckReport(False, 0, 0, failures=[(-1, e.reason)]),
                SimpleNamespace(gamma=g), 0, 0)


def _default_pages() -> dict:
    """page name -> sign branch, for building sources without any PageMap."""
    from .arm import SIGN_BRANCHES
    return {page_name(sg): sg for sg in SIGN_BRANCHES}


def point_in_wall(cfg: Config, poly: Polytope, point) -> bool | None:
    """Is (y, theta) inside the verified wall? None when undecidable.

    Even-odd ray casting along +y at fixed theta. Valid only on a PLANAR page.
    With theta_periodic the page is a cylinder, where a wrapping loop has no
    interior in the plane and a chord joining two points that are the SAME
    page point destroys the parity count -- so this refuses to answer rather
    than answer wrongly. Invariant 5 measures the current placements as
    exactly planar (no valid cell touches theta = +/-pi), which is a
    PRECONDITION to re-test whenever the scene changes.
    """
    if getattr(cfg.pagemap, "theta_periodic", True):
        return None
    if not poly.closed or not poly.loops:
        return None
    y, th = float(point[0]), float(point[1])
    inside = False
    for loop in poly.loops:
        v = np.asarray(loop, dtype=float)
        # DIMENSION GUARD. Even-odd ray casting over (y, theta) is a PLANAR
        # test. At d >= 3 the proof object is a triangulated surface, and this
        # loop would happily index columns 0 and 1 of 3-D vertices and return a
        # silently wrong boolean. That boolean drives exit 1: a wrong "outside"
        # skips a door the robot CAN reach, the flood terminates early, and the
        # result is a FALSE INFEASIBILITY PROOF. Refuse instead -- None means
        # undecided, which routes to exit 2/3/4 and ultimately to following the
        # door, which is always safe.
        if v.ndim != 2 or v.shape[1] != 2:
            return None
        if len(v) < 3:
            continue
        count = 0
        for a, b in zip(v[:-1], v[1:]):
            if (a[1] > th) == (b[1] > th):
                continue
            t = (th - a[1]) / (b[1] - a[1])
            if a[0] + t * (b[0] - a[0]) > y:
                count += 1
        if count % 2 == 1:
            inside = not inside
    return inside


def disc_ring(disc: SigmaDisc, n: int = 24, scale: float = 1.0) -> np.ndarray:
    """Points on a door's ELLIPTICAL boundary.

    Elliptical, not circular: y and theta differ by ~6x in units, and a
    circular ring of radius `disc.radius` would be a third of the page wide
    (invariant 6). This is the rest class for the second wall, so getting it
    wrong makes the wall detour around a region that is not there.
    """
    c = np.array([disc.y, disc.theta], dtype=float)
    out = [c + scale * np.array([disc.radius * np.cos(a),
                                 disc.r_theta * np.sin(a)])
           for a in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)]
    return np.asarray(out, dtype=float)


def second_wall(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    seed_pts: np.ndarray, inv_pts: np.ndarray, target: SigmaDisc,
    others: list, alg4_iters: int = 4, verbose: bool = False,
) -> tuple:
    """Exit 2: a wall whose only job is to separate the start from ONE door.

    Definition 1 says nothing about the second point being the goal; the proof
    of Proposition 1 works for any pair. Substituting the door gives

        closed  /\\  in C_obs  /\\  start inside  /\\  door outside
            =>  start and door lie in disconnected components of C_free
            =>  no path from the start to this door exists

    which is the strict proof a failed path search can never supply.

    Three things learned the hard way:

    1. `others` must EXCLUDE the target. The Sigma guard rejects any facet
       touching a disc, so leaving the target's own disc in rejects every
       facet of the wall that is trying to hug it. The target's RING goes into
       the rest class; its DISC stops being a guard.
    2. The rest class holds the door and the virtual ring ONLY -- not the goal
       component. One wall threading between the start and two targets at once
       is invariant 15 in reverse: gamma climbs and facet centres land in
       valid cells.
    3. Failing is safe and expected. A door genuinely reachable from the start
       has NO separating wall, so this must fail there; the caller then falls
       through to exits 3 and 4. It can never manufacture a proof.
    """
    ring = virtual_boundary_samples(cfg, 80)
    rest_pts = np.vstack([disc_ring(target, n=32), ring])
    poly, rep, wmodel, iters, rounds = budgeted_alg4(
        cfg, f"wall2 door y={target.y:.4f}", verbose=verbose,
        arm=arm, weld=weld, scene=scene, signs=signs, seed_pts=seed_pts,
        rest_pts=rest_pts, inv_pts=inv_pts, discs=others, max_iters=alg4_iters,
    )
    ok = bool(rep.ok and poly.closed and poly.loops)
    outside = False
    if ok:
        # Test the WHOLE disc, not its centre.
        #
        # For wall1 the centre would be enough: the Sigma guard is the first
        # check in Algorithm 3, so a VERIFIED wall cannot have a facet touching
        # a live disc, hence the disc lies wholly on one side and any point of
        # it speaks for all of it.
        #
        # wall2 has no such protection, because `others` deliberately drops the
        # target from the guard -- without that, every facet is rejected for
        # hugging the disc it is trying to exclude. So this wall MAY pass
        # through the disc, leaving part of the door inside and part outside.
        # A door is only excluded when ALL of it is excluded: one reachable
        # patch is a usable escape hatch, and skipping on a centre that
        # happened to fall outside would be a false infeasibility proof.
        probes = np.vstack([np.array([[target.y, target.theta]]),
                            disc_ring(target, n=32)])
        verdicts = [point_in_wall(cfg, poly, p) for p in probes]
        outside = bool(verdicts) and all(v is False for v in verdicts)
    return poly, rep, wmodel, bool(ok and outside)


def build_witness(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    route: list, goal_y: float, verbose: bool = False,
) -> tuple:
    """Splice the legs of a route into one end-to-end path.

    `route` is forward order: [(signs, entry_yt, disc, partner_signs), ...]
    with disc None on the final hop. Each hop contributes an in-page leg to
    the door anchor and then a delta-sweep across the door; the final hop
    contributes an in-page leg to the goal.

    Returns (WitnessPath, "") or (None, why). A failure here is INCONCLUSIVE,
    never a proof that no path exists -- PRM is probabilistically complete,
    so a missed narrow passage looks exactly like this.
    """
    legs = []
    # The START POSTURE on a partner page is NOT that page's IK representative
    # -- it is wherever the crossing actually left the robot. The two differ by
    # 2*pi on one joint: the s2 mirror is (q1 + pi, q3 + pi), while traversing
    # to delta = +pi lands on (q1 + pi, q3 - pi). Same posture, different joint
    # values, and the robot is holding the second one.
    #
    # So each page leg after a crossing is planned in that spelling: the offset
    # is carried into IK, and limits are checked on the carried values. Without
    # it the legs would not meet in Q and verification would (correctly, but
    # needlessly) reject a route that is actually executable. This is invariant
    # 21 applied to the continuation rather than to the crossing.
    offset = None
    for signs, entry_yt, disc, partner in route:
        if disc is None:
            leg = plan_in_page(cfg, arm, weld, scene, signs, entry_yt,
                               goal_y=goal_y, q_offset=offset)
            if leg is None:
                return None, (f"no in-page path on {page_name(signs)} from "
                              f"y={entry_yt[0]:.4f} to the goal")
            legs.append(leg)
            break
        leg = plan_in_page(cfg, arm, weld, scene, signs, entry_yt,
                           target_yt=disc.anchor, q_offset=offset)
        if leg is None:
            return None, (f"no in-page path on {page_name(signs)} from "
                          f"y={entry_yt[0]:.4f} to the door at "
                          f"y={disc.anchor[0]:.4f}")
        legs.append(leg)
        dleg, cert = door_leg(cfg, arm, weld, scene, disc, signs, partner,
                              q_offset=offset)
        if dleg is None:
            return None, (f"door at y={disc.anchor[0]:.4f} on "
                          f"{page_name(signs)} is not crossable: {cert.reason}")
        legs.append(dleg)
        # Where the crossing left us, minus what the partner page would call
        # this point. Rounded to whole turns: anything else would mean the two
        # are not the same posture, and the path is broken rather than wrapped.
        rep = arm.ik_branch(*weld.pose(disc.anchor[0]), disc.anchor[1], partner)
        if not rep.ok:
            return None, (f"partner page {page_name(partner)} has no IK at the "
                          f"door anchor y={disc.anchor[0]:.4f}")
        diff = dleg.q[-1] - rep.q
        turns = np.round(diff / (2.0 * np.pi))
        residual = float(np.abs(diff - 2.0 * np.pi * turns).max())
        if residual > 1e-9:
            return None, (f"the crossing endpoint is not the partner page's "
                          f"posture (residual {residual:.3e} rad)")
        offset = 2.0 * np.pi * turns
    return WitnessPath(legs=legs), ""


def run_proof(cfg: Config, *args, **kw) -> ProofResult:
    """_run_proof under cfg.budget (findings 44). A run that exhausts its
    budget is INCONCLUSIVE "stopped by budget: ..."; a wall that exhausts
    its own is a failed wall inside an otherwise normal run."""
    try:
        with _budget.run_budget(cfg.budget, "run_proof"):
            return _run_proof(cfg, *args, **kw)
    except _budget.BudgetExceeded as e:
        if kw.get("verbose", True):
            print(f"[budget] {e.reason}")
        return ProofResult(proven=False, reason=e.reason, verdict=INCONCLUSIVE)


def _run_proof(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    pages: dict | None, sigma: dict, n_train_max: int = 900,
    n_invalid_max: int = 160,
    verbose: bool = True, alg4_iters: int = 4, seeds: list | None = None,
    use_second_wall: bool = False, use_witness: bool = True,
    start_yt: tuple | None = None, sources: dict | None = None,
) -> ProofResult:
    """IJRR Algorithm 4 wrapped in the page/door flood fill.

    alg4_iters is the Algorithm 4 budget PER REGION. It is the knob that
    decides whether a wall gets pushed deep enough into C_obs to pass
    Algorithm 3, because a facet that fails is only fixed by feeding its
    centre back into the training data and retraining (invariant 11 --
    subdividing cannot help, pd is a property of where the wall sits). Four
    is enough for the easy regions and not enough for the ones that graze a
    valid band, so callers that want a verdict rather than a picture should
    raise it.

    What a wall must show (findings 47). For the region it is built for:

      closed, every facet in C_obs, no facet touching a live door disc
        (Algorithm 3, Sigma guard first)
      every ENTRY inside: the start region, or -- on a page entered through
        a door -- the WHOLE door disc (probe), so every component touching
        the door is inside whether the roadmap saw it or not
      the WHOLE goal band y >= goal_y - goal_y_tol outside (wall y_max below
        it, exact at the vertices) -- not just the goal components sampled
      every live door on the page probed: outside -> excluded, inside (or
        undecidable) -> followed

    Doors enter the training set as REST by default (door_policy
    "exclude_first") and move to SEED only on evidence; see FloodParams.
    use_second_wall (exit 2) is off by default: with doors already in the
    rest class, the one wall per region does that job.
    """
    rng = np.random.default_rng(cfg.planner.rng_seed)
    # The five questions go through a SOURCE. GridSource is the behaviour that
    # shipped; RoadmapSource answers the same five from a PRM with no cells at
    # all (findings 38). Pass `sources` to choose; the default stays the grid
    # so existing callers are untouched.
    if sources is None:
        if cfg.planner.page_source == "grid":
            if pages is None:
                raise ValueError("page_source='grid' needs `pages`")
            sources = grid_sources(cfg, pages)
        else:
            # Grid-free by default. Door anchors are injected as roadmap
            # vertices, which is what turns `regions_under` from a guess into
            # a lookup -- callers that want the anchors right should build
            # sources with bootstrap_roadmap and pass them in, so that Sigma
            # and the roadmap come from the same pass.
            sources = roadmap_sources(
                cfg, arm, weld, scene,
                pages if pages is not None else {
                    n: type("P", (), {"signs": sg})()
                    for n, sg in _default_pages().items()},
                sigma, n_samples=cfg.planner.source_samples,
                verbose=verbose)
    regions_by_page = {n: src.regions() for n, src in sources.items()}
    on_roadmap = any(getattr(s_, "kind", "grid") == "roadmap"
                     for s_ in sources.values())
    fp = cfg.flood
    goal_on = bool(getattr(fp, "goal_check", True))
    goal_edge = fp.goal_y - fp.goal_y_tol
    goal_limit = goal_edge - float(getattr(fp, "goal_margin", 0.0)) if goal_on else None
    policy = getattr(fp, "door_policy", "exclude_first")
    n_rounds = int(getattr(fp, "door_promotion_rounds", 1))

    # Seed the queue. By default: every region that touches the start of the
    # weld, on every page -- which proves the STRONGER claim that no posture
    # anywhere at y = 0 can finish. Pass `seeds` to prove the claim the task
    # actually makes: the robot starts in ONE posture, so the proof starts in
    # the one region holding it and reaches the other pages only through doors
    # it can actually cross. That is the same claim EXP 2's single wall makes,
    # and it does not ask the method to wall regions the robot can never be in.
    if seeds is not None:
        queue = [(n, k) for n, k in seeds if n in sources]
        if verbose:
            print(f"\n[stage4] seeding from the start posture's region(s) "
                  f"only: {queue}")
    else:
        queue = []
        for name, regs in regions_by_page.items():
            for r in regs:
                if r.reaches_start:
                    queue.append((name, r.index))
    if not queue:
        return ProofResult(True, "no region touches the start; trivially infeasible")

    if verbose and seeds is None:
        print(f"\n[stage4] seeding queue with {len(queue)} start-touching region(s): "
              f"{[(n, k) for n, k in queue]}")

    visited = set()
    certificate, visited_list, doors_followed = [], [], []
    decisions, second_walls = [], []
    # How each region was entered, so a route can be rebuilt when the goal is
    # reached: (prev_page, prev_region, disc, partner_signs, entry_yt,
    # crossing_verified). The seed regions have no parent.
    parent: dict = {}
    entry_point: dict = {}
    # The doors a node was entered through. Each goes WHOLE into that node's
    # seed class, and the wall must enclose it (findings 47).
    node_entry: dict = {}
    # Per page: the doors a VERIFIED wall there encloses (they were in its
    # Sigma guard, so wholly on one side, and the probe says inside). Every
    # component touching such a door is inside that wall, so following the
    # door onto this page again needs no new wall.
    walled: dict = {}

    def enclosed_on(page_name_: str, d: SigmaDisc) -> bool:
        return any(same_door(d, e) for e in walled.get(page_name_, []))

    for n_, k_ in queue:
        if start_yt is not None:
            entry_point[(n_, k_)] = tuple(start_yt)
        else:
            ep = sources[n_].entry_point(k_)
            if ep is not None:
                entry_point[(n_, k_)] = ep
    q_start_cfg = None
    if queue:
        n0, k0 = queue[0]
        yt0 = entry_point.get((n0, k0))
        if yt0 is not None:
            r0 = arm.ik_branch(*weld.pose(float(yt0[0])), float(yt0[1]),
                               sources[n0].signs)
            q_start_cfg = r0.q if r0.ok else None

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        name, k = node
        src = sources[name]
        pm = pages.get(name) if pages else None
        entries = list(node_entry.get(node, []))
        is_comp = isinstance(k, (int, np.integer))
        if is_comp:
            reg = next(r for r in regions_by_page[name] if r.index == k)
            pts = src.seed_pool(k)
            k_lab = int(k)
        else:
            # Entered through a door that touches no sampled component: the
            # wall is built around the door alone. Something valid may touch
            # it between samples -- enclosing the door encloses that too.
            d0 = entries[0]
            reg = Region(page=name, index=-1, n_cells=0, y_min=float(d0.y),
                         y_max=float(d0.y), reaches_start=False, reaches_goal=False)
            pts = np.zeros((0, 2))
            k_lab = -1
        visited_list.append(dict(asdict(reg), node=str(k),
                                 entry_doors=[door_key(e) for e in entries]))

        if verbose:
            print(f"\n[stage4] region ({name}, {k}): {reg.n_cells} cells, "
                  f"y in [{reg.y_min:.3f}, {reg.y_max:.3f}]"
                  + (f", entered through {len(entries)} door(s)" if entries else ""))

        # A door node whose every entry door is already inside a verified
        # wall on this page adds nothing: the components touching those doors
        # are inside that wall, and its doors were resolved there.
        if entries and all(enclosed_on(name, e) for e in entries):
            if verbose:
                print(f"[stage4] region ({name}, {k}): every entry door is inside a "
                      f"verified wall on this page already -- no new wall")
            continue

        if is_comp and reg.reaches_goal:
            # The goal is reachable ACCORDING TO THE REGION LABELS. That is a
            # chain of local guesses, not a path: every door on the route here
            # may have been followed on exit 4, i.e. without any proof that
            # the robot can reach it or cross it. Composing those guesses into
            # "feasible" is the one place this algorithm could return a wrong
            # answer, so the claim now has to be cashed in for an actual
            # motion. If the splice cannot be built, the honest answer is
            # INCONCLUSIVE -- never "infeasible", and never "feasible".
            if verbose:
                print(f"[stage4] goal set reached in region ({name}, {k}); "
                      f"building a witness path")
            route, chain, cur = [], [], (name, k)
            while cur is not None and len(chain) <= len(visited) + 1:
                chain.append(cur)
                p = parent.get(cur)
                cur = (p[0], p[1]) if p is not None else None
            chain.reverse()
            for idx, (pn, pk) in enumerate(chain):
                signs_i = sources[pn].signs
                entry_i = entry_point.get((pn, pk))
                if entry_i is None:
                    route = []
                    break
                if idx + 1 < len(chain):
                    nxt = chain[idx + 1]
                    _, _, disc_i, partner_i, _, _ = parent[nxt]
                    route.append((signs_i, entry_i, disc_i, partner_i))
                else:
                    route.append((signs_i, entry_i, None, None))

            if not use_witness or not cfg.witness.enable:
                return ProofResult(
                    False, "goal set reached (region labels); witness path "
                           "disabled, so feasibility is NOT certified",
                    regions_visited=visited_list, certificate=certificate,
                    doors_followed=doors_followed, goal_reached_via=f"{name}/{k}",
                    verdict=INCONCLUSIVE, door_decisions=decisions,
                    second_walls=second_walls)

            path, why = (build_witness(cfg, arm, weld, scene, route,
                                       cfg.flood.goal_y, verbose)
                         if route else (None, "could not rebuild the route"))
            if path is None:
                if verbose:
                    print(f"[stage4] INCONCLUSIVE: {why}")
                return ProofResult(
                    False, f"goal set reached by region labels, but no verified "
                           f"path exists for that route: {why}",
                    regions_visited=visited_list, certificate=certificate,
                    doors_followed=doors_followed, goal_reached_via=f"{name}/{k}",
                    verdict=INCONCLUSIVE, door_decisions=decisions,
                    second_walls=second_walls)
            wrep = verify_witness(cfg, arm, weld, scene, path,
                                  q_start=q_start_cfg, goal_y=cfg.flood.goal_y)
            if verbose:
                print(f"[witness] {wrep.reason}")
            if not wrep.ok:
                return ProofResult(
                    False, "goal set reached by region labels, but the path "
                           "failed verification",
                    regions_visited=visited_list, certificate=certificate,
                    doors_followed=doors_followed, goal_reached_via=f"{name}/{k}",
                    verdict=INCONCLUSIVE, witness_path=path.as_dict(),
                    witness_report=wrep.as_dict(), door_decisions=decisions,
                    second_walls=second_walls)
            return ProofResult(
                False, "FEASIBLE: a verified end-to-end witness path exists",
                regions_visited=visited_list, certificate=certificate,
                doors_followed=doors_followed, goal_reached_via=f"{name}/{k}",
                verdict=FEASIBLE, witness_path=path.as_dict(),
                witness_report=wrep.as_dict(), door_decisions=decisions,
                second_walls=second_walls)

        # ---- the training set ------------------------------------------
        seed_base = thin(pts, n_train_max, rng)
        rest = src.rest_pool(k_lab, cfg.wall.rest_mode)
        rest_pts0 = thin(rest, n_train_max, rng)
        # eta-separation, roadmap only. A grid already guarantees one cell of
        # vacuum between the classes; a roadmap does not, and the kernel width
        # Algorithm 1 must reach is 1/d^2 in that gap. See eta_separate.
        if on_roadmap and cfg.wall.eta_separation > 0.0 and len(seed_base):
            scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
            seed_base, rest_pts0, n_cut = eta_separate(
                seed_base, rest_pts0, cfg.wall.eta_separation, scale)
            if verbose and n_cut:
                print(f"[eta] region ({name}, {k}): dropped {n_cut} point(s) "
                      f"within {cfg.wall.eta_separation} of the other class")
        ring = virtual_boundary_samples(cfg, 80)
        # The whole goal band, valid or not (findings 47): it steers the wall
        # off the goal where no goal component was sampled. The exact test is
        # the y_max check inside find_inf_proof_region.
        rest_base = np.vstack([rest_pts0, ring, goal_line_samples(cfg)])
        inv = thin(src.obstacle_points(), n_invalid_max, rng)
        # Discharge the doors that fiber certification has proved impassable,
        # exactly as the demo's prove() does. A disc exists to stop the wall
        # routing through a door that MIGHT be passable (invariant 7); a door
        # whose whole fiber is invalid carries no such risk, and leaving its
        # disc in forces the wall to dodge an opening nothing can use. On this
        # page that is 8 of 16 doors, and the detour is what drove gamma from
        # 1.3 to 334 while prove() -- which does discharge them -- settled at
        # 1.3 (findings 29).
        discs, discharged = live_discs(cfg, arm, weld, scene, src.signs,
                                       sigma.get(name, []), verbose=False)
        if verbose and discharged:
            print(f"[stage5-fiber] {len(discharged)} disc(s) discharged on "
                  f"{name}: their doors are certified blocked")
        # The guard covers every live door AND every entry door: the wall may
        # cut neither, so each lies wholly on one side and one probe decides.
        guard = list(discs) + [e for e in entries
                               if not any(same_door(e, d) for d in discs)]
        candidates = [d for d in discs if not any(same_door(d, e) for e in entries)]
        entry = entry_point.get(node)
        comps = {int(k)} if is_comp else set()

        # ---- which doors go in which class -------------------------------
        promoted: dict = {}           # door_key -> (disc, how, in-page leg or None)
        if policy == "legacy":
            s_leg, r_leg = sigma_disc_samples(discs, pts, src.tol())
        else:
            # Evidence first: a door the roadmap connects to this region is
            # worth a verified path; a door the robot provably reaches cannot
            # be walled off. Connectivity alone decides nothing here.
            for d in candidates:
                if comps and comps & set(touching_regions(src, d)):
                    pleg = door_reach_evidence(cfg, arm, weld, scene, src.signs, entry, d)
                    if pleg is not None:
                        promoted[door_key(d)] = (d, "verified in-page path", pleg)

        poly = Polytope()
        rep = CheckReport(False, 0, 0)
        wmodel = None
        iters = rounds = 0
        entry_ok, entry_why = False, ""
        attempt_log = []
        for attempt in range(n_rounds + 1):
            seed_pts, rest_pts = seed_base, rest_base
            if policy == "legacy":
                if len(s_leg):
                    seed_pts = np.vstack([seed_pts, s_leg])
                if len(r_leg):
                    rest_pts = np.vstack([rest_pts, r_leg])
            else:
                inside_doors = entries + [v[0] for v in promoted.values()]
                excluded = [d for d in candidates if door_key(d) not in promoted]
                if inside_doors:
                    seed_pts = np.vstack([seed_pts] + [door_disc_samples(d)
                                                       for d in inside_doors])
                if excluded:
                    rest_pts = np.vstack([rest_pts] + [door_disc_samples(d)
                                                       for d in excluded])
            if not len(seed_pts):
                seed_pts = np.vstack([door_disc_samples(e) for e in entries])

            labeller = (
                RoadmapLabeller(cfg, arm, weld, scene, src, k_lab)
                if getattr(src, "kind", "grid") == "roadmap" else
                PageLabeller(cfg, arm, weld, scene, src.signs, pm, src.lab, k_lab,
                             search_cells=cfg.wall.label_search_cells,
                             motion_samples=cfg.wall.label_motion_samples)
            )
            poly, rep, wmodel, iters, rounds = budgeted_alg4(
                cfg, f"{name}/{k}", verbose=verbose,
                arm=arm, weld=weld, scene=scene, signs=src.signs, seed_pts=seed_pts,
                rest_pts=rest_pts, inv_pts=inv, discs=guard, max_iters=alg4_iters,
                labeller=labeller, goal_y_limit=goal_limit, goal_y_feed=goal_edge,
            )
            # Every entry -- and every promoted door -- must be INSIDE. The
            # guard keeps each disc wholly on one side, so its centre decides.
            entry_ok, entry_why = bool(rep.ok and poly.closed), ""
            if entry_ok and policy != "legacy":
                for e in entries + [v[0] for v in promoted.values()]:
                    side = point_in_wall(cfg, poly, (e.y, e.theta))
                    if side is not True:
                        entry_ok = False
                        entry_why = (f"door y={e.y:.4f} th={e.theta:+.4f} that must be "
                                     f"inside is {'outside' if side is False else 'undecidable'}")
                        break
            attempt_log.append(dict(attempt=attempt, promoted=sorted(promoted),
                                    facets=f"{rep.n_passed}/{rep.n_facets}",
                                    ok=entry_ok, note=entry_why or (
                                        rep.failures[0][1] if rep.failures else poly.note)))
            if entry_ok or policy == "legacy" or attempt == n_rounds:
                break
            # The wall failed. "Cannot be excluded" is the second kind of
            # evidence: promote the doors the roadmap connects to this region
            # (and that the first pass could not prove reachable), retry once.
            # Following an unreachable door costs walls, never soundness.
            new = [d for d in candidates if door_key(d) not in promoted
                   and comps and comps & set(touching_regions(src, d))]
            if not new:
                break
            for d in new:
                promoted[door_key(d)] = (d, "could not be excluded (roadmap-connected)", None)
            if verbose:
                print(f"[doors] region ({name}, {k}): wall failed with "
                      f"{len(new)} roadmap-connected door(s) excluded -> promoting them "
                      f"to the seed class and retrying")

        wall_ok = bool(entry_ok)
        y_hi = wall_y_max(poly) if poly.closed and poly.loops else None
        if verbose:
            print(f"[stage3] region ({name}, {k}): alg4 done in {iters} iter(s); "
                  f"closed={poly.closed}, facets {rep.n_passed}/{rep.n_facets}"
                  + (f", y_max {y_hi:.4f}" if y_hi is not None else "")
                  + ("" if wall_ok else f" -- NOT verified ({entry_why or 'see note'})"))

        certificate.append(asdict(CertificateEntry(
            page=name, region=int(k_lab), n_facets=rep.n_facets, n_passed=rep.n_passed,
            closed=poly.closed, gamma=float(getattr(wmodel, "gamma", float("nan"))),
            alg2_rounds=rounds, ok=wall_ok,
            note=entry_why or poly.note or (rep.failures[0][1] if rep.failures else ""),
            y_max=y_hi, goal_limit=goal_limit,
            entry_doors=[door_key(e) for e in entries],
            seed_doors=[{"door": kk, "why": v[1]} for kk, v in promoted.items()],
            excluded_doors=[door_key(d) for d in candidates if door_key(d) not in promoted],
            attempts=attempt_log,
        )))

        # --- resolve the doors: the exits ------------------------------------
        #
        # Exit 1  d outside the wall           -> PROVEN unreachable, free
        # Exit 2  wall2 verifies, d outside it -> PROVEN unreachable (opt-in)
        # Exit 3  path to d AND crossing leg   -> PROVEN reachable, follow
        # Exit 4  none of the above            -> UNKNOWN, follow anyway
        #
        # With a VERIFIED wall every door in the guard is probed -- not only
        # the doors near the region's samples (findings 47): a door inside
        # the wall but far from every sample is one the robot may reach
        # through space the roadmap never saw, and skipping it is the one way
        # this loop could end in a false INFEASIBLE. With a failed wall the
        # run is INCONCLUSIVE anyway; the doors near the region are followed
        # only to give the FEASIBLE side a chance.
        if wall_ok:
            here = guard
        else:
            here = discs_in_region(discs, pts, src.tol()) + [
                v[0] for v in promoted.values()]
        _seen: set = set()
        here = [d for d in here if not (door_key(d) in _seen or _seen.add(door_key(d)))]
        for d in here:
            ex, follow, verified, why = "4-unknown", True, False, ""

            side = point_in_wall(cfg, poly, (d.y, d.theta)) if wall_ok else None
            if side is False:
                ex, follow, why = "1-wall1", False, (
                    "outside the verified wall for this region: Definition 1 "
                    "with X = this door proves the start cannot reach it")
            if wall_ok and side is True:
                walled.setdefault(name, []).append(d)
            is_entry = any(same_door(d, e) for e in entries)

            # Exit 3 runs BEFORE exit 2 although it is numbered after it. The
            # two conclusions are mutually exclusive, and exit 3 is far
            # cheaper: if a witness path proves the door reachable then NO
            # separating wall exists, so exit 2 on such a door is not merely
            # wasted, it is the EXPENSIVE waste -- a wall that cannot exist
            # burns its whole Algorithm 4 budget before saying so (measured on
            # EXP 3's own door: 153.9 s to reach 0/0 facets at gamma 399.9).
            #
            # Within exit 3 the crossing leg goes first for the same reason:
            # it is closed form and costs microseconds, while the PRM costs
            # seconds, and a door that cannot be crossed makes the path to it
            # pointless.
            if follow and use_witness and cfg.witness.enable and not is_entry:
                target = [o for o in d.pages if o != name and o in sources]
                partner = sources[target[0]].signs if target else None
                if partner is None:
                    why = "no partner page on this door"
                else:
                    dleg, cert = door_leg(cfg, arm, weld, scene, d, src.signs,
                                          partner)
                    if dleg is None:
                        why = f"not crossable -- {cert.reason}"
                    else:
                        pv = promoted.get(door_key(d))
                        pleg = pv[2] if pv is not None and pv[2] is not None else (
                            plan_in_page(cfg, arm, weld, scene, src.signs,
                                         entry, target_yt=d.anchor)
                            if entry is not None else None)
                        if pleg is not None:
                            ex, verified, why = "3-witness", True, (
                                f"reachable (PRM path, {len(pleg.waypoints)} "
                                f"waypoints) and crossable "
                                f"({np.degrees(cert.margin):.2f} deg of slack)")
                        else:
                            why = ("crossable, but no in-page path to it was "
                                   "found -- INCONCLUSIVE, not unreachable")

            # Exit 2 -- opt-in (use_second_wall). Retired by default: the
            # region's one wall already tries to exclude every door.
            if follow and not verified and use_second_wall and not is_entry:
                others = [o for o in guard if o is not d]
                p2, r2, w2, ok2 = second_wall(
                    cfg, arm, weld, scene, src.signs, seed_base if len(seed_base)
                    else door_disc_samples(d), inv, d, others,
                    alg4_iters=max(1, alg4_iters // 3), verbose=False)
                second_walls.append({
                    "page": name, "region": int(k_lab), "door_y": float(d.y),
                    "door_theta": float(d.theta),
                    "n_facets": r2.n_facets, "n_passed": r2.n_passed,
                    "closed": bool(p2.closed), "gamma": float(w2.gamma),
                    "separates": bool(ok2),
                    "note": (r2.failures[0][1] if r2.failures else p2.note),
                })
                if ok2:
                    ex, follow, why = "2-wall2", False, (
                        "a second wall, closed and entirely in C_obs, has the "
                        "start inside and this door outside")
                elif verbose:
                    print(f"[exit2] door y={d.y:.4f}: no separating wall "
                          f"({r2.n_passed}/{r2.n_facets} facets, "
                          f"closed={p2.closed}) -> no conclusion")

            if not follow:
                decisions.append(asdict(DoorDecision(
                    page=name, region=int(k_lab), y=float(d.y), theta=float(d.theta),
                    kind=d.kind, exit=ex, followed=False,
                    crossing_verified=False, reason=why)))
                if verbose and d in discs_in_region(discs, pts, src.tol()):
                    print(f"[exit{ex[0]}] door y={d.y:.4f} th={d.theta:+.4f}: "
                          f"SKIPPED -- {why}")
                continue

            for other in d.pages:
                if other == name or other not in sources:
                    continue
                osrc = sources[other]
                if enclosed_on(other, d):
                    # Already inside a verified wall over there: every
                    # component touching it is inside that wall.
                    decisions.append(asdict(DoorDecision(
                        page=name, region=int(k_lab), y=float(d.y),
                        theta=float(d.theta), kind=d.kind, exit=ex, followed=True,
                        crossing_verified=verified,
                        reason=why + ("; " if why else "") + f"already enclosed by a "
                               f"verified wall on {other}",
                        to=f"{other}/enclosed")))
                    continue
                # Which components to start from is a HINT (touching_regions):
                # the partner wall gets the whole door in its seed class and
                # must enclose it, so a component missed here is still inside.
                targets = touching_regions(osrc, d) or [("door", door_key(d))]
                for kk in targets:
                    nxt = (other, kk)
                    doors_followed.append({
                        "from": f"{name}/{k}", "to": f"{other}/{kk}",
                        "y": d.y, "theta": d.theta, "kind": d.kind,
                        "exit": ex, "crossing_verified": verified,
                    })
                    decisions.append(asdict(DoorDecision(
                        page=name, region=int(k_lab), y=float(d.y), theta=float(d.theta),
                        kind=d.kind, exit=ex, followed=True,
                        crossing_verified=verified, reason=why,
                        to=f"{other}/{kk}")))
                    if nxt in visited:
                        continue
                    lst = node_entry.setdefault(nxt, [])
                    if not any(same_door(d, e) for e in lst):
                        lst.append(d)
                    if nxt not in queue:
                        queue.append(nxt)
                        parent[nxt] = (name, k, d, osrc.signs, d.anchor, verified)
                        entry_point[nxt] = tuple(d.anchor)
                        if verbose:
                            print(f"[exit{ex[0]}] door ({d.kind}) at "
                                  f"(y={d.y:.4f}, th={d.theta:+.4f}): "
                                  f"{name}/{k} -> {other}/{kk} "
                                  f"[crossing_verified={verified}]")

    all_ok = bool(certificate) and all(c["ok"] for c in certificate)
    # Note what is NOT required here. Doors followed on exit 4 carry no proof
    # that the robot can reach or cross them, and that is fine for this side
    # of the verdict: following a door it cannot use makes the proof cover
    # regions the robot can never occupy, which is a STRONGER claim than the
    # task needs (invariant 2). The unverified crossings can only cost work,
    # never soundness -- they bite solely on the FEASIBLE side, which is why
    # the goal branch above refuses to report feasibility without a path.
    n_unverified = sum(1 for d in decisions
                       if d["followed"] and not d["crossing_verified"])
    n_stopped = sum(1 for c in certificate if str(c["note"]).startswith(_budget.STOP))
    return ProofResult(
        proven=all_ok,
        reason=("queue empty, goal never reached, every wall verified"
                + (" (goal band outside every wall)" if goal_on else "")
                + (f" ({n_unverified} door(s) followed without a crossing "
                   f"proof -- conservative, so the certificate still holds)"
                   if n_unverified else "")
                if all_ok else
                "queue empty and goal never reached, but some wall failed "
                "verification" + (f" ({n_stopped} {_budget.STOP})" if n_stopped else "")
                + " -- INCONCLUSIVE, not a proof"),
        regions_visited=visited_list, certificate=certificate,
        doors_followed=doors_followed,
        verdict=INFEASIBLE if all_ok else INCONCLUSIVE,
        door_decisions=decisions, second_walls=second_walls,
    )


def write_certificate(result: ProofResult, sigma: dict, path) -> None:
    """Serialise the certificate so a standalone verifier can re-check it."""
    payload = {
        # `verdict` is the authoritative field. `proven` is kept for callers
        # written before the three-way verdict existed, but it must not be
        # read as "not proven => feasible": that is the whole point of the
        # INCONCLUSIVE state.
        "verdict": result.verdict,
        "proven": result.proven,
        "reason": result.reason,
        "regions_visited": result.regions_visited,
        "walls": result.certificate,
        "second_walls": result.second_walls,
        "doors_followed": result.doors_followed,
        "door_decisions": result.door_decisions,
        "goal_reached_via": result.goal_reached_via,
        # Present only on a FEASIBLE verdict. A verifier re-checks this with
        # FK + limits + collision alone -- no SVM, no topology, no resolution.
        "witness_path": result.witness_path,
        "witness_report": result.witness_report,
        "sigma_cover": {
            name: [
                {"kind": d.kind, "y": d.y, "theta": d.theta, "radius": d.radius,
                 "vanishing": sorted(d.vanishing), "pages": list(d.pages)}
                for d in discs
            ]
            for name, discs in sigma.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def prove_line_escalating(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
                          start_page: str = "++-", n_samples: int = 1200,
                          verbose: bool = False, **run_kw) -> tuple:
    """The grid-free 2-D proof, re-run on an escalation ladder (findings 44).

    Each rung bootstraps its OWN roadmap (escalate_factor times the samples,
    a new seed), seeds the regions of `start_page` touching the start edge,
    and runs run_proof under that rung's budget. Every rung is a complete,
    independent proof, so the first conclusive one is the answer; a rung
    stopped by its budget is INCONCLUSIVE and only ever leads to the next
    rung. Returns (ProofResult, log)."""
    from .source import bootstrap_roadmap

    def attempt(n, seed, bp):
        c = dataclasses.replace(cfg, budget=bp,
                                planner=dataclasses.replace(cfg.planner, rng_seed=seed))
        try:
            # One clock for the whole rung: the roadmap counts against it too.
            with _budget.run_budget(bp, "rung"):
                srcs, sig = bootstrap_roadmap(c, arm, weld, scene, n_samples=n)
                src = srcs[start_page]
                seeds = [(start_page, r.index) for r in src.regions() if r.reaches_start]
                if not seeds:
                    return ProofResult(proven=False, verdict=INCONCLUSIVE,
                                       reason="no valid start-edge region on the start page")
                return run_proof(c, arm, weld, scene, None, sig, sources=srcs,
                                 seeds=seeds, verbose=verbose, **run_kw)
        except _budget.BudgetExceeded as e:
            return ProofResult(proven=False, reason=e.reason, verdict=INCONCLUSIVE)

    return _budget.escalate(attempt, cfg.budget, n_samples, cfg.planner.rng_seed,
                            verbose=True)
