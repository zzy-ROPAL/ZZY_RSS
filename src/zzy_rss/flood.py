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

from .arm import SrsArm
from .params import Config
from .pagemap import PageMap, page_name
from .scene import Scene
from .doors import SigmaDisc, door_fiber_validity, door_passable
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
class Region:
    """One page-restricted connected component of the valid set."""

    page: str
    index: int
    n_cells: int
    y_min: float
    y_max: float
    reaches_start: bool
    reaches_goal: bool


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


def find_inf_proof_region(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    seed_pts: np.ndarray, rest_pts: np.ndarray, inv_pts: np.ndarray,
    discs: list, max_iters: int = 4, verbose: bool = True, labeller=None,
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
        wall, wall_pts, seed_pts, rest_pts, rounds = sample_wall(
            cfg, arm, weld, scene, signs, wall, inv_pts, seed_pts, rest_pts,
            verbose=False, labeller=labeller,
        )
        alg2_total += rounds

        if not tc_ok:
            poly = reconstruct_wall(cfg, wall)
            tc_ok = poly.closed and bool(poly.loops)
        if not tc_ok:
            poly = reconstruct_wall(cfg, wall, grid_res=361)
            tc_ok = poly.closed and bool(poly.loops)
            if not tc_ok:
                if verbose:
                    print(f"[stage3] alg4 iter {it}: wall not closed "
                          f"({poly.note or 'open loop'})")
                continue

        rep = check_polytope(cfg, arm, weld, scene, signs, poly, discs,
                             seed_pts, verbose=False)
        if verbose:
            print(f"[stage3] alg4 iter {it}: facets {rep.n_passed}/{rep.n_facets} "
                  f"passed (max recursion {rep.max_recursion})")
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
    poly, rep, wmodel, iters, rounds = find_inf_proof_region(
        cfg, arm, weld, scene, signs, seed_pts, rest_pts, inv_pts, others,
        max_iters=alg4_iters, verbose=verbose,
    )
    ok = bool(rep.ok and poly.closed and poly.loops)
    outside = None
    if ok:
        outside = point_in_wall(cfg, poly, (target.y, target.theta))
        outside = (outside is False)
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
    for signs, entry_yt, disc, partner in route:
        if disc is None:
            leg = plan_in_page(cfg, arm, weld, scene, signs, entry_yt,
                               goal_y=goal_y)
            if leg is None:
                return None, (f"no in-page path on {page_name(signs)} from "
                              f"y={entry_yt[0]:.4f} to the goal")
            legs.append(leg)
            break
        leg = plan_in_page(cfg, arm, weld, scene, signs, entry_yt,
                           target_yt=disc.anchor)
        if leg is None:
            return None, (f"no in-page path on {page_name(signs)} from "
                          f"y={entry_yt[0]:.4f} to the door at "
                          f"y={disc.anchor[0]:.4f}")
        legs.append(leg)
        dleg, cert = door_leg(cfg, arm, weld, scene, disc, signs, partner)
        if dleg is None:
            return None, (f"door at y={disc.anchor[0]:.4f} on "
                          f"{page_name(signs)} is not crossable: {cert.reason}")
        legs.append(dleg)
    return WitnessPath(legs=legs), ""


def run_proof(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    pages: dict, sigma: dict, n_train_max: int = 900, n_invalid_max: int = 160,
    verbose: bool = True, alg4_iters: int = 4, seeds: list | None = None,
    use_second_wall: bool = True, use_witness: bool = True,
    start_yt: tuple | None = None,
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
    """
    rng = np.random.default_rng(cfg.planner.rng_seed)
    labels, regions_by_page = {}, {}
    for name, pm in pages.items():
        lab, regs = label_regions(cfg, pm)
        labels[name] = lab
        regions_by_page[name] = regs

    # Seed the queue. By default: every region that touches the start of the
    # weld, on every page -- which proves the STRONGER claim that no posture
    # anywhere at y = 0 can finish. Pass `seeds` to prove the claim the task
    # actually makes: the robot starts in ONE posture, so the proof starts in
    # the one region holding it and reaches the other pages only through doors
    # it can actually cross. That is the same claim EXP 2's single wall makes,
    # and it does not ask the method to wall regions the robot can never be in.
    if seeds is not None:
        queue = [(n, k) for n, k in seeds if n in pages]
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
    for n_, k_ in queue:
        pm_ = pages[n_]
        pts_ = region_points(pm_, labels[n_], k_)
        if start_yt is not None:
            entry_point[(n_, k_)] = tuple(start_yt)
        elif len(pts_):
            entry_point[(n_, k_)] = tuple(pts_[int(np.argmin(pts_[:, 0]))])
    q_start_cfg = None
    if queue:
        n0, k0 = queue[0]
        yt0 = entry_point.get((n0, k0))
        if yt0 is not None:
            r0 = arm.ik_branch(*weld.pose(float(yt0[0])), float(yt0[1]),
                               pages[n0].signs)
            q_start_cfg = r0.q if r0.ok else None

    while queue:
        name, k = queue.pop(0)
        if (name, k) in visited:
            continue
        visited.add((name, k))
        pm = pages[name]
        lab = labels[name]
        reg = next(r for r in regions_by_page[name] if r.index == k)
        visited_list.append(asdict(reg))

        pts = region_points(pm, lab, k)
        if verbose:
            print(f"\n[stage4] region ({name}, {k}): {reg.n_cells} cells, "
                  f"y in [{reg.y_min:.3f}, {reg.y_max:.3f}]")

        if reg.reaches_goal:
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
            route, chain, node = [], [], (name, k)
            while node is not None and len(chain) <= len(visited) + 1:
                chain.append(node)
                p = parent.get(node)
                node = (p[0], p[1]) if p is not None else None
            chain.reverse()
            for idx, (pn, pk) in enumerate(chain):
                signs_i = pages[pn].signs
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

        seed_pts = thin(pts, n_train_max, rng)
        rest = other_valid_points(pm, lab, k, regions_by_page[name],
                                  cfg.wall.rest_mode)
        ring = virtual_boundary_samples(cfg, 80)
        rest_pts = np.vstack([thin(rest, n_train_max, rng), ring])
        inv = thin(invalid_points(pm), n_invalid_max, rng)
        # Discharge the doors that fiber certification has proved impassable,
        # exactly as the demo's prove() does. A disc exists to stop the wall
        # routing through a door that MIGHT be passable (invariant 7); a door
        # whose whole fiber is invalid carries no such risk, and leaving its
        # disc in forces the wall to dodge an opening nothing can use. On this
        # page that is 8 of 16 doors, and the detour is what drove gamma from
        # 1.3 to 334 while prove() -- which does discharge them -- settled at
        # 1.3 (findings 29).
        discs, discharged = live_discs(cfg, arm, weld, scene, pm.signs,
                                       sigma.get(name, []), verbose=False)
        if verbose and discharged:
            print(f"[stage5-fiber] {len(discharged)} disc(s) discharged on "
                  f"{name}: their doors are certified blocked")

        # Teach the wall where the doors are, so it cannot route through one.
        tol_d = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                               pm.theta_grid[1] - pm.theta_grid[0]))
        s_extra, r_extra = sigma_disc_samples(discs, pts, tol_d)
        if len(s_extra):
            seed_pts = np.vstack([seed_pts, s_extra])
        if len(r_extra):
            rest_pts = np.vstack([rest_pts, r_extra])

        labeller = PageLabeller(
            cfg, arm, weld, scene, pm.signs, pm, lab, k,
            search_cells=cfg.wall.label_search_cells,
            motion_samples=cfg.wall.label_motion_samples,
        )
        poly, rep, wmodel, iters, rounds = find_inf_proof_region(
            cfg, arm, weld, scene, pm.signs, seed_pts, rest_pts, inv, discs,
            max_iters=alg4_iters, verbose=verbose, labeller=labeller,
        )
        if verbose:
            print(f"[stage3] region ({name}, {k}): alg4 done in {iters} iter(s); "
                  f"closed={poly.closed}, facets {rep.n_passed}/{rep.n_facets}")

        certificate.append(asdict(CertificateEntry(
            page=name, region=k, n_facets=rep.n_facets, n_passed=rep.n_passed,
            closed=poly.closed, gamma=wmodel.gamma, alg2_rounds=rounds,
            ok=bool(rep.ok and poly.closed),
            note=poly.note or (rep.failures[0][1] if rep.failures else ""),
        )))

        # --- resolve every door in this region: the four exits ---------------
        #
        # Exit 1  d outside wall1              -> PROVEN unreachable, free
        # Exit 2  wall2 verifies, d outside it -> PROVEN unreachable, ~seconds
        # Exit 3  path to d AND crossing leg   -> PROVEN reachable, follow
        # Exit 4  none of the above            -> UNKNOWN, follow anyway
        #
        # Exits 1 and 2 reach the same conclusion by the same theorem and
        # differ only in which wall; exit 3 reaches the opposite one. They are
        # four independent proof ATTEMPTS, not a decision tree: falling
        # through them means nothing was proved, which is exit 4, not a
        # verdict. Both exits 3 and 4 follow the door -- only the tag differs,
        # and the tag is what decides whether a goal reached beyond it may be
        # called FEASIBLE.
        tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                             pm.theta_grid[1] - pm.theta_grid[0]))
        here = discs_in_region(discs, pts, tol)
        for d in here:
            ex, follow, verified, why = "4-unknown", True, False, ""

            # Exit 1 -- the wall we already built may have done the job.
            side = point_in_wall(cfg, poly, (d.y, d.theta)) if rep.ok else None
            if side is False:
                ex, follow, why = "1-wall1", False, (
                    "outside the verified wall for this region: Definition 1 "
                    "with X = this door proves the start cannot reach it")

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
            if follow and use_witness and cfg.witness.enable:
                target = [o for o in d.pages if o != name and o in pages]
                partner = pages[target[0]].signs if target else None
                if partner is None:
                    why = "no partner page on this door"
                else:
                    dleg, cert = door_leg(cfg, arm, weld, scene, d, pm.signs,
                                          partner)
                    if dleg is None:
                        why = f"not crossable -- {cert.reason}"
                    else:
                        entry = entry_point.get((name, k))
                        pleg = (plan_in_page(cfg, arm, weld, scene, pm.signs,
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

            # Exit 2 -- pay for a wall whose only job is this door. Reached
            # only when exit 3 produced nothing, so the door may genuinely be
            # unreachable. Budgeted well below wall1's: failing is expected
            # and costs only time, because exit 4 follows the door anyway.
            if follow and not verified and use_second_wall:
                others = [o for o in discs if o is not d]
                p2, r2, w2, ok2 = second_wall(
                    cfg, arm, weld, scene, pm.signs, seed_pts, inv, d, others,
                    alg4_iters=max(1, alg4_iters // 3), verbose=False)
                second_walls.append({
                    "page": name, "region": k, "door_y": float(d.y),
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
                    page=name, region=k, y=float(d.y), theta=float(d.theta),
                    kind=d.kind, exit=ex, followed=False,
                    crossing_verified=False, reason=why)))
                if verbose:
                    print(f"[exit{ex[0]}] door y={d.y:.4f} th={d.theta:+.4f}: "
                          f"SKIPPED -- {why}")
                continue

            for other in d.pages:
                if other == name or other not in pages:
                    continue
                opm, olab = pages[other], labels[other]
                i = int(np.argmin(np.abs(opm.y_grid - d.y)))
                j = int(np.argmin(np.abs(opm.theta_grid - d.theta)))
                # Conservative: enqueue every region within the disc footprint.
                rad = max(1, int(np.ceil(d.radius / (opm.y_grid[1] - opm.y_grid[0]))))
                sub = olab[max(0, i - rad): i + rad + 1, max(0, j - rad): j + rad + 1]
                for kk in sorted(set(int(v) for v in np.unique(sub) if v > 0)):
                    doors_followed.append({
                        "from": f"{name}/{k}", "to": f"{other}/{kk}",
                        "y": d.y, "theta": d.theta, "kind": d.kind,
                        "exit": ex, "crossing_verified": verified,
                    })
                    decisions.append(asdict(DoorDecision(
                        page=name, region=k, y=float(d.y), theta=float(d.theta),
                        kind=d.kind, exit=ex, followed=True,
                        crossing_verified=verified, reason=why,
                        to=f"{other}/{kk}")))
                    if (other, kk) not in visited:
                        queue.append((other, kk))
                        parent[(other, kk)] = (name, k, d, opm.signs,
                                               d.anchor, verified)
                        entry_point[(other, kk)] = tuple(d.anchor)
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
    return ProofResult(
        proven=all_ok,
        reason=("queue empty, goal never reached, every wall verified"
                + (f" ({n_unverified} door(s) followed without a crossing "
                   f"proof -- conservative, so the certificate still holds)"
                   if n_unverified else "")
                if all_ok else
                "queue empty and goal never reached, but some wall failed "
                "verification -- INCONCLUSIVE, not a proof"),
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
