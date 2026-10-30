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
from .doors import door_fiber_validity
from .wall import (
    sigma_disc_samples,
    CheckReport, Polytope, check_polytope, reconstruct_wall, sample_wall,
    train_wall, virtual_boundary_samples,
)
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
class ProofResult:
    proven: bool
    reason: str
    regions_visited: list = field(default_factory=list)
    certificate: list = field(default_factory=list)
    doors_followed: list = field(default_factory=list)
    goal_reached_via: str = ""


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


def other_valid_points(pm: PageMap, lab: np.ndarray, k: int) -> np.ndarray:
    idx = np.argwhere(pm.valid_mask & (lab != k))
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
    if not cfg.sigma_policy.enable_fiber_certification:
        return list(discs), []
    live, discharged = [], []
    for d in discs:
        if not d.vanishing:
            live.append(d)
            continue
        fib = door_fiber_validity(cfg, arm, weld, scene, d, signs)
        (live if fib["ok"] else discharged).append(d)
        if verbose:
            ay, _ = d.anchor
            print(f"[stage5-fiber] door y={ay:.4f}: "
                  f"{'OPEN, wall must avoid it' if fib['ok'] else 'BLOCKED, disc discharged'}")
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
            for m in add:
                if assign_label(m, seed_pts, rest_pts, wall.scale, lb) > 0:
                    seed_pts = np.vstack([seed_pts, m])
                else:
                    rest_pts = np.vstack([rest_pts, m])
            wall = train_wall(cfg, seed_pts, rest_pts)
            tc_ok = False
            if verbose:
                print(f"[stage3] alg4 iter {it}: fed {len(add)} valid facet "
                      f"centre(s) back into training -> retrain "
                      f"(gamma={wall.gamma:.2f})")
            continue

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


def run_proof(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    pages: dict, sigma: dict, n_train_max: int = 900, n_invalid_max: int = 160,
    verbose: bool = True,
) -> ProofResult:
    """IJRR Algorithm 4 wrapped in the page/door flood fill."""
    rng = np.random.default_rng(cfg.planner.rng_seed)
    labels, regions_by_page = {}, {}
    for name, pm in pages.items():
        lab, regs = label_regions(cfg, pm)
        labels[name] = lab
        regions_by_page[name] = regs

    # Seed the queue with every region that touches the start of the weld.
    queue = []
    for name, regs in regions_by_page.items():
        for r in regs:
            if r.reaches_start:
                queue.append((name, r.index))
    if not queue:
        return ProofResult(True, "no region touches the start; trivially infeasible")

    if verbose:
        print(f"\n[stage4] seeding queue with {len(queue)} start-touching region(s): "
              f"{[(n, k) for n, k in queue]}")

    visited = set()
    certificate, visited_list, doors_followed = [], [], []

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
            if verbose:
                print(f"[stage4] GOAL REACHED in region ({name}, {k}) -> no proof; "
                      f"the planner returns a path")
            return ProofResult(
                False, "goal set reached: the problem is feasible",
                regions_visited=visited_list, certificate=certificate,
                doors_followed=doors_followed, goal_reached_via=f"{name}/{k}",
            )

        seed_pts = thin(pts, n_train_max, rng)
        rest = other_valid_points(pm, lab, k)
        ring = virtual_boundary_samples(cfg)
        rest_pts = np.vstack([thin(rest, n_train_max, rng), ring])
        inv = thin(invalid_points(pm), n_invalid_max, rng)
        discs = sigma.get(name, [])

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
            verbose=verbose, labeller=labeller,
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

        # --- follow every door inside this region ---------------------------
        tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                             pm.theta_grid[1] - pm.theta_grid[0]))
        for d in discs_in_region(discs, pts, tol):
            # Door reachability is a FIBER property. The IK representative can
            # be invalid while other fiber points are valid, so never decide a
            # door from the representative alone. With fiber certification off
            # (the default) doors are simply assumed open, which is the
            # conservative direction for an infeasibility proof.
            if cfg.sigma_policy.enable_fiber_certification:
                fib = door_fiber_validity(cfg, arm, weld, scene, d, pm.signs)
                if not fib["ok"]:
                    if verbose:
                        print(f"[stage5-fiber] door at (y={d.y:.4f}, "
                              f"th={d.theta:+.4f}) on {name}: {fib['reason']}")
                    continue
                if verbose:
                    print(f"[stage5-fiber] door at (y={d.y:.4f}, th={d.theta:+.4f}) "
                          f"on {name}: OPEN ({fib['valid_fraction'] * 100:.1f}% of "
                          f"fiber valid)")
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
                    })
                    if (other, kk) not in visited:
                        queue.append((other, kk))
                        if verbose:
                            print(f"[stage4] door ({d.kind}) at (y={d.y:.4f}, "
                                  f"th={d.theta:+.4f}): {name}/{k} -> {other}/{kk}")

    all_ok = bool(certificate) and all(c["ok"] for c in certificate)
    return ProofResult(
        proven=all_ok,
        reason=("queue empty, goal never reached, every wall verified"
                if all_ok else
                "queue empty and goal never reached, but some wall failed "
                "verification -- INCONCLUSIVE, not a proof"),
        regions_visited=visited_list, certificate=certificate,
        doors_followed=doors_followed,
    )


def write_certificate(result: ProofResult, sigma: dict, path) -> None:
    """Serialise the certificate so a standalone verifier can re-check it."""
    payload = {
        "proven": result.proven,
        "reason": result.reason,
        "regions_visited": result.regions_visited,
        "walls": result.certificate,
        "doors_followed": result.doors_followed,
        "goal_reached_via": result.goal_reached_via,
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
