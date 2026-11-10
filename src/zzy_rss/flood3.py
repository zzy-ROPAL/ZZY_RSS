"""Cross-page flood on the 3-D pages of a weld patch (findings 43).

The d = 3 counterpart of flood.run_proof. A region is (page, component of
that page's roadmap). Starting from the start posture's region it repeats:

    goal reached?        -> splice a path back to the start and VERIFY it
    else build the wall  -> closed, every triangle in C_obs, every entry
                            state inside, the goal slab outside
    every LIVE door curve inside the wall must be followed: the robot can
    reach it and leave the page there, so every component of the partner
    page that touches the curve is queued. A curve outside the wall is
    unreachable from this region (exit 1).

INFEASIBLE when the queue empties with every wall verified and the goal never
reached; FEASIBLE only with a verified end-to-end path; INCONCLUSIVE else.

HOW A DOOR IS CROSSED. Not by the fiber delta-sweep. At a q_j door the two
axes next to joint j coincide, so the IK's spelling AT the door is an
atan2(0, 0) artifact (0 on one page, -180 deg on the other at EXP 3's door),
and the "delta = 0 -> pi" picture measures from that artifact. A robot
arriving from direction d holds the spelling of the limit along d and simply
passes through: q_j changes sign, every other joint is continuous, nothing
moves along the fiber. So a crossing is the straight page segment

        x = p + eps d   (page P)   ->   p   ->   x' = p - eps d   (page P')

with P's spelling before p and P''s after. Every page is built with the same
symmetric set of such states injected as vertices (p + eps d for d and -d),
so each crossing endpoint is a REAL vertex in a real component -- the
partner region is a lookup, not a footprint guess (the d = 3 version of
findings 38's regions_under fix).

Coverage, and why it no longer decides anything (findings 47): the curve is
sampled (every cert_stride-th point, eight directions across it), so a
partner component that touches the curve only between samples is not found
by the injected states. That used to be the one direction that could make
INFEASIBLE false. Phase 2 now puts the WHOLE door piece into the partner
region's seed class and requires the verified wall to enclose it; a free
component touching the piece cannot cross a wall that lies in C_obs, so it
is inside whether or not a sample found it. The injected states remain what
phase 1 and the witness use to actually cross.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import budget as _budget
from .arm import SrsArm
from .doors3 import certify_curve, door_curves, patch_singularities
from .page3 import (FEASIBLE, INCONCLUSIVE, INFEASIBLE, Page3, choose_start_theta,
                    clip_to_box, densify3, door_pieces, piece_vertices, theta_cut_check,
                    wall_for_region)
from .params import Config, Page3Params
from .scene import Scene
from .weld import WeldSurface
from .witness import LEG_PAGE, Leg, WitnessPath, verify_witness


def flip(signs: tuple, idx: int) -> tuple:
    out = list(signs)
    out[idx] = -out[idx]
    return tuple(out)


def page_name(signs: tuple) -> str:
    return "".join("+" if s > 0 else "-" for s in signs)


# --------------------------------------------------------------------------
# near-door states: the crossing endpoints, injected into every page
# --------------------------------------------------------------------------


@dataclass
class DoorSamples:
    """For every curve sample p and direction d: the state p + eps d.

    pts[n] is one injected state; key[n] = (curve, sample, direction).
    `mirror[n]` is the index of p - eps d, the other end of the crossing."""

    pts: np.ndarray
    key: list
    mirror: np.ndarray
    p: np.ndarray                       # the door point of each state
    sign: np.ndarray                    # sign index of each state's curve

    def same_sample(self, n: int) -> list:
        """Every injected state around the same door point as state n."""
        ci, si, _k = self.key[n]
        if not hasattr(self, "_by"):
            by: dict = {}
            for m, (c, s_, _kk) in enumerate(self.key):
                by.setdefault((c, s_), []).append(m)
            self._by = by
        return self._by[(ci, si)]


def door_samples(curves: list, scale: np.ndarray, eps: float, stride: int,
                 n_dirs: int = 8) -> DoorSamples:
    """Symmetric crossing endpoints across each curve, in SCALED distance eps.

    Directions span the plane normal to the curve (moving along it stays on
    the door), in scaled coordinates so they are isotropic."""
    pts, key, door, sign = [], [], [], []
    for ci, c in enumerate(curves):
        sc_pts = c.pts * scale
        for si in range(0, len(c.pts), max(1, stride)):
            j0, j1 = max(0, si - 1), min(len(c.pts) - 1, si + 1)
            t = sc_pts[j1] - sc_pts[j0]
            t = t / max(np.linalg.norm(t), 1e-12)
            a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n1 = np.cross(t, a)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(t, n1)
            for k in range(n_dirs):
                ang = 2.0 * np.pi * k / n_dirs
                d = np.cos(ang) * n1 + np.sin(ang) * n2
                pts.append((sc_pts[si] + eps * d) / scale)
                key.append((ci, si, k))
                door.append(c.pts[si])
                sign.append(c.sign_index)
    index = {k: n for n, k in enumerate(key)}
    half = n_dirs // 2
    mirror = np.array([index[(ci, si, (k + half) % n_dirs)] for ci, si, k in key],
                      dtype=int)
    return DoorSamples(pts=np.asarray(pts).reshape(-1, 3), key=key, mirror=mirror,
                       p=np.asarray(door).reshape(-1, 3),
                       sign=np.asarray(sign, dtype=int))


# --------------------------------------------------------------------------
# the flood
# --------------------------------------------------------------------------


@dataclass
class FloodResult3:
    verdict: str
    reason: str
    regions: list = field(default_factory=list)       # visited, in order
    certificate: list = field(default_factory=list)   # one wall per region
    crossings: list = field(default_factory=list)     # doors followed
    doors: dict = field(default_factory=dict)         # per page: curve verdicts
    witness: dict | None = None
    stats: dict = field(default_factory=dict)
    path: object = None        # the verified WitnessPath behind a FEASIBLE verdict
    polys: list = field(default_factory=list)   # (page, comp, Polytope) per wall built


def run_flood3(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
               start_signs: tuple, p3: Page3Params, start_theta: float | None = None,
               max_regions: int = 12, verbose: bool = True) -> FloodResult3:
    """The flood, inside a lazy-PRM loop: a path that fails verification
    blacklists its bad roadmap edges, the pages are rebuilt, and the flood
    runs again (Page3Params.lazy_rounds). Walls are only ever built for
    regions of the CURRENT graphs, so a certificate never rests on an edge a
    witness has refuted.

    Runs under cfg.budget (findings 44): exhausting it gives INCONCLUSIVE
    "stopped by budget: ...", and a region whose wall was stopped ends the
    flood the same way any unverified wall does."""
    try:
        with _budget.run_budget(cfg.budget, "flood3"):
            return _run_flood3(cfg, arm, surf, scene, start_signs, p3, start_theta,
                               max_regions, verbose)
    except _budget.BudgetExceeded as e:
        if verbose:
            print(f"[budget] {e.reason}")
        return FloodResult3(INCONCLUSIVE, e.reason, stats={"budget_stop": e.reason})


def _run_flood3(cfg, arm, surf, scene, start_signs, p3, start_theta, max_regions,
                verbose) -> FloodResult3:
    state: dict = {}
    res = None
    for attempt in range(int(p3.lazy_rounds) + 1):
        res = _flood_once(cfg, arm, surf, scene, start_signs, p3, start_theta,
                          max_regions, verbose, state)
        bad = state.pop("bad_edges", None)
        if not bad:
            break
        n_bad = sum(len(v) for v in bad.values())
        if verbose:
            print(f"[flood3] lazy PRM: {n_bad} roadmap edge(s) refuted by the witness; "
                  f"rebuilding and re-running (attempt {attempt + 2})")
        for signs, edges in bad.items():
            pg = state["pages"][signs]
            pg.blacklist |= edges
            pg.rebuild()
        res.stats["lazy_rounds"] = attempt + 1
    return res


def _flood_once(cfg, arm, surf, scene, start_signs, p3, start_theta, max_regions,
                verbose, state) -> FloodResult3:
    t0 = time.time()
    st: dict = {}
    sing = patch_singularities(cfg, arm, surf)
    if not sing["ok"]:
        return FloodResult3(INCONCLUSIVE, f"unsupported singular set: {sing['reason']}")
    if start_theta is None:
        start_theta = choose_start_theta(cfg, arm, surf, scene, start_signs, p3)
        if start_theta is None:
            return FloodResult3(INCONCLUSIVE, "no valid start posture at the start point")
    st["start"] = [p3.start_u, p3.start_v, float(start_theta)]

    if "geometry" not in state:
        state["geometry"] = door_curves(cfg, arm, surf, p3)
        state["ds"] = door_samples(state["geometry"], np.asarray(p3.scale, dtype=float),
                                   float(p3.cross_eps), int(p3.cert_stride))
        state["pages"], state["certified"] = {}, {}
    geometry, ds = state["geometry"], state["ds"]
    st["door_curves"] = len(geometry)
    st["injected_per_page"] = len(ds.pts)

    pages: dict = state["pages"]
    certified: dict = state["certified"]   # (signs, curve) -> DoorCurve + verdict

    def get_page(signs):
        if signs not in pages:
            t = time.time()
            pg = Page3(cfg, arm, surf, scene, signs, p3,
                       start_theta if signs == start_signs else None, inject=ds.pts)
            pages[signs] = pg
            cut = theta_cut_check(pg)
            pg.cut_ok, pg.cut_reason = cut["ok"], cut["reason"]
            if verbose:
                print(f"[flood3] page {page_name(signs)}: {len(pg.valid)} valid vertices, "
                      f"{int((pg.inject_index >= 0).sum())}/{len(ds.pts)} near-door states "
                      f"valid ({time.time() - t:.1f} s)")
        return pages[signs]

    def live_curves(signs):
        out = []
        for ci, g in enumerate(geometry):
            if (signs, ci) not in certified:
                c = type(g)(sign_index=g.sign_index, pts=g.pts, closed=g.closed, tube=g.tube)
                certify_curve(cfg, arm, surf, scene, c, signs, p3)
                certified[(signs, ci)] = c
            if not certified[(signs, ci)].certified_blocked:
                out.append((ci, certified[(signs, ci)]))
        return out

    start_page = get_page(start_signs)
    if not start_page.cut_ok:
        return FloodResult3(INCONCLUSIVE, f"bounded theta not justified on "
                            f"{page_name(start_signs)}: {start_page.cut_reason}", stats=st)
    root = (start_signs, int(start_page.comp[0]))

    # ---- phase 1: FEASIBLE by connectivity, before any wall -------------
    # (CLAUDE.md, correction A: "feasible" comes from reaching the goal, not
    # from a wall failing.) Breadth-first over regions through the injected
    # crossing pairs: x = p + eps d in this region, x' = p - eps d valid on
    # the partner page. A goal region found this way is cashed in for a
    # verified path; a refuted path feeds the lazy-PRM loop. No wall is built.
    reach_parent: dict = {root: None}
    reach_pairs: dict = {}            # region -> every (n, n2) from its parent
    frontier = [root]
    qsum: dict = state.setdefault("qsum", {})

    def pair_sum(signs, n):
        """q_i + q_k of injected state n on page `signs` (cached). The fiber
        motion keeps it fixed, so a crossing needs it equal on both sides --
        a 2*pi difference cannot be joined within the joint limits."""
        key = (signs, int(n))
        if key not in qsum:
            x = ds.pts[n]
            r = arm.ik_branch(*surf.pose(float(x[0]), float(x[1])), float(x[2]), signs)
            i, _j, k = FIBER_PAIR[int(ds.sign[n])]
            qsum[key] = float(r.q[i] + r.q[k]) if r.ok else np.nan
        return qsum[key]
    goal_node = None
    while frontier and goal_node is None:
        node = frontier.pop(0)
        signs, comp = node
        page = get_page(signs)
        if np.any(page.goal_mask() & (page.comp == comp)):
            goal_node = node
            break
        for n in np.flatnonzero(page.inject_index >= 0):
            if int(page.comp[page.inject_index[n]]) != comp:
                continue
            ci = ds.key[n][0]
            partner = flip(signs, geometry[ci].sign_index)
            # Straight through first (no fiber motion), then every other
            # departure around the same door point, on either page.
            order = [int(ds.mirror[n])] + [m for m in ds.same_sample(n)
                                           if m != int(ds.mirror[n])]
            for tsigns in (partner, signs):
                tp = get_page(tsigns)
                for n2 in order:
                    vb = tp.inject_index[n2]
                    if vb < 0:
                        continue
                    if not abs(pair_sum(signs, n) - pair_sum(tsigns, n2)) < 0.5:
                        continue
                    nxt = (tsigns, int(tp.comp[vb]))
                    if nxt == node:
                        continue
                    if nxt not in reach_parent:
                        reach_parent[nxt] = (node, int(n), int(n2))
                        frontier.append(nxt)
                    if reach_parent[nxt] is not None and reach_parent[nxt][0] == node:
                        reach_pairs.setdefault(nxt, []).append((int(n), int(n2)))
    st["phase1_regions"] = len(reach_parent)
    if goal_node is not None:
        hops = 0
        cur = goal_node
        while reach_parent[cur] is not None:
            hops += 1
            cur = reach_parent[cur][0]
        if verbose:
            print(f"[flood3] phase 1: goal region {page_name(goal_node[0])}/{goal_node[1]} "
                  f"reached by connectivity across {hops} door crossing(s); verifying")
        # Each hop: the first recorded pair may be refused (a 2*pi split
        # mismatch, a sweep out of limits). Try every pair into the region
        # until one makes a crossing leg.
        why = ""
        cur = goal_node
        while reach_parent[cur] is not None:
            prev = reach_parent[cur][0]
            for n, n2 in reach_pairs.get(cur, [])[: int(p3.cross_tries)]:
                leg, why = door_crossing_leg(cfg, arm, surf, pages[prev[0]], pages[cur[0]],
                                             ds, n, n2)
                if leg is None:
                    continue
                # A leg that builds can still fail: the door POSTURE may touch an
                # obstacle while both endpoints, cross_eps away, are free (a door
                # curve grazing a box; findings 45). Verify the leg on its own,
                # as refuted_edges does for roadmap edges, and try the next pair.
                lrep = verify_witness(cfg, arm, surf, scene, WitnessPath(legs=[leg]))
                if lrep.ok:
                    reach_parent[cur] = (prev, n, n2)
                    break
                why = f"crossing leg fails verification: {lrep.failures[0][1]}"
            cur = prev
        path, why2 = splice_path(pages, ds, reach_parent, goal_node, None, cfg, arm, surf)
        why = why2 or why
        if path is not None:
            rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start,
                                 goal_y=1.0)
            if verbose:
                print(f"[witness-3d] {rep.reason}")
            st["seconds"] = round(time.time() - t0, 1)
            if rep.ok:
                return FloodResult3(
                    FEASIBLE, f"verified path across {hops} door crossing(s) "
                    f"({' -> '.join(page_name(s) for s in _chain_pages(reach_parent, goal_node))})",
                    [dict(page=page_name(s), comp=c) for s, c in reach_parent],
                    witness=rep.as_dict(), stats=st, path=path)
            bad = refuted_edges(cfg, arm, surf, scene, pages, path)
            if bad:
                state["bad_edges"] = bad
                return FloodResult3(INCONCLUSIVE, "phase 1 path refuted; lazy PRM retry",
                                    witness=rep.as_dict(), stats=st)
            why = "the spliced path failed verification at a door crossing"
        if verbose:
            print(f"[flood3] phase 1 could not cash the route in ({why}); building walls")

    # ---- phase 2: walls, region by region -------------------------------
    #
    # A node is (page, component), or (page, ("door", curve, piece)) for a
    # page entered through a door piece that touches no sampled component.
    # Findings 47 changes three things here:
    #
    #   * every node entered through a door gets that WHOLE door piece in its
    #     seed class, and its wall must enclose it -- so every component
    #     touching the door is inside, whether or not an injected sample
    #     happened to land in it (the coverage caveat in this module's
    #     docstring is now a matter of efficiency only);
    #   * every live piece is probed against the verified wall, and a piece
    #     already enclosed by a verified wall on the page it leads to is not
    #     followed again;
    #   * the door is no longer followed back onto its OWN page: the piece is
    #     wholly inside this wall, so every own-page component touching it is
    #     inside this wall too.
    if "sample_piece" not in state:
        lo = np.array([0.0, 0.0, p3.theta_min])
        hi = np.array([1.0, 1.0, p3.theta_max])
        scale = np.asarray(p3.scale, dtype=float)
        polys = {ci: clip_to_box(np.asarray(g.tube.polyline, dtype=float), lo, hi)
                 for ci, g in enumerate(geometry)}
        sp = []
        for n, (ci, _si, _k) in enumerate(ds.key):
            best, best_d = None, 1e-6
            for j, poly in enumerate(polys[ci]):
                d_ = float(np.min(np.linalg.norm((poly - ds.p[n]) * scale, axis=1)))
                if d_ <= best_d:
                    best, best_d = (ci, j), d_
            sp.append(best)
        state["sample_piece"] = sp
    sample_piece = state["sample_piece"]

    queue = [root]
    entries = {root: [0]}                 # vertex indices that enter the region
    node_entry: dict = {root: set()}      # door pieces the region was entered through
    parent: dict = {root: None}           # region -> (prev region, n_from, n_to)
    walled: dict = {}                     # signs -> piece keys inside a verified wall there
    visited, regions, certificate, crossings = set(), [], [], []

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        if len(visited) >= max_regions:
            return FloodResult3(INCONCLUSIVE, f"region budget {max_regions} exhausted",
                                regions, certificate, crossings, stats=st)
        visited.add(node)
        signs, comp = node
        page = get_page(signs)
        is_comp = isinstance(comp, (int, np.integer))
        page.set_focus(comp if is_comp else None)
        ekeys = set(node_entry.get(node, set()))
        n_pts = int((page.comp == comp).sum()) if is_comp else 0
        cid = int(comp) if is_comp else f"door{comp[1]}.{comp[2]}"
        if ekeys and ekeys <= walled.get(signs, set()):
            regions.append(dict(page=page_name(signs), comp=cid, vertices=n_pts,
                                skipped="every entry door is inside a verified wall here"))
            if verbose:
                print(f"\n[flood3] region {page_name(signs)}/{cid}: every entry door is "
                      f"already inside a verified wall on this page -- no new wall")
            continue
        regions.append(dict(page=page_name(signs), comp=cid, vertices=n_pts,
                            entry_doors=[list(k) for k in sorted(ekeys)]))
        if verbose:
            print(f"\n[flood3] region {page_name(signs)}/{cid}: {n_pts} vertices, "
                  f"entered via {len(entries.get(node, []))} state(s)"
                  + (f" and {len(ekeys)} door piece(s)" if ekeys else ""))
        if not page.cut_ok:
            return FloodResult3(INCONCLUSIVE, f"bounded theta not justified on "
                                f"{page_name(signs)}: {page.cut_reason}",
                                regions, certificate, crossings, stats=st)

        goal_here = (page.goal_mask() & (page.comp == comp)) if is_comp else np.zeros(1, bool)
        if goal_here.any():
            if verbose:
                print(f"[flood3] goal reached in {page_name(signs)}/{comp}: splicing a path")
            path, why = splice_path(pages, ds, parent, node, entries, cfg, arm, surf)
            if path is None:
                return FloodResult3(INCONCLUSIVE, f"goal reached on the roadmaps, but no "
                                    f"path could be spliced: {why}",
                                    regions, certificate, crossings, stats=st)
            rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start,
                                 goal_y=1.0)
            if verbose:
                print(f"[witness-3d] {rep.reason}")
            st["seconds"] = round(time.time() - t0, 1)
            if not rep.ok:
                bad = refuted_edges(cfg, arm, surf, scene, pages, path)
                if bad:
                    state["bad_edges"] = bad
                return FloodResult3(INCONCLUSIVE, "goal reached, but the spliced path "
                                    "failed verification", regions, certificate,
                                    crossings, witness=rep.as_dict(), stats=st)
            return FloodResult3(FEASIBLE, f"verified path across "
                                f"{sum(1 for l in path.legs if l.note == 'crossing')} "
                                f"door crossing(s)", regions, certificate, crossings,
                                witness=rep.as_dict(), stats=st, path=path)

        live = live_curves(signs)
        pieces = door_pieces(page, live)
        have = {pc.key for pc in pieces}
        # An entry piece is guarded and seeded even where it is not live on
        # this page (e.g. certified blocked in this spelling).
        for ci in sorted({k[0] for k in ekeys}):
            for pc in door_pieces(page, [(ci, geometry[ci])]):
                if pc.key in ekeys and pc.key not in have:
                    pieces.append(pc)
                    have.add(pc.key)
        entry_idx = [int(v) for v in entries.get(node, []) if int(v) >= 0]
        entry_pts = page.valid[entry_idx] if entry_idx else np.zeros((0, 3))
        ev = next((v for v in entry_idx if is_comp and int(page.comp[v]) == int(comp)), None)
        w = wall_for_region(cfg, page, pieces, entry_pts, entry_keys=ekeys,
                            entry_vertex=ev, verbose=verbose)
        certificate.append(dict(page=page_name(signs), comp=cid, **w["wall"]))
        state.setdefault("polys", []).append((page_name(signs), cid, w["poly"]))
        if not w["ok"]:
            st["seconds"] = round(time.time() - t0, 1)
            return FloodResult3(INCONCLUSIVE, f"{page_name(signs)}/{cid}: {w['reason']}",
                                regions, certificate, crossings, stats=st)
        inside_keys = {k for k, v in w["piece_inside"].items() if v}
        walled.setdefault(signs, set()).update(inside_keys)

        for pc in pieces:
            info = dict(page=page_name(signs), comp=cid, curve=pc.curve, piece=pc.piece,
                        joint=f"q{pc.joint}")
            if pc.key not in inside_keys:
                crossings.append(dict(info, followed=False,
                                      why="outside the verified wall: unreachable"))
                continue
            partner = flip(signs, pc.sign_index)
            if pc.key in walled.get(partner, set()):
                crossings.append(dict(info, followed=True, to_page=page_name(partner),
                                      to_components=[],
                                      why="already inside a verified wall on the partner page"))
                continue
            tp = get_page(partner)
            # Which partner components to start from is a HINT: the injected
            # crossing endpoints of this piece, and roadmap vertices inside its
            # tube. The partner wall must enclose the whole piece anyway.
            targets: dict = {}
            for n, key in enumerate(sample_piece):
                if key != pc.key:
                    continue
                vb = tp.inject_index[n]
                if vb < 0:
                    continue
                m = int(tp.comp[vb])
                va = page.inject_index[ds.mirror[n]]
                # Prefer a straight-through pair whose near end is in THIS
                # region: the witness can use exactly that crossing.
                usable = bool(is_comp and va >= 0 and int(page.comp[va]) == int(comp))
                if m not in targets or (usable and not targets[m][2]):
                    targets[m] = (int(ds.mirror[n]), n, usable)
            for j in piece_vertices(tp, pc):
                targets.setdefault(int(tp.comp[j]), (None, int(j), False))
            crossings.append(dict(info, followed=True, to_page=page_name(partner),
                                  to_components=sorted(targets)))
            if verbose:
                print(f"[flood3] live q{pc.joint} piece {pc.key} INSIDE the wall -> "
                      f"{page_name(partner)} components {sorted(targets) or 'none (door only)'}")
            if not targets:
                targets = {("door", pc.curve, pc.piece): (None, None, False)}
            for m, (n_from, n_to, usable) in targets.items():
                nxt = (partner, m)
                if nxt in visited:
                    continue
                node_entry.setdefault(nxt, set()).add(pc.key)
                entries.setdefault(nxt, [])
                if n_to is not None:
                    # an injected state (n_from given) or a tube vertex
                    vb = int(tp.inject_index[n_to]) if n_from is not None else int(n_to)
                    if vb >= 0 and vb not in entries[nxt]:
                        entries[nxt].append(vb)
                if nxt not in parent:
                    parent[nxt] = ((node, n_from, n_to) if usable
                                   else (node, None, n_to if n_from is not None else None))
                if nxt not in queue:
                    queue.append(nxt)

    st["seconds"] = round(time.time() - t0, 1)
    st["pages_built"] = [page_name(s) for s in pages]
    n_walls = sum(1 for r in regions if "skipped" not in r)
    return FloodResult3(INFEASIBLE, f"queue empty: {len(certificate)} verified wall(s) over "
                        f"{len({r['page'] for r in regions if 'skipped' not in r})} page(s), "
                        f"goal never reached, goal slab outside every wall",
                        regions, certificate, crossings,
                        doors={page_name(s): [dict(curve=ci, live=not c.certified_blocked,
                                                    reason=c.reason)
                                               for (s2, ci), c in certified.items() if s2 == s]
                               for s in pages},
                        stats=dict(st, walls=n_walls), polys=list(state.get("polys", [])))


# --------------------------------------------------------------------------
# FEASIBLE: one spliced, verified path
# --------------------------------------------------------------------------


def refuted_edges(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
                  pages: dict, path: WitnessPath) -> dict:
    """The roadmap edges of a failed path that fail verification on their
    own: {signs: {(i, j), ...}} in vertex indices of each page."""
    out: dict = {}
    for leg in path.legs:
        if leg.note == "crossing" or leg.waypoints is None:
            continue
        page = pages[leg.page]
        way = leg.waypoints
        idx = [int(np.argmin(np.linalg.norm(page.valid - w, axis=1))) for w in way]
        for i, j in zip(idx[:-1], idx[1:]):
            seg = np.vstack([page.valid[i], page.valid[j]])
            q, uv = densify3(arm, surf, cfg.witness, page.scale, seg,
                             lambda _a, _t: leg.page)
            ok = q is not None and verify_witness(
                cfg, arm, surf, scene,
                WitnessPath(legs=[Leg(kind=LEG_PAGE, page=leg.page, q=q, y=uv)])).ok
            if not ok:
                out.setdefault(leg.page, set()).add((min(i, j), max(i, j)))
    return out


def _chain_pages(parent: dict, node) -> list:
    out = []
    while node is not None:
        out.append(node[0])
        node = parent[node][0] if parent[node] is not None else None
    return out[::-1]


FIBER_PAIR = {0: (0, 1, 2), 2: (4, 5, 6)}   # sign index -> (i, j, k): q_j = 0, q_i/q_k coincide


def door_crossing_leg(cfg: Config, arm: SrsArm, surf: WeldSurface, page, tpage,
                      ds: DoorSamples, n_from: int, n_to: int) -> tuple:
    """Arrive at door point p from x = ds.pts[n_from] on `page`, leave toward
    x' = ds.pts[n_to] on `tpage` (the partner page or the same page).

    At p the axes of q_i and q_k coincide, so the posture depends on
    q_i + q_k only and counter-rotating them is a pure null motion. The leg:

        x -> p+  on page's spelling (densified IK)
        snap to the exact door posture, split as the arrival holds it
        counter-rotate q_i/q_k to the split the departure needs
        exact door posture -> p+' -> x' on tpage's spelling

    Each joint of the pair moves linearly between two in-range values, so the
    sweep respects the limits; collision is constant along it. The sum
    q_i + q_k must agree on both sides -- a 2*pi difference would mean the
    joint values cannot be joined, and the crossing is refused. Straight
    through (x' = mirror of x on the partner page) is the delta = 0 case.
    """
    wp = cfg.witness
    p = ds.p[n_from]
    a, b = ds.pts[n_from], ds.pts[n_to]
    i, _j, k = FIBER_PAIR[int(ds.sign[n_from])]
    hair = float(wp.cross_hair)
    qa, uva = densify3(arm, surf, wp, page.scale, np.vstack([a, p + hair * (a - p)]),
                       lambda _s, _t: page.signs)
    qb, uvb = densify3(arm, surf, wp, page.scale, np.vstack([p + hair * (b - p), b]),
                       lambda _s, _t: tpage.signs)
    if qa is None or qb is None:
        return None, "IK failed next to the door"
    q_in, q_out = qa[-1], qb[0]
    rin = arm.ik_branch(*surf.pose(float(p[0]), float(p[1])), float(p[2]), page.signs)
    rout = arm.ik_branch(*surf.pose(float(p[0]), float(p[1])), float(p[2]), tpage.signs)
    if not (rin.ok and rout.ok):
        return None, "no IK at the door point"
    # The pair's SUM at the door, read where it is defined. At the q2 door the
    # IK computes q3 relative to q1, so q1 + q3 is right even though q1 is
    # atan2(0, 0). At the q6 door q5 AND q7 are both read from entries of
    # R04^T R that vanish, so their sum is atan2 noise too (findings 45: 16
    # deg of tool rotation on EXP 4's patch). There R04^T R = Rz(q5 + q7)
    # exactly, and R04 depends on q1..q4 only, which are well defined.
    _, rot_door = surf.pose(float(p[0]), float(p[1]))

    def door_sum(q_door):
        if i == 4:
            m = arm.frames(q_door)["R04"].T @ rot_door
            return float(np.arctan2(m[1, 0], m[0, 0]))
        return float(q_door[i] + q_door[k])

    # The exact door posture in the split each side actually holds.
    def resplit(q_door, q_near):
        q = q_door.copy()
        tot = door_sum(q_door)
        q[i] = q_near[i]
        q[k] = tot - q_near[i]
        q[k] += 2.0 * np.pi * np.round((q_near[k] - q[k]) / (2.0 * np.pi))
        return q
    din, dout = resplit(rin.q, q_in), resplit(rout.q, q_out)
    others = [m for m in range(len(din)) if m not in (i, k)]
    if float(np.abs(din[others] - dout[others]).max()) > 1e-6:
        return None, "the two sides are not one door posture"
    if abs((din[i] + din[k]) - (dout[i] + dout[k])) > 1e-6:
        return None, "q_i + q_k differs by a full turn across the door"
    delta = float(dout[i] - din[i])
    n = max(2, int(np.ceil(abs(delta) / wp.step_max)) + 1)
    sweep = np.repeat(din[None, :], n, axis=0)
    ds_ = np.linspace(0.0, delta, n)
    sweep[:, i] = din[i] + ds_
    sweep[:, k] = din[k] - ds_
    q = np.vstack([qa, sweep, qb])
    uv = np.vstack([uva, np.repeat(p[None, :2], n, axis=0), uvb])
    return Leg(kind=LEG_PAGE, page=page.signs, q=q, y=uv, waypoints=np.vstack([a, p, b]),
               note="crossing"), ""


def splice_path(pages: dict, ds: DoorSamples, parent: dict, node, entries: dict,
                cfg: Config, arm: SrsArm, surf: WeldSurface) -> tuple:
    """Start -> ... -> goal across the recorded crossings, as one WitnessPath."""
    chain = []
    cur = node
    while cur is not None:
        chain.append(cur)
        par = parent[cur]
        cur = par[0] if par is not None else None
    chain.reverse()
    legs = []
    at = 0                                      # vertex we stand on (start: 0)
    for i, (signs, comp) in enumerate(chain):
        page = pages[signs]
        if i + 1 < len(chain):
            _prev, n_from, n_to = parent[chain[i + 1]]
            if n_from is None:
                return None, (f"no crossing of the door into {page_name(chain[i + 1][0])} "
                              f"starts in {page_name(signs)}/{comp}")
            target = int(page.inject_index[n_from])
            mask = np.zeros(len(page.valid), dtype=bool)
            mask[target] = True
        else:
            mask = page.goal_mask() & (page.comp == comp)
        idx = page.path(at, mask)
        if idx is None:
            return None, f"no roadmap path inside {page_name(signs)}/{comp}"
        way = page.valid[idx]
        if len(way) >= 2:
            q, uv = densify3(arm, surf, cfg.witness, page.scale, way, lambda _i, _t: signs)
            if q is None:
                return None, f"IK failed along {page_name(signs)}/{comp}"
            legs.append(Leg(kind=LEG_PAGE, page=signs, q=q, y=uv, waypoints=way))
        if i + 1 < len(chain):
            nsigns = chain[i + 1][0]
            npage = pages[nsigns]
            leg, why = door_crossing_leg(cfg, arm, surf, page, npage, ds, n_from, n_to)
            if leg is None:
                return None, why
            legs.append(leg)
            at = int(npage.inject_index[n_to])
            continue
    return WitnessPath(legs=legs), ""
