"""The recursive infeasibility proof on a weld patch (docs/recursive_proof_design.md, step 3).

    prove_task  1. FEASIBLE side: a breadth-first search over the glued
                   roadmap (every page's components, joined by crossing
                   states either side of each door), cashed in for ONE
                   spliced path that verify_witness accepts (invariant 25).
                   A refuted path blacklists its bad edges and re-runs
                   (lazy PRM), as flood3 does.
                2. INFEASIBLE side: inf_proof(start, goal).

    inf_proof(A, B, branch):
        r = check(A, B)
        CONNECTED          -> NOT PROVEN (the roadmap joins A and B)
        FAILED             -> NOT PROVEN (reason)
        WALLED, no doors   -> PROVEN, a leaf
        WALLED with doors  -> for EVERY door d inside (AND):
                                  d on the branch already  -> NOT PROVEN (cycle)
                                  inf(A, d) proven, or else inf(d, B) proven (OR)
                              all doors cut -> PROVEN, a node with one child per door

Why it is sound: the walls of check(A, B) enclose A, exclude B and touch no
live door, so a path from A to B must leave through a door d inside them;
it then reaches d from A and B from d. Cutting either half for every such d
leaves no path. A call still on the branch is never assumed proven
(that would be circular), so a cycle can only cost a verdict.

PROVEN results are cached by (A, B): they rest on no assumption. NOT PROVEN
ones are not (a cycle verdict depends on the branch).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import budget as _budget
from .arm import SrsArm
from .check import (CONNECTED, FAILED, WALLED, DoorNode, GoalSet, StartSet, World,
                    check)
from .flood3 import (FIBER_PAIR, door_crossing_leg, flip, page_name, refuted_edges,
                     splice_path)
from .params import Config, Page3Params
from .scene import Scene
from .weld import WeldSurface
from .witness import WitnessPath, verify_witness

FEASIBLE, INFEASIBLE, INCONCLUSIVE = "FEASIBLE", "INFEASIBLE", "INCONCLUSIVE"
PROVEN, NOT_PROVEN = "PROVEN", "NOT_PROVEN"


@dataclass
class ProofNode:
    A: str
    B: str
    status: str                       # PROVEN / NOT_PROVEN
    reason: str
    walls: list = field(default_factory=list)       # one dict per page
    children: list = field(default_factory=list)    # dict(door, cut, node)
    geometry: list = field(default_factory=list, repr=False)   # wall meshes (not in the JSON)

    def as_dict(self) -> dict:
        return dict(A=self.A, B=self.B, status=self.status, reason=self.reason,
                    walls=self.walls,
                    children=[dict(door=c["door"], cut=c["cut"], node=c["node"].as_dict())
                              for c in self.children])

    def wall_meshes(self) -> list:
        """Every wall mesh of the tree: dict(node, page, ok, vertices, simplices),
        vertices in page coordinates (u, v, theta)."""
        out = [dict(node=f"{self.A} -/-> {self.B}", **g) for g in self.geometry]
        for c in self.children:
            out += c["node"].wall_meshes()
        return out

    def count_walls(self) -> int:
        return len(self.walls) + sum(c["node"].count_walls() for c in self.children)

    def lines(self, indent: str = "") -> list:
        w = ", ".join(f"{x['page']} {x['facets']}" for x in self.walls)
        out = [f"{indent}{self.status:10s} {self.A} -/-> {self.B}"
               + (f"   walls: {w}" if w else "") + f"   ({self.reason})"]
        for c in self.children:
            out.append(f"{indent}  door {c['door']}: cut {c['cut']}")
            out += c["node"].lines(indent + "    ")
        return out


@dataclass
class TaskResult:
    verdict: str
    reason: str
    tree: ProofNode | None = None
    path: object = None
    witness: dict | None = None
    stats: dict = field(default_factory=dict)

    def certificate(self) -> dict:
        return dict(verdict=self.verdict, reason=self.reason,
                    tree=None if self.tree is None else self.tree.as_dict(),
                    witness=self.witness, stats=self.stats)


# --------------------------------------------------------------------------
# the recursion
# --------------------------------------------------------------------------


class Prover:
    def __init__(self, world: World, max_checks: int = 24, verbose: bool = True):
        self.world, self.max_checks, self.verbose = world, max_checks, verbose
        self.cache: dict = {}
        self.checks = 0

    def inf_proof(self, A, B, branch: frozenset = frozenset(), depth: int = 0) -> ProofNode:
        key = (A, B)
        if key in self.cache:
            return self.cache[key]
        pad = "  " * depth
        if self.checks >= self.max_checks:
            return ProofNode(A.label(), B.label(), NOT_PROVEN,
                             f"check budget {self.max_checks} exhausted")
        self.checks += 1
        if self.verbose:
            print(f"\n[proof]{pad} inf({A.label()} -> {B.label()})  [check {self.checks}]")
        r = check(self.world, A, B, verbose=self.verbose)
        walls = [dict(page=page_name(w.signs), ok=w.ok, reason=w.reason,
                      facets=w.info.get("facets"), gamma=w.info.get("gamma"),
                      u_max=w.info.get("u_max"), doors_inside=w.info.get("doors_inside", []))
                 for w in r.walls]
        if self.verbose:
            print(f"[proof]{pad}   -> {r.kind}: {r.reason}"
                  + (f"; doors inside: {[d.label() for d in r.doors]}" if r.doors else ""))
        geom = [dict(page=page_name(w.signs), ok=w.ok,
                     vertices=np.asarray(w.poly.vertices, dtype=float),
                     simplices=np.asarray(w.poly.simplices, dtype=int))
                for w in r.walls if w.poly is not None and w.poly.vertices is not None]
        if r.kind != WALLED:
            return ProofNode(A.label(), B.label(), NOT_PROVEN, f"{r.kind}: {r.reason}", walls,
                             geometry=geom)
        node = ProofNode(A.label(), B.label(), PROVEN, "walls verified", walls, geometry=geom)
        if not r.doors:
            node.reason = "walls verified, no live door inside"
            self.cache[key] = node
            return node
        for d in r.doors:
            if d in branch:
                return ProofNode(A.label(), B.label(), NOT_PROVEN,
                                 f"cycle: {d.label()} is already being cut on this branch",
                                 walls, geometry=geom)
            left = self.inf_proof(A, d, branch | {d}, depth + 1)
            if left.status == PROVEN:
                node.children.append(dict(door=d.label(), cut=f"{A.label()} -/-> door",
                                          node=left))
                continue
            right = self.inf_proof(d, B, branch | {d}, depth + 1)
            if right.status == PROVEN:
                node.children.append(dict(door=d.label(), cut=f"door -/-> {B.label()}",
                                          node=right))
                continue
            return ProofNode(A.label(), B.label(), NOT_PROVEN,
                             f"{d.label()}: neither {A.label()} -/-> door ({left.reason}) "
                             f"nor door -/-> {B.label()} ({right.reason})", walls,
                             geometry=geom)
        node.reason = f"walls verified; all {len(r.doors)} door(s) inside cut"
        self.cache[key] = node
        return node


# --------------------------------------------------------------------------
# the FEASIBLE side: one verified path over the glued roadmap
# --------------------------------------------------------------------------


def connect(world: World, verbose: bool = True) -> tuple:
    """Breadth-first over (page, component) through crossing states; a goal
    component found is spliced into one path and verified. Returns
    (verdict or None, reason, path, witness dict, bad_edges)."""
    cfg, arm, surf, scene, ds = world.cfg, world.arm, world.surf, world.scene, world.ds
    start = world.page(world.start_signs)
    root = (world.start_signs, int(start.comp[0]))
    parent: dict = {root: None}
    pairs: dict = {}
    frontier = [root]
    qsum: dict = {}

    def pair_sum(signs, n):
        key = (signs, int(n))
        if key not in qsum:
            x = ds.pts[n]
            r = arm.ik_branch(*surf.pose(float(x[0]), float(x[1])), float(x[2]), signs)
            i, _j, k = FIBER_PAIR[int(ds.sign[n])]
            qsum[key] = float(r.q[i] + r.q[k]) if r.ok else np.nan
        return qsum[key]

    goal = None
    while frontier and goal is None:
        node = frontier.pop(0)
        signs, comp = node
        page = world.page(signs)
        if np.any(page.goal_mask() & (page.comp == comp)):
            goal = node
            break
        for n in np.flatnonzero(page.inject_index >= 0):
            if int(page.comp[page.inject_index[n]]) != comp:
                continue
            k = int(world.sample_door[n])
            if world.discharged(signs, k)[0]:
                continue
            partner = flip(signs, int(ds.sign[n]))
            order = [int(ds.mirror[n])] + [m for m in ds.same_sample(n)
                                           if m != int(ds.mirror[n])]
            for tsigns in (partner, signs):
                tp = world.page(tsigns)
                for n2 in order:
                    vb = tp.inject_index[n2]
                    if vb < 0 or not abs(pair_sum(signs, n) - pair_sum(tsigns, n2)) < 0.5:
                        continue
                    nxt = (tsigns, int(tp.comp[vb]))
                    if nxt == node:
                        continue
                    if nxt not in parent:
                        parent[nxt] = (node, int(n), int(n2))
                        frontier.append(nxt)
                    if parent[nxt] is not None and parent[nxt][0] == node:
                        pairs.setdefault(nxt, []).append((int(n), int(n2)))
    if goal is None:
        return None, f"no goal component reachable over the glued roadmap " \
                     f"({len(parent)} region(s) reached)", None, None, None
    hops, cur = 0, goal
    while parent[cur] is not None:
        hops += 1
        prev = parent[cur][0]
        for n, n2 in pairs.get(cur, [])[: int(world.p3.cross_tries)]:
            leg, _ = door_crossing_leg(cfg, arm, surf, world.pages[prev[0]],
                                       world.pages[cur[0]], ds, n, n2)
            if leg is not None and verify_witness(cfg, arm, surf, scene,
                                                  WitnessPath(legs=[leg])).ok:
                parent[cur] = (prev, n, n2)
                break
        cur = prev
    if verbose:
        print(f"[connect] goal region {page_name(goal[0])}/{goal[1]} reached across "
              f"{hops} door crossing(s); splicing and verifying")
    path, why = splice_path(world.pages, ds, parent, goal, None, cfg, arm, surf)
    if path is None:
        return None, f"goal reached on the roadmap but no path spliced: {why}", None, None, None
    rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start, goal_y=1.0)
    if verbose:
        print(f"[witness] {rep.reason}")
    if rep.ok:
        return FEASIBLE, f"verified path across {hops} door crossing(s)", path, \
            rep.as_dict(), None
    bad = refuted_edges(cfg, arm, surf, scene, world.pages, path)
    return None, "spliced path refuted", None, rep.as_dict(), bad


# --------------------------------------------------------------------------
# the task
# --------------------------------------------------------------------------


def prove_task(cfg: Config, arm: SrsArm, surf: WeldSurface, scene: Scene,
               start_signs: tuple, p3: Page3Params, start_theta: float | None = None,
               max_checks: int = 24, verbose: bool = True,
               also_prove: bool = False) -> TaskResult:
    """also_prove: after a FEASIBLE verdict, run the recursion anyway and
    attach its tree (illustration only: it must come out NOT PROVEN, since a
    verified path exists; the verdict stays FEASIBLE)."""
    t0 = time.time()
    try:
        with _budget.run_budget(cfg.budget, "prove_task"):
            world = World(cfg, arm, surf, scene, start_signs, p3, start_theta, verbose)
            st = dict(door_regions=len(world.doorset.doors),
                      door_seconds=world.door_seconds)
            if not world.doorset.complete:
                return TaskResult(INCONCLUSIVE, "sigma search did not finish", stats=st)
            if world.start_theta is None:
                return TaskResult(INCONCLUSIVE, "no valid start posture", stats=st)
            if not world.page(world.start_signs).cut_ok:
                return TaskResult(INCONCLUSIVE, "bounded theta not justified on the start page",
                                  stats=st)
            reason = ""
            for attempt in range(int(p3.lazy_rounds) + 1):
                verdict, reason, path, wit, bad = connect(world, verbose)
                if verdict == FEASIBLE:
                    tree = None
                    if also_prove:
                        prover = Prover(world, max_checks=max_checks, verbose=verbose)
                        tree = prover.inf_proof(StartSet(world.start_signs), GoalSet())
                        st.update(checks=prover.checks, walls_built=world.walls_built)
                        if tree.status == PROVEN:
                            raise AssertionError("a proof tree next to a verified path: "
                                                 "one of them is wrong")
                    st.update(seconds=round(time.time() - t0, 1),
                              pages_built=[page_name(s) for s in world.pages])
                    return TaskResult(FEASIBLE, reason, tree=tree, path=path, witness=wit,
                                      stats=st)
                if not bad:
                    break
                for signs, edges in bad.items():
                    world.pages[signs].blacklist |= edges
                    world.pages[signs].rebuild()
                st["lazy_rounds"] = attempt + 1
            st["connect"] = reason
            prover = Prover(world, max_checks=max_checks, verbose=verbose)
            tree = prover.inf_proof(StartSet(world.start_signs), GoalSet())
            st.update(seconds=round(time.time() - t0, 1), checks=prover.checks,
                      walls_built=world.walls_built,
                      pages_built=[page_name(s) for s in world.pages])
            if tree.status == PROVEN:
                return TaskResult(INFEASIBLE, f"proof tree with {tree.count_walls()} verified "
                                              f"wall(s)", tree=tree, stats=st)
            return TaskResult(INCONCLUSIVE, tree.reason, tree=tree, stats=st)
    except _budget.BudgetExceeded as e:
        if verbose:
            print(f"[budget] {e.reason}")
        return TaskResult(INCONCLUSIVE, e.reason, stats=dict(budget_stop=e.reason))


def write_geometry(res: TaskResult, path) -> None:
    """The pictures' data, next to the JSON certificate: every wall mesh of the
    proof tree and the verified path's joint configurations (pickle)."""
    import pickle
    q = None
    if res.path is not None and getattr(res.path, "legs", None):
        q = np.concatenate([np.asarray(leg.q, dtype=float) for leg in res.path.legs])
    with open(path, "wb") as f:
        pickle.dump(dict(verdict=res.verdict, reason=res.reason,
                         walls=[] if res.tree is None else res.tree.wall_meshes(),
                         path_q=q), f)


def write_certificate(res: TaskResult, path) -> None:
    with open(path, "w") as f:
        json.dump(res.certificate(), f, indent=1, default=str)
