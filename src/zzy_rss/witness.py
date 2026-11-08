"""Witness paths -- the constructive certificate for FEASIBLE.

The wall is a NON-constructive certificate for INFEASIBLE: "there exists a
closed curve in C_obs with the start inside and X outside". It works because
infeasibility is a statement about ALL paths, which cannot be checked one at
a time, so Definition 1 converts it into a statement about one separating
object and lets topology supply the universal quantifier.

Feasibility needs no such conversion. It is already existential -- "there
exists a path" -- so its witness is the path itself. IJRR reports a plan from
the PLANNER (Figure 2, the green box), never from the proof side.

Our flood fill broke that separation. It returns "feasible" from region
labels, which composes unverified local guesses:

    start ->? door   (page 1)   guessed, never verified
    door  ->  partner           guessed, never verified
    door  ->  goal   (page 2)   a region label
    ------------------------------------------------
    therefore start -> goal     a non sequitur

and the FEASIBLE return carries only a string, so there is nothing to check.
A path cannot be fooled that way, because it is ONE object: if the robot
cannot actually reach the door in page 1, the splice has no first leg and
there is nothing to verify.

Structure
---------
A witness path alternates two kinds of leg:

    [page leg]  PRM + Dijkstra inside one page; the arm moves
    [door leg]  a delta-sweep along a self-motion fiber at a fixed task
                point; the arm does NOT move, only the joint spelling

The door leg cannot be a PRM edge. In Y+ a door crossing is a single POINT --
both sides share (y, theta) -- so it has length only in Q. That is exactly
where the [IROS] bijection fails, and it is why the path must be carried as
joint vectors rather than page coordinates.

Verification
------------
Deliberately independent of how the path was found: N x FK + N x limits +
N x collision, plus junction and step bounds. No SVM, no topology, no
resolution parameter. That is the whole reason a path is worth more than a
certificate to a reader: it can be re-checked without understanding any of
the machinery that produced it.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .arm import SrsArm
from .doors import SigmaDisc, certify_fiber
from .params import Config
from .prm import classify
from .scene import Scene, segment_box_distance
from .weld import WeldLine

LEG_PAGE = "page"
LEG_DOOR = "door"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Leg:
    """One segment of a witness path, carried as the configurations traversed."""

    kind: str
    page: tuple
    q: np.ndarray                              # (n, 7) densified
    y: np.ndarray                              # (n,) weld parameter per sample
    waypoints: np.ndarray | None = None        # (m, 2) page coords, page legs
    anchor: tuple | None = None                # (y, theta), door legs
    deltas: np.ndarray | None = None           # (n,), door legs
    partner: tuple | None = None               # partner page, door legs
    note: str = ""

    def as_dict(self) -> dict:
        d = {"kind": self.kind, "page": list(self.page), "n": int(len(self.q)),
             "q_start_deg": [round(float(v), 4) for v in np.degrees(self.q[0])],
             "q_end_deg": [round(float(v), 4) for v in np.degrees(self.q[-1])],
             "y_start": float(self.y[0]), "y_end": float(self.y[-1]),
             "note": self.note}
        if self.kind == LEG_DOOR:
            d["anchor"] = list(self.anchor)
            d["partner"] = list(self.partner)
            d["delta_end_deg"] = float(np.degrees(self.deltas[-1]))
        return d


@dataclass
class WitnessPath:
    """A spliced sequence of legs from the start configuration to a target."""

    legs: list = field(default_factory=list)

    @property
    def q_start(self) -> np.ndarray:
        return self.legs[0].q[0]

    @property
    def q_end(self) -> np.ndarray:
        return self.legs[-1].q[-1]

    def configurations(self) -> np.ndarray:
        return np.vstack([leg.q for leg in self.legs])

    def as_dict(self) -> dict:
        return {
            "n_legs": len(self.legs),
            "n_configurations": int(sum(len(l.q) for l in self.legs)),
            "n_door_crossings": sum(1 for l in self.legs if l.kind == LEG_DOOR),
            "legs": [l.as_dict() for l in self.legs],
        }


@dataclass
class WitnessReport:
    """Result of re-checking a path from scratch."""

    ok: bool = False
    reason: str = ""
    n_configs: int = 0
    max_step: float = 0.0
    max_junction_gap: float = 0.0
    max_tool_pos_err: float = 0.0
    max_tool_rot_err: float = 0.0
    min_clearance: float = float("inf")
    swept_certified: bool = False
    failures: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "reason": self.reason, "n_configurations": self.n_configs,
            "max_step_rad": self.max_step,
            "max_junction_gap_rad": self.max_junction_gap,
            "max_tool_pos_err_mm": self.max_tool_pos_err * 1e3,
            "max_tool_rot_err_rad": self.max_tool_rot_err,
            "min_clearance_mm": (self.min_clearance * 1e3
                                 if np.isfinite(self.min_clearance) else None),
            "swept_segments_certified": self.swept_certified,
            "failures": self.failures[:20],
        }


# --------------------------------------------------------------------------
# Exact clearance -- NOT Scene.check
# --------------------------------------------------------------------------


def exact_clearance(scene: Scene, frames: dict) -> tuple[float, str]:
    """Clearance using the exact segment-box distance, not the sampled one.

    Scene.check samples n_coarse points along each capsule axis and takes the
    min, so the value it reports is an UPPER bound on the true clearance.
    scene.py argues that bias is safe "everywhere it is used", and for wall
    verification it is: it can only enlarge the valid set or make a facet
    check fail, never manufacture a proof.

    For a swept path segment the sign FLIPS. Certifying "nothing moves more
    than the clearance" with an optimistic clearance would certify a segment
    that actually collides. On a 0.33 m link with 49 samples the error reaches
    ~3.4 mm, which is not small next to the gaps in these scenes.

    segment_box_distance is exact (convexity + ternary refinement) and costs
    ~1.9 ms per call. A witness path is a one-off, so it pays for it; the fast
    sampled routine exists to serve millions of grid cells, which is a
    different problem.
    """
    caps = scene.capsules(frames)
    if not scene.boxes or not caps:
        return float("inf"), ""
    worst, witness = float("inf"), ""
    for a, b, radius, label in caps:
        for center, half, blabel in scene.boxes:
            # Tighter than segment_box_distance's defaults (17 coarse samples,
            # 24 ternary steps converge to only ~1e-6 m). A path has a few
            # hundred configurations, not millions of grid cells, so the
            # refinement is free here and the residual drops below 1e-10 m.
            dist, _ = segment_box_distance(a, b, np.asarray(center, dtype=float),
                                           np.asarray(half, dtype=float),
                                           iters=48, n_coarse=33)
            clear = float(dist - radius)
            if clear < worst:
                worst, witness = clear, f"{label}/{blabel}"
    return worst, witness


def _motion_radius(arm: SrsArm) -> float:
    """Conservative bound on how far a link point sits from any joint axis.

    A point at distance r from the axis of joint i moves at most r*|dq_i| when
    that joint turns, so total motion is bounded by R_max * ||dq||_1. Taking
    the full kinematic reach plus any off-axis hub offset over-estimates r,
    which is the safe direction for a clearance argument.
    """
    p = arm.p
    reach = float(p.d_se + p.d_ew + p.d_wt)
    off = max(float(getattr(p, "upper_arm_hub_offset", 0.0)),
              float(getattr(p, "elbow_box_offset", 0.0)))
    return reach + off


# --------------------------------------------------------------------------
# Producing legs
# --------------------------------------------------------------------------


PAGE_SCALE = np.array([1.0, 1.0 / np.pi])


def _edge_ok(cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
             a: np.ndarray, b: np.ndarray, signs: tuple) -> bool:
    """Straight-line local plan in page coordinates, resolution-checked.

    This is what makes a PRM leg better evidence than a grid path: a grid path
    is a sequence of adjacent valid CELLS, and the segment between two cell
    centres was never checked at all.
    """
    step = cfg.planner.edge_resolution
    dist = float(np.linalg.norm((b - a) * PAGE_SCALE))
    n = max(2, int(np.ceil(dist / step)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = a + t * (b - a)
        ok, _ = classify(cfg, arm, weld, scene, float(p[0]), float(p[1]), signs)
        if not ok:
            return False
    return True


def plan_in_page(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, signs: tuple,
    start_yt: tuple, target_yt: tuple | None = None,
    goal_y: float | None = None, rng: np.random.Generator | None = None,
    extra_nodes: list | None = None,
) -> Leg | None:
    """PRM + Dijkstra inside one page, from start_yt to a target.

    Either target_yt (a specific point, e.g. a door anchor) or goal_y (any
    configuration at or beyond that weld parameter) must be given.

    Returns None when no path is found. That is NOT a proof that none exists
    -- PRM is probabilistically complete, so a failure here is INCONCLUSIVE
    and the caller must treat it as such.
    """
    wp = cfg.witness
    rng = rng or np.random.default_rng(wp.rng_seed)
    pm = cfg.pagemap

    start = np.asarray(start_yt, dtype=float)
    ok, _ = classify(cfg, arm, weld, scene, float(start[0]), float(start[1]), signs)
    if not ok:
        return None

    nodes = [start]
    if target_yt is not None:
        tgt = np.asarray(target_yt, dtype=float)
        ok_t, _ = classify(cfg, arm, weld, scene, float(tgt[0]), float(tgt[1]), signs)
        if not ok_t:
            return None
        nodes.append(tgt)
        target_index = 1
    else:
        target_index = None
    for extra in (extra_nodes or []):
        e = np.asarray(extra, dtype=float)
        ok_e, _ = classify(cfg, arm, weld, scene, float(e[0]), float(e[1]), signs)
        if ok_e:
            nodes.append(e)

    ys = rng.uniform(pm.y_min, pm.y_max, wp.prm_samples)
    ths = rng.uniform(pm.theta_min, pm.theta_max, wp.prm_samples)
    for y, th in zip(ys, ths):
        ok_s, _ = classify(cfg, arm, weld, scene, float(y), float(th), signs)
        if ok_s:
            nodes.append(np.array([y, th]))
    pts = np.asarray(nodes, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return None

    goal_set = ([target_index] if target_index is not None
                else [i for i in range(n)
                      if pts[i, 0] >= (goal_y - cfg.flood.goal_y_tol)])
    if not goal_set:
        return None

    scaled = pts * PAGE_SCALE
    order = [np.argsort(np.linalg.norm(scaled - scaled[i], axis=1))[1: wp.prm_k + 1]
             for i in range(n)]

    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[0] = 0.0
    heap = [(0.0, 0)]
    checked: dict = {}
    goal_hit = -1
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u] + 1e-12:
            continue
        if u in goal_set:
            goal_hit = u
            break
        for v in order[u]:
            v = int(v)
            w = float(np.linalg.norm(scaled[v] - scaled[u]))
            if dist[u] + w >= dist[v]:
                continue
            key = (min(u, v), max(u, v))
            if key not in checked:
                checked[key] = _edge_ok(cfg, arm, weld, scene, pts[u], pts[v], signs)
            if not checked[key]:
                continue
            dist[v] = dist[u] + w
            prev[v] = u
            heapq.heappush(heap, (dist[v], v))

    if goal_hit < 0:
        return None

    chain = [goal_hit]
    while prev[chain[-1]] >= 0:
        chain.append(int(prev[chain[-1]]))
    chain.reverse()
    waypoints = pts[chain]
    q, yy = _densify_page(cfg, arm, weld, signs, waypoints)
    if q is None:
        return None
    return Leg(kind=LEG_PAGE, page=signs, q=q, y=yy, waypoints=waypoints,
               note=f"PRM/Dijkstra, {len(waypoints)} waypoints")


def _densify_page(cfg: Config, arm: SrsArm, weld: WeldLine, signs: tuple,
                  waypoints: np.ndarray):
    """Sample each waypoint segment and solve IK at every sample."""
    wp = cfg.witness
    qs, ys = [], []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        d = float(np.linalg.norm((b - a) * PAGE_SCALE))
        n = max(2, int(np.ceil(d / wp.page_step)))
        # page_step bounds the step in PAGE coordinates, which says nothing
        # about the step in Q -- the map (y, theta) -> q stretches, badly near
        # a degeneracy. Refine until the JOINT step respects step_max, since
        # that is the bound the swept-segment certificate actually rests on.
        for _ in range(8):
            seg = []
            for t in np.linspace(0.0, 1.0, n + 1):
                p = a + t * (b - a)
                res = arm.ik_branch(*weld.pose(float(p[0])), float(p[1]), signs)
                if not res.ok:
                    return None, None
                seg.append((res.q, float(p[0])))
            qq = np.asarray([s[0] for s in seg])
            if len(qq) < 2 or float(np.abs(np.diff(qq, axis=0)).max()) <= wp.step_max:
                break
            n *= 2
        for q_i, y_i in seg[:-1]:
            qs.append(q_i)
            ys.append(y_i)
    res = arm.ik_branch(*weld.pose(float(waypoints[-1][0])),
                        float(waypoints[-1][1]), signs)
    if not res.ok:
        return None, None
    qs.append(res.q)
    ys.append(float(waypoints[-1][0]))
    return np.asarray(qs, dtype=float), np.asarray(ys, dtype=float)


def door_leg(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene,
    disc: SigmaDisc, page: tuple, partner: tuple,
) -> tuple:
    """The delta-sweep across a door, or None with the reason it is impossible.

    Uses the exact interval from certify_fiber: a crossing needs ONE connected
    admissible interval holding both delta = 0 (this page's spelling) and
    delta = +/- pi (the partner's). Limits are checked on the joint values
    actually TRAVERSED, never on a 2*pi-wrapped representative -- invariant 21.
    """
    cert = certify_fiber(cfg, arm, weld, scene, disc, page)
    if not cert.crossable:
        return None, cert
    i, k = cert.pair
    anchor_y, anchor_theta = disc.anchor
    res = arm.ik_branch(*weld.pose(anchor_y), anchor_theta, page)
    if not res.ok:
        return None, cert
    # The two moving joints change by exactly |delta|, so the joint step is
    # pi/(n-1); size n so it respects step_max rather than trusting a count.
    n_steps = max(cfg.witness.door_samples,
                  int(np.ceil(np.pi / cfg.witness.step_max)) + 1)
    deltas = np.linspace(0.0, cert.direction * np.pi, n_steps)
    q = np.repeat(res.q[None, :], len(deltas), axis=0)
    q[:, i] = res.q[i] + deltas
    q[:, k] = res.q[k] - deltas
    leg = Leg(kind=LEG_DOOR, page=page, q=q,
              y=np.full(len(deltas), float(anchor_y)),
              anchor=(float(anchor_y), float(anchor_theta)),
              deltas=deltas, partner=partner,
              note=(f"self-motion, delta 0 -> {cert.direction:+d}*pi, "
                    f"{np.degrees(cert.margin):.2f} deg of limit slack"))
    return leg, cert


# --------------------------------------------------------------------------
# Verification -- the whole point
# --------------------------------------------------------------------------


def verify_witness(
    cfg: Config, arm: SrsArm, weld: WeldLine, scene: Scene, path: WitnessPath,
    q_start: np.ndarray | None = None, goal_y: float | None = None,
) -> WitnessReport:
    """Re-check a path from scratch. Grid-free, SVM-free, resolution-free.

    Six checks:
      1. on-constraint   -- the tool stays on the weld with the right pose,
                            so this is a path ON the constraint manifold and
                            not merely in Q
      2. joint limits    -- on the values actually traversed, no wrapping
      3. collision-free  -- exact clearance at every sample
      4. step bound      -- consecutive samples are close, so the discrete
                            sample can stand in for a curve
      5. junctions       -- consecutive legs agree in Q, not just in page
                            coordinates (this is where a 2*pi wrap would show)
      6. endpoints       -- starts at q_start, ends at or beyond goal_y
    """
    wp = cfg.witness
    rep = WitnessReport()
    if not path.legs:
        rep.reason = "empty path"
        return rep

    r_max = _motion_radius(arm)
    swept_ok = True

    for li, leg in enumerate(path.legs):
        for j in range(len(leg.q)):
            q = leg.q[j]
            rep.n_configs += 1

            # (1) on-constraint
            pos, rot = weld.pose(float(leg.y[j]))
            fk_pos, fk_rot = arm.fk(q)
            e_pos = float(np.linalg.norm(fk_pos - pos))
            e_rot = float(np.linalg.norm(fk_rot - rot))
            rep.max_tool_pos_err = max(rep.max_tool_pos_err, e_pos)
            rep.max_tool_rot_err = max(rep.max_tool_rot_err, e_rot)
            if e_pos > wp.tol_pos or e_rot > wp.tol_rot:
                rep.failures.append(
                    (f"leg{li}[{j}]", f"off the constraint: {e_pos * 1e3:.4f} mm, "
                                      f"rot {e_rot:.2e}"))

            # (2) joint limits, on the traversed values
            if not arm.within_limits(q):
                bad = [f"q{m + 1}={np.degrees(q[m]):.2f}" for m in range(len(q))
                       if q[m] < arm.q_min[m] or q[m] > arm.q_max[m]]
                rep.failures.append((f"leg{li}[{j}]", f"joint limit: {', '.join(bad)}"))

            # (3) collision
            frames = arm.frames(q)
            if wp.exact_collision:
                clear, witness = exact_clearance(scene, frames)
            else:
                cr = scene.check(frames)
                clear, witness = cr.clearance, cr.witness
            rep.min_clearance = min(rep.min_clearance, clear)
            if clear < 0.0:
                rep.failures.append((f"leg{li}[{j}]", f"collision: {witness}"))

            # (4) step bound + (3b) swept-segment certificate
            if j + 1 < len(leg.q):
                dq = leg.q[j + 1] - q
                step = float(np.abs(dq).max())
                rep.max_step = max(rep.max_step, step)
                if step > wp.step_max:
                    rep.failures.append(
                        (f"leg{li}[{j}]", f"step {step:.4f} rad exceeds "
                                          f"step_max {wp.step_max}"))
                # A point at distance r from joint i's axis moves at most
                # r*|dq_i|, so the whole robot moves at most R_max*||dq||_1.
                # If that is below the clearance here, the SEGMENT is clear,
                # not just its endpoints.
                if clear < r_max * float(np.abs(dq).sum()):
                    swept_ok = False

    # (5) junctions -- in Q, where a 2*pi wrap cannot hide
    for li in range(len(path.legs) - 1):
        gap = float(np.abs(path.legs[li].q[-1] - path.legs[li + 1].q[0]).max())
        rep.max_junction_gap = max(rep.max_junction_gap, gap)
        if gap > wp.tol_join:
            rep.failures.append(
                (f"junction {li}->{li + 1}",
                 f"legs do not meet in Q: {gap:.3e} rad "
                 f"({np.degrees(gap):.4f} deg)"))

    # (6) endpoints
    if q_start is not None:
        gap = float(np.abs(path.q_start - q_start).max())
        if gap > wp.tol_join:
            rep.failures.append(("start", f"path does not start at q_start: "
                                          f"{gap:.3e} rad"))
    if goal_y is not None:
        reached = float(path.legs[-1].y[-1])
        if reached < goal_y - cfg.flood.goal_y_tol:
            rep.failures.append(("goal", f"path ends at y={reached:.4f}, "
                                         f"short of goal_y={goal_y}"))

    rep.swept_certified = bool(swept_ok and not rep.failures)
    rep.ok = not rep.failures
    rep.reason = (
        f"verified: {rep.n_configs} configurations, "
        f"{sum(1 for l in path.legs if l.kind == LEG_DOOR)} door crossing(s), "
        f"max step {rep.max_step:.4f} rad, "
        f"min clearance {rep.min_clearance * 1e3:.2f} mm"
        + ("; every segment swept-certified" if rep.swept_certified
           else "; sample-verified only (segments not swept-certified)")
    ) if rep.ok else f"{len(rep.failures)} check(s) failed"
    return rep
