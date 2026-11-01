"""Scene search for EXP 4: a q6 door the robot can ACTUALLY cross, plus an
obstacle pair that leaves the wall somewhere to be.

Four screens, in increasing cost. Each one exists because the cheaper screen
above it passed something that later failed, and the comment says what.

    --mode doors    weld placements with a TRAVERSABLE q6 door.
                    Obstacle-free, so this is the cheapest screen there is.

    --mode scenes   + an obstacle that makes the EXP 4 structure:
                    the start region on page P cannot reach the goal, the
                    partner P' can, the door lies in the start region.

    --mode split    + a SECOND obstacle that gives the wall room, scored by
                    the bottleneck below.

    --mode walls    the ground truth: run Algorithms 1-4 and see whether the
                    wall actually closes with every facet verified.

WHY EACH SCREEN IS NEEDED
-------------------------
1.  TRAVERSABILITY.  A q6 door is a wrist gimbal: q6 = 0 makes the q5 and q7
    axes collinear, so (q5 + d, q7 - d) moves no link. The partner page's
    representative of that one arm shape sits at d = +/-pi, so crossing means
    traversing the fiber, and every intermediate d must be in limits:

        d: 0 -> +pi   needs   q5_0 <= L5 - pi   and   q7_0 >= -L7 + pi
        d: 0 -> -pi   needs   q5_0 >= -L5 + pi  and   q7_0 <=  L7 - pi

    With L5 = 178.85 and L7 = 168.45 deg those windows are 1.15 and 11.55 deg
    wide. docs/exp4_brief.md checked only the ENDPOINT, through `mirror_q`,
    which WRAPS q7 from -316.62 to +43.38 deg -- a value inside the limit that
    the joint cannot reach, because it would have to pass -168.45 on the way
    (findings 37). Of 1660 interior q6 doors over 5544 placements, 2233
    page/door pairs pass the real test, so this costs nothing but must be the
    first screen, not the last.

2.  ROOM FOR THE WALL.  Algorithm 3 needs every facet inside C_obs. The wall
    is a closed curve in the invalid set separating the start region R1 from
    the goal component G, and for any such curve C and any path P from R1 to
    G, C crosses P, so min_C depth <= max_P depth -- with equality attainable.
    Hence the room available to the best possible wall is

        bottleneck = min over R1->G paths of ( max depth-into-C_obs along path )

    a minimax-path value. Measured in SCALED page units (y in [0,1],
    theta/pi in [-1,1]) so it does not move with resolution.

    Screen this at res 61 or finer. At res 41 one theta-cell is already 0.05
    scaled, so the theta ~ 0 joint-limit hair -- which is 1.2-1.7 deg wide,
    under a third of a cell -- reads the same as a real collision band and the
    screen passes everything.

3.  TWO BOXES.  On every placement found by --mode scenes, R1 and G are
    separated only by that hair, and a sweep of 16128 single boxes produced
    no bottleneck above one cell. The fix is findings 24's ZGAP split: one box
    makes the structure, the other sits on the theta ~ 0 elbow track and turns
    the hair into a collision band. Collision is page-independent, so the
    second box removes those cells from P' as well, which costs nothing
    because P''s spanning route runs at theta < 0, through the door.

4.  GROUND TRUTH.  Even the bottleneck is a proxy. --mode walls trains the
    real wall; that is what decides. On the 11 two-box candidates this found,
    10 closed at 100% of facets with gamma between 2.7 and 19.7, most in a
    single Algorithm 4 iteration.

Results are written to --out as JSON and each mode reads the previous one's
file, so the modes chain:

    python3 scripts/search_exp4_scene.py --mode doors
    python3 scripts/search_exp4_scene.py --mode scenes
    python3 scripts/search_exp4_scene.py --mode split
    python3 scripts/search_exp4_scene.py --mode walls
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, find_interior_doors_analytic  # noqa: E402
from zzy_rss.fasteval import Precomputed  # noqa: E402
from zzy_rss.flood import (  # noqa: E402
    find_inf_proof_region, invalid_points, label_regions, live_discs,
    other_valid_points, region_points, thin,
)
from zzy_rss.pagemap import build_all_pages_shared  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.wall import (  # noqa: E402
    PageLabeller, sigma_disc_samples, virtual_boundary_samples,
)
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
BASE = Config()


def weld_params(x, z, half_y, tilt_deg):
    a = np.radians(tilt_deg)
    return WeldParams(p0=(x, -half_y, z), p1=(x, half_y, z),
                      approach=(float(np.sin(a)), 0.0, float(-np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def config_for(c, res, boxes=()):
    obstacles = tuple(BoxObstacle(center=tuple(cc), half_extent=tuple(hh),
                                  label=f"b{n}")
                      for n, (cc, hh) in enumerate(boxes))
    return dataclasses.replace(
        BASE, weld=weld_params(c["x"], c["z"], c["half_y"], c["tilt"]),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=obstacles),
        sigma_policy=dataclasses.replace(BASE.sigma_policy,
                                         enable_fiber_certification=True),
        wall=dataclasses.replace(BASE.wall, pd_mode="lipschitz"))


def crossing_direction(arm, q0, scene=None, n=121):
    """Which way round the gimbal fiber reaches the partner representative?

    Returns -1.0 or +1.0, or None if neither direction stays legal. This is
    the test docs/exp4_brief.md replaced with an endpoint check.
    """
    lim = arm.q_max
    for sign in (-1.0, +1.0):
        ok = True
        for d in np.linspace(0.0, sign * np.pi, n):
            q = q0.copy()
            q[4] += d
            q[6] -= d
            if not np.all(np.abs(q) <= lim):
                ok = False
                break
            if scene is not None and scene.check(arm.frames(q)).in_collision:
                ok = False
                break
        if ok:
            return sign
    return None


def bottleneck(valid, R1, G, sampling):
    """min over R1->G paths of (max depth-into-C_obs along the path).

    The largest clearance the best possible separating wall can have at its
    tightest point. See the module docstring for why this is the right
    quantity.
    """
    dV = ndimage.distance_transform_edt(~valid, sampling=sampling)
    ny, nt = valid.shape
    best = np.full(valid.shape, np.inf)
    heap = []
    for i, j in np.argwhere(R1):
        c = dV[i, j]
        if c < best[i, j]:
            best[i, j] = c
            heapq.heappush(heap, (c, int(i), int(j)))
    while heap:
        c, i, j = heapq.heappop(heap)
        if c > best[i, j]:
            continue
        if G[i, j]:
            return float(c)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nt:
                nc = max(c, dV[ni, nj])
                if nc < best[ni, nj]:
                    best[ni, nj] = nc
                    heapq.heappush(heap, (nc, ni, nj))
    return float("inf")


# ------------------------------------------------------- mode: doors ----

def _doors_one(task):
    x, z, hy, tl, res = task
    c = dict(x=float(x), z=float(z), half_y=float(hy), tilt=float(tl))
    cfg = config_for(c, res)
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    stub = SimpleNamespace(y_grid=np.linspace(0, 1, res),
                           theta_grid=np.linspace(-np.pi, np.pi, res),
                           name="stub")
    try:
        doors = find_interior_doors_analytic(cfg, arm, weld, stub, 2)
    except Exception:
        return []
    out = []
    for d in doors:
        ay, at = d.anchor
        if not (0.03 < ay < 0.97):
            continue
        for s2, s4 in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            r = arm.ik_branch(*weld.pose(ay), at, (s2, s4, 1))
            if not r.ok or not arm.within_limits(r.q):
                continue
            sign = crossing_direction(arm, r.q)
            if sign is None:
                continue
            page = ("+" if s2 > 0 else "-") + ("+" if s4 > 0 else "-")
            out.append(dict(c, page=page + "+", partner=page + "-",
                            door_y=float(ay), door_theta=float(at),
                            q5=float(np.degrees(r.q[4])),
                            q7=float(np.degrees(r.q[6])),
                            cross=float(sign)))
    return out


def mode_doors(args):
    tasks = [(x, z, hy, tl, args.res)
             for x in np.round(np.arange(0.22, 0.481, 0.02), 3)
             for z in np.round(np.arange(0.14, 0.601, 0.04), 3)
             for hy in (0.26, 0.30, 0.34)
             for tl in (0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60)]
    print(f"[doors] {len(tasks)} weld placements", flush=True)
    hits = []
    with Pool(args.jobs) as p:
        for n, r in enumerate(p.imap_unordered(_doors_one, tasks, chunksize=16)):
            hits.extend(r)
            if (n + 1) % 500 == 0:
                print(f"  {n+1}/{len(tasks)} -> {len(hits)}", flush=True)
    lim = np.degrees(SrsArm(BASE.arm).q_max)
    for h in hits:
        up = min(lim[4] - h["q5"], h["q7"] + lim[6]) - 180.0
        dn = min(h["q5"] + lim[4], lim[6] - h["q7"]) - 180.0
        h["margin"] = max(up, dn)
    hits.sort(key=lambda h: -h["margin"])
    return hits


# ------------------------------------------------------ mode: scenes ----

def _scene_structure(pre, page, partner, ij, coll):
    """The EXP 4 region structure, or None. Shared by --mode scenes/split."""
    i0, j0 = ij
    mP = pre.page_map(page, coll).valid_mask
    mPp = pre.page_map(partner, coll).valid_mask
    if not (mP[i0, j0] and mPp[i0, j0]):
        return None
    labP, _ = ndimage.label(mP)
    labPp, _ = ndimage.label(mPp)
    kP = int(labP[i0, j0])
    R1 = labP == kP
    if not R1[0, :].any() or R1[-1, :].any():
        return None                                   # starts, must not finish
    RP = labPp == int(labPp[i0, j0])
    if not (RP[0, :].any() and RP[-1, :].any()):
        return None                                   # partner must span
    if any((pre.ok & pre.limit_ok[pg] & ~coll)[:, [0, -1]].any()
           for pg in pre.limit_ok):
        return None                                   # invariant 5
    G = np.zeros_like(R1)
    for k in range(1, labP.max() + 1):
        if k != kP and (labP == k)[-1, :].any():
            G |= (labP == k)
    return mP, R1, G, kP


def _scenes_one(task):
    c, res, grid = task
    cfg = config_for(c, res)
    pre = Precomputed(cfg)
    arm = pre.arm
    q0 = arm.ik_branch(*pre.weld.pose(c["door_y"]), c["door_theta"],
                       tuple(1 if ch == "+" else -1 for ch in c["page"]))
    if not q0.ok:
        return []
    ij = (int(np.argmin(np.abs(pre.y_grid - c["door_y"]))),
          int(np.argmin(np.abs(pre.theta_grid - c["door_theta"]))))
    samp = (1.0 / (res - 1), 2.0 / (res - 1))
    out = []
    for half, centre in grid:
        boxes = ((centre, half),)
        coll = pre.collide([BoxObstacle(center=tuple(centre),
                                        half_extent=tuple(half), label="b0")])
        st = _scene_structure(pre, c["page"], c["partner"], ij, coll)
        if st is None:
            continue
        mP, R1, G, kP = st
        bn = bottleneck(mP, R1, G, samp) if G.any() else float("inf")
        sc = Scene(cfg.arm, dataclasses.replace(cfg.scene, obstacles=(
            BoxObstacle(center=tuple(centre), half_extent=tuple(half),
                        label="b0"),)))
        if crossing_direction(arm, q0.q, sc, n=91) is None:
            continue
        out.append(dict(c, boxes=[[list(centre), list(half)]],
                        bottleneck=float(bn),
                        y_max=float(pre.y_grid[np.argwhere(R1)[:, 0].max()]),
                        n_R1=int(R1.sum())))
    return out


def _box_grid(step, extents, x_range, y_range, z_range):
    xs = np.round(np.arange(*x_range, step), 3)
    ys = np.round(np.arange(*y_range, step), 3)
    zs = np.round(np.arange(*z_range, step), 3)
    return [(h, tuple(float(v) for v in c))
            for h in extents for c in itertools.product(xs, ys, zs)]


def mode_scenes(args):
    src = json.loads(Path(args.doors_file).read_text())
    seen, tasks = set(), []
    grid = _box_grid(0.06, [(0.07, 0.07, 0.07), (0.09, 0.09, 0.09),
                            (0.06, 0.13, 0.06)],
                     (-0.04, 0.50), (-0.40, 0.41), (0.08, 0.70))
    for h in src:
        key = (h["x"], h["z"], h["half_y"], h["tilt"], h["page"],
               round(h["door_y"], 4))
        if key in seen:
            continue
        seen.add(key)
        tasks.append((h, args.res, grid))
        if len(tasks) >= args.placements:
            break
    print(f"[scenes] {len(tasks)} placements x {len(grid)} boxes", flush=True)
    hits = []
    with Pool(args.jobs) as p:
        for n, r in enumerate(p.imap_unordered(_scenes_one, tasks)):
            hits.extend(r)
            print(f"  [{n+1}/{len(tasks)}] +{len(r)} -> {len(hits)}", flush=True)
    hits.sort(key=lambda d: (-d["bottleneck"], d["y_max"]))
    return hits


# ------------------------------------------------------- mode: split ----

def _split_one(task):
    c, res, grid, min_bn = task
    cfg = config_for(c, res)
    pre = Precomputed(cfg)
    arm = pre.arm
    sg = tuple(1 if ch == "+" else -1 for ch in c["page"])
    q0 = arm.ik_branch(*pre.weld.pose(c["door_y"]), c["door_theta"], sg)
    if not q0.ok:
        return []
    ij = (int(np.argmin(np.abs(pre.y_grid - c["door_y"]))),
          int(np.argmin(np.abs(pre.theta_grid - c["door_theta"]))))
    samp = (1.0 / (res - 1), 2.0 / (res - 1))
    b1c, b1h = c["boxes"][0]
    out = []
    for half, centre in grid:
        obs = [BoxObstacle(center=tuple(b1c), half_extent=tuple(b1h),
                           label="cut"),
               BoxObstacle(center=tuple(centre), half_extent=tuple(half),
                           label="split")]
        coll = pre.collide(obs)
        st = _scene_structure(pre, c["page"], c["partner"], ij, coll)
        if st is None:
            continue
        mP, R1, G, kP = st
        bn = bottleneck(mP, R1, G, samp) if G.any() else float("inf")
        if bn < min_bn:
            continue
        sc = Scene(cfg.arm, dataclasses.replace(cfg.scene, obstacles=tuple(obs)))
        if crossing_direction(arm, q0.q, sc, n=91) is None:
            continue
        out.append(dict(c, boxes=[[list(b1c), list(b1h)],
                                  [list(centre), list(half)]],
                        bottleneck=float(bn),
                        y_max=float(pre.y_grid[np.argwhere(R1)[:, 0].max()]),
                        n_R1=int(R1.sum())))
    return out


def mode_split(args):
    src = json.loads(Path(args.scenes_file).read_text())
    if args.res < 61:
        print("[split] WARNING: res < 61 cannot tell the theta~0 hair from a "
              "real band -- one theta-cell is already 0.05 scaled at res 41.")
    # Aimed at the theta = 0 elbow and wrist tracks, which is where the hair
    # lives; see the module docstring, screen 3.
    grid = _box_grid(0.04, [(0.05, 0.05, 0.05), (0.06, 0.06, 0.06),
                            (0.08, 0.08, 0.08), (0.05, 0.20, 0.05),
                            (0.04, 0.14, 0.04)],
                     (0.06, 0.47), (-0.34, 0.35), (0.18, 0.63))
    seen, tasks = set(), []
    for h in src:
        key = (h["x"], h["z"], h["half_y"], h["tilt"], h["page"],
               tuple(map(tuple, h["boxes"][0])))
        if key in seen:
            continue
        seen.add(key)
        tasks.append((h, args.res, grid, args.min_bottleneck))
        if len(tasks) >= args.placements:
            break
    print(f"[split] {len(tasks)} scenes x {len(grid)} second boxes, "
          f"bottleneck >= {args.min_bottleneck} at res {args.res}", flush=True)
    hits = []
    with Pool(args.jobs) as p:
        for n, r in enumerate(p.imap_unordered(_split_one, tasks)):
            hits.extend(r)
            print(f"  [{n+1}/{len(tasks)}] +{len(r)} -> {len(hits)}", flush=True)
    hits.sort(key=lambda d: (-d["bottleneck"], d["y_max"]))
    return hits


# ------------------------------------------------------- mode: walls ----

def _wall_one(task):
    c, res, iters, fbm = task
    cfg = config_for(c, res, boxes=[tuple(map(tuple, b)) for b in c["boxes"]])
    cfg = dataclasses.replace(
        cfg, wall=dataclasses.replace(cfg.wall, feedback_margin=fbm))
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)
    built = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
    pms = {n: built[n] for n in PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    P, Pp = c["page"], c["partner"]
    sg = tuple(1 if ch == "+" else -1 for ch in P)
    cands = [d for d in sig.get(P, []) if 2 in d.vanishing]
    if not cands:
        return dict(c, ok=False, why="no q6 door")
    d = min(cands, key=lambda z: abs(z.anchor[0] - c["door_y"]))
    ay, at = d.anchor
    pmA, pmB = pms[P], pms[Pp]
    labA, regsA = label_regions(cfg, pmA)
    labB, regsB = label_regions(cfg, pmB)
    i = int(np.argmin(np.abs(pmA.y_grid - ay)))
    j = int(np.argmin(np.abs(pmA.theta_grid - at)))
    kA, kB = int(labA[i, j]), int(labB[i, j])
    if kA == 0 or kB == 0:
        return dict(c, ok=False, why="door cell invalid")
    rA = next(r for r in regsA if r.index == kA)
    rB = next(r for r in regsB if r.index == kB)
    if not (rA.reaches_start and not rA.reaches_goal and rB.reaches_goal):
        return dict(c, ok=False, why="region structure")
    q0 = arm.ik_branch(*weld.pose(ay), at, sg)
    if crossing_direction(arm, q0.q, scene, n=121) is None:
        return dict(c, ok=False, why="door not traversable")
    rng = np.random.default_rng(cfg.planner.rng_seed)
    pts = region_points(pmA, labA, kA)
    live, _ = live_discs(cfg, arm, weld, scene, sg, sig[P], verbose=False)
    tol = float(np.hypot(pmA.y_grid[1] - pmA.y_grid[0],
                         pmA.theta_grid[1] - pmA.theta_grid[0]))
    se, re_ = sigma_disc_samples(live, pts, tol)
    seed = thin(pts, 450, rng)
    rest = np.vstack([thin(other_valid_points(pmA, labA, kA, regsA,
                                              cfg.wall.rest_mode), 450, rng),
                      virtual_boundary_samples(cfg, 80)])
    if len(se):
        seed = np.vstack([seed, se])
    if len(re_):
        rest = np.vstack([rest, re_])
    inv = thin(invalid_points(pmA), 150, rng)
    lb = PageLabeller(cfg, arm, weld, scene, sg, pmA, labA, kA,
                      search_cells=cfg.wall.label_search_cells,
                      motion_samples=cfg.wall.label_motion_samples)
    poly, rep, w, it, _ = find_inf_proof_region(
        cfg, arm, weld, scene, sg, seed, rest, inv, live, max_iters=iters,
        verbose=False, labeller=lb)
    return dict(c, ok=bool(rep.ok and poly.closed), why="",
                facets=[rep.n_passed, rep.n_facets], closed=bool(poly.closed),
                gamma=float(w.gamma), accuracy=float(w.accuracy), alg4_iters=it,
                kA=kA, kB=kB, y_max=rA.y_max, n_R1=rA.n_cells)


def mode_walls(args):
    src = json.loads(Path(args.split_file).read_text())
    tasks = [(c, args.res, args.iters, args.feedback_margin)
             for c in src[:args.placements]]
    print(f"[walls] training {len(tasks)} walls at res {args.res}, "
          f"iters {args.iters}, feedback_margin {args.feedback_margin}",
          flush=True)
    good = []
    with Pool(args.jobs) as p:
        for n, r in enumerate(p.imap_unordered(_wall_one, tasks)):
            f = r.get("facets")
            print(f"  [{n+1}/{len(tasks)}] boxes={r['boxes']}  "
                  f"{('%d/%d' % (f[0], f[1])) if f else '-':>9}  "
                  f"gamma={r.get('gamma', float('nan')):7.1f}  "
                  + ("*** CLOSED 100% ***" if r.get("ok")
                     else (r.get("why") or "incomplete")), flush=True)
            if r.get("ok"):
                good.append(r)
    good.sort(key=lambda r: r["gamma"])
    print(f"\n[walls] {len(good)} scene(s) with a fully verified wall")
    for r in good:
        print(f"  gamma={r['gamma']:6.1f}  {r['facets'][0]}/{r['facets'][1]} "
              f"facets  alg4_iters={r['alg4_iters']}  y_max={r['y_max']:.3f}  "
              f"boxes={r['boxes']}")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="doors",
                    choices=("doors", "scenes", "split", "walls"))
    ap.add_argument("--res", type=int, default=0,
                    help="0 picks the mode's default: 61 for doors/split/"
                         "walls, 41 for the scenes sweep")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--placements", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--feedback-margin", type=float, default=0.019)
    ap.add_argument("--min-bottleneck", type=float, default=0.07,
                    help="scaled page units; one theta-cell at res 61 is "
                         "0.033, and the theta~0 hair never exceeds that")
    ap.add_argument("--out", default="out/exp4_search")
    args = ap.parse_args()
    if not args.res:
        args.res = 41 if args.mode == "scenes" else 61
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.doors_file = out / "doors.json"
    args.scenes_file = out / "scenes.json"
    args.split_file = out / "split.json"

    fn = {"doors": mode_doors, "scenes": mode_scenes,
          "split": mode_split, "walls": mode_walls}[args.mode]
    hits = fn(args)
    path = out / f"{args.mode}.json"
    path.write_text(json.dumps(hits, indent=1, default=str))
    print(f"\n[out] {len(hits)} result(s) -> {path}")


if __name__ == "__main__":
    main()
