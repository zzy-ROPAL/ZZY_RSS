"""Targeted EXP 4 scene search: maximise the SEPARATION of the two classes.

docs/exp4_brief.md's four conditions decide whether a scene is a valid EXP 4.
They say nothing about whether the WALL can be built, and on the scene it
picked the wall is close to impossible:

    EXP 3  page ++-  start region 1   nearest seed<->rest distance  0.2539
    EXP 4  page -++  start region 1   nearest seed<->rest distance  0.0667

An RBF needs gamma ~ 1/d^2 to resolve two classes d apart, so EXP 3 needs ~16
(it uses 1.0) while EXP 4 needs ~225 -- and measured runs pin gamma at the 400
cap with accuracy below 1, i.e. the wall stops separating its own training
data (findings 29). That is invariant 14's "hard page" exactly.

So this search adds condition (E): maximise the distance between the start
region and the goal-reaching components. The weld placement is kept, which
keeps the q6 door where the brief verified it; only the obstacle moves.

Screening is at low resolution and uses grid connectivity, which is NOT an
infeasibility criterion (invariant 8) and gives false positives at a single
resolution (invariant 23, findings 34e: 13 of 13 evaporated between res 41 and
61). Survivors are therefore re-checked at two further resolutions before
anything is reported.

    python3 -u scripts/search_exp4.py --res 41 --verify-res 61,81
"""
from __future__ import annotations
import argparse, dataclasses, itertools, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zzy_rss.params import (Config, WeldParams, PageMapParams, SceneParams,
                            BoxObstacle)
from zzy_rss.arm import SrsArm
from zzy_rss.weld import WeldLine
from zzy_rss.scene import Scene
from zzy_rss.fasteval import Precomputed
from zzy_rss.flood import label_regions, region_points, other_valid_points
from zzy_rss.doors import build_sigma, door_fiber_validity

A = np.radians(45.0)
WELD = WeldParams(p0=(0.40, -0.34, 0.22), p1=(0.40, 0.34, 0.22),
                  approach=(float(np.sin(A)), 0.0, -float(np.cos(A))),
                  reference=(float(np.cos(A)), 0.0, float(np.sin(A))))
START_PAGE, PARTNER_PAGE = "-++", "-+-"
SCALE = np.array([1.0, 1.0 / np.pi])


def make_cfg(res, obs=()):
    base = Config()
    return dataclasses.replace(
        base, weld=WELD, pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=tuple(obs)),
        wall=dataclasses.replace(base.wall, rest_mode="goal_only"))


def start_thetas(pm, lab, k):
    """Which arm angles at y = 0 land in region k -- the concrete start postures."""
    i = 0
    js = np.where(lab[i, :] == k)[0]
    return [round(float(pm.theta_grid[j]), 3) for j in js]


def evaluate(cfg, pms, discs, arm, weld, scene, fiber_n):
    """Conditions A-E for every start-touching region that cannot finish.

    (A) the region holding the start posture does NOT reach the goal
    (B) a q6 door lies in it and is OPEN
    (C) the door's partner page has a region that DOES reach the goal
    (E) the separation between that region and the goal-reaching components

    The weak proposition (invariant 16c / findings 36): it is enough that the
    ONE region holding the start posture is blocked, not every region touching
    y = 0. So each such region is a candidate scenario in its own right, and
    the arm angles that land in it are reported so the posture is concrete.
    """
    pmA = pms[START_PAGE]
    labA, regsA = label_regions(cfg, pmA)
    rest = None
    out = []
    tol = float(np.hypot(pmA.y_grid[1] - pmA.y_grid[0],
                         pmA.theta_grid[1] - pmA.theta_grid[0]))
    _, regsB = label_regions(cfg, pms[PARTNER_PAGE])
    if not any(r.reaches_start and r.reaches_goal for r in regsB):
        return []                                  # (C) partner cannot finish
    for k in regsA:
        if not (k.reaches_start and not k.reaches_goal):
            continue                               # (A)
        if k.n_cells < 50:
            continue                               # ignore specks
        seed = region_points(pmA, labA, k.index)
        if rest is None:
            rest = other_valid_points(pmA, labA, k.index, regsA, "goal_only")
        if len(rest) == 0 or len(seed) == 0:
            continue
        doors = []
        for d in discs:
            if 2 not in d.vanishing:               # q6 doors only (invariant 21)
                continue
            c = np.array(d.anchor)
            if float(np.min(np.linalg.norm(seed - c, axis=1))) > max(tol, d.radius):
                continue
            f = door_fiber_validity(cfg, arm, weld, scene, d, pmA.signs, n=fiber_n)
            if f["ok"]:
                doors.append(dict(y=round(float(c[0]), 4),
                                  theta=round(float(c[1]), 4),
                                  fiber=round(float(f["valid_fraction"]), 3)))
        if not doors:
            continue                               # (B)
        d_min = float(cKDTree(rest * SCALE).query(seed * SCALE)[0].min())
        out.append(dict(region=k.index, n_cells=k.n_cells,
                        y_max=round(k.y_max, 4), d_min=round(d_min, 4),
                        gamma_needed=round(1.0 / d_min ** 2, 1),
                        n_seed=len(seed), n_rest=len(rest),
                        start_thetas=start_thetas(pmA, labA, k.index)[:6],
                        doors=doors,
                        frac_valid=round(float(pmA.valid_mask.mean()), 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=41)
    ap.add_argument("--verify-res", dest="verify", default="61,81")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--fiber", type=int, default=73,
                    help="fiber samples during the screen; the verify pass "
                         "uses the full default")
    ap.add_argument("--out", default="out/search_exp4.json")
    args = ap.parse_args()

    cfg = make_cfg(args.res)
    arm, wl = SrsArm(cfg.arm), WeldLine(cfg.weld)
    t0 = time.perf_counter()
    pre = Precomputed(cfg)
    # Door POSITIONS depend only on the weld, never on the obstacle, so the
    # analytic finder runs once. (A fasteval PageMap leaves q2/q4/q6 NaN, so
    # build_sigma returns the analytic doors only -- which is the primary
    # finder anyway; the grid scan is only its cross-check.)
    discs = build_sigma(cfg, arm, wl, {START_PAGE: pre.page_map(
        START_PAGE, np.zeros(pre.shape, dtype=bool))},
        verbose=False)[START_PAGE]
    nq6 = sum(1 for d in discs if 2 in d.vanishing)
    print(f"[search] precomputed res {args.res} in "
          f"{time.perf_counter()-t0:.1f} s; {len(discs)} discs on "
          f"{START_PAGE}, {nq6} of them q6", flush=True)

    # Obstacle grid. The brief complains the chosen cube sits almost on the
    # base (x=0.095, shoulder at x=0), so push the sweep outward.
    xs = np.round(np.arange(0.10, 0.34, 0.03), 3)
    ys = np.round(np.arange(0.12, 0.42, 0.03), 3)
    zs = np.round(np.arange(0.18, 0.60, 0.04), 3)
    hs = (0.05, 0.07, 0.09)
    cands = list(itertools.product(xs, ys, zs, hs))
    print(f"[search] {len(cands)} candidate obstacles", flush=True)

    hits, t0 = [], time.perf_counter()
    for n, (x, y, z, h) in enumerate(cands, 1):
        obs = (BoxObstacle(center=(float(x), float(y), float(z)),
                           half_extent=(h, h, h), label="e4"),)
        scene_c = Scene(cfg.arm, dataclasses.replace(cfg.scene, obstacles=obs))
        for rec in evaluate(cfg, pre.all_pages(pre.collide(obs)), discs,
                            arm, wl, scene_c, args.fiber):
            rec.update(center=[float(x), float(y), float(z)], half=h)
            hits.append(rec)
        if n % 250 == 0:
            print(f"[search]   {n}/{len(cands)}  hits {len(hits)}  "
                  f"({time.perf_counter()-t0:.0f} s)", flush=True)
    print(f"[search] screen done: {len(hits)} pass (A)+(B)+(C) "
          f"in {time.perf_counter()-t0:.0f} s", flush=True)
    if not hits:
        json.dump([], open(args.out, "w")); return

    hits.sort(key=lambda r: -r["d_min"])
    print(f"\n[search] best {min(args.top, len(hits))} by separation "
          f"(EXP 4's own scene scores d_min=0.0667):", flush=True)
    for r in hits[:args.top]:
        print(f"   c={r['center']} h={r['half']}  d_min={r['d_min']:.4f}  "
              f"gamma~{r['gamma_needed']:.0f}  y_max={r['y_max']:.3f}  "
              f"cells={r['n_cells']}  valid={r['frac_valid']*100:.0f}%",
              flush=True)

    # --- re-check the leaders at higher resolution (invariant 23) ---------
    print(f"\n[search] re-checking the top {args.top} at res "
          f"{args.verify} and testing the q6 door", flush=True)
    keep = []
    for r in hits[:args.top]:
        obs = (BoxObstacle(center=tuple(r["center"]),
                           half_extent=(r["half"],) * 3, label="e4"),)
        rows, ok = [], True
        for rv in [int(v) for v in args.verify.split(",")]:
            c2 = make_cfg(rv, obs)
            arm, wl, sc = SrsArm(c2.arm), WeldLine(c2.weld), Scene(c2.arm, c2.scene)
            p2 = Precomputed(dataclasses.replace(c2, scene=dataclasses.replace(
                c2.scene, obstacles=())))
            pms = p2.all_pages(p2.collide(obs))
            rec = evaluate(c2, pms)
            if rec is None:
                ok = False; rows.append((rv, None)); break
            rows.append((rv, rec))
        if not ok:
            print(f"   c={r['center']} h={r['half']}: EVAPORATED at higher res "
                  f"(invariant 23)", flush=True); continue
        # (B)+(D): is a q6 door open and inside the start region?
        rv = int(args.verify.split(",")[0])
        c2 = make_cfg(rv, obs)
        arm, wl, sc = SrsArm(c2.arm), WeldLine(c2.weld), Scene(c2.arm, c2.scene)
        p2 = Precomputed(dataclasses.replace(c2, scene=dataclasses.replace(
            c2.scene, obstacles=())))
        pms = p2.all_pages(p2.collide(obs))
        sig = build_sigma(c2, arm, wl, pms, verbose=False)
        labA, regsA = label_regions(c2, pms[START_PAGE])
        kk = rows[0][1]["region"]
        pts = region_points(pms[START_PAGE], labA, kk)
        tol = float(np.hypot(pms[START_PAGE].y_grid[1]-pms[START_PAGE].y_grid[0],
                             pms[START_PAGE].theta_grid[1]-pms[START_PAGE].theta_grid[0]))
        doors = []
        for d in sig.get(START_PAGE, []):
            if 2 not in d.vanishing:
                continue
            c = np.array(d.anchor)
            if float(np.min(np.linalg.norm(pts - c, axis=1))) > max(tol, d.radius):
                continue
            f = door_fiber_validity(c2, arm, wl, sc, d, pms[START_PAGE].signs)
            if f["ok"]:
                doors.append(dict(y=round(float(c[0]), 4),
                                  theta=round(float(c[1]), 4),
                                  fiber=round(float(f["valid_fraction"]), 3)))
        r["verify"] = {str(a): b for a, b in rows}
        r["q6_doors_in_start_region"] = doors
        keep.append(r)
        print(f"   c={r['center']} h={r['half']}: survives; "
              f"d_min(res {rows[0][0]})={rows[0][1]['d_min']:.4f}  "
              f"q6 doors in start region: {len(doors)}  {doors[:2]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(keep, open(args.out, "w"), indent=2)
    print(f"\n[out] {args.out}  ({len(keep)} survivors)", flush=True)


if __name__ == "__main__":
    main()
