#!/usr/bin/env python3
"""First d = 3 experiment: a flat weld PATCH, single-page proof (findings 42).

The patch is EXP 2's weld line (x = 0.28, z = 0.46, vertical torch) swept
+/-5 cm in z, so v = 0.5 is EXP 2 itself. The task: from one start posture at
the u = 0 edge, reach the u = 1 edge anywhere on the patch, tool on the patch
throughout. The page is (u, v, theta) on sign branch (++-).

    python3 scripts/run_3d.py --obstacle zgap            # INFEASIBLE
    python3 scripts/run_3d.py --obstacle none            # FEASIBLE
    python3 scripts/run_3d.py --obstacle zgap --dense 20000   # + PRM cross-check

Output: one JSON line with the verdict, the wall, the doors and timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.budget import escalate  # noqa: E402
from zzy_rss.flood3 import page_name, run_flood3  # noqa: E402
from zzy_rss.page3 import Page3, prove3  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, Page3Params, PageMapParams, SceneParams, SurfaceParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

ZGAP = (BoxObstacle(center=(0.14, 0.0, 0.38), half_extent=(0.06,) * 3, label="low"),
        BoxObstacle(center=(0.14, 0.0, 0.66), half_extent=(0.06,) * 3, label="high"))
# EXP 3's scene: ZGAP shifted +0.10 in y; its weld (x 0.32, z 0.40, 15 deg
# torch) swept +/-5 cm in z. Its q2 door curve crosses the patch and is LIVE
# under the posture rule (findings 43), so the proof has to cross pages.
ZGAP_Y = tuple(BoxObstacle(center=(b.center[0], b.center[1] + 0.10, b.center[2]),
                           half_extent=b.half_extent, label=b.label) for b in ZGAP)
# EXP 4's scene (docs/exp4_brief.md): one 14 cm cube; start page (-++) stops
# short of the goal, and a q6 door leads to (-+-), which reaches it. The
# brief's "door not crossable" (findings 37) is the atan2(0, 0) artifact;
# straight-through crossings verify (findings 43).
EXP4_OBS = (BoxObstacle(center=(0.095, 0.29, 0.377), half_extent=(0.07, 0.07, 0.07),
                        label="exp4"),)
_T4 = np.radians(45.0)
_T = np.radians(15.0)
PATCHES = {
    "exp4": SurfaceParams(p0=(0.40, -0.34, 0.21), p1=(0.40, 0.34, 0.21),
                          sweep=(0.0, 0.0, 0.02),
                          approach=(float(np.sin(_T4)), 0.0, -float(np.cos(_T4))),
                          reference=(float(np.cos(_T4)), 0.0, float(np.sin(_T4)))),
    "exp2": SurfaceParams(),
    "exp3": SurfaceParams(p0=(0.32, -0.26, 0.35), p1=(0.32, 0.26, 0.35),
                          sweep=(0.0, 0.0, 0.10),
                          approach=(float(np.sin(_T)), 0.0, -float(np.cos(_T))),
                          reference=(float(np.cos(_T)), 0.0, float(np.sin(_T)))),
}
START_SIGNS = {"exp2": (1, 1, -1), "exp3": (1, 1, -1), "exp4": (-1, 1, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", choices=sorted(PATCHES), default="exp2")
    ap.add_argument("--obstacle", choices=("none", "zgap"), default="zgap")
    ap.add_argument("--flood", action="store_true",
                    help="cross-page flood (flood3) instead of the single-page proof")
    ap.add_argument("--sweep", type=float, default=None,
                    help="override the patch's z sweep (m), centred on the same line")
    ap.add_argument("--high-z", type=float, default=None,
                    help="move ZGAP's high box to this centre height (exp2 patch); "
                         "at 0.66 its 7.8 mm overlap with the door posture blocks "
                         "every q2 door, higher frees them")
    ap.add_argument("--tilt", type=float, default=None,
                    help="torch tilt in degrees (exp2 patch; 0 = vertical). A vertical "
                         "torch puts q5 at +/-180 deg at every q2 door on an s6=- page")
    ap.add_argument("--start-page", default=None,
                    help="sign branch of the start posture, e.g. +++ (default per patch)")
    ap.add_argument("--start-u", type=float, default=None)
    ap.add_argument("--start-v", type=float, default=None)
    ap.add_argument("--start-theta", type=float, default=None,
                    help="explicit start arm angle; default: middle of the widest "
                         "valid theta interval at (start_u, start_v)")
    ap.add_argument("--samples", type=int, default=Page3Params().n_samples)
    ap.add_argument("--seed", type=int, default=Page3Params().rng_seed)
    ap.add_argument("--dense", type=int, default=0,
                    help="also build a PRM of this size from the same start and "
                         "report whether IT reaches the goal (a cross-check)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--scene", default=None,
                    help="custom patch 'x,z,tilt,roll[,half_y,sweep]' (search_scene3.surface); "
                         "overrides --patch geometry and obstacles")
    ap.add_argument("--box", action="append", default=[],
                    help="obstacle 'cx,cy,cz,hx,hy,hz' (repeatable; with --scene)")
    ap.add_argument("--two-page", action="store_true",
                    help="findings 45: prove the start page ALONE cannot reach the goal "
                         "(doors as exits), then run the cross-page flood")
    ap.add_argument("--save", default=None,
                    help="with --two-page: pickle the wall, the verified path and the "
                         "scene for scripts/fig_two_page.py")
    ap.add_argument("--escalate", action="store_true",
                    help="re-run on a ladder of more samples / new seeds until "
                         "conclusive (findings 44)")
    ap.add_argument("--no-budget", action="store_true",
                    help="disable the fail-fast budgets (findings 44)")
    ap.add_argument("--alg4-iters", type=int, default=None,
                    help="Algorithm 4 iterations per wall (Page3Params.alg4_iters)")
    ap.add_argument("--gamma-search", choices=("linear", "gallop"), default="linear",
                    help="Algorithm 1's gamma search (findings 45)")
    ap.add_argument("--wall-seconds", type=float, default=None,
                    help="per-wall budget (BudgetParams.wall_seconds)")
    ap.add_argument("--no-batch", action="store_true",
                    help="scalar validity loops instead of batch.py (findings 44)")
    args = ap.parse_args()

    obstacles = () if args.obstacle == "none" else {
        "exp2": ZGAP, "exp3": ZGAP_Y, "exp4": EXP4_OBS}[args.patch]
    if args.scene:
        obstacles = tuple(BoxObstacle(center=tuple(v[:3]), half_extent=tuple(v[3:6]),
                                      label=f"box{i}")
                          for i, v in enumerate([[float(c) for c in b.split(",")]
                                                 for b in args.box]))
    signs = (tuple(1 if c == "+" else -1 for c in args.start_page)
             if args.start_page else START_SIGNS[args.patch])
    if args.high_z is not None and args.patch == "exp2" and obstacles:
        obstacles = tuple(dataclasses.replace(b, center=(b.center[0], b.center[1], args.high_z))
                          if b.label == "high" else b for b in obstacles)
    cfg = dataclasses.replace(
        Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
        scene=SceneParams(gap_open=True, obstacles=obstacles))
    if args.no_batch:
        cfg = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner,
                                                                   batch_classify=False))
    if args.no_budget:
        cfg = dataclasses.replace(cfg, budget=dataclasses.replace(cfg.budget, enabled=False))
    if args.wall_seconds is not None:
        cfg = dataclasses.replace(cfg, budget=dataclasses.replace(
            cfg.budget, wall_seconds=args.wall_seconds))
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall,
                                                            gamma_search=args.gamma_search))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    sp = PATCHES[args.patch]
    if args.scene:
        from search_scene3 import surface
        sp = surface(*[float(c) for c in args.scene.split(",")])
    if args.tilt is not None:
        a = np.radians(args.tilt)
        sp = dataclasses.replace(sp, approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                                 reference=(float(np.cos(a)), 0.0, float(np.sin(a))))
    if args.sweep is not None:
        zc = sp.p0[2] + 0.5 * sp.sweep[2]
        sp = dataclasses.replace(sp, p0=(sp.p0[0], sp.p0[1], zc - 0.5 * args.sweep),
                                 p1=(sp.p1[0], sp.p1[1], zc - 0.5 * args.sweep),
                                 sweep=(0.0, 0.0, args.sweep))
    surf = WeldSurface(sp)
    p3 = dataclasses.replace(Page3Params(), n_samples=args.samples, rng_seed=args.seed)
    if args.alg4_iters is not None:
        p3 = dataclasses.replace(p3, alg4_iters=args.alg4_iters)
    if args.start_u is not None:
        p3 = dataclasses.replace(p3, start_u=args.start_u)
    if args.start_v is not None:
        p3 = dataclasses.replace(p3, start_v=args.start_v)

    t = time.time()
    if args.two_page:
        alone = prove3(cfg, arm, surf, scene, signs, p3, start_theta=args.start_theta,
                       verbose=not args.quiet, doors_as_exits=True)
        t1 = time.time()
        task = run_flood3(cfg, arm, surf, scene, signs, p3, start_theta=args.start_theta,
                          verbose=not args.quiet)
        out = dict(mode="two-page", scene=args.scene, boxes=args.box, patch=args.patch,
                   start_page=page_name(signs), samples=args.samples, seed=args.seed,
                   page_alone=dict(verdict=alone.verdict, scope=alone.scope,
                                   reason=alone.reason, walls=alone.walls,
                                   doors=alone.doors, seconds=round(t1 - t, 1),
                                   stats=alone.stats),
                   task=dict(verdict=task.verdict, reason=task.reason,
                             regions=task.regions, crossings=task.crossings,
                             walls=task.certificate, witness=task.witness,
                             seconds=round(time.time() - t1, 1), stats=task.stats))
        if args.save:
            import pickle
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            poly = alone.poly
            legs = ([dict(q=leg.q, y=leg.y, page=tuple(leg.page), note=leg.note)
                     for leg in task.path.legs] if task.path is not None else [])
            with open(args.save, "wb") as fh:
                pickle.dump(dict(out=out, scene=args.scene, boxes=args.box, signs=signs,
                                 samples=args.samples, seed=args.seed,
                                 start=alone.stats.get("start"),
                                 wall=None if poly is None or poly.vertices is None else
                                 dict(vertices=poly.vertices, simplices=poly.simplices),
                                 legs=legs), fh)
        print(json.dumps(out, default=str))
        return
    if args.escalate:
        def attempt(n, seed, bp):
            c = dataclasses.replace(cfg, budget=bp)
            q = dataclasses.replace(p3, n_samples=n, rng_seed=seed)
            if args.flood:
                return run_flood3(c, arm, surf, scene, signs, q,
                                  start_theta=args.start_theta, verbose=not args.quiet)
            return prove3(c, arm, surf, scene, signs, q, start_theta=args.start_theta,
                          verbose=not args.quiet)
        res, log = escalate(attempt, cfg.budget, args.samples, args.seed)
        out = dict(patch=args.patch, obstacle=args.obstacle, mode="escalate",
                   flood=args.flood, verdict=res.verdict, reason=res.reason,
                   seconds=round(time.time() - t, 1), escalation=log,
                   stats={k: v for k, v in res.stats.items() if k != "escalation"})
        print(json.dumps(out, default=str))
        return
    if args.flood:
        res = run_flood3(cfg, arm, surf, scene, signs, p3, start_theta=args.start_theta,
                         verbose=not args.quiet)
        out = dict(patch=args.patch, start_page=args.start_page, sweep=sp.sweep[2],
                   high_z=args.high_z, tilt=args.tilt,
                   obstacle=args.obstacle,
                   samples=args.samples, seed=args.seed, mode="flood", verdict=res.verdict, reason=res.reason,
                   seconds=round(time.time() - t, 1), regions=res.regions,
                   walls=res.certificate, crossings=res.crossings, doors=res.doors,
                   witness=res.witness, stats=res.stats)
        print(json.dumps(out, default=str))
        return
    res = prove3(cfg, arm, surf, scene, signs, p3, start_theta=args.start_theta,
                     verbose=not args.quiet)
    out = dict(patch=args.patch, obstacle=args.obstacle, samples=args.samples,
               seed=args.seed, verdict=res.verdict, reason=res.reason,
               seconds=round(time.time() - t, 1),
               walls=res.walls, doors=res.doors, witness=res.witness,
               stats={k: v for k, v in res.stats.items()})
    if args.dense:
        t = time.time()
        dense = Page3(cfg, arm, surf, scene, signs,
                      dataclasses.replace(p3, rng_seed=args.seed + 777),
                      res.stats["start"][2], n_samples=args.dense)
        out["dense_prm"] = dict(samples=args.dense, valid=len(dense.valid),
                                start_component=int((dense.comp == dense.start_comp).sum()),
                                reaches_goal=dense.reaches_goal(),
                                seconds=round(time.time() - t, 1))
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
