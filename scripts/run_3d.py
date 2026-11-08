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
from zzy_rss.flood3 import run_flood3  # noqa: E402
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
    args = ap.parse_args()

    obstacles = () if args.obstacle == "none" else {
        "exp2": ZGAP, "exp3": ZGAP_Y, "exp4": EXP4_OBS}[args.patch]
    signs = (tuple(1 if c == "+" else -1 for c in args.start_page)
             if args.start_page else START_SIGNS[args.patch])
    if args.high_z is not None and args.patch == "exp2" and obstacles:
        obstacles = tuple(dataclasses.replace(b, center=(b.center[0], b.center[1], args.high_z))
                          if b.label == "high" else b for b in obstacles)
    cfg = dataclasses.replace(
        Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
        scene=SceneParams(gap_open=True, obstacles=obstacles))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    sp = PATCHES[args.patch]
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
    if args.start_u is not None:
        p3 = dataclasses.replace(p3, start_u=args.start_u)
    if args.start_v is not None:
        p3 = dataclasses.replace(p3, start_v=args.start_v)

    t = time.time()
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
