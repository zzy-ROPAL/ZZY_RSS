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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.page3 import Page3, prove3  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, Page3Params, PageMapParams, SceneParams, SurfaceParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

ZGAP = (BoxObstacle(center=(0.14, 0.0, 0.38), half_extent=(0.06,) * 3, label="low"),
        BoxObstacle(center=(0.14, 0.0, 0.66), half_extent=(0.06,) * 3, label="high"))
SIGNS = (1, 1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle", choices=("none", "zgap"), default="zgap")
    ap.add_argument("--samples", type=int, default=Page3Params().n_samples)
    ap.add_argument("--seed", type=int, default=Page3Params().rng_seed)
    ap.add_argument("--dense", type=int, default=0,
                    help="also build a PRM of this size from the same start and "
                         "report whether IT reaches the goal (a cross-check)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
        scene=SceneParams(gap_open=True,
                          obstacles=ZGAP if args.obstacle == "zgap" else ()))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(SurfaceParams())
    p3 = dataclasses.replace(Page3Params(), n_samples=args.samples, rng_seed=args.seed)

    t = time.time()
    res = prove3(cfg, arm, surf, scene, SIGNS, p3, verbose=not args.quiet)
    out = dict(obstacle=args.obstacle, samples=args.samples, seed=args.seed,
               verdict=res.verdict, reason=res.reason, seconds=round(time.time() - t, 1),
               walls=res.walls, doors=res.doors, witness=res.witness,
               stats={k: v for k, v in res.stats.items()})
    if args.dense:
        t = time.time()
        dense = Page3(cfg, arm, surf, scene, SIGNS,
                      dataclasses.replace(p3, rng_seed=args.seed + 777),
                      res.stats["start"][2], n_samples=args.dense)
        out["dense_prm"] = dict(samples=args.dense, valid=len(dense.valid),
                                start_component=int((dense.comp == dense.start_comp).sum()),
                                reaches_goal=dense.reaches_goal(),
                                seconds=round(time.time() - t, 1))
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
