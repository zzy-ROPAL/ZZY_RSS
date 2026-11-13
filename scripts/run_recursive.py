#!/usr/bin/env python3
"""The recursive infeasibility proof (docs/recursive_proof_design.md) on 3-D weld patches.

Doors come from the chart's singularity function sigma (no door list):

    python3 scripts/run_recursive.py --case exp2-none     # FEASIBLE (control)
    python3 scripts/run_recursive.py --case exp2-zgap     # INFEASIBLE, one leaf
    python3 scripts/run_recursive.py --case exp3-new      # FEASIBLE through the q6 door
    python3 scripts/run_recursive.py --case two-wall      # INFEASIBLE, the proof splits at the door

Prints the proof tree and one JSON line; --save writes the certificate.
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_3d import PATCHES, ZGAP  # noqa: E402
from search_scene3 import surface  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.params import (BoxObstacle, Config, Page3Params, PageMapParams,  # noqa: E402
                            SceneParams)
from zzy_rss.proof import prove_task, write_certificate, write_geometry  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402


def _box(c, h, label):
    return BoxObstacle(center=c, half_extent=h, label=label)


BOX45 = _box((0.2, 0.0, 0.34), (0.06, 0.06, 0.06), "box0")
BOX46 = _box((0.3, 0.2, 0.3), (0.06, 0.06, 0.06), "box1")
CASES = {
    # name: (surface params, obstacles, start signs, what today's pipeline says)
    "exp2-none": (PATCHES["exp2"], (), (1, 1, -1), "FEASIBLE (prove3, findings 42)"),
    "exp2-zgap": (PATCHES["exp2"], ZGAP, (1, 1, -1),
                  "INFEASIBLE, one wall 2014-2156 triangles (findings 42)"),
    "exp3-new": (surface(0.40, 0.21, 45, 305), (BOX45,), (1, 1, 1),
                 "FEASIBLE across the q6 door (run_flood3, findings 45)"),
    "two-wall": (surface(0.40, 0.21, 45, 305), (BOX45, BOX46), (1, 1, 1),
                 "INFEASIBLE, four walls (run_flood3, findings 46)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alg4-iters", type=int, default=8)
    ap.add_argument("--max-checks", type=int, default=24)
    ap.add_argument("--wall-seconds", type=float, default=1200.0)
    ap.add_argument("--run-seconds", type=float, default=7200.0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--save-geometry", default=None,
                    help="pickle with every wall mesh and the verified path (for figures)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--also-prove", action="store_true",
                    help="on a FEASIBLE task, run the recursion anyway to show where it "
                         "stops (it must not prove)")
    args = ap.parse_args()

    sp, obstacles, signs, today = CASES[args.case]
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=tuple(obstacles)))
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall, gamma_search="gallop"))
    cfg = dataclasses.replace(cfg, budget=dataclasses.replace(
        cfg.budget, wall_seconds=args.wall_seconds, run_seconds=args.run_seconds))
    p3 = dataclasses.replace(Page3Params(), n_samples=args.samples, alg4_iters=args.alg4_iters)
    if args.seed is not None:
        p3 = dataclasses.replace(p3, rng_seed=args.seed)
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(sp)

    t = time.time()
    res = prove_task(cfg, arm, surf, scene, signs, p3, max_checks=args.max_checks,
                     verbose=not args.quiet, also_prove=args.also_prove)
    dt = time.time() - t
    print(f"\n[recursive] {args.case}: {res.verdict} -- {res.reason}  ({dt:.1f} s)")
    print(f"[recursive] today's pipeline: {today}")
    if res.tree is not None:
        print("[recursive] proof tree:")
        for line in res.tree.lines("    "):
            print(line)
    if args.save:
        write_certificate(res, args.save)
    if args.save_geometry:
        write_geometry(res, args.save_geometry)
    print(json.dumps(dict(case=args.case, verdict=res.verdict, reason=res.reason,
                          seconds=round(dt, 1), stats={k: v for k, v in res.stats.items()
                                                       if k != "connect"}),
                     default=str))


if __name__ == "__main__":
    main()
