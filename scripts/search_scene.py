"""Search for a scene that is infeasible by CONNECTIVITY, not reachability.

findings 22: the current EXP 2 / EXP 3 obstacles make a stretch of the weld
unreachable on every page at every arm angle (a "common curtain"), so
infeasibility follows from the page map alone and the SVM wall is decoration.

A scene that actually exercises the method must satisfy BOTH:

  (A) no common curtain -- for every y, some page has at least one valid
      arm angle, so no single weld point is unreachable; and
  (B) the start region still cannot reach the goal -- the valid set is split
      so that start and goal lie in different components.

(B) is checked here with grid connectivity, which per CLAUDE.md invariant 8
is a SEARCH FILTER and a spot check, never a verdict. The verdict still comes
from the learned wall.

Usage:  python3 scripts/search_scene.py --res 31 --out out/search.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.flood import label_regions  # noqa: E402
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
START_PAGE = "++-"


def weld_for(tilt_deg):
    a = np.radians(tilt_deg)
    return WeldParams(p0=(0.28, -0.26, 0.46), p1=(0.28, 0.26, 0.46),
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def analyse(tilt, obs, res):
    base = Config()
    cfg = dataclasses.replace(
        base, weld=weld_for(tilt),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=tuple(obs)))
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    per_y_best = None          # max valid cells in a y column, over pages
    start_goal_same = False    # some page has one region reaching start AND goal
    start_pages = 0
    pages = {}
    for n in PAGES:
        pm = build_page_map(cfg, arm, weld, scene,
                            tuple(1 if c == "+" else -1 for c in n),
                            with_kin=False)
        col = pm.valid_mask.sum(axis=1)
        per_y_best = col if per_y_best is None else np.maximum(per_y_best, col)
        lab, regs = label_regions(cfg, pm)
        if any(r.reaches_start and r.reaches_goal for r in regs):
            start_goal_same = True
        if any(r.reaches_start for r in regs):
            start_pages += 1
        pages[n] = dict(valid=float(pm.valid_mask.mean()),
                        n_regions=len(regs),
                        spanning=bool(any(r.reaches_start and r.reaches_goal
                                          for r in regs)))
    curtain = int(np.sum(per_y_best == 0))
    return dict(curtain_cols=curtain,
                pinch=int(per_y_best.min()),
                pinch_frac=float(per_y_best.min()) / res,
                start_goal_same=bool(start_goal_same),
                start_pages=int(start_pages),
                mean_valid=float(np.mean([p["valid"] for p in pages.values()])),
                pages=pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=31)
    ap.add_argument("--out", default="out/search.json")
    ap.add_argument("--tilts", default="0,30")
    args = ap.parse_args()

    xs = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
    zs = (0.50, 0.54, 0.58, 0.62, 0.66)
    hs = ((0.030, 0.030, 0.030), (0.040, 0.040, 0.040),
          (0.050, 0.050, 0.050), (0.060, 0.060, 0.060),
          (0.035, 0.100, 0.035), (0.030, 0.180, 0.030),
          (0.025, 0.025, 0.080), (0.080, 0.030, 0.025))
    tilts = tuple(float(t) for t in args.tilts.split(","))

    rows = []
    t0 = time.time()
    combos = list(itertools.product(tilts, xs, zs, hs))
    print(f"{len(combos)} candidates at res {args.res}", flush=True)
    for i, (tilt, x, z, h) in enumerate(combos):
        obs = (BoxObstacle(center=(x, 0.0, z), half_extent=h, label="block"),)
        try:
            r = analyse(tilt, obs, args.res)
        except Exception as exc:                      # keep the sweep going
            print(f"   [{i}] FAILED {exc}", flush=True)
            continue
        r.update(tilt=tilt, x=x, z=z, h=list(h))
        rows.append(r)
        good = (r["curtain_cols"] == 0 and not r["start_goal_same"])
        if good:
            print(f"   *** tilt {tilt:4.0f} x {x:.2f} z {z:.2f} h {h}  "
                  f"curtain {r['curtain_cols']}  pinch {r['pinch']}/{args.res}  "
                  f"mean valid {100*r['mean_valid']:.1f}%", flush=True)
        if i % 40 == 0:
            print(f"   [{i}/{len(combos)}] {time.time()-t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["curtain_cols"] > 0, r["start_goal_same"],
                             r["pinch"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    keep = [r for r in rows if r["curtain_cols"] == 0 and not r["start_goal_same"]]
    print(f"\n{len(keep)} candidate(s) are infeasible by CONNECTIVITY "
          f"(no curtain, no spanning region) out of {len(rows)}")
    for r in keep[:15]:
        print(f"   tilt {r['tilt']:4.0f}  x {r['x']:.2f}  z {r['z']:.2f}  "
              f"h {tuple(r['h'])}  pinch {r['pinch']}/{args.res}  "
              f"start-pages {r['start_pages']}/8  "
              f"mean valid {100*r['mean_valid']:.1f}%")
    print(f"\n[json] {args.out}   total {time.time()-t0:.0f}s")


main()
