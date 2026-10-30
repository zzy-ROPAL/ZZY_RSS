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


def one_box():
    """One box: the family the current EXP 2 / EXP 3 obstacles come from."""
    xs = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
    zs = (0.50, 0.54, 0.58, 0.62, 0.66)
    hs = ((0.030, 0.030, 0.030), (0.040, 0.040, 0.040),
          (0.050, 0.050, 0.050), (0.060, 0.060, 0.060),
          (0.035, 0.100, 0.035), (0.030, 0.180, 0.030),
          (0.025, 0.025, 0.080), (0.080, 0.030, 0.025))
    for x, z, h in itertools.product(xs, zs, hs):
        yield (f"1box x{x:.2f} z{z:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, z), half_extent=h, label="block"),))


def two_box_zgap():
    """Two boxes stacked in z, leaving a GAP the elbow must thread.

    theta swings the elbow around the shoulder-wrist axis, so blocking the
    high and low elbow arcs while leaving a middle channel is the natural way
    to split the valid set in theta WITHOUT making any weld point
    unreachable -- condition (A) kept, condition (B) induced.
    """
    xs = (0.14, 0.20, 0.26)
    z1s = (0.38, 0.44, 0.50)          # lower box
    z2s = (0.60, 0.66, 0.72)          # upper box
    hs = ((0.050, 0.120, 0.050), (0.040, 0.200, 0.040),
          (0.060, 0.060, 0.060), (0.035, 0.260, 0.035))
    for x, z1, z2, h in itertools.product(xs, z1s, z2s, hs):
        yield (f"2box-zgap x{x:.2f} z{z1:.2f}/{z2:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, z1), half_extent=h, label="low"),
                BoxObstacle(center=(x, 0.0, z2), half_extent=h, label="high")))


def two_box_ygap():
    """Two boxes separated along y, pinching the middle of the weld."""
    xs = (0.14, 0.20, 0.26)
    zs = (0.52, 0.58, 0.64)
    yoffs = (0.10, 0.16, 0.22)
    hs = ((0.050, 0.060, 0.070), (0.040, 0.040, 0.120),
          (0.060, 0.080, 0.050))
    for x, z, yo, h in itertools.product(xs, zs, yoffs, hs):
        yield (f"2box-ygap x{x:.2f} z{z:.2f} y+-{yo:.2f} h{h}",
               (BoxObstacle(center=(x, -yo, z), half_extent=h, label="left"),
                BoxObstacle(center=(x, +yo, z), half_extent=h, label="right")))


def three_box_slot():
    """Two in z plus one behind: a slot the arm must enter and cannot leave."""
    xs = (0.16, 0.22)
    gaps = (0.14, 0.20)
    hs = ((0.045, 0.160, 0.045), (0.035, 0.240, 0.035))
    for x, g, h in itertools.product(xs, gaps, hs):
        zc = 0.55
        yield (f"3box-slot x{x:.2f} gap{g:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, zc - g), half_extent=h, label="low"),
                BoxObstacle(center=(x, 0.0, zc + g), half_extent=h, label="high"),
                BoxObstacle(center=(x + 0.12, 0.0, zc),
                            half_extent=(0.035, h[1], g), label="back")))


FAMILIES = {"1box": one_box, "2box-zgap": two_box_zgap,
            "2box-ygap": two_box_ygap, "3box-slot": three_box_slot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=31)
    ap.add_argument("--out", default="out/search.json")
    ap.add_argument("--tilts", default="0,30")
    ap.add_argument("--families", default="2box-zgap,2box-ygap,3box-slot",
                    help="comma-separated: " + ",".join(FAMILIES))
    args = ap.parse_args()

    tilts = tuple(float(t) for t in args.tilts.split(","))
    combos = []
    for fam in args.families.split(","):
        for tilt in tilts:
            for tag, obs in FAMILIES[fam.strip()]():
                combos.append((tilt, tag, obs))

    rows = []
    t0 = time.time()
    print(f"{len(combos)} candidates at res {args.res} "
          f"(families: {args.families})", flush=True)
    for i, (tilt, tag, obs) in enumerate(combos):
        x = z = 0.0
        h = []
        try:
            r = analyse(tilt, obs, args.res)
        except Exception as exc:                      # keep the sweep going
            print(f"   [{i}] FAILED {exc}", flush=True)
            continue
        r.update(tilt=tilt, tag=tag,
                 obstacles=[dict(center=list(b.center),
                                 half_extent=list(b.half_extent),
                                 label=b.label) for b in obs])
        rows.append(r)
        good = (r["curtain_cols"] == 0 and not r["start_goal_same"])
        if good:
            print(f"   *** tilt {tilt:4.0f}  {tag}  "
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
    for r in keep[:20]:
        print(f"   tilt {r['tilt']:4.0f}  {r['tag']:<46}  "
              f"pinch {r['pinch']}/{args.res}  "
              f"start-pages {r['start_pages']}/8  "
              f"mean valid {100*r['mean_valid']:.1f}%")
    if not keep:
        near = [r for r in rows if r["curtain_cols"] <= 2
                and not r["start_goal_same"]][:10]
        print("   nearest misses (small curtain, still disconnected):")
        for r in near:
            print(f"   tilt {r['tilt']:4.0f}  {r['tag']:<46}  "
                  f"curtain {r['curtain_cols']}  pinch {r['pinch']}")
    print(f"\n[json] {args.out}   total {time.time()-t0:.0f}s")


main()
