"""Full experiment: Stage 0-5 on both worked-illustration scenes.

gap_open   -- a theta gap remains and the q2 door is open: the method must NOT
              claim infeasibility.
gap_closed -- the box is widened along y, closing both the gap and the door:
              infeasibility must be proven, with a verified wall per region.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma  # noqa: E402
from zzy_rss.flood import label_regions, run_proof, write_certificate  # noqa: E402
from zzy_rss.pagemap import build_all_pages, page_census  # noqa: E402
from zzy_rss.params import Config, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def make_cfg(res: int, gap_open: bool, fiber: bool, pd_mode: str) -> Config:
    base = Config()
    return dataclasses.replace(
        base,
        arm=dataclasses.replace(base.arm, use_off_axis_upper_arm=True),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams.worked_illustration(gap_open),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=fiber),
        wall=dataclasses.replace(base.wall, pd_mode=pd_mode),
    )


def run_scene(res: int, gap_open: bool, fiber: bool, pd_mode: str,
              out_dir: Path) -> dict:
    tag = "gap_open" if gap_open else "gap_closed"
    cfg = make_cfg(res, gap_open, fiber, pd_mode)
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)
    box = cfg.scene.obstacles[0]

    print("\n" + "=" * 78)
    print(f"SCENE {tag}   grid {res}x{res}   pd_mode=[pd:{pd_mode}]   "
          f"fiber_cert={fiber}")
    print(f"box center={box.center} half={box.half_extent}")
    print("=" * 78)

    t0 = time.time()
    pages = build_all_pages(cfg, arm, weld, scene, with_kin=False)
    t1 = time.time()
    census = page_census(pages)
    print(f"\n[census] {census['n_pages_non_empty']}/8 pages non-empty")

    region_counts = {}
    for name, pm in pages.items():
        _lab, regs = label_regions(cfg, pm)
        region_counts[name] = len(regs)
        start = [r.index for r in regs if r.reaches_start]
        goal = [r.index for r in regs if r.reaches_goal]
        print(f"[regions] page {name}: {len(regs)} region(s), "
              f"start-touching {start}, goal-touching {goal}")

    sigma = build_sigma(cfg, arm, weld, pages, verbose=False)
    n_doors = sum(1 for ds in sigma.values() for d in ds if d.vanishing)
    print(f"[stage2] {n_doors} door discs across all pages")
    t2 = time.time()

    result = run_proof(cfg, arm, weld, scene, pages, sigma)
    t3 = time.time()

    print("\n" + "-" * 78)
    print(f"RESULT [{tag}]: "
          f"{'INFEASIBILITY PROVEN' if result.proven else 'NO PROOF'}")
    print(f"  reason: {result.reason}")
    print(f"  regions visited : {len(result.regions_visited)}")
    print(f"  walls in cert   : {len(result.certificate)} "
          f"({sum(1 for c in result.certificate if c['ok'])} verified)")
    print(f"  doors followed  : {len(result.doors_followed)}")
    if result.goal_reached_via:
        print(f"  goal reached via: region {result.goal_reached_via}")
    for c in result.certificate:
        print(f"   wall {c['page']}/{c['region']}: facets {c['n_passed']}/"
              f"{c['n_facets']}  closed={c['closed']}  gamma={c['gamma']:.2f}  "
              f"alg2_rounds={c['alg2_rounds']}  ok={c['ok']}"
              + (f"  note={c['note']}" if c["note"] else ""))
    print(f"  timing: stage1 {t1 - t0:.1f}s  stage2 {t2 - t1:.1f}s  "
          f"stage3-4 {t3 - t2:.1f}s")
    print("-" * 78)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_certificate(result, sigma, out_dir / f"certificate_{tag}.json")
    return {
        "scene": tag, "grid": res, "pd_mode": pd_mode, "fiber_cert": fiber,
        "census": census, "region_counts": region_counts, "n_door_discs": n_doors,
        "proven": result.proven, "reason": result.reason,
        "n_regions_visited": len(result.regions_visited),
        "walls": result.certificate, "n_doors_followed": len(result.doors_followed),
        "goal_reached_via": result.goal_reached_via,
        "timing": {"stage1": t1 - t0, "stage2": t2 - t1, "stage34": t3 - t2},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=81)
    ap.add_argument("--pd-mode", default="ijrr_local_opt",
                    choices=["ijrr_local_opt", "lipschitz"])
    ap.add_argument("--fiber", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    out = Path(args.out)
    results = [run_scene(args.res, g, args.fiber, args.pd_mode, out)
               for g in (True, False)]
    (out / "experiment_results.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'scene':<12} {'pages':>6} {'doors':>6} {'regions':>8} "
          f"{'walls ok':>9} {'verdict':>22}")
    for r in results:
        print(f"{r['scene']:<12} {r['census']['n_pages_non_empty']:>6} "
              f"{r['n_door_discs']:>6} {r['n_regions_visited']:>8} "
              f"{sum(1 for c in r['walls'] if c['ok'])}/{len(r['walls']):<7} "
              f"{'INFEASIBILITY PROVEN' if r['proven'] else 'NO PROOF (feasible)':>22}")
    print(f"\n[out] wrote {out / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
