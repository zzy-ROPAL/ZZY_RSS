"""Stage 1-2 + door-graph reachability pre-scan on both scenes.

This is the cheap half of the certificate: build the page maps, find Sigma,
then BFS over (region, door) connectivity from the start set. It mirrors IJRR
running the planner in parallel with proof construction -- if the goal is
reachable there is nothing to prove, and no wall need be built.

Doors are followed conservatively (assumed open) unless fiber certification is
switched on, in which case a door whose whole self-motion fiber is invalid is
certified blocked and its edge removed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import label_regions, region_points, discs_in_region  # noqa: E402
from zzy_rss.pagemap import build_all_pages, page_census  # noqa: E402
from zzy_rss.params import Config, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def run(res: int, gap_open: bool, fiber: bool) -> dict:
    tag = "gap_open" if gap_open else "gap_closed"
    base = Config()
    cfg = dataclasses.replace(
        base,
        arm=dataclasses.replace(base.arm, use_off_axis_upper_arm=True),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams.worked_illustration(gap_open),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=fiber),
    )
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)
    box = cfg.scene.obstacles[0]

    print("\n" + "=" * 76)
    print(f"SCENE {tag}   grid {res}x{res}   box half={box.half_extent}")
    print("=" * 76)

    t0 = time.time()
    pages = build_all_pages(cfg, arm, weld, scene, with_kin=False, verbose=False)
    census = page_census(pages)
    labels, regions = {}, {}
    total_regions = 0
    for name, pm in pages.items():
        lab, regs = label_regions(cfg, pm)
        labels[name], regions[name] = lab, regs
        total_regions += len(regs)
        print(f"[stage1] page {name}  valid {pm.stats['frac_valid'] * 100:5.2f}%  "
              f"regions {len(regs):2d}  start {[r.index for r in regs if r.reaches_start]}"
              f"  goal {[r.index for r in regs if r.reaches_goal]}")
    sigma = build_sigma(cfg, arm, weld, pages, verbose=False)
    n_doors = sum(1 for ds in sigma.values() for d in ds if d.vanishing)
    print(f"\n[stage2] {n_doors} door discs; {total_regions} regions total "
          f"({census['n_pages_non_empty']}/8 pages non-empty)")

    # Door edges, with optional fiber certification.
    edges, blocked, open_doors = [], [], []
    for name, pm in pages.items():
        lab = labels[name]
        for r in regions[name]:
            pts = region_points(pm, lab, r.index)
            tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                                 pm.theta_grid[1] - pm.theta_grid[0]))
            for d in discs_in_region(sigma.get(name, []), pts, tol):
                if not d.vanishing:
                    continue
                if fiber:
                    fib = door_fiber_validity(cfg, arm, weld, scene, d, pm.signs)
                    if not fib["ok"]:
                        blocked.append({"page": name, "region": r.index, "y": d.y,
                                        "theta": d.theta,
                                        "blockers": fib.get("blockers", {})})
                        continue
                    open_doors.append({"page": name, "region": r.index, "y": d.y,
                                       "theta": d.theta,
                                       "valid_fraction": fib["valid_fraction"]})
                for other in d.pages:
                    if other == name or other not in pages:
                        continue
                    opm, olab = pages[other], labels[other]
                    i = int(np.argmin(np.abs(opm.y_grid - d.y)))
                    j = int(np.argmin(np.abs(opm.theta_grid - d.theta)))
                    rad = max(1, int(np.ceil(d.radius /
                                             (opm.y_grid[1] - opm.y_grid[0]))))
                    sub = olab[max(0, i - rad): i + rad + 1,
                               max(0, j - rad): j + rad + 1]
                    for kk in sorted({int(v) for v in np.unique(sub) if v > 0}):
                        edges.append(((name, r.index), (other, kk),
                                      round(d.y, 4), round(d.theta, 4)))

    print(f"[stage2] door edges: {len(edges)}   "
          f"certified BLOCKED: {len(blocked)}   open: {len(open_doors)}")
    for o in open_doors[:4]:
        print(f"         OPEN  {o['page']}/{o['region']} at y={o['y']:.4f} "
              f"th={o['theta']:+.4f}  ({o['valid_fraction'] * 100:.1f}% of fiber valid)")
    for b in blocked[:4]:
        bl = ",".join(f"{k}:{v}" for k, v in list(b["blockers"].items())[:2])
        print(f"         BLOCKED {b['page']}/{b['region']} at y={b['y']:.4f} "
              f"th={b['theta']:+.4f}  by {bl}")

    # BFS from every start-touching region.
    adj = {}
    for a, b, _y, _t in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    start_nodes = [(n, r.index) for n in pages for r in regions[n] if r.reaches_start]
    goal_nodes = {(n, r.index) for n in pages for r in regions[n] if r.reaches_goal}

    seen, q = set(start_nodes), deque(start_nodes)
    hit = None
    while q:
        node = q.popleft()
        if node in goal_nodes:
            hit = node
            break
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    dt = time.time() - t0
    print("\n" + "-" * 76)
    if hit:
        print(f"VERDICT [{tag}]: NO PROOF -- goal set reachable at region "
              f"{hit[0]}/{hit[1]}")
        print("          the planner returns a path; nothing to prove")
    else:
        print(f"VERDICT [{tag}]: goal set UNREACHABLE in the door graph")
        print(f"          {len(seen)} region(s) reachable from the start, "
              f"none touches the goal")
        print("          -> infeasibility follows once a wall is verified for "
              "each of those regions")
    print(f"          ({dt:.1f}s)")
    print("-" * 76)

    return {
        "scene": tag, "grid": res, "fiber_cert": fiber,
        "pages_non_empty": census["n_pages_non_empty"],
        "total_regions": total_regions, "n_door_discs": n_doors,
        "n_door_edges": len(edges), "n_blocked": len(blocked),
        "n_open": len(open_doors),
        "regions_reachable_from_start": len(seen),
        "goal_reachable": bool(hit),
        "goal_reached_at": f"{hit[0]}/{hit[1]}" if hit else None,
        "open_doors": open_doors[:8], "blocked_doors": blocked[:8],
        "seconds": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=81)
    ap.add_argument("--fiber", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    rows = [run(args.res, g, args.fiber) for g in (True, False)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reachability.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "=" * 76)
    print(f"{'scene':<12}{'pages':>6}{'regions':>9}{'doors':>7}{'edges':>7}"
          f"{'blocked':>9}{'reach':>7}  verdict")
    for r in rows:
        print(f"{r['scene']:<12}{r['pages_non_empty']:>6}{r['total_regions']:>9}"
              f"{r['n_door_discs']:>7}{r['n_door_edges']:>7}{r['n_blocked']:>9}"
              f"{r['regions_reachable_from_start']:>7}  "
              f"{'NO PROOF (feasible)' if r['goal_reachable'] else 'goal unreachable'}")
    print(f"\n[out] wrote {out / 'reachability.json'}")


if __name__ == "__main__":
    main()
