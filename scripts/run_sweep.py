"""Sweep obstacle width and locate the feasibility transition.

Experiment 3/4 of the plan: the method must NOT claim infeasibility when a
plan exists. Sweeping the obstacle's half-width along y moves the problem from
feasible to infeasible, and the transition is where the claim must flip.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import discs_in_region, label_regions, region_points  # noqa: E402
from zzy_rss.pagemap import build_all_pages  # noqa: E402
from zzy_rss.params import BoxObstacle, Config, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def analyse(res: int, half_y: float, fiber: bool) -> dict:
    base = Config()
    # half_y == 0 must mean NO obstacle. A zero-thickness plate still collides
    # through the capsule radii, which silently made the "obstacle-free"
    # baseline infeasible and hid the real cause.
    if half_y <= 0.0:
        sc = SceneParams(gap_open=True, obstacles=())
    else:
        sc = SceneParams(gap_open=True, obstacles=(
            BoxObstacle(center=(0.34, 0.0, 0.80),
                        half_extent=(0.13, half_y, 0.17), label="mid_block"),))
    cfg = dataclasses.replace(
        base,
        arm=dataclasses.replace(base.arm, use_off_axis_upper_arm=True),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=sc,
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=fiber),
    )
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, sc)
    pages = build_all_pages(cfg, arm, weld, scene, with_kin=False, verbose=False)
    sigma = build_sigma(cfg, arm, weld, pages, verbose=False)

    labels, regions = {}, {}
    n_regions = 0
    direct = []           # regions touching BOTH start and goal: feasible in-page
    for name, pm in pages.items():
        lab, regs = label_regions(cfg, pm)
        labels[name], regions[name] = lab, regs
        n_regions += len(regs)
        direct += [f"{name}/{r.index}" for r in regs
                   if r.reaches_start and r.reaches_goal]

    edges, n_blocked, n_open = [], 0, 0
    for name, pm in pages.items():
        lab = labels[name]
        tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                             pm.theta_grid[1] - pm.theta_grid[0]))
        for r in regions[name]:
            pts = region_points(pm, lab, r.index)
            for d in discs_in_region(sigma.get(name, []), pts, tol):
                if not d.vanishing:
                    continue
                if fiber:
                    fib = door_fiber_validity(cfg, arm, weld, scene, d, pm.signs)
                    if not fib["ok"]:
                        n_blocked += 1
                        continue
                    n_open += 1
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
                        edges.append(((name, r.index), (other, kk)))

    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    start_nodes = [(n, r.index) for n in pages for r in regions[n] if r.reaches_start]
    goal_nodes = {(n, r.index) for n in pages for r in regions[n] if r.reaches_goal}
    seen, q, hit = set(start_nodes), deque(start_nodes), None
    while q:
        node = q.popleft()
        if node in goal_nodes:
            hit = node
            break
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

    mean_valid = float(np.mean([p.stats["frac_valid"] for p in pages.values()]))
    return {
        "half_y": half_y, "mean_valid_frac": mean_valid, "n_regions": n_regions,
        "n_door_edges": len(edges), "n_doors_open": n_open,
        "n_doors_blocked": n_blocked,
        "direct_regions": direct, "n_reachable": len(seen),
        "goal_reachable": bool(hit),
        "goal_at": f"{hit[0]}/{hit[1]}" if hit else None,
        "via_door": bool(hit) and not direct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--fiber", action="store_true")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    widths = [0.0, 0.02, 0.04, 0.06, 0.085, 0.11, 0.15]
    rows = []
    print(f"{'half_y':>7}{'valid%':>8}{'regions':>9}{'edges':>7}{'open':>6}"
          f"{'blockd':>8}{'reach':>7}  verdict")
    print("-" * 76)
    for w in widths:
        r = analyse(args.res, w, args.fiber)
        rows.append(r)
        verdict = ("FEASIBLE in-page" if r["direct_regions"]
                   else "FEASIBLE via door" if r["goal_reachable"]
                   else "goal unreachable -> prove it")
        print(f"{w:>7.3f}{r['mean_valid_frac'] * 100:>8.2f}{r['n_regions']:>9}"
              f"{r['n_door_edges']:>7}{r['n_doors_open']:>6}"
              f"{r['n_doors_blocked']:>8}{r['n_reachable']:>7}  {verdict}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[out] wrote {out / 'sweep.json'}")


if __name__ == "__main__":
    main()
