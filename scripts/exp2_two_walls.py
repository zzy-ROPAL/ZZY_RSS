#!/usr/bin/env python3
"""EXP 2 as a ONE-PAGE, TWO-WALL infeasibility proof (findings 47).

Same task, obstacle, robot, page and wall settings as EXP 2 of run_three.py
(`make(0.0, OBS_CLOSED, 61)`, page (++-), single start posture at y = 0), with
ONE change: the doors are NOT discharged by fiber certification. Every Sigma
disc on the page stays LIVE, i.e. the proof assumes the robot might change
page there (invariant 2). EXP 2's published single wall closes only because
its two q2 doors are certified blocked; here that certificate is replaced by
walls:

    wall 1  seed = the start region, rest = the goal components + ring.
            A door near the start region goes into the SEED class (the
            wall encloses it, it must not cut through a door); a door far
            from it into the REST class (the wall excludes it: exit 1).
    wall 2  one per door that wall 1 encloses: seed = the start region,
            rest = that door's ring + outside ring, nothing else. A verified
            wall 2 with the start inside and the whole disc outside proves
            the start cannot reach the door (IJRR Definition 1, X = door).

    INFEASIBLE (on this page, with every door live)  iff  wall 1 verifies
    and every live door is outside wall 1 or outside a verified wall 2.

The verdict is cross-checked by flood.run_proof on the same page, which
reaches the same doors through its exits 1 and 2.

    PYTHONPATH=src python3 scripts/exp2_two_walls.py --out out/exp2_two_walls
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
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

import run_three as m  # noqa: E402
from zzy_rss.doors import build_sigma  # noqa: E402
from zzy_rss.flood import (find_inf_proof_region, invalid_points, label_regions,  # noqa: E402
                           other_valid_points, point_in_wall, region_points,
                           run_proof, second_wall, thin)
from zzy_rss.pagemap import build_all_pages_shared  # noqa: E402
from zzy_rss.wall import (PageLabeller, sigma_disc_samples,  # noqa: E402
                          virtual_boundary_samples)


def config(res: int, ring: str = "ellipse", wall1_doors: str = "enclose"):
    cfg, arm, weld, scene = m.make(0.0, m.OBS_CLOSED, res)
    cfg = dataclasses.replace(
        cfg,
        wall=dataclasses.replace(cfg.wall, sigma_ring=ring, sigma_side=wall1_doors),
        # Every door live: no fiber discharge, exact or sampled.
        sigma_policy=dataclasses.replace(cfg.sigma_policy,
                                         enable_fiber_certification=False),
        crossing=dataclasses.replace(cfg.crossing, exact=False),
        # EXP 2's prove() runs without a budget (gamma up to gamma_max 400).
        budget=dataclasses.replace(cfg.budget, enabled=False))
    return cfg, arm, weld, scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--iters", type=int, default=14, help="Alg 4 rounds per wall")
    ap.add_argument("--out", default="out/exp2_two_walls")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--ring", choices=("circle", "ellipse"), default="ellipse",
                    help="shape of the door training ring (circle = legacy)")
    ap.add_argument("--wall1-doors", choices=("proximity", "enclose"), default="enclose",
                    help="enclose: wall 1 = start vs goal only, every door left to a wall 2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cfg, arm, weld, scene = config(args.res, args.ring, args.wall1_doors)
    built = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False, verbose=False)
    pms = {n: built[n] for n in m.PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    pm = pms[m.START_PAGE]
    signs = m.START_SIGNS
    lab, regs = label_regions(cfg, pm)
    k = sorted([r for r in regs if r.reaches_start and not r.reaches_goal],
               key=lambda r: -r.n_cells)[0].index
    pts = region_points(pm, lab, k)
    discs = [d for d in sig[m.START_PAGE] if d.vanishing]
    print(f"[exp2-2w] page {m.START_PAGE}: start region {k}, {len(pts)} cells, "
          f"y in [{pts[:, 0].min():.3f}, {pts[:, 0].max():.3f}]; "
          f"{len(discs)} live door(s) (fiber discharge OFF)")

    # ---- wall 1: start vs goal (run_three.prove, same recipe as EXP 2) ----
    t = time.perf_counter()
    poly1, rep1, live, disch = m.prove(cfg, arm, weld, scene, pm, lab, k,
                                       sig[m.START_PAGE], iters=args.iters)
    assert not disch, "fiber discharge must be off"
    ok1 = bool(rep1.ok and poly1.closed)
    t_w1 = time.perf_counter() - t
    print(f"[wall1] start vs goal: {rep1.n_passed}/{rep1.n_facets} facets, "
          f"closed={poly1.closed} -> {'VERIFIED' if ok1 else 'FAILED'} ({t_w1:.1f} s)")

    # ---- each door: outside wall 1, or a wall 2 of its own ---------------
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    rng = np.random.default_rng(0)
    seed_core = thin(pts, 450, rng)
    inv = thin(invalid_points(pm), 150, rng)
    doors, walls2 = [], []
    for i, d in enumerate(discs):
        rec = dict(door=i, kind=d.kind, y=round(float(d.y), 4),
                   theta=round(float(d.theta), 4),
                   dist_to_region=round(float(np.min(np.linalg.norm(
                       (pts - [d.y, d.theta]) * [cfg.wall.scale_y, cfg.wall.scale_theta],
                       axis=1))), 4))
        side = point_in_wall(cfg, poly1, (d.y, d.theta)) if ok1 else None
        rec["inside_wall1"] = side
        if side is False:
            rec.update(excluded_by="wall 1 (exit 1)")
            print(f"[door {i}] y={d.y:.4f}: OUTSIDE wall 1 -> excluded by wall 1")
            doors.append(rec)
            continue
        others = [o for o in live if o is not d]
        # Wall 2 judges the OTHER doors by proximity even when wall 1
        # encloses them all: enclosing a far door would only make it harder.
        s_oth, _ = sigma_disc_samples(others, pts, tol, ring=cfg.wall.sigma_ring,
                                      side="proximity")
        seed2 = np.vstack([seed_core, s_oth]) if len(s_oth) else seed_core
        t = time.perf_counter()
        p2, r2, w2, ok2 = second_wall(cfg, arm, weld, scene, signs, seed2, inv, d,
                                      others, alg4_iters=args.iters, verbose=False)
        t_w2 = time.perf_counter() - t
        rec.update(wall2=dict(facets=f"{r2.n_passed}/{r2.n_facets}", closed=bool(p2.closed),
                              gamma=round(float(w2.gamma), 2), separates=bool(ok2),
                              seconds=round(t_w2, 1),
                              note=(r2.failures[0][1] if r2.failures else p2.note)))
        rec["excluded_by"] = "wall 2 (exit 2)" if ok2 else None
        walls2.append((i, p2, ok2))
        print(f"[door {i}] y={d.y:.4f}: INSIDE wall 1 -> wall 2: "
              f"{r2.n_passed}/{r2.n_facets} facets, closed={p2.closed}, "
              f"gamma {w2.gamma:.2f} -> "
              f"{'door EXCLUDED' if ok2 else 'no separating wall'} ({t_w2:.1f} s)")
        doors.append(rec)

    proven = ok1 and all(r["excluded_by"] for r in doors)
    verdict = "INFEASIBLE" if proven else "INCONCLUSIVE"
    print(f"[exp2-2w] {verdict}: wall 1 {'verified' if ok1 else 'FAILED'}; "
          + ", ".join(f"door {r['door']} -> {r['excluded_by'] or 'NOT excluded'}"
                      for r in doors))

    # ---- cross-check with the flood's own exits ---------------------------
    cross = None
    if not args.no_crosscheck:
        t = time.perf_counter()
        cfg_g = dataclasses.replace(cfg, planner=dataclasses.replace(
            cfg.planner, page_source="grid"))
        res = run_proof(cfg_g, arm, weld, scene, pms, sig, n_train_max=450,
                        n_invalid_max=150, alg4_iters=args.iters,
                        seeds=[(m.START_PAGE, k)], verbose=False)
        cross = dict(verdict=res.verdict, reason=res.reason,
                     regions=len(res.regions_visited),
                     walls=[f"{c['page']}/{c['region']} {c['n_passed']}/{c['n_facets']}"
                            for c in res.certificate],
                     doors=[f"y={dd['y']:.4f} exit {dd['exit']} "
                            f"{'followed' if dd['followed'] else 'skipped'}"
                            for dd in res.door_decisions],
                     seconds=round(time.perf_counter() - t, 1))
        print(f"[run_proof] {res.verdict}: {res.reason}")
        for line in cross["doors"]:
            print(f"[run_proof]   door {line}")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6.5))
    m.draw_page(ax, pm, title=f"EXP 2, page ({m.START_PAGE}), every door live: {verdict}")
    for loop in poly1.loops:
        ax.plot(loop[:, 0], loop[:, 1], color="#9d0208", lw=3,
                label="wall 1: start | goal" if loop is poly1.loops[0] else None)
    styles = [("#1d3557", "--"), ("#7b2cbf", "-."), ("#0077b6", ":")]
    for j, (i, p2, ok2) in enumerate(walls2):
        col, ls = styles[j % len(styles)]
        for n, loop in enumerate(p2.loops):
            ax.plot(loop[:, 0], loop[:, 1], color=col, lw=2.4, ls=ls,
                    label=f"wall 2: start | door {i}" if n == 0 else None)
    for i, d in enumerate(discs):
        ax.add_patch(Ellipse((d.y, d.theta), 2 * d.radius, 2 * d.r_theta, fill=False,
                             ec="#ffb703", lw=2.2))
        ax.annotate(f"door {i}", (d.y, d.theta + d.r_theta), xytext=(0, 6),
                    textcoords="offset points", ha="center", color="#ffb703",
                    fontweight="bold")
    ax.axvline(1.0 - cfg.flood.goal_y_tol, color="#2a9d8f", lw=1.5, ls=":")
    ax.plot(0.0, float(np.median(pts[pts[:, 0] == pts[:, 0].min(), 1])), "o", ms=11,
            mfc="#ffffff", mec="#000000", mew=2, zorder=9, label="start region (y = 0)")
    ax.text(0.99, -2.9, "goal y = 1", ha="right", color="#2a9d8f", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "exp2_two_walls.png", dpi=160)
    plt.close(fig)

    report = dict(experiment="EXP 2, one page, two walls (fiber discharge off)",
                  ring=args.ring, wall1_doors=args.wall1_doors,
                  page=m.START_PAGE, start_region=int(k), verdict=verdict,
                  wall1=dict(facets=f"{rep1.n_passed}/{rep1.n_facets}",
                             closed=bool(poly1.closed), verified=ok1, seconds=round(t_w1, 1)),
                  doors=doors, run_proof=cross,
                  seconds=round(time.perf_counter() - t0, 1))
    (out / "exp2_two_walls.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[exp2-2w] wrote {out / 'exp2_two_walls.png'} and exp2_two_walls.json "
          f"({report['seconds']} s)")


if __name__ == "__main__":
    main()
