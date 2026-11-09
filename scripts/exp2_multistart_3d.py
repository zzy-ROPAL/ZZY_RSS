#!/usr/bin/env python3
"""3-D EXP 2 with the start given as a TORCH POSE (findings 48).

The task fixes where the torch is at the start (u = start_u, v = start_v),
not the arm. Every sign branch and arm angle that puts the torch there
without collision or a limit violation is a start the robot could be in, so
an infeasibility proof must cover ALL of them -- one posture, as in every
experiment so far, proves a weaker claim.

Per page (page3.prove3_starts): the valid start postures are theta arcs on
the start line; each start region (roadmap component holding some of them)
gets wall 1 (start region | goal) and, with the doors live, one wall 2 per
live door inside it; and every densely sampled start posture must lie inside
a verified wall 1 (coverage).

    INFEASIBLE   every page INFEASIBLE (or no start posture on it)
    FEASIBLE     some start region has a verified path to the goal
    INCONCLUSIVE otherwise

The doors on each page are either discharged (--discharge, the published
3-D EXP 2 setting) or all kept live and excluded by second walls (default,
as in exp2_two_walls_3d.py). Either way no proof object relies on a page
change, so the eight page proofs are independent and run in parallel.

    PYTHONPATH=src python3 scripts/exp2_multistart_3d.py --jobs 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import exp2_two_walls_3d as E  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES  # noqa: E402
from zzy_rss.page3 import FEASIBLE, INCONCLUSIVE, INFEASIBLE, prove3_starts  # noqa: E402


def _job(a):
    signs, samples, seed, live, n_inject, n_check, alg4_iters = a
    import dataclasses
    cfg, arm, scene, surf, p3 = E.setup(samples, seed, "enclose", True, live)
    p3 = dataclasses.replace(p3, second_wall_iters=8, alg4_iters=alg4_iters)
    t = time.time()
    try:
        r = prove3_starts(cfg, arm, surf, scene, signs, p3, n_inject=n_inject,
                          n_check=n_check, verbose=True)
    except Exception as e:                          # report, never hide
        return dict(page="".join("+" if s > 0 else "-" for s in signs),
                    verdict=INCONCLUSIVE, reason=f"error: {e!r}", regions=[],
                    seconds=round(time.time() - t, 1))
    for reg in r["regions"]:
        reg["polys"] = [(lab, None if p is None or p.vertices is None else
                         dict(vertices=p.vertices, simplices=np.asarray(p.simplices)))
                        for lab, p in reg.get("polys", [])]
    return r


def figure(results, setup_args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp2_two_walls_3d import slice_segments
    from zzy_rss.batch import classify_surface_batch
    from zzy_rss.doors3 import door_curves
    cfg, arm, scene, surf, p3 = E.setup(*setup_args)
    v = p3.start_v
    U, T = np.meshgrid(np.linspace(0, 1, 121), np.linspace(-np.pi, np.pi, 145), indexing="ij")
    P = np.column_stack([U.ravel(), np.full(U.size, v), T.ravel()])
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    by = {r["page"]: r for r in results}
    for ax, sg in zip(axes.ravel(), SIGN_BRANCHES):
        name = "".join("+" if s > 0 else "-" for s in sg)
        ok = classify_surface_batch(cfg, arm, surf, scene, P, sg)[0].reshape(U.shape)
        ax.imshow(ok.T, origin="lower", aspect="auto", extent=[0, 1, -np.pi, np.pi],
                  cmap="Greys", vmin=-0.3, vmax=1.6)
        r = by.get(name, {})
        w2_styles = [("#1d3557", "--"), ("#7b2cbf", "-.")]
        notes = []
        for reg in r.get("regions", []):
            k2 = 0
            sw = reg.get("second_walls", [])
            for lab, pl in reg.get("polys", []):
                if lab == "wall 1":
                    col, ls, lw = "#9d0208", "-", 2.2
                else:
                    ok2 = sw[k2]["excluded"] if k2 < len(sw) else False
                    col, ls = w2_styles[k2 % 2]
                    lw = 1.8
                    if not ok2:
                        notes.append(f"wall 2 #{k2}: FAILED")
                    k2 += 1
                if pl is None:
                    continue
                for seg in slice_segments(pl["vertices"], pl["simplices"], v):
                    ax.plot(seg[:, 0], seg[:, 1], color=col, ls=ls, lw=lw)
        for c in door_curves(cfg, arm, surf, p3):
            near = np.abs(c.pts[:, 1] - v) < 0.02
            if near.any():
                ax.plot(c.pts[near, 0], c.pts[near, 2], "o", ms=7, mfc="none", mec="#ffb703",
                        mew=2.2)
        if notes:
            ax.text(0.5, -2.85, "; ".join(notes), ha="center", color="#d62828", fontsize=9,
                    fontweight="bold")
        for a, b in r.get("arcs", []):
            ax.plot([0, 0], np.radians([a, b]), color="#1d6fd8", lw=6, solid_capstyle="butt")
        ax.axvline(1 - p3.goal_tol, color="#2a9d8f", ls=":", lw=1.4)
        ax.set_title(f"page {name}: {r.get('verdict', '?')}  "
                     f"({len(r.get('regions', []))} start region(s), coverage {r.get('coverage')})",
                     fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("u (along the weld)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta$ (arm angle)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#1d6fd8", lw=6, label="valid start postures (u = 0)"),
                        Line2D([], [], color="#9d0208", lw=2, label="wall 1: start region | goal"),
                        Line2D([], [], color="#1d3557", lw=1.8, ls="--",
                               label="wall 2 #0: start region | first door"),
                        Line2D([], [], color="#7b2cbf", lw=1.8, ls="-.",
                               label="wall 2 #1: start region | second door"),
                        Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec="#ffb703",
                               mew=2.2, label="q2 door curves (live)"),
                        Line2D([], [], color="#2a9d8f", ls=":", label="goal u = 1")],
               loc="lower center", ncol=6, fontsize=10)
    verdict = combine(results)[0]
    mode = "every door LIVE" if setup_args[4] else "doors discharged"
    fig.suptitle(f"3-D EXP 2, start = torch pose (all postures on all 8 pages), {mode}: {verdict} -- "
                 f"slices at v = {v}, dark grey = valid", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def combine(results) -> tuple:
    if any(r["verdict"] == FEASIBLE for r in results):
        return FEASIBLE, "a start posture has a verified path to the goal"
    if all(r["verdict"] == INFEASIBLE for r in results):
        n = sum(len(r["regions"]) for r in results)
        return INFEASIBLE, (f"all 8 pages: every start posture is enclosed ({n} start "
                            f"region(s), every wall verified)")
    bad = [r["page"] for r in results if r["verdict"] != INFEASIBLE]
    return INCONCLUSIVE, f"page(s) {', '.join(bad)} not proved"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--discharge", action="store_true",
                    help="discharge certified-blocked doors (published 3-D EXP 2 setting)")
    ap.add_argument("--n-inject", type=int, default=181)
    ap.add_argument("--n-check", type=int, default=3601)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--alg4-iters", type=int, default=8,
                    help="Algorithm 4 rounds per wall 1 (as the 45/46 runs)")
    ap.add_argument("--out", default="out/exp2_multistart_3d")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    live = not args.discharge
    t0 = time.time()
    jobs = [(sg, args.samples, args.seed, live, args.n_inject, args.n_check, args.alg4_iters)
            for sg in SIGN_BRANCHES]
    with Pool(args.jobs) as pool:
        results = pool.map(_job, jobs)
    verdict, reason = combine(results)
    wall_s = round(time.time() - t0, 1)
    for r in results:
        print(f"[multistart] page {r['page']}: {r['verdict']} ({r['seconds']} s) -- {r['reason']}")
        for reg in r["regions"]:
            w = reg.get("wall", {})
            print(f"[multistart]    start region {reg['comp']}: {reg['n_starts']} starts, theta "
                  f"{reg['theta_deg']} deg; wall 1 ({reg.get('wall1_side')}) {w.get('facets')} "
                  f"gamma {w.get('gamma')}; "
                  + ", ".join(f"wall 2 {s['curve']} {s['wall']['facets']}"
                              f"{'' if s['excluded'] else ' NOT excluded'}"
                              for s in reg.get("second_walls", [])))
    print(f"[multistart] {verdict}: {reason}  (wall clock {wall_s} s, "
          f"sum of page times {round(sum(r['seconds'] for r in results), 1)} s)")
    slim = [{k: v for k, v in r.items() if k != "regions"} |
            dict(regions=[{k: v for k, v in g.items() if k != "polys"} for g in r["regions"]])
            for r in results]
    (out / "exp2_multistart_3d.json").write_text(json.dumps(
        dict(verdict=verdict, reason=reason, live_doors=live, samples=args.samples,
             seed=args.seed, wall_clock_s=wall_s, pages=slim), indent=2, default=str))
    with open(out / "exp2_multistart_3d.pkl", "wb") as fh:
        pickle.dump(dict(verdict=verdict, results=results), fh)
    figure(results, (args.samples, args.seed, "enclose", True, live), out / "exp2_multistart_3d.png")
    print(f"[multistart] wrote {out}")


if __name__ == "__main__":
    main()
