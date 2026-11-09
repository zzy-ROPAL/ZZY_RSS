#!/usr/bin/env python3
"""3-D EXP 2 as a ONE-PAGE, TWO-WALL infeasibility proof (findings 47c).

The d = 3 counterpart of scripts/exp2_two_walls.py. Same scene as
`run_3d.py --obstacle zgap` (findings 42): the EXP 2 weld line swept +/-5 cm in
z into a patch, ZGAP, start page (++-), page (u, v, theta). Its published
proof is ONE wall, and it closes because both q2 door curves are DISCHARGED
(Lipschitz-guarded certification). Here every door curve stays LIVE:

    wall 1  seed = the start region + every live door tube (sigma_side
            "enclose"), rest = the goal components + goal face + outside
            ring. It separates start from goal and keeps every door inside.
    wall 2  one per live curve inside wall 1 (page3.second_wall3): seed =
            the start region, rest = that curve's tube + outside ring. It
            excludes the door only if no triangle touches the tube (exact).

    INFEASIBLE (one page, every door live) iff wall 1 verifies and every
    live curve is excluded by a verified wall 2.

    PYTHONPATH=src python3 scripts/exp2_two_walls_3d.py --out out/exp2_two_walls_3d
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

from run_3d import PATCHES, START_SIGNS, ZGAP  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import prove3  # noqa: E402
from zzy_rss.params import Config, Page3Params, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402


def setup(samples: int, seed: int, wall1_doors: str, second_wall: bool, live_doors: bool):
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=ZGAP))
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(
        cfg.wall, gamma_search="gallop", sigma_side=wall1_doors))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(PATCHES["exp2"])
    p3 = dataclasses.replace(Page3Params(), n_samples=samples, rng_seed=seed,
                             second_wall=second_wall,
                             discharge_doors=not live_doors)
    return cfg, arm, scene, surf, p3


def slice_segments(verts: np.ndarray, simp: np.ndarray, v0: float) -> list:
    """Intersection of a triangle mesh with the plane v = v0, as (u, theta)
    segments."""
    out = []
    for t in simp:
        P = verts[t]
        d = P[:, 1] - v0
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (d[i] > 0) != (d[j] > 0):
                s = d[i] / (d[i] - d[j])
                pts.append(P[i] + s * (P[j] - P[i]))
        if len(pts) == 2:
            out.append(np.array([[pts[0][0], pts[0][2]], [pts[1][0], pts[1][2]]]))
    return out


def figure(cfg, arm, scene, surf, p3, res, path, title):
    signs = START_SIGNS["exp2"]
    curves = door_curves(cfg, arm, surf, p3)
    vs = (0.15, 0.5, 0.85)
    fig, axes = plt.subplots(1, len(vs), figsize=(5.2 * len(vs), 4.6), sharey=True)
    U, T = np.meshgrid(np.linspace(0, 1, 151), np.linspace(-np.pi, np.pi, 181), indexing="ij")
    colours = ["#1d3557", "#7b2cbf", "#0077b6", "#2a9d8f"]
    for ax, v in zip(axes, vs):
        P = np.column_stack([U.ravel(), np.full(U.size, v), T.ravel()])
        ok = classify_surface_batch(cfg, arm, surf, scene, P, signs)[0].reshape(U.shape)
        ax.imshow(ok.T, origin="lower", aspect="auto", extent=[0, 1, -np.pi, np.pi],
                  cmap="Greys", vmin=-0.3, vmax=1.6, alpha=0.9)
        walls = [("wall 1: start | goal", res.poly, "#9d0208", "-", 2.6)]
        walls += [(f"wall 2: start | {lab}", pl, colours[i % 4], "--", 2.0)
                  for i, (lab, pl) in enumerate(res.polys2)]
        for lab, pl, col, ls, lw in walls:
            if pl is None or pl.vertices is None:
                continue
            for n, seg in enumerate(slice_segments(pl.vertices, np.asarray(pl.simplices), v)):
                ax.plot(seg[:, 0], seg[:, 1], color=col, ls=ls, lw=lw,
                        label=lab if (n == 0 and v == vs[0]) else None)
        for ci, c in enumerate(curves):
            near = np.abs(c.pts[:, 1] - v) < 0.02
            if near.any():
                ax.plot(c.pts[near, 0], c.pts[near, 2], "o", ms=7, mfc="none",
                        mec="#ffb703", mew=2.2,
                        label="q2 door curves (live)" if (v == vs[0] and ci == 0) else None)
        ax.axvline(1 - p3.goal_tol, color="#2a9d8f", ls=":", lw=1.4)
        ax.set_title(f"slice v = {v}")
        ax.set_xlabel("u (along the weld)")
    axes[0].set_ylabel(r"$\theta$ (arm angle)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=Page3Params().n_samples)
    ap.add_argument("--seed", type=int, default=Page3Params().rng_seed)
    ap.add_argument("--wall1-doors", choices=("proximity", "enclose"), default="enclose")
    ap.add_argument("--discharge", action="store_true",
                    help="allow door discharge (the published single-wall proof)")
    ap.add_argument("--no-second-wall", action="store_true")
    ap.add_argument("--out", default="out/exp2_two_walls_3d")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, arm, scene, surf, p3 = setup(args.samples, args.seed, args.wall1_doors,
                                      not args.no_second_wall, not args.discharge)
    t0 = time.time()
    res = prove3(cfg, arm, surf, scene, START_SIGNS["exp2"], p3, verbose=True)
    secs = round(time.time() - t0, 1)
    print(f"[exp2-3d-2w] {res.verdict}: {res.reason} ({secs} s)")
    report = dict(experiment="3-D EXP 2 (ZGAP patch), one page, every door live",
                  wall1_doors=args.wall1_doors, live_doors=not args.discharge,
                  second_wall=not args.no_second_wall, samples=args.samples, seed=args.seed,
                  verdict=res.verdict, reason=res.reason, walls=res.walls, doors=res.doors,
                  second_walls=res.stats.get("second_walls"), seconds=secs,
                  live_inside=res.stats.get("live_inside"))
    (out / "exp2_two_walls_3d.json").write_text(json.dumps(report, indent=2, default=str))
    if res.poly is not None and res.poly.vertices is not None:
        figure(cfg, arm, scene, surf, p3, res, out / "exp2_two_walls_3d.png",
               f"3-D EXP 2, page (++-), every door live: {res.verdict}")
    print(f"[exp2-3d-2w] wrote {out}")


if __name__ == "__main__":
    main()
