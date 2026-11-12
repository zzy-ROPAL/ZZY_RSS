#!/usr/bin/env python3
"""Where the singularity doors are, found from sigma alone, drawn on a page.

Per patch, three panels:
  1. the whole 3-D page (u, v, theta): certified door-region cells (faint),
     the door curves found by minimising sigma (doorset.py), and the
     closed-form curves (doors3.door_curves, the independent cross-check);
  2. the slice v = v0: sigma = min(|q2|, |q6|) as a heat map, the region
     cells cut by the slice, and the curve points near the slice;
  3. the same slice of one page's validity (valid / joint limit / collision /
     no IK) with the door on top -- where on the page the robot can cross.

The validity map is a picture-only grid; nothing in a proof reads it.

    python3 scripts/fig_sigma_doors.py --out out/fig_sigma_doors.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from run_3d import PATCHES, ZGAP  # noqa: E402
from run_recursive import BOX45  # noqa: E402
from search_scene3 import surface  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.doorset import find_doorset  # noqa: E402
from zzy_rss.params import Config, Page3Params, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CASES = [
    ("3-D EXP 2 patch, ZGAP (two q2 doors)", PATCHES["exp2"], ZGAP, (1, 1, -1)),
    ("new EXP 3 patch (q6 doors)", surface(0.40, 0.21, 45, 305), (BOX45,), (1, 1, 1)),
]
NAMES = ["valid", "no IK", "joint limit", "collision"]
COLORS = ["#d9f0d3", "#bdbdbd", "#fdd0a2", "#9e9ac8"]


def draw(axs, title, sp, obstacles, signs, v0, n_grid):
    cfg = dataclasses.replace(Config(), scene=SceneParams(gap_open=True, obstacles=obstacles))
    arm, p3 = SrsArm(cfg.arm), Page3Params()
    scene = Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(sp)
    ch = SrsChart.for_surface(arm, surf, p3)
    ds = find_doorset(ch, p3.tube_radius)
    closed = door_curves(cfg, arm, surf, p3)
    th_deg = np.degrees

    # 1. the whole page
    ax = axs[0]
    for d in ds.doors:
        C = d.region.centres
        ax.scatter(C[:, 0], C[:, 1], th_deg(C[:, 2]), s=1, c="#fdae6b", alpha=0.04,
                   depthshade=False)
        for c in d.curves:
            ax.plot(c.pts[:, 0], c.pts[:, 1], th_deg(c.pts[:, 2]), ".", ms=2.5, c="#d62728")
    for c in closed:
        on = np.all((c.pts[:, :2] >= 0) & (c.pts[:, :2] <= 1), axis=1)
        ax.plot(c.pts[on, 0], c.pts[on, 1], th_deg(c.pts[on, 2]), "-", lw=0.8, c="k")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xlabel("u"), ax.set_ylabel("v"), ax.set_zlabel("theta (deg)")
    n_cells = sum(len(d.region.centres) for d in ds.doors)
    ax.set_title(f"{title}\n{len(ds.doors)} door region(s), {n_cells} cells (orange); "
                 f"sigma curves (red)\nclosed form (black line, cross-check only)", fontsize=8)

    # 2. sigma on the slice v = v0
    U, T = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(-np.pi, np.pi, n_grid))
    Z = np.column_stack([U.ravel(), np.full(U.size, v0), T.ravel()])
    S = ch.sigma(Z)
    smin = np.fmin(S[:, 0], S[:, 1]).reshape(U.shape)
    ax = axs[1]
    im = ax.pcolormesh(U, th_deg(T), th_deg(smin), cmap="viridis_r", vmin=0, vmax=60,
                       shading="auto")
    plt.colorbar(im, ax=ax, label="sigma = min(|q2|, |q6|) (deg)")
    _overlay(ax, ds, closed, v0, th_deg)
    ax.set_title(f"slice v = {v0}: sigma, certified cells (orange boxes),\n"
                 f"sigma curve points (red), closed form (black x)", fontsize=8)

    # 3. the page's validity on the same slice
    ok, tag, _ = classify_surface_batch(cfg, arm, surf, scene, Z, signs)
    ax = axs[2]
    ax.pcolormesh(U, th_deg(T), np.asarray(tag).reshape(U.shape),
                  cmap=ListedColormap(COLORS), vmin=-0.5, vmax=3.5, shading="auto")
    _overlay(ax, ds, closed, v0, th_deg)
    page = "".join("+" if s > 0 else "-" for s in signs)
    ax.set_title(f"page {page}, slice v = {v0}: " + ", ".join(
        f"{n} ({c})" for n, c in zip(NAMES, ["green", "grey", "orange", "purple"])),
        fontsize=7)
    return ds


def _overlay(ax, ds, closed, v0, th_deg):
    for d in ds.doors:
        r = d.region
        cut = np.abs(r.centres[:, 1] - v0) <= r.half[:, 1]
        for c, h in zip(r.centres[cut], r.half[cut]):
            ax.add_patch(Rectangle((c[0] - h[0], th_deg(c[2] - h[2])), 2 * h[0],
                                   th_deg(2 * h[2]), fill=False, ec="#e6550d", lw=0.4))
        for cv in d.curves:
            near = np.abs(cv.pts[:, 1] - v0) < 0.02
            ax.plot(cv.pts[near, 0], th_deg(cv.pts[near, 2]), "o", ms=3, c="#d62728")
    for c in closed:
        near = (np.abs(c.pts[:, 1] - v0) < 0.02) & (c.pts[:, 0] >= 0) & (c.pts[:, 0] <= 1)
        ax.plot(c.pts[near, 0], th_deg(c.pts[near, 2]), "x", ms=5, c="k")
    ax.set_xlabel("u"), ax.set_ylabel("theta (deg)")
    ax.set_xlim(0, 1), ax.set_ylim(-180, 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fig_sigma_doors.png")
    ap.add_argument("--v0", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=361)
    args = ap.parse_args()
    fig = plt.figure(figsize=(17, 10))
    for row, (title, sp, obs, signs) in enumerate(CASES):
        axs = [fig.add_subplot(2, 3, 3 * row + 1, projection="3d"),
               fig.add_subplot(2, 3, 3 * row + 2), fig.add_subplot(2, 3, 3 * row + 3)]
        ds = draw(axs, title, sp, obs, signs, args.v0, args.grid)
        print(f"[fig] {title}: {[d.label() for d in ds.doors]}")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
