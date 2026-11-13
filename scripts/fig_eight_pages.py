#!/usr/bin/env python3
"""The eight pages of a weld-patch task on one slice v = v0, with the doors the
proof found from sigma (method 2: certified box search with the guaranteed L).

Each panel is one sign branch (s2, s4, s6), drawn as (u, theta) at v = v0 and
coloured by WHY a state is invalid. On top:

    orange boxes  the certified door regions of the sigma search, cut at v0
                  (every door is inside one; every other box was PROVEN door-free)
    red dots      the door curve points (sigma = 0) refined inside them
    star          the start posture (start page only)
    red line      the goal edge u = 1

The door regions are the same on all eight pages (sigma is page-independent:
the eight pages are eight spellings of one physical posture); what differs is
WHICH pages a door glues: a q2 door flips s2, a q6 door flips s6.

The colouring is a grid picture, never a proof (invariant 8).

    python3 scripts/fig_eight_pages.py --case exp3-new --out docs/fig_eight_pages_exp3.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_recursive import CASES  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.doorset import find_doorset  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402
from zzy_rss.params import Config, Page3Params, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

# 0 valid, 1 collision, 2 q7 limit, 3 q2/q4/q6 limit or no IK, 4 q1/q3/q5 limit only
COLORS = ["#f4f4f4", "#e76f51", "#2a9d8f", "#e9c46a", "#b392d6"]
NAMES = ["valid", "collision (box)", "q7 beyond limit", "q2/q4/q6 limit or no IK",
         "q1/q3/q5 limit only"]
FLIP = {"q2": 0, "q4": 1, "q6": 2}


def page_name(sg) -> str:
    return "".join("+" if s > 0 else "-" for s in sg)


def classify_slice(cfg, arm, scene, surf, signs, U, T, v0):
    Z = np.column_stack([U.ravel(), np.full(U.size, v0), T.ravel()])
    ok, tag, q = classify_surface_batch(cfg, arm, surf, scene, Z, signs)
    bad = (q < arm.q_min) | (q > arm.q_max)
    code = np.zeros(len(Z), dtype=int)
    code[tag == 3] = 1
    code[(tag == 2) & bad[:, 6]] = 2
    thick = bad[:, 1] | bad[:, 3] | bad[:, 5]
    thin = bad[:, 0] | bad[:, 2] | bad[:, 4]
    code[(tag == 2) & ~bad[:, 6] & thick] = 3
    code[(tag == 2) & ~bad[:, 6] & ~thick & thin] = 4
    code[tag == 1] = 3
    return code.reshape(U.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="exp3-new")
    ap.add_argument("--v0", type=float, default=None, help="slice (default: the start's v)")
    ap.add_argument("--grid", type=int, default=241)
    ap.add_argument("--out", default="docs/fig_eight_pages_exp3.png")
    args = ap.parse_args()

    sp, obstacles, start_signs, _ = CASES[args.case]
    cfg = dataclasses.replace(Config(), scene=SceneParams(gap_open=True, obstacles=tuple(obstacles)))
    arm, p3 = SrsArm(cfg.arm), Page3Params()
    scene = Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(sp)
    v0 = p3.start_v if args.v0 is None else args.v0

    t0 = time.time()
    ch = SrsChart.for_surface(arm, surf, p3)
    ds = find_doorset(ch, p3.tube_radius)
    t_doors = time.time() - t0
    print(f"[fig8] sigma door search: {len(ds.doors)} region(s), complete={ds.complete} "
          f"({t_doors:.1f} s)")
    th0 = choose_start_theta(cfg, arm, surf, scene, start_signs, p3)

    U, T = np.meshgrid(np.linspace(0, 1, args.grid), np.linspace(-np.pi, np.pi, args.grid))
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.6), sharex=True, sharey=True)
    cmap = ListedColormap(COLORS)
    for ax, sg in zip(axs.ravel(), SIGN_BRANCHES):
        code = classify_slice(cfg, arm, scene, surf, sg, U, T, v0)
        ax.pcolormesh(U, np.degrees(T), code, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto",
                      rasterized=True)
        partners = []
        for i, d in enumerate(ds.doors):
            r = d.region
            cut = np.abs(r.centres[:, 1] - v0) <= r.half[:, 1]
            for c, h in zip(r.centres[cut], r.half[cut]):
                ax.add_patch(Rectangle((c[0] - h[0], np.degrees(c[2] - h[2])), 2 * h[0],
                                       np.degrees(2 * h[2]), fill=False, ec="#e6550d", lw=0.5))
            for cv in d.curves:
                near = np.abs(cv.pts[:, 1] - v0) < 0.01
                ax.plot(cv.pts[near, 0], np.degrees(cv.pts[near, 2]), ".", ms=4, c="#c1121f")
            for k in sorted(d.region.kinds):
                if k in FLIP:
                    other = list(sg)
                    other[FLIP[k]] *= -1
                    partners.append(f"D{i}:{k}->{page_name(other)}")
        ax.axvline(1.0, color="#c1121f", lw=2.5)
        is_start = tuple(sg) == tuple(start_signs)
        if is_start and th0 is not None:
            ax.plot([p3.start_u], [np.degrees(th0)], "*", ms=16, mfc="#ffd60a", mec="k")
        frac = float(np.mean(code == 0))
        ax.set_title(f"page {page_name(sg)}{'  (START)' if is_start else ''}   "
                     f"valid {100 * frac:.0f}%\n" + ", ".join(sorted(set(partners))),
                     fontsize=9, fontweight="bold" if is_start else "normal")
        ax.set_xlim(0, 1)
        ax.set_ylim(-180, 180)
    for ax in axs[1]:
        ax.set_xlabel("u (along the weld)")
    for ax in axs[:, 0]:
        ax.set_ylabel("arm angle theta (deg)")
    handles = [Patch(fc=c, ec="k", lw=0.3, label=n) for c, n in zip(COLORS, NAMES)]
    handles += [Patch(fill=False, ec="#e6550d", label="certified door region (sigma + L)"),
                plt.Line2D([], [], ls="", marker=".", c="#c1121f", ms=8, label="door curve (sigma = 0)"),
                plt.Line2D([], [], ls="", marker="*", mfc="#ffd60a", mec="k", ms=12, label="start"),
                plt.Line2D([], [], c="#c1121f", lw=2.5, label="goal edge u = 1")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle(f"{args.case}: the eight pages at v = {v0:.2f}  --  doors found by the sigma "
                 f"search with the guaranteed L ({len(ds.doors)} region(s), complete="
                 f"{ds.complete}, {t_doors:.1f} s)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[fig8] saved {args.out} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
