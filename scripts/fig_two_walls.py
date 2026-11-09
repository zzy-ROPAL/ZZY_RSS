#!/usr/bin/env python3
"""The two-wall infeasibility proof in 3-D (findings 46), from run_3d.py --two-page --save.

Top: page 1 (start) and page 2 (after the wrist flip) as (u, v, theta) volumes
with the whole verified wall of each page's main region. Bottom: the same at
v = 0.5 (white valid, red collision, teal joint limits / no IK) with the
walls' cross-sections.

    python3 scripts/fig_two_walls.py out/exp2_two_wall.pkl --out out/fig_exp2_two_walls.png
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_scene3 as S  # noqa: E402
from fig_3d_scene import slice_segments  # noqa: E402
from fig_pages3d import classify_grid, draw_volume, frame, doors, goal_face  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="out/fig_exp2_two_walls.png")
    args = ap.parse_args()
    d = pickle.load(open(args.data, "rb"))
    x, z, tilt, roll = (float(c) for c in d["scene"].split(",")[:4])
    boxes = [[float(c) for c in b.split(",")] for b in d["boxes"]]
    cfg, arm, scene, surf = S.world(dict(x=x, z=z, tilt=tilt, roll=roll, boxes=boxes))
    A = tuple(d["signs"])
    B = (A[0], A[1], -A[2])
    p3 = Page3Params()
    curves = door_curves(cfg, arm, surf, p3)
    th0 = choose_start_theta(cfg, arm, surf, scene, A, p3)
    walls = {(w["page"], w["comp"]): w for w in d["task_walls"]}
    t = d["out"]["task"]
    info = {(w["page"], w["comp"]): w for w in t["walls"]}
    name = lambda sg: "".join("+" if s > 0 else "-" for s in sg)  # noqa: E731
    us, vs, ts = np.linspace(0, 1, 41), np.linspace(0, 1, 13), np.linspace(-np.pi, np.pi, 81)

    fig = plt.figure(figsize=(17, 14))
    panels = ((A, "page 1", "start region", "#1d6fd8"),
              (B, "page 2", "region entered through the q6 door", "#d62828"))
    for i, (sg, pl, what, col) in enumerate(panels):
        key = (name(sg), 0)
        w, wi = walls[key], info[key]
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        draw_volume(ax, classify_grid(cfg, arm, scene, surf, sg, us, vs, ts), us, vs, ts)
        vv = w["vertices"].copy()
        vv[:, 2] /= np.pi
        ax.add_collection3d(Poly3DCollection(vv[w["simplices"]], facecolor=col, alpha=0.30,
                                             edgecolor=col, linewidths=0.05))
        doors(ax, curves)
        goal_face(ax)
        if sg == A:
            ax.scatter([0], [0.5], [th0 / np.pi], s=80, color="#1d6fd8", edgecolor="k",
                       depthshade=False)
        frame(ax, "")
        ax.set_title(f"{pl} ({name(sg)}): wall around the {what}\n"
                     f"closed, {wi['facets']} triangles verified in C_obs, gamma {wi['gamma']}, "
                     f"ends at u = {wi['u_max']} < goal", fontsize=11)

        ax = fig.add_subplot(2, 2, i + 3)
        U, T = np.meshgrid(np.linspace(0, 1, 301), np.linspace(-np.pi, np.pi, 361), indexing="ij")
        P = np.column_stack([U.ravel(), np.full(U.size, 0.5), T.ravel()])
        _ok, tag, _q = classify_surface_batch(cfg, arm, surf, scene, P, sg)
        code = np.where(tag == 0, 0, np.where(tag == 3, 1, 2)).reshape(U.shape)
        ax.imshow(code.T, origin="lower", extent=[0, 1, -1, 1], aspect="auto", interpolation="nearest",
                  cmap=matplotlib.colors.ListedColormap(["#ffffff", "#e76f51", "#2a9d8f"]),
                  vmin=0, vmax=2)
        for a, b in slice_segments(w["vertices"], w["simplices"], 0.5):
            ax.plot([a[0], b[0]], [a[2] / np.pi, b[2] / np.pi], color=col, lw=2.6)
        for c in curves:
            j = int(np.argmin(np.abs(c.pts[:, 1] - 0.5)))
            ax.plot(c.pts[j, 0], c.pts[j, 2] / np.pi, "*", color="#7b2cbf", ms=18, mec="k")
        if sg == A:
            ax.plot(0, th0 / np.pi, "o", color="#1d6fd8", ms=11, mec="k")
        ax.axvline(1.0, color="#d62828", lw=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("u (along the weld)")
        ax.set_ylabel("theta / pi")
        ax.set_title(f"{pl} ({name(sg)}), slice v = 0.5", fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color="#1d6fd8", lw=3, label="wall 1 (page 1)"),
        Line2D([], [], color="#d62828", lw=3, label="wall 2 (page 2)"),
        Line2D([], [], color="#e76f51", lw=8, alpha=0.6, label="collision (2 boxes)"),
        Line2D([], [], color="#2a9d8f", lw=8, alpha=0.6, label="q7 / joint limits"),
        Line2D([], [], ls="", marker="*", ms=14, color="#7b2cbf", mec="k", label="q6 door"),
        Line2D([], [], ls="", marker="o", ms=9, color="#1d6fd8", mec="k", label="start"),
        Line2D([], [], color="#d62828", lw=4, label="goal u = 1")],
        loc="lower center", ncol=7, fontsize=11)
    fig.suptitle(f"Two-wall infeasibility proof: {t['verdict']} -- {t['reason']}", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.07, hspace=0.18, wspace=0.12)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
