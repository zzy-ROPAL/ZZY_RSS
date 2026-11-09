#!/usr/bin/env python3
"""3-D EXP 2 (ZGAP weld patch): all eight pages, and the two-wall proof (findings 47c).

Figure 1 -- the eight pages (s2, s4, s6) as volumes in (u, v, theta),
coloured by WHY a state is invalid (same colour code as fig_pages3d.py):
collision is identical on every page (invariant 49), the limits move. The
q2 door curves are drawn on every page; the start posture on (++-).

Figure 2 -- the start page (++-) with the verified proof surfaces from
scripts/exp2_two_walls_3d.py: wall 1 (start | goal, both live doors inside)
and one wall 2 per door curve (start | that door), next to the same page
without walls for reference.

Surfaces of the page volume are marching cubes on a grid: a picture, never a
proof (invariant 8). The walls are the verified TDC meshes themselves.

    PYTHONPATH=src python3 scripts/exp2_two_walls_3d.py      # writes the .pkl
    PYTHONPATH=src python3 scripts/fig_pages3d_exp2.py \\
        --data out/exp2_two_walls_3d/exp2_two_walls_3d.pkl --out docs/figs/exp2_pages3d.png
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

import exp2_two_walls_3d as E  # noqa: E402
from fig_pages3d import CATS, classify_grid, draw_volume, frame, goal_face, page_name  # noqa: E402
from run_3d import START_SIGNS  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402

DOOR_COL = "#ff8800"
WALL1 = ("#9d0208", 0.22)
WALL2 = (("#1d3557", 0.16), ("#7b2cbf", 0.16))


def draw_doors(ax, curves):
    for c in curves:
        ax.plot(c.pts[:, 0], c.pts[:, 1], c.pts[:, 2] / np.pi, color=DOOR_COL, lw=2.6)


def draw_mesh(ax, verts, simp, col, alpha):
    v = np.asarray(verts, dtype=float).copy()
    v[:, 2] /= np.pi
    ax.add_collection3d(Poly3DCollection(v[np.asarray(simp)], facecolor=col, alpha=alpha,
                                         edgecolor=col, linewidth=0.05))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/exp2_two_walls_3d/exp2_two_walls_3d.pkl")
    ap.add_argument("--res", default="41,13,81", help="grid u,v,theta")
    ap.add_argument("--out", default="docs/figs/exp2_pages3d.png")
    args = ap.parse_args()
    cfg, arm, scene, surf, p3 = E.setup(6000, 20261108, "enclose", True, True)
    curves = door_curves(cfg, arm, surf, p3)
    nu, nv, nt = (int(c) for c in args.res.split(","))
    us, vs, ts = np.linspace(0, 1, nu), np.linspace(0, 1, nv), np.linspace(-np.pi, np.pi, nt)
    start = START_SIGNS["exp2"]
    th0 = choose_start_theta(cfg, arm, surf, scene, start, p3)
    codes = {sg: classify_grid(cfg, arm, scene, surf, sg, us, vs, ts) for sg in SIGN_BRANCHES}

    # ---- figure 1: the eight pages -------------------------------------
    fig = plt.figure(figsize=(22, 10.5))
    for k, sg in enumerate(SIGN_BRANCHES):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        draw_volume(ax, codes[sg], us, vs, ts)
        draw_doors(ax, curves)
        goal_face(ax)
        if sg == start and th0 is not None:
            ax.scatter([0.0], [p3.start_v], [th0 / np.pi], s=80, color="#1d6fd8",
                       edgecolor="k", depthshade=False)
        valid = float((codes[sg] == 0).mean())
        tag = "  START PAGE" if sg == start else (
            "  q2 partner (s2 flipped)" if sg == (-start[0], start[1], start[2]) else "")
        frame(ax, f"page {page_name(sg)}  (s2 s4 s6){tag}\nvalid {100 * valid:.0f}% of the box")
    fig.legend(handles=[Line2D([], [], color=c, lw=8, alpha=0.6, label=lab)
                        for _c, c, _a, lab in CATS]
               + [Line2D([], [], color=DOOR_COL, lw=2.6, label="q2 door curves (s2 flips)"),
                  Line2D([], [], ls="", marker="o", ms=8, color="#1d6fd8", mec="k",
                         label="start posture"),
                  Line2D([], [], color="#d62828", label="goal face u = 1")],
               loc="lower center", ncol=7, fontsize=10)
    fig.suptitle("3-D EXP 2 -- all eight pages (u, v, theta) of the ZGAP weld patch "
                 "(EXP 2 seam swept +/-5 cm in z, two 12 cm boxes). White = valid. "
                 "Collision is identical on all pages; only the joint limits move.", fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.07, wspace=0.0, hspace=0.12)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- figure 2: the start page with the verified walls --------------
    if not args.data or not Path(args.data).exists():
        return
    d = pickle.load(open(args.data, "rb"))
    fig = plt.figure(figsize=(22, 15))
    panels = [("page only", None)]
    if d["wall1"] is not None:
        panels.append(("wall 1: start | goal", [(d["wall1"], *WALL1)]))
    for i, w in enumerate(d["walls2"]):
        panels.append((f"wall 2: start | door curve {i}", [(w, *WALL2[i % 2])]))
    # Row 1: perspective. Row 2: the same, seen ALONG v (theta up, u right):
    # the door curves run along v, so each projects to a short mark at
    # theta = 0, and the gap between it and a wall 2 is visible -- in
    # perspective a see-through wall seems to swallow a curve just below it.
    for j, (title, meshes) in [(j + len(panels) * row, pm) for row in (0, 1)
                               for j, pm in enumerate(panels)]:
        side = j >= len(panels)
        ax = fig.add_subplot(2, len(panels), j + 1, projection="3d")
        if meshes is None:
            draw_volume(ax, codes[start], us, vs, ts)
        for w, col, alpha in meshes or []:
            draw_mesh(ax, w["vertices"], w["simplices"], col, alpha)
        draw_doors(ax, curves)
        goal_face(ax)
        st = d.get("start")
        if st is not None:
            ax.scatter([st[0]], [st[1]], [st[2] / np.pi], s=80, color="#1d6fd8",
                       edgecolor="k", depthshade=False)
        frame(ax, f"page {page_name(start)}: {title}" + ("  (seen along v)" if side else ""))
        ax.set_xlim(-0.12, 1.0)
        if side:
            ax.view_init(elev=0, azim=-90)
            ax.set_ylabel("")
            ax.set_yticks([])
    rep = d["report"]
    w1 = rep["walls"][0]
    fig.suptitle(f"3-D EXP 2, start page {page_name(start)}, both q2 door curves LIVE -- "
                 f"{d['verdict']} on one page: wall 1 {w1['facets']} triangles, "
                 + ", ".join(f"wall 2 #{i} {s['wall']['facets']}"
                             for i, s in enumerate(rep["second_walls"] or []))
                 + " -- every triangle verified in C_obs", fontsize=12)
    fig.legend(handles=[Line2D([], [], color=WALL1[0], lw=8, alpha=0.5, label="wall 1 (start | goal)"),
                        Line2D([], [], color=WALL2[0][0], lw=8, alpha=0.5, label="wall 2 (start | door 0)"),
                        Line2D([], [], color=WALL2[1][0], lw=8, alpha=0.5, label="wall 2 (start | door 1)"),
                        Line2D([], [], color=DOOR_COL, lw=2.6, label="q2 door curves (live)"),
                        Line2D([], [], ls="", marker="o", ms=8, color="#1d6fd8", mec="k",
                               label="start posture"),
                        Line2D([], [], color="#d62828", label="goal face u = 1")],
               loc="lower center", ncol=6, fontsize=10)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.06, wspace=0.0, hspace=0.08)
    out2 = out.with_name(out.stem + "_walls" + out.suffix)
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
