#!/usr/bin/env python3
"""Figure: the first d = 3 experiment (findings 42) -- scene, page and wall.

  (a) WORKSPACE. EXP 2's weld line swept +/-5 cm in z into a flat patch; the
      ZGAP boxes; the LWA 4D at its start posture; the two q2 door curves
      drawn where the tool would be on the patch.
  (b) THE 3-D PAGE (u, v, theta) of sign branch (++-): roadmap states by
      component and the verified wall -- a closed triangle mesh, every
      triangle checked inside C_obs by Algorithm 3.
  (c) THE SLICE v = 0.5 of that page, which is exactly the 2-D page (y, theta)
      of EXP 2: the wall's cross-section is a closed curve around the start.

    python3 scripts/fig_3d_scene.py --out out/fig_3d_scene.png
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_3d as R  # noqa: E402
import visualize as V  # noqa: E402
import zzy_rss.page3 as P  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.params import Config, Page3Params, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

SIGNS = (1, 1, -1)


def solve():
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=R.ZGAP))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(R.PATCHES["exp2"])
    p3 = Page3Params()
    keep = {}

    class Capture(P.Page3):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            keep["page"] = self

    P.Page3 = Capture
    res = P.prove3(cfg, arm, surf, scene, SIGNS, p3, verbose=False)
    curves = door_curves(cfg, arm, surf, p3)
    return cfg, arm, scene, surf, p3, res, keep["page"], curves


def slice_segments(verts, tris, v0):
    """Cross-section of a triangle mesh with the plane v = v0, as segments."""
    segs = []
    for t in tris:
        p = verts[t]
        s = p[:, 1] - v0
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (s[a] > 0) != (s[b] > 0):
                w = s[a] / (s[a] - s[b])
                pts.append(p[a] + w * (p[b] - p[a]))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fig_3d_scene.png")
    args = ap.parse_args()
    cfg, arm, scene, surf, p3, res, page, curves = solve()
    poly = res.poly
    start = np.asarray(res.stats["start"])

    fig = plt.figure(figsize=(17, 5.8))

    # ---- (a) workspace -------------------------------------------------
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    corners = np.array([surf.position(0, 0), surf.position(1, 0),
                        surf.position(1, 1), surf.position(0, 1)])
    ax.add_collection3d(Poly3DCollection([corners], facecolor="#e9c46a", alpha=0.35,
                                         edgecolor="#b08a20", linewidths=1.0))
    line = np.array([surf.position(0, 0.5), surf.position(1, 0.5)])
    ax.plot(*line.T, color="#e76f51", lw=2.5, label="EXP 2 weld line (v = 0.5)")
    e0 = np.array([surf.position(0, 0), surf.position(0, 1)])
    e1 = np.array([surf.position(1, 0), surf.position(1, 1)])
    ax.plot(*e0.T, color="#2a9d8f", lw=4, label="start edge u = 0")
    ax.plot(*e1.T, color="#d62828", lw=4, label="goal edge u = 1")
    for c in curves:
        w = np.array([surf.position(u, v) for u, v, _ in c.pts])
        ax.plot(*w.T, color="#7b2cbf", lw=1.6, ls="--")
    ax.plot([], [], color="#7b2cbf", ls="--", label="q2 door curves (discharged)")
    for c_, h_, _lab in scene.boxes:
        V.draw_box(ax, c_, h_)
    r = arm.ik_branch(*surf.pose(start[0], start[1]), start[2], SIGNS)
    V.draw_arm(ax, arm, scene, r.q, label="LWA 4D at the start posture")
    ax.plot([0, 0], [0, 0], [0, 0.24], color="#888888", lw=8, solid_capstyle="round")
    ax.set_xlim(-0.25, 0.55)
    ax.set_ylim(-0.40, 0.40)
    ax.set_zlim(0.0, 0.85)
    ax.set_box_aspect((0.8, 0.8, 0.85))
    ax.set_xlabel("x [m]", labelpad=-4)
    ax.set_ylabel("y [m]", labelpad=-4)
    ax.set_zlabel("z [m]", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=18, azim=-60)
    ax.set_title("(a) workspace: a flat weld PATCH (2-D task)\n"
                 "EXP 2's line swept +/-5 cm in z, ZGAP boxes", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # ---- (b) the 3-D page --------------------------------------------
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    goal_comps = set(int(c) for c in page.comp[page.goal_mask()])
    mstart = page.comp == page.start_comp
    mgoal = np.isin(page.comp, sorted(goal_comps))
    other = ~(mstart | mgoal)
    th = np.pi
    ax.scatter(page.valid[other, 0], page.valid[other, 1], page.valid[other, 2] / th,
               s=1, color="#bbbbbb", alpha=0.5, label="other valid states")
    ax.scatter(page.valid[mgoal, 0], page.valid[mgoal, 1], page.valid[mgoal, 2] / th,
               s=2, color="#d62828", alpha=0.6, label="goal component (rest class)")
    ax.scatter(page.valid[mstart, 0], page.valid[mstart, 1], page.valid[mstart, 2] / th,
               s=3, color="#1d6fd8", alpha=0.9, label="start component (seed class)")
    vv = poly.vertices.copy()
    vv[:, 2] /= th
    ax.add_collection3d(Poly3DCollection(vv[poly.simplices], facecolor="#48cae4",
                                         alpha=0.10, edgecolor="#0077b6",
                                         linewidths=0.08))
    for c in curves:
        ax.plot(c.pts[:, 0], c.pts[:, 1], c.pts[:, 2] / th, color="#7b2cbf", lw=1.6,
                ls="--")
    ax.scatter(*start[:2], start[2] / th, color="#2a9d8f", s=60, edgecolor="k",
               depthshade=False, label="start posture")
    ax.set_xlabel("u (along the weld)", labelpad=-2)
    ax.set_ylabel("v (across the patch)", labelpad=-2)
    ax.set_zlabel("theta / pi (arm angle)", labelpad=-2)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=22, azim=-58)
    w = res.walls[0]
    ax.set_title(f"(b) the 3-D page (u, v, theta) of branch (++-)\n"
                 f"wall: {len(poly.simplices)} triangles, {w['facets']} in C_obs, "
                 f"gamma {w['gamma']}  ->  {res.verdict}", fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4, color="#1d6fd8", label="start component (seed class)"),
        Line2D([], [], ls="", marker="o", ms=4, color="#d62828", label="goal component (rest class)"),
        Line2D([], [], ls="", marker="o", ms=3, color="#bbbbbb", label="other valid states"),
        Line2D([], [], ls="", marker="o", ms=7, color="#2a9d8f", mec="k", label="start posture"),
        Line2D([], [], color="#48cae4", lw=6, alpha=0.5, label="verified wall (triangle mesh)"),
        Line2D([], [], color="#7b2cbf", ls="--", label="q2 door curves (discharged)"),
    ], fontsize=7, loc="upper left")

    # ---- (c) the slice v = 0.5 = the 2-D page of EXP 2 -----------------
    ax = fig.add_subplot(1, 3, 3)
    band = np.abs(page.valid[:, 1] - 0.5) < 0.04
    for m, col, lab in ((other, "#bbbbbb", "other valid"), (mgoal, "#d62828", "goal comp."),
                        (mstart, "#1d6fd8", "start comp.")):
        sel = band & m
        ax.scatter(page.valid[sel, 0], page.valid[sel, 2] / th, s=8, color=col,
                   label=f"{lab} (|v-0.5|<0.04)")
    for a, b in slice_segments(poly.vertices, poly.simplices, 0.5):
        ax.plot([a[0], b[0]], [a[2] / th, b[2] / th], color="#0077b6", lw=1.6)
    ax.plot([], [], color="#0077b6", lw=1.6, label="wall cross-section at v=0.5")
    for c in curves:
        k = np.argmin(np.abs(c.pts[:, 1] - 0.5))
        ax.scatter(c.pts[k, 0], c.pts[k, 2] / th, marker="x", s=70, color="#7b2cbf")
    ax.scatter([], [], marker="x", color="#7b2cbf", label="q2 door (discharged)")
    ax.scatter(start[0], start[2] / th, s=80, color="#2a9d8f", edgecolor="k", zorder=5,
               label="start (u=0)")
    ax.axvspan(1 - p3.goal_tol, 1.0, color="#d62828", alpha=0.12)
    ax.axvline(0, color="#2a9d8f", lw=1)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("u  (= y of the 2-D page)")
    ax.set_ylabel("theta / pi")
    ax.set_title("(c) slice v = 0.5 of the 3-D page = EXP 2's 2-D page (y, theta)\n"
                 "the 3-D wall cut by this plane is a closed curve around the start",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({res.verdict}: {res.reason})")


if __name__ == "__main__":
    main()
