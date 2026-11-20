#!/usr/bin/env python3
"""A rendered 3-D scene of a weld-patch case, in the style of the paper's
experiment figures: the arm at the start posture, a faded arm at a valid goal
posture, the weld patch, the obstacle cubes, and START / GOAL labels.

    python3 scripts/fig_scene3d.py --case two-wall --out docs/fig_scene_two_wall.png

Geometry is the collision model the proofs use (capsules on the
shoulder-elbow-wrist-tool chain); the pedestal and floor are drawing only.
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

from run_recursive import CASES  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.batch import collision_batch, ik_batch  # noqa: E402
from zzy_rss.chart import limit_margins  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402
from zzy_rss.params import Config, Page3Params, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

LINK = "#d9d9d9"
JOINT = "#3a3a3a"
CUBE_COLORS = ["#f2d22e", "#d62828", "#f2d22e", "#d62828"]


def valid_postures(arm, scene, surf, u, v, signs_list, n_theta=361):
    """Valid (q, signs, theta) at task point (u, v): IK exists, limits, no collision."""
    pos, rot = surf.pose(u, v)
    th = np.linspace(-np.pi, np.pi, n_theta)
    out = []
    for signs in signs_list:
        ok, q = ik_batch(arm, np.repeat(pos[None], len(th), 0), rot, th, tuple(signs))
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        inlim = np.all(limit_margins(arm, q[idx]) <= 0.0, axis=1)
        idx = idx[inlim]
        if not len(idx):
            continue
        free = ~np.asarray(collision_batch(arm, scene, q[idx]), dtype=bool)
        for k in idx[free]:
            out.append((q[k], tuple(signs), th[k]))
    return out


def cylinder(ax, a, b, r, color, alpha, n=24):
    a, b = np.asarray(a, float), np.asarray(b, float)
    axis = b - a
    L = np.linalg.norm(axis)
    if L < 1e-9:
        return
    w = axis / L
    t = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, t)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    ang = np.linspace(0, 2 * np.pi, n)
    s = np.linspace(0, 1, 2)
    A, S = np.meshgrid(ang, s)
    P = (a[:, None, None] + axis[:, None, None] * S[None]
         + r * (np.cos(A)[None] * u[:, None, None] + np.sin(A)[None] * v[:, None, None]))
    ax.plot_surface(P[0], P[1], P[2], color=color, alpha=alpha, linewidth=0, shade=True)


def sphere(ax, c, r, color, alpha, n=16):
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n))
    ax.plot_surface(c[0] + r * np.cos(uu) * np.sin(vv), c[1] + r * np.sin(uu) * np.sin(vv),
                    c[2] + r * np.cos(vv), color=color, alpha=alpha, linewidth=0, shade=True)


def box(ax, center, half, color, alpha=1.0, edge="#333333"):
    c, h = np.asarray(center, float), np.asarray(half, float)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    V = c + corners * h
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([V[f] for f in faces], facecolor=color, alpha=alpha,
                                         edgecolor=edge, linewidths=0.6))


def draw_arm(ax, arm, scene, q, alpha):
    f = arm.frames(q)
    p = arm.p
    cylinder(ax, [0, 0, 0], f["shoulder"], 0.075, "#8c8c8c", alpha)       # pedestal
    for a, b, r, _ in scene.capsules(f):
        cylinder(ax, a, b, r, LINK, alpha)
    for key, r in (("shoulder", p.r_upper_arm * 1.25), ("elbow", p.r_upper_arm * 1.2),
                   ("wrist", p.r_forearm * 1.2)):
        sphere(ax, f[key], r, JOINT, alpha)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="two-wall")
    ap.add_argument("--out", default="docs/fig_scene_two_wall.png")
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=-128.0)
    args = ap.parse_args()

    sp, obstacles, start_signs, _ = CASES[args.case]
    cfg = dataclasses.replace(Config(), scene=SceneParams(gap_open=True,
                                                          obstacles=tuple(obstacles)))
    arm, scene, p3 = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene), Page3Params()
    surf = WeldSurface(sp)

    th0 = choose_start_theta(cfg, arm, surf, scene, start_signs, p3)
    pos, rot = surf.pose(p3.start_u, p3.start_v)
    r0 = arm.ik_branch(pos, rot, th0, tuple(start_signs))
    assert r0.ok, "no start posture"
    goal = valid_postures(arm, scene, surf, 1.0, p3.start_v, SIGN_BRANCHES)
    print(f"[scene] start theta {np.degrees(th0):.1f} deg on {start_signs}; "
          f"{len(goal)} valid goal postures at u=1")

    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp", focal_length=0.35)

    # Floor grid.
    g = np.linspace(-0.4, 0.8, 13)
    for x in g:
        ax.plot([x, x], [-0.6, 0.6], [0, 0], c="#bbbbbb", lw=0.5)
    for y in np.linspace(-0.6, 0.6, 13):
        ax.plot([-0.4, 0.8], [y, y], [0, 0], c="#bbbbbb", lw=0.5)

    # Weld patch: the tool-tip surface p(u, v), and the seam's start/end edges.
    U, V = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 5))
    P = np.array([[surf.pose(u, v)[0] for u, v in zip(ru, rv)] for ru, rv in zip(U, V)])
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color="#f4a261", alpha=0.55, linewidth=0)
    for uu, col in ((0.0, "#2a9d8f"), (1.0, "#c1121f")):
        E = np.array([surf.pose(uu, v)[0] for v in np.linspace(0, 1, 10)])
        ax.plot(E[:, 0], E[:, 1], E[:, 2], c=col, lw=3)
    # Workpiece under the seam (drawing only).
    lo = P.reshape(-1, 3).min(axis=0)
    hi = P.reshape(-1, 3).max(axis=0)
    box(ax, [(lo[0] + 0.62) / 2, (lo[1] + hi[1]) / 2, lo[2] / 2],
        [(0.62 - lo[0]) / 2, (hi[1] - lo[1]) / 2 + 0.03, lo[2] / 2], "#5c6b73", 0.85)

    for k, b in enumerate(cfg.scene.obstacles):
        box(ax, b.center, b.half_extent, CUBE_COLORS[k % len(CUBE_COLORS)], 0.95)

    if goal:
        qg = goal[len(goal) // 2][0]
        fg = draw_arm(ax, arm, scene, qg, 0.28)
        ax.text(fg["tool"][0], fg["tool"][1] + 0.08, fg["tool"][2] + 0.12, "GOAL",
                fontsize=11, fontweight="bold", ha="center",
                bbox=dict(fc="white", ec="black", lw=0.8))
    fs = draw_arm(ax, arm, scene, r0.q, 1.0)
    ax.text(fs["tool"][0], fs["tool"][1] - 0.08, fs["tool"][2] + 0.25, "START",
            fontsize=11, fontweight="bold", ha="center",
            bbox=dict(fc="white", ec="black", lw=0.8))

    ax.set_xlim(-0.4, 0.8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(0.0, 0.9)
    ax.set_box_aspect((1.2, 1.2, 0.9))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[scene] saved {args.out}")


if __name__ == "__main__":
    main()
