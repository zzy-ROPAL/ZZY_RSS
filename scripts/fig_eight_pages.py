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
WALL_COLORS = ["#000000", "#1d4ed8", "#be185d"]


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


def slice_mesh(V, S, v0):
    """Segments where the triangles of a wall mesh cross the plane v = v0."""
    segs = []
    d = V[:, 1] - v0
    for tri in S:
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[i], tri[j]
            if (d[a] < 0) != (d[b] < 0):
                t = d[a] / (d[a] - d[b])
                pts.append(V[a] + t * (V[b] - V[a]))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def path_page_coords(arm, surf, Q):
    """(u, v, theta, page) of every configuration of a path: the tool position
    on the patch gives (u, v), the elbow gives theta, the joint signs the page."""
    from zzy_rss.chart import arm_angle
    M = np.column_stack([surf.du, surf.dv])
    u, v, th, pg = [], [], [], []
    for q in Q:
        T, _ = arm.fk(q)
        uv = np.linalg.lstsq(M, T - surf.p0, rcond=None)[0]
        u.append(uv[0])
        v.append(uv[1])
        th.append(arm_angle(arm, q))
        pg.append(page_name(np.where(q[[1, 3, 5]] >= 0, 1, -1)))
    return dict(u=np.array(u), v=np.array(v), theta=np.array(th), page=np.array(pg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="exp3-new")
    ap.add_argument("--v0", type=float, default=None, help="slice (default: the start's v)")
    ap.add_argument("--grid", type=int, default=241)
    ap.add_argument("--out", default="docs/fig_eight_pages_exp3.png")
    ap.add_argument("--geometry", default=None,
                    help="pickle from run_recursive.py --save-geometry: draws the walls and the path")
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

    walls, path = [], None
    if args.geometry:
        import pickle
        g = pickle.load(open(args.geometry, "rb"))
        walls = g["walls"]
        if g.get("path_q") is not None:
            path = path_page_coords(arm, surf, g["path_q"])
    nodes = list(dict.fromkeys(w["node"] for w in walls))
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
        for w in walls:
            if w["page"] != page_name(sg):
                continue
            col = WALL_COLORS[nodes.index(w["node"]) % len(WALL_COLORS)]
            for a, b in slice_mesh(w["vertices"], w["simplices"], v0):
                ax.plot([a[0], b[0]], np.degrees([a[2], b[2]]), "-", c=col, lw=1.8)
        if path is not None:
            on = path["page"] == page_name(sg)
            if on.any():
                ax.plot(path["u"][on], np.degrees(path["theta"][on]), ".", ms=1.2, c="#1d3557")
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
    for k, nd in enumerate(nodes):
        handles.append(plt.Line2D([], [], c=WALL_COLORS[k % len(WALL_COLORS)], lw=1.8,
                                  label=f"wall for {nd} (cut at v)"))
    if path is not None:
        handles.append(plt.Line2D([], [], ls="", marker=".", c="#1d3557", ms=6,
                                  label="verified path (all v, projected)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle(f"{args.case}: the eight pages at v = {v0:.2f}  --  doors found by the sigma "
                 f"search with the guaranteed L ({len(ds.doors)} region(s), complete="
                 f"{ds.complete}, {t_doors:.1f} s)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[fig8] saved {args.out} ({time.time() - t0:.1f} s)")
    if walls:
        out3 = str(Path(args.out).with_name(Path(args.out).stem + "_wall3d.png"))
        wall_3d(walls, path, ds, p3, th0, start_signs, out3)
        print(f"[fig8] saved {out3}")


def wall_3d(walls, path, ds, p3, th0, start_signs, out):
    """Every wall of the proof tree in (u, v, theta), with the doors and the path."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(7.5 * len(walls), 7))
    for k, w in enumerate(walls):
        ax = fig.add_subplot(1, len(walls), k + 1, projection="3d")
        V = w["vertices"].copy()
        V[:, 2] = np.degrees(V[:, 2])
        ax.add_collection3d(Poly3DCollection(V[w["simplices"]], facecolor="#8ecae6", alpha=0.25,
                                             edgecolor="#219ebc", lw=0.15))
        for d in ds.doors:
            for cv in d.curves:
                ax.plot(cv.pts[:, 0], cv.pts[:, 1], np.degrees(cv.pts[:, 2]), ".", ms=2,
                        c="#c1121f")
        if path is not None:
            on = path["page"] == w["page"]
            ax.plot(path["u"][on], path["v"][on], np.degrees(path["theta"][on]), ".", ms=1,
                    c="#1d3557")
        if w["page"] == page_name(start_signs) and th0 is not None:
            ax.scatter([p3.start_u], [p3.start_v], [np.degrees(th0)], marker="*", s=200,
                       c="#ffd60a", edgecolors="k")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-180, 180)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel("theta (deg)")
        ax.view_init(elev=22, azim=-60)
        ax.set_title(f"wall on page {w['page']} ({'verified' if w['ok'] else 'FAILED'}): "
                     f"{len(w['simplices'])} triangles\n{w['node']}\n"
                     f"red: door curves (sigma = 0), dots: verified path on this page, star: start",
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
