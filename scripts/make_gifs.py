#!/usr/bin/env python3
"""GIFs of the two 3-D experiments (findings 45/46).

  EXP 3 (feasible via a wrist flip): the arm executing the verified path, next
         to the v = 0.5 slice of the page it is on (page 1, then page 2).
  EXP 2 (infeasible, two pages): the two verified walls in (u, v, theta),
         rotating.

    python3 scripts/make_gifs.py --exp3 out/exp3_two_page.pkl --exp2 out/exp2_two_wall.pkl
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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_scene3 as S  # noqa: E402
import visualize as V  # noqa: E402
from fig_pages3d import classify_grid, doors, draw_volume, frame, goal_face  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors import _arm_angle_of_elbow  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402

CMAP = matplotlib.colors.ListedColormap(["#ffffff", "#e76f51", "#2a9d8f"])


def world(d):
    x, z, tilt, roll = (float(c) for c in d["scene"].split(",")[:4])
    boxes = [[float(c) for c in b.split(",")] for b in d["boxes"]]
    return S.world(dict(x=x, z=z, tilt=tilt, roll=roll, boxes=boxes)), boxes


def slice_image(cfg, arm, scene, surf, sg):
    U, T = np.meshgrid(np.linspace(0, 1, 201), np.linspace(-np.pi, np.pi, 241), indexing="ij")
    P = np.column_stack([U.ravel(), np.full(U.size, 0.5), T.ravel()])
    _ok, tag, _q = classify_surface_batch(cfg, arm, surf, scene, P, sg)
    return np.where(tag == 0, 0, np.where(tag == 3, 1, 2)).reshape(U.shape).T


def gif_exp3(path, out, frames=120):
    d = pickle.load(open(path, "rb"))
    (cfg, arm, scene, surf), boxes = world(d)
    A = tuple(d["signs"])
    B = (A[0], A[1], -A[2])
    Q = np.vstack([lg["q"] for lg in d["legs"]])
    UV = np.vstack([lg["y"] for lg in d["legs"]])
    TH = np.array([_arm_angle_of_elbow(arm, surf.line(float(v)), float(u), arm.frames(q)["elbow"])
                   for q, (u, v) in zip(Q, UV)]) / np.pi
    k = int(np.flatnonzero(np.diff(np.sign(Q[:, 5])) != 0)[0]) + 1
    idx = np.unique(np.concatenate([np.linspace(0, k, frames // 2).astype(int),
                                    np.linspace(k, len(Q) - 1, frames // 2).astype(int)]))
    imgs = {A: slice_image(cfg, arm, scene, surf, A), B: slice_image(cfg, arm, scene, surf, B)}
    curves = door_curves(cfg, arm, surf, Page3Params())
    tool = np.array([arm.frames(q)["tool"] for q in Q])

    fig = plt.figure(figsize=(13, 5.6))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    def draw(i):
        j = idx[i]
        on_b = j >= k
        ax3.cla()
        c = np.array([surf.position(0, 0), surf.position(1, 0), surf.position(1, 1), surf.position(0, 1)])
        ax3.add_collection3d(Poly3DCollection([c], facecolor="#e9c46a", alpha=0.35))
        for b in boxes:
            V.draw_box(ax3, b[:3], b[3:6], color="#d62828", alpha=0.25)
        for cv in curves:
            w = np.array([surf.position(u, v) for u, v, _ in cv.pts])
            ax3.plot(*w.T, color="#7b2cbf", ls="--", lw=1.5)
        ax3.plot(*tool[:j + 1].T, color="k", lw=1.2)
        f = arm.frames(Q[j])
        P = np.array([f["shoulder"], f["elbow"], f["wrist"], f["tool"]])
        ax3.plot(*P.T, "-o", color="#d62828" if on_b else "#1d6fd8", lw=4, ms=5)
        ax3.plot([0, 0], [0, 0], [0, 0.24], color="#888888", lw=8)
        ax3.set_xlim(-0.2, 0.6)
        ax3.set_ylim(-0.4, 0.4)
        ax3.set_zlim(0, 0.7)
        ax3.set_box_aspect((0.8, 0.8, 0.7))
        ax3.view_init(elev=22, azim=-62)
        ax3.tick_params(labelsize=6)
        ax3.set_title(f"EXP 3: verified path, configuration {j + 1}/{len(Q)}\n"
                      f"q6 = {np.degrees(Q[j, 5]):+6.1f} deg   q7 = {np.degrees(Q[j, 6]):+6.1f} deg",
                      fontsize=10)
        ax2.cla()
        sg = B if on_b else A
        ax2.imshow(imgs[sg], origin="lower", extent=[0, 1, -1, 1], aspect="auto", cmap=CMAP,
                   vmin=0, vmax=2, interpolation="nearest")
        ax2.plot(UV[:min(j, k) + 1, 0], TH[:min(j, k) + 1], color="#1d6fd8", lw=1.5)
        if on_b:
            ax2.plot(UV[k:j + 1, 0], TH[k:j + 1], color="#d62828", lw=1.5)
        ax2.plot(UV[j, 0], TH[j], "o", color="k", ms=8)
        ax2.axvline(1, color="#d62828", lw=3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_xlabel("u (along the weld)")
        ax2.set_ylabel("theta / pi")
        name = "".join("+" if s > 0 else "-" for s in sg)
        ax2.set_title(f"{'page 2 (' + name + ') after the wrist flip' if on_b else 'page 1 (' + name + ')'}"
                      f", slice v = 0.5\nred: collision, teal: joint limits", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=len(idx))
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"wrote {out}")


def gif_exp2(path, out, frames=48):
    d = pickle.load(open(path, "rb"))
    (cfg, arm, scene, surf), _boxes = world(d)
    A = tuple(d["signs"])
    B = (A[0], A[1], -A[2])
    curves = door_curves(cfg, arm, surf, Page3Params())
    walls = {(w["page"], w["comp"]): w for w in d["task_walls"]}
    us, vs, ts = np.linspace(0, 1, 33), np.linspace(0, 1, 11), np.linspace(-np.pi, np.pi, 65)
    codes = {sg: classify_grid(cfg, arm, scene, surf, sg, us, vs, ts) for sg in (A, B)}
    t = d["out"]["task"]
    fig = plt.figure(figsize=(13, 6))
    axs = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    pages = ((A, "page 1 (+++): wall around the start", "#1d6fd8"),
             (B, "page 2 (++-): wall behind the q6 door", "#d62828"))

    def draw(i):
        for ax, (sg, title, col) in zip(axs, pages):
            ax.cla()
            draw_volume(ax, codes[sg], us, vs, ts)
            w = walls[("".join("+" if s > 0 else "-" for s in sg), 0)]
            vv = w["vertices"].copy()
            vv[:, 2] /= np.pi
            ax.add_collection3d(Poly3DCollection(vv[w["simplices"]], facecolor=col, alpha=0.35,
                                                 edgecolor="none"))
            doors(ax, curves)
            goal_face(ax)
            frame(ax, title)
            ax.view_init(elev=22, azim=-60 + 360.0 * i / frames)
        fig.suptitle(f"EXP 2 (3-D): {t['verdict']} -- both walls closed, every triangle verified "
                     f"in C_obs", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(out, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp3", default="out/exp3_two_page.pkl")
    ap.add_argument("--exp2", default="out/exp2_two_wall.pkl")
    args = ap.parse_args()
    gif_exp3(args.exp3, "out/exp3_path.gif")
    gif_exp2(args.exp2, "out/exp2_walls.gif")


if __name__ == "__main__":
    main()
