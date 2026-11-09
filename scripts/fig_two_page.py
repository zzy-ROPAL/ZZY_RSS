#!/usr/bin/env python3
"""Figure for the two-page EXP 3 (findings 45), from run_3d.py --two-page --save.

  (a) WORKSPACE: the weld patch, the box, the q6 door curves (where the wrist
      straightens) and the arm at three postures of the verified path: the
      start (page A), the wrist flip, the end (page B).
  (b) PAGE A (u, v, theta): its start region and the verified wall -- the
      first-page proof that the goal cannot be reached without a page change.
  (c) The verified path drawn over page A's v-slice: the part after the
      crossing runs straight through A's q7 limit band, which is legal only
      because the wrist has flipped.
  (d) q6 and q7 along the path, with q7 as page A would spell the same
      postures: it would leave the +/-168.45 deg limit.

    python3 scripts/fig_two_page.py out/exp3_two_page.pkl --out out/fig_exp3_two_page.png
"""

from __future__ import annotations

import argparse
import dataclasses
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
import visualize as V  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors import _arm_angle_of_elbow  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import Page3  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402

LIM7 = 168.45


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", default="out/fig_exp3_two_page.png")
    args = ap.parse_args()
    d = pickle.load(open(args.data, "rb"))
    x, z, tilt, roll = (float(c) for c in d["scene"].split(",")[:4])
    boxes = [[float(c) for c in b.split(",")] for b in d["boxes"]]
    spec = dict(x=x, z=z, tilt=tilt, roll=roll, boxes=boxes)
    cfg, arm, scene, surf = S.world(spec)
    A = tuple(d["signs"])
    B = (A[0], A[1], -A[2])
    p3 = dataclasses.replace(Page3Params(), n_samples=d["samples"], rng_seed=d["seed"])
    curves = door_curves(cfg, arm, surf, p3)
    out = d["out"]

    legs = d["legs"]
    Q = np.vstack([lg["q"] for lg in legs])
    UV = np.vstack([lg["y"] for lg in legs])
    TH = np.array([_arm_angle_of_elbow(arm, surf.line(float(v)), float(u), arm.frames(q)["elbow"])
                   for q, (u, v) in zip(Q, UV)], dtype=float)
    k_cross = int(np.flatnonzero(np.diff(np.sign(Q[:, 5])) != 0)[0]) + 1
    on_b = np.arange(len(Q)) >= k_cross

    fig = plt.figure(figsize=(19, 5.6))

    # ---- (a) workspace --------------------------------------------------
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    corners = np.array([surf.position(0, 0), surf.position(1, 0), surf.position(1, 1),
                        surf.position(0, 1)])
    ax.add_collection3d(Poly3DCollection([corners], facecolor="#e9c46a", alpha=0.35,
                                         edgecolor="#b08a20"))
    ax.plot(*np.array([surf.position(0, 0.5), surf.position(1, 0.5)]).T, color="#e76f51",
            lw=2, label="weld (v = 0.5)")
    for c in curves:
        w = np.array([surf.position(u, v) for u, v, _ in c.pts])
        ax.plot(*w.T, color="#7b2cbf", lw=1.8, ls="--")
    ax.plot([], [], color="#7b2cbf", ls="--", label="q6 door curves (wrist straight)")
    for b in boxes:
        V.draw_box(ax, b[:3], b[3:6], color="#d62828", alpha=0.25)
    for idx, col, lab in ((0, "#1d6fd8", "start (page A)"), (k_cross, "#7b2cbf", "wrist flip"),
                          (len(Q) - 1, "#d62828", "end (page B)")):
        f = arm.frames(Q[idx])
        P = np.array([f["shoulder"], f["elbow"], f["wrist"], f["tool"]])
        ax.plot(*P.T, "-o", color=col, lw=3, ms=4, label=lab)
    tool = np.array([arm.frames(q)["tool"] for q in Q[::max(1, len(Q) // 400)]])
    ax.plot(*tool.T, color="k", lw=1)
    ax.plot([0, 0], [0, 0], [0, 0.24], color="#888888", lw=8)
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(0.0, 0.7)
    ax.set_box_aspect((0.8, 0.8, 0.7))
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=7)
    ax.set_title(f"(a) weld patch x {x}, z {z}, torch {tilt:.0f} deg, roll {roll:.0f} deg\n"
                 f"one box {boxes[0][3]*2:.2f} m at ({boxes[0][0]}, {boxes[0][1]}, "
                 f"{boxes[0][2]})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # ---- (b) page A and its wall ------------------------------------------
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    page = Page3(cfg, arm, surf, scene, A, p3, d["start"][2])
    sc = page.comp == page.start_comp
    ax.scatter(page.valid[~sc, 0], page.valid[~sc, 1], page.valid[~sc, 2] / np.pi, s=1,
               color="#bbbbbb", alpha=0.4)
    ax.scatter(page.valid[sc, 0], page.valid[sc, 1], page.valid[sc, 2] / np.pi, s=2,
               color="#1d6fd8", alpha=0.7)
    if d["wall"] is not None:
        vv = d["wall"]["vertices"].copy()
        vv[:, 2] /= np.pi
        ax.add_collection3d(Poly3DCollection(vv[d["wall"]["simplices"]], facecolor="#48cae4",
                                             alpha=0.12, edgecolor="#0077b6", linewidths=0.08))
    for c in curves:
        ax.plot(c.pts[:, 0], c.pts[:, 1], c.pts[:, 2] / np.pi, color="#7b2cbf", lw=1.8, ls="--")
    gu = 1.0 - p3.goal_tol
    ax.plot([gu] * 5, [0, 1, 1, 0, 0], [-1, -1, 1, 1, -1], color="#d62828", lw=1.5)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("theta / pi", labelpad=-2)
    ax.view_init(elev=20, azim=-60)
    ax.tick_params(labelsize=7)
    pa = out["page_alone"]
    w = pa["walls"][0] if pa["walls"] else {}
    ax.set_title(f"(b) page A {''.join('+' if s > 0 else '-' for s in A)} alone: "
                 f"{pa['verdict']}\nwall {w.get('facets', '?')} triangles in C_obs, "
                 f"gamma {w.get('gamma', '?')}, u_max {w.get('u_max', '?')}", fontsize=10)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", color="#1d6fd8", label="start region (page A)"),
        Line2D([], [], color="#48cae4", lw=6, alpha=0.5, label="verified wall"),
        Line2D([], [], color="#7b2cbf", ls="--", label="q6 door curves"),
        Line2D([], [], color="#d62828", label="goal face u = 1 - tol")], fontsize=7,
        loc="upper left")

    # ---- (c) the path over page A's slice -----------------------------------
    ax = fig.add_subplot(1, 4, 3)
    nu, nt = 201, 241
    us, ts = np.linspace(0, 1, nu), np.linspace(-np.pi, np.pi, nt)
    U, T = np.meshgrid(us, ts, indexing="ij")
    P = np.column_stack([U.ravel(), np.full(U.size, 0.5), T.ravel()])
    ok, tag, q = classify_surface_batch(cfg, arm, surf, scene, P, A)
    bad = (q < arm.q_min) | (q > arm.q_max)
    code = np.zeros(len(P), int)
    code[tag == 3] = 1
    code[(tag == 2) & bad[:, 6]] = 2
    code[(tag == 2) & ~bad[:, 6]] = 3
    code[tag == 1] = 3
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#f4a582", "#66c2a5", "#fee08b"])
    ax.imshow(code.reshape(nu, nt).T, origin="lower", extent=[0, 1, -1, 1], aspect="auto",
              cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.plot(UV[~on_b, 0], TH[~on_b] / np.pi, color="#1d6fd8", lw=2.2, label="path on page A")
    ax.plot(UV[on_b, 0], TH[on_b] / np.pi, color="#d62828", lw=2.2,
            label="path on page B (wrist flipped)")
    ax.plot(UV[k_cross, 0], TH[k_cross] / np.pi, "*", color="#7b2cbf", ms=16, mec="k",
            label="crossing the q6 door")
    for c in curves:
        k = int(np.argmin(np.abs(c.pts[:, 1] - 0.5)))
        ax.plot(c.pts[k, 0], c.pts[k, 2] / np.pi, "k*", ms=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("u (along the weld)")
    ax.set_ylabel("theta / pi")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color="#f4a582", lw=8, label="collision (box)"),
        Line2D([], [], color="#66c2a5", lw=8, label="page A: q7 beyond its limit"),
        Line2D([], [], color="#fee08b", lw=8, label="other limits / no IK")], fontsize=7,
        loc="lower right")
    ax.set_title("(c) the verified path over page A (slice v = 0.5)\n"
                 "after the flip it runs through A's q7 band", fontsize=10)

    # ---- (d) joints along the path ---------------------------------------
    ax = fig.add_subplot(1, 4, 4)
    s = np.arange(len(Q))
    q7 = np.degrees(Q[:, 6])
    q7_as_a = np.degrees(np.angle(np.exp(1j * (Q[:, 6] + np.pi))))
    ax.plot(s, np.degrees(Q[:, 5]), color="#7b2cbf", lw=1.6, label="q6 (changes sign = flip)")
    ax.plot(s, q7, color="#d62828", lw=1.8, label="q7 as executed")
    ax.plot(s[on_b], q7_as_a[on_b], color="#1d6fd8", lw=1.4, ls="--",
            label="q7 of the same postures, unflipped wrist")
    ax.axhspan(LIM7, 200, color="#66c2a5", alpha=0.3)
    ax.axhspan(-200, -LIM7, color="#66c2a5", alpha=0.3)
    ax.axvline(k_cross, color="#7b2cbf", ls=":")
    ax.set_ylim(-195, 195)
    ax.set_xlabel("configuration index along the verified path")
    ax.set_ylabel("deg")
    ax.legend(fontsize=7, loc="lower left")
    t = out["task"]
    wit = t.get("witness") or {}
    ax.set_title(f"(d) task: {t['verdict']} -- {wit.get('n_configurations', '?')} "
                 f"configurations verified\nshaded: q7 beyond +/-{LIM7} deg", fontsize=10)
    ax.grid(alpha=0.25)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10, wspace=0.30)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
