#!/usr/bin/env python3
"""All eight pages of a weld-patch task as 3-D maps of (u, v, theta) (findings 45).

Each page is one sign branch (s2, s4, s6). The volume of the page is coloured
by WHY a state is invalid:

    red     collision (the box)   -- the same on every page (invariant 49)
    teal    q7 beyond its limit   -- moves by pi when s6 flips (the wrist)
    yellow  q2/q4/q6 limits, or no IK
    purple  q1/q3/q5 limit slivers only (faint)

Surfaces are marching-cubes boundaries of a grid classification: a picture,
never a proof (invariant 8). Hair-thin q1/q3/q5 limit slivers (2.3 deg) are
below the grid's resolution: they are drawn faintly, as broken fragments. Dashed purple: the q6 door
curves (wrist straight). With --data (run_3d.py --two-page --save) a second
figure shows the start page with its verified wall and the verified path
across both pages in (u, v, theta).

    python3 scripts/fig_pages3d.py --scene 0.40,0.21,45,305 \\
        --box 0.2,0.0,0.34,0.06,0.06,0.06 --data out/exp3_two_page.pkl \\
        --out out/fig_pages3d.png
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
from skimage.measure import marching_cubes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_scene3 as S  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors import _arm_angle_of_elbow  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402

CATS = ((2, "#e76f51", 0.30, "collision (box)"),
        (3, "#2a9d8f", 0.45, "q7 beyond its limit"),
        (4, "#e9c46a", 0.16, "q2/q4/q6 limits, no IK"),
        (5, "#9d4edd", 0.10, "q1/q3/q5 slivers (2.3 deg, under-resolved)"))


def page_name(sg) -> str:
    return "".join("+" if s > 0 else "-" for s in sg)


def classify_grid(cfg, arm, scene, surf, signs, us, vs, ts):
    U, V, T = np.meshgrid(us, vs, ts, indexing="ij")
    P = np.column_stack([U.ravel(), V.ravel(), T.ravel()])
    ok, tag, q = classify_surface_batch(cfg, arm, surf, scene, P, signs)
    bad = (q < arm.q_min) | (q > arm.q_max)
    code = np.zeros(len(P), dtype=int)
    code[tag == 3] = 2                                    # collision
    code[(tag == 2) & bad[:, 6]] = 3                      # q7 limit
    thin = bad[:, 0] | bad[:, 2] | bad[:, 4]
    thick = bad[:, 1] | bad[:, 3] | bad[:, 5]
    code[(tag == 2) & ~bad[:, 6] & thick] = 4             # q2/q4/q6 limits
    code[(tag == 2) & ~bad[:, 6] & ~thick & thin] = 5     # hair-thin slivers only
    code[tag == 1] = 4                                    # no IK
    return code.reshape(U.shape)


def draw_volume(ax, code, us, vs, ts):
    sp = (us[1] - us[0], vs[1] - vs[0], (ts[1] - ts[0]) / np.pi)
    for c, col, alpha, _lab in CATS:
        m = np.pad((code == c).astype(float), 1)
        if m.sum() == 0:
            continue
        verts, faces, _n, _v = marching_cubes(m, 0.5, spacing=sp, step_size=1)
        verts -= np.array(sp)                             # undo the padding
        verts += np.array([us[0], vs[0], ts[0] / np.pi])
        verts = np.clip(verts, [us[0], vs[0], -1.0], [us[-1], vs[-1], 1.0])
        ax.add_collection3d(Poly3DCollection(verts[faces], facecolor=col, alpha=alpha,
                                             edgecolor="none"))


def frame(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("u (along weld)", fontsize=8, labelpad=-6)
    ax.set_ylabel("v (across)", fontsize=8, labelpad=-6)
    ax.set_zlabel("theta / pi", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-3)
    ax.view_init(elev=24, azim=-52)
    ax.set_box_aspect((1.5, 1.0, 1.2))
    ax.set_title(title, fontsize=9)


def doors(ax, curves):
    for c in curves:
        ax.plot(c.pts[:, 0], c.pts[:, 1], c.pts[:, 2] / np.pi, color="#7b2cbf", lw=2.0,
                ls="--")


def goal_face(ax):
    ax.plot([1, 1, 1, 1, 1], [0, 1, 1, 0, 0], [-1, -1, 1, 1, -1], color="#d62828", lw=1.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="0.40,0.21,45,305")
    ap.add_argument("--box", action="append", default=None)
    ap.add_argument("--data", default=None, help="pickle from run_3d.py --two-page --save")
    ap.add_argument("--start-page", default="+++")
    ap.add_argument("--res", default="41,13,81", help="grid u,v,theta")
    ap.add_argument("--out", default="out/fig_pages3d.png")
    args = ap.parse_args()
    boxes = [[float(c) for c in b.split(",")] for b in (args.box or ["0.2,0.0,0.34,0.06,0.06,0.06"])]
    x, z, tilt, roll = (float(c) for c in args.scene.split(",")[:4])
    spec = dict(x=x, z=z, tilt=tilt, roll=roll, boxes=boxes)
    cfg, arm, scene, surf = S.world(spec)
    p3 = Page3Params()
    curves = door_curves(cfg, arm, surf, p3)
    nu, nv, nt = (int(c) for c in args.res.split(","))
    us, vs, ts = np.linspace(0, 1, nu), np.linspace(0, 1, nv), np.linspace(-np.pi, np.pi, nt)
    start_signs = tuple(1 if ch == "+" else -1 for ch in args.start_page)
    th0 = choose_start_theta(cfg, arm, surf, scene, start_signs, p3)

    codes = {sg: classify_grid(cfg, arm, scene, surf, sg, us, vs, ts) for sg in SIGN_BRANCHES}

    # ---- figure 1: the eight pages ------------------------------------
    fig = plt.figure(figsize=(22, 10.5))
    for k, sg in enumerate(SIGN_BRANCHES):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        code = codes[sg]
        draw_volume(ax, code, us, vs, ts)
        doors(ax, curves)
        goal_face(ax)
        name = page_name(sg)
        if sg == start_signs and th0 is not None:
            ax.scatter([0.0], [0.5], [th0 / np.pi], s=70, color="#1d6fd8", edgecolor="k",
                       depthshade=False)
        valid = float((code == 0).mean())
        tag = ""
        if sg == start_signs:
            tag = "  START PAGE (A)"
        elif sg == (start_signs[0], start_signs[1], -start_signs[2]):
            tag = "  wrist-flipped (B)"
        frame(ax, f"page {name}  (s2 s4 s6){tag}\nvalid {100 * valid:.0f}% of the box")
    fig.legend(handles=[Line2D([], [], color=c, lw=8, alpha=0.6, label=lab)
                        for _c, c, _a, lab in CATS]
               + [Line2D([], [], color="#7b2cbf", ls="--", lw=2, label="q6 door curves"),
                  Line2D([], [], ls="", marker="o", ms=8, color="#1d6fd8", mec="k",
                         label="start posture"),
                  Line2D([], [], color="#d62828", label="goal face u = 1")],
               loc="lower center", ncol=7, fontsize=10)
    fig.suptitle(f"All eight pages (u, v, theta) -- weld patch x {x}, z {z}, torch {tilt:.0f} deg, "
                 f"roll {roll:.0f} deg, box at ({boxes[0][0]}, {boxes[0][1]}, {boxes[0][2]}); "
                 f"white = valid. Collision is identical on all pages; only the limits move.",
                 fontsize=12)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.07, wspace=0.0, hspace=0.12)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # ---- figure 2: pages A and B with the wall and the verified path ---
    if not args.data:
        return
    d = pickle.load(open(args.data, "rb"))
    B = (start_signs[0], start_signs[1], -start_signs[2])
    legs = d["legs"]
    Q = np.vstack([lg["q"] for lg in legs])
    UV = np.vstack([lg["y"] for lg in legs])
    TH = np.array([_arm_angle_of_elbow(arm, surf.line(float(v)), float(u),
                                       arm.frames(q)["elbow"]) for q, (u, v) in zip(Q, UV)])
    k = int(np.flatnonzero(np.diff(np.sign(Q[:, 5])) != 0)[0]) + 1
    fig = plt.figure(figsize=(18, 8))
    for i, (sg, what) in enumerate(((start_signs, "wall"), (B, "path"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        draw_volume(ax, codes[sg], us, vs, ts)
        doors(ax, curves)
        goal_face(ax)
        if what == "wall" and d.get("wall") is not None:
            vv = d["wall"]["vertices"].copy()
            vv[:, 2] /= np.pi
            ax.add_collection3d(Poly3DCollection(vv[d["wall"]["simplices"]], facecolor="#48cae4",
                                                 alpha=0.22, edgecolor="#0077b6",
                                                 linewidths=0.05))
        ax.plot(UV[:k, 0], UV[:k, 1], TH[:k] / np.pi, color="#1d6fd8", lw=2.2)
        ax.plot(UV[k:, 0], UV[k:, 1], TH[k:] / np.pi, color="#d62828", lw=2.2)
        ax.scatter([UV[k, 0]], [UV[k, 1]], [TH[k] / np.pi], s=180, marker="*",
                   color="#7b2cbf", edgecolor="k", depthshade=False)
        ax.scatter([UV[0, 0]], [UV[0, 1]], [TH[0] / np.pi], s=70, color="#1d6fd8",
                   edgecolor="k", depthshade=False)
        pa = d["out"]["page_alone"]
        w = pa["walls"][0] if pa["walls"] else {}
        if what == "wall":
            title = (f"page {page_name(sg)} (A, start): {pa['verdict']} on its own -- verified wall "
                     f"{w.get('facets')} triangles in C_obs (cyan)\nthe q7 band (teal) and the box "
                     f"(red) close the start region; the live q6 door is inside the wall")
        else:
            t = d["out"]["task"]
            title = (f"page {page_name(sg)} (B, wrist flipped): the q7 band has moved away\n"
                     f"verified path {t['witness']['n_configurations']} configurations: blue on A, "
                     f"star = q6 crossing, red on B -> goal ({t['verdict']})")
        frame(ax, title)
        ax.set_title(title, fontsize=10)
    fig.legend(handles=[Line2D([], [], color=c, lw=8, alpha=0.6, label=lab)
                        for _c, c, _a, lab in CATS]
               + [Line2D([], [], color="#48cae4", lw=8, alpha=0.5, label="verified wall (page A)"),
                  Line2D([], [], color="#1d6fd8", lw=2, label="path on page A"),
                  Line2D([], [], color="#d62828", lw=2, label="path on page B"),
                  Line2D([], [], color="#7b2cbf", ls="--", lw=2, label="q6 door curves")],
               loc="lower center", ncol=8, fontsize=10)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.08, wspace=0.0)
    out2 = out.with_name(out.stem + "_AB" + out.suffix)
    fig.savefig(out2, dpi=120)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
