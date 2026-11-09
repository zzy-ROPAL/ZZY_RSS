#!/usr/bin/env python3
"""Two v = 0.5 slices of the new EXP 3 (findings 45): page 1 (start) and page 2.

white = valid, red = collision (box), teal = any joint limit / no IK.
Page 1 carries wall A's cross-section (the verified wall); both carry the
verified path projected to (u, theta).

    python3 scripts/fig_two_page_2d.py out/exp3_two_page.pkl --out out/fig_exp3_slices.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_scene3 as S  # noqa: E402
from fig_3d_scene import slice_segments  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors import _arm_angle_of_elbow  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--out", default="out/fig_exp3_slices.png")
    args = ap.parse_args()
    d = pickle.load(open(args.data, "rb"))
    x, z, tilt, roll = (float(c) for c in d["scene"].split(",")[:4])
    boxes = [[float(c) for c in b.split(",")] for b in d["boxes"]]
    cfg, arm, scene, surf = S.world(dict(x=x, z=z, tilt=tilt, roll=roll, boxes=boxes))
    A = tuple(d["signs"])
    B = (A[0], A[1], -A[2])
    curves = door_curves(cfg, arm, surf, Page3Params())

    Q = np.vstack([lg["q"] for lg in d["legs"]])
    UV = np.vstack([lg["y"] for lg in d["legs"]])
    TH = np.array([_arm_angle_of_elbow(arm, surf.line(float(v)), float(u), arm.frames(q)["elbow"])
                   for q, (u, v) in zip(Q, UV)]) / np.pi
    k = int(np.flatnonzero(np.diff(np.sign(Q[:, 5])) != 0)[0]) + 1

    us, ts = np.linspace(0, 1, 301), np.linspace(-np.pi, np.pi, 361)
    U, T = np.meshgrid(us, ts, indexing="ij")
    P = np.column_stack([U.ravel(), np.full(U.size, args.v), T.ravel()])
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#e76f51", "#2a9d8f"])

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, sg, title in ((axs[0], A, "page 1 (+++), start page"),
                          (axs[1], B, "page 2 (++-), after the wrist flip")):
        _ok, tag, _q = classify_surface_batch(cfg, arm, surf, scene, P, sg)
        code = np.where(tag == 0, 0, np.where(tag == 3, 1, 2)).reshape(U.shape)
        ax.imshow(code.T, origin="lower", extent=[0, 1, -1, 1], aspect="auto", cmap=cmap,
                  vmin=0, vmax=2, interpolation="nearest")
        ax.axvline(1.0, color="#d62828", lw=4)
        ax.text(0.985, 0.93, "goal", color="#d62828", ha="right", fontsize=11, weight="bold")
        for c in curves:
            j = int(np.argmin(np.abs(c.pts[:, 1] - args.v)))
            ax.plot(c.pts[j, 0], c.pts[j, 2] / np.pi, "*", color="#7b2cbf", ms=18, mec="k")
        ax.plot(UV[0, 0], TH[0], "o", color="#1d6fd8", ms=11, mec="k")
        if sg == A:
            if d.get("wall") is not None:
                for a, b in slice_segments(d["wall"]["vertices"], d["wall"]["simplices"], args.v):
                    ax.plot([a[0], b[0]], [a[2] / np.pi, b[2] / np.pi], color="#0077b6", lw=2.5)
            ax.plot(UV[:k + 1, 0], TH[:k + 1], color="#1d6fd8", lw=2)
            ax.plot([], [], color="#0077b6", lw=2.5, label="wall A (verified): start inside, goal outside")
            ax.plot([], [], color="#1d6fd8", lw=2, label="verified path start -> door (so no wall B exists)")
        else:
            ax.plot(UV[k:, 0], TH[k:], color="#d62828", lw=2)
            ax.plot([], [], color="#d62828", lw=2, label="verified path door -> goal (so no wall on page 2)")
        ax.plot([], [], "*", color="#7b2cbf", ms=14, mec="k", ls="", label="q6 door (wrist flip)")
        ax.plot([], [], "o", color="#1d6fd8", ms=9, mec="k", ls="", label="start")
        ax.plot([], [], color="#e76f51", lw=8, label="collision (box)")
        ax.plot([], [], color="#2a9d8f", lw=8, label="joint limits / no IK")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("u (along the weld)")
        ax.set_ylabel("theta / pi (arm angle)")
        ax.set_title(f"{title}, slice v = {args.v}", fontsize=12)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Paths are projected onto the slice (the real path also moves in v).", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
