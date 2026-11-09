#!/usr/bin/env python3
"""Where the samples of a 3-D proof come from (findings 48).

One page (default: 3-D EXP 2, page ++-), drawn twice -- in 3-D and as a
slab |v - v0| < half_width -- with every kind of sample the proof uses:

    roadmap   uniform random states (valid -> vertices, invalid -> kept for
              Algorithm 2), the exact start postures (start line u = 0), and,
              in the cross-page flood only, the states either side of each
              door curve
    training  goal-face samples, face anchors (labelled by connectivity),
              door-tube samples, and the outside ring -- these steer the
              SVM wall, they are NOT roadmap vertices

    PYTHONPATH=src python3 scripts/fig_sample_types.py --out docs/figs/sample_types.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import exp2_two_walls_3d as E  # noqa: E402
from zzy_rss.flood3 import door_samples  # noqa: E402
from zzy_rss.page3 import (Page3, door_curves, face_anchors, start_states,  # noqa: E402
                           tube_samples, virtual_ring3)

# (key, label, colour, marker, size, group)
KINDS = [
    ("invalid", "uniform sample, INVALID (kept for Alg 2)", "#c8c8c8", "o", 6, "roadmap"),
    ("valid", "uniform sample, valid -> roadmap vertex", "#343a40", "o", 12, "roadmap"),
    ("start", "START: exact start postures (u = 0)", "#1d6fd8", "o", 26, "roadmap"),
    ("door_x", "DOOR (flood only): states either side of the curve", "#7b2cbf", "D", 14, "roadmap"),
    ("goal", "GOAL: goal-face samples (u = 1 - tol)", "#2a9d8f", "s", 12, "training"),
    ("anchor_s", "face anchor -> start side", "#48cae4", "^", 18, "training"),
    ("anchor_r", "face anchor -> goal side", "#e76f51", "v", 18, "training"),
    ("tube", "DOOR: tube samples (keep the wall off the door)", "#ff8800", "*", 22, "training"),
    ("ring", "OUTSIDE: ring just outside the page box", "#000000", "x", 10, "training"),
]


def collect(signs, samples, seed):
    cfg, arm, scene, surf, p3 = E.setup(samples, seed, "enclose", True, True)
    starts = start_states(cfg, arm, surf, scene, signs, p3, 181)
    curves = door_curves(cfg, arm, surf, p3)
    ds = door_samples(curves, np.asarray(p3.scale), float(p3.cross_eps), int(p3.cert_stride))
    page = Page3(cfg, arm, surf, scene, signs, p3, None, inject=starts)
    vin = page.inject_index[page.inject_index >= 0]
    page.set_focus(int(page.comp[vin[0]]))
    rng = np.random.default_rng(p3.rng_seed + 3)
    region = page.seed_points()
    a_seed, a_rest = face_anchors(page, rng)
    goal = rng.uniform(page.lo, page.hi, size=(int(p3.goal_face_samples), 3))
    goal[:, 0] = 1.0 - p3.goal_tol
    t_seed, t_rest = tube_samples(curves, region, page.scale, 0.03, page.lo, page.hi,
                                  side="enclose")
    ring = virtual_ring3(cfg, page, int(p3.ring_per_face), rng)
    n_inj = len(starts)
    pts = dict(invalid=page.invalid, valid=page.valid[n_inj:], start=starts,
               door_x=ds.pts, goal=goal, anchor_s=a_seed, anchor_r=a_rest,
               tube=np.vstack([t_seed, t_rest]), ring=ring)
    return pts, curves, p3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="++-")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--v0", type=float, default=0.5)
    ap.add_argument("--half-width", type=float, default=0.06)
    ap.add_argument("--out", default="docs/figs/sample_types.png")
    args = ap.parse_args()
    signs = tuple(1 if c == "+" else -1 for c in args.page)
    pts, curves, p3 = collect(signs, args.samples, args.seed)
    for k, v in pts.items():
        print(f"[samples] {k:9s} {len(v)}")

    fig = plt.figure(figsize=(21, 9))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    for key, lab, col, mk, sz, _grp in KINDS:
        P = np.asarray(pts[key]).reshape(-1, 3)
        if not len(P):
            continue
        Q = P[np.random.default_rng(0).choice(len(P), min(len(P), 2500), replace=False)] \
            if key in ("invalid", "valid") else P
        ax3.scatter(Q[:, 0], Q[:, 1], Q[:, 2] / np.pi, c=col, marker=mk, s=sz * 0.6,
                    alpha=0.35 if key in ("invalid", "valid") else 0.9, depthshade=False,
                    linewidths=1.0 if mk == "x" else 0)
        m = np.abs(P[:, 1] - args.v0) < args.half_width
        if key == "start":
            m = np.ones(len(P), dtype=bool)          # the start line is at v0 already
        ax2.scatter(P[m, 0], P[m, 2] / np.pi, c=col, marker=mk, s=sz,
                    alpha=(0.55 if key == "invalid" else 0.9) if key in ("invalid", "valid") else 0.95,
                    linewidths=1.0 if mk == "x" else 0, zorder=3 if key != "invalid" else 1)
    for c in curves:
        ax3.plot(c.pts[:, 0], c.pts[:, 1], c.pts[:, 2] / np.pi, color="#ff8800", lw=2.5)
        near = np.abs(c.pts[:, 1] - args.v0) < args.half_width
        if near.any():
            ax2.plot(c.pts[near, 0], c.pts[near, 2] / np.pi, color="#ff8800", lw=3, zorder=4)
    ax3.set_xlabel("u (along weld)")
    ax3.set_ylabel("v (across)")
    ax3.set_zlabel(r"$\theta / \pi$")
    ax3.view_init(elev=22, azim=-58)
    ax3.set_box_aspect((1.4, 1.0, 1.1))
    ax3.set_title(f"page {args.page}: every sample in (u, v, theta)", fontsize=12)
    for x in (0.0, 1.0):
        ax2.axvline(x, color="#999999", lw=0.8, ls=":")
    ax2.axvline(1 - p3.goal_tol, color="#2a9d8f", lw=1.2, ls="--")
    ax2.text(0.003, 1.08, "start line u = 0", color="#1d6fd8", fontsize=10)
    ax2.text(0.995, 1.08, "goal u = 1", color="#2a9d8f", fontsize=10, ha="right")
    ax2.set_xlim(-0.16, 1.16)
    ax2.set_ylim(-1.18, 1.18)
    ax2.set_xlabel("u (along the weld)")
    ax2.set_ylabel(r"$\theta / \pi$ (arm angle)")
    ax2.set_title(f"slab |v - {args.v0}| < {args.half_width} (orange line = q2 door curve)",
                  fontsize=12)
    handles = []
    for grp, title in (("roadmap", "ROADMAP (vertices / Alg 2 obstacle samples)"),
                       ("training", "WALL TRAINING SET ONLY")):
        handles.append(Line2D([], [], ls="", label=title))
        for key, lab, col, mk, sz, g in KINDS:
            if g == grp:
                handles.append(Line2D([], [], ls="", marker=mk, color=col, mfc=col, mec=col,
                                      ms=9,
                                      label=f"{lab}  ({len(pts[key])})"))
    leg = fig.legend(handles=handles, loc="center right", fontsize=10, frameon=True)
    for t in leg.get_texts():
        if t.get_text().startswith(("ROADMAP", "WALL")):
            t.set_fontweight("bold")
    fig.suptitle("Where the samples come from: uniform free-space sampling + exact points at "
                 "the start, the goal, the doors and the page boundary", fontsize=14)
    fig.subplots_adjust(left=0.0, right=0.70, top=0.9, bottom=0.07, wspace=0.08)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
