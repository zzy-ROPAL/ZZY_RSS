"""Schematic of every door/wall configuration the four exits have to handle.

Six panels, each a page of Y+ drawn flat. These are SCHEMATIC -- the shapes are
hand-drawn blobs, not computed walls -- because the point is the topology, not
any particular scene. Real walls are produced by run_three.py / run_proof.

What each panel is for is written in docs/findings.md (section 37, appendix 2).

    python3 scripts/fig_door_cases.py --out out/door_cases.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

# Validated categorical slots (dataviz reference palette, light surface).
SURFACE = "#fcfcfb"
W1 = "#2a78d6"      # wall 1 -- excludes the GOAL
W2 = "#eb6834"      # wall 2 -- excludes ONE DOOR
DOOR = "#1baf7a"    # doors / door patches
GOAL = "#eda100"    # goal set
OTHER = "#6250d6"   # partner page / second component
INK = "#1a1a19"
MUTED = "#6b6b68"
OBS = "#d8d8d4"     # obstacle region of the page


def blob(cx, cy, rx, ry, wobble=0.0, phase=0.0, n=240):
    """A closed curve. Walls are closed by Definition 1 (I), so these are too."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    r = 1.0 + wobble * np.sin(3.0 * t + phase)
    return cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)


def draw_wall(ax, xy, color, label=None, ls="-", lw=2.2, z=3, band=True,
              at=0.13, band_lw=13.0):
    """A closed curve, drawn inside the obstacle band it is required to occupy.

    Definition 1 asks for (I) closed AND (II) entirely in C_obs. Drawing the
    wall as a bare line beside a separate obstacle block would show only (I);
    the grey halo is (II), and it is the whole reason the curve blocks
    anything -- a closed curve through FREE space separates nothing.
    """
    x, y = xy
    if band:
        ax.plot(x, y, color=OBS, lw=band_lw, zorder=z - 1,
                solid_capstyle="round")
    ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
    if label:
        k = int(at * len(x))
        ax.annotate(label, (x[k], y[k]), color=color, fontsize=8.5,
                    fontweight="bold", ha="center", va="center", zorder=z + 1,
                    bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE,
                              ec=color, lw=1.0))


def mark(ax, x, y, color, text, dx=0.0, dy=0.10, size=78, marker="o"):
    ax.scatter([x], [y], s=size, c=color, marker=marker, zorder=6,
               edgecolors=SURFACE, linewidths=1.6)
    ax.annotate(text, (x + dx, y + dy), color=INK, fontsize=8.5, ha="center",
                va="center", zorder=7)


def obstacle(ax, x, y, w, h, label="C_obs"):
    ax.add_patch(Rectangle((x, y), w, h, fc=OBS, ec="#b9b9b4", lw=1.0,
                           zorder=1))
    if label:
        ax.annotate(label, (x + w / 2, y + h / 2), color=MUTED, fontsize=7.5,
                    ha="center", va="center", style="italic", zorder=2)


def patches_along(ax, xs, ys, r=0.055, live=None):
    """A door that is a LINE, covered by a chain of discs (find_fold_doors)."""
    live = [True] * len(xs) if live is None else live
    for x, y, on in zip(xs, ys, live):
        ax.add_patch(Circle((x, y), r, fc=DOOR if on else "#ffffff",
                            ec=DOOR, lw=1.6, alpha=0.95 if on else 1.0,
                            zorder=5))


def frame(ax, title, verdict, vcolor):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color("#d4d4d0")
    ax.set_title(title, fontsize=9.5, color=INK, pad=7, loc="left",
                 fontweight="bold")
    ax.annotate(verdict, (0.5, -0.075), xycoords="axes fraction", fontsize=8.5,
                ha="center", va="top", color=vcolor, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/door_cases.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.4))
    fig.patch.set_facecolor(SURFACE)

    # ---------------------------------------------------------------- (a)
    # Exit 1: wall1 was trained to exclude the GOAL, and happens to leave the
    # door outside too. Definition 1 re-read with X = the door, for free.
    ax = axes[0][0]
    draw_wall(ax, blob(0.38, 0.52, 0.25, 0.27, 0.07, 0.6), W1, "wall 1", at=0.60)
    ax.annotate("the wall lies in C_obs\n(grey band)", (0.17, 0.12),
                color=MUTED, fontsize=7.6, ha="center", style="italic")
    mark(ax, 0.36, 0.52, INK, "start", dy=0.095)
    mark(ax, 0.86, 0.72, GOAL, "goal", dy=0.095)
    ax.add_patch(Circle((0.82, 0.30), 0.052, fc=DOOR, ec=DOOR, lw=1.6, zorder=5))
    ax.annotate("door", (0.82, 0.18), color=INK, fontsize=8.5, ha="center")
    frame(ax, "(a)  Exit 1 — one wall does both jobs",
          "door OUTSIDE wall 1  →  proven unreachable  →  skip   (free)", W1)

    # ---------------------------------------------------------------- (b)
    # Exit 2: wall1 could not exclude the door, so a second wall is trained
    # for that one job. NESTED -- and nesting is the normal case, not a defect.
    ax = axes[0][1]
    draw_wall(ax, blob(0.44, 0.52, 0.31, 0.30, 0.05, 1.3), W1, "wall 1", at=0.02)
    draw_wall(ax, blob(0.30, 0.50, 0.14, 0.15, 0.04, 2.1), W2, "wall 2",
              at=0.55, band_lw=9.0)
    mark(ax, 0.29, 0.50, INK, "start", dy=0.090)
    mark(ax, 0.90, 0.80, GOAL, "goal", dy=0.090)
    ax.add_patch(Circle((0.58, 0.44), 0.052, fc=DOOR, ec=DOOR, lw=1.6, zorder=5))
    ax.annotate("door", (0.58, 0.33), color=INK, fontsize=8.5, ha="center")
    ax.annotate("wall 2 is nested INSIDE wall 1:\nthey share interior, and that is fine",
                (0.50, 0.10), color=MUTED, fontsize=7.6, ha="center",
                style="italic")
    frame(ax, "(b)  Exit 2 — a second wall, nested inside the first",
          "two independent statements; overlap is NOT a defect", W2)

    # ---------------------------------------------------------------- (c)
    # An EXTENDED door (a q4 fold line) tiled into patches. Some outside the
    # wall, some inside. The door is skipped only if EVERY patch is excluded.
    ax = axes[0][2]
    draw_wall(ax, blob(0.36, 0.52, 0.25, 0.28, 0.05, 0.2), W1, "wall 1", at=0.55)
    ys = np.linspace(0.12, 0.90, 9)
    xs = np.full_like(ys, 0.50)
    patches_along(ax, xs, ys, r=0.034)
    mark(ax, 0.30, 0.52, INK, "start", dy=0.090)
    mark(ax, 0.88, 0.84, GOAL, "goal", dy=0.090)
    ax.annotate("ONE door,\ntiled into patches", (0.76, 0.52), color=DOOR,
                fontsize=8.2, ha="center", fontweight="bold")
    ax.annotate("3 patches inside the wall, 6 outside\n→ not every patch is excluded",
                (0.28, 0.10), color=MUTED, fontsize=7.4, ha="center",
                style="italic")
    frame(ax, "(c)  Extended door — union semantics",
          "skip ONLY if every patch is excluded; here → FOLLOW", DOOR)

    # ---------------------------------------------------------------- (d)
    # Why wall2 must probe the whole ring: it drops the target from the Sigma
    # guard, so it MAY cut the disc. Centre outside, part of the ring inside.
    ax = axes[1][0]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.add_patch(Circle((0.58, 0.50), 0.20, fc="none", ec=DOOR, lw=2.0,
                        ls="--", zorder=4))
    ax.scatter(0.58 + 0.20 * np.cos(t[::12]), 0.50 + 0.20 * np.sin(t[::12]),
               s=34, c=DOOR, zorder=5, edgecolors=SURFACE, linewidths=1.0)
    ax.plot([0.30, 0.52, 0.66, 0.86], [0.14, 0.42, 0.60, 0.90], color=W2,
            lw=2.4, zorder=6, solid_capstyle="round")
    ax.annotate("wall 2", (0.17, 0.33), color=W2, fontsize=8.5,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.18",
                                             fc=SURFACE, ec=W2, lw=1.0))
    ax.scatter([0.58], [0.50], s=140, c=INK, marker="X", zorder=7,
               edgecolors=SURFACE, linewidths=1.8)
    ax.annotate("centre reads\nOUTSIDE", (0.58, 0.50), xytext=(0.86, 0.26),
                color=INK, fontsize=8.0, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="#d4d4d0"))
    ax.annotate("but this arc of the ring\nis INSIDE wall 2", (0.30, 0.76),
                color="#d03b3b", fontsize=8.0, ha="center", fontweight="bold")
    ax.annotate("door disc — the RING is the rest class", (0.52, 0.08),
                color=DOOR, fontsize=7.6, ha="center", style="italic")
    frame(ax, "(d)  Why wall 2 probes the WHOLE ring",
          "centre-only test → skip a reachable door → FALSE INFEASIBLE", "#d03b3b")

    # ---------------------------------------------------------------- (e)
    # The Sigma guard is Algorithm 3's first check, so a VERIFIED wall never
    # touches a LIVE disc -- which is what makes the centre test exact there.
    ax = axes[1][1]
    draw_wall(ax, blob(0.50, 0.50, 0.29, 0.26, 0.16, 2.6), W1, "wall 1",
              at=0.78)
    ax.add_patch(Circle((0.50, 0.80), 0.075, fc="none", ec=DOOR, lw=2.0,
                        zorder=5))
    ax.annotate("live disc", (0.50, 0.93), color=DOOR, fontsize=8.0,
                ha="center", fontweight="bold")
    ax.annotate("wall 1 DETOURS around it\n(Σ guard — Alg 3's FIRST check)", (0.50, 0.60),
                color=W1, fontsize=7.8, ha="center", fontweight="bold")
    ax.add_patch(Circle((0.22, 0.22), 0.075, fc="none", ec=DOOR, lw=2.0,
                        ls="--", zorder=5))
    ax.plot([0.10, 0.34], [0.14, 0.31], color=W2, lw=2.4, zorder=6,
            solid_capstyle="round")
    ax.annotate("wall 2 may CUT its own target\n(guard dropped for d)",
                (0.52, 0.17), color=W2, fontsize=7.8, ha="center",
                fontweight="bold")
    frame(ax, "(e)  The Σ guard is what makes a point test exact",
          "wall 1: centre suffices   ·   wall 2: must probe the ring", MUTED)

    # ---------------------------------------------------------------- (f)
    # A door bridging two components of the SAME page, around an obstacle,
    # by way of the partner page. The flood fill is built for exactly this.
    ax = axes[1][2]
    ax.annotate("page A", (0.06, 0.95), color=INK, fontsize=8.5,
                fontweight="bold")
    obstacle(ax, 0.42, 0.56, 0.17, 0.40, "C_obs")
    draw_wall(ax, blob(0.24, 0.76, 0.14, 0.16, 0.05, 0.0), W1, None, lw=1.8)
    ax.annotate("C1", (0.24, 0.76), color=MUTED, fontsize=8.5, ha="center")
    ax.annotate("C2", (0.78, 0.76), color=MUTED, fontsize=8.5, ha="center")
    draw_wall(ax, blob(0.78, 0.76, 0.14, 0.16, 0.05, 1.0), OTHER, None, lw=1.8)
    patches_along(ax, [0.37, 0.50, 0.63], [0.50, 0.50, 0.50], r=0.045)
    ax.annotate("door (extended)", (0.50, 0.40), color=DOOR, fontsize=8.0,
                ha="center", fontweight="bold")
    ax.annotate("page B", (0.06, 0.26), color=INK, fontsize=8.5,
                fontweight="bold")
    draw_wall(ax, blob(0.50, 0.17, 0.30, 0.10, 0.03, 0.0), OTHER, None, lw=1.8)
    ax.annotate("one region, connects both", (0.50, 0.17), color=MUTED,
                fontsize=7.8, ha="center", style="italic")
    for x0, x1 in ((0.33, 0.40), (0.67, 0.60)):
        ax.add_patch(FancyArrowPatch((x0, 0.58), (x1, 0.28), color=DOOR,
                                     lw=1.8, arrowstyle="-|>", mutation_scale=12,
                                     connectionstyle="arc3,rad=0.25", zorder=7))
    mark(ax, 0.20, 0.80, INK, "start", dy=0.085)
    mark(ax, 0.84, 0.80, GOAL, "goal", dy=0.085)
    frame(ax, "(f)  A door bridging two components, via the partner page",
          "C1 → door → page B → door → C2 : the flood fill handles it", OTHER)

    fig.suptitle("Door / wall configurations the four exits must handle",
                 fontsize=13, color=INK, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "schematic — shapes are illustrative, not computed walls; "
             "every wall shown is closed and lies entirely in C_obs",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.01, 1, 0.935))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
