#!/usr/bin/env python3
"""Schematic (not data): how walls, doors and obstacles may and may not meet.

  (a) NOT allowed: a live door crossing wall A. Inside -> door -> turn the
      wrist at the door -> leave on the other side: the wall is bypassed, so
      Algorithm 3 rejects every wall triangle touching the door tube.
  (b) Allowed: the door lies wholly inside wall A (it is the page's exit);
      the part of the door curve inside an obstacle is simply unusable.
  (c) Allowed: two verified walls crossing each other. Both lie in C_obs, so
      their crossing does too; leaving either still means passing C_obs.

    python3 scripts/fig_wall_rules.py --out out/fig_wall_rules.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

OBS = "#f4a582"
WALL_A = "#1d6fd8"
WALL_B = "#2b9348"
DOOR = "#7b2cbf"


def blob(cx, cy, rx, ry, wob=0.08, n=200, ph=0.0):
    t = np.linspace(0, 2 * np.pi, n)
    r = 1 + wob * np.sin(3 * t + ph)
    return np.column_stack([cx + rx * r * np.cos(t), cy + ry * r * np.sin(t)])


def base(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.add_patch(Polygon(blob(5, 4, 3.6, 3.0, ph=0.4), closed=True, fc=OBS, ec="none",
                         alpha=0.9, zorder=0))
    ax.add_patch(Polygon(blob(5, 4, 2.4, 1.9, ph=1.0), closed=True, fc="white", ec="none",
                         zorder=1))
    ax.plot(2.9, 4.0, "o", color="#1d6fd8", ms=12, mec="k", zorder=6)
    ax.text(2.9, 3.45, "start", ha="center", fontsize=10, zorder=6)
    ax.text(9.6, 7.5, "goal\n(valid)", ha="right", va="top", fontsize=10, color="#d62828")
    ax.text(5, 7.55, "C_obs (obstacle / joint limits)", ha="center", fontsize=9, color="#a0522d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fig_wall_rules.png")
    args = ap.parse_args()
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.8))

    # (a) door crossing wall A -- forbidden
    ax = axs[0]
    base(ax, "(a) NOT allowed: a live door crosses wall A")
    w = blob(5, 4, 3.0, 2.45, wob=0.05, ph=0.7)
    ax.plot(w[:, 0], w[:, 1], color=WALL_A, lw=3, zorder=3)
    xs = np.linspace(4.2, 9.4, 50)
    ys = 5.0 + 0.25 * np.sin(xs)
    ax.fill_between(xs, ys - 0.35, ys + 0.35, color=DOOR, alpha=0.15, zorder=2)
    ax.plot(xs, ys, color=DOOR, lw=2.5, ls="--", zorder=4)
    ax.text(9.3, 5.6, "door curve\n+ its tube", color=DOOR, ha="right", fontsize=9)
    cross_x = 7.85
    ax.plot(cross_x, 5.0 + 0.25 * np.sin(cross_x), "X", color="#d62828", ms=22, mec="k",
            zorder=7)
    ax.annotate("wall triangles touching the tube\nare REJECTED by Algorithm 3",
                (cross_x, 4.9), (6.2, 1.1), fontsize=9, color="#d62828",
                arrowprops=dict(arrowstyle="->", color="#d62828"))
    path = np.array([[2.9, 4.0], [3.8, 4.6], [4.7, 5.05]])
    ax.plot(*path.T, color="k", lw=1.8, zorder=5)
    ax.plot([4.7, 9.0], [5.05, 5.0 + 0.25 * np.sin(9.0)], color="k", lw=1.8, ls=":",
            zorder=5)
    ax.text(6.0, 6.05, "inside -> door -> along the door -> outside:\n"
                       "the wall would be bypassed", fontsize=9, ha="center")

    # (b) door wholly inside wall A, partly through the obstacle -- allowed
    ax = axs[1]
    base(ax, "(b) Allowed: the door is wholly inside wall A")
    ax.plot(w[:, 0], w[:, 1], color=WALL_A, lw=3, zorder=3)
    xs = np.linspace(4.0, 7.4, 40)
    ys = 4.7 + 0.5 * np.sin(1.3 * xs)
    ax.fill_between(xs, ys - 0.3, ys + 0.3, color=DOOR, alpha=0.15, zorder=2)
    free = xs < 6.6
    ax.plot(xs[free], ys[free], color=DOOR, lw=2.5, ls="--", zorder=4)
    ax.plot(xs[~free], ys[~free], color="#888888", lw=2.5, ls="--", zorder=4)
    ax.text(4.0, 5.9, "door: the only exit\n-> follow it to page 2", color=DOOR, fontsize=9)
    ax.annotate("part of the door curve inside C_obs:\nposture collides, unusable (fine)",
                (7.2, ys[-3]), (6.0, 1.0), fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(5, 7.0, "wall A: closed, every triangle in C_obs,\nstart inside, goal outside",
            ha="center", fontsize=9, color=WALL_A)

    # (c) two walls crossing -- allowed
    ax = axs[2]
    base(ax, "(c) Allowed: two verified walls cross each other")
    wa = blob(5, 4, 3.0, 2.45, wob=0.05, ph=0.7)
    wb = blob(5.0, 4.0, 3.0, 2.45, wob=0.10, ph=2.4)
    ax.plot(wa[:, 0], wa[:, 1], color=WALL_A, lw=3, zorder=3, label="wall A")
    ax.plot(wb[:, 0], wb[:, 1], color=WALL_B, lw=3, zorder=3, label="wall B")
    ax.text(5, 0.55, "both walls lie in C_obs, so where they cross is C_obs too.\n"
                     "Each is a separate statement; leaving either still crosses C_obs.",
            ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=10)

    fig.suptitle("Schematic of the wall rules (illustration, not data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
