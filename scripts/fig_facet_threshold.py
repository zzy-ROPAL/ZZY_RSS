#!/usr/bin/env python3
"""Why the door check on a facet needs a threshold tied to the facet's size.

A real triangle facet placed across a real q2 door on the 3-D EXP 2 patch,
sigma = |q2| (sigma = 0 is the door) evaluated on its face, and the chart's
real guaranteed Lipschitz bound L:

  (a) the face: all three nodes sit outside a thin door zone sigma < eps
      (eps chosen by hand), yet the door pierces the middle -- a node-only
      check passes a facet that cuts through a door;
  (b) a profile across the face: from a node's value, the only guarantee
      between nodes is the cone sigma >= sigma(node) - L * distance; it
      reaches 0 inside the facet, so no threshold below L * r can certify it;
  (c) what the test does instead: split the facet; pieces whose ball passes
      sigma(c) > L * r_piece are certified (green), pieces at the door keep
      failing (red) -- the door is found, never missed.

    python3 scripts/fig_facet_threshold.py --out docs/fig_facet_threshold.png
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
from matplotlib.patches import Polygon  # noqa: E402

from run_3d import PATCHES  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import I_Q2, SrsChart  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.facets import enclosing_ball, subdivide  # noqa: E402
from zzy_rss.params import Config, Page3Params  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/fig_facet_threshold.png")
    ap.add_argument("--radius", type=float, default=0.03)
    args = ap.parse_args()

    cfg, p3 = Config(), Page3Params()
    arm = SrsArm(cfg.arm)
    surf = WeldSurface(PATCHES["exp2"])
    ch = SrsChart.for_surface(arm, surf, p3)
    s = ch.scale
    c = door_curves(cfg, arm, surf, p3)[0]
    on = c.pts[np.all((c.pts[:, :2] >= 0) & (c.pts[:, :2] <= 1), axis=1)]
    i = len(on) // 2
    p = on[i]
    t = (on[i + 1] - on[i - 1]) * s
    t /= np.linalg.norm(t)
    a = np.cross(t, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(t, a)                       # the facet's plane is spanned by a, b
    R = args.radius

    def to_page(xy):                          # 2-D facet-plane coords -> page state
        xy = np.atleast_2d(xy)
        return p[None, :] + (xy[:, :1] * a + xy[:, 1:2] * b) / s

    def sig(xy):
        return ch.sigma(to_page(xy))[:, I_Q2]

    ang = np.array([90.0, 210.0, 330.0]) + 17.0
    V2 = R * np.column_stack([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    sv = sig(V2)
    L = float(np.nanmax(ch.lipschitz(to_page(np.zeros((1, 2))), np.array([R]))[:, I_Q2]))
    eps_hand = 0.6 * float(sv.min())          # a hand-picked threshold the nodes pass
    eps_safe = L * R

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))

    # (a) the face
    ax = axs[0]
    g = np.linspace(-1.25 * R, 1.25 * R, 301)
    X, Y = np.meshgrid(g, g)
    S = sig(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    im = ax.pcolormesh(X, Y, np.degrees(S), cmap="viridis_r", shading="auto",
                       vmin=0, vmax=np.degrees(S.max()))
    plt.colorbar(im, ax=ax, label="sigma = |q2|  (deg)", fraction=0.046)
    ax.contour(X, Y, S, levels=[eps_hand], colors="w", linewidths=1.5, linestyles="--")
    ax.add_patch(Polygon(V2, closed=True, fill=False, ec="k", lw=2))
    ax.plot(0, 0, "r*", ms=16, label="door pierces the face (sigma = 0)")
    for v, val in zip(V2, sv):
        ax.plot(*v, "o", ms=9, mfc="w", mec="k")
        off = (-120, 10) if v[0] > 0.5 * R else ((8, 8) if v[1] > 0 else (8, -24))
        ax.annotate(f"node: sigma={np.degrees(val):.2f} deg\n> eps -> passes",
                    v, textcoords="offset points", xytext=off, fontsize=8,
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    ax.plot([], [], "w--", label=f"hand-picked door zone sigma < eps = {np.degrees(eps_hand):.2f} deg")
    ax.set_aspect("equal")
    ax.set_title("(a) Node check with a hand-picked threshold:\nall 3 nodes pass, "
                 "but a door pierces the middle of the face", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlabel("in the facet's plane (scaled page units)")

    # (b) profile node -> centre -> opposite edge
    ax = axs[1]
    v0 = V2[0]
    end = -0.5 * v0                            # midpoint of the opposite edge
    ts = np.linspace(0.0, 1.0, 400)
    line = v0[None, :] + ts[:, None] * (end - v0)[None, :]
    dist = np.linalg.norm(line - v0, axis=1)
    ax.plot(dist, np.degrees(sig(line)), "k-", lw=2, label="true sigma along the line")
    ax.plot(dist, np.degrees(sv[0] - L * dist), "C3--", lw=2,
            label=f"guaranteed lower bound from the node:\nsigma(node) - L * distance   (L = {L:.2f})")
    ax.axhline(0, color="gray", lw=1)
    ax.axhline(np.degrees(eps_hand), color="C0", ls=":", lw=1.5,
               label=f"hand-picked eps = {np.degrees(eps_hand):.2f} deg")
    ax.axhline(np.degrees(eps_safe), color="C2", ls="-.", lw=1.5,
               label=f"safe threshold eps = L * r = {np.degrees(eps_safe):.2f} deg")
    dc = float(np.linalg.norm(v0))
    ax.axvline(dc, color="r", lw=1)
    ax.annotate("door\n(centre)", (dc, 0.05 * np.degrees(sv[0])), color="r", fontsize=9)
    ax.set_xlabel("distance from the node, across the face (scaled)")
    ax.set_ylabel("sigma (deg)")
    ax.set_title("(b) From a node, only the cone sigma - L*distance is guaranteed:\n"
                 "it drops below 0 inside the facet, so the node proves nothing\n"
                 "unless sigma(node) > L * r (the green line)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(np.degrees(min(-0.2 * eps_safe, (sv[0] - L * dist).min())) * 1.05,
                np.degrees(eps_safe) * 1.15)

    # (c) split until certified or the door is pinned
    ax = axs[2]
    pieces = [(V2, 0)]
    leaves = []
    max_depth = 4
    while pieces:
        tri, dpt = pieces.pop()
        cs, r = enclosing_ball(tri)
        ok = float(sig(cs[None, :])[0]) > L * r
        if ok or dpt == max_depth:
            leaves.append((tri, ok))
        else:
            pieces += [(q, dpt + 1) for q in subdivide(tri)]
    for tri, ok in leaves:
        ax.add_patch(Polygon(tri, closed=True, fc=("#a1d99b" if ok else "#fb6a4a"),
                             ec="k", lw=0.4))
    ax.plot(0, 0, "r*", ms=16)
    n_ok = sum(1 for _, ok in leaves if ok)
    ax.set_xlim(-1.25 * R, 1.25 * R)
    ax.set_ylim(-1.25 * R, 1.25 * R)
    ax.set_aspect("equal")
    ax.set_title(f"(c) The real test: split, certify each piece with sigma(c) > L * r_piece\n"
                 f"green = certified door-free ({n_ok} pieces), red = still touches the door\n"
                 f"({len(leaves) - n_ok} pieces at depth {max_depth}): the facet is rejected, "
                 f"the door is not missed", fontsize=10)
    ax.set_xlabel("in the facet's plane (scaled page units)")

    fig.suptitle(f"A real facet (radius r = {R}) across a real q2 door, 3-D EXP 2 patch.  "
                 f"Nodes are checked with sigma = |q2|; the door is where sigma = 0.",
                 fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[fig] nodes sigma = {np.degrees(sv).round(3)} deg, centre sigma = "
          f"{np.degrees(sig(np.zeros((1, 2)))[0]):.2e} deg, L = {L:.3f}, "
          f"eps_hand = {np.degrees(eps_hand):.3f} deg, L*r = {np.degrees(eps_safe):.3f} deg")
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
