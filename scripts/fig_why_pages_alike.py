"""The figure behind findings 31/33/37: why the eight pages place the SAME arm,
and what a capsule cannot see.

Four panels, all but one computed from the real kinematics:

  A  how (y, theta) pins shoulder, elbow, wrist and tool -- the two spheres
     whose intersection is the elbow circle, and theta picking a point on it.
     Nothing is left free, which is why the eight branches cannot differ.
  B  all eight branches drawn on top of each other at the same (y, theta).
     They coincide; the caption carries the measured displacement.
  C  what DOES differ: a point rigidly attached to the upper-arm link, drawn
     for all eight branches. The skeleton is fixed but the link is ROLLED 180
     degrees about its own axis, so an off-axis feature swings to the far side.
  D  the consequence, measured (findings 37): how many cells of the page differ
     between two pages by collision alone, as a function of how far a feature
     protrudes past the capsule -- against the joint-limit effects of findings
     31 as reference lines.

usage: python3 scripts/fig_why_pages_alike.py [--out out/why_pages_alike.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.pagemap import q_for_branch, page_name  # noqa: E402
from zzy_rss.params import DEFAULT  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

Y_DEMO, TH_DEMO = 0.50, 1.00
REF = SIGN_BRANCHES[0]
# findings 37, measured by scripts in the scratchpad; reproduced in the docstring
DELTA_MM = [0, 5, 10, 20, 35, 50, 85]
CELLS = [0, 58, 116, 354, 376, 376, 372]
N_CELLS = 3721
# findings 31: the joint-limit effects, for scale
LIMIT_EFFECT = {"flip s2 (limits)": 60, "flip s4 (limits)": 70,
                "flip s6 (limits)": 340}

PALETTE = ["#1d3557", "#457b9d", "#e76f51", "#f4a261",
           "#2a9d8f", "#8ab17d", "#9d4edd", "#c1121f"]


def elbow_circle(arm: SrsArm, shoulder, wrist):
    """Centre, radius and axis of the self-motion circle of the elbow."""
    p = arm.p
    d = wrist - shoulder
    L = float(np.linalg.norm(d))
    u = d / L
    a = (L * L + p.d_se ** 2 - p.d_ew ** 2) / (2.0 * L)
    rho = float(np.sqrt(max(p.d_se ** 2 - a * a, 0.0)))
    return shoulder + a * u, rho, u


def ring(centre, radius, axis, n=200):
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return (centre[None, :] + radius * (np.cos(t)[:, None] * e1[None, :]
                                        + np.sin(t)[:, None] * e2[None, :]))


def sphere(centre, radius, n=26):
    u = np.linspace(0.0, 2.0 * np.pi, 2 * n)
    v = np.linspace(0.0, np.pi, n)
    return (centre[0] + radius * np.outer(np.cos(u), np.sin(v)),
            centre[1] + radius * np.outer(np.sin(u), np.sin(v)),
            centre[2] + radius * np.outer(np.ones_like(u), np.cos(v)))


def equal_3d(ax, pts, pad=0.05):
    pts = np.asarray(pts)
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    c, r = 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo))
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/why_pages_alike.png")
    ap.add_argument("--hub-offset", type=float, default=0.105,
                    help="offset of the illustrative off-axis feature, metres")
    args = ap.parse_args()

    cfg = DEFAULT
    arm = SrsArm(cfg.arm)
    weld = WeldLine(cfg.weld)
    pos, rot = weld.pose(Y_DEMO)
    res = arm.ik_branch(pos, rot, TH_DEMO, REF)
    assert res.ok, "the demo cell must have IK"

    qs, frames = {}, {}
    for s in SIGN_BRANCHES:
        q = res.q if s == REF else q_for_branch(res.q, REF, s)
        qs[page_name(s)] = q
        frames[page_name(s)] = arm.frames(q)
    f0 = frames[page_name(REF)]
    S, E, W, T = (f0["shoulder"], f0["elbow"], f0["wrist"], f0["tool"])

    # measured spread of the skeleton across the eight pages
    spread = max(
        float(np.linalg.norm(frames[n][k] - f0[k]))
        for n in frames for k in ("shoulder", "elbow", "wrist", "tool"))

    # a point rigidly attached to the UPPER-ARM link, off its axis
    def feature(name):
        fr = frames[name]
        base = S + cfg.arm.upper_arm_hub_along * cfg.arm.d_se * (fr["R02"]
                                                                 @ np.array([0, 0, 1.0]))
        return base + fr["R02"] @ np.array([args.hub_offset, 0.0, 0.0])

    feats = {n: feature(n) for n in frames}
    feat_spread = max(float(np.linalg.norm(feats[n] - feats[page_name(REF)]))
                      for n in feats)

    fig = plt.figure(figsize=(17.5, 9.2))
    fig.suptitle("Why the eight pages place the SAME arm — and what a capsule "
                 "cannot see", fontsize=15, fontweight="bold", y=0.975)

    # ---------------------------------------------------------------- A ----
    axA = fig.add_subplot(2, 2, 1, projection="3d")
    axA.set_title("A.  $(y,\\theta)$ pins every joint centre — nothing is left "
                  "free\n(so the eight branches have nothing to differ by)",
                  fontsize=11)
    p0, p1 = weld.p0, weld.p1
    axA.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
             color="#e9c46a", lw=7, solid_capstyle="round",
             label="weld line (the task)", zorder=0)
    axA.plot([0, 0], [0, 0], [0, cfg.arm.d_bs], color="#999999", lw=8,
             solid_capstyle="round", zorder=0)
    # great circles instead of shaded spheres: same information, far less ink
    for c, rad, col, lab in ((S, cfg.arm.d_se, "#457b9d", "$|E-S| = d_{se}$"),
                             (W, cfg.arm.d_ew, "#e76f51", "$|W-E| = d_{ew}$")):
        for axis in (np.array([0, 0, 1.0]), np.array([0, 1.0, 0]),
                     np.array([1.0, 0, 0])):
            g = ring(c, rad, axis)
            axA.plot(g[:, 0], g[:, 1], g[:, 2], color=col, lw=0.8, alpha=0.33)
        axA.plot([], [], color=col, lw=2, alpha=0.8, label=lab)
    cen, rho, axis = elbow_circle(arm, S, W)
    circ = ring(cen, rho, axis)
    axA.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="#2a9d8f", lw=3.0,
             label="elbow circle: $\\theta$ picks ONE point", zorder=6)
    axA.plot([S[0], W[0]], [S[1], W[1]], [S[2], W[2]], color="#2a9d8f",
             lw=1.0, ls=":", zorder=4)
    axA.plot([S[0], E[0], W[0], T[0]], [S[1], E[1], W[1], T[1]],
             [S[2], E[2], W[2], T[2]], color="#1d3557", lw=4.0, alpha=0.95,
             zorder=7)
    # labels pushed apart along distinct directions so they cannot collide
    for pt, nm, col, off in ((S, "S", "#1d3557", (-0.07, 0.00, -0.05)),
                             (E, "E", "#2a9d8f", (0.00, 0.00, 0.055)),
                             (W, "W", "#e76f51", (0.00, 0.06, 0.03)),
                             (T, "tool", "#c1121f", (0.05, -0.05, -0.03))):
        axA.scatter(*pt, s=70, color=col, depthshade=False, zorder=10)
        axA.text(pt[0] + off[0], pt[1] + off[1], pt[2] + off[2], nm,
                 fontsize=11, color=col, fontweight="bold", zorder=11)
    axA.legend(loc="upper left", fontsize=8, framealpha=0.92)
    equal_3d(axA, np.array([S, E, W, T, p0, p1]))
    axA.set_xlabel("x"); axA.set_ylabel("y"); axA.set_zlabel("z")
    axA.view_init(elev=20, azim=-62)

    # ---------------------------------------------------------------- B ----
    axB = fig.add_subplot(2, 2, 2, projection="3d")
    axB.set_title(f"B.  all EIGHT branches at the same $(y,\\theta)$, overlaid\n"
                  f"skeletons coincide to {spread * 1000:.3f} mm — it is one arm",
                  fontsize=11)
    for i, n in enumerate(sorted(frames)):
        fr = frames[n]
        pts = np.array([fr["shoulder"], fr["elbow"], fr["wrist"], fr["tool"]])
        axB.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=PALETTE[i],
                 lw=8.5 - 0.9 * i, alpha=0.95, solid_capstyle="round",
                 label=f"page ({n})", zorder=i)
    for pt in (S, E, W, T):
        axB.scatter(*pt, s=46, color="#111111", depthshade=False, zorder=20)
    axB.legend(loc="upper left", fontsize=7.6, ncol=2, framealpha=0.9)
    equal_3d(axB, np.array([S, E, W, T]))
    axB.set_xlabel("x"); axB.set_ylabel("y"); axB.set_zlabel("z")
    axB.view_init(elev=20, azim=-62)

    # ---------------------------------------------------------------- C ----
    axC = fig.add_subplot(2, 2, 3)
    axC.set_title("C.  what DOES differ: flipping $s_2$ ROLLS frame 2 "
                  "(the upper-arm housing,\njoint 2$\\to$3) by 180° about its "
                  "own axis — cross-section looking down $S\\rightarrow E$",
                  fontsize=10.5)
    # project the computed feature positions into the plane normal to S->E
    u = (E - S) / np.linalg.norm(E - S)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, u))) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    proj = {}
    for n, h in feats.items():
        d = h - S
        d = d - float(np.dot(d, u)) * u
        proj[n] = np.array([float(np.dot(d, e1)), float(np.dot(d, e2))])
    r_arm = cfg.arm.r_upper_arm

    axC.add_patch(plt.Circle((0, 0), r_arm, facecolor="#1d3557", alpha=0.18,
                             edgecolor="#1d3557", lw=2.2, zorder=1))
    axC.plot([0], [0], marker="+", ms=13, color="#111", mew=2, zorder=5)
    axC.text(0.0, -0.016, "link axis", fontsize=8.6, color="#111",
             ha="center", va="top")
    # the two clusters: four pages coincide at each, so label them to the side
    groups = {}
    for n, xy in proj.items():
        key = tuple(np.round(xy, 6))
        groups.setdefault(key, []).append(n)
    ordered = sorted(groups.items(), key=lambda kv: -kv[0][1])
    for (xy, names), col, va in zip(ordered, ("#c1121f", "#2a9d8f"),
                                    ("bottom", "top")):
        xy = np.array(xy)
        axC.scatter(*xy, s=300, color=col, edgecolor="#222", lw=1.1, zorder=8)
        axC.annotate(f"{len(names)} pages\n" + ", ".join(sorted(names)),
                     xy=xy, xytext=(0.105, xy[1]), fontsize=8.4, color=col,
                     ha="left", va="center", fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=col, lw=1.1))
    k = [np.array(kk) for kk, _ in ordered]
    a0, a1 = k[0], k[1]
    rr = float(np.linalg.norm(a0))
    th0 = np.arctan2(a0[1], a0[0])
    ths = np.linspace(th0, th0 + np.pi, 90)
    axC.plot(rr * np.cos(ths), rr * np.sin(ths), ls="--", color="#666", lw=1.4,
             zorder=3)
    mid = th0 + 0.5 * np.pi
    axC.annotate("", xy=tuple(rr * np.array([np.cos(mid + 0.18),
                                             np.sin(mid + 0.18)])),
                 xytext=tuple(rr * np.array([np.cos(mid - 0.18),
                                             np.sin(mid - 0.18)])),
                 arrowprops=dict(arrowstyle="->", color="#666", lw=2.0))
    axC.text(-0.172, 0.0, f"roll 180°\nfeature swings\n"
             f"{feat_spread * 1000:.0f} mm",
             fontsize=8.6, color="#444", ha="left", va="center",
             fontweight="bold")
    # a realistic small flange, drawn at both rolls, to the same scale
    for ang, col in ((th0, "#c1121f"), (th0 + np.pi, "#2a9d8f")):
        c5 = r_arm * np.array([np.cos(ang), np.sin(ang)])
        axC.add_patch(plt.Circle(tuple(c5), 0.011, facecolor=col, alpha=0.5,
                                 edgecolor=col, lw=1.2, zorder=6))
    axC.plot([], [], marker="o", ls="none", color="#999", alpha=0.6, ms=8,
             label="realistic ~5 mm flange (to scale)")
    axC.plot([], [], marker="o", ls="none", color="#c1121f", ms=11,
             label="illustrative feature, 105 mm off-axis")
    axC.add_patch(plt.Circle((0, 0), r_arm, facecolor="none",
                             edgecolor="#1d3557", lw=2.4, zorder=7))
    axC.text(0.062, 0.030,
             "capsule outline is\nIDENTICAL under the roll\n⇒  blind to it",
             fontsize=8.8, color="#1d3557", fontweight="bold", va="center",
             ha="left",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1d3557",
                       lw=0.8, alpha=0.9))
    axC.set_aspect("equal")
    lim = max(np.abs(np.array(list(proj.values()))).max(), r_arm)
    axC.set_xlim(-lim * 2.1, lim * 2.35)
    axC.set_ylim(-lim * 1.55, lim * 1.55)
    axC.set_xlabel("distance from the link axis  [m]")
    axC.grid(alpha=0.22)
    axC.legend(loc="lower left", fontsize=7.6, framealpha=0.94)

    # ---------------------------------------------------------------- D ----
    axD = fig.add_subplot(2, 2, 4)
    axD.set_title("D.  the consequence, measured (findings 37)\n"
                  "cells where two pages differ by COLLISION alone, vs how far "
                  "a feature protrudes", fontsize=11)
    pct = [100.0 * c / N_CELLS for c in CELLS]
    axD.plot(DELTA_MM, pct, "o-", color="#c1121f", lw=2.4, ms=7,
             label="collision-only page difference")
    for x, y, c in zip(DELTA_MM, pct, CELLS):
        axD.annotate(f"{c}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5, color="#c1121f")
    styles = ["--", "-.", ":"]
    xpos = [40, 62, 86]
    vpos = ["top", "bottom", "bottom"]
    dy = [-0.17, 0.12, 0.12]
    for ((lab, cells), st, xp, vp, d) in zip(LIMIT_EFFECT.items(), styles,
                                             xpos, vpos, dy):
        axD.axhline(100.0 * cells / N_CELLS, color="#1d3557", ls=st, lw=1.5,
                    alpha=0.85)
        axD.text(xp, 100.0 * cells / N_CELLS + d, f"{lab}: {cells} cells",
                 fontsize=8.4, color="#1d3557", ha="right", va=vp,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                           alpha=0.85))
    axD.set_xlabel("how far the feature protrudes past the capsule  [mm]")
    axD.set_ylabel(f"% of the {N_CELLS}-cell page that differs")
    axD.grid(alpha=0.28)
    axD.legend(loc="center right", fontsize=9)
    axD.set_xlim(-3, 90); axD.set_ylim(-0.4, 11.4)
    axD.annotate("a 5 mm asymmetry already matches\nthe WHOLE s2 limit effect",
                 xy=(5.5, 100.0 * 58 / N_CELLS), xytext=(16, 5.0),
                 fontsize=9.0, color="#444", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=165)
    plt.close(fig)
    print(f"[fig] {out}")
    print(f"[fig] skeleton spread over the eight pages : {spread * 1000:.4f} mm")
    print(f"[fig] off-axis feature spread              : {feat_spread * 1000:.1f} mm")


if __name__ == "__main__":
    main()
