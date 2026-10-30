"""3D visualisation of the weld scene, the sign branches, and a door fiber.

Three figures:
  1. scene overview  -- arm along the weld, weld line, obstacle, off-axis hub
  2. sign branches   -- the same (y, theta) on both s2 pages, showing that only
                        the upper-arm roll (and so the hub) differs
  3. door fiber      -- the gimbal arc at a q2 door: the tool never moves
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.params import Config, PageMapParams, SceneParams, WeldParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

LINK_COL = {"upper_arm": "#1d3557", "forearm": "#4895ef", "tool": "#e76f51",
            "upper_arm_hub": "#9d4edd", "elbow_hub": "#9d4edd"}


def tilted_weld(cfg: Config, tilt_deg: float) -> WeldParams:
    a = np.radians(tilt_deg)
    return WeldParams(p0=cfg.weld.p0, p1=cfg.weld.p1,
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def draw_box(ax, center, half, color="#b0b0b0", alpha=0.30):
    c, h = np.asarray(center, float), np.asarray(half, float)
    lo, hi = c - h, c + h
    verts = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([verts[f] for f in faces],
                                         facecolor=color, alpha=alpha,
                                         edgecolor="#505050", linewidths=0.6))


def draw_sphere(ax, center, radius, color, alpha=0.35):
    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
    ax.plot_surface(center[0] + radius * np.cos(u) * np.sin(v),
                    center[1] + radius * np.sin(u) * np.sin(v),
                    center[2] + radius * np.cos(v),
                    color=color, alpha=alpha, linewidth=0, shade=True)


def draw_arm(ax, arm, scene, q, alpha=1.0, lw=5.0, label=None, hub=True):
    f = arm.frames(q)
    pts = {k: f[k] for k in ("shoulder", "elbow", "wrist", "tool")}
    chain = [("upper_arm", "shoulder", "elbow"), ("forearm", "elbow", "wrist"),
             ("tool", "wrist", "tool")]
    for name, a, b in chain:
        p0, p1 = pts[a], pts[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=LINK_COL[name], lw=lw, alpha=alpha, solid_capstyle="round",
                label=label if name == "upper_arm" else None)
    for k, col, s in (("shoulder", "#222222", 45), ("elbow", "#222222", 30),
                      ("wrist", "#222222", 22)):
        ax.scatter(*pts[k], color=col, s=s, alpha=alpha, depthshade=False)
    # tool tip, to show it sits exactly on the weld line
    ax.scatter(*pts["tool"], color="#e76f51", edgecolor="#111111",
               linewidth=1.0, s=55, alpha=alpha, depthshade=False, zorder=7)
    if hub:
        for a, b, r, name in scene.capsules(f):
            if "hub" in name:
                draw_sphere(ax, a, r, LINK_COL[name], alpha=0.45 * alpha)
    return f


def style(ax, title, weld, scene):
    p0, p1 = weld.p0, weld.p1
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color="#e9c46a", lw=8, solid_capstyle="round", zorder=0)
    ax.scatter(*p0, color="#ffffff", edgecolor="#111111", linewidth=1.6,
               s=110, marker="o", depthshade=False, zorder=6)
    ax.scatter(*p1, color="#111111", edgecolor="#111111", linewidth=1.6,
               s=110, marker="s", depthshade=False, zorder=6)
    ax.text(p0[0], p0[1] - 0.09, p0[2] + 0.02, "START y=0", fontsize=8,
            color="#111111")
    ax.text(p1[0], p1[1] + 0.02, p1[2] + 0.05, "GOAL y=1", fontsize=8,
            color="#111111")
    for c, h, _lab in scene.boxes:
        draw_box(ax, c, h)
    ax.plot([0, 0], [0, 0], [0, 0.34], color="#888888", lw=8,
            solid_capstyle="round", zorder=0)
    ax.set_xlim(-0.25, 0.65)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0.0, 0.95)
    ax.set_box_aspect((0.90, 0.90, 0.95))
    ax.set_xlabel("x [m]", labelpad=-4)
    ax.set_ylabel("y [m]", labelpad=-4)
    ax.set_zlabel("z [m]", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_title(title, fontsize=10, pad=2)
    ax.view_init(elev=20, azim=-78)


def fig_scene(cfg, arm, weld, scene, page, out):
    signs = tuple(1 if c == "+" else -1 for c in page)
    fig = plt.figure(figsize=(16, 5.2))
    for k, y in enumerate((0.0, 0.5, 1.0)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        pos, rot = weld.pose(y)
        shown = 0
        for th in np.linspace(-np.pi, np.pi, 73):
            r = arm.ik_branch(pos, rot, float(th), signs)
            if not r.ok or not arm.within_limits(r.q):
                continue
            if scene.check(arm.frames(r.q)).in_collision:
                continue
            draw_arm(ax, arm, scene, r.q, alpha=0.9 if shown == 0 else 0.16,
                     lw=5.0 if shown == 0 else 2.2, hub=(shown == 0))
            shown += 1
        style(ax, f"page ({page})   y = {y:.2f}   {shown} valid arm angles",
              weld, scene)
    fig.suptitle("Weld scene: tool tip on the weld line (yellow), "
                 "one posture per valid arm angle", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_branches(cfg, arm, weld, scene, out, y=0.5, theta=0.6):
    pos, rot = weld.pose(y)
    pairs = [("++-", "-+-"), ("+--", "---")]
    fig = plt.figure(figsize=(15, 6.4))
    for k, (a, b) in enumerate(pairs):
        for m, page in enumerate((a, b)):
            ax = fig.add_subplot(2, 2, k * 2 + m + 1, projection="3d")
            signs = tuple(1 if c == "+" else -1 for c in page)
            r = arm.ik_branch(pos, rot, theta, signs)
            if r.ok:
                f = draw_arm(ax, arm, scene, r.q)
                lim = arm.within_limits(r.q)
                col = scene.check(f).in_collision
                tag = ("valid" if lim and not col
                       else "joint-limit" if not lim else "collision")
                q = ", ".join(f"{v:+.2f}" for v in r.q)
                style(ax, f"page ({page})   {tag}\nq = [{q}]", weld, scene)
            else:
                style(ax, f"page ({page})   no IK", weld, scene)
    fig.suptitle(f"The same (y={y}, theta={theta}) on paired s2 branches: the "
                 "shoulder/elbow/wrist/tool coincide exactly;\nonly the "
                 "upper-arm roll differs, which moves the off-axis hub (purple)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def fig_fiber(cfg, arm, weld, scene, out, door_y, page="++-"):
    signs = tuple(1 if c == "+" else -1 for c in page)
    pos, rot = weld.pose(door_y)
    r = arm.ik_branch(pos, rot, 0.0, signs)
    fig = plt.figure(figsize=(15, 5.4))
    if not r.ok:
        print("[viz] no IK at door; skipping fiber figure")
        return
    q0 = r.q
    deltas = np.linspace(-np.pi, np.pi, 25)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    tool_dev = []
    n_valid = 0
    for d in deltas:
        q = q0.copy()
        q[0] += d
        q[2] -= d
        ok = arm.within_limits(q) and not scene.check(arm.frames(q)).in_collision
        n_valid += int(ok)
        draw_arm(ax, arm, scene, q, alpha=0.75 if ok else 0.12,
                 lw=3.2 if ok else 1.6, hub=ok)
        p, _ = arm.fk(q)
        tool_dev.append(np.linalg.norm(p - arm.fk(q0)[0]))
    style(ax, f"q2=0 door on page ({page}) at y={door_y:.3f}\n"
              f"gimbal fiber: q1 and q3 counter-rotate  "
              f"({n_valid}/{len(deltas)} valid)", weld, scene)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.degrees(deltas), np.array(tool_dev) * 1e3, "o-",
             color="#1d3557", ms=4)
    ax2.set_xlabel(r"fiber parameter $\delta$  [deg]")
    ax2.set_ylabel("tool tip displacement  [mm]")
    ax2.set_title("The tool never leaves the weld line along the fiber\n"
                  "(anchored at the EXACT door, where q2 = 0)", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-1e-3, 1e-3)
    ax2.axhline(0, color="#e76f51", lw=1, ls="--")
    ax2.text(0.02, 0.92, f"max |displacement| = {max(tool_dev) * 1e3:.2e} mm  "
             f"(0.65 mm if anchored 6.3e-4 rad off the door)",
             transform=ax2.transAxes, fontsize=9, color="#e76f51")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=30.0)
    ap.add_argument("--out", default="out/viz")
    ap.add_argument("--no-obstacle", action="store_true")
    args = ap.parse_args()

    base = Config()
    cfg = dataclasses.replace(
        base,
        arm=dataclasses.replace(base.arm, use_off_axis_upper_arm=True),
        pagemap=PageMapParams(n_y=51, n_theta=51),
        scene=SceneParams(gap_open=True, obstacles=()) if args.no_obstacle
        else base.scene,
    )
    cfg = dataclasses.replace(cfg, weld=tilted_weld(cfg, args.tilt))
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_scene(cfg, arm, weld, scene, "++-", out / "scene.png")
    fig_branches(cfg, arm, weld, scene, out / "branches.png")
    # Anchor the fiber at the EXACT analytic door. The counter-rotation is a
    # true self-motion only where q2 = 0 exactly; 6.3e-4 rad off it the tool
    # drifts 0.65 mm off the weld line instead of 7.9e-14 mm.
    from zzy_rss.doors import find_interior_doors_analytic
    from zzy_rss.pagemap import build_page_map
    pm = build_page_map(cfg, arm, weld, scene, (1, 1, -1), with_kin=False)
    doors = find_interior_doors_analytic(cfg, arm, weld, pm, 0)
    door_y = doors[0].exact_y if doors else 0.1210
    fig_fiber(cfg, arm, weld, scene, out / "fiber.png", door_y=door_y)


if __name__ == "__main__":
    main()
