"""The three experiments, on the real Schunk LWA 4D.

    EXP 1  no obstacle                 robot FINISHES the weld
    EXP 2  obstacle, doors CLOSED      infeasible; proof closes in ONE page
    EXP 3  obstacle, door OPEN         infeasible; the proof CROSSES the door
                                       and EXPANDS to a second page

EXP 1 and EXP 2 share a scene: weld x = 0.28, z = 0.46, y in [-0.26, 0.26];
a SINGLE start posture on page (++-) at y = 0; goal is y = 1.

EXP 3 needs its own weld placement, and that is not a convenience -- it is
what makes the experiment possible at all (docs/findings.md 26-27). Two
things have to hold at once, and they pull against each other: the obstacle
must disconnect the start from the goal, and some door must stay OPEN.
A door is open iff some point of its self-motion fiber is collision-free and
within limits, and that fiber is the forearm sweeping a cone about the
shoulder-wrist axis -- the same motion an obstacle has to block in order to
disconnect anything. At the shared placement the doors sit at y = 0.088,
right against the start, and ZGAP sits on their fiber: all 16 doors close
(witness collision:forearm/high) and EXP 3 collapses into a second copy of
EXP 2. Door position is set by |W(y) - E*| = d_ew, so pushing the seam out to
x = 0.36 moves the door to y = 0.429, mid-seam, with 97.4% of its fiber
valid. That leaves room for an obstacle to cut off the goal end of the seam
without touching the door.

The torch tilt remains the joint-limit half of the switch (findings 12): a q2
door pins the configuration so q5 is determined, and the s6 mirror shifts it
by pi. At tilt 0 that lands on +/-180 deg, outside the real +/-178.85 deg
limit, so the start page's doors are shut no matter where the obstacle is.
Tilt 30 deg moves q5 off zero and opens them.

Outputs per experiment: an mp4 and an annotated 8-page figure. EXP 3 also
writes exp3_door_crossing.png (wall #1 -> the fiber -> wall #2) and
exp3_certificate.json (the verdict from flood.run_proof, which walks every
page reachable through every open door -- the two walls in the figure are the
illustration, not the proof).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import (  # noqa: E402
    find_inf_proof_region, invalid_points, label_regions, live_discs,
    other_valid_points, region_points, run_proof, thin, write_certificate,
)
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.wall import (  # noqa: E402
    PageLabeller, sigma_disc_samples, virtual_boundary_samples)  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
START_PAGE, START_SIGNS = "++-", (1, 1, -1)
PARTNER_PAGE = "-+-"                      # differs from START_PAGE in s2 only
PARTNER_SIGNS = (-1, 1, -1)
# Two 12 cm cubes stacked at x = 0.14 with a 16 cm vertical gap between them
# (z = 0.38 and z = 0.66). findings 24: a SINGLE box cannot work here. Shrink
# it and the arm threads through (feasible); grow it slightly and the weld
# point becomes unreachable, which is infeasible for a trivial reason and
# never exercises the wall (findings 22). The gap between those outcomes is
# narrower than the grid resolves -- every single-box candidate flipped
# verdict between 61, 91 and 121 cells.
#
# Two boxes separate the two effects: they block the HIGH and LOW elbow arcs
# and leave a channel between them, so every weld point stays reachable
# (curtain 0) while the elbow cannot cross from one side of the gap to the
# other. That is connectivity failure without reachability failure, the only
# kind that needs a wall. Verified stable at 61/91/121 cells and both tilts.
ZGAP = (BoxObstacle(center=(0.14, 0.0, 0.38),
                    half_extent=(0.06, 0.06, 0.06), label="low"),
        BoxObstacle(center=(0.14, 0.0, 0.66),
                    half_extent=(0.06, 0.06, 0.06), label="high"))
OBS_CLOSED = ZGAP

# EXP 3: its own weld placement and its own obstacle (findings 26-29).
#
# Weld at x = 0.32, z = 0.40, tilt 15 deg. Door position is set by
# |W(y) - E*| = d_ew, so the PLACEMENT -- not the obstacle -- decides where the
# pages touch. At the EXP 1/2 placement the doors sit at y = 0.088, hard
# against the start, and ZGAP at y = 0 lies straight across their fiber: all 16
# shut, witness collision:forearm/high, and EXP 3 collapsed into a copy of
# EXP 2. Here the door lands at y = 0.336 -- a third of the way down the seam,
# with 94.7% of its fiber valid, so an obstacle has to block nearly the whole
# elbow swing to close it.
#
# Obstacle: ZGAP's two-box idea slid to y = +0.10. The pair still blocks the
# HIGH and LOW elbow arcs so the elbow cannot cross the gap (connectivity
# failure, findings 24), but it now sits over the GOAL half of the seam and
# clears the door's fiber cone entirely.
#
# Measured at res 61 / 91 / 121 -- all three agree (invariant 8, findings 24):
#   (A) no common curtain    curtain 0; pinch 10 / 15 / 20 cells. The channel
#                            WIDENS with resolution, so it is a real gap and
#                            not the grid noise that flipped every single-box
#                            candidate in findings 24. Widest margin of the
#                            ten scenes that qualified.
#   (B) start cannot reach   no spanning region on any page.
#       the goal
#   (C) a door is open       8 of 16 doors open, and exactly ONE of them lies
#                            in the start region: y = 0.3356, theta = 0,
#                            vanishing s2, fiber 94.7% valid. Exactly one is
#                            what makes the certificate genuinely two-page.
#   (D) the partner is       it lands in region 2 of (-+-), which does not
#       blocked too          reach the goal. Without this the door would
#                            RESCUE the robot and the task would be feasible.
#   theta seam clear         no valid cell touches theta = +/-pi on any page at
#                            any of the three resolutions, so the flat
#                            rectangle is exactly the cylinder (invariant 5).
#
# Of the ten scenes satisfying (A)-(D), this one's start page is by far the
# cheapest to wall: 162/162 facets in 7 Algorithm 4 iterations at gamma 1.10.
# The runner-up (weld x = 0.36, door y = 0.429, fiber 97.4%) looked better on
# geometry and on initial separability, but its start-page wall drives gamma to
# 229 and stalls near 250/274 facets (findings 29).
EXP3_TILT = 15.0
EXP3_WELD = dict(x=0.32, z=0.40)
OBS_OPEN = (BoxObstacle(center=(0.20, 0.10, 0.34),
                        half_extent=(0.06, 0.08, 0.06), label="low"),
            BoxObstacle(center=(0.20, 0.10, 0.62),
                        half_extent=(0.06, 0.08, 0.06), label="high"))


# Algorithm 4 feed-back labelling; set from --label-mode in main().
LABEL_MODE = "metric"
REST_MODE = "goal_only"
CMAP = ListedColormap(["#2a9d8f", "#e9e9e9", "#e9c46a", "#e76f51"])


def weld_for(tilt_deg, x=0.28, z=0.46, half_y=0.26):
    """The weld seam. x/z place it in the workspace; the tilt rolls the torch.

    The placement is not cosmetic: door position is set by |W(y) - E*| = d_ew,
    so x and z decide WHERE along the seam the pages touch, and the tilt
    decides whether the door's fiber clears the joint limits (findings 12).
    EXP 1 and EXP 2 keep the original placement; EXP 3 needs its own, and
    says so (findings 26).
    """
    a = np.radians(tilt_deg)
    return WeldParams(p0=(x, -half_y, z), p1=(x, half_y, z),
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def make(tilt, obstacles, res, weld_kw=None):
    base = Config()
    cfg = dataclasses.replace(
        base, weld=weld_for(tilt, **(weld_kw or {})),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=tuple(obstacles or ())),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz",
                                 label_mode=LABEL_MODE,
                                 rest_mode=REST_MODE))
    return cfg, SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)


def theta_seam_check(pms, label=""):
    """CLAUDE.md invariant 5: the page is a CYLINDER in theta.

    Treating it as a flat rectangle is EXACT, not approximate, only while no
    valid cell touches theta = +/-pi -- the joint limits must cut the theta
    circle before the seam. That is a measured property of a placement, not a
    theorem, so every new scene has to be re-measured. If a valid cell does
    touch the seam, a wall that looks closed on the rectangle is actually
    open around the back of the cylinder and the proof is void (findings 23).
    """
    touching = []
    for n, pm in pms.items():
        lo = int(pm.valid_mask[:, 0].sum())
        hi = int(pm.valid_mask[:, -1].sum())
        if lo or hi:
            touching.append((n, lo, hi))
    if touching:
        print(f"[seam]{label} WARNING: valid cells touch theta=+/-pi on "
              f"{len(touching)} page(s): "
              + ", ".join(f"{n}({lo}+{hi})" for n, lo, hi in touching)
              + " -- the flat-rectangle page is NOT equivalent to the "
                "cylinder here; theta_periodic must be True.")
    else:
        print(f"[seam]{label} no valid cell touches theta=+/-pi on any of the "
              f"{len(pms)} pages: flat rectangle == cylinder, exactly.")
    return touching


def bfs_path(pm, lab=None, region=None, to_goal=True):
    """Path in the valid grid from y=0 toward y=1 (theta periodic)."""
    m = pm.valid_mask if lab is None else (lab == region)
    ny, nt = m.shape
    starts = [(0, j) for j in range(nt) if m[0, j]]
    if not starts:
        return None
    prev = {s: None for s in starts}
    q, best = deque(starts), starts[0]
    while q:
        i, j = q.popleft()
        if i > best[0]:
            best = (i, j)
        if to_goal and i == ny - 1:
            best = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, (j + dj) % nt
            if 0 <= ni < ny and m[ni, nj] and (ni, nj) not in prev:
                prev[(ni, nj)] = (i, j)
                q.append((ni, nj))
    out, cur = [], best
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return np.array([[pm.y_grid[i], pm.theta_grid[j]] for i, j in out])


def prove(cfg, arm, weld, scene, pm, lab, k, discs, iters=6, verbose=False):
    sg = tuple(1 if c == "+" else -1 for c in pm.name)
    rng = np.random.default_rng(0)
    pts = region_points(pm, lab, k)
    live, disch = live_discs(cfg, arm, weld, scene, sg, discs)
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    se, re_ = sigma_disc_samples(live, pts, tol)
    seed = thin(pts, 450, rng)
    _, regs_here = label_regions(cfg, pm)
    rest = np.vstack([thin(other_valid_points(pm, lab, k, regs_here,
                                              cfg.wall.rest_mode), 450, rng),
                      virtual_boundary_samples(cfg, 80)])
    if len(se):
        seed = np.vstack([seed, se])
    if len(re_):
        rest = np.vstack([rest, re_])
    inv = thin(invalid_points(pm), 150, rng)
    labeller = PageLabeller(cfg, arm, weld, scene, sg, pm, lab, k,
                            search_cells=cfg.wall.label_search_cells,
                            motion_samples=cfg.wall.label_motion_samples)
    poly, rep, w, it, rr = find_inf_proof_region(
        cfg, arm, weld, scene, sg, seed, rest, inv, live, max_iters=iters,
        verbose=verbose, labeller=labeller)
    return poly, rep, live, disch


def draw_box(ax, c, h):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c, h = np.asarray(c, float), np.asarray(h, float)
    lo, hi = c - h, c + h
    v = np.array([[x, y, z] for x in (lo[0], hi[0])
                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor="#c1121f",
                                         alpha=0.45, edgecolor="#780000", lw=0.8))


def draw_arm(ax, cfg, arm, scene, weld, q, trail):
    p0, p1 = weld.p0, weld.p1
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color="#e9c46a", lw=9, solid_capstyle="round", zorder=0)
    ax.plot([0, 0], [0, 0], [0, cfg.arm.d_bs], color="#888888", lw=10,
            solid_capstyle="round", zorder=0)
    for c, h, _l in scene.boxes:
        draw_box(ax, c, h)
    f = arm.frames(q)
    pts = [f["shoulder"], f["elbow"], f["wrist"], f["tool"]]
    for i, col in enumerate(("#1d3557", "#4895ef", "#e76f51")):
        a, b = pts[i], pts[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col, lw=7,
                solid_capstyle="round")
    for pt in pts[:3]:
        ax.scatter(*pt, color="#222", s=40, depthshade=False)
    if trail:
        T = np.array(trail)
        ax.plot(T[:, 0], T[:, 1], T[:, 2], color="#e63946", lw=4)
    ax.scatter(*f["tool"], color="#e63946", s=95, depthshade=False,
               edgecolor="#111", zorder=9)
    ax.set_xlim(-0.25, 0.60); ax.set_ylim(-0.42, 0.42); ax.set_zlim(0, 0.80)
    ax.set_box_aspect((0.85, 0.84, 0.80))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=20, azim=-72)
    return f


def draw_page(ax, pm, path=None, upto=None, wall=None, discs=None, title=""):
    ax.imshow(pm.tag.T, origin="lower", aspect="auto", cmap=CMAP, vmin=-.5,
              vmax=3.5, extent=[0, 1, pm.theta_grid[0], pm.theta_grid[-1]])
    if wall is not None:
        for loop in wall:
            ax.plot(loop[:, 0], loop[:, 1], color="#9d0208", lw=3)
    for d in (discs or []):
        if not d.vanishing:
            continue
        ay, at = d.anchor
        st = d.__dict__.get("_state", "open")
        ax.plot(ay, at, "o", ms=12, mfc="none",
                mec="#1d3557" if st == "open" else "#9d0208", mew=2.6)
        if st != "open":
            ax.plot(ay, at, "x", ms=10, mec="#9d0208", mew=2.6)
    if path is not None:
        n = len(path) if upto is None else upto + 1
        ax.plot(path[:n, 0], path[:n, 1], color="#e63946", lw=3)
        ax.scatter(path[n - 1, 0], path[n - 1, 1], color="#e63946", s=80,
                   edgecolor="#111", zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(pm.theta_grid[0], pm.theta_grid[-1])
    ax.set_xlabel("y  (weld coordinate)"); ax.set_ylabel(r"$\theta$  (arm angle)")
    ax.set_title(title, fontsize=11)


# ------------------------------------------------------------ figures ----

PAGE_NOTE = (
    "Each panel is ONE IK branch of the same task. The triple is "
    "(s2, s4, s6) = the signs of joints 2, 4, 6 — shoulder, elbow and wrist. "
    "All eight put the tool on the same weld point; they differ only in joint "
    "values, so they differ by JOINT LIMITS and by which link rolls where.\n"
    "Axes: y = position along the weld (0 = start, 1 = goal), "
    r"$\theta$ = arm angle, the one redundant freedom. "
    "A point is one complete robot posture."
)


def fig8(pms, sigma, walls, title, note, out_path, path=None):
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    for ax, n in zip(axes.ravel(), PAGES):
        pm = pms[n]
        w = walls.get(n)
        draw_page(ax, pm, path=path if n == START_PAGE else None,
                  wall=w, discs=sigma.get(n, []),
                  title="")
        role = ""
        if n == START_PAGE:
            role = "   ← START PAGE"
        elif n == PARTNER_PAGE:
            role = "   ← partner across the s2 door"
        ax.set_title(f"({n})  valid {pm.stats['frac_valid'] * 100:.1f}%"
                     f"{'   WALL' if w else ''}{role}", fontsize=11,
                     fontweight="bold" if n in (START_PAGE, PARTNER_PAGE)
                     else "normal")
        if n == START_PAGE:
            ax.scatter([0.0], [0.0], marker="*", s=240, color="#ffffff",
                       edgecolor="#111", zorder=8)
            ax.text(0.02, 0.15, "start", fontsize=9)
            ax.axvline(1.0, color="#111", lw=2, ls=":")
            ax.text(0.80, -2.9, "goal y=1", fontsize=9)
    handles = [
        mpatches.Patch(color="#2a9d8f", label="valid posture"),
        mpatches.Patch(color="#e9c46a", label="invalid: joint limit"),
        mpatches.Patch(color="#e76f51", label="invalid: collision"),
        mpatches.Patch(color="#e9e9e9", label="invalid: no IK"),
        plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#1d3557",
                   mew=2.6, ms=12, label=r"$\Sigma$ door OPEN"),
        plt.Line2D([], [], marker="x", ls="none", mec="#9d0208", mew=2.6,
                   ms=10, label=r"$\Sigma$ door BLOCKED"),
        plt.Line2D([], [], color="#9d0208", lw=3, label="learned wall (the proof)"),
        plt.Line2D([], [], color="#e63946", lw=3, label="robot motion"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(title + "\n" + note, fontsize=12)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_door_crossing(arm, weld, scene, pmA, pmB, wallA, wallB, discsA, discsB,
                      door, fiber, pathA, pathB, out_path, drift_mm=None,
                      pageA=START_PAGE, pageB=PARTNER_PAGE):
    """How the infeasibility proof gets from one page to the next.

    1. the start page. Wall #1 seals the start region everywhere EXCEPT at the
       door -- a wall may not cross a door (invariant 7), because a door is a
       real way out.
    2. the robot at the door, in space. Nothing here moves: measured over the
       whole sweep, every link point travels 0.000 mm and the tool 1e-13 mm.
    3. where the motion actually is -- joint space. q1 and q3 each slide 180
       deg in opposite directions. The product is a GIMBAL NULL-MOTION: with
       q2 = 0 the q1 and q3 axes are collinear, so the two rotations cancel
       exactly and the arm holds still while its coordinates travel. At
       delta = 180 deg the s2 sign has flipped, so the identical physical
       posture is now read on the partner page. Changing page costs no motion
       at all -- only joint range, which is why the torch tilt (a joint-limit
       effect) is what opens and closes these doors (findings 12, 30).
    4. the partner page, where the proof builds wall #2. Both walls together
       are the certificate.
    """
    from matplotlib.patches import ConnectionPatch

    fig = plt.figure(figsize=(22, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.12, 1.0),
                          height_ratios=(1.25, 1.0), wspace=0.28, hspace=0.42)
    axA = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    axq = fig.add_subplot(gs[1, 1])
    axB = fig.add_subplot(gs[:, 2])
    ay, at = door.anchor

    # ---- 1. start page, wall #1 -----------------------------------------
    draw_page(axA, pmA, path=pathA, wall=wallA, discs=discsA,
              title=f"1.  page ({pageA}):  wall #1 seals the start region")
    axA.scatter([0.0], [0.0], marker="*", s=260, color="#ffffff",
                edgecolor="#111", zorder=9)
    axA.text(0.03, 0.22, "start", fontsize=9, fontweight="bold")
    axA.annotate("OPEN DOOR\nthe wall is forbidden\nto cross it",
                 xy=(ay, at), xytext=(min(ay + 0.20, 0.52), at + 1.45),
                 fontsize=9, fontweight="bold", color="#1d3557",
                 ha="left", arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                            color="#1d3557"))

    # ---- 2. the robot at the door, in space: it does not move ------------
    p0, p1 = weld.p0, weld.p1
    ax3.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
             color="#e9c46a", lw=9, solid_capstyle="round", zorder=0)
    ax3.plot([0, 0], [0, 0], [0, 0.240], color="#888888", lw=10,
             solid_capstyle="round", zorder=0)
    for c, h, _l in scene.boxes:
        draw_box(ax3, c, h)
    n = len(fiber)
    step = max(1, n // 8)
    for idx in list(range(0, n, step)) + [n - 1]:
        f = arm.frames(fiber[idx])
        pts = [f["shoulder"], f["elbow"], f["wrist"], f["tool"]]
        for i, col in enumerate(("#1d3557", "#4895ef", "#e76f51")):
            u, v = pts[i], pts[i + 1]
            ax3.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], color=col,
                     lw=5.5, alpha=0.9, solid_capstyle="round")
    tip = arm.fk(fiber[0])[0]
    ax3.scatter(*tip, color="#e63946", s=190, depthshade=False,
                edgecolor="#111", zorder=10)
    ax3.set_xlim(-0.25, 0.60); ax3.set_ylim(-0.42, 0.42); ax3.set_zlim(0, 0.80)
    ax3.set_box_aspect((0.85, 0.84, 0.80))
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.view_init(elev=20, azim=-72)
    drift = ("" if drift_mm is None else f"tool {drift_mm:.0e} mm")
    ax3.set_title(f"2.  at the door, y={ay:.3f}: all {n} usable fiber postures "
                  f"are drawn\nand they COINCIDE -- every link point moves "
                  f"0.000 mm ({drift})", fontsize=10.5, pad=10)

    # ---- 3. where the motion is: joint space -----------------------------
    # Re-derive the FULL sweep so the panel can show the part that joint
    # limits exclude; `fiber` itself is already clipped to the usable part.
    q0 = np.asarray(fiber)[0].copy()
    full = []
    for d_ in np.linspace(0.0, np.pi, 181):
        qq = q0.copy(); qq[0] += d_; qq[2] -= d_
        full.append(qq)
    q = np.asarray(full)
    dlt = np.degrees(np.linspace(0.0, np.pi, len(q)))
    ok = np.array([arm.within_limits(qq) for qq in q])
    axq.plot(dlt, np.degrees(q[:, 0]), lw=2.6, color="#1d3557", label=r"$q_1$")
    axq.plot(dlt, np.degrees(q[:, 2]), lw=2.6, color="#e76f51", label=r"$q_3$")
    axq.plot(dlt, np.degrees(q[:, 0] + q[:, 2]), lw=2.0, ls="--",
             color="#2a9d8f", label=r"$q_1+q_3$  (constant)")
    bad = ~ok
    if bad.any():
        axq.fill_between(dlt, -200, 200, where=bad, color="#e9c46a",
                         alpha=0.55, step="mid",
                         label="outside joint limits")
    axq.set_xlim(0, 180); axq.set_ylim(-210, 210)
    axq.set_xlabel(r"self-motion parameter $\delta$  [deg]")
    axq.set_ylabel("joint angle [deg]")
    n_bad = int((~ok).sum())
    axq.set_title(r"3.  the motion is in JOINT space: $q_1$ and $q_3$ slide "
                  r"and cancel ($q_1+q_3$ fixed)" "\n"
                  + (r"the $s_2$ mirror at $\delta=180^\circ$ needs "
                     r"$|q_1|=180^\circ$ > the $178.85^\circ$ limit: door open "
                     r"only in the CONSERVATIVE sense"
                     if n_bad else
                     r"($q_2=0$ makes their axes collinear, so the arm holds "
                     r"still; at $\delta=180^\circ$ the $s_2$ sign flips)"),
                  fontsize=10.5)
    axq.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    axq.grid(alpha=0.25)

    # ---- 4. partner page, wall #2 ---------------------------------------
    if pmB is not None:
        draw_page(axB, pmB, path=pathB, wall=wallB, discs=discsB,
                  title=f"4.  page ({pageB}):  wall #2 seals it there too")
        axB.annotate("arrive here\nsame posture in space,\nflipped $s_2$",
                     xy=(ay, at), xytext=(min(ay + 0.20, 0.50), at - 2.1),
                     fontsize=9, fontweight="bold", color="#1d3557",
                     ha="left", arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                                color="#1d3557"))
    for (src, dst) in ((axA, ax3), (axq, axB)):
        try:
            con = ConnectionPatch(
                xyA=(1.03, 0.5), coordsA=src.transAxes,
                xyB=(-0.05, 0.5), coordsB=dst.transAxes,
                arrowstyle="-|>", mutation_scale=26, lw=2.6, color="#1d3557")
            fig.add_artist(con)
        except Exception:                 # 3-D axes transforms can refuse
            pass

    fig.suptitle(
        "HOW THE PROOF PASSES THROUGH THE DOOR\n"
        "A door is a posture where a sign vanishes, so ONE robot configuration is "
        "read by two pages at once.\nThe wall may not cross it, so the PROOF "
        "crosses it instead: it follows the robot onto the partner page\nand "
        "closes a second wall there.   Certificate = wall #1 + the door + wall #2.\n"
        "A door counts as open when ANY point of its fiber is usable -- the "
        "conservative direction, since assuming\na door might be passable can "
        "only make the proof do more work, never less.",
        fontsize=11.5, y=1.13)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")



# -------------------------------------------------------------- video ----

def render(segments, out_path, fps=18):
    """segments: list of dicts with keys
         cfg, arm, weld, scene, pm, q_list, path, upto_list, wall, discs,
         caption, page
    """
    fig = plt.figure(figsize=(14, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    writer = FFMpegWriter(fps=fps, bitrate=2600)
    with writer.saving(fig, str(out_path), dpi=100):
        for seg in segments:
            trail = []
            for idx, q in enumerate(seg["q_list"]):
                ax.clear(); ax2.clear()
                f = draw_arm(ax, seg["cfg"], seg["arm"], seg["scene"],
                             seg["weld"], q, trail)
                trail.append(f["tool"])
                ax.set_title(seg["caption"](idx), fontsize=11)
                draw_page(ax2, seg["pm"], path=seg.get("path"),
                          upto=seg["upto_list"][idx] if seg.get("upto_list") else None,
                          wall=seg.get("wall"), discs=seg.get("discs"),
                          title=f"page ({seg['page']})   coordinate space $Y^+$")
                fig.tight_layout()
                writer.grab_frame()
    plt.close(fig)
    print(f"[video] {out_path}")


def qs_from_path(arm, weld, path, signs):
    out, keep = [], []
    for i, (y, t) in enumerate(path):
        r = arm.ik_branch(*weld.pose(float(y)), float(t), signs)
        if r.ok:
            out.append(r.q); keep.append(i)
    return out, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--out", default="out/three")
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--only", default="123",
                    help="which experiments to run, e.g. 3 or 23 (default all)")
    ap.add_argument("--rest-mode", default="goal_only",
                    choices=("goal_only", "all_valid"),
                    help="which valid cells the wall must keep outside: "
                         "'goal_only' = IJRR's start-vs-goal split, "
                         "'all_valid' = every other component too")
    ap.add_argument("--label-mode", default="metric",
                    choices=("metric", "topological"),
                    help="how Algorithm 4 classes a fed-back facet centre: "
                         "'metric' = nearest class (the IJRR heuristic), "
                         "'topological' = which component it connects to")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    want = args.only
    global LABEL_MODE, REST_MODE
    LABEL_MODE = args.label_mode
    REST_MODE = args.rest_mode
    print(f"[cfg] rest class: {REST_MODE}")
    print(f"[cfg] Algorithm 4 feed-back labelling: {LABEL_MODE}")
    # keep the other experiments' entries when only some are re-run
    rpath = out / "three_report.json"
    report = json.loads(rpath.read_text()) if rpath.exists() else {}

    def pages_and_sigma(cfg, arm, weld, scene):
        pms = {n: build_page_map(cfg, arm, weld, scene,
                                 tuple(1 if c == "+" else -1 for c in n),
                                 with_kin=False) for n in PAGES}
        sig = build_sigma(cfg, arm, weld, pms, verbose=False)
        n_open = n_blk = 0
        for n in PAGES:
            sg = tuple(1 if c == "+" else -1 for c in n)
            for d in sig.get(n, []):
                if d.vanishing:
                    ok = door_fiber_validity(cfg, arm, weld, scene, d, sg)["ok"]
                    d.__dict__["_state"] = "open" if ok else "blocked"
                    n_open += ok; n_blk += (not ok)
        return pms, sig, n_open, n_blk

    if "1" in want:
        # ------------------------------- EXP 1 -------------------------------
        print("\n=== EXP 1 : no obstacle -> the robot FINISHES the weld ===")
        cfg, arm, weld, scene = make(0.0, None, args.res)
        pms, sig, no_, nb = pages_and_sigma(cfg, arm, weld, scene)
        pm = pms[START_PAGE]
        path = bfs_path(pm)
        qlist, keep = qs_from_path(arm, weld, path, START_SIGNS)
        print(f"[1] page ({START_PAGE}) valid {pm.stats['frac_valid']*100:.1f}%; "
              f"path {len(path)} waypoints reaching y={path[-1,0]:.2f}; "
              f"doors {no_} open / {nb} blocked")
        fig8(pms, sig, {}, "EXPERIMENT 1 — no obstacle: the robot FINISHES the weld",
             PAGE_NOTE, out / "exp1_pages.png", path=path)
        render([dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pm, q_list=qlist,
                     path=path, upto_list=keep, discs=sig[START_PAGE],
                     page=START_PAGE,
                     caption=lambda i, p=path, k=keep: (
                         f"EXP 1  no obstacle   weld "
                         f"{100*p[k[i],0]:.0f}% complete"))],
               out / "exp1_feasible.mp4")
        report["exp1"] = {"verdict": "feasible", "waypoints": len(path),
                          "doors_open": no_, "doors_blocked": nb}

    if "2" in want:
        # ------------------------------- EXP 2 -------------------------------
        print("\n=== EXP 2 : obstacle, doors CLOSED -> proof in ONE page ===")
        cfg2, arm2, weld2, scene2 = make(0.0, OBS_CLOSED, args.res)
        pms2, sig2, no2, nb2 = pages_and_sigma(cfg2, arm2, weld2, scene2)
        pm2 = pms2[START_PAGE]
        lab2, regs2 = label_regions(cfg2, pm2)
        k2 = sorted([r for r in regs2 if r.reaches_start and not r.reaches_goal],
                    key=lambda r: -r.n_cells)[0].index
        path2 = bfs_path(pm2, lab2, k2, to_goal=False)
        q2l, keep2 = qs_from_path(arm2, weld2, path2, START_SIGNS)
        poly2, rep2, live2, disch2 = prove(cfg2, arm2, weld2, scene2, pm2, lab2, k2,
                                           sig2[START_PAGE], iters=args.iters)
        ok2 = rep2.ok and poly2.closed
        print(f"[2] doors {no2} open / {nb2} blocked; robot stalls at y={path2[-1,0]:.2f}")
        print(f"[2] wall: {rep2.n_passed}/{rep2.n_facets} facets, closed={poly2.closed}, "
              f"{len(disch2)} disc(s) discharged -> "
              f"{'PROOF COMPLETE' if ok2 else 'not yet a proof'}")
        fig8(pms2, sig2, {START_PAGE: poly2.loops},
             "EXPERIMENT 2 — obstacle, doors CLOSED: the proof closes INSIDE ONE PAGE",
             PAGE_NOTE, out / "exp2_pages.png")
        render([dict(cfg=cfg2, arm=arm2, weld=weld2, scene=scene2, pm=pm2,
                     q_list=q2l, path=path2, upto_list=keep2, wall=poly2.loops,
                     discs=sig2[START_PAGE], page=START_PAGE,
                     caption=lambda i, p=path2, k=keep2: (
                         f"EXP 2  obstacle, doors CLOSED   "
                         f"blocked at y={p[k[i],0]:.2f}"))],
               out / "exp2_infeasible_onepage.mp4")
        report["exp2"] = {"verdict": "proven_one_page" if ok2 else "incomplete",
                          "facets": [rep2.n_passed, rep2.n_facets],
                          "closed": bool(poly2.closed),
                          "doors_open": no2, "doors_blocked": nb2,
                          "discs_discharged": len(disch2)}

    if "3" in want:
        # ------------------------------- EXP 3 -------------------------------
        print("\n=== EXP 3 : obstacle, door OPEN -> the proof CROSSES to a 2nd page ===")
        cfg3, arm3, weld3, scene3 = make(EXP3_TILT, OBS_OPEN, args.res, EXP3_WELD)
        pms3, sig3, no3, nb3 = pages_and_sigma(cfg3, arm3, weld3, scene3)
        seam3 = theta_seam_check(pms3, " exp3")
        pmA, pmB = pms3[START_PAGE], pms3[PARTNER_PAGE]
        labA, regsA = label_regions(cfg3, pmA)
        labB, regsB = label_regions(cfg3, pmB)
        kA = sorted([r for r in regsA if r.reaches_start and not r.reaches_goal],
                    key=lambda r: -r.n_cells)[0].index
        pathA = bfs_path(pmA, labA, kA, to_goal=False)
        qA, keepA = qs_from_path(arm3, weld3, pathA, START_SIGNS)

        # the open door out of the start region
        door = None
        ptsA = region_points(pmA, labA, kA)
        tolA = float(np.hypot(pmA.y_grid[1] - pmA.y_grid[0],
                              pmA.theta_grid[1] - pmA.theta_grid[0]))
        for d in sig3.get(START_PAGE, []):
            if not d.vanishing or d.__dict__.get("_state") != "open":
                continue
            ay, at = d.anchor
            if float(np.min(np.linalg.norm(ptsA - np.array([ay, at]), axis=1))) <= \
                    max(tolA, d.radius):
                door = d
                break
        print(f"[3] doors {no3} open / {nb3} blocked; "
              f"door out of the start region: "
              f"{'y=%.4f' % door.anchor[0] if door is not None else 'NONE'}")

        segs = [dict(cfg=cfg3, arm=arm3, weld=weld3, scene=scene3, pm=pmA,
                     q_list=qA, path=pathA, upto_list=keepA,
                     discs=sig3[START_PAGE], page=START_PAGE,
                     caption=lambda i, p=pathA, k=keepA: (
                         f"EXP 3  obstacle, door OPEN   "
                         f"page ({START_PAGE}) blocked at y={p[k[i],0]:.2f}"))]

        # the door crossing itself: a SELF-MOTION. q1 and q3 counter-rotate while
        # the tool stands still on the weld; at delta = pi we are on the s2 mirror,
        # i.e. the partner page. This segment exists only in Q -- in Y+ it is ONE
        # point, which is exactly where the [IROS] bijection fails.
        kB, fiber, drift, mirror = None, None, None, {}
        if door is not None:
            ay, at = door.anchor
            r0 = arm3.ik_branch(*weld3.pose(ay), at, START_SIGNS)
            if r0.ok:
                # Only the part of the fiber the robot can actually occupy.
                # The sweep is a GIMBAL NULL-MOTION -- q2 = 0 makes the q1 and
                # q3 axes collinear, so the two rotations cancel and the arm
                # does not move in space at all (measured: every link point
                # 0.000 mm, tool 1e-13 mm). What varies along it is only the
                # joint VALUES, so joint limits are the whole story, which is
                # why the torch tilt is the door switch (findings 12, 30).
                fiber, deltas = [], []
                for dlt in np.linspace(0.0, np.pi, 26):
                    q = r0.q.copy(); q[0] += dlt; q[2] -= dlt
                    if not arm3.within_limits(q):
                        continue
                    fiber.append(q); deltas.append(dlt)
                tip0 = arm3.fk(r0.q)[0]
                drift = max(float(np.linalg.norm(arm3.fk(q)[0] - tip0)) for q in fiber)
                print(f"[3] door crossing: {len(fiber)}-step self-motion over "
                      f"delta 0..{np.degrees(deltas[-1]):.1f} deg, tool drift "
                      f"{drift*1e3:.2e} mm (the arm does not move at all)")

                # The partner page's own representative of this same posture.
                # s2 flips it by exactly pi in q1, and every door in this task
                # sits at theta = 0 where q1 = 0, so the mirror lands at
                # |q1| = 180 deg -- 1.15 deg outside the real +/-178.85 limit.
                # The door is therefore OPEN only in the conservative sense
                # invariant 2 demands (its fiber has valid points, so it MIGHT
                # be passable and the proof must assume it is). Saying so is
                # the difference between a sound certificate and an overclaim.
                rp = arm3.ik_branch(*weld3.pose(ay), at, PARTNER_SIGNS)
                mirror = {"ik_ok": bool(rp.ok),
                          "within_limits": bool(rp.ok and arm3.within_limits(rp.q)),
                          "q_deg": ([round(float(v), 3)
                                     for v in np.degrees(rp.q)] if rp.ok else None),
                          "delta_valid_max_deg": float(np.degrees(deltas[-1]))}
                print(f"[3] partner-page representative of the SAME posture: "
                      f"q1={mirror['q_deg'][0] if mirror['q_deg'] else float('nan'):+.2f} "
                      f"deg, within_limits={mirror['within_limits']}"
                      + ("" if mirror["within_limits"] else
                         "  <- the s2 mirror is outside the q1 limit, so the "
                         "door is open only in the CONSERVATIVE sense "
                         "(invariant 2); the proof must still cross it"))
                segs.append(dict(cfg=cfg3, arm=arm3, weld=weld3, scene=scene3,
                                 pm=pmA, q_list=fiber, path=pathA,
                                 upto_list=[len(keepA) - 1] * len(fiber),
                                 discs=sig3[START_PAGE], page=START_PAGE,
                                 caption=lambda i, n=len(fiber): (
                                     "EXP 3  CROSSING THE DOOR: self-motion, "
                                     f"tool fixed  ({100*i/(n-1):.0f}%)")))
            i = int(np.argmin(np.abs(pmB.y_grid - ay)))
            j = int(np.argmin(np.abs(pmB.theta_grid - at)))
            sub = labB[max(0, i-2):i+3, max(0, j-2):j+3]
            ks = sorted({int(v) for v in np.unique(sub) if v > 0})
            kB = ks[0] if ks else None

        pathB, polyB, repB = None, None, None
        if kB is not None:
            pathB = bfs_path(pmB, labB, kB, to_goal=False)
            qB, keepB = qs_from_path(arm3, weld3, pathB,
                                     tuple(1 if c == "+" else -1 for c in PARTNER_PAGE))
            segs.append(dict(cfg=cfg3, arm=arm3, weld=weld3, scene=scene3, pm=pmB,
                             q_list=qB, path=pathB, upto_list=keepB,
                             discs=sig3[PARTNER_PAGE], page=PARTNER_PAGE,
                             caption=lambda i, p=pathB, k=keepB: (
                                 f"EXP 3  now on page ({PARTNER_PAGE})   "
                                 f"still blocked, at y={p[k[i],0]:.2f}")))

        print(f"[3] training wall #1 on ({START_PAGE}) region {kA} ...")
        polyA, repA, liveA, dischA = prove(cfg3, arm3, weld3, scene3, pmA, labA, kA,
                                           sig3[START_PAGE], iters=args.iters,
                                           verbose=True)
        print(f"[3] wall on ({START_PAGE}) region {kA}: {repA.n_passed}/{repA.n_facets} "
              f"facets, closed={polyA.closed}")
        walls3 = {START_PAGE: polyA.loops}
        if kB is not None:
            print(f"[3] training wall #2 on ({PARTNER_PAGE}) region {kB} ...")
            polyB, repB, liveB, dischB = prove(cfg3, arm3, weld3, scene3, pmB, labB,
                                               kB, sig3[PARTNER_PAGE],
                                               iters=args.iters, verbose=True)
            walls3[PARTNER_PAGE] = polyB.loops
            print(f"[3] wall on ({PARTNER_PAGE}) region {kB}: "
                  f"{repB.n_passed}/{repB.n_facets} facets, closed={polyB.closed}")

        # ---- the VERDICT, from the transitive page/door flood ------------------
        # The two walls above are what the figure and the video show. They are NOT
        # the proof: with more than one open door the certificate has to cover
        # every region reachable through every chain of doors, and only
        # flood.run_proof walks that graph. Printing closed=True off two walls
        # while a third region sits unvisited would be the same class of error as
        # calling this experiment "doors OPEN" while all 16 were shut.
        print("\n[3] --- verdict from flood.run_proof (all pages, all doors) ---")
        # Same training budget as prove() above, so the verdict is about the
        # same two walls the figure and the video show. run_proof's default of
        # 900 seeds puts twice as many cells in the fit, and on this region
        # that alone drives gamma from 1.3 to 323 (findings 29).
        res3 = run_proof(cfg3, arm3, weld3, scene3, pms3, sig3,
                         verbose=True, alg4_iters=args.iters,
                         seeds=[(START_PAGE, kA)],
                         n_train_max=450, n_invalid_max=150)
        print(f"[3] run_proof: proven={res3.proven} -- {res3.reason}")
        print(f"[3] regions visited {len(res3.regions_visited)}, "
              f"walls in certificate {len(res3.certificate)}, "
              f"doors followed {len(res3.doors_followed)}")
        for c in res3.certificate:
            print(f"[3]   wall on {c['page']}/{c['region']}: "
                  f"{c['n_passed']}/{c['n_facets']} facets, closed={c['closed']}, "
                  f"gamma={c['gamma']:.1f}, ok={c['ok']}"
                  + (f"  ({c['note']})" if c['note'] else ""))
        for d in res3.doors_followed:
            print(f"[3]   door followed: {d['from']} -> {d['to']} "
                  f"at y={d['y']:.4f}, th={d['theta']:+.4f}")
        write_certificate(res3, sig3, out / "exp3_certificate.json")

        # the video panel draws seg["wall"], so hand each segment the wall of
        # the page it plays on: the viewer sees the robot stall against wall #1,
        # cross the door, and stall again against wall #2.
        for seg in segs:
            seg["wall"] = walls3.get(seg["page"])

        fig8(pms3, sig3, walls3,
             "EXPERIMENT 3 — obstacle, door OPEN: the proof CROSSES a door and "
             "EXPANDS to a second page", PAGE_NOTE, out / "exp3_pages.png")
        if door is not None and fiber is not None:
            fig_door_crossing(arm3, weld3, scene3, pmA, pmB if kB is not None else None,
                              polyA.loops, polyB.loops if polyB else None,
                              sig3[START_PAGE], sig3[PARTNER_PAGE], door, fiber,
                              pathA, pathB, out / "exp3_door_crossing.png",
                              drift_mm=drift * 1e3 if drift is not None else None)
        render(segs, out / "exp3_infeasible_twopage.mp4")
        report["exp3"] = {
            "weld": dict(EXP3_WELD, tilt=EXP3_TILT),
            "doors_open": no3, "doors_blocked": nb3,
            "door_y": float(door.anchor[0]) if door is not None else None,
            "door_fiber_drift_mm": (drift * 1e3) if drift is not None else None,
            "pages_walled": list(walls3),
            "facetsA": [repA.n_passed, repA.n_facets], "closedA": bool(polyA.closed),
            "facetsB": ([repB.n_passed, repB.n_facets] if repB else None),
            "closedB": (bool(polyB.closed) if polyB else None),
            "theta_seam_touching_pages": [n for n, _, _ in seam3],
        "partner_representative": mirror,
            "verdict": {"proven": bool(res3.proven), "reason": res3.reason,
                        "regions_visited": len(res3.regions_visited),
                        "certificate": res3.certificate,
                        "doors_followed": res3.doors_followed}}

    rpath.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[out] {rpath}")


if __name__ == "__main__":
    main()
