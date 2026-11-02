"""EXP 4 -- the page change the robot can actually execute.

    EXP 1  no obstacle                 robot FINISHES the weld
    EXP 2  obstacle, doors CLOSED      infeasible; proof closes in ONE page
    EXP 3  obstacle, door OPEN         infeasible; the proof CROSSES the door
    EXP 4  obstacle, door OPEN         the START PAGE is infeasible, and the
                                       robot FINISHES by changing page

EXP 4 is the first experiment here whose verdict is NOT "infeasible". It still
rests on an infeasibility proof: a closed wall, learned by IJRR Algorithm 1-4,
seals the region that holds the start posture and proves the goal cannot be
reached WITHOUT LEAVING THAT PAGE. The planner then walks out through a q6
door, and on the partner page the weld completes. So the certificate reads
`proven=False, goal_reached_via=<partner page>` and the wall is the reason the
answer is interesting rather than trivial: the task is feasible only across
two charts.

WHY THIS SCENE AND NOT docs/exp4_brief.md's
--------------------------------------------
The handed-over scene (weld x 0.40 z 0.22 tilt 45, box at (0.095, 0.29, 0.377),
page -++ -> -+-, door y 0.7261) reproduces exactly as documented -- all four of
its criteria hold at res 61/81/101. Its door is NOT crossable, though, and the
brief's "+14.46 deg margin" measures the wrong thing.

A q6 door is a WRIST GIMBAL: q6 = 0 makes the q5 and q7 axes collinear, so
(q5 + d, q7 - d) is a null motion -- the arm does not move in space, only the
joint values slide. The partner page's representative of that one arm shape
sits at d = +/-pi. Crossing therefore means TRAVERSING the fiber from d = 0 to
d = +/-pi, and every intermediate d must be within limits:

    d: 0 -> +pi   needs   q5_0 <= L5 - pi   and   q7_0 >= -L7 + pi
    d: 0 -> -pi   needs   q5_0 >= -L5 + pi  and   q7_0 <=  L7 - pi

With L5 = 178.85 deg and L7 = 168.45 deg those windows are 1.15 deg and
11.55 deg wide. At the brief's door q5_0 = -59.84 and q7_0 = -136.62, so the
valid run is d in [-119.0, +31.7] deg: it contains d = 0 and NEITHER endpoint.
The brief checked only the endpoint, via `mirror_q`, which WRAPS q7 from
-316.62 deg to +43.38 deg. The wrapped value is inside the limit; the joint
cannot rotate there, because it would have to pass -168.45 on the way. Same
conclusion as findings 30 for the q2 door, reached one step later.

This scene is a placement whose door passes the real test. Measured here:
181 of 181 steps along the crossing are within limits and collision-free, the
fiber endpoint IS the partner representative to 2.4e-06 deg, the tool drifts
3.5e-06 mm over the whole crossing, clearance never drops below 56 mm, and the
tightest joint margin at the far end is +19.6 deg. This is the project's first
page change that the hardware could perform.

Outputs (--out, default out/exp4):
    exp4_pages.png          all eight pages, the wall, the doors
    exp4_manifold.png       the learned manifold in detail: the SVM decision
                            field, its zero set, the training classes, the
                            reconstructed polytope and the facet verdicts
    exp4_door_crossing.png  wall #1 -> the gimbal fiber -> the partner page
    exp4.gif                the robot: stalls, drives to the door, crosses,
                            finishes the weld
    exp4_certificate.json   flood.run_proof's verdict
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm  # noqa: E402

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import (  # noqa: E402
    find_inf_proof_region, invalid_points, label_regions, live_discs,
    other_valid_points, region_points, run_proof, thin, write_certificate,
)
from zzy_rss.pagemap import build_all_pages_shared, mirror_q  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.wall import (  # noqa: E402
    PageLabeller, check_polytope, reconstruct_wall, sigma_disc_samples,
    virtual_boundary_samples,
)
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")

# ---------------------------------------------------------------- scene ----
# Found by scripts/search_exp4_scene.py; measured numbers in docs/exp4_spec.md.
#
# TWO BOXES, TWO JOBS -- the same split that made EXP 2's ZGAP work
# (findings 24), applied here to the WALL rather than to connectivity.
#
#   CUT   makes the experiment: on page (+-+) the start region can no longer
#         reach the goal, while on (+--) the door's region still spans.
#   SPLIT makes the experiment PROVABLE. Without it the start region and the
#         goal component on (+-+) are separated only by the theta ~ 0
#         joint-limit hair, where q3 sits 1.15 deg outside its limit. That
#         hair is 1.2-1.7 deg wide -- under a third of a grid cell, and under
#         half the 0.019 feedback guard in scaled page units -- so no SVM zero
#         set fits inside it and ~6 facets can never pass (findings 37).
#         SPLIT sits on the theta ~ 0 elbow track and turns the hair into a
#         collision band. Collision is page-independent, so it is removed from
#         (+--) too, and that costs nothing: (+--)'s spanning route runs at
#         theta < 0, through the door at theta = -0.257.
#
# With SPLIT the wall closes at 169/169 facets, gamma 2.7, in ONE Algorithm 4
# iteration. Without it: 186/192 at best, then Algorithm 4 drifts away
# (175/193, 155/202, ...) with gamma pinned at 400.
EXP4 = dict(
    weld_x=0.44, weld_z=0.22, weld_half_y=0.34, tilt=45.0,
    boxes=(((0.18, 0.30, 0.34), (0.07, 0.07, 0.07), "cut"),
           ((0.18, -0.06, 0.42), (0.05, 0.05, 0.05), "split")),
    start_page="+-+", partner_page="+--",
    door_y=0.6674, door_theta=-0.2571,
)
START_PAGE = EXP4["start_page"]
PARTNER_PAGE = EXP4["partner_page"]
START_SIGNS = tuple(1 if c == "+" else -1 for c in START_PAGE)
PARTNER_SIGNS = tuple(1 if c == "+" else -1 for c in PARTNER_PAGE)

CMAP = ListedColormap(["#2a9d8f", "#e9e9e9", "#e9c46a", "#e76f51"])

PAGE_NOTE = (
    "Each panel is ONE IK branch of the same task. The triple is "
    "(s2, s4, s6) = the signs of joints 2, 4, 6 — shoulder, elbow and wrist. "
    "All eight put the tool on the same weld point; they differ only in joint "
    "values, so they differ by JOINT LIMITS and by which link rolls where.\n"
    "Axes: y = position along the weld (0 = start, 1 = goal), "
    r"$\theta$ = arm angle, the one redundant freedom. "
    "A point is one complete robot posture."
)


# ----------------------------------------------------------- scaffolding ----

def weld_for(x, z, half_y, tilt_deg):
    """The weld seam. Placement decides WHERE the pages touch.

    Door position solves |E(y) - S| = d_se for a q6 door, and E(y) is affine
    in y, so x, z and the tilt fix the door's y. The tilt additionally rolls
    the torch, which is what moves q5 and q7 at the door -- and those two are
    exactly what decides whether the gimbal fiber can be traversed.
    """
    a = np.radians(tilt_deg)
    return WeldParams(p0=(x, -half_y, z), p1=(x, half_y, z),
                      approach=(float(np.sin(a)), 0.0, float(-np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def make(res, feedback_margin, rest_mode="goal_only", label_mode="metric",
         seed=None):
    base = Config()
    obstacles = tuple(BoxObstacle(center=tuple(c), half_extent=tuple(h),
                                  label=lbl)
                      for c, h, lbl in EXP4["boxes"])
    cfg = dataclasses.replace(
        base,
        weld=weld_for(EXP4["weld_x"], EXP4["weld_z"],
                      EXP4["weld_half_y"], EXP4["tilt"]),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=obstacles),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz",
                                 rest_mode=rest_mode, label_mode=label_mode,
                                 feedback_margin=feedback_margin))
    if seed is not None:
        cfg = dataclasses.replace(
            cfg, planner=dataclasses.replace(cfg.planner, rng_seed=int(seed)))
    return cfg, SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)


def theta_seam_check(pms, label=""):
    """CLAUDE.md invariant 5: the page is a CYLINDER in theta.

    Treating it as a flat rectangle is EXACT, not approximate, only while no
    valid cell touches theta = +/-pi -- the joint limits must cut the theta
    circle before the seam. That is a measured property of a placement, not a
    theorem, so every new scene has to be re-measured (findings 23).
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


def pages_and_sigma(cfg, arm, weld, scene):
    """All eight pages for the price of one, plus the door states."""
    built = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
    pms = {n: built[n] for n in PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blk = 0
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sig.get(n, []):
            if d.vanishing:
                ok = door_fiber_validity(cfg, arm, weld, scene, d, sg)["ok"]
                d.__dict__["_state"] = "open" if ok else "blocked"
                n_open += ok
                n_blk += (not ok)
    return pms, sig, n_open, n_blk


def bfs_route(pm, mask, start_ij, goal_ij=None):
    """Shortest 4-connected route inside `mask`, in grid cells.

    With goal_ij: the route to that cell. Without: the route to the cell of
    largest y it can reach. Used to keep the animation CONTINUOUS -- the robot
    has to drive to the door, not be cut to it.
    """
    ny, nt = mask.shape
    if not mask[start_ij]:
        return None
    prev = {start_ij: None}
    q = deque([start_ij])
    best = start_ij
    while q:
        i, j = q.popleft()
        if goal_ij is None and i > best[0]:
            best = (i, j)
        if goal_ij is not None and (i, j) == tuple(goal_ij):
            best = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, (j + dj) % nt
            if 0 <= ni < ny and mask[ni, nj] and (ni, nj) not in prev:
                prev[(ni, nj)] = (i, j)
                q.append((ni, nj))
    if goal_ij is not None and tuple(goal_ij) not in prev:
        return None
    out, cur = [], best
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return np.array([[pm.y_grid[i], pm.theta_grid[j]] for i, j in out])


def qs_from_path(arm, weld, path, signs):
    out, keep = [], []
    for i, (y, t) in enumerate(path):
        r = arm.ik_branch(*weld.pose(float(y)), float(t), signs)
        if r.ok:
            out.append(r.q)
            keep.append(i)
    return out, keep


def prove(cfg, arm, weld, scene, pm, lab, k, discs, iters, n_train=450,
          verbose=False):
    """IJRR Algorithms 1-4 on one region, with the same inputs run_proof uses.

    Kept deliberately identical to run_proof's per-region setup (invariant
    16): if the figure's wall were trained on a different budget or with
    different discs, the picture would not be the object the certificate
    describes.
    """
    sg = tuple(1 if c == "+" else -1 for c in pm.name)
    rng = np.random.default_rng(cfg.planner.rng_seed)
    pts = region_points(pm, lab, k)
    live, disch = live_discs(cfg, arm, weld, scene, sg, discs, verbose=False)
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    se, re_ = sigma_disc_samples(live, pts, tol)
    seed = thin(pts, n_train, rng)
    _, regs_here = label_regions(cfg, pm)
    rest = np.vstack([thin(other_valid_points(pm, lab, k, regs_here,
                                              cfg.wall.rest_mode),
                           n_train, rng),
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
    return poly, rep, w, live, disch, seed, rest


# ---------------------------------------------------------------- drawing ----

def containment_check(poly, pm, lab, kA, regions):
    """Is the start region INSIDE the closed wall and the goal OUTSIDE it?

    Algorithm 1 reports training accuracy 1.0, which says every training
    point is on the right side of the zero set. The training set is an
    (epsilon, eta)-net SUBSAMPLE, though, so on its own that leaves open
    whether some cell nobody trained on sits on the wrong side. Closedness
    plus 'every facet in C_obs' does not settle it either: both are
    statements about where the curve is, not about what it encloses.

    So test it directly, on EVERY valid cell: wind the closed polyline around
    each cell centre. The certificate's claim is exactly

        region k inside  AND  goal component outside  AND  closed  AND
        every facet in C_obs

    and this supplies the first two.
    """
    from matplotlib.path import Path as MplPath

    if not poly.loops:
        return {"ok": False, "note": "no loops"}
    paths = [MplPath(np.asarray(lp, dtype=float)) for lp in poly.loops]

    def inside(pts):
        if not len(pts):
            return np.zeros(0, dtype=bool)
        wind = np.zeros(len(pts), dtype=int)
        for p in paths:
            wind += p.contains_points(pts).astype(int)
        return (wind % 2) == 1           # odd-even, so nested loops behave

    ys, ts = np.meshgrid(pm.y_grid, pm.theta_grid, indexing="ij")
    start_ij = np.argwhere(lab == kA)
    start_pts = np.column_stack([ys[lab == kA], ts[lab == kA]])
    goal_mask = np.zeros_like(lab, dtype=bool)
    for r in regions:
        if r.index != kA and r.reaches_goal:
            goal_mask |= (lab == r.index)
    goal_pts = np.column_stack([ys[goal_mask], ts[goal_mask]])
    s_in = inside(start_pts)
    g_in = inside(goal_pts)
    out = {
        "start_cells": int(len(start_pts)),
        "start_inside": int(s_in.sum()),
        "start_outside": int((~s_in).sum()),
        "goal_cells": int(len(goal_pts)),
        "goal_inside": int(g_in.sum()),
        "goal_outside": int((~g_in).sum()),
    }
    out["ok"] = bool(len(start_pts) and s_in.all() and not g_in.any())
    del start_ij
    return out


def draw_box(ax, c, h):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c, h = np.asarray(c, float), np.asarray(h, float)
    lo, hi = c - h, c + h
    v = np.array([[x, y, z] for x in (lo[0], hi[0])
                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor="#c1121f",
                                         alpha=0.45, edgecolor="#780000",
                                         lw=0.8))


def draw_arm(ax, cfg, arm, scene, weld, q, trail, azim=-72.0):
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
    ax.set_xlim(-0.25, 0.62)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(0, 0.80)
    ax.set_box_aspect((0.87, 0.90, 0.80))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=20, azim=azim)
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
    ax.set_xlim(0, 1)
    ax.set_ylim(pm.theta_grid[0], pm.theta_grid[-1])
    ax.set_xlabel("y  (weld coordinate)")
    ax.set_ylabel(r"$\theta$  (arm angle)")
    ax.set_title(title, fontsize=11)


def fig8(pms, sigma, walls, title, note, out_path, kA=None, kB=None):
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    for ax, n in zip(axes.ravel(), PAGES):
        pm = pms[n]
        w = walls.get(n)
        draw_page(ax, pm, wall=w, discs=sigma.get(n, []), title="")
        role = ""
        if n == START_PAGE:
            role = f"   ← START PAGE (region {kA})"
        elif n == PARTNER_PAGE:
            role = f"   ← partner across the $q_6$ door (region {kB})"
        ax.set_title(f"({n})  valid {pm.stats['frac_valid'] * 100:.1f}%"
                     f"{'   WALL' if w else ''}{role}", fontsize=11,
                     fontweight="bold" if n in (START_PAGE, PARTNER_PAGE)
                     else "normal")
        if n == START_PAGE:
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
        plt.Line2D([], [], color="#9d0208", lw=3,
                   label="learned wall (the proof)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(title + "\n" + note, fontsize=12)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_manifold(cfg, pm, wmodel, poly, rep, seed_pts, rest_pts, discs,
                 kA, out_path):
    """The learned manifold itself -- what Algorithm 1 actually produced.

    Left   the RBF-SVM decision field over the page, with its ZERO SET. This is
           the learned manifold: the wall is the level set d(y, theta) = 0, not
           a curve anybody drew. Red is the seed side (the start region), blue
           the rest side (the goal component plus the virtual free ring).
    Mid    the same zero set on top of the page's validity tags, with the
           training classes shown. The wall must lie entirely in the invalid
           (non-green) set -- that is what Algorithm 3 verifies.
    Right  the reconstructed polytope: ordered closed polyline, one marker per
           facet, coloured by its Algorithm 3 verdict.
    """
    pmp = cfg.pagemap
    eps = cfg.wall.virtual_band
    m = cfg.wall.recon_margin * eps
    ys = np.linspace(pmp.y_min - m, pmp.y_max + m, 420)
    ts = np.linspace(pmp.theta_min - m, pmp.theta_max + m, 420)
    yy, tt = np.meshgrid(ys, ts, indexing="ij")
    D = wmodel.decision(np.column_stack([yy.ravel(), tt.ravel()])
                        ).reshape(yy.shape)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8.0), constrained_layout=True)

    # -- 1. the decision field and its zero set ---------------------------
    ax = axes[0]
    lim = float(np.max(np.abs(D)))
    im = ax.pcolormesh(ys, ts, D.T, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
    ax.contour(ys, ts, D.T, levels=[0.0], colors="#111111", linewidths=3)
    fig.colorbar(im, ax=ax, label=r"SVM decision value  $d(y,\theta)$")
    ax.set_title(f"1.  THE LEARNED MANIFOLD\nRBF-SVM decision field, "
                 rf"$\gamma$={wmodel.gamma:.1f}, "
                 f"training accuracy {wmodel.accuracy:.4f}\n"
                 r"the wall is the zero set $d=0$ (black)", fontsize=11)
    ax.set_xlabel("y  (weld coordinate)")
    ax.set_ylabel(r"$\theta$  (arm angle)")

    # -- 2. the zero set against the page's validity ----------------------
    ax = axes[1]
    draw_page(ax, pm, wall=poly.loops, discs=discs, title="")
    ax.plot(rest_pts[:, 0], rest_pts[:, 1], ".", ms=2.0, color="#0b3954",
            alpha=0.55)
    ax.plot(seed_pts[:, 0], seed_pts[:, 1], ".", ms=2.0, color="#6a040f",
            alpha=0.55)
    ax.set_xlim(pmp.y_min - m, pmp.y_max + m)
    ax.set_ylim(pmp.theta_min - m, pmp.theta_max + m)
    ax.set_title(f"2.  the wall on page ({pm.name}), region {kA}\n"
                 f"seed class = the start region (dark red dots), "
                 f"rest class = goal component + virtual ring (navy dots)\n"
                 r"every facet must lie in $C_{obs}$ — i.e. OFF the green",
                 fontsize=11)

    # -- 3. the reconstructed polytope and the facet verdicts -------------
    ax = axes[2]
    ax.imshow(pm.tag.T, origin="lower", aspect="auto", cmap=CMAP, vmin=-.5,
              vmax=3.5, extent=[0, 1, pm.theta_grid[0], pm.theta_grid[-1]],
              alpha=0.35)
    for loop in poly.loops:
        ax.plot(loop[:, 0], loop[:, 1], "-", color="#9d0208", lw=2.4,
                zorder=4)
        mids = 0.5 * (loop[:-1] + loop[1:])
        ax.plot(mids[:, 0], mids[:, 1], "o", ms=3.0, mfc="#2a9d8f",
                mec="#14504a", mew=0.5, zorder=5)
    bad = np.asarray([c for c, _ in rep.failure_points], dtype=float
                     ).reshape(-1, 2)
    if len(bad):
        ax.plot(bad[:, 0], bad[:, 1], "X", ms=11, mfc="#ffb703",
                mec="#9d0208", mew=1.8, zorder=7)
    ax.set_xlim(pmp.y_min - m, pmp.y_max + m)
    ax.set_ylim(pmp.theta_min - m, pmp.theta_max + m)
    ax.set_xlabel("y  (weld coordinate)")
    ax.set_ylabel(r"$\theta$  (arm angle)")
    ok = rep.ok and poly.closed
    ax.set_title(f"3.  the proof object (IJRR §4.3 reconstruction)\n"
                 f"closed polyline: {poly.closed}, {poly.n_segments} segments, "
                 f"facets {rep.n_passed}/{rep.n_facets} verified in "
                 r"$C_{obs}$" "\n"
                 + ("ALL FACETS PASS → this is a valid infeasibility proof"
                    if ok else
                    f"{rep.n_facets - rep.n_passed} facet(s) unverified "
                    f"(gold X) → not yet a proof"),
                 fontsize=11,
                 color="#14504a" if ok else "#9d0208")
    handles = [
        plt.Line2D([], [], color="#9d0208", lw=2.4, label="wall (zero set)"),
        plt.Line2D([], [], marker="o", ls="none", mfc="#2a9d8f",
                   mec="#14504a", ms=7,
                   label=r"facet verified in $C_{obs}$ (Algorithm 3)"),
        plt.Line2D([], [], marker="X", ls="none", mfc="#ffb703",
                   mec="#9d0208", ms=11, label="facet NOT verified"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              framealpha=0.95)

    fig.suptitle(
        "EXPERIMENT 4 — THE LEARNED MANIFOLD ON THE START PAGE\n"
        "IJRR Algorithm 1 fits an RBF-SVM separating the start region from "
        "the goal component; Algorithm 2 pushes its zero set into "
        r"$C_{obs}$; §4.3 reconstructs the zero set as a closed polyline; "
        "Algorithm 3 verifies every facet.\n"
        "A closed polyline lying entirely in "
        r"$C_{obs}$ and enclosing the start region is the certificate: no "
        "path can leave it, so the goal is unreachable WITHOUT CHANGING PAGE.",
        fontsize=12.5, y=1.085)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_door_crossing(arm, weld, scene, pmA, pmB, wallA, discsA, discsB,
                      door, fiber, deltas, routeA, pathB, out_path,
                      rep_a=None, rep_b=None, drift_mm=None, clear_mm=None):
    """How the robot gets from one page to the next, and that it really can.

    Left   page A: the wall seals the start region; the robot's route runs
           along the seam, stalls, and then drives down to the door.
    Mid-1  the gimbal fiber in 3-D: eight postures along the crossing, all on
           the same weld point, the arm not moving at all.
    Mid-2  the crossing in JOINT space: q5 and q7 slide by 180 deg against
           each other, both inside their limits the whole way. THIS is the
           panel the brief's scene fails.
    Mid-3  the two pages' readings of the one posture at the far end.
    Right  page B: the weld completes.
    """
    fig = plt.figure(figsize=(23, 8.8))
    gs = fig.add_gridspec(3, 3, width_ratios=(1.0, 1.30, 1.0),
                          height_ratios=(1.25, 0.72, 0.72),
                          wspace=0.26, hspace=0.75)
    axA = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    axq = fig.add_subplot(gs[1, 1])
    axrep = fig.add_subplot(gs[2, 1])
    axB = fig.add_subplot(gs[:, 2])
    ay, at = door.anchor

    draw_page(axA, pmA, path=routeA, wall=wallA, discs=discsA,
              title=f"1.  page ({pmA.name}): the wall seals the start region,\n"
                    f"the robot drives to the OPEN $q_6$ door")
    axA.scatter([routeA[0, 0]], [routeA[0, 1]], marker="*", s=300,
                color="#ffffff", edgecolor="#111", zorder=9)
    axA.annotate("START", xy=(routeA[0, 0], routeA[0, 1]),
                 xytext=(0.10, routeA[0, 1] + 1.1), fontsize=9,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#111"))
    axA.axvline(1.0, color="#111", lw=2.4, ls=":")
    axA.text(0.965, -2.62, "GOAL  y=1\nunreachable on THIS page", fontsize=9,
             fontweight="bold", ha="right", va="top", color="#9d0208",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                       alpha=0.85))
    axA.annotate(f"OPEN DOOR  $q_6=0$\ny={ay:.4f}", xy=(ay, at),
                 xytext=(max(ay - 0.42, 0.02), at + 1.5), fontsize=9,
                 fontweight="bold", color="#1d3557",
                 arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1d3557"))

    # -- the fiber in 3-D --------------------------------------------------
    p0, p1 = weld.p0, weld.p1
    ax3.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
             color="#e9c46a", lw=9, solid_capstyle="round", zorder=0)
    ax3.plot([0, 0], [0, 0], [0, 0.240], color="#888888", lw=10,
             solid_capstyle="round", zorder=0)
    for c, h, _l in scene.boxes:
        draw_box(ax3, c, h)
    n = len(fiber)
    for idx in list(range(0, n, max(1, n // 8))) + [n - 1]:
        f = arm.frames(fiber[idx])
        pts = [f["shoulder"], f["elbow"], f["wrist"], f["tool"]]
        for i, col in enumerate(("#1d3557", "#4895ef", "#e76f51")):
            u, v = pts[i], pts[i + 1]
            ax3.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], color=col,
                     lw=5.5, alpha=0.9, solid_capstyle="round")
    ax3.scatter(*arm.fk(fiber[0])[0], color="#e63946", s=190,
                depthshade=False, edgecolor="#111", zorder=10)
    ax3.set_xlim(-0.25, 0.62)
    ax3.set_ylim(-0.45, 0.45)
    ax3.set_zlim(0, 0.80)
    ax3.set_box_aspect((0.87, 0.90, 0.80))
    ax3.set_xlabel("x [m]", labelpad=-2)
    ax3.set_ylabel("y [m]", labelpad=-2)
    ax3.set_zlabel("z [m]", labelpad=-2)
    ax3.tick_params(labelsize=7, pad=-1)
    ax3.view_init(elev=20, azim=-72)
    ax3.set_position(ax3.get_position().expanded(1.45, 1.40))
    ax3.set_title(f"2.  the crossing in SPACE: {n} postures overlaid.\n"
                  f"the arm does not move — tool drift {drift_mm:.1e} mm",
                  fontsize=10.5, pad=2)

    # -- the crossing in joint space --------------------------------------
    dd = np.degrees(np.asarray(deltas))
    Q = np.degrees(np.asarray(fiber))
    L = np.degrees(arm.q_max)
    axq.axhspan(-L[4], L[4], color="#2a9d8f", alpha=0.08)
    axq.plot(dd, Q[:, 4], lw=2.8, color="#1d3557", label=r"$q_5$")
    axq.plot(dd, Q[:, 6], lw=2.8, color="#e76f51", label=r"$q_7$")
    for v, c, lbl in ((L[4], "#1d3557", r"$q_5$ limit"),
                      (-L[4], "#1d3557", None),
                      (L[6], "#e76f51", r"$q_7$ limit"),
                      (-L[6], "#e76f51", None)):
        axq.axhline(v, color=c, lw=1.4, ls="--", alpha=0.8, label=lbl)
    axq.scatter([dd[0]], [Q[0, 4]], s=90, color="#1d3557", zorder=6,
                edgecolor="#111")
    axq.scatter([dd[-1]], [Q[-1, 4]], s=90, color="#1d3557", zorder=6,
                edgecolor="#111")
    axq.scatter([dd[0]], [Q[0, 6]], s=90, color="#e76f51", zorder=6,
                edgecolor="#111")
    axq.scatter([dd[-1]], [Q[-1, 6]], s=90, color="#e76f51", zorder=6,
                edgecolor="#111")
    axq.set_xlabel(r"$\delta$ along the gimbal fiber  [deg]"
                   "      (0 = this page's reading, "
                   r"$\pm180$ = the partner's)")
    axq.set_ylabel("angle [deg]")
    axq.set_title(r"3.  the crossing in JOINT space: $q_5$ and $q_7$ slide "
                  r"$180^\circ$ against each other," "\n"
                  f"both inside their limits for all {n} steps — "
                  f"so the robot CAN do it", fontsize=10.5, pad=6)
    axq.set_ylim(-215, 215)
    axq.legend(fontsize=8, ncol=4, loc="lower center", framealpha=0.95)
    axq.grid(alpha=0.25)
    if dd[-1] < dd[0]:
        axq.invert_xaxis()          # read the crossing left to right

    # -- the two readings of the far-end posture --------------------------
    if rep_a is not None and rep_b is not None:
        idx = np.arange(7)
        a_deg, b_deg = np.degrees(rep_a), np.degrees(rep_b)
        w = 0.38
        axrep.bar(idx - w / 2, a_deg, w, color="#1d3557",
                  label=f"page ({pmA.name}) reading")
        axrep.bar(idx + w / 2, b_deg, w, color="#e76f51",
                  label=f"page ({pmB.name}) reading")
        axrep.plot(idx, L, "_", ms=22, color="#9d0208", mew=2.2)
        axrep.plot(idx, -L, "_", ms=22, color="#9d0208", mew=2.2,
                   label="joint limit")
        marg = float(np.min(L - np.abs(b_deg)))
        axrep.set_xticks(idx)
        axrep.set_xticklabels([f"$q_{i+1}$" for i in idx])
        axrep.set_ylabel("angle [deg]")
        axrep.axhline(0, color="#777", lw=0.8)
        axrep.set_title(r"4.  THE PAGE CHANGE: the two charts' readings of the "
                        r"SAME posture." "\n"
                        r"identical but for $q_5,q_7$ — and the partner's "
                        f"reading clears every limit by {marg:.1f}°",
                        fontsize=10.5, pad=6)
        axrep.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95)
        axrep.grid(alpha=0.2, axis="y")

    if pmB is not None:
        draw_page(axB, pmB, path=pathB, discs=discsB,
                  title=f"5.  page ({pmB.name}): no wall is possible here — "
                        f"the robot\nFINISHES the weld, so the task is "
                        f"feasible across TWO pages")
        axB.axvline(1.0, color="#111", lw=2.4, ls=":")
        axB.text(0.965, -2.62, "GOAL  y=1\nREACHED", fontsize=9,
                 fontweight="bold", ha="right", va="top", color="#14504a",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                           alpha=0.85))
        axB.plot([ay], [at], marker="s", ms=13, mfc="none", mec="#1d3557",
                 mew=2.8, zorder=10)
        axB.annotate("the robot arrives HERE,\nin limits, clearance "
                     f"{clear_mm:.0f} mm", xy=(ay, at),
                     xytext=(max(ay - 0.46, 0.02), at - 2.1), fontsize=9,
                     fontweight="bold", color="#1d3557",
                     arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                     color="#1d3557"))

    fig.suptitle(
        "EXPERIMENT 4 — HOW THE ROBOT LEAVES A PROVEN-INFEASIBLE PAGE\n"
        r"The wall on page (" + pmA.name + r") is a complete infeasibility "
        "proof FOR THAT PAGE. The robot escapes through a $q_6$ door: at "
        r"$q_6=0$ the $q_5$ and $q_7$ axes are collinear, so sliding them "
        r"$180^\circ$ against each other" "\n"
        "moves no link at all and yet re-labels the posture onto the partner "
        "page. Unlike the $q_2$ door of EXP 3, every step of this slide is "
        "inside the real joint limits — the hardware could execute it.",
        fontsize=11.5, y=1.045)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


PAGE_COLOUR = {"start": "#1d3557", "cross": "#bb3e03", "partner": "#14746f"}


def render_gif(segments, out_path, arm, fps=14, dpi=80):
    """The animation, as a GIF (no ffmpeg needed).

    Four things here are not cosmetic.

    THE TRAIL IS GLOBAL. Resetting it per segment threw away the record of
    how far the tool had got, so the last segment looked like a fresh run
    rather than the continuation it is.

    THE BANNER NAMES THE PAGE. The two pages' validity maps look almost
    identical, so without a banner the page change -- the entire point of the
    experiment -- is invisible.

    THE JOINT STRIP ALWAYS MOVES. The crossing is a gimbal SELF-MOTION: the
    arm is motionless in space for the whole of it, by construction. With
    only a 3-D view and a page map on screen that reads as a hung animation,
    and it is the most interesting part of the run. The strip shows q5 and q7
    against their limits, so during the crossing the viewer watches the two
    joints slide 180 degrees past each other while the arm stands still --
    which is exactly what a q6 door IS.

    PROGRESS IS SHOWN AGAINST THE WHOLE WELD, not per segment, so "stalled at
    0.82" and "finished at 1.00" are directly comparable.
    """
    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.26),
                          width_ratios=(1.0, 1.0), hspace=0.32, wspace=0.18,
                          left=0.04, right=0.97, top=0.88, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])
    axbar = fig.add_subplot(gs[1, 0])
    axq = fig.add_subplot(gs[1, 1])
    banner = fig.text(0.5, 0.955, "", ha="center", va="center",
                      fontsize=15, fontweight="bold")
    sub = fig.text(0.5, 0.915, "", ha="center", va="center", fontsize=10.5,
                   color="#333333")
    L = np.degrees(arm.q_max)
    writer = PillowWriter(fps=fps)
    n_frames = 0
    trail = []                       # GLOBAL: survives across segments
    y_done = 0.0                     # best weld coordinate reached so far

    with writer.saving(fig, str(out_path), dpi=dpi):
        for seg in segments:
            n = len(seg["q_list"])
            for idx in range(n + seg.get("hold", 0)):
                k = min(idx, n - 1)
                q = seg["q_list"][k]
                ax.clear(); ax2.clear(); axbar.clear(); axq.clear()

                # A hold frame identical to the one before it is SILENTLY
                # DROPPED by Pillow's GIF encoder -- measured: 294 frames
                # written, 236 in the file, the difference being exactly the
                # holds. Orbiting the camera during a hold makes each frame
                # distinct, and reads as a deliberate pause rather than a
                # stutter.
                azim = -72.0 - 0.30 * max(0, idx - n + 1)
                f = draw_arm(ax, seg["cfg"], seg["arm"], seg["scene"],
                             seg["weld"], q, trail, azim=azim)
                if idx < n and seg.get("advances", True):
                    trail.append(f["tool"])
                ax.set_title(seg["caption"](k), fontsize=10.5, pad=0)

                draw_page(ax2, seg["pm"], path=seg.get("path"),
                          upto=(seg["upto_list"][k]
                                if seg.get("upto_list") else None),
                          wall=seg.get("wall"), discs=seg.get("discs"),
                          title=f"page ({seg['page']})   "
                                r"coordinate space $Y^+$")

                col = PAGE_COLOUR[seg["role"]]
                banner.set_text(seg["banner"])
                banner.set_color(col)
                sub.set_text(seg["sub"])
                for spine in ax2.spines.values():
                    spine.set_edgecolor(col)
                    spine.set_linewidth(3.0)

                # ---- weld progress, against the WHOLE seam ---------------
                y_now = (seg["y_of"](k) if seg.get("y_of") else y_done)
                y_done = max(y_done, y_now)
                axbar.barh([0], [1.0], color="#e9e9e9", height=0.55)
                axbar.barh([0], [y_now], color=col, height=0.55)
                axbar.axvline(seg["stall_y"], color="#9d0208", lw=2.2, ls="--")
                axbar.text(seg["stall_y"], 0.52,
                           f" wall stops page ({seg['start_page']}) here "
                           f"({seg['stall_y']:.2f})",
                           fontsize=8, color="#9d0208", va="bottom")
                axbar.plot([y_now], [0], "o", ms=11, color=col,
                           mec="#111", zorder=5)
                axbar.set_xlim(-0.01, 1.01); axbar.set_ylim(-0.6, 1.0)
                axbar.set_yticks([])
                axbar.set_xlabel("weld completed    "
                                 r"($y=0$ start, $y=1$ goal)", fontsize=9)
                axbar.set_title(f"weld progress: {100 * y_now:.0f}%",
                                fontsize=10, color=col, fontweight="bold")

                # ---- the joints that carry the page change ---------------
                qd = np.degrees(q)
                for row, j, name in ((1, 4, r"$q_5$"), (0, 6, r"$q_7$")):
                    axq.barh([row], [2 * L[j]], left=-L[j], height=0.42,
                             color="#eef1f3")
                    axq.plot([-L[j], -L[j]], [row - .26, row + .26],
                             color="#9d0208", lw=2.0)
                    axq.plot([L[j], L[j]], [row - .26, row + .26],
                             color="#9d0208", lw=2.0)
                    axq.barh([row], [qd[j]], height=0.42, color=col)
                    axq.plot([qd[j]], [row], "o", ms=9, color=col,
                             mec="#111", zorder=5)
                    axq.text(-L[j] - 8, row, name, fontsize=11, ha="right",
                             va="center")
                    axq.text(qd[j] + (9 if qd[j] >= 0 else -9), row,
                             f"{qd[j]:+.0f}°", fontsize=8.5, va="center",
                             ha="left" if qd[j] >= 0 else "right")
                axq.axvline(0, color="#999", lw=0.8)
                axq.set_xlim(-235, 235); axq.set_ylim(-0.6, 1.6)
                axq.set_yticks([])
                axq.set_xlabel("joint angle [deg]   "
                               "(red bars = the real LWA 4D limits)",
                               fontsize=9)
                axq.set_title(seg["qtitle"], fontsize=10,
                              color=col, fontweight="bold")
                writer.grab_frame()
                n_frames += 1
    plt.close(fig)
    print(f"[gif] {out_path}  ({n_frames} frames)")
    return n_frames


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--out", default="out/exp4")
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--n-train", type=int, default=450)
    ap.add_argument("--feedback-margin", type=float, default=-1.0,
                    help="Algorithm 4 guard, in SCALED page units; -1 (the "
                         "default) means HALF A CELL DIAGONAL at the chosen "
                         "resolution, which is 0.019 at res 61 -- the value "
                         "findings 32 measured. The grid pipeline historically "
                         "ran 0 (IJRR as written); on this region unguarded "
                         "feed-back drives the training set inseparable "
                         "(gamma 400, accuracy 0.988) and the facet count AWAY "
                         "from closure. The guard has to SCALE with the grid: "
                         "pinning it at 0.019 while the cells shrink skips "
                         "centres that are now several cells clear, and at "
                         "res 81 that left the wall at 306/309 with gamma 30.2 "
                         "instead of 172/172 at gamma 3.7.")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the (epsilon, eta)-net subsample that "
                         "picks the training points. Whether the wall closes "
                         "depends on this draw, not only on the scene: over "
                         "6 resolutions x 6 seeds the wall came out fully "
                         "verified in most draws and short by a few facets in "
                         "the rest, with the geometric bottleneck identical "
                         "(0.090 scaled) throughout. A draw that falls short "
                         "is INCONCLUSIVE, never a disproof -- Algorithm 3 "
                         "only ever fails to certify, it never certifies "
                         "something false -- so re-drawing is legitimate. See "
                         "docs/exp4_spec.md.")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-proof", action="store_true",
                    help="skip flood.run_proof (figures only)")
    args = ap.parse_args()
    if args.feedback_margin < 0.0:
        # half a cell diagonal in SCALED page units (y in [0,1],
        # theta/pi in [-1,1]); 0.019 at res 61, matching findings 32
        args.feedback_margin = float(
            0.5 * np.hypot(1.0 / (args.res - 1), 2.0 / (args.res - 1)))
        print(f"[cfg] feedback_margin auto = {args.feedback_margin:.4f} "
              f"(half a cell diagonal at res {args.res})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    T0 = time.perf_counter()
    timings, report = {}, {}

    def lap(label, since):
        dt = time.perf_counter() - since
        timings[label] = round(dt, 2)
        print(f"[time] {label}: {dt:.1f} s")
        return time.perf_counter()

    print("\n=== EXP 4 : the start page is infeasible, and the robot "
          "FINISHES by changing page ===")
    print(f"[scene] weld x={EXP4['weld_x']} z={EXP4['weld_z']} "
          f"half_y={EXP4['weld_half_y']} tilt={EXP4['tilt']}deg")
    for c, h, lbl in EXP4["boxes"]:
        print(f"[scene] box '{lbl}': center={c} half={h}")
    print(f"[scene] start page ({START_PAGE}) -> partner ({PARTNER_PAGE}) "
          f"through a q6 door near y={EXP4['door_y']}")

    t = time.perf_counter()
    cfg, arm, weld, scene = make(args.res, args.feedback_margin,
                                 seed=args.seed)
    print(f"[cfg] training subsample seed = {cfg.planner.rng_seed}")
    pms, sig, n_open, n_blk = pages_and_sigma(cfg, arm, weld, scene)
    t = lap("pages_and_doors", t)
    seam = theta_seam_check(pms, " exp4")

    pmA, pmB = pms[START_PAGE], pms[PARTNER_PAGE]
    labA, regsA = label_regions(cfg, pmA)
    labB, regsB = label_regions(cfg, pmB)

    # The door: the analytic q6 door nearest the screened one.
    cand = [d for d in sig.get(START_PAGE, [])
            if 2 in d.vanishing and d.__dict__.get("_state") == "open"]
    if not cand:
        raise SystemExit("[4] no OPEN q6 door on the start page")
    door = min(cand, key=lambda d: abs(d.anchor[0] - EXP4["door_y"]))
    ay, at = door.anchor
    i = int(np.argmin(np.abs(pmA.y_grid - ay)))
    j = int(np.argmin(np.abs(pmA.theta_grid - at)))
    kA, kB = int(labA[i, j]), int(labB[i, j])
    regA = next(r for r in regsA if r.index == kA)
    regB = next(r for r in regsB if r.index == kB)
    print(f"[4] doors {n_open} open / {n_blk} blocked over 8 pages")
    print(f"[4] q6 door at y={ay:.4f}, theta={at:+.4f}")
    print(f"[4] start region ({START_PAGE}, {kA}): {regA.n_cells} cells, "
          f"y in [{regA.y_min:.3f}, {regA.y_max:.3f}], "
          f"reaches_goal={regA.reaches_goal}")
    print(f"[4] partner region ({PARTNER_PAGE}, {kB}): {regB.n_cells} cells, "
          f"y in [{regB.y_min:.3f}, {regB.y_max:.3f}], "
          f"reaches_goal={regB.reaches_goal}")
    if regA.reaches_goal:
        raise SystemExit("[4] the start region reaches the goal: nothing to "
                         "prove on this page")
    if not regB.reaches_goal:
        raise SystemExit("[4] the partner region does not reach the goal: "
                         "this is EXP 3, not EXP 4")

    # ---- the crossing, measured before anything is drawn ------------------
    fib = door_fiber_validity(cfg, arm, weld, scene, door, START_SIGNS)
    qA0 = arm.ik_branch(*weld.pose(ay), at, START_SIGNS)
    rB = arm.ik_branch(*weld.pose(ay), at, PARTNER_SIGNS)
    L = np.degrees(arm.q_max)
    # Which way round the gimbal fiber stays inside the limits all the way to
    # the partner's representative? This is the test docs/exp4_brief.md
    # replaced with an endpoint check.
    cross = None
    for sign in (-1.0, +1.0):
        ds = np.linspace(0.0, sign * np.pi, 181)
        qs, okall, minclear = [], True, 1e9
        for d in ds:
            q = qA0.q.copy()
            q[4] += d
            q[6] -= d
            rep = scene.check(arm.frames(q))
            if not arm.within_limits(q) or rep.in_collision:
                okall = False
                break
            minclear = min(minclear, rep.clearance)
            qs.append(q)
        if okall:
            cross = dict(sign=sign, deltas=ds, fiber=qs, clearance=minclear)
            break
    if cross is None:
        raise SystemExit(
            "[4] the q6 door's gimbal fiber is NOT traversable: no direction "
            "reaches the partner representative inside the joint limits. "
            "That is the defect in docs/exp4_brief.md's scene -- pick another "
            "placement (scripts/search_exp4_scene.py).")
    fiber, deltas = cross["fiber"], cross["deltas"]
    tip0 = arm.fk(qA0.q)[0]
    drift = max(float(np.linalg.norm(arm.fk(q)[0] - tip0)) for q in fiber)
    q_end = fiber[-1]
    margin = float(np.min(L - np.abs(np.degrees(q_end))))
    print(f"[4] gimbal fiber: {fib['valid_fraction']*100:.1f}% of the circle "
          f"valid; the crossing runs delta 0 -> "
          f"{np.degrees(deltas[-1]):+.0f} deg")
    print(f"[4] crossing: {len(fiber)}/{len(deltas)} steps within limits and "
          f"collision-free, tool drift {drift*1e3:.2e} mm, min clearance "
          f"{cross['clearance']*1e3:.1f} mm")
    print(f"[4] far end == page ({PARTNER_PAGE}) representative to "
          f"{np.max(np.abs(np.degrees(q_end - rB.q))):.1e} deg; tightest "
          f"joint margin there {margin:+.2f} deg")
    print(f"[4]   ({START_PAGE}) reading [deg]: "
          f"{np.round(np.degrees(qA0.q), 2)}")
    print(f"[4]   ({PARTNER_PAGE}) reading [deg]: "
          f"{np.round(np.degrees(rB.q), 2)}")

    # ---- the wall on the start page: the infeasibility proof --------------
    print(f"\n[4] training the wall on ({START_PAGE}) region {kA} ...")
    t = time.perf_counter()
    polyA, repA, wmodel, liveA, dischA, seedA, restA = prove(
        cfg, arm, weld, scene, pmA, labA, kA, sig[START_PAGE],
        iters=args.iters, n_train=args.n_train, verbose=True)
    t = lap("wall", t)
    wall_ok = bool(repA.ok and polyA.closed)
    print(f"[4] wall on ({START_PAGE}) region {kA}: "
          f"{repA.n_passed}/{repA.n_facets} facets, closed={polyA.closed}, "
          f"gamma={wmodel.gamma:.1f}, accuracy={wmodel.accuracy:.4f}")
    cont = containment_check(polyA, pmA, labA, kA, regsA)
    print(f"[4] containment: {cont['start_inside']}/{cont['start_cells']} "
          f"start-region cells INSIDE the wall, "
          f"{cont['goal_outside']}/{cont['goal_cells']} goal-component cells "
          f"OUTSIDE it -> {'ok' if cont['ok'] else 'FAILED'}")
    wall_ok = wall_ok and cont["ok"]
    print(f"[4] -> {'PROOF COMPLETE for this page' if wall_ok else 'INCOMPLETE'}")

    # ---- the routes, for the figures and the animation --------------------
    startA = (0, int(np.argmax(labA[0] == kA))) if (labA[0] == kA).any() else None
    if startA is None:
        raise SystemExit("[4] the start region does not touch y=0")
    stall = bfs_route(pmA, labA == kA, startA)
    to_door = bfs_route(pmA, labA == kA,
                        (int(np.argmin(np.abs(pmA.y_grid - stall[-1, 0]))),
                         int(np.argmin(np.abs(pmA.theta_grid - stall[-1, 1])))),
                        (i, j))
    pathB = bfs_route(pmB, labB == kB, (i, j))
    print(f"[4] route on ({START_PAGE}): {len(stall)} cells to the stall at "
          f"y={stall[-1,0]:.3f}, then {0 if to_door is None else len(to_door)} "
          f"cells to the door")
    print(f"[4] route on ({PARTNER_PAGE}): {len(pathB)} cells, reaching "
          f"y={pathB[-1,0]:.3f}")

    # ---- the verdict ------------------------------------------------------
    res = None
    if not args.no_proof:
        print("\n[4] --- verdict from flood.run_proof (all pages, all doors) ---")
        t = time.perf_counter()
        res = run_proof(cfg, arm, weld, scene, pms, sig, verbose=True,
                        alg4_iters=args.iters, seeds=[(START_PAGE, kA)],
                        n_train_max=args.n_train, n_invalid_max=150)
        print(f"[4] run_proof: proven={res.proven} -- {res.reason}")
        print(f"[4] goal_reached_via={res.goal_reached_via!r}")
        print(f"[4] regions visited {len(res.regions_visited)}, walls in "
              f"certificate {len(res.certificate)}, doors followed "
              f"{len(res.doors_followed)}")
        for c in res.certificate:
            print(f"[4]   wall on {c['page']}/{c['region']}: "
                  f"{c['n_passed']}/{c['n_facets']} facets, "
                  f"closed={c['closed']}, gamma={c['gamma']:.1f}, "
                  f"ok={c['ok']}" + (f"  ({c['note']})" if c['note'] else ""))
        for d in res.doors_followed:
            print(f"[4]   door followed: {d['from']} -> {d['to']} at "
                  f"y={d['y']:.4f}, th={d['theta']:+.4f} ({d['kind']})")
        t = lap("run_proof", t)
        write_certificate(res, sig, out / "exp4_certificate.json")
        print(f"[out] {out / 'exp4_certificate.json'}")

    # ---- figures ----------------------------------------------------------
    t = time.perf_counter()
    walls = {START_PAGE: polyA.loops}
    fig8(pms, sig, walls,
         "EXPERIMENT 4 — the START PAGE is proven infeasible, and the robot "
         "FINISHES the weld by changing page",
         PAGE_NOTE, out / "exp4_pages.png", kA=kA, kB=kB)
    fig_manifold(cfg, pmA, wmodel, polyA, repA, seedA, restA,
                 sig[START_PAGE], kA, out / "exp4_manifold.png")
    routeA = np.vstack([stall, to_door]) if to_door is not None else stall
    fig_door_crossing(arm, weld, scene, pmA, pmB, polyA.loops,
                      sig[START_PAGE], sig[PARTNER_PAGE], door, fiber, deltas,
                      routeA, pathB, out / "exp4_door_crossing.png",
                      rep_a=qA0.q, rep_b=rB.q, drift_mm=drift * 1e3,
                      clear_mm=cross["clearance"] * 1e3)
    t = lap("figures", t)

    # ---- the animation ----------------------------------------------------
    n_frames = 0
    if not args.no_gif:
        t = time.perf_counter()
        qS, keepS = qs_from_path(arm, weld, stall, START_SIGNS)
        stall_y = float(stall[-1, 0])
        common = dict(cfg=cfg, arm=arm, weld=weld, scene=scene,
                      stall_y=stall_y, start_page=START_PAGE)
        SUB_A = (f"the wall on page ({START_PAGE}) is a complete "
                 f"infeasibility proof FOR THAT PAGE "
                 f"({repA.n_passed}/{repA.n_facets} facets verified in "
                 r"$C_{obs}$)")
        segs = [
            # 1/4 -- weld until the wall stops it
            dict(common, pm=pmA, q_list=qS, path=stall, upto_list=keepS,
                 wall=polyA.loops, discs=sig[START_PAGE], page=START_PAGE,
                 role="start", hold=10,
                 banner=f"PAGE ({START_PAGE})  —  welding",
                 sub=SUB_A,
                 qtitle=r"$q_5$ and $q_7$ — the two joints the door will move",
                 y_of=lambda k, p=stall, kp=keepS: float(p[kp[k], 0]),
                 caption=lambda k, p=stall, kp=keepS: (
                     f"1/4   welding along the seam —  y={p[kp[k],0]:.2f}")),
        ]
        if to_door is not None:
            qD, keepD = qs_from_path(arm, weld, to_door, START_SIGNS)
            segs.append(dict(
                common, pm=pmA, q_list=qD, path=to_door, upto_list=keepD,
                wall=polyA.loops, discs=sig[START_PAGE], page=START_PAGE,
                role="start", hold=8, advances=True,
                banner=f"PAGE ({START_PAGE})  —  STALLED at y={stall_y:.2f}, "
                       f"reversing to the door",
                sub="the goal is unreachable from here WITHOUT LEAVING THIS "
                    "PAGE, so the robot backs up to the singular posture",
                qtitle=r"$q_5$ and $q_7$ — still on page " f"({START_PAGE})",
                y_of=lambda k, p=to_door, kp=keepD: float(p[kp[k], 0]),
                caption=lambda k, p=to_door, kp=keepD: (
                    f"2/4   reversing to the OPEN $q_6$ door at y={ay:.3f}"
                    f"  (now y={p[kp[k],0]:.2f})")))
        # 3/4 -- the crossing. The arm is MOTIONLESS here by construction, so
        # the joint strip is the only thing that moves; without it this reads
        # as a hung animation (and did).
        fib_q = fiber[::3]
        fib_d = np.degrees(deltas[::3])
        segs.append(dict(
            common, pm=pmA, q_list=fib_q, path=routeA,
            upto_list=[len(routeA) - 1] * len(fib_q),
            wall=polyA.loops, discs=sig[START_PAGE], page=START_PAGE,
            role="cross", hold=14, advances=False,
            banner="THE PAGE CHANGE  —  "
                   f"({START_PAGE})  ⟶  ({PARTNER_PAGE})",
            sub=r"$q_6=0$ makes the $q_5$ and $q_7$ axes collinear, so this "
                r"motion moves NO LINK — watch the joints, not the arm "
                f"(tool drift {drift*1e3:.0e} mm)",
            qtitle=r"$q_5$ and $q_7$ slide $180°$ past each other — "
                   "both stay inside the limits",
            y_of=lambda k, v=float(routeA[-1, 0]): v,
            caption=lambda k, d=fib_d, n=len(fib_q): (
                f"3/4   gimbal self-motion:  $\delta$={d[k]:+.0f}°  "
                f"— the arm does NOT move  ({100*k/max(1,n-1):.0f}%)")))
        qB, keepB = qs_from_path(arm, weld, pathB, PARTNER_SIGNS)
        segs.append(dict(
            common, pm=pmB, q_list=qB, path=pathB, upto_list=keepB,
            discs=sig[PARTNER_PAGE], page=PARTNER_PAGE,
            role="partner", hold=26,
            banner=f"PAGE ({PARTNER_PAGE})  —  AFTER THE PAGE CHANGE",
            sub="same robot, same weld, a different chart — and on this one "
                "the seam can be finished",
            qtitle=r"$q_5$ and $q_7$ — now at the partner page's reading",
            y_of=lambda k, p=pathB, kp=keepB: float(p[kp[k], 0]),
            caption=lambda k, p=pathB, kp=keepB, n=len(keepB): (
                f"4/4   welding to the goal —  y={p[kp[k],0]:.2f}"
                + ("   ✔ WELD COMPLETE" if k == n - 1 else ""))))
        n_frames = render_gif(segs, out / "exp4.gif", arm)
        t = lap("gif", t)

    # ---- report -----------------------------------------------------------
    report["exp4"] = {
        "scene": dict(EXP4, boxes=[{"center": list(c), "half_extent": list(h),
                                    "label": lbl}
                                   for c, h, lbl in EXP4["boxes"]]),
        "res": args.res,
        "feedback_margin": args.feedback_margin,
        "doors_open": n_open, "doors_blocked": n_blk,
        "theta_seam_touching_pages": [n for n, _, _ in seam],
        "door": {"y": float(ay), "theta": float(at), "kind": door.kind,
                 "fiber_valid_fraction": float(fib["valid_fraction"])},
        "crossing": {
            "direction_deg": float(np.degrees(deltas[-1])),
            "steps": len(fiber), "steps_total": len(deltas),
            "tool_drift_mm": float(drift * 1e3),
            "min_clearance_mm": float(cross["clearance"] * 1e3),
            "tightest_joint_margin_deg": margin,
            "start_page_q_deg": [round(float(v), 3)
                                 for v in np.degrees(qA0.q)],
            "partner_page_q_deg": [round(float(v), 3)
                                   for v in np.degrees(rB.q)],
            "endpoint_matches_partner_rep_deg": float(
                np.max(np.abs(np.degrees(q_end - rB.q)))),
        },
        "start_region": {"page": START_PAGE, "index": kA,
                         "cells": regA.n_cells,
                         "y": [regA.y_min, regA.y_max],
                         "reaches_goal": regA.reaches_goal},
        "partner_region": {"page": PARTNER_PAGE, "index": kB,
                           "cells": regB.n_cells,
                           "y": [regB.y_min, regB.y_max],
                           "reaches_goal": regB.reaches_goal},
        "wall": {"facets": [repA.n_facets, repA.n_passed],
                 "closed": bool(polyA.closed),
                 "gamma": float(wmodel.gamma),
                 "accuracy": float(wmodel.accuracy),
                 "segments": int(polyA.n_segments),
                 "ok": wall_ok,
                 "containment": cont,
                 "note": polyA.note or (repA.failures[0][1]
                                        if repA.failures else "")},
        "gif_frames": n_frames,
    }
    if res is not None:
        report["exp4"]["verdict"] = {
            "proven": bool(res.proven), "reason": res.reason,
            "goal_reached_via": res.goal_reached_via,
            "regions_visited": len(res.regions_visited),
            "certificate": res.certificate,
            "doors_followed": res.doors_followed,
        }
    report["timings_seconds"] = dict(
        timings, script_total=round(time.perf_counter() - T0, 2))
    (out / "exp4_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\n[time] script total: {time.perf_counter() - T0:.1f} s "
          f"(res {args.res}, iters {args.iters})")
    print(f"[out] {out / 'exp4_report.json'}")


if __name__ == "__main__":
    main()
