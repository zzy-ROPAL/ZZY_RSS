"""EXPERIMENT 5 -- the q4 FOLD DOOR: a door that is a LINE on the page.

    EXP 5a  obstacles leave the task FEASIBLE, but every plan has to flip the
            elbow, and the flip is only affordable at the fold. The continuous
            joint path that flips the elbow CHANGES PAGE: measured, the smooth
            continuation through the fold is (theta + pi, s4 flipped) at
            0.019 deg of joint discontinuity, while keeping the page costs
            exactly 180 deg. So here the page change is forced by continuity,
            not by bookkeeping.

    EXP 5b  one obstacle disconnects the goal end of the seam. The proof is
            INFEASIBLE and it is a TWO-PAGE proof: the fold line sits inside
            the start region, so the flood fill has to follow it to the s4
            partner page and close a second wall there.

Why this experiment needed new code
-----------------------------------
EXP 1-4 used q2 and q6 doors, which are POINTS on the page and LINES in Q (a
self-motion fiber). A q4 fold is the dual: a LINE on the page and a POINT in Q.
Three consequences, all of them the point of the experiment.

1.  It cannot exist on a straight seam. Invariant 22: a straight seam with a
    constant tool pose makes the wrist centre affine in y, so |W(y) - S| is
    convex, so the reachable y set is one interval and the fold |W - S| =
    reach_max can only sit at an endpoint, where the task leaves the
    workspace. 2280 straight placements measured: 0 non-convex. So weld.py
    grew an ARC seam, on which |W - S| has an interior maximum, and the
    fixture is placed so that maximum equals reach_max EXACTLY -- the limiting
    workcell layout in which the arm just reaches the farthest point of the
    seam. That is a codimension-1 condition, so it is SOLVED (brentq on the
    fixture's x), not searched; residual 0.0 to 1.1e-16 m.

2.  Its certification is EXACT, which closes todo 5 for line doors. At the
    fold the elbow circle has radius 0, so every theta at that y is the same
    arm in space, and q4 = 0 also makes the q3 and q5 axes collinear, so
    (q3 + d, q5 - d) is a null motion. Measured drift of every link point over
    BOTH families: 2.0e-17 m. So collision is ONE query and joint limits are a
    line-meets-box intersection, computed by intersecting intervals. No
    sampling anywhere -- unlike a point door, whose fiber is a measure-zero set
    that can only be sampled.

    And the fiber is not a formality here: the IK representative at the fold
    needs q5 = -180 deg, which is 1.145 deg outside the real +/-178.855 limit
    -- the same 1.15 deg that kills every q2 door in this task (findings 30).
    The q3/q5 null motion moves q5 off that bound and the door is open with
    43.11 deg to spare. Invariant 3, with the margin measured exactly.

3.  It forces the page to be treated as a CYLINDER. On this placement each
    page's forbidden theta window is thin and the s4 mirror shifts q3 and q5
    by pi, so a page's window and its s4 PARTNER's window sit pi apart: no
    single cut in theta can land inside both. Since the fold door makes the
    proof visit both pages, one of them always has valid cells at the cut, and
    invariant 5 says a wall "closed" across such a cut is open around the back
    of the cylinder. Measured seam-touching valid cells over the eight pages:
    116 / 168 / 218 / 278 at res 41 / 61 / 81 / 101 -- never zero. So
    theta_periodic is True here, the SVM embeds theta as (cos, sin) to be a
    function on the cylinder at all, and the wall comes out as TWO WRAPPING
    LOOPS rather than one enclosing loop.

Outputs per experiment: an animated GIF, the eight-page map with the learned
manifold drawn on it, a fold-door anatomy figure, and (5b) the certificate JSON
that flood.run_proof produced.
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
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from zzy_rss.arm import SrsArm, VALID  # noqa: E402
from zzy_rss.doors import (  # noqa: E402
    build_sigma, certify_fold_door, door_fiber_validity)
from zzy_rss.flood import (  # noqa: E402
    find_inf_proof_region, invalid_points, label_regions, live_discs,
    other_valid_points, region_points, run_proof, thin, write_certificate)
from zzy_rss.pagemap import build_all_pages_shared, q_for_branch  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.wall import (  # noqa: E402
    PageLabeller, is_periodic, sigma_disc_samples, virtual_boundary_samples)
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
START_PAGE, START_SIGNS = "++-", (1, 1, -1)
# The fold door flips s4 and nothing else, so the partner page differs from the
# start page in its MIDDLE sign only.
FOLD_PARTNER, PARTNER_SIGNS = "+--", (1, -1, -1)
CMAP = ListedColormap(["#2a9d8f", "#e9e9e9", "#e9c46a", "#e76f51"])

# ---------------------------------------------------------------- placement --
#
# A circumferential seam: a horizontal arc of radius R at height CZ, which is
# the commonest curved weld there is (pipe-to-plate, flange rings). The torch
# is held at a fixed 30 deg tilt. y = 0 and y = 1 are the ends of the arc and
# PHI0/PHI1 are chosen so the arc's FAR point -- the one at phi = 0, farthest
# from the robot -- lands at y* = 4/9 of the way along, i.e. in the interior.
CZ, R_ARC, TILT = 0.30, 0.20, 30.0
PHI0, PHI1 = -0.80, 1.00

# EXP 5b: one box. Found by scanning 2264 candidates and keeping those that
# (i) clear the fold posture, (ii) leave no page with a region reaching both
# ends, (iii) keep the fold INSIDE the start region -- otherwise the flood
# never follows it and the proof is single-page -- and (iv) still leave the
# goal end reachable in SOME posture, so the infeasibility is a connectivity
# failure and not a curtain (findings 24). Ranked by how deep inside the
# invalid set a wrapping wall can stay (0.1833 scaled units against a cell
# diagonal of 0.0373, i.e. about 5 cells of room), this was the best.
OBS_5B = (BoxObstacle(center=(0.66, 0.12, 0.50),
                      half_extent=(0.07, 0.07, 0.07), label="post"),)

# EXP 5a: two boxes, one each side of the seam plane, both aimed at an elbow
# position. The elbow circle translates from -y to +y as the torch runs along
# the arc, so the -y post blocks the elbow over the FIRST half of the seam and
# the +y post blocks it over the SECOND half. The arm therefore has to swing
# its arm angle by about pi, and the set of arm angles it can reach below the
# fold is DISJOINT from the set the goal end admits above it, so the swing
# cannot be completed anywhere else.
OBS_5A = (BoxObstacle(center=(0.281, -0.142, 0.334),
                      half_extent=(0.07, 0.07, 0.07), label="post_-y"),
          BoxObstacle(center=(0.278, 0.146, 0.334),
                      half_extent=(0.055, 0.055, 0.055), label="post_+y"))


def weld_arc(cx, cy, cz, radius, tilt_deg, phi0, phi1) -> WeldParams:
    a = np.radians(tilt_deg)
    return WeldParams(
        p0=(cx + radius, cy, cz), p1=(cx + radius, cy, cz),   # unused for arcs
        approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
        reference=(float(np.cos(a)), 0.0, float(np.sin(a))),
        curve="arc", arc_center=(cx, cy, cz), arc_radius=radius,
        arc_normal=(0.0, 0.0, 1.0), arc_ref=(1.0, 0.0, 0.0),
        arc_phi0=phi0, arc_phi1=phi1)


def solve_fixture_x(cy, cz, radius, tilt) -> tuple:
    """Place the fixture so the seam's far point is EXACTLY at full extension.

    |W - S| on a horizontal arc is  hypot(dz, |C_xy - G_xy| + R)  at the far
    point, with G = S + d_wt * (tool axis). Setting that equal to reach_max is
    one equation in the fixture's x, solved to the last representable bit. The
    result is a DOUBLE root of |W(y) - S| - reach_max: both halves of the seam
    stay reachable and the fold door lands in the task's interior, which is
    docs/exp5_spec.md condition B and the thing a straight seam cannot have.
    """
    arm_p = Config().arm
    shoulder = np.array([0.0, 0.0, arm_p.d_bs])

    def residual(cx):
        w = WeldLine(weld_arc(cx, cy, cz, radius, tilt, 0.0, 1.0))
        g = shoulder + arm_p.d_wt * (w.rot @ np.array([0.0, 0.0, 1.0]))
        c = np.array([cx, cy, cz])
        rho = float(np.linalg.norm((g - c)[:2]))
        return float(np.hypot(g[2] - cz, rho + radius)) - arm_p.reach_max

    cx = brentq(residual, 0.05, 1.2, xtol=1e-15, rtol=8.9e-16)
    return float(cx), float(residual(cx))


CX, CX_RESIDUAL = solve_fixture_x(0.0, CZ, R_ARC, TILT)
Y_STAR = (0.0 - PHI0) / (PHI1 - PHI0)      # the far point sits at phi = 0


def make(obstacles=(), res=61):
    base = Config()
    cfg = dataclasses.replace(
        base,
        weld=weld_arc(CX, 0.0, CZ, R_ARC, TILT, PHI0, PHI1),
        # theta_periodic=True is NOT a preference here; see the module
        # docstring, point 3. Valid cells touch theta = +/-pi at every
        # resolution, so the flat rectangle is not equivalent to the cylinder.
        pagemap=PageMapParams(n_y=res, n_theta=res, theta_periodic=True),
        scene=SceneParams(gap_open=True, obstacles=tuple(obstacles)),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz"))
    return cfg, SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)


# ----------------------------------------------------------------- plumbing --

def pages_and_sigma(cfg, arm, weld, scene):
    """Stage 1 + Stage 2, with every door's state decided.

    Doors are counted as DOORS, not as discs. A fold is a LINE door: it needs
    ~190 overlapping discs to cover the whole theta range (the facet guard has
    to refuse a crossing anywhere along it), but it is ONE door -- one posture,
    one certificate. Counting discs would report 1512 doors on 8 pages where
    there are 8.
    """
    built = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
    pms = {n: built[n] for n in PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blk = 0
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        decided: dict = {}
        for d in sig.get(n, []):
            if not d.vanishing:
                continue
            key = ("fold", round(float(d.exact_y), 12)) if d.fold else ("pt", id(d))
            if key not in decided:
                decided[key] = door_fiber_validity(
                    cfg, arm, weld, scene, d, sg)["ok"]
            d.__dict__["_state"] = "open" if decided[key] else "blocked"
        for ok in decided.values():
            n_open += bool(ok)
            n_blk += (not ok)
    return pms, sig, n_open, n_blk


def seam_report(pms, label=""):
    """Invariant 5, re-measured. Here it is expected to FAIL, and that is the
    finding: the page must be carried as a cylinder, not as a rectangle."""
    touching = {n: int(p.valid_mask[:, 0].sum() + p.valid_mask[:, -1].sum())
                for n, p in pms.items()}
    total = sum(touching.values())
    if total:
        print(f"[seam]{label} {total} valid cell(s) touch theta=+/-pi over the "
              f"eight pages {touching}")
        print(f"[seam]{label} -> the flat rectangle is NOT equivalent to the "
              f"cylinder here, so theta_periodic=True is REQUIRED (invariant 5)")
    else:
        print(f"[seam]{label} no valid cell touches theta=+/-pi")
    return touching


def prove(cfg, arm, weld, scene, pm, lab, k, discs, iters=6, verbose=False):
    """One region's wall, with the same budget run_proof uses (invariant 16)."""
    sg = tuple(1 if c == "+" else -1 for c in pm.name)
    # The SAME seed run_proof uses, not a local 0. Invariant 16: if the figure's
    # wall is trained from a different thinning of the same region it is a
    # different wall, and the picture then illustrates something the
    # certificate does not contain. Measured cost of getting this wrong on the
    # partner page here: gamma 1.0 -> 229.
    rng = np.random.default_rng(cfg.planner.rng_seed)
    pts = region_points(pm, lab, k)
    live, disch = live_discs(cfg, arm, weld, scene, sg, discs)
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    se, re_ = sigma_disc_samples(live, pts, tol, periodic=is_periodic(cfg))
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
    return poly, rep, w, live, disch


def bfs_route(pm, mask, start_ij, goal_row=None):
    """4-connected route in `mask`, theta periodic; to goal_row or as far as it gets."""
    ny, nt = mask.shape
    if not mask[start_ij]:
        return None
    prev = {start_ij: None}
    q = deque([start_ij])
    best = start_ij
    while q:
        i, j = q.popleft()
        if goal_row is None and i > best[0]:
            best = (i, j)
        if goal_row is not None and i == goal_row:
            best = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, (j + dj) % nt
            if 0 <= ni < ny and mask[ni, nj] and (ni, nj) not in prev:
                prev[(ni, nj)] = (i, j)
                q.append((ni, nj))
    out, cur = [], best
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return out


def two_page_route(pmA, pmB, i_star):
    """A plan that USES the fold door: page A below the fold, page B above it.

    The edge across the fold is what the door means. At y* the elbow circle has
    radius 0, so every theta at that y is the same posture and so is either s4
    spelling; the plan may therefore step from (A, i_star, any theta) to
    (B, i_star + 1, ANY theta). Nothing else in the graph connects the halves,
    so a route that uses that edge is a route through the door.
    """
    vA, vB = pmA.tag == VALID, pmB.tag == VALID
    ny, nt = vA.shape
    starts = [(0, 0, j) for j in range(nt) if vA[0, j]]
    if not starts:
        return None
    prev = {s: None for s in starts}
    q = deque(starts)
    goal = None
    while q:
        pg, i, j = q.popleft()
        if pg == 1 and i == ny - 1:
            goal = (pg, i, j)
            break
        v = vA if pg == 0 else vB
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, (j + dj) % nt
            if 0 <= ni < ny and v[ni, nj] and (pg, ni, nj) not in prev:
                prev[(pg, ni, nj)] = (pg, i, j)
                q.append((pg, ni, nj))
        if pg == 0 and i == i_star:
            for jj in range(nt):
                if vB[i_star + 1, jj] and (1, i_star + 1, jj) not in prev:
                    prev[(1, i_star + 1, jj)] = (pg, i, j)
                    q.append((1, i_star + 1, jj))
    if goal is None:
        return None
    out, cur = [], goal
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return out


def qs_from_cells(arm, weld, pm, cells, signs):
    """Joint vectors for a list of (i, j) cells on one page."""
    qs, keep = [], []
    for n, (i, j) in enumerate(cells):
        r = arm.ik_branch(*weld.pose(float(pm.y_grid[i])),
                          float(pm.theta_grid[j]), signs)
        if r.ok:
            qs.append(r.q)
            keep.append(n)
    return qs, keep


def densify(arm, weld, pm, cells, signs, n_target: int):
    """Resample a grid route to ~n_target frames, re-solving IK as it goes.

    A route is a list of grid cells, so its length is set by the grid and not
    by how much motion it contains: page (+--) is entered AT the fold and has
    only six cells left before wall #2, which is six frames of animation for a
    stretch the viewer needs to see. Interpolating in (y, theta) and solving
    the IK at each sample gives as many frames as wanted and every one of them
    is an exact posture on the seam, not a blend of two.

    Returns (q_list, path, upto_list) ready for a render_gif segment.
    """
    pts = np.array([[pm.y_grid[i], pm.theta_grid[j]] for i, j in cells],
                   dtype=float)
    if len(pts) < 2:
        qs, keep = qs_from_cells(arm, weld, pm, cells, signs)
        return qs, pts, keep
    # arc length along the route, taking the short way round in theta
    d = np.diff(pts, axis=0)
    d[:, 1] = (d[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    seg = np.linalg.norm(d * np.array([1.0, 1.0 / np.pi]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0.0:
        qs, keep = qs_from_cells(arm, weld, pm, cells, signs)
        return qs, pts, keep
    targets = np.linspace(0.0, cum[-1], max(2, int(n_target)))
    qs, path, upto = [], [], []
    for t in targets:
        k = int(np.clip(np.searchsorted(cum, t) - 1, 0, len(seg) - 1))
        w = 0.0 if seg[k] <= 0 else (t - cum[k]) / seg[k]
        y = float(pts[k, 0] + w * d[k, 0])
        th = float(pts[k, 1] + w * d[k, 1])
        th = (th + np.pi) % (2.0 * np.pi) - np.pi
        r = arm.ik_branch(*weld.pose(y), th, signs)
        if not r.ok:
            continue
        qs.append(r.q)
        path.append([y, th])
        upto.append(len(path) - 1)
    return qs, np.asarray(path, dtype=float), upto


def fold_posture(cfg, arm, weld, cert):
    """The fold posture, spelled on the q3/q5 null motion so limits hold.

    The IK representative at the fold needs q5 = -180 deg, 1.145 deg outside
    the real limit -- the same deficit that shuts every q2 door in this task.
    The null motion (q3 + d, q5 - d) leaves every link point where it is
    (measured drift 2.0e-17 m) and moves q5 off the bound, so the posture the
    robot actually holds is the one at the best delta, not the one the IK
    happens to return. Invariant 3.
    """
    pos, rot = weld.pose(cert.y)
    r = arm.ik_branch(pos, rot, 0.0, START_SIGNS)
    q = r.q.copy()
    v = cert.pages.get(START_PAGE, {})
    d = v.get("best_delta_rad")
    if d is not None:
        q[2] = float(((q[2] + d + np.pi) % (2 * np.pi)) - np.pi)
        q[4] = float(((q[4] - d + np.pi) % (2 * np.pi)) - np.pi)
    return q, (float(d) if d is not None else 0.0)


# ------------------------------------------------------------------ drawing --

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


def draw_arm(ax, cfg, arm, scene, weld, q, trail, mark_fold=None):
    seam = weld.polyline(161)
    ax.plot(seam[:, 0], seam[:, 1], seam[:, 2], color="#e9c46a", lw=9,
            solid_capstyle="round", zorder=0)
    ax.plot([0, 0], [0, 0], [0, cfg.arm.d_bs], color="#888888", lw=10,
            solid_capstyle="round", zorder=0)
    if mark_fold is not None:
        ax.scatter(*mark_fold, color="#1d3557", s=130, marker="D",
                   depthshade=False, edgecolor="#fff", zorder=8)
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
        t = np.array(trail)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="#e63946", lw=4)
    ax.scatter(*f["tool"], color="#e63946", s=95, depthshade=False,
               edgecolor="#111", zorder=9)
    ax.set_xlim(-0.05, 0.80)
    ax.set_ylim(-0.40, 0.40)
    ax.set_zlim(0, 0.70)
    ax.set_box_aspect((0.85, 0.80, 0.70))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-62)
    return f


# How far outside the page to draw, so the virtual-boundary band -- and the
# wall loop that closes inside it -- are visible at all. IJRR's Fig. 5 trick
# puts an invalid band of positive thickness just outside y in [0, 1] with a
# virtual FREE ring beyond it; on a cylinder page the wall is two wrapping
# loops and ONE OF THEM LIVES IN THAT BAND, so a plot clipped to [0, 1] shows
# half the proof.
Y_VIEW = 0.10


def draw_page(ax, pm, path=None, upto=None, wall=None, kinds=None,
              fold_y=None, title="", mark=None, band=None):
    """One page. The wall is the LEARNED MANIFOLD: the SVM's zero level set,
    reconstructed and verified facet by facet."""
    if band:
        ax.axvspan(-Y_VIEW, 0.0, color="#b0b0b0", alpha=0.55, zorder=0)
        ax.axvspan(1.0, 1.0 + Y_VIEW, color="#b0b0b0", alpha=0.55, zorder=0)
        ax.axvspan(-band, 0.0, color="#7a7a7a", alpha=0.75, zorder=1)
        ax.axvspan(1.0, 1.0 + band, color="#7a7a7a", alpha=0.75, zorder=1)
    ax.imshow(pm.tag.T, origin="lower", aspect="auto", cmap=CMAP, vmin=-.5,
              vmax=3.5, extent=[0, 1, pm.theta_grid[0], pm.theta_grid[-1]],
              zorder=2)
    if fold_y is not None:
        ax.axvspan(fold_y - 0.008, fold_y + 0.008, color="#1d3557", alpha=0.85,
                   zorder=4)
        ax.text(fold_y, pm.theta_grid[-1] * 0.80, r" $q_4\!=\!0$ fold LINE",
                fontsize=8.5, color="#1d3557", fontweight="bold", rotation=90,
                va="top", ha="left", zorder=6)
    for n, loop in enumerate(wall or []):
        kind = (kinds[n] if kinds and n < len(kinds) else "")
        if kind == "wrapping":
            # closes through the theta seam, so draw the closing segment too
            ll = np.vstack([loop, loop[:1] + np.array([0.0, 2 * np.pi])])
            ax.plot(ll[:, 0], ll[:, 1], color="#9d0208", lw=3, zorder=5)
            ax.plot(ll[:, 0], ll[:, 1] - 2 * np.pi, color="#9d0208", lw=3,
                    alpha=0.35, zorder=5)
        else:
            ax.plot(loop[:, 0], loop[:, 1], color="#9d0208", lw=3, zorder=5)
    if path is not None:
        p = np.asarray(path)
        n = len(p) if upto is None else upto + 1
        ax.plot(p[:n, 0], p[:n, 1], color="#e63946", lw=2.6, zorder=6)
        ax.scatter(p[n - 1, 0], p[n - 1, 1], color="#e63946", s=80,
                   edgecolor="#111", zorder=7)
    if mark is not None:
        ax.scatter([mark[0]], [mark[1]], marker="*", s=240, color="#ffffff",
                   edgecolor="#111", zorder=8)
    ax.set_xlim(-Y_VIEW, 1.0 + Y_VIEW)
    ax.set_ylim(pm.theta_grid[0], pm.theta_grid[-1])
    ax.set_xlabel("y  (weld coordinate)")
    ax.set_ylabel(r"$\theta$  (arm angle)")
    ax.set_title(title, fontsize=11)


PAGE_NOTE = (
    "Each panel is ONE IK branch of the same task. The triple is "
    r"$(s_2,s_4,s_6)$ = the signs of joints 2, 4, 6 — shoulder, elbow, wrist. "
    "All eight put the tool on the same seam point; they differ only in joint "
    "values, so they differ by JOINT LIMITS.\n"
    r"The page is a CYLINDER in $\theta$ (invariant 5 re-measured: valid cells "
    r"DO touch $\theta=\pm\pi$ here), so a separating wall WRAPS it instead of "
    "enclosing a blob — hence two horizontal loops, not one ring.\n"
    r"The dark vertical band is the $q_4=0$ FOLD: there the elbow circle has "
    r"radius 0, so the whole $\theta$ line is ONE posture. A wall may not cross "
    "it; the proof must go through it."
)


def fig8(pms, walls, kinds, title, out_path, fold_y=None, path=None,
         start_mark=None, band=None):
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.4), constrained_layout=True)
    for ax, n in zip(axes.ravel(), PAGES):
        pm = pms[n]
        w = walls.get(n)
        draw_page(ax, pm, wall=w, kinds=kinds.get(n), fold_y=fold_y,
                  path=path if n == START_PAGE else None,
                  mark=start_mark if n == START_PAGE else None, band=band)
        role = ""
        if n == START_PAGE:
            role = "   <- START PAGE"
        elif n == FOLD_PARTNER:
            role = "   <- partner across the FOLD door"
        ax.set_title(f"({n})  valid {pm.stats['frac_valid'] * 100:.1f}%"
                     f"{'   WALL' if w else ''}{role}", fontsize=11,
                     fontweight="bold" if n in (START_PAGE, FOLD_PARTNER)
                     else "normal")
    handles = [
        mpatches.Patch(color="#2a9d8f", label="valid posture"),
        mpatches.Patch(color="#e9c46a", label="invalid: joint limit"),
        mpatches.Patch(color="#e76f51", label="invalid: collision"),
        mpatches.Patch(color="#e9e9e9", label="invalid: no IK"),
        mpatches.Patch(color="#1d3557", label=r"$q_4=0$ fold LINE door"),
        mpatches.Patch(color="#7a7a7a", alpha=0.75,
                       label="virtual invalid band (IJRR Fig. 5)"),
        plt.Line2D([], [], color="#9d0208", lw=3,
                   label="learned manifold (RBF-SVM zero set = the proof)"),
        plt.Line2D([], [], color="#e63946", lw=3, label="robot motion"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(title + "\n" + PAGE_NOTE, fontsize=11.5)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_fold_door(cfg, arm, weld, scene, cert, out_path, q_fold):
    """The anatomy of a LINE door, and why its certificate is exact."""
    fig = plt.figure(figsize=(21, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.12, 1.0, 1.0),
                          height_ratios=(1.0, 1.0), wspace=0.26, hspace=0.42)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axd = fig.add_subplot(gs[0, 1])
    axq = fig.add_subplot(gs[1, 1])
    axe = fig.add_subplot(gs[0, 2])
    axt = fig.add_subplot(gs[1, 2])

    # --- 1. the posture, with the whole theta line drawn on top of itself ---
    seam = weld.polyline(161)
    ax3.plot(seam[:, 0], seam[:, 1], seam[:, 2], color="#e9c46a", lw=8,
             solid_capstyle="round", zorder=0)
    ax3.plot([0, 0], [0, 0], [0, cfg.arm.d_bs], color="#888888", lw=9,
             solid_capstyle="round", zorder=0)
    for c, h, _l in scene.boxes:
        draw_box(ax3, c, h)
    pos, rot = weld.pose(cert.y)
    for th in np.linspace(-np.pi, np.pi, 17):
        r = arm.ik_branch(pos, rot, float(th), START_SIGNS)
        if not r.ok:
            continue
        f = arm.frames(r.q)
        pts = [f["shoulder"], f["elbow"], f["wrist"], f["tool"]]
        for i, col in enumerate(("#1d3557", "#4895ef", "#e76f51")):
            u, v = pts[i], pts[i + 1]
            ax3.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], color=col,
                     lw=5.5, alpha=0.8, solid_capstyle="round")
    f0 = arm.frames(q_fold)
    ax3.scatter(*f0["elbow"], color="#1d3557", s=150, marker="D",
                depthshade=False, edgecolor="#fff", zorder=10)
    ax3.scatter(*f0["tool"], color="#e63946", s=150, depthshade=False,
                edgecolor="#111", zorder=10)
    ax3.set_xlim(-0.05, 0.80); ax3.set_ylim(-0.40, 0.40); ax3.set_zlim(0, 0.70)
    ax3.set_box_aspect((0.85, 0.80, 0.70))
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.view_init(elev=22, azim=-62)
    ax3.set_title(f"1.  the fold posture at y* = {cert.y:.6f}.\n"
                  f"All 17 arm angles are drawn: they LAND ON TOP OF EACH "
                  f"OTHER.\nMeasured drift of every link point over the whole "
                  f"$\\theta$ line\nAND the $q_3/q_5$ null motion: "
                  f"{cert.posture_drift_m:.1e} m", fontsize=10, pad=10)

    # --- 2. the exact limit intervals along the null motion ----------------
    deltas = np.linspace(-np.pi, np.pi, 1441)
    styles = {START_PAGE: ("#1d3557", 3.4, "-"), FOLD_PARTNER: ("#e76f51", 1.8, "--")}
    for name, col in ((START_PAGE, "#1d3557"), (FOLD_PARTNER, "#e76f51")):
        signs = tuple(1 if c == "+" else -1 for c in name)
        q = q_for_branch(arm.ik_branch(pos, rot, 0.0, (1, 1, 1)).q,
                         (1, 1, 1), signs)
        m = []
        for d in deltas:
            qq = q.copy()
            qq[2] = ((qq[2] + d + np.pi) % (2 * np.pi)) - np.pi
            qq[4] = ((qq[4] - d + np.pi) % (2 * np.pi)) - np.pi
            m.append(np.degrees(arm.limit_margin(qq)))
        c, lw, ls = styles[name]
        axd.plot(deltas, m, lw=lw, ls=ls, color=c, label=f"page ({name})")
        for a, b in cert.pages[name]["delta_intervals"]:
            axd.axvspan(a, b, color=col, alpha=0.10)
    axd.axhline(0.0, color="#9d0208", lw=2.0, ls="--")
    axd.scatter([0.0], [np.degrees(arm.limit_margin(
        arm.ik_branch(pos, rot, 0.0, START_SIGNS).q))], s=120, color="#9d0208",
        zorder=6, edgecolor="#111")
    axd.set_ylim(-16, 62)
    axd.annotate("the IK representative, $\\delta=0$:  $q_5=-180°$,\n"
                 "$1.145°$ OUTSIDE the real $\\pm178.855°$ limit.\n"
                 "The same deficit shuts every $q_2$ door here.",
                 xy=(0.0, -1.145), xytext=(-3.1, -15.0),
                 fontsize=8.2, fontweight="bold", color="#9d0208",
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#9d0208"))
    axd.text(0.0, 60.0, "the two pages' curves COINCIDE: the $s_4$ mirror IS "
             "the $\\delta=\\pi$ point of this very null motion",
             fontsize=7.8, color="#444", ha="center", va="top")
    axd.set_xlabel(r"$\delta$  along the $q_3/q_5$ null motion  "
                   r"$(q_3+\delta,\ q_5-\delta)$")
    axd.set_ylabel("slack to the\nnearest limit [deg]")
    axd.set_title("2.  the door is open BY THE FIBER, not by the "
                  "representative\n(shaded = the admissible set, computed by "
                  "INTERSECTING INTERVALS — exact)", fontsize=10, pad=6)
    axd.legend(fontsize=8, loc="center right", framealpha=0.96,
               bbox_to_anchor=(1.0, 0.62))
    axd.grid(alpha=0.25)

    # --- 3. the two pages' spellings of that one posture -------------------
    idx = np.arange(7)
    a_deg = np.array(cert.pages[START_PAGE]["q_deg"])
    b_deg = np.array(cert.pages[FOLD_PARTNER]["q_deg"])
    lim = np.degrees(arm.q_max)
    w = 0.38
    axq.bar(idx - w / 2, a_deg, w, color="#1d3557", label=f"({START_PAGE})")
    axq.bar(idx + w / 2, b_deg, w, color="#e76f51", label=f"({FOLD_PARTNER})")
    axq.plot(idx, lim, "_", ms=20, color="#9d0208", mew=2.2)
    axq.plot(idx, -lim, "_", ms=20, color="#9d0208", mew=2.2, label="limit")
    axq.set_xticks(idx)
    axq.set_xticklabels([f"$q_{i+1}$" for i in idx])
    axq.set_ylabel("angle [deg]")
    axq.set_ylim(-265, 360)
    axq.axhline(0, color="#777", lw=0.8)
    # The two spellings differ by exactly 180 deg in q3 and q5 -- and that
    # offset is the delta = pi point of the q3/q5 null motion, verified to
    # 0.0 deg. So the page change is not an extra freedom on top of the door:
    # it is a point of the door's OWN one-parameter family, which is why it
    # costs nothing and is available whenever the door is open at all.
    axq.text(3.0, 352, "the two spellings differ by exactly $180°$ in $q_3$ "
             "and $q_5$ — the $s_4$ mirror.\n"
             "That offset is the $\\delta=\\pi$ point of the null motion in "
             "panel 2 (verified to $0.0°$),\n"
             "so the PAGE CHANGE is a point of the door's own family, not an "
             "extra freedom.",
             fontsize=7.6, ha="center", va="top", color="#444")
    j5 = 4
    axq.annotate(f"$q_5={a_deg[j5]:.0f}°$: outside the\n"
                 f"$\\pm178.855°$ limit by $1.145°$.\n"
                 f"BOTH spellings are, equally.",
                 xy=(j5, a_deg[j5]), xytext=(4.6, -255), fontsize=7.8,
                 fontweight="bold", color="#9d0208",
                 arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#9d0208"))
    axq.set_title(r"3.  the two pages' readings of that ONE posture",
                  fontsize=10, pad=6)
    axq.legend(fontsize=7.5, ncol=3, loc="lower left", framealpha=0.95)
    axq.grid(alpha=0.2, axis="y")

    # --- 4. the elbow circle collapsing, i.e. what the flip costs ----------
    ys = np.linspace(0.0, 1.0, 401)
    rr, disp = [], []
    for yv in ys:
        r = arm.ik_branch(*weld.pose(float(yv)), 0.0, START_SIGNS)
        rr.append(r.elbow_circle_radius * 1000 if r.ok else np.nan)
        disp.append(2 * r.elbow_circle_radius * 1000 if r.ok else np.nan)
    axe.plot(ys, disp, lw=2.4, color="#4895ef")
    axe.axvline(cert.y, color="#1d3557", lw=2.4)
    axe.scatter([cert.y], [0.0], s=110, color="#1d3557", zorder=6,
                edgecolor="#fff")
    axe.annotate("0.00 mm  <- the flip is FREE\nonly here", xy=(cert.y, 0.0),
                 xytext=(cert.y + 0.08, 70), fontsize=9, fontweight="bold",
                 color="#1d3557",
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#1d3557"))
    axe.set_xlabel("y  (weld coordinate)")
    axe.set_ylabel("elbow displacement [mm]")
    axe.set_title(r"4.  what an elbow flip ($\theta\to\theta+\pi$) COSTS,"
                  "\nas a function of where it is attempted", fontsize=10, pad=6)
    axe.grid(alpha=0.25)

    # --- 5. |W - S| touching reach_max, which is the whole construction ----
    sh = arm.shoulder
    L = np.array([float(np.linalg.norm(weld.wrist_center(float(y), arm.p.d_wt)
                                       - sh)) for y in ys])
    axt.plot(ys, L * 1000, lw=2.4, color="#2a9d8f")
    axt.axhline(arm.p.reach_max * 1000, color="#9d0208", lw=2.0, ls="--")
    axt.axvline(cert.y, color="#1d3557", lw=2.0)
    axt.annotate(f"reach_max = {arm.p.reach_max*1000:.1f} mm\n"
                 f"touched from BELOW: a DOUBLE root,\nso both halves stay "
                 f"reachable\n(residual {abs(CX_RESIDUAL):.1e} m)",
                 xy=(cert.y, arm.p.reach_max * 1000),
                 xytext=(0.06, arm.p.reach_max * 1000 - 46), fontsize=8.5,
                 fontweight="bold", color="#9d0208",
                 arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#9d0208"))
    axt.set_xlabel("y  (weld coordinate)")
    axt.set_ylabel(r"$|W(y)-S|$ [mm]")
    axt.set_title("5.  why the fold is INTERIOR at all: the ARC makes "
                  r"$|W-S|$ non-convex." "\n"
                  r"On a straight seam it is convex, so the fold can only sit "
                  r"at an END (invariant 22).", fontsize=10, pad=6)
    axt.grid(alpha=0.25)

    fig.suptitle(
        "ANATOMY OF A LINE DOOR  ($q_4=0$, full extension)\n"
        "A point door is a POINT on the page and a LINE in Q; this is the "
        "dual — a LINE on the page and a POINT in Q. That is why its "
        "certificate is EXACT:\none collision query for the whole line, plus "
        "an interval intersection for the limits. No sampling of a "
        "measure-zero fiber anywhere.", fontsize=12, y=1.045)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


# -------------------------------------------------------------- animation ---

def render_gif(segments, out_path, fps=12, dpi=84):
    """Animated GIF: the robot on the left, the page with the wall on the right.

    PillowWriter rather than ffmpeg, so the GIF is produced directly and the
    deliverable does not depend on an external binary.
    """
    fig = plt.figure(figsize=(14.2, 5.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.01, right=0.97, wspace=0.26)
    writer = PillowWriter(fps=fps)
    n_frames = 0
    with writer.saving(fig, str(out_path), dpi=dpi):
        for seg in segments:
            trail = list(seg.get("trail0") or [])
            for idx, q in enumerate(seg["q_list"]):
                ax.clear()
                ax2.clear()
                f = draw_arm(ax, seg["cfg"], seg["arm"], seg["scene"],
                             seg["weld"], q, trail, mark_fold=seg.get("fold_xyz"))
                trail.append(f["tool"])
                ax.set_title(seg["caption"](idx), fontsize=10.5)
                draw_page(ax2, seg["pm"], path=seg.get("path"),
                          upto=(seg["upto_list"][idx]
                                if seg.get("upto_list") else None),
                          wall=seg.get("wall"), kinds=seg.get("kinds"),
                          fold_y=seg.get("fold_y"), band=seg.get("band"),
                          title=f"page ({seg['page']})   "
                                r"coordinate space $Y^+$ (a cylinder in $\theta$)")
                fig.subplots_adjust(left=0.01, right=0.97, wspace=0.26,
                                    top=0.84, bottom=0.12)
                writer.grab_frame()
                n_frames += 1
    plt.close(fig)
    print(f"[gif] {out_path}  ({n_frames} frames)")


def elbow_path_length(arm, qs) -> float:
    pts = np.array([arm.frames(q)["elbow"] for q in qs])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="out/exp5")
    ap.add_argument("--only", default="ab", help="'a', 'b' or 'ab'")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "exp5_report.json"
    report = json.loads(rpath.read_text()) if rpath.exists() else {}
    timings = {}
    T0 = time.perf_counter()

    def lap(label, since):
        dt = time.perf_counter() - since
        timings[label] = round(dt, 2)
        print(f"[time] {label}: {dt:.1f} s")
        return time.perf_counter()

    print("=" * 78)
    print("EXPERIMENT 5 -- the q4 FOLD DOOR (a door that is a LINE on the page)")
    print("=" * 78)
    print(f"[place] circumferential seam: centre ({CX:.9f}, 0.000, {CZ}), "
          f"R = {R_ARC}, torch tilt {TILT:.0f} deg, phi in [{PHI0}, {PHI1}]")
    print(f"[place] the fixture's x was SOLVED so the seam's far point sits "
          f"exactly at full extension: residual {CX_RESIDUAL:+.2e} m")
    print(f"[place] => interior fold door at y* = {Y_STAR:.9f}")
    report["placement"] = {
        "seam": "arc", "arc_center": [CX, 0.0, CZ], "arc_radius": R_ARC,
        "tilt_deg": TILT, "phi0": PHI0, "phi1": PHI1,
        "fixture_x_residual_m": CX_RESIDUAL, "y_star": Y_STAR,
        "note": "fixture x solved so max_y |W(y)-S| == reach_max exactly; "
                "that double root is what puts the fold INSIDE the task",
    }

    # ---------------------------------------------------------------- 5 b ---
    if "b" in args.only:
        print("\n" + "-" * 78)
        print("EXP 5b : obstacle disconnects the goal end -> INFEASIBLE, and "
              "the proof is TWO-PAGE")
        print("-" * 78)
        t_e = t_s = time.perf_counter()
        cfg, arm, weld, scene = make(OBS_5B, args.res)
        pms, sig, n_open, n_blk = pages_and_sigma(cfg, arm, weld, scene)
        t_s = lap("5b_pages_and_doors", t_s)
        seam = seam_report(pms, " 5b")
        pmA, pmB = pms[START_PAGE], pms[FOLD_PARTNER]
        labA, regsA = label_regions(cfg, pmA)
        labB, regsB = label_regions(cfg, pmB)
        kA = sorted([r for r in regsA if r.reaches_start and not r.reaches_goal],
                    key=lambda r: -r.n_cells)[0].index
        regA = next(r for r in regsA if r.index == kA)
        print(f"[5b] doors: {n_open} open / {n_blk} blocked   (all of them are "
              f"q4 FOLD discs -- this placement has no q2 and no q6 door)")
        print(f"[5b] start region ({START_PAGE}, {kA}): {regA.n_cells} cells, "
              f"y in [{regA.y_min:.3f}, {regA.y_max:.3f}]; the fold at "
              f"y*={Y_STAR:.4f} is INSIDE it "
              f"({regA.y_min <= Y_STAR <= regA.y_max}) -- which is why the "
              f"flood has to follow it")

        fold_disc = next(d for d in sig[START_PAGE] if d.fold)
        cert = certify_fold_door(cfg, arm, weld, scene, fold_disc,
                                 pages=fold_disc.pages)
        q_fold, delta_used = fold_posture(cfg, arm, weld, cert)
        print(f"[5b] FOLD DOOR, certified EXACTLY: open={cert.open}, "
              f"exact={cert.exact}, collision-free={cert.collision_free}")
        print(f"[5b]   link drift over the whole theta line AND the q3/q5 null "
              f"motion: {cert.posture_drift_m:.2e} m  => the line is ONE posture")
        for n, v in cert.pages.items():
            print(f"[5b]   page {n}: within limits = {v['within_limits']}, "
                  f"best margin {v['best_margin_deg']:.2f} deg, admissible "
                  f"delta {v['delta_intervals']}")
        print(f"[5b]   the IK representative (delta=0) is 1.145 deg OUTSIDE the "
              f"q5 limit -- the same deficit that shuts every q2 door here "
              f"(findings 30). The null motion fixes it, and that is invariant 3.")

        # ---- the VERDICT, and the walls it is made of ---------------------
        # run_proof is asked for its polytopes (invariant 16 / todo 9b) rather
        # than retrained here for the picture: the figure then shows exactly
        # the walls the certificate contains, and the second training pass is
        # saved. Retraining the partner page independently gave gamma 57-240
        # for the same region run_proof fits at gamma 1.0.
        print(f"\n[5b] --- verdict from flood.run_proof (all pages, all doors) ---")
        collected: dict = {}
        res = run_proof(cfg, arm, weld, scene, pms, sig, verbose=True,
                        alg4_iters=args.iters, seeds=[(START_PAGE, kA)],
                        n_train_max=450, n_invalid_max=150, collect=collected)
        print(f"[5b] run_proof: proven={res.proven} -- {res.reason}")
        print(f"[5b] regions visited {len(res.regions_visited)}, walls in the "
              f"certificate {len(res.certificate)}")
        for c in res.certificate:
            print(f"[5b]   wall on {c['page']}/{c['region']}: "
                  f"{c['n_passed']}/{c['n_facets']} facets, "
                  f"closed={c['closed']}, gamma={c['gamma']:.1f}, ok={c['ok']}")
        seen = set()
        for d in res.doors_followed:
            key = (d["from"], d["to"], round(d["y"], 6))
            if key in seen:
                continue
            seen.add(key)
            print(f"[5b]   door followed ({d['kind']}): {d['from']} -> "
                  f"{d['to']} at y={d['y']:.6f}")
        t_s = lap("5b_run_proof", t_s)
        write_certificate(res, sig, out / "exp5b_certificate.json")
        print(f"[cert] {out / 'exp5b_certificate.json'}")

        polyA, repA, wA = collected[(START_PAGE, kA)]
        print(f"[5b] wall #1 on ({START_PAGE}) region {kA}: "
              f"{repA.n_passed}/{repA.n_facets} facets, closed={polyA.closed}, "
              f"loops={polyA.loop_kinds}, gamma={wA.gamma:.2f}")
        kB = next((k for (n, k) in collected if n == FOLD_PARTNER), None)
        polyB = repB = wB = None
        if kB is not None:
            polyB, repB, wB = collected[(FOLD_PARTNER, kB)]
            print(f"[5b] wall #2 on ({FOLD_PARTNER}) region {kB}: "
                  f"{repB.n_passed}/{repB.n_facets} facets, "
                  f"closed={polyB.closed}, loops={polyB.loop_kinds}, "
                  f"gamma={wB.gamma:.2f}")

        # ---- the motion the figure and the GIF show -----------------------
        routeA = bfs_route(pmA, labA == kA, (0, int(np.argmin(np.abs(
            pmA.theta_grid - 0.0)))))
        pathA = np.array([[pmA.y_grid[i], pmA.theta_grid[j]] for i, j in routeA])
        qA, keepA = qs_from_cells(arm, weld, pmA, routeA, START_SIGNS)
        print(f"[5b] the robot stalls at y = {pathA[-1, 0]:.3f}")
        pathB = None
        if kB is not None:
            # Start the partner-page walk at the fold's own row, but NOT at
            # theta = 0: the s4 mirror shifts q3 and q5 by pi, so this page's
            # forbidden theta window sits where the start page's does not --
            # which is the same fact that forces the cylinder treatment. Take
            # the cell of region kB nearest the fold row instead.
            i = int(np.argmin(np.abs(pmB.y_grid - cert.y)))
            cand = np.flatnonzero(labB[i] == kB)
            if not len(cand):
                rows = np.flatnonzero((labB == kB).any(axis=1))
                i = int(rows[np.argmin(np.abs(pmB.y_grid[rows] - cert.y))])
                cand = np.flatnonzero(labB[i] == kB)
            # Put the walk in the MIDDLE of the widest valid run of that row,
            # not at theta = 0: this page's forbidden window sits at theta ~ 0,
            # so a route started there runs along the band and is invisible
            # under it. (The whole theta line is one posture at the fold, so
            # any column of the row is an equally honest place to emerge.)
            runs, cur = [], [int(cand[0])]
            for a, b in zip(cand[:-1], cand[1:]):
                if b == a + 1:
                    cur.append(int(b))
                else:
                    runs.append(cur)
                    cur = [int(b)]
            runs.append(cur)
            best = max(runs, key=len)
            j = int(best[len(best) // 2])
            rB = bfs_route(pmB, labB == kB, (i, j))
            pathB = np.array([[pmB.y_grid[a], pmB.theta_grid[b]] for a, b in rB])
            qB, keepB = qs_from_cells(arm, weld, pmB, rB, PARTNER_SIGNS)

        walls = {START_PAGE: polyA.loops}
        kinds = {START_PAGE: polyA.loop_kinds}
        if polyB is not None:
            walls[FOLD_PARTNER] = polyB.loops
            kinds[FOLD_PARTNER] = polyB.loop_kinds
        fig8(pms, walls, kinds,
             "EXPERIMENT 5b - INFEASIBLE. The $q_4$ fold LINE door sits inside "
             "the start region, so the proof is forced onto TWO pages.",
             out / "exp5b_pages.png", fold_y=cert.y, path=pathA,
             start_mark=(0.0, 0.0),
             band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer)
        fig_fold_door(cfg, arm, weld, scene, cert,
                      out / "exp5b_fold_door.png", q_fold)
        t_s = lap("5b_figures", t_s)

        if not args.no_gif:
            fold_xyz = arm.frames(q_fold)["tool"]
            qA_d, pathA_d, keepA_d = densify(arm, weld, pmA, routeA,
                                             START_SIGNS, 34)
            segs = [
                dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                     q_list=qA_d, path=pathA_d, upto_list=keepA_d,
                     wall=polyA.loops, kinds=polyA.loop_kinds,
                     fold_y=cert.y, fold_xyz=fold_xyz, page=START_PAGE,
                     band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer,
                     caption=lambda n, p=pathA_d: (
                         f"1/3  page ({START_PAGE}): welding the seam, "
                         f"y={p[min(n, len(p) - 1), 0]:.2f}"
                         f"  -- stopped by WALL #1")),
                dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                     q_list=[q_fold] * 10, path=pathA_d,
                     upto_list=[len(pathA_d) - 1] * 10,
                     wall=polyA.loops, kinds=polyA.loop_kinds,
                     fold_y=cert.y, fold_xyz=fold_xyz, page=START_PAGE,
                     band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer,
                     caption=lambda n: (
                         f"2/3  the FOLD at y*={cert.y:.3f}: arm straight, "
                         f"elbow circle radius 0.\n"
                         f"The whole theta line is ONE posture -> the door is "
                         f"certified EXACTLY")),
            ]
            if pathB is not None:
                qB_d, pathB_d, keepB_d = densify(arm, weld, pmB, rB,
                                                 PARTNER_SIGNS, 26)
                segs.append(dict(
                    cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmB,
                    q_list=qB_d, path=pathB_d, upto_list=keepB_d,
                    wall=polyB.loops if polyB else None,
                    kinds=polyB.loop_kinds if polyB else None,
                    fold_y=cert.y, fold_xyz=fold_xyz, page=FOLD_PARTNER,
                    band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer,
                    caption=lambda n, p=pathB_d: (
                        f"3/3  page ({FOLD_PARTNER}), reached through the fold "
                        f"door: y={p[min(n, len(p) - 1), 0]:.2f}.\n"
                        f"The proof must seal this page too -- WALL #2")))
            render_gif(segs, out / "exp5b_infeasible_twopage.gif")
            t_s = lap("5b_gif", t_s)

        lap("5b_total", t_e)
        report["exp5b"] = {
            "obstacles": [{"center": list(o.center),
                           "half_extent": list(o.half_extent),
                           "label": o.label} for o in OBS_5B],
            "doors_open": int(n_open), "doors_blocked": int(n_blk),
            "start_region": {"page": START_PAGE, "index": int(kA),
                             "cells": int(regA.n_cells),
                             "y_max": float(regA.y_max),
                             "fold_inside": bool(regA.y_min <= Y_STAR <= regA.y_max)},
            "fold_certificate": {
                "y": cert.y, "open": cert.open, "exact": cert.exact,
                "collision_free": cert.collision_free,
                "posture_drift_m": cert.posture_drift_m,
                "pages": cert.pages, "reason": cert.reason},
            "wall1": {"facets": [repA.n_passed, repA.n_facets],
                      "closed": bool(polyA.closed),
                      "loop_kinds": polyA.loop_kinds, "gamma": wA.gamma},
            "wall2": (None if repB is None else
                      {"facets": [repB.n_passed, repB.n_facets],
                       "closed": bool(polyB.closed),
                       "loop_kinds": polyB.loop_kinds, "gamma": wB.gamma}),
            "theta_seam_touching_cells": seam,
            "verdict": {"proven": bool(res.proven), "reason": res.reason,
                        "regions_visited": len(res.regions_visited),
                        "certificate": res.certificate,
                        "doors_followed": res.doors_followed},
        }

    # ---------------------------------------------------------------- 5 a ---
    if "a" in args.only:
        print("\n" + "-" * 78)
        print("EXP 5a : FEASIBLE, but every plan must FLIP THE ELBOW, and the "
              "flip is free only at the fold")
        print("-" * 78)
        t_e = t_s = time.perf_counter()
        cfg, arm, weld, scene = make(OBS_5A, args.res)
        pms, sig, n_open, n_blk = pages_and_sigma(cfg, arm, weld, scene)
        t_s = lap("5a_pages_and_doors", t_s)
        seam_report(pms, " 5a")
        pmA, pmB = pms[START_PAGE], pms[FOLD_PARTNER]
        labA, regsA = label_regions(cfg, pmA)
        regA = sorted([r for r in regsA if r.reaches_start],
                      key=lambda r: -r.n_cells)[0]
        kA = regA.index
        fold_disc = next(d for d in sig[START_PAGE] if d.fold)
        cert = certify_fold_door(cfg, arm, weld, scene, fold_disc,
                                 pages=fold_disc.pages)
        q_fold, delta_used = fold_posture(cfg, arm, weld, cert)
        print(f"[5a] doors {n_open} open / {n_blk} blocked; fold door open="
              f"{cert.open} (exact={cert.exact}), margin "
              f"{cert.pages[START_PAGE]['best_margin_deg']:.2f} deg")

        # ---- what the flip costs, as a function of where it is done -------
        print(f"[5a] cost of an elbow flip (theta -> theta+pi) along the seam:")
        for yv in (0.0, 0.25, 0.40, Y_STAR, 0.60, 1.0):
            r = arm.ik_branch(*weld.pose(float(yv)), 0.0, START_SIGNS)
            print(f"[5a]   y={yv:6.4f}  elbow circle radius "
                  f"{r.elbow_circle_radius*1000:7.2f} mm  -> elbow moves "
                  f"{2*r.elbow_circle_radius*1000:7.2f} mm")

        # ---- the C1 statement: which continuation through the fold is smooth?
        print(f"[5a] continuity of the joint path through the fold "
              f"(y* -/+ 1e-4, approach along theta=0):")
        eps = 1e-4
        qm = arm.ik_branch(*weld.pose(Y_STAR - eps), 0.0, START_SIGNS).q
        cont = {}
        for nm, (th, sg) in {
                f"theta kept,   page kept     ({START_PAGE})": (0.0, START_SIGNS),
                f"theta kept,   s4 FLIPPED    ({FOLD_PARTNER})": (0.0, PARTNER_SIGNS),
                f"theta+pi,     page kept     ({START_PAGE})": (np.pi, START_SIGNS),
                f"theta+pi,     s4 FLIPPED    ({FOLD_PARTNER})": (np.pi, PARTNER_SIGNS),
        }.items():
            r = arm.ik_branch(*weld.pose(Y_STAR + eps), th, sg)
            d = (np.degrees(np.max(np.abs((r.q - qm + np.pi) % (2*np.pi) - np.pi)))
                 if r.ok else float("nan"))
            cont[nm.strip()] = float(d)
            print(f"[5a]   {nm}: max |dq| = {d:8.3f} deg"
                  + ("   <-- SMOOTH" if d < 1.0 else ""))
        print(f"[5a] => the only smooth way to CHANGE the arm angle through the "
              f"fold also changes the PAGE. Keeping the page costs 180 deg.\n"
              f"[5a]    That is new: at a q2/q6 point door nothing moves at all "
              f"and the page change is pure relabelling (findings 30).")

        # ---- two plans: single page, and through the fold door -------------
        i_star = int(np.argmin(np.abs(pmA.y_grid - Y_STAR)))
        if pmA.y_grid[i_star] > Y_STAR:
            i_star -= 1
        single = bfs_route(pmA, labA == kA, (0, int(np.argmin(np.abs(
            pmA.theta_grid - np.array([[t] for t in [0]])[0][0])))),
            goal_row=pmA.tag.shape[0] - 1)
        if single is None:
            j0 = int(np.flatnonzero(pmA.tag[0] == VALID)[0])
            single = bfs_route(pmA, labA == kA, (0, j0),
                               goal_row=pmA.tag.shape[0] - 1)
        q_single, _ = qs_from_cells(arm, weld, pmA, single, START_SIGNS)
        twop = two_page_route(pmA, pmB, i_star)
        print(f"[5a] single-page plan on ({START_PAGE}): {len(single)} cells, "
              f"total elbow path {elbow_path_length(arm, q_single)*1000:.1f} mm")
        cells_a = [(i, j) for pg, i, j in twop if pg == 0]
        cells_b = [(i, j) for pg, i, j in twop if pg == 1]
        qa, ka_ = qs_from_cells(arm, weld, pmA, cells_a, START_SIGNS)
        qb, kb_ = qs_from_cells(arm, weld, pmB, cells_b, PARTNER_SIGNS)
        th_in = pmA.theta_grid[cells_a[-1][1]]
        th_out = pmB.theta_grid[cells_b[0][1]]
        d_th = float(abs(((th_out - th_in + np.pi) % (2*np.pi)) - np.pi))
        e_in = arm.frames(qa[-1])["elbow"]
        e_out = arm.frames(qb[0])["elbow"]
        print(f"[5a] THROUGH THE FOLD DOOR: {len(cells_a)} cells on "
              f"({START_PAGE}) then {len(cells_b)} on ({FOLD_PARTNER})")
        print(f"[5a]   the door crossing changes the arm angle by "
              f"{np.degrees(d_th):.1f} deg and moves the elbow by "
              f"{np.linalg.norm(e_out-e_in)*1000:.2f} mm")
        print(f"[5a]   total elbow path of the two-page plan: "
              f"{(elbow_path_length(arm, qa)+elbow_path_length(arm, qb))*1000:.1f} mm")

        # ---- the forced-switch statement -----------------------------------
        print(f"[5a] obstacles: a post each side of the seam plane. The elbow "
              f"circle translates from -y to +y along the arc, so the -y post "
              f"blocks\n[5a]    the elbow over the FIRST half of the seam and "
              f"the +y post over the SECOND half.")
        print(f"[5a] route arm angle: {pmA.theta_grid[single[0][1]]:+.3f} -> "
              f"{pmA.theta_grid[single[-1][1]]:+.3f} rad "
              f"(swing {np.ptp([pmA.theta_grid[j] for _i, j in single]):.3f} rad "
              f"= {np.degrees(np.ptp([pmA.theta_grid[j] for _i, j in single])):.0f} deg)")

        print(f"\n[5a] --- verdict from flood.run_proof ---")
        res_a = run_proof(cfg, arm, weld, scene, pms, sig, verbose=False,
                          alg4_iters=args.iters, seeds=[(START_PAGE, kA)],
                          n_train_max=450, n_invalid_max=150)
        print(f"[5a] run_proof: proven={res_a.proven} -- {res_a.reason}")
        print(f"[5a] goal reached via {res_a.goal_reached_via!r}  => NO "
              f"infeasibility certificate exists, and that is the correct "
              f"answer: the planner returns a path.")
        write_certificate(res_a, sig, out / "exp5a_certificate.json")
        t_s = lap("5a_run_proof", t_s)

        pathA = np.array([[pmA.y_grid[i], pmA.theta_grid[j]] for i, j in cells_a])
        pathB = np.array([[pmB.y_grid[i], pmB.theta_grid[j]] for i, j in cells_b])
        fig8(pms, {}, {},
             "EXPERIMENT 5a - FEASIBLE. The plan must flip the elbow, and the "
             "flip is free only at the $q_4$ fold line; the smooth "
             "continuation through it CHANGES PAGE.",
             out / "exp5a_pages.png", fold_y=cert.y, path=pathA,
             start_mark=(0.0, float(pmA.theta_grid[cells_a[0][1]])),
             band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer)
        fig_fold_door(cfg, arm, weld, scene, cert,
                      out / "exp5a_fold_door.png", q_fold)
        t_s = lap("5a_figures", t_s)

        if not args.no_gif:
            fold_xyz = arm.frames(q_fold)["tool"]
            qa_d, patha_d, ka_d = densify(arm, weld, pmA, cells_a,
                                          START_SIGNS, 30)
            qb_d, pathb_d, kb_d = densify(arm, weld, pmB, cells_b,
                                          PARTNER_SIGNS, 30)
            segs = [
                dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                     q_list=qa_d, path=patha_d,
                     upto_list=ka_d, fold_y=cert.y, fold_xyz=fold_xyz,
                     page=START_PAGE,
                     caption=lambda n, p=patha_d: (
                         f"1/3  page ({START_PAGE}): welding the first half, "
                         f"y={p[min(n, len(p) - 1), 0]:.2f}. The -y post keeps "
                         f"the elbow on one side.")),
                dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                     q_list=[q_fold] * 12, path=patha_d,
                     upto_list=[len(patha_d) - 1] * 12, fold_y=cert.y,
                     fold_xyz=fold_xyz, page=START_PAGE,
                     caption=lambda n: (
                         f"2/3  AT THE FOLD y*={cert.y:.3f}: the arm is "
                         f"STRAIGHT, the elbow circle has radius 0.\n"
                         f"Flipping the elbow here costs 0.00 mm; at y=0 it "
                         f"would cost 230 mm.")),
                dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmB,
                     q_list=qb_d, path=pathb_d, upto_list=kb_d,
                     fold_y=cert.y, fold_xyz=fold_xyz, page=FOLD_PARTNER,
                     band=cfg.wall.virtual_band * cfg.wall.virtual_band_outer,
                     trail0=[arm.frames(q)["tool"] for q in qa_d],
                     caption=lambda n, p=pathb_d: (
                         f"3/3  page ({FOLD_PARTNER}): elbow FLIPPED, weld "
                         f"finished, y={p[min(n, len(p) - 1), 0]:.2f}.\n"
                         f"The page changed because that is the SMOOTH "
                         f"continuation (0.019 deg, vs 180 deg to stay).")),
            ]
            render_gif(segs, out / "exp5a_feasible_elbowflip.gif")
            t_s = lap("5a_gif", t_s)

        lap("5a_total", t_e)
        report["exp5a"] = {
            "obstacles": [{"center": list(o.center),
                           "half_extent": list(o.half_extent),
                           "label": o.label} for o in OBS_5A],
            "doors_open": int(n_open), "doors_blocked": int(n_blk),
            "fold_certificate": {
                "y": cert.y, "open": cert.open, "exact": cert.exact,
                "posture_drift_m": cert.posture_drift_m, "pages": cert.pages},
            "continuity_through_fold_deg": cont,
            "flip_cost_mm": {
                f"{yv:.4f}": float(2 * arm.ik_branch(
                    *weld.pose(float(yv)), 0.0, START_SIGNS
                ).elbow_circle_radius * 1000)
                for yv in (0.0, 0.25, 0.40, Y_STAR, 0.60, 1.0)},
            "door_crossing": {"dtheta_deg": float(np.degrees(d_th)),
                              "elbow_motion_mm": float(
                                  np.linalg.norm(e_out - e_in) * 1000)},
            "plan_single_page_elbow_path_mm": float(
                elbow_path_length(arm, q_single) * 1000),
            "plan_two_page_elbow_path_mm": float(
                (elbow_path_length(arm, qa) + elbow_path_length(arm, qb)) * 1000),
            "route_theta": [float(pmA.theta_grid[single[0][1]]),
                            float(pmA.theta_grid[single[-1][1]])],
            "verdict": {"proven": bool(res_a.proven), "reason": res_a.reason,
                        "goal_reached_via": res_a.goal_reached_via},
        }

    report.setdefault("timings_seconds", {}).update(timings)
    report["timings_seconds"]["script_total"] = round(time.perf_counter() - T0, 2)
    rpath.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[time] script total: {time.perf_counter() - T0:.1f} s "
          f"(res {args.res}, iters {args.iters}, only {args.only})")
    print(f"[out] {rpath}")


if __name__ == "__main__":
    main()
