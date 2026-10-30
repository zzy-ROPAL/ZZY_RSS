"""Demo deliverables on the real Schunk LWA 4D.

Fixed scene for every case:
    weld      straight, x = 0.28, z = 0.46, y in [-0.26, 0.26]
    start     a SINGLE posture on page (++-) at y = 0
    goal      reach y = 1
    obstacle  cube at (0.14, 0, 0.62), half-extent 0.07      [cases B and C]

The torch tilt is the door switch, through the admissibility lemma: a q2 door
pins the whole configuration, so q5 is determined, and the s6 mirror shifts it
by pi.
    tilt  0 deg -> q5 = +/-180 deg on the paired page -> doors BLOCKED
    tilt 30 deg -> q5 moves off zero                  -> doors OPEN

    case A   no obstacle, tilt  0   robot finishes the weld   -> path + video
    case B   obstacle,    tilt  0   infeasible, doors BLOCKED -> proof, ONE page
    case C   obstacle,    tilt 30   infeasible, doors OPEN    -> proof SPANS pages
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
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import (  # noqa: E402
    find_inf_proof_region, invalid_points, label_regions, live_discs,
    other_valid_points, region_points, thin,
)
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.wall import sigma_disc_samples, virtual_boundary_samples  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
START_PAGE, START_SIGNS = "++-", (1, 1, -1)
# Two obstacles: the two cases need different things of them.
#
# The q2 door's elbow sits at (0, 0, 0.568), directly above the shoulder,
# so any obstacle near the base closes every door. Case B WANTS that
# (doors blocked -> proof inside one page). Case C must keep a door OPEN,
# so its obstacle is pushed out to x = 0.26, clear of that elbow, while
# still cutting the arm's route along the weld.
#
#   measured on page (++-) at tilt 30:
#     (0.14, 0, 0.62) h=0.07 -> does not span, 0 doors open
#     (0.26, 0, 0.58) h=0.06 -> does not span, 2 doors OPEN
OBSTACLE_B = (BoxObstacle(center=(0.14, 0.0, 0.62),
                          half_extent=(0.07, 0.07, 0.07), label="block"),)
OBSTACLE_C = (BoxObstacle(center=(0.26, 0.0, 0.58),
                          half_extent=(0.06, 0.06, 0.06), label="block"),)
TAGCMAP = ListedColormap(["#2a9d8f", "#e9e9e9", "#e9c46a", "#e76f51"])
LEG = [mpatches.Patch(color="#2a9d8f", label="valid"),
       mpatches.Patch(color="#e9e9e9", label="invalid: no IK"),
       mpatches.Patch(color="#e9c46a", label="invalid: joint limit"),
       mpatches.Patch(color="#e76f51", label="invalid: collision")]


def weld_for(tilt_deg: float) -> WeldParams:
    a = np.radians(tilt_deg)
    return WeldParams(p0=(0.28, -0.26, 0.46), p1=(0.28, 0.26, 0.46),
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def make(tilt: float, obstacle, res: int):
    base = Config()
    cfg = dataclasses.replace(
        base, weld=weld_for(tilt), pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=(tuple(obstacle) if obstacle else ())),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz"))
    return cfg, SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)


def all_pages(cfg, arm, weld, scene):
    out = {}
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        out[n] = build_page_map(cfg, arm, weld, scene, sg, with_kin=False)
    return out


# ------------------------------------------------------------- figures ----

def fig_pages(pms, sigma, title, path_overlay, out_path, walls=None):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for ax, n in zip(axes.ravel(), PAGES):
        pm = pms[n]
        ax.imshow(pm.tag.T, origin="lower", aspect="auto", cmap=TAGCMAP,
                  vmin=-0.5, vmax=3.5,
                  extent=[0, 1, pm.theta_grid[0], pm.theta_grid[-1]])
        for d in sigma.get(n, []):
            if not d.vanishing:
                continue
            ay, at = d.anchor
            st = d.__dict__.get("_state", "")
            col = "#1d3557" if st != "blocked" else "#9d0208"
            ax.plot(ay, at, "o", ms=11, mfc="none", mec=col, mew=2.4)
            if st == "blocked":
                ax.plot(ay, at, "x", ms=9, mec="#9d0208", mew=2.4)
        if walls and n in walls:
            for loop in walls[n]:
                ax.plot(loop[:, 0], loop[:, 1], color="#9d0208", lw=2.6)
        if path_overlay is not None and n == START_PAGE:
            ax.plot(path_overlay[:, 0], path_overlay[:, 1], color="#e63946", lw=3)
            ax.scatter([0], [path_overlay[0, 1]], color="#ffffff",
                       edgecolor="#111", s=90, zorder=6)
            ax.scatter([1], [path_overlay[-1, 1]], color="#111111",
                       marker="s", s=90, zorder=6)
        mark = "  <-- START PAGE" if n == START_PAGE else ""
        ax.set_title(f"page ({n})   valid {pm.stats['frac_valid'] * 100:.1f}%{mark}",
                     fontsize=11,
                     fontweight="bold" if n == START_PAGE else "normal")
        ax.set_xlabel("y  (weld coordinate)")
        ax.set_ylabel(r"$\theta$  (arm angle)")
    handles = LEG + [
        plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#1d3557",
                   mew=2.4, ms=11, label="door OPEN"),
        plt.Line2D([], [], marker="x", ls="none", mec="#9d0208", mew=2.4,
                   ms=9, label="door BLOCKED"),
        plt.Line2D([], [], color="#9d0208", lw=2.6, label="verified wall"),
        plt.Line2D([], [], color="#e63946", lw=3, label="plan"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path}")


# ---------------------------------------------------------------- path ----

def grid_path(pm, j0=None):
    m = pm.valid_mask
    ny, nt = m.shape
    starts = [(0, j) for j in range(nt) if m[0, j]]
    if j0 is not None and m[0, j0]:
        starts = [(0, j0)]
    if not starts:
        return None
    prev = {s: None for s in starts}
    q, goal = deque(starts), None
    while q:
        i, j = q.popleft()
        if i == ny - 1:
            goal = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, (j + dj) % nt          # theta is periodic
            if 0 <= ni < ny and m[ni, nj] and (ni, nj) not in prev:
                prev[(ni, nj)] = (i, j)
                q.append((ni, nj))
    if goal is None:
        return None
    out, cur = [], goal
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return np.array([[pm.y_grid[i], pm.theta_grid[j]] for i, j in out])


# --------------------------------------------------------------- video ----

def draw_box(ax, center, half):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c, h = np.asarray(center, float), np.asarray(half, float)
    lo, hi = c - h, c + h
    v = np.array([[x, y, z] for x in (lo[0], hi[0])
                  for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor="#b0b0b0",
                                         alpha=0.4, edgecolor="#505050", lw=0.7))


def make_video(cfg, arm, weld, scene, pm, path, out_path, fps=20):
    from matplotlib.animation import FFMpegWriter
    qs = [arm.ik_branch(*weld.pose(float(y)), float(t), START_SIGNS).q
          for y, t in path]
    qs = [q for q in qs if q is not None]
    fig = plt.figure(figsize=(13.5, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    p0, p1 = weld.p0, weld.p1
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(out_path), dpi=100):
        trail = []
        for k, q in enumerate(qs):
            ax.clear(); ax2.clear()
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
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col,
                        lw=7, solid_capstyle="round")
            for pt in pts[:3]:
                ax.scatter(*pt, color="#222", s=40, depthshade=False)
            trail.append(f["tool"])
            T = np.array(trail)
            ax.plot(T[:, 0], T[:, 1], T[:, 2], color="#e63946", lw=4)
            ax.scatter(*f["tool"], color="#e63946", s=95, depthshade=False,
                       edgecolor="#111", zorder=9)
            ax.set_xlim(-0.25, 0.60); ax.set_ylim(-0.42, 0.42); ax.set_zlim(0, 0.80)
            ax.set_box_aspect((0.85, 0.84, 0.80))
            ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
            ax.view_init(elev=20, azim=-72)
            ax.set_title(f"Schunk LWA 4D, page ({START_PAGE})   "
                         f"weld {100 * k / max(1, len(qs) - 1):.0f}% complete",
                         fontsize=11)
            ax2.imshow(pm.tag.T, origin="lower", aspect="auto", cmap=TAGCMAP,
                       vmin=-0.5, vmax=3.5,
                       extent=[0, 1, pm.theta_grid[0], pm.theta_grid[-1]])
            ax2.plot(path[:k + 1, 0], path[:k + 1, 1], color="#e63946", lw=2.8)
            ax2.scatter(path[k, 0], path[k, 1], color="#e63946", s=80,
                        edgecolor="#111", zorder=5)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(pm.theta_grid[0], pm.theta_grid[-1])
            ax2.set_xlabel("y  (weld coordinate)")
            ax2.set_ylabel(r"$\theta$  (arm angle)")
            ax2.set_title(r"the same motion in coordinate space $Y^+$", fontsize=11)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    print(f"[video] {out_path}  ({len(qs)} frames)")


# --------------------------------------------------------------- proof ----

def prove_region(cfg, arm, weld, scene, pms, sigma, page, k, lab, iters=9):
    pm = pms[page]
    sg = tuple(1 if c == "+" else -1 for c in page)
    rng = np.random.default_rng(0)
    pts = region_points(pm, lab, k)
    discs_all = sigma.get(page, [])
    live, discharged = live_discs(cfg, arm, weld, scene, sg, discs_all)
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    se, re_ = sigma_disc_samples(live, pts, tol)
    seed = thin(pts, 350, rng)
    rest = np.vstack([thin(other_valid_points(pm, lab, k), 350, rng),
                      virtual_boundary_samples(cfg, 50)])
    if len(se):
        seed = np.vstack([seed, se])
    if len(re_):
        rest = np.vstack([rest, re_])
    inv = thin(invalid_points(pm), 100, rng)
    poly, rep, wmodel, it, rr = find_inf_proof_region(
        cfg, arm, weld, scene, sg, seed, rest, inv, live,
        max_iters=iters, verbose=False)
    return poly, rep, live, discharged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--out", default="out/demo")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # ======================= case A : feasible ==========================
    print("\n=== CASE A : no obstacle, tilt 0 -> the robot finishes the weld ===")
    cfg, arm, weld, scene = make(0.0, None, args.res)
    pms = all_pages(cfg, arm, weld, scene)
    sigma = build_sigma(cfg, arm, weld, pms, verbose=False)
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sigma.get(n, []):
            if d.vanishing:
                d.__dict__["_state"] = ("open" if door_fiber_validity(
                    cfg, arm, weld, scene, d, sg)["ok"] else "blocked")
    pm = pms[START_PAGE]
    _lab, regs = label_regions(cfg, pm)
    spans = [r.index for r in regs if r.reaches_start and r.reaches_goal]
    path = grid_path(pm)
    print(f"[A] page ({START_PAGE}): valid {pm.stats['frac_valid'] * 100:.1f}%, "
          f"{len(regs)} region(s), spanning {spans}")
    print(f"[A] VERDICT: FEASIBLE -- plan with {len(path)} waypoints, "
          f"theta sweeps [{path[:,1].min():+.2f}, {path[:,1].max():+.2f}]")
    fig_pages(pms, sigma,
              "CASE A  no obstacle, torch tilt 0 deg  --  the robot FINISHES "
              "the weld (plan in red on the start page)",
              path, out / "A_pages_feasible.png")
    report["A"] = {"verdict": "feasible", "waypoints": int(len(path)),
                   "valid": {n: pms[n].stats["frac_valid"] for n in PAGES},
                   "spanning_regions": spans}
    if not args.skip_video:
        t = time.time()
        make_video(cfg, arm, weld, scene, pm, path, out / "A_weld.mp4")
        print(f"[A] video rendered in {time.time() - t:.0f}s")

    # ============ case B : obstacle, tilt 0, doors BLOCKED ==============
    print("\n=== CASE B : obstacle, tilt 0 -> infeasible, doors BLOCKED ===")
    cfgB, armB, weldB, sceneB = make(0.0, OBSTACLE_B, args.res)
    pmsB = all_pages(cfgB, armB, weldB, sceneB)
    sigB = build_sigma(cfgB, armB, weldB, pmsB, verbose=False)
    nb = no_ = 0
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sigB.get(n, []):
            if d.vanishing:
                ok = door_fiber_validity(cfgB, armB, weldB, sceneB, d, sg)["ok"]
                d.__dict__["_state"] = "open" if ok else "blocked"
                no_ += ok; nb += (not ok)
    pmB = pmsB[START_PAGE]
    labB, regsB = label_regions(cfgB, pmB)
    encl = [r for r in regsB if r.reaches_start and not r.reaches_goal]
    spansB = [r.index for r in regsB if r.reaches_start and r.reaches_goal]
    print(f"[B] page ({START_PAGE}): valid {pmB.stats['frac_valid'] * 100:.1f}%, "
          f"{len(regsB)} region(s), spanning {spansB}, "
          f"enclosed start region {[r.index for r in encl]}")
    print(f"[B] doors on this page: {no_} open / {nb} blocked (all pages)")
    walls_B = {}
    if encl:
        t = time.time()
        poly, rep, live, disch = prove_region(cfgB, armB, weldB, sceneB, pmsB,
                                              sigB, START_PAGE, encl[0].index, labB)
        walls_B[START_PAGE] = poly.loops
        ok = rep.ok and poly.closed
        print(f"[B] wall: {rep.n_passed}/{rep.n_facets} facets verified "
              f"({100 * rep.n_passed / max(1, rep.n_facets):.0f}%), "
              f"closed={poly.closed}, {len(disch)} disc(s) discharged as "
              f"blocked doors, {time.time() - t:.0f}s")
        print(f"[B] VERDICT: {'INFEASIBILITY PROVEN within ONE page' if ok else 'INCONCLUSIVE (wall incomplete)'}")
        report["B"] = {"verdict": "proven_one_page" if ok else "inconclusive",
                       "facets": [rep.n_passed, rep.n_facets],
                       "closed": bool(poly.closed),
                       "discs_discharged": len(disch),
                       "doors_open": int(no_), "doors_blocked": int(nb)}
    fig_pages(pmsB, sigB,
              "CASE B  obstacle, torch tilt 0 deg  --  doors BLOCKED (red x), "
              "so the proof closes INSIDE ONE PAGE",
              None, out / "B_pages_proof_onepage.png", walls=walls_B)

    # ============ case C : obstacle, tilt 30, doors OPEN ================
    print("\n=== CASE C : obstacle, tilt 30 -> infeasible, doors OPEN ===")
    cfgC, armC, weldC, sceneC = make(30.0, OBSTACLE_C, args.res)
    pmsC = all_pages(cfgC, armC, weldC, sceneC)
    sigC = build_sigma(cfgC, armC, weldC, pmsC, verbose=False)
    nb = no_ = 0
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sigC.get(n, []):
            if d.vanishing:
                ok = door_fiber_validity(cfgC, armC, weldC, sceneC, d, sg)["ok"]
                d.__dict__["_state"] = "open" if ok else "blocked"
                no_ += ok; nb += (not ok)
    labs, walls_C, visited = {}, {}, []
    for n in PAGES:
        labs[n] = label_regions(cfgC, pmsC[n])
    pmC = pmsC[START_PAGE]
    labC, regsC = labs[START_PAGE]
    enclC = [r for r in regsC if r.reaches_start and not r.reaches_goal]
    print(f"[C] page ({START_PAGE}): valid {pmC.stats['frac_valid'] * 100:.1f}%, "
          f"{len(regsC)} region(s), enclosed start "
          f"{[r.index for r in enclC]}")
    print(f"[C] doors: {no_} open / {nb} blocked")
    # follow the open door from the start region to its partner page
    partner, pk = None, None
    if enclC:
        pts = region_points(pmC, labC, enclC[0].index)
        tol = float(np.hypot(pmC.y_grid[1] - pmC.y_grid[0],
                             pmC.theta_grid[1] - pmC.theta_grid[0]))
        for d in sigC.get(START_PAGE, []):
            if not d.vanishing or d.__dict__.get("_state") != "open":
                continue
            ay, at = d.anchor
            if float(np.min(np.linalg.norm(pts - np.array([ay, at]), axis=1))) > \
                    max(tol, d.radius):
                continue
            cand = [o for o in d.pages if o != START_PAGE and o in pmsC]
            if not cand:
                continue
            partner = cand[0]
            plab, pregs = labs[partner]
            i = int(np.argmin(np.abs(pmsC[partner].y_grid - ay)))
            j = int(np.argmin(np.abs(pmsC[partner].theta_grid - at)))
            sub = plab[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
            ks = sorted({int(v) for v in np.unique(sub) if v > 0})
            if ks:
                pk = ks[0]
            print(f"[C] open door at y={ay:.4f} links ({START_PAGE}) region "
                  f"{enclC[0].index} -> ({partner}) region {pk}")
            break
    todo = [(START_PAGE, enclC[0].index)] if enclC else []
    if partner and pk:
        todo.append((partner, pk))
    facet_tot = [0, 0]
    for pg, kk in todo:
        t = time.time()
        poly, rep, live, disch = prove_region(cfgC, armC, weldC, sceneC, pmsC,
                                              sigC, pg, kk, labs[pg][0])
        walls_C[pg] = poly.loops
        facet_tot[0] += rep.n_passed; facet_tot[1] += rep.n_facets
        print(f"[C] wall on ({pg}) region {kk}: {rep.n_passed}/{rep.n_facets} "
              f"facets ({100 * rep.n_passed / max(1, rep.n_facets):.0f}%), "
              f"closed={poly.closed}, {time.time() - t:.0f}s")
    print(f"[C] VERDICT: proof SPANS {len(todo)} page(s); "
          f"facets {facet_tot[0]}/{facet_tot[1]}")
    report["C"] = {"pages_walled": [p for p, _ in todo],
                   "facets": facet_tot, "doors_open": int(no_),
                   "doors_blocked": int(nb),
                   "door_link": f"{START_PAGE}->{partner}" if partner else None}
    fig_pages(pmsC, sigC,
              "CASE C  obstacle, torch tilt 30 deg  --  doors OPEN, so the "
              "proof must CROSS a door and wall BOTH pages",
              None, out / "C_pages_proof_twopage.png", walls=walls_C)

    (out / "demo_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[out] {out / 'demo_report.json'}")


if __name__ == "__main__":
    main()
