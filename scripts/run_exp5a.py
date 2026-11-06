"""EXPERIMENT 5a -- the spec's shape: START PAGE INFEASIBLE, PARTNER PAGE REACHES
THE GOAL, and the two are joined by the q4 FOLD LINE DOOR.

What this run certifies, and what it does not:

  * On page (++-) the start posture's region CANNOT finish the weld, and that
    is PROVED: a closed learned manifold around it, every facet verified
    inside C_obs (301/301). That is a genuine single-page infeasibility
    certificate -- the same object EXP 2 and EXP 5b produce, on one page.
  * The task as a whole is FEASIBLE: flood fill follows the fold door to the
    s4 partner page (+--) and reaches y = 1 there, so run_proof returns
    proven=False. That is the CORRECT answer and the point of 5a -- the
    planner returns a path, and the path must change page at the fold.

So "5a" is not a failed 5b. It is the case where a per-page proof exists and
the global proof correctly does not.

SCENE (resolution-stable; see docs/exp5a_scene.md section 9)

    weld      arc, centre (0.15, -0.10, 0.45), radius R = 0.449530
              arc plane normal = z, arc_ref = x
              torch approach from tilt = 170 deg, azimuth = 90 deg
              span centred on the far point: [-1.4, +1.7]
              => fold line door at y* = 0.451613
    obstacle  one 12 cm cube, centre (0.2988, -0.0121, 0.3747), half 0.06
    pages     A = (++-) start, B = (+--) partner  (they differ in s4, which is
              exactly the pair the fold door joins)

R is SOLVED, not searched: with the centre and tool axis fixed, the tangency
condition hypot(dn, rho + R) = reach_max is closed form in R, so the placement
carries an exact interior fold door by construction.

WHY THIS CONFIGURATION (findings 37r, docs/exp5a_scene.md sections 10-11)

The wall sat at 96-99% for a long time, every failure "facet centre is VALID",
and the reason was measured, not guessed: the crossed facets sit in SUB-CELL
slivers of valid space near the theta seam -- 6 of 7 landed in cells the
labeller calls invalid while the exact classify() calls the facet centre
valid. The grid cannot see those slivers, so nothing in the training data
stopped the zero set from entering them.

Two settings fix it together, and neither works alone:

  boundary_anchors = 8   exact VALID postures bisected onto the TRUE class
                         boundary, 8 per cell edge, so the surface cannot
                         bulge between them (16 crossings -> 6 -> 4)
  feedback_margin = 0.25 cell diagonals, i.e. SMALL. The guard exists only to
                         keep the training set separable (findings 29), which
                         is now the anchors' job; shrinking it lets Algorithm
                         4 do its actual repair -- feed the failed facet
                         centre back and push the wall into C_obs (invariant
                         11: subdividing cannot help).

Measured: the 6 crossings at res 81 / dens 4 are IDENTICAL for guard 1.0, 1.5
and 2.0 and for alg4_iters 16 and 24, i.e. Algorithm 4 was never touching
them. Dropping the guard to 0.25 with dens 8 takes it to 301/301.

usage:
    python3 scripts/run_exp5a.py --out out/exp5a
    python3 scripts/run_exp5a.py --style bw --no-gif
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_exp5 as R
from zzy_rss.arm import VALID, SrsArm
from zzy_rss.doors import build_sigma, certify_fold_door
from zzy_rss.flood import label_regions, run_proof, write_certificate
from zzy_rss.pagemap import build_all_pages_shared
from zzy_rss.params import (BoxObstacle, Config, PageMapParams, SceneParams,
                            WeldParams)
from zzy_rss.scene import Scene
from zzy_rss.weld import WeldLine

A, A_SIGNS = "++-", (1, 1, -1)
B, B_SIGNS = "+--", (1, -1, -1)
CX, CY, CZ = 0.15, -0.10, 0.45
TILT, AZIM, SPAN_BEFORE, SPAN_AFTER = 170.0, 90.0, 1.4, 1.7
OBSTACLE = BoxObstacle(center=(0.2988, -0.0121, 0.3747),
                       half_extent=(0.06, 0.06, 0.06), label="post")


def tool_axis(tilt_deg: float, azim_deg: float) -> np.ndarray:
    t, b = np.radians(tilt_deg), np.radians(azim_deg)
    return np.array([np.sin(t) * np.cos(b), np.sin(t) * np.sin(b), -np.cos(t)])


def solve_radius(arm_p) -> tuple[float, float]:
    """The arc radius that makes the farthest seam point exactly reach_max.

    One equation, one unknown, closed form -- so |W(y) - S| - reach_max has a
    DOUBLE root and the fold door lands strictly inside the task (invariant
    22: a straight seam can never do this).
    """
    S = np.array([0.0, 0.0, arm_p.d_bs])
    g = S + arm_p.d_wt * tool_axis(TILT, AZIM)
    c = np.array([CX, CY, CZ])
    n = np.array([0.0, 0.0, 1.0])
    d = g - c
    dn = float(np.dot(d, n))
    dp = d - dn * n
    rho = float(np.linalg.norm(dp))
    rad = float(np.sqrt(arm_p.reach_max ** 2 - dn ** 2) - rho)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(n, e1)
    phi_far = float(np.arctan2(float(np.dot(dp, e2)),
                               float(np.dot(dp, e1)))) + np.pi
    return rad, phi_far


def make(res: int, anchors: int, guard_cells: float,
         routing: str = "connectivity"):
    base = Config()
    rad, phi_far = solve_radius(base.arm)
    a = np.radians(TILT)
    weld_p = WeldParams(
        p0=(CX + rad, CY, CZ), p1=(CX + rad, CY, CZ),
        approach=tuple(tool_axis(TILT, AZIM)),
        reference=(float(np.cos(a)), 0.0, float(np.sin(a))),
        curve="arc", arc_center=(CX, CY, CZ), arc_radius=rad,
        arc_normal=(0.0, 0.0, 1.0), arc_ref=(1.0, 0.0, 0.0),
        arc_phi0=phi_far - SPAN_BEFORE, arc_phi1=phi_far + SPAN_AFTER)
    diag = float(np.hypot(1.0 / (res - 1), 2.0 / (res - 1)))
    cfg = dataclasses.replace(
        base, weld=weld_p,
        # theta_periodic: the fold door forces the proof onto both s4 pages and
        # their forbidden theta windows are pi apart, so no single cut works --
        # the page must be a cylinder (invariant 24).
        pagemap=PageMapParams(n_y=res, n_theta=res, theta_periodic=True),
        scene=SceneParams(gap_open=True, obstacles=(OBSTACLE,)),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        flood=dataclasses.replace(base.flood, door_routing=routing),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz",
                                 feedback_margin=guard_cells * diag,
                                 boundary_anchors=anchors))
    return (cfg, SrsArm(cfg.arm), WeldLine(cfg.weld),
            Scene(cfg.arm, cfg.scene), rad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=81)
    ap.add_argument("--anchors", type=int, default=8,
                    help="exact boundary anchors per cell edge (findings 37r)")
    ap.add_argument("--guard", type=float, default=0.25,
                    help="feedback_margin in cell diagonals; SMALL on purpose")
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--budget", type=int, default=450)
    ap.add_argument("--style", choices=("color", "bw"), default="color")
    ap.add_argument("--no-gif", dest="no_gif", action="store_true")
    ap.add_argument("--out", default="out/exp5a")
    ap.add_argument("--door-routing", dest="door_routing",
                    default="connectivity",
                    choices=("connectivity", "wall_side"),
                    help="who decides a door's side: grid connected "
                         "components, or the verified wall itself")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sfx = "_bw" if args.style == "bw" else ""
    R.STYLE = args.style
    R.START_PAGE, R.START_SIGNS = A, A_SIGNS
    R.FOLD_PARTNER, R.PARTNER_SIGNS = B, B_SIGNS

    timings: dict[str, float] = {}
    t_all = time.perf_counter()
    t = t_all

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timings[name] = round(now - t, 3)
        t = now

    cfg, arm, weld, scene, rad = make(args.res, args.anchors, args.guard,
                                      args.door_routing)
    print(f"[cfg] door routing = {cfg.flood.door_routing}")
    y_star = SPAN_BEFORE / (SPAN_BEFORE + SPAN_AFTER)
    print(f"[5a] arc centre ({CX}, {CY}, {CZ})  R = {rad:.6f}  "
          f"tilt {TILT:.0f} azim {AZIM:.0f}   fold at y* = {y_star:.6f}")
    print(f"[5a] res {args.res}, boundary anchors {args.anchors}/edge, "
          f"feedback guard {args.guard} cell diagonals "
          f"(= {cfg.wall.feedback_margin:.4f})")

    pms = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                 verbose=False)
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    pmA, pmB = pms[A], pms[B]
    labA, regsA = label_regions(cfg, pmA)
    cands = [r for r in regsA if r.reaches_start]
    if not cands:
        print("[5a] FAILED: no start region on page A")
        return 1
    regA = sorted(cands, key=lambda r: -r.n_cells)[0]
    kA = regA.index
    print(f"[5a] page ({A}) start region {kA}: {regA.n_cells} cells, "
          f"y in [{regA.y_min:.3f}, {regA.y_max:.3f}], "
          f"reaches_goal={regA.reaches_goal}  <- must be False")
    lap("stage12")

    fold = next((d for d in sig[A] if d.fold), None)
    if fold is None:
        print("[5a] FAILED: no fold door")
        return 1
    cert = certify_fold_door(cfg, arm, weld, scene, fold, pages=fold.pages)
    q_fold = arm.ik_branch(*weld.pose(cert.y), 0.0, A_SIGNS).q
    print(f"[5a] fold LINE door: open={cert.open} exact={cert.exact} "
          f"collision_free={cert.collision_free} y={cert.y:.6f}  "
          f"drift={cert.posture_drift_m:.2e} m")
    for nm, v in cert.pages.items():
        mg = v["best_margin_deg"]
        print(f"[5a]    {nm}: within_limits={v['within_limits']}"
              + (f" margin={mg:.2f} deg" if mg is not None else ""))

    collected: dict = {}
    res_a = run_proof(cfg, arm, weld, scene, pms, sig, verbose=False,
                      alg4_iters=args.iters, seeds=[(A, kA)],
                      n_train_max=args.budget, n_invalid_max=150,
                      collect=collected)
    lap("run_proof")
    print(f"\n[5a] run_proof: proven={res_a.proven} -- {res_a.reason}")
    print(f"[5a] goal reached via {res_a.goal_reached_via!r}")

    walls, kinds, wall_ok = {}, {}, False
    if (A, kA) in collected:
        poly, rep, wmod = collected[(A, kA)]
        walls[A] = poly.loops
        kinds[A] = poly.loop_kinds
        wall_ok = bool(rep.ok and poly.closed)
        print(f"[5a] WALL on ({A})/{kA}: {rep.n_passed}/{rep.n_facets} facets "
              f"verified, closed={poly.closed}, loops={poly.loop_kinds}, "
              f"gamma={wmod.gamma:.1f}, accuracy={wmod.accuracy:.4f}")
        if wall_ok:
            print(f"[5a] => page ({A}) alone is PROVED INFEASIBLE: the start "
                  f"region is enclosed by a closed manifold lying entirely "
                  f"in C_obs.")
        else:
            reasons: dict[str, int] = {}
            for _i, why in rep.failures:
                key = why.split(";")[0]
                reasons[key] = reasons.get(key, 0) + 1
            print(f"[5a] => page ({A})'s wall is NOT a certificate yet: "
                  f"{reasons}")
    else:
        print(f"[5a] no wall was built on ({A}) -- the region reached the goal")

    write_certificate(res_a, sig, out / f"exp5a_certificate{sfx}.json")
    summary = {
        "scene": {"arc_center": [CX, CY, CZ], "arc_radius": rad,
                  "tilt_deg": TILT, "azim_deg": AZIM,
                  "span": [-SPAN_BEFORE, SPAN_AFTER],
                  "obstacle": {"center": list(OBSTACLE.center),
                               "half_extent": list(OBSTACLE.half_extent)},
                  "start_page": A, "partner_page": B},
        "settings": {"res": args.res, "boundary_anchors": args.anchors,
                     "guard_cell_diagonals": args.guard,
                     "feedback_margin": cfg.wall.feedback_margin,
                     "alg4_iters": args.iters, "n_train_max": args.budget},
        "start_region": {"index": kA, "cells": regA.n_cells,
                         "y_min": regA.y_min, "y_max": regA.y_max,
                         "reaches_goal": regA.reaches_goal},
        "fold_door": {"y": cert.y, "open": cert.open, "exact": cert.exact,
                      "collision_free": cert.collision_free,
                      "drift_m": cert.posture_drift_m},
        "start_page_wall_is_certificate": wall_ok,
        "proven": res_a.proven, "goal_reached_via": res_a.goal_reached_via,
    }
    if (A, kA) in collected:
        poly, rep, wmod = collected[(A, kA)]
        summary["start_page_wall"] = {
            "n_passed": rep.n_passed, "n_facets": rep.n_facets,
            "closed": poly.closed, "loop_kinds": list(poly.loop_kinds),
            "gamma": wmod.gamma, "accuracy": wmod.accuracy}

    # ---- the plan that uses the door ------------------------------------
    i_star = int(np.argmin(np.abs(pmA.y_grid - cert.y)))
    if pmA.y_grid[i_star] > cert.y:
        i_star -= 1
    twop = R.two_page_route(pmA, pmB, i_star)
    pathA = pathB = None
    if twop is None:
        print("[5a] no two-page route found on the grid")
    else:
        cells_a = [(i, j) for pg, i, j in twop if pg == 0]
        cells_b = [(i, j) for pg, i, j in twop if pg == 1]
        print(f"[5a] plan THROUGH THE FOLD DOOR: {len(cells_a)} cells on "
              f"({A}) then {len(cells_b)} on ({B}) to y=1")
        pathA = np.array([[pmA.y_grid[i], pmA.theta_grid[j]]
                          for i, j in cells_a])
        pathB = np.array([[pmB.y_grid[i], pmB.theta_grid[j]]
                          for i, j in cells_b])
        summary["route"] = {"cells_on_start_page": len(cells_a),
                            "cells_on_partner_page": len(cells_b)}

    band = cfg.wall.virtual_band * cfg.wall.virtual_band_outer
    verdict = ("page (%s) PROVED INFEASIBLE (%d/%d facets)" %
               (A, summary.get("start_page_wall", {}).get("n_passed", 0),
                summary.get("start_page_wall", {}).get("n_facets", 0))
               if wall_ok else "page (%s) wall incomplete" % A)
    R.fig8(pms, walls, kinds,
           f"EXPERIMENT 5a -- {verdict}; the $q_4$ FOLD LINE door at "
           f"y*={cert.y:.3f} carries the plan to ({B}), which reaches the "
           f"goal.\nrun_proof: proven={res_a.proven} via "
           f"{res_a.goal_reached_via} -- a per-page certificate exists and "
           f"the global proof correctly does not.",
           out / f"exp5a_pages{sfx}.png", fold_y=cert.y, path=pathA,
           start_mark=((0.0, float(pathA[0][1])) if pathA is not None
                       else None),
           band=band)
    R.fig_fold_door(cfg, arm, weld, scene, cert,
                    out / f"exp5a_fold_door{sfx}.png", q_fold)
    lap("figures")
    print(f"[5a] [fig] {out / ('exp5a_pages%s.png' % sfx)}")
    print(f"[5a] [fig] {out / ('exp5a_fold_door%s.png' % sfx)}")

    if not args.no_gif and twop is not None:
        fold_xyz = arm.frames(q_fold)["tool"]
        qa_d, patha_d, ka_d = R.densify(arm, weld, pmA, cells_a, A_SIGNS, 30)
        qb_d, pathb_d, kb_d = R.densify(arm, weld, pmB, cells_b, B_SIGNS, 30)
        segs = [
            dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                 q_list=qa_d, path=patha_d, upto_list=ka_d, fold_y=cert.y,
                 fold_xyz=fold_xyz, page=A, wall=walls.get(A),
                 kinds=kinds.get(A), band=band,
                 caption=lambda n, p=patha_d: (
                     f"1/3  page ({A}): the start region is ENCLOSED BY THE "
                     f"WALL, y={p[min(n, len(p)-1), 0]:.2f}.\n"
                     f"It cannot finish the weld on this page -- that is "
                     f"proved, {summary.get('start_page_wall', {}).get('n_passed', 0)}"
                     f"/{summary.get('start_page_wall', {}).get('n_facets', 0)}"
                     f" facets in $C_{{obs}}$.")),
            dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmA,
                 q_list=[q_fold] * 12, path=patha_d,
                 upto_list=[len(patha_d) - 1] * 12, fold_y=cert.y,
                 fold_xyz=fold_xyz, page=A, wall=walls.get(A),
                 kinds=kinds.get(A), band=band,
                 caption=lambda n: (
                     f"2/3  AT THE FOLD y*={cert.y:.3f}: the arm is STRAIGHT "
                     f"and the elbow circle has radius 0.\n"
                     f"The whole theta line is ONE posture, so this is the "
                     f"door -- certified EXACTLY, margin "
                     f"{max((v['best_margin_deg'] or 0) for v in cert.pages.values()):.1f} deg.")),
            dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=pmB,
                 q_list=qb_d, path=pathb_d, upto_list=kb_d, fold_y=cert.y,
                 fold_xyz=fold_xyz, page=B, band=band,
                 trail0=[arm.frames(q)["tool"] for q in qa_d],
                 caption=lambda n, p=pathb_d: (
                     f"3/3  page ({B}) -- the s4 PARTNER across the fold: "
                     f"y={p[min(n, len(p)-1), 0]:.2f} and rising to 1.00.\n"
                     f"No wall here; this page REACHES THE GOAL, so the task "
                     f"is feasible and the plan had to change page.")),
        ]
        R.render_gif(segs, out / f"exp5a{sfx}.gif")
        lap("gif")
        print(f"[5a] [gif] {out / ('exp5a%s.gif' % sfx)}")

    timings["total"] = round(time.perf_counter() - t_all, 3)
    summary["timings_seconds"] = timings
    json.dump(summary, open(out / f"exp5a_report{sfx}.json", "w"), indent=1)
    print(f"\n[5a] [out] {out / ('exp5a_report%s.json' % sfx)}")
    print(f"[5a] timings: {timings}")
    print(f"\n[5a] VERDICT: start page ({A}) wall is a certificate: "
          f"{wall_ok};  task feasible via {res_a.goal_reached_via} "
          f"(proven={res_a.proven})")
    return 0 if wall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
