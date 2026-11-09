#!/usr/bin/env python3
"""Design aid: place a box so that page A is walled off THICKLY (findings 45).

Why a box is needed at all. On a weld patch the q7 limit band of a page is
bounded by the q6 door curves: around a door q5 and q7 take every value, so
every band and sliver is a sheet that ends on a door. Without obstacles the
band therefore always separates page A *through a door point*, the start and
goal regions of A touch there, and no wall can exist (search_scene3: gap
0.0031 -- the door-sample spacing -- in every obstacle-free scene).

The fix: let a box (the same on every page) seal everything between start and
goal except one corridor, and choose the torch roll so the THICK far end of
A's q7 band plugs that corridor. On page B the wrist flip moves the band away
and the corridor is open.

This screen works on v-slices with a grid. Grid connectivity is NOT a proof
and not even a reliable screen (invariant 23): it is used only to rank box
placements; every winner is re-checked by search_scene3.screen (PRM, two
sample sizes) and then by the full proofs (run_3d.py --two-page).

    python3 scripts/design_obstacle3.py --jobs 4 --out out/design3.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_scene3 as S  # noqa: E402
from zzy_rss.batch import classify_surface_batch  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.page3 import choose_start_theta  # noqa: E402
from zzy_rss.params import Page3Params  # noqa: E402

NU, NT = 121, 181
US = np.linspace(0.0, 1.0, NU)
TS = np.linspace(-np.pi, np.pi, NT)


def slice_valid(cfg, arm, scene, surf, signs, v):
    U, T = np.meshgrid(US, TS, indexing="ij")
    P = np.column_stack([U.ravel(), np.full(U.size, v), T.ravel()])
    return classify_surface_batch(cfg, arm, surf, scene, P, signs)[0].reshape(NU, NT)


def evaluate(spec: dict, A=(1, 1, 1), B=(1, 1, -1), vs=(0.0, 0.25, 0.5, 0.75, 1.0),
             goal_tol: float = 0.02) -> dict:
    cfg, arm, scene, surf = S.world(spec)
    p3 = Page3Params()
    th0 = choose_start_theta(cfg, arm, surf, scene, A, p3)
    if th0 is None:
        return dict(ok=False, why="no start posture")
    curves = [c for c in door_curves(cfg, arm, surf, p3) if c.joint == 6]
    if not curves:
        return dict(ok=False, why="no q6 door")
    du, dt = US[1] - US[0], (TS[1] - TS[0]) / np.pi
    gaps, b_ok, door_ok = [], [], []
    for v in vs:
        va, vb = slice_valid(cfg, arm, scene, surf, A, v), slice_valid(cfg, arm, scene, surf, B, v)
        la, _ = ndimage.label(va)
        lb, _ = ndimage.label(vb)
        # start: the column u = 0 at the start theta (or the valid cell nearest it)
        col = np.flatnonzero(va[0])
        if not len(col):
            return dict(ok=False, why=f"u=0 invalid at v={v}")
        j0 = col[np.argmin(np.abs(TS[col] - th0))]
        s_lab = la[0, j0]
        smask = la == s_lab
        goal_cols = US >= 1.0 - goal_tol
        if smask[goal_cols].any():
            return dict(ok=False, why=f"page A reaches the goal at v={v}")
        # The wall's two classes as region_wall builds them: INSIDE is the
        # start region plus every region touching a live door it holds (the
        # door tube goes in the seed class, and a pocket that meets the start
        # region only at the door point must then be enclosed too); OUTSIDE is
        # every region that reaches the goal, and the goal face itself.
        inside_m = smask.copy()
        for c in curves:
            k = int(np.argmin(np.abs(c.pts[:, 1] - v)))
            i = int(round(c.pts[k, 0] / du))
            j = int(round((c.pts[k, 2] + np.pi) / (TS[1] - TS[0])))
            if not (0 <= i < NU and 0 <= j < NT):
                continue
            win = la[max(0, i - 3):i + 4, max(0, j - 3):j + 4]
            if (win == s_lab).any():
                for lab in np.unique(win[win > 0]):
                    inside_m |= la == lab
        if inside_m[goal_cols].any():
            return dict(ok=False, why=f"a door pocket reaches the goal at v={v}")
        goal_labs = set(np.unique(la[goal_cols][la[goal_cols] > 0]).tolist())
        gmask = np.isin(la, sorted(goal_labs))
        dist = ndimage.distance_transform_edt(~inside_m, sampling=(du, dt))
        gap = (1.0 - goal_tol) - float(US[np.any(inside_m, axis=1)].max())
        if gmask.any():
            gap = min(gap, float(dist[gmask].min()))
        gaps.append(gap)
        # the live door in the start region, and where crossing it lands on B
        best = False
        dok = False
        for c in curves:
            k = int(np.argmin(np.abs(c.pts[:, 1] - v)))
            u, th = c.pts[k, 0], c.pts[k, 2]
            i, j = int(round(u / du)), int(round((th + np.pi) / (TS[1] - TS[0])))
            if not (2 <= i < NU - 2 and 2 <= j < NT - 2):
                continue
            ring = [(i + a, j + b) for a in (-2, 0, 2) for b in (-2, 0, 2) if (a, b) != (0, 0)]
            on_a = [(a, b) for a, b in ring if smask[a, b]]
            if not on_a:
                continue
            dok = True
            for a, b in on_a:                          # straight through: mirror cell
                a2, b2 = 2 * i - a, 2 * j - b
                if vb[a2, b2] and (lb[goal_cols] == lb[a2, b2]).any():
                    best = True
        door_ok.append(dok)
        b_ok.append(best)
    return dict(ok=bool(all(door_ok) and all(b_ok)), gap=round(min(gaps), 4),
                gaps=[round(g, 4) for g in gaps], door=door_ok, b=b_ok)


def _job(spec):
    try:
        return spec, evaluate(spec)
    except Exception as e:  # noqa: BLE001
        return spec, dict(ok=False, why=repr(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="out/design3.jsonl")
    ap.add_argument("--place", default="0.40,0.21,45")
    ap.add_argument("--roll", type=float, nargs="+", default=[240.0, 255.0, 270.0, 285.0, 300.0])
    ap.add_argument("--bx", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--by", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--bz", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.45, 0.6])
    ap.add_argument("--half", default="0.06,0.06,0.06;0.10,0.10,0.10;0.15,0.05,0.15;0.05,0.15,0.15")
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.06],
                    help="patch width in z (the v extent)")
    args = ap.parse_args()
    x, z, tilt = (float(c) for c in args.place.split(","))
    halves = [tuple(float(c) for c in h.split(",")) for h in args.half.split(";")]
    specs = [dict(x=x, z=z, tilt=tilt, roll=r, sweep=sw, boxes=[[bx, by, bz, *h]])
             for r, bx, by, bz, h, sw in itertools.product(args.roll, args.bx, args.by, args.bz,
                                                           halves, args.sweep)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0, n_ok = time.time(), 0
    with Pool(args.jobs) as pool, open(args.out, "w") as fh:
        for i, (spec, res) in enumerate(pool.imap_unordered(_job, specs), 1):
            fh.write(json.dumps(dict(spec=spec, **res)) + "\n")
            if res.get("ok") and res.get("gap", 0) > 0.03:
                n_ok += 1
                print(f"[design3] gap {res['gap']:.3f} {spec}", flush=True)
            if i % 100 == 0:
                print(f"[design3] {i}/{len(specs)} ({time.time() - t0:.0f} s), {n_ok} usable",
                      flush=True)
    print(f"[design3] done: {n_ok} usable of {len(specs)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
