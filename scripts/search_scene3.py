#!/usr/bin/env python3
"""Search 3-D (u, v, theta) weld-patch scenes for a TWO-PAGE structure (findings 45).

Wanted: from the start posture on page A the goal edge u = 1 is NOT reachable
without changing page, but a live door lies in the start region and crossing
it lands in a component of page B that DOES reach the goal.

The eight pages are eight joint spellings of one arm posture, so collision is
the same on every page and only JOINT LIMITS can make A and B differ. The q6
door (wrist flip) shifts q5 and q7 by pi, and q7's limit band |q7| > 168.45
deg is 23 deg wide, so the knob that matters most is the torch ROLL about its
own axis: it is a pure offset of q7.

Screening is grid-free (invariant 23): the pages are PRMs (Page3) with the
near-door crossing endpoints injected as vertices, exactly as flood3 uses
them, and a candidate must pass at two sample sizes. Nothing here is a proof;
run_3d.py --scene re-runs the winners through the full machinery.

    python3 scripts/search_scene3.py --jobs 4 --out out/search3.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.doors3 import door_curves, patch_singularities  # noqa: E402
from zzy_rss.flood3 import (FIBER_PAIR, door_crossing_leg, door_samples, flip,  # noqa: E402
                            page_name)
from zzy_rss.page3 import Page3, choose_start_theta, theta_cut_check  # noqa: E402
from zzy_rss.params import (BoxObstacle, Config, Page3Params, PageMapParams,  # noqa: E402
                            SceneParams, SurfaceParams)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402


def surface(x, z, tilt_deg, roll_deg, half_y=0.30, sweep=0.06, y0=0.0) -> SurfaceParams:
    """Weld along +y at (x, z), torch tilted by `tilt` about y (toward +x) and
    rolled by `roll` about its own axis; the patch is swept in z."""
    t, r = np.radians(tilt_deg), np.radians(roll_deg)
    a = np.array([np.sin(t), 0.0, -np.cos(t)])
    ref = np.array([np.cos(t), 0.0, np.sin(t)])
    ref = np.cos(r) * ref + np.sin(r) * np.cross(a, ref)
    zc = z - 0.5 * sweep
    return SurfaceParams(p0=(x, y0 - half_y, zc), p1=(x, y0 + half_y, zc),
                         sweep=(0.0, 0.0, sweep), approach=tuple(float(c) for c in a),
                         reference=tuple(float(c) for c in ref))


def boxes(spec) -> tuple:
    return tuple(BoxObstacle(center=tuple(b[:3]), half_extent=tuple(b[3:6]),
                             label=f"box{i}") for i, b in enumerate(spec or ()))


def world(spec: dict):
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True,
                                                obstacles=boxes(spec.get("boxes"))))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(surface(spec["x"], spec["z"], spec["tilt"], spec["roll"],
                               spec.get("half_y", 0.30), spec.get("sweep", 0.06),
                               spec.get("y0", 0.0)))
    return cfg, arm, scene, surf


def comp_of(page: Page3, x: np.ndarray, k: int = 8):
    """Component of a state that is not a vertex: first nearby vertex it
    connects to with a checked edge."""
    if len(page.valid) == 0:
        return None
    from scipy.spatial import cKDTree
    _, js = cKDTree(page.valid * page.scale).query(x * page.scale, k=min(k, len(page.valid)))
    for j in np.atleast_1d(js):
        if page._edge(np.asarray(x, dtype=float), page.valid[int(j)]):
            return int(page.comp[int(j)])
    return None


def screen(spec: dict, n_samples: int = 2000, seed: int = 20261108) -> list:
    """Every start page with the two-page structure, for one scene."""
    t0 = time.time()
    cfg, arm, scene, surf = world(spec)
    sing = patch_singularities(cfg, arm, surf)
    if not sing["ok"]:
        return []
    p3 = dataclasses.replace(Page3Params(), n_samples=n_samples, rng_seed=seed)
    curves = door_curves(cfg, arm, surf, p3)
    if not curves:
        return []
    ds = door_samples(curves, np.asarray(p3.scale), float(p3.cross_eps), int(p3.cert_stride))
    pages: dict = {}

    def page(sg):
        if sg not in pages:
            pages[sg] = Page3(cfg, arm, surf, scene, sg, p3, None, inject=ds.pts)
        return pages[sg]

    qsum: dict = {}

    def psum(sg, n):
        if (sg, n) not in qsum:
            x = ds.pts[n]
            r = arm.ik_branch(*surf.pose(float(x[0]), float(x[1])), float(x[2]), sg)
            i, _j, k = FIBER_PAIR[int(ds.sign[n])]
            qsum[(sg, n)] = float(r.q[i] + r.q[k]) if r.ok else np.nan
        return qsum[(sg, n)]

    out = []
    for A in SIGN_BRANCHES:
        th = choose_start_theta(cfg, arm, surf, scene, A, p3)
        if th is None:
            continue
        pa = page(A)
        x0 = np.array([p3.start_u, p3.start_v, th])
        c0 = comp_of(pa, x0)
        if c0 is None:
            continue
        goal_a = set(int(c) for c in pa.comp[pa.goal_mask()])
        if c0 in goal_a:
            continue                       # page A alone reaches the goal
        hits = {}
        for n in np.flatnonzero(pa.inject_index >= 0):
            if int(pa.comp[pa.inject_index[n]]) != c0:
                continue
            ci = ds.key[n][0]
            B = flip(A, curves[ci].sign_index)
            pb = page(B)
            goal_b = set(int(c) for c in pb.comp[pb.goal_mask()])
            n2 = int(ds.mirror[n])                  # straight through only
            vb = pb.inject_index[n2]
            if vb < 0 or int(pb.comp[vb]) not in goal_b:
                continue
            if not abs(psum(A, n) - psum(B, n2)) < 0.5:
                continue
            key = (page_name(B), f"q{curves[ci].joint}", int(pb.comp[vb]))
            hits[key] = hits.get(key, 0) + 1
        if not hits:
            continue
        (bname, joint, comp_b), n_hits = max(hits.items(), key=lambda kv: kv[1])
        B = tuple(1 if ch == "+" else -1 for ch in bname)
        pb = page(B)
        cut_a, cut_b = theta_cut_check(pa)["ok"], theta_cut_check(pb)["ok"]
        # The page-A wall must fit between the start region and the goal set
        # (A's goal components and the goal face u = 1 - goal_tol): gamma
        # ~ 1/gap^2, so this gap is what makes the first-page proof cheap.
        S0 = pa.valid[pa.comp == c0]
        gap = (1.0 - p3.goal_tol - float(S0[:, 0].max())) * float(pa.scale[0])
        G = pa.valid[np.isin(pa.comp, sorted(goal_a))]
        if len(G):
            from scipy.spatial import cKDTree
            gap = min(gap, float(cKDTree(G * pa.scale).query(S0 * pa.scale)[0].min()))
        # Crossing legs that actually build (closed-form resplit + sweep).
        legs = 0
        tried = 0
        for n in np.flatnonzero(pa.inject_index >= 0):
            if tried >= 12:
                break
            if int(pa.comp[pa.inject_index[n]]) != c0:
                continue
            n2 = int(ds.mirror[n])
            vb = pb.inject_index[n2]
            if vb < 0 or int(pb.comp[vb]) != comp_b:
                continue
            tried += 1
            leg, _why = door_crossing_leg(cfg, arm, surf, pa, pb, ds, int(n), n2)
            legs += leg is not None
        out.append(dict(spec=spec, A=page_name(A), B=bname, door=joint, start_theta=th,
                        start_comp=int(len(S0)), start_umax=round(float(S0[:, 0].max()), 3),
                        gap=round(gap, 4), hits=n_hits, legs=f"{legs}/{tried}",
                        cut_ok=bool(cut_a and cut_b), samples=n_samples, seed=seed,
                        seconds=round(time.time() - t0, 1)))
    return out


MIN_GAP = 0.05


def _job(spec):
    try:
        rows = screen(spec, n_samples=1200)
        confirmed = []
        good = [r for r in rows if r["cut_ok"] and r["gap"] >= MIN_GAP
                and not r["legs"].startswith("0/")]
        if good:
            again = screen(spec, n_samples=2500, seed=7)
            for r in good:
                m = [s for s in again if s["A"] == r["A"] and s["B"] == r["B"]]
                if m:
                    r["confirm"] = dict(hits=m[0]["hits"], gap=m[0]["gap"], legs=m[0]["legs"],
                                        start_umax=m[0]["start_umax"])
                    confirmed.append(r)
        return spec, rows, confirmed
    except Exception as e:                       # a degenerate placement
        return spec, [], [dict(error=repr(e))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="out/search3.jsonl")
    ap.add_argument("--x", type=float, nargs="+", default=[0.30, 0.38, 0.46])
    ap.add_argument("--z", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45])
    ap.add_argument("--tilt", type=float, nargs="+", default=[0.0, 30.0, 45.0, 60.0])
    ap.add_argument("--roll", type=float, nargs="+",
                    default=[float(r) for r in range(0, 360, 30)])
    ap.add_argument("--places", default=None,
                    help="explicit 'x,z,tilt;x,z,tilt;...' instead of the x/z/tilt grid")
    ap.add_argument("--all-rows", default=None, help="also write every screened row here")
    args = ap.parse_args()
    if args.places:
        places = [tuple(float(v) for v in p.split(",")) for p in args.places.split(";")]
    else:
        places = list(itertools.product(args.x, args.z, args.tilt))
    specs = [dict(x=x, z=z, tilt=t, roll=r) for (x, z, t), r in
             itertools.product(places, args.roll)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_found = 0
    with Pool(args.jobs) as pool, open(args.out, "w") as fh:
        rows_fh = open(args.all_rows, "w") if args.all_rows else None
        for i, (spec, rows, conf) in enumerate(pool.imap_unordered(_job, specs), 1):
            if rows_fh:
                for r in rows:
                    rows_fh.write(json.dumps(r) + "\n")
                rows_fh.flush()
            for r in conf:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if "error" not in r:
                    n_found += 1
                    print(f"[search3] FOUND {spec}: {r['A']} -> {r['B']} via {r['door']}, "
                          f"gap {r['gap']}/{r['confirm']['gap']}, legs {r['legs']}/"
                          f"{r['confirm']['legs']}, start u_max {r['start_umax']}", flush=True)
            if i % 8 == 0:
                print(f"[search3] {i}/{len(specs)} scenes, {n_found} found "
                      f"({time.time() - t0:.0f} s)", flush=True)
    print(f"[search3] done: {n_found} confirmed scene/page pairs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
