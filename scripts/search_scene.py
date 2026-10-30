"""Search for a scene that is infeasible by CONNECTIVITY and has an OPEN DOOR.

findings 22: an obstacle that makes a stretch of the weld unreachable on every
page at every arm angle (a "common curtain") is infeasible from the page map
alone, and the SVM wall is decoration. findings 26: the EXP 3 obstacle ZGAP
closes every door, so the proof never crosses a page boundary and EXP 3
degenerates into a second copy of EXP 2.

A scene that exercises the cross-page part of the method must satisfy ALL of:

  (A) no common curtain -- for every y, some page has at least one valid arm
      angle, so no single weld point is unreachable;
  (B) the start region still cannot reach the goal, so the valid set is split;
  (C) at least one door OUT OF THE START REGION is open, i.e. some point of
      its self-motion fiber is collision-free and within limits; and
  (D) the partner page reached through that door cannot reach the goal either
      -- otherwise the door is an escape route and the task is FEASIBLE.

(B)/(D) are checked here with grid connectivity, which per CLAUDE.md invariant
8 is a SEARCH FILTER and a spot check, never a verdict. The verdict still
comes from the learned wall, and the transitive version of (D) -- every page
reachable through any chain of open doors -- comes from flood.run_proof.

Two modes:

  --mode welds    sweep weld placements with NO obstacle and report the door
                  landscape: how many doors are open, where they sit, and how
                  much fiber margin each one has. Door position is set by
                  |W(y) - E*| = d_ew, so the placement decides where the doors
                  are before any obstacle exists.
  --mode scenes   sweep obstacle families at a fixed placement and apply the
                  (A)-(D) filter.

Usage:
  python3 scripts/search_scene.py --mode welds  --res 61 --jobs 4
  python3 scripts/search_scene.py --mode scenes --res 61 --jobs 4 \
      --families asym-zgap --weld-x 0.28 --weld-z 0.46 --tilts 30
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import label_regions, region_points  # noqa: E402
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
START_PAGE = "++-"


def signs_of(name):
    return tuple(1 if c == "+" else -1 for c in name)


def partner_page(name, vanishing):
    """The page on the other side of a door: flip every vanishing sign."""
    s = list(name)
    for idx in sorted(vanishing):
        s[idx] = "-" if s[idx] == "+" else "+"
    return "".join(s)


def weld_for(tilt_deg, x=0.28, z=0.46, half_y=0.26):
    a = np.radians(tilt_deg)
    return WeldParams(p0=(x, -half_y, z), p1=(x, +half_y, z),
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


# --------------------------------------------------------------------------
# one candidate
# --------------------------------------------------------------------------

def analyse(tilt, obs, res, weld_kw=None, with_doors=True):
    base = Config()
    cfg = dataclasses.replace(
        base, weld=weld_for(tilt, **(weld_kw or {})),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=tuple(obs)),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True))
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    per_y_best = None          # max valid cells in a y column, over pages
    start_goal_same = False    # some page has one region reaching start AND goal
    start_pages = 0
    pages, pms, labels, regions = {}, {}, {}, {}
    for n in PAGES:
        pm = build_page_map(cfg, arm, weld, scene, signs_of(n), with_kin=False)
        pms[n] = pm
        col = pm.valid_mask.sum(axis=1)
        per_y_best = col if per_y_best is None else np.maximum(per_y_best, col)
        lab, regs = label_regions(cfg, pm)
        labels[n], regions[n] = lab, regs
        if any(r.reaches_start and r.reaches_goal for r in regs):
            start_goal_same = True
        if any(r.reaches_start for r in regs):
            start_pages += 1
        pages[n] = dict(valid=float(pm.valid_mask.mean()),
                        n_regions=len(regs),
                        spanning=bool(any(r.reaches_start and r.reaches_goal
                                          for r in regs)))
    curtain = int(np.sum(per_y_best == 0))
    row = dict(curtain_cols=curtain,
               pinch=int(per_y_best.min()),
               pinch_frac=float(per_y_best.min()) / res,
               start_goal_same=bool(start_goal_same),
               start_pages=int(start_pages),
               mean_valid=float(np.mean([p["valid"] for p in pages.values()])),
               pages=pages)
    if with_doors:
        row.update(door_landscape(cfg, arm, weld, scene, pms, labels, regions))
    return row


def door_landscape(cfg, arm, weld, scene, pms, labels, regions):
    """Conditions (C) and (D): which doors are open, and where they lead.

    A door's usability is a property of its whole self-motion fiber, never of
    the single IK representative (CLAUDE.md invariant 3), so every door is
    scored with door_fiber_validity. Cheap: ~60 ms per door against ~5.7 s for
    the eight page maps, so the full 721-sample fiber is used, not a screen.
    """
    sigma = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blocked = 0
    all_doors = []
    for n in PAGES:
        sg = signs_of(n)
        for d in sigma.get(n, []):
            if not d.vanishing:
                continue
            r = door_fiber_validity(cfg, arm, weld, scene, d, sg)
            n_open += bool(r["ok"])
            n_blocked += (not r["ok"])
            all_doors.append(dict(
                page=n, y=float(d.anchor[0]), theta=float(d.anchor[1]),
                vanishing=sorted(int(v) for v in d.vanishing),
                kind=d.kind, open=bool(r["ok"]),
                radius=float(d.radius),
                valid_fraction=float(r["valid_fraction"]),
                blockers=r.get("blockers") or {}))

    # (C) open doors inside a start-touching region that cannot reach the goal,
    # and (D) where each one leads. Only the START page can hold the start
    # posture, so that is the page the proof must leave.
    pm, lab = pms[START_PAGE], labels[START_PAGE]
    tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                         pm.theta_grid[1] - pm.theta_grid[0]))
    start_regions = [r for r in regions[START_PAGE]
                     if r.reaches_start and not r.reaches_goal]
    usable = []
    for reg in sorted(start_regions, key=lambda r: -r.n_cells):
        pts = region_points(pm, lab, reg.index)
        if pts.size == 0:
            continue
        for d in all_doors:
            if d["page"] != START_PAGE or not d["open"]:
                continue
            a = np.array([d["y"], d["theta"]])
            # same reach test run_three.py uses to find the door it crosses
            if float(np.min(np.linalg.norm(pts - a, axis=1))) > max(
                    tol, d["radius"]):
                continue
            pname = partner_page(START_PAGE, d["vanishing"])
            # which partner-page region does the door land in, and can it
            # reach the goal? (one step; run_proof does the transitive version)
            pmB, labB = pms[pname], labels[pname]
            i = int(np.argmin(np.abs(pmB.y_grid - d["y"])))
            j = int(np.argmin(np.abs(pmB.theta_grid - d["theta"])))
            sub = labB[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
            ks = sorted({int(v) for v in np.unique(sub) if v > 0})
            kB = ks[0] if ks else None
            regB = next((r for r in regions[pname] if r.index == kB), None)
            usable.append(dict(
                region=int(reg.index), y=d["y"], theta=d["theta"],
                vanishing=d["vanishing"], valid_fraction=d["valid_fraction"],
                partner=pname, partner_region=kB,
                partner_valid=bool(kB is not None),
                partner_reaches_goal=bool(regB.reaches_goal) if regB else False))
    return dict(doors_open=int(n_open), doors_blocked=int(n_blocked),
                doors_start_region=len(usable),
                doors_usable=sum(1 for u in usable
                                 if u["partner_valid"]
                                 and not u["partner_reaches_goal"]),
                start_doors=usable, doors=all_doors)


def verdict(r):
    """(A) and (B) and (C) and (D)."""
    return (r["curtain_cols"] == 0
            and not r["start_goal_same"]
            and r.get("doors_usable", 0) > 0)


# --------------------------------------------------------------------------
# obstacle families
# --------------------------------------------------------------------------

def one_box():
    """One box: the family the current EXP 2 / EXP 3 obstacles come from."""
    xs = (0.10, 0.14, 0.18, 0.22, 0.26, 0.30)
    zs = (0.50, 0.54, 0.58, 0.62, 0.66)
    hs = ((0.030, 0.030, 0.030), (0.040, 0.040, 0.040),
          (0.050, 0.050, 0.050), (0.060, 0.060, 0.060),
          (0.035, 0.100, 0.035), (0.030, 0.180, 0.030),
          (0.025, 0.025, 0.080), (0.080, 0.030, 0.025))
    for x, z, h in itertools.product(xs, zs, hs):
        yield (f"1box x{x:.2f} z{z:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, z), half_extent=h, label="block"),))


def two_box_zgap():
    """Two boxes stacked in z, leaving a GAP the elbow must thread.

    theta swings the elbow around the shoulder-wrist axis, so blocking the
    high and low elbow arcs while leaving a middle channel is the natural way
    to split the valid set in theta WITHOUT making any weld point
    unreachable -- condition (A) kept, condition (B) induced.
    """
    xs = (0.14, 0.20, 0.26)
    z1s = (0.38, 0.44, 0.50)          # lower box
    z2s = (0.60, 0.66, 0.72)          # upper box
    hs = ((0.050, 0.120, 0.050), (0.040, 0.200, 0.040),
          (0.060, 0.060, 0.060), (0.035, 0.260, 0.035))
    for x, z1, z2, h in itertools.product(xs, z1s, z2s, hs):
        yield (f"2box-zgap x{x:.2f} z{z1:.2f}/{z2:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, z1), half_extent=h, label="low"),
                BoxObstacle(center=(x, 0.0, z2), half_extent=h, label="high")))


def two_box_ygap():
    """Two boxes separated along y, pinching the middle of the weld."""
    xs = (0.14, 0.20, 0.26)
    zs = (0.52, 0.58, 0.64)
    yoffs = (0.10, 0.16, 0.22)
    hs = ((0.050, 0.060, 0.070), (0.040, 0.040, 0.120),
          (0.060, 0.080, 0.050))
    for x, z, yo, h in itertools.product(xs, zs, yoffs, hs):
        yield (f"2box-ygap x{x:.2f} z{z:.2f} y+-{yo:.2f} h{h}",
               (BoxObstacle(center=(x, -yo, z), half_extent=h, label="left"),
                BoxObstacle(center=(x, +yo, z), half_extent=h, label="right")))


def three_box_slot():
    """Two in z plus one behind: a slot the arm must enter and cannot leave."""
    xs = (0.16, 0.22)
    gaps = (0.14, 0.20)
    hs = ((0.045, 0.160, 0.045), (0.035, 0.240, 0.035))
    for x, g, h in itertools.product(xs, gaps, hs):
        zc = 0.55
        yield (f"3box-slot x{x:.2f} gap{g:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, zc - g), half_extent=h, label="low"),
                BoxObstacle(center=(x, 0.0, zc + g), half_extent=h, label="high"),
                BoxObstacle(center=(x + 0.12, 0.0, zc),
                            half_extent=(0.035, h[1], g), label="back")))


def fine_1box():
    """Fine grid around the measured transition.

    At res 61 the coarse winners all turned out SPANNING (connected after
    all) or CURTAINED (unreachable), with the boundary between them lying
    between h = (0.025, 0.025, 0.080) and h = (0.040, 0.040, 0.040) near
    x = 0.14, z = 0.50-0.58. The usable scene, if it exists, is in that gap.
    """
    xs = (0.14, 0.16)
    zs = (0.50, 0.52, 0.54, 0.56, 0.58)
    hs = ((0.040, 0.040, 0.040), (0.045, 0.045, 0.045),
          (0.050, 0.050, 0.050), (0.040, 0.070, 0.040),
          (0.045, 0.070, 0.045), (0.040, 0.040, 0.060),
          (0.045, 0.045, 0.060))
    for x, z, h in itertools.product(xs, zs, hs):
        yield (f"fine x{x:.2f} z{z:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, z), half_extent=h, label="post"),))


def asym_zgap():
    """ZGAP shifted along y, off the door fiber.

    findings 26: ZGAP closes all 16 doors, and the witness is always
    collision:forearm/high -- the UPPER box sweeps through the forearm as the
    fiber swings. The doors sit near the seam ENDS (y = 0.09 and 0.91 at tilt
    30), while ZGAP sits at y = 0, dead centre. Sliding both boxes toward +y
    leaves the -y end of the seam -- and the door that lives there -- clear,
    while still blocking the elbow from crossing in the rest of the seam.
    """
    xs = (0.14, 0.18, 0.22)
    yoffs = (0.06, 0.10, 0.14, 0.18)
    zpairs = ((0.38, 0.66), (0.42, 0.62), (0.44, 0.68))
    hs = ((0.060, 0.060, 0.060), (0.050, 0.100, 0.050),
          (0.060, 0.140, 0.060))
    for x, yo, (z1, z2), h in itertools.product(xs, yoffs, zpairs, hs):
        yield (f"asym-zgap x{x:.2f} y+{yo:.2f} z{z1:.2f}/{z2:.2f} h{h}",
               (BoxObstacle(center=(x, yo, z1), half_extent=h, label="low"),
                BoxObstacle(center=(x, yo, z2), half_extent=h, label="high")))


def asym_high():
    """Keep ZGAP's low box centred; move only the HIGH box off the fiber.

    The low box never appeared as a collision witness at any door, so it is
    free to stay where it blocks the low elbow arc. Only the high box has to
    move.
    """
    xs = (0.14, 0.18)
    yoffs = (0.08, 0.12, 0.16, 0.20)
    z2s = (0.60, 0.66, 0.72)
    hs = ((0.060, 0.060, 0.060), (0.060, 0.100, 0.060))
    for x, yo, z2, h in itertools.product(xs, yoffs, z2s, hs):
        yield (f"asym-high x{x:.2f} y+{yo:.2f} z{z2:.2f} h{h}",
               (BoxObstacle(center=(x, 0.0, 0.38),
                            half_extent=(0.06, 0.06, 0.06), label="low"),
                BoxObstacle(center=(x, yo, z2), half_extent=h, label="high")))


def zgap_rel():
    """ZGAP's two-box idea, swept wide and offset along +y.

    Two boxes stacked in z block the HIGH and LOW elbow arcs while leaving a
    channel between them, which splits the valid set in theta WITHOUT making
    any weld point unreachable -- (A) kept, (B) induced (findings 24). The
    y offset is the new part: findings 26 showed ZGAP closes every door
    because the upper box sweeps the forearm at the door's fiber, so the pair
    is pushed toward +y, away from the door and toward the goal end of the
    seam that actually has to be cut off.
    """
    xs = (0.16, 0.20, 0.24, 0.28)
    yoffs = (0.06, 0.10, 0.14, 0.18)
    zpairs = ((0.30, 0.58), (0.34, 0.62), (0.38, 0.66), (0.42, 0.70))
    hs = ((0.060, 0.080, 0.060), (0.060, 0.120, 0.060),
          (0.070, 0.070, 0.070))
    for x, yo, (z1, z2), h in itertools.product(xs, yoffs, zpairs, hs):
        yield (f"zgap-rel x{x:.2f} y+{yo:.2f} z{z1:.2f}/{z2:.2f} h{h}",
               (BoxObstacle(center=(x, yo, z1), half_extent=h, label="low"),
                BoxObstacle(center=(x, yo, z2), half_extent=h, label="high")))


FAMILIES = {"1box": one_box, "2box-zgap": two_box_zgap,
            "2box-ygap": two_box_ygap, "3box-slot": three_box_slot,
            "fine": fine_1box, "asym-zgap": asym_zgap,
            "asym-high": asym_high,
            "zgap-rel": zgap_rel}

# Weld placements to scan in --mode welds. Door position is set by
# |W(y) - E*| = d_ew, so x and z move the doors along the seam and the tilt
# decides whether the fiber clears the joint limits (findings 12).
WELD_X = (0.24, 0.28, 0.32, 0.36)
WELD_Z = (0.40, 0.46, 0.52)
WELD_TILT = (0.0, 15.0, 30.0, 45.0)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _job(spec):
    tilt, tag, obs, weld_kw, res = spec
    try:
        r = analyse(tilt, obs, res, weld_kw)
    except Exception as exc:                            # keep the sweep going
        return dict(tag=tag, tilt=tilt, failed=repr(exc))
    r.update(tilt=tilt, tag=tag, weld=dict(weld_kw or {}),
             obstacles=[dict(center=list(b.center),
                             half_extent=list(b.half_extent),
                             label=b.label) for b in obs])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="scenes", choices=("scenes", "welds"))
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--out", default="out/search.json")
    ap.add_argument("--tilts", default="0,30")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--weld-x", type=float, default=0.28)
    ap.add_argument("--weld-z", type=float, default=0.46)
    ap.add_argument("--weld-half-y", type=float, default=0.26)
    ap.add_argument("--families", default="asym-zgap,asym-high,2box-zgap",
                    help="comma-separated: " + ",".join(FAMILIES))
    args = ap.parse_args()

    specs = []
    if args.mode == "welds":
        for x, z, tilt in itertools.product(WELD_X, WELD_Z, WELD_TILT):
            specs.append((tilt, f"weld x{x:.2f} z{z:.2f} tilt{tilt:.0f}", (),
                          dict(x=x, z=z, half_y=args.weld_half_y), args.res))
    else:
        weld_kw = dict(x=args.weld_x, z=args.weld_z, half_y=args.weld_half_y)
        tilts = tuple(float(t) for t in args.tilts.split(","))
        for fam in args.families.split(","):
            for tilt in tilts:
                for tag, obs in FAMILIES[fam.strip()]():
                    specs.append((tilt, tag, obs, weld_kw, args.res))

    t0 = time.time()
    print(f"[{args.mode}] {len(specs)} candidates at res {args.res}, "
          f"{args.jobs} job(s)", flush=True)

    rows = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            it = pool.imap_unordered(_job, specs)
            rows = _drain(it, len(specs), t0, args.mode)
    else:
        rows = _drain((_job(s) for s in specs), len(specs), t0, args.mode)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    ok = [r for r in rows if not r.get("failed")]
    if args.mode == "welds":
        ok.sort(key=lambda r: -r.get("doors_open", 0))
        print(f"\ndoor landscape, best first ({len(ok)} placements):")
        for r in ok[:20]:
            ds = r.get("doors") or []
            near = sorted((d for d in ds if d["open"]), key=lambda d: d["y"])
            where = ", ".join(f"y={d['y']:.3f}@{d['page']}({100*d['valid_fraction']:.0f}%)"
                              for d in near[:3])
            print(f"   {r['tag']:<34}  doors {r.get('doors_open',0):2d} open / "
                  f"{r.get('doors_blocked',0):2d} blocked   "
                  f"mean valid {100*r['mean_valid']:.1f}%   {where}")
    else:
        keep = [r for r in ok if verdict(r)]
        keep.sort(key=lambda r: (-r.get("doors_usable", 0), -r["pinch"]))
        print(f"\n{len(keep)} candidate(s) satisfy (A) no curtain, (B) no "
              f"spanning region, (C) open door out of the start region, "
              f"(D) partner cannot reach the goal -- out of {len(ok)}")
        for r in keep[:25]:
            sd = r.get("start_doors") or []
            d0 = sd[0] if sd else {}
            print(f"   tilt {r['tilt']:4.0f}  {r['tag']:<46}  "
                  f"usable-doors {r.get('doors_usable',0)}  "
                  f"door y={d0.get('y',float('nan')):.3f}->{d0.get('partner','?')}  "
                  f"fiber {100*d0.get('valid_fraction',0):.0f}%  "
                  f"pinch {r['pinch']}/{args.res}")
        if not keep:
            near = [r for r in ok if r["curtain_cols"] == 0
                    and not r["start_goal_same"]]
            print(f"   {len(near)} reached (A)+(B) but failed (C)/(D); best by "
                  f"open doors:")
            near.sort(key=lambda r: -r.get("doors_open", 0))
            for r in near[:10]:
                print(f"   tilt {r['tilt']:4.0f}  {r['tag']:<46}  "
                      f"doors {r.get('doors_open',0):2d} open, "
                      f"{r.get('doors_start_region',0)} in start region, "
                      f"{r.get('doors_usable',0)} usable")
    print(f"\n[json] {args.out}   total {time.time()-t0:.0f}s")


def _drain(it, n, t0, mode):
    rows = []
    for i, r in enumerate(it):
        rows.append(r)
        if r.get("failed"):
            print(f"   [{i}] FAILED {r['tag']}: {r['failed']}", flush=True)
        elif mode == "scenes" and verdict(r):
            sd = (r.get("start_doors") or [{}])[0]
            print(f"   *** tilt {r['tilt']:4.0f}  {r['tag']}  "
                  f"usable-doors {r.get('doors_usable')}  "
                  f"door y={sd.get('y', float('nan')):.3f} -> "
                  f"{sd.get('partner','?')}  pinch {r['pinch']}", flush=True)
        if i % 25 == 0:
            print(f"   [{i}/{n}] {time.time()-t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    main()
