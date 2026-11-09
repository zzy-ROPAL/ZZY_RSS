"""The 3-D flood's door crossings, end to end (findings 43).

Scene: EXP 2's patch with a 15 deg torch and ZGAP's high box at z = 0.675.
There the q2 door curves are live and the start region on (++-) touches
them, and a crossing leads into a (-+-) component that reaches the goal --
so a path that CHANGES PAGES can be built and checked.

What is pinned:

  * A spliced start -> door -> crossing -> goal path passes verify_witness
    (tool on the patch, limits, exact clearance, step bound), and q2 really
    changes sign across the crossing leg: the robot changes page.
  * Both kinds of crossing verify: straight through (no fiber motion) and
    with a fiber sweep (q1/q3 counter-rotating at the door posture).
  * A pair whose q1 + q3 differs by a full turn is REFUSED, not stitched.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_3d as R  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.flood3 import (FIBER_PAIR, door_crossing_leg, door_samples,  # noqa: E402
                            splice_path)
from zzy_rss.page3 import Page3, choose_start_theta  # noqa: E402
from zzy_rss.params import Config, Page3Params, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402
from zzy_rss.witness import verify_witness  # noqa: E402

A, B = (1, 1, -1), (-1, 1, -1)


def _world():
    t = np.radians(15.0)
    sp = dataclasses.replace(R.PATCHES["exp2"],
                             approach=(float(np.sin(t)), 0.0, -float(np.cos(t))),
                             reference=(float(np.cos(t)), 0.0, float(np.sin(t))))
    obs = tuple(dataclasses.replace(b, center=(b.center[0], b.center[1], 0.675))
                if b.label == "high" else b for b in R.ZGAP)
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=obs))
    arm, scene, surf = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene), WeldSurface(sp)
    p3 = dataclasses.replace(Page3Params(), n_samples=3000)
    th = choose_start_theta(cfg, arm, surf, scene, A, p3)
    ds = door_samples(door_curves(cfg, arm, surf, p3), np.asarray(p3.scale),
                      p3.cross_eps, p3.cert_stride)
    pa = Page3(cfg, arm, surf, scene, A, p3, th, inject=ds.pts)
    pb = Page3(cfg, arm, surf, scene, B, p3, None, inject=ds.pts)
    return cfg, arm, surf, scene, ds, pa, pb


def _pairs(arm, surf, ds, pa, pb):
    """(n on A in the start region, n2 on B in a goal component, delta, sum gap)."""
    goal_b = set(int(c) for c in pb.comp[pb.goal_mask()])
    out = []
    for n in np.flatnonzero(pa.inject_index >= 0):
        if pa.comp[pa.inject_index[n]] != pa.start_comp:
            continue
        for n2 in ds.same_sample(n):
            vb = pb.inject_index[n2]
            if vb < 0 or int(pb.comp[vb]) not in goal_b:
                continue
            i, _j, k = FIBER_PAIR[int(ds.sign[n])]
            qa = arm.ik_branch(*surf.pose(*ds.pts[n][:2]), ds.pts[n][2], A).q
            qb = arm.ik_branch(*surf.pose(*ds.pts[n2][:2]), ds.pts[n2][2], B).q
            out.append((int(n), int(n2), float(qb[i] - qa[i]),
                        float((qb[i] + qb[k]) - (qa[i] + qa[k]))))
    return out


def test_page_changing_paths_verify() -> dict:
    cfg, arm, surf, scene, ds, pa, pb = _world()
    pairs = _pairs(arm, surf, ds, pa, pb)
    assert pairs, "no crossing from the start region into a goal component"
    matched = [p for p in pairs if abs(p[3]) < 0.5]
    straight = [p for p in matched if p[1] == int(ds.mirror[p[0]])]
    swept = sorted([p for p in matched if p[1] != int(ds.mirror[p[0]])],
                   key=lambda p: -abs(p[2]))
    out = {}
    for label, cand in (("straight through", straight), ("fiber sweep", swept)):
        ok = None
        for n, n2, delta, _gap in cand[:40]:
            root = (A, int(pa.start_comp))
            nxt = (B, int(pb.comp[pb.inject_index[n2]]))
            path, why = splice_path({A: pa, B: pb}, ds, {root: None, nxt: (root, n, n2)},
                                    nxt, None, cfg, arm, surf)
            if path is None:
                continue
            rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start,
                                 goal_y=1.0)
            if not rep.ok:
                continue
            cross = [leg for leg in path.legs if leg.note == "crossing"][0]
            assert cross.q[0, 1] > 0.0 > cross.q[-1, 1], "q2 must change sign"
            ok = dict(n_configs=rep.n_configs, delta_deg=float(np.degrees(delta)),
                      clearance_mm=float(rep.min_clearance * 1e3))
            break
        assert ok is not None, f"no {label} crossing produced a verified path"
        out[label] = ok
    return out


def test_full_turn_mismatch_is_refused() -> dict:
    """EXP 4's q6 door: around the same door point, some departures on (-+-)
    hold q5 + q7 a full turn away from the arrival on (-++). Joining them
    would need q7 to wind through 2*pi (its limit is +/-168 deg), so the
    crossing must be refused -- and the matched ones must still be built."""
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=R.EXP4_OBS))
    arm, scene, surf = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene), WeldSurface(R.PATCHES["exp4"])
    p3 = dataclasses.replace(Page3Params(), n_samples=50)
    ds = door_samples(door_curves(cfg, arm, surf, p3), np.asarray(p3.scale),
                      p3.cross_eps, p3.cert_stride)
    pa = Page3(cfg, arm, surf, scene, (-1, 1, 1), p3, None, inject=ds.pts)
    pb = Page3(cfg, arm, surf, scene, (-1, 1, -1), p3, None, inject=ds.pts)
    refused = built = 0
    why_seen = ""
    for n in np.flatnonzero(pa.inject_index >= 0)[::3]:
        for n2 in ds.same_sample(n):
            if pb.inject_index[n2] < 0:
                continue
            i, _j, k = FIBER_PAIR[int(ds.sign[n])]
            qa = arm.ik_branch(*surf.pose(*ds.pts[n][:2]), ds.pts[n][2], pa.signs).q
            qb = arm.ik_branch(*surf.pose(*ds.pts[n2][:2]), ds.pts[n2][2], pb.signs).q
            gap = (qb[i] + qb[k]) - (qa[i] + qa[k])
            leg, why = door_crossing_leg(cfg, arm, surf, pa, pb, ds, int(n), int(n2))
            if abs(gap) > 3.0:
                assert leg is None and "full turn" in why, (gap, why)
                refused += 1
                why_seen = why
            elif abs(gap) < 0.5 and leg is not None:
                built += 1
    assert refused > 0, "no full-turn pair found to refuse"
    assert built > 0, "no matched pair was built"
    return dict(refused=refused, built=built, why=why_seen)


def test_q6_crossing_keeps_the_tool_pose() -> dict:
    """findings 45: at the exact q6 = 0 posture the IK reads q5 AND q7 from
    vanishing entries, so their SUM is atan2(0, 0) noise as well (unlike the
    q2 door, where q3 is computed relative to q1). The crossing must take
    q5 + q7 from R04^T R instead -- otherwise the 'null' sweep turns the
    torch (16 deg measured) and every q6 crossing fails verification.

    No obstacle: EXP 4's weld (x 0.40, z 0.21, 45 deg torch) swept 6 cm,
    start page (+++) -> wrist flip -> (++-)."""
    from search_scene3 import world
    cfg, arm, scene, surf = world(dict(x=0.40, z=0.21, tilt=45.0, roll=0.0))
    p3 = dataclasses.replace(Page3Params(), n_samples=1500)
    A6, B6 = (1, 1, 1), (1, 1, -1)
    ds = door_samples(door_curves(cfg, arm, surf, p3), np.asarray(p3.scale),
                      p3.cross_eps, p3.cert_stride)
    th = choose_start_theta(cfg, arm, surf, scene, A6, p3)
    pa = Page3(cfg, arm, surf, scene, A6, p3, th, inject=ds.pts)
    pb = Page3(cfg, arm, surf, scene, B6, p3, None, inject=ds.pts)
    worst, built = 0.0, 0
    for n in np.flatnonzero(pa.inject_index >= 0)[:40]:
        n2 = int(ds.mirror[n])
        if pb.inject_index[n2] < 0:
            continue
        leg, _why = door_crossing_leg(cfg, arm, surf, pa, pb, ds, int(n), n2)
        if leg is None:
            continue
        built += 1
        for q, uv in zip(leg.q, leg.y):
            p, R = arm.fk(q)
            p0, R0 = surf.pose(*uv)
            worst = max(worst, float(np.linalg.norm(R - R0)), float(np.linalg.norm(p - p0)))
    assert built and worst < 1e-6, (built, worst)      # was 0.40 rad before the fix
    goal_b = set(int(c) for c in pb.comp[pb.goal_mask()])
    assert not pa.reaches_goal(), "the start page alone must not reach the goal here"
    for n in np.flatnonzero(pa.inject_index >= 0):
        n2 = int(ds.mirror[n])
        vb = pb.inject_index[n2]
        if (pa.comp[pa.inject_index[n]] != pa.start_comp or vb < 0
                or int(pb.comp[vb]) not in goal_b):
            continue
        root, nxt = (A6, int(pa.start_comp)), (B6, int(pb.comp[vb]))
        path, _w = splice_path({A6: pa, B6: pb}, ds, {root: None, nxt: (root, int(n), n2)},
                               nxt, None, cfg, arm, surf)
        if path is None:
            continue
        rep = verify_witness(cfg, arm, surf, scene, path, q_start=path.q_start, goal_y=1.0)
        assert rep.ok, rep.failures[:3]
        cross = [leg for leg in path.legs if leg.note == "crossing"][0]
        assert cross.q[0, 5] > 0.0 > cross.q[-1, 5], "q6 must change sign"
        return dict(legs=built, worst=worst, n_configs=rep.n_configs)
    raise AssertionError("no start-region crossing into a goal component of (++-)")


if __name__ == "__main__":
    for k, v in test_page_changing_paths_verify().items():
        print(f"[flood3] {k:16s}: verified path of {v['n_configs']} configurations "
              f"across (++-) -> (-+-), fiber delta {v['delta_deg']:+.1f} deg, "
              f"clearance {v['clearance_mm']:.1f} mm")
    r = test_full_turn_mismatch_is_refused()
    print(f"[flood3] mismatch        : EXP 4 q6 door -- {r['refused']} full-turn pairs "
          f"refused ({r['why']}), {r['built']} matched pairs built")
    r = test_q6_crossing_keeps_the_tool_pose()
    print(f"[flood3] q6 wrist flip  : {r['legs']} crossing legs on the constraint "
          f"(worst {r['worst']:.1e}); verified (+++) -> (++-) path of {r['n_configs']} "
          f"configurations")
    print("[flood3] ALL TESTS PASSED")
