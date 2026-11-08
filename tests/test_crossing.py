"""Doors are crossed STRAIGHT THROUGH; only posture facts discharge (findings 43).

What is pinned:

  * THE ARTIFACT. At a q2 door the q1/q3 axes coincide, and the IK's q1 there
    is atan2(0, 0): 0 on page (++-), -180 deg on (-+-). One step off the door
    the two pages agree to ~1e-3 rad -- the "half a turn apart" is not real.
  * THE CROSSING. Straight lines through EXP 3's q2 door, on (++-) before the
    door and (-+-) after it, pass verify_witness (tool on the weld, limits,
    exact clearance, step bound) in at least 20 of 24 directions. The legacy
    rule calls this door "short by 1.15 deg -> NOT crossable".
  * THE RULE. door_passable no longer discharges that door; it still
    discharges EXP 2's, whose door posture is in collision.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_witness import _doors, _scene, _zgap, _zgap_y  # noqa: E402
from zzy_rss.doors import door_passable  # noqa: E402
from zzy_rss.witness import LEG_PAGE, Leg, WitnessPath, verify_witness  # noqa: E402

A, B = (1, 1, -1), (-1, 1, -1)


def _exp3():
    cfg, arm, weld, scene = _scene(tilt=15.0, x=0.32, z=0.40, half_y=0.26,
                                   obstacles=_zgap_y())
    door = [d for d in _doors(cfg, arm, weld, scene, "++-") if 0 in d.vanishing][0]
    return cfg, arm, weld, scene, door


def test_the_half_turn_is_an_artifact() -> dict:
    _cfg, arm, weld, _scene_, door = _exp3()
    y0, t0 = door.anchor
    at_a = arm.ik_branch(*weld.pose(y0), t0, A).q
    at_b = arm.ik_branch(*weld.pose(y0), t0, B).q
    gap_at = abs(np.degrees(at_a[0] - at_b[0]))
    worst = 0.0
    for a in np.radians(np.arange(15, 360, 30)):
        d = 1e-5 * np.array([np.cos(a), 6.0 * np.sin(a)])
        qa = arm.ik_branch(*weld.pose(y0 + d[0]), t0 + d[1], A).q
        qb = arm.ik_branch(*weld.pose(y0 - d[0]), t0 - d[1], B).q
        worst = max(worst, float(np.abs(qa - qb).max()))
    assert abs(gap_at - 180.0) < 1e-6, gap_at
    assert worst < 1e-3, worst
    return dict(q1_gap_at_door_deg=gap_at, worst_gap_off_door_rad=worst)


def test_straight_through_crossings_verify() -> tuple:
    cfg, arm, weld, scene, door = _exp3()
    y0, t0 = door.anchor
    ok = 0
    for a in np.radians(np.arange(0, 360, 15)):
        d = np.array([0.002 * np.cos(a), 0.012 * np.sin(a)])
        ts = np.r_[np.linspace(1.0, 1e-7, 150), -np.linspace(1e-7, 1.0, 150)]
        qs, ys = [], []
        for t in ts:
            y, th = np.array([y0, t0]) + t * d
            qs.append(arm.ik_branch(*weld.pose(y), th, A if t > 0 else B).q)
            ys.append(y)
        qs = np.asarray(qs)
        assert qs[0, 1] > 0 > qs[-1, 1], "the path must change the sign of q2"
        rep = verify_witness(cfg, arm, weld, scene,
                             WitnessPath(legs=[Leg(kind=LEG_PAGE, page=A, q=qs,
                                                   y=np.asarray(ys))]))
        ok += rep.ok
    assert ok >= 20, f"only {ok}/24 straight-through crossings verify"
    return ok, 24


def test_only_posture_facts_discharge() -> dict:
    cfg, arm, weld, scene, door = _exp3()
    now = door_passable(cfg, arm, weld, scene, door, A)
    legacy = door_passable(dataclasses.replace(cfg, crossing=dataclasses.replace(
        cfg.crossing, discharge="fiber_interval")), arm, weld, scene, door, A)
    assert not now["certified_blocked"] and now["passable"], now["reason"]
    assert legacy["certified_blocked"], "legacy rule should reproduce the old verdict"
    cfg2, arm2, weld2, scene2 = _scene(tilt=0.0, x=0.28, z=0.46, half_y=0.26,
                                       obstacles=_zgap())
    blocked = [door_passable(cfg2, arm2, weld2, scene2, d, A)
               for d in _doors(cfg2, arm2, weld2, scene2, "++-") if 0 in d.vanishing]
    assert blocked and all(b["certified_blocked"] for b in blocked)
    assert all("collision" in b["reason"] for b in blocked)
    return dict(exp3_now=now["reason"][:80], exp3_legacy=legacy["reason"][:80],
                exp2=blocked[0]["reason"][:80])


if __name__ == "__main__":
    r = test_the_half_turn_is_an_artifact()
    print(f"[crossing] artifact : q1 differs by {r['q1_gap_at_door_deg']:.2f} deg AT the "
          f"door, by <= {r['worst_gap_off_door_rad']:.1e} rad one step off it")
    ok, n = test_straight_through_crossings_verify()
    print(f"[crossing] crossing : {ok}/{n} straight-through crossings of EXP 3's q2 "
          f"door pass verify_witness")
    for k, v in test_only_posture_facts_discharge().items():
        print(f"[crossing] rule     : {k:10s} -> {v}")
    print("[crossing] ALL TESTS PASSED")
