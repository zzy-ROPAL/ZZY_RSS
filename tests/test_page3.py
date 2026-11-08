"""The 3-D page of a weld patch (findings 42).

What is pinned:

  * DOOR CURVES ARE EXACT AND COMPLETE. On every random v-slice, the 2-D
    analytic door finder -- independent code -- must find its doors ON our
    circles, and q_j must vanish on every curve point. A missed door is the
    one way to a false INFEASIBLE (invariant 30), so completeness is tested
    against a second method, not assumed.
  * ONLY A POSITIVE PROOF DISCHARGES A CURVE. An inexact or crossable sample
    keeps it live; so does a margin too close to its own sample-to-sample
    change.
  * THE PATCH CHECKS ARE EXACT: the |W - S| range and the axis distance
    agree with dense sampling and are never optimistic.
  * FEASIBLE MEANS A VERIFIED PATH, and the verifier still rejects a tampered
    one when the task coordinate is (u, v).
  * THE THETA CUT is refused when valid states touch it.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import zzy_rss.doors3 as D3  # noqa: E402
from test_witness import _scene, _zgap  # noqa: E402
from zzy_rss.doors import find_interior_doors_analytic  # noqa: E402
from zzy_rss.params import Page3Params, SurfaceParams, WeldParams  # noqa: E402
from zzy_rss.page3 import (FEASIBLE, Page3, choose_start_theta, prove3,  # noqa: E402
                           theta_cut_check, witness_path3)
from zzy_rss.weld import WeldSurface  # noqa: E402
from zzy_rss.witness import verify_witness  # noqa: E402

SIGNS = (1, 1, -1)


def _setup(obstacles=()):
    cfg, arm, _w, scene = _scene(tilt=0.0, x=0.28, z=0.46, half_y=0.26,
                                 obstacles=obstacles)
    return cfg, arm, scene, WeldSurface(SurfaceParams()), Page3Params()


def test_door_curves_are_exact_and_complete() -> dict:
    cfg, arm, _scene_, surf, p3 = _setup()
    curves = D3.door_curves(cfg, arm, surf, p3)
    assert curves, "no door curve on the patch"
    worst_q = 0.0
    for c in curves:
        for u, v, th in c.pts:
            r = arm.ik_branch(*surf.pose(u, v), th, SIGNS)
            assert r.ok, "no IK on a door curve point"
            worst_q = max(worst_q, abs(float(r.q[c.joint - 1])))
    assert worst_q < 1e-6, f"q_j = {worst_q} on a door curve"

    rng = np.random.default_rng(0)
    worst_gap, n_found = 0.0, 0
    for v in rng.uniform(0.0, 1.0, 40):
        line = surf.line(float(v))
        cfg_v = dataclasses.replace(cfg, weld=WeldParams(
            p0=tuple(line.p0), p1=tuple(line.p1),
            approach=surf.p.approach, reference=surf.p.reference))
        for idx in (0, 2):
            for d in find_interior_doors_analytic(cfg_v, arm, line, None, idx,
                                                  radius=0.01, r_th=0.01, name="slice"):
                n_found += 1
                same = [c for c in curves if c.sign_index == idx]
                gap = min(float(np.min(np.hypot(c.pts[:, 0] - d.exact_y,
                                                c.pts[:, 1] - v))) for c in same)
                worst_gap = max(worst_gap, gap)
                assert gap < 0.5 * min(p3.tube_radius[:2]), (
                    f"the 2-D finder has a q{2 if idx == 0 else 6} door at "
                    f"(u={d.exact_y:.4f}, v={v:.4f}) that no curve covers (gap {gap:.4f})")
    assert n_found >= 40, "the slice finder found too few doors to compare"
    return dict(curves=len(curves), max_qj=worst_q, slice_doors=n_found,
                worst_gap=worst_gap)


def test_patch_checks_are_exact() -> dict:
    cfg, arm, _s, surf, _p3 = _setup()
    sing = D3.patch_singularities(cfg, arm, surf)
    uu, vv = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    W = np.array([surf.wrist_center(u, v, arm.p.d_wt)
                  for u, v in zip(uu.ravel(), vv.ravel())])
    r = np.linalg.norm(W - arm.shoulder, axis=1)
    lo, hi = sing["w_range"]
    assert lo <= r.min() + 1e-12 and hi >= r.max() - 1e-12, "range is optimistic"
    assert abs(lo - r.min()) < 1e-3 and abs(hi - r.max()) < 1e-12
    ax = np.hypot(W[:, 0], W[:, 1]).min()
    assert sing["axis_dist"] <= ax + 1e-12
    # A patch pushed past full extension must be refused.
    far = WeldSurface(dataclasses.replace(surf.p, p0=(0.7, -0.26, 0.41),
                                          p1=(0.7, 0.26, 0.41)))
    assert not D3.patch_singularities(cfg, arm, far)["ok"]
    return dict(w_range=(lo, hi), axis_dist=sing["axis_dist"])


def test_only_a_positive_proof_discharges() -> dict:
    cfg, arm, scene, surf, p3 = _setup(_zgap())
    out = {}
    curve = D3.door_curves(cfg, arm, surf, p3)[0]
    D3.certify_curve(cfg, arm, surf, scene, curve, SIGNS, p3)
    assert curve.certified_blocked, curve.reason
    out["zgap"] = curve.reason

    # One inexact sample in the middle: the curve must stay live.
    real = D3.certify_fiber
    calls = [0]

    def flaky(*a, **k):
        calls[0] += 1
        c = real(*a, **k)
        if calls[0] == 10:
            c = dataclasses.replace(c, exact=False, reason="forced inexact")
        return c
    D3.certify_fiber = flaky
    try:
        c2 = D3.door_curves(cfg, arm, surf, p3)[0]
        D3.certify_curve(cfg, arm, surf, scene, c2, SIGNS, p3)
    finally:
        D3.certify_fiber = real
    assert not c2.certified_blocked and c2.unknown, "an inexact sample discharged a curve"
    out["inexact sample"] = c2.reason

    # A guard that cannot be met: the margin must exceed 1e9 x its own steps.
    c3 = D3.door_curves(cfg, arm, surf, p3)[0]
    D3.certify_curve(cfg, arm, surf, scene, c3, SIGNS,
                     dataclasses.replace(p3, cert_lipschitz_safety=1e9))
    assert not c3.certified_blocked
    out["tight guard"] = c3.reason
    return out


def test_feasible_is_a_verified_path() -> dict:
    cfg, arm, scene, surf, p3 = _setup()
    p3 = dataclasses.replace(p3, n_samples=2000)
    res = prove3(cfg, arm, surf, scene, SIGNS, p3, verbose=False)
    assert res.verdict == FEASIBLE, res.reason
    assert res.witness["ok"] and res.witness["swept_segments_certified"]

    th = choose_start_theta(cfg, arm, surf, scene, SIGNS, p3)
    page = Page3(cfg, arm, surf, scene, SIGNS, p3, th)
    path = witness_path3(page, page.path_to_goal())
    leg = path.legs[0]
    # Tamper: claim the path is one sample further along v than it is.
    bad_y = leg.y.copy()
    bad_y[len(bad_y) // 2, 1] += 0.05
    bad = dataclasses.replace(path, legs=[dataclasses.replace(leg, y=bad_y)])
    rep = verify_witness(cfg, arm, surf, scene, bad, q_start=path.q_start, goal_y=1.0)
    assert not rep.ok and any("off the constraint" in w for _, w in rep.failures)
    # Tamper: stop short of the goal edge.
    short = dataclasses.replace(path, legs=[dataclasses.replace(
        leg, q=leg.q[: len(leg.q) // 2], y=leg.y[: len(leg.y) // 2])])
    rep2 = verify_witness(cfg, arm, surf, scene, short, q_start=path.q_start, goal_y=1.0)
    assert not rep2.ok and any(k == "goal" for k, _ in rep2.failures)
    return dict(n=res.witness["n_configurations"],
                tool_err_mm=res.witness["max_tool_pos_err_mm"])


def test_infeasible_is_a_verified_closed_wall() -> dict:
    """ZGAP on the patch: a closed triangle wall, every facet in C_obs, the
    start inside, the goal slab exactly outside, both door curves
    discharged. And the certificate's own claims, re-checked here."""
    from zzy_rss.page3 import INFEASIBLE
    from zzy_rss.tangential import inside
    cfg, arm, scene, surf, p3 = _setup(_zgap())
    p3 = dataclasses.replace(p3, n_samples=2000)
    res = prove3(cfg, arm, surf, scene, SIGNS, p3, verbose=False)
    assert res.verdict == INFEASIBLE, res.reason
    poly = res.poly
    sc = np.asarray(p3.scale)
    # Re-check: closed + oriented surface, start inside, goal slab outside.
    from zzy_rss.tangential import closure_report
    ok, note = closure_report(poly.simplices, 2)
    assert ok, note
    start = np.asarray(res.stats["start"])
    assert inside((start * sc)[None, :], poly.vertices * sc, poly.simplices, 2)[0]
    assert poly.vertices[:, 0].max() < 1.0 - p3.goal_tol
    assert all(not d["live"] for d in res.doors)
    return dict(triangles=len(poly.simplices), gamma=res.walls[0]["gamma"],
                u_max=float(poly.vertices[:, 0].max()), seconds=res.stats["seconds"])


def test_theta_cut_is_refused_when_valid_states_touch_it() -> str:
    cfg, arm, scene, surf, p3 = _setup()
    th = choose_start_theta(cfg, arm, surf, scene, SIGNS, p3)
    # Move the cut into the middle of the valid set: theta in [th - 1, th + 1].
    p3c = dataclasses.replace(p3, n_samples=300, theta_min=th - 1.0, theta_max=th + 1.0,
                              seam_samples=500)
    page = Page3(cfg, arm, surf, scene, SIGNS, p3c, th)
    cut = theta_cut_check(page)
    assert not cut["ok"], "a cut through valid states was accepted"
    page_ok = Page3(cfg, arm, surf, scene, SIGNS,
                    dataclasses.replace(p3, n_samples=300, seam_samples=500), th)
    assert theta_cut_check(page_ok)["ok"]
    return cut["reason"]


if __name__ == "__main__":
    r = test_door_curves_are_exact_and_complete()
    print(f"[page3] door curves  : {r['curves']} curve(s), max |q_j| {r['max_qj']:.1e}; "
          f"{r['slice_doors']} slice doors from the 2-D finder all on a curve "
          f"(worst gap {r['worst_gap']:.1e})")
    r = test_patch_checks_are_exact()
    print(f"[page3] patch checks : |W-S| in [{r['w_range'][0]:.4f}, {r['w_range'][1]:.4f}], "
          f"axis {r['axis_dist']:.3f} m; never optimistic; far patch refused")
    for k, v in test_only_a_positive_proof_discharges().items():
        print(f"[page3] discharge    : {k:15s} -> {v}")
    r = test_feasible_is_a_verified_path()
    print(f"[page3] FEASIBLE     : verified path of {r['n']} configurations "
          f"(tool error {r['tool_err_mm']:.1e} mm); tampered paths rejected")
    r = test_infeasible_is_a_verified_closed_wall()
    print(f"[page3] INFEASIBLE   : ZGAP -- {r['triangles']} triangles all in C_obs, "
          f"gamma {r['gamma']}, wall ends at u={r['u_max']:.3f} (< goal), "
          f"{r['seconds']} s")
    print(f"[page3] theta cut    : refused -- {test_theta_cut_is_refused_when_valid_states_touch_it()}")
    print("[page3] ALL TESTS PASSED")
