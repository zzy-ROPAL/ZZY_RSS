"""IK round-trip and branch-structure tests (Stage 0 acceptance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SIGN_BRANCHES, SrsArm, rot_y, rot_z  # noqa: E402
from zzy_rss.params import DEFAULT  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def test_round_trip() -> tuple[int, int, float]:
    """FK(IK(pose, psi, signs)) must reproduce the pose on every branch."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(7)

    n_ok = 0
    n_tried = 0
    worst = 0.0
    for _ in range(400):
        y = float(rng.uniform(0.0, 1.0))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            n_tried += 1
            if not res.ok:
                continue
            pos_fk, rot_fk = arm.fk(res.q)
            e_pos = float(np.linalg.norm(pos_fk - pos))
            e_rot = float(np.linalg.norm(rot_fk - rot))
            err = max(e_pos, e_rot)
            worst = max(worst, err)
            assert err < 1e-8, f"round-trip failed: {err:.3e} at y={y} psi={psi} {signs}"
            n_ok += 1
    return n_ok, n_tried, worst


def test_sign_consistency() -> float:
    """The requested signs must match the signs of the returned joint angles."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        y = float(rng.uniform(0.05, 0.95))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            if not res.ok:
                continue
            q = res.q
            for idx, want in zip((1, 3, 5), signs):
                got = q[idx]
                if abs(got) < 1e-7:
                    continue
                assert np.sign(got) == want, (
                    f"sign mismatch at q{idx + 1}: wanted {want}, got {got:.4f}"
                )
            worst = max(worst, abs(q[1]), abs(q[3]), abs(q[5]))
    return worst


def test_branch_geometry_identical() -> float:
    """All eight branches must place shoulder, elbow, wrist and tool alike."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(100):
        y = float(rng.uniform(0.05, 0.95))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        pts = []
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            if res.ok:
                f = arm.frames(res.q)
                pts.append(np.concatenate([f["elbow"], f["wrist"], f["tool"]]))
        if len(pts) > 1:
            ref = pts[0]
            for other in pts[1:]:
                worst = max(worst, float(np.linalg.norm(other - ref)))
    assert worst < 1e-8, f"branches place geometry differently: {worst:.3e}"
    return worst


def test_degeneracy_geometry() -> None:
    """q2 = 0 must pin the elbow; q4 = 0 must be full extension."""
    arm = SrsArm(DEFAULT.arm)
    p = DEFAULT.arm

    # q2 = 0: elbow sits on the joint-1 axis at height d_bs + d_se,
    # independently of q1 and q3.
    for q1 in (-1.0, 0.0, 2.0):
        for q3 in (-2.0, 0.5, 3.0):
            q = np.array([q1, 0.0, q3, 1.0, 0.2, 0.7, -0.3])
            f = arm.frames(q)
            expected = np.array([0.0, 0.0, p.d_bs + p.d_se])
            assert np.linalg.norm(f["elbow"] - expected) < 1e-12

    # q2 = 0: the gimbal arc leaves the tool pose fixed when q1 + q3 is held.
    base = np.array([0.4, 0.0, -0.9, 1.1, 0.3, 0.8, -0.2])
    pos0, rot0 = arm.fk(base)
    for delta in (0.3, -0.7, 1.9):
        q = base.copy()
        q[0] += delta
        q[2] -= delta
        pos1, rot1 = arm.fk(q)
        assert np.linalg.norm(pos1 - pos0) < 1e-12, "gimbal arc moved the tool"
        assert np.linalg.norm(rot1 - rot0) < 1e-12, "gimbal arc rotated the tool"

    # q4 = 0: shoulder-to-wrist distance equals full reach.
    q = np.array([0.3, 0.6, -0.4, 0.0, 0.1, 0.2, 0.3])
    f = arm.frames(q)
    assert abs(np.linalg.norm(f["wrist"] - f["shoulder"]) - p.reach_max) < 1e-12

    # q6 = 0: the wrist gimbal leaves the tool pose fixed when q5 + q7 is held.
    base = np.array([0.2, 0.9, -0.5, 1.0, 0.6, 0.0, -0.4])
    pos0, rot0 = arm.fk(base)
    for delta in (0.4, -1.1):
        q = base.copy()
        q[4] += delta
        q[6] -= delta
        pos1, rot1 = arm.fk(q)
        assert np.linalg.norm(pos1 - pos0) < 1e-12, "wrist gimbal moved the tool"
        assert np.linalg.norm(rot1 - rot0) < 1e-12, "wrist gimbal rotated the tool"


def test_mirror_identities() -> None:
    """s2/s4/s6 act as the stated representational mirrors."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    pos, rot = weld.pose(0.5)
    psi = 0.8

    pairs = [
        ((+1, +1, +1), (-1, +1, +1), (0, 1, 2)),
        ((+1, +1, +1), (+1, -1, +1), (2, 3, 4)),
        ((+1, +1, +1), (+1, +1, -1), (4, 5, 6)),
    ]
    for a, b, idx in pairs:
        ra = arm.ik_branch(pos, rot, psi, a)
        rb = arm.ik_branch(pos, rot, psi, b)
        if not (ra.ok and rb.ok):
            continue
        i, j, k = idx
        # Middle angle negates; the outer two shift by pi (mod 2 pi).
        assert abs(rb.q[j] + ra.q[j]) < 1e-9, f"middle angle not negated for {b}"
        for m in (i, k):
            d = (rb.q[m] - ra.q[m]) % (2.0 * np.pi)
            assert min(abs(d - np.pi), abs(d + np.pi - 2 * np.pi)) < 1e-9, (
                f"outer angle q{m + 1} not shifted by pi for {b}: delta={d:.6f}"
            )


def test_shared_page_build_matches() -> int:
    """build_all_pages_shared must be EXACT, not an approximation.

    It replaces 8 x (IK + FK + collision) per cell with 1 x (IK + FK +
    collision) plus 8 joint-limit comparisons, on the strength of findings 31:
    IK existence and collision are page-independent, and the mirror identities
    give the other seven joint vectors in closed form. If that is right the two
    builders agree in every cell of every page; if it is wrong this test says
    where.
    """
    import dataclasses

    from zzy_rss.pagemap import build_all_pages, build_all_pages_shared
    from zzy_rss.params import BoxObstacle, PageMapParams, SceneParams
    from zzy_rss.scene import Scene

    worst = 0
    for obstacles in (
        (),
        (BoxObstacle(center=(0.14, 0.0, 0.38), half_extent=(0.06, 0.06, 0.06)),
         BoxObstacle(center=(0.14, 0.0, 0.66), half_extent=(0.06, 0.06, 0.06))),
    ):
        cfg = dataclasses.replace(
            DEFAULT,
            pagemap=PageMapParams(n_y=31, n_theta=31),
            scene=SceneParams(gap_open=True, obstacles=obstacles),
        )
        arm = SrsArm(cfg.arm)
        weld = WeldLine(cfg.weld)
        scene = Scene(cfg.arm, cfg.scene)
        a = build_all_pages(cfg, arm, weld, scene, with_kin=False,
                            verbose=False, shared=False)
        b = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
        assert set(a) == set(b), "page sets differ"
        for name in a:
            diff = int(np.sum(a[name].tag != b[name].tag))
            worst = max(worst, diff)
            assert diff == 0, (
                f"page {name}: shared build differs in {diff} cells"
            )
            for field in ("q2", "q4", "q6", "limit_margin"):
                x = np.nan_to_num(getattr(a[name], field))
                y = np.nan_to_num(getattr(b[name], field))
                assert np.max(np.abs(x - y)) < 1e-9, (
                    f"page {name}: {field} differs"
                )
    return worst


def test_fasteval_matches_page_build() -> int:
    """fasteval.Precomputed must agree with the real page build, cell by cell.

    It is a search accelerator, so it is only useful if it is EXACT. It skips
    the per-obstacle inverse kinematics on the strength of findings 31, and
    this is the test that keeps that honest.
    """
    import dataclasses

    from zzy_rss.fasteval import Precomputed
    from zzy_rss.pagemap import build_all_pages_shared
    from zzy_rss.params import BoxObstacle, PageMapParams, SceneParams
    from zzy_rss.scene import Scene

    obstacles = (
        BoxObstacle(center=(0.14, 0.0, 0.38), half_extent=(0.06, 0.06, 0.06)),
        BoxObstacle(center=(0.14, 0.0, 0.66), half_extent=(0.06, 0.06, 0.06)),
    )
    worst = 0
    for obs in ((), obstacles):
        cfg = dataclasses.replace(
            DEFAULT, pagemap=PageMapParams(n_y=31, n_theta=31),
            scene=SceneParams(gap_open=True, obstacles=obs))
        arm = SrsArm(cfg.arm)
        weld = WeldLine(cfg.weld)
        truth = build_all_pages_shared(cfg, arm, weld, Scene(cfg.arm, cfg.scene),
                                       with_kin=False, verbose=False)
        pre = Precomputed(cfg)
        fast = pre.all_pages(pre.collide(obs))
        assert set(truth) == set(fast), "page sets differ"
        for name in truth:
            diff = int(np.sum(truth[name].tag != fast[name].tag))
            worst = max(worst, diff)
            assert diff == 0, f"page {name}: fasteval differs in {diff} cells"
    return worst


def test_unevaluable_doors_are_left_open() -> int:
    """A door we cannot evaluate must be left OPEN, never discharged.

    live_discs reads door_fiber_validity()["ok"] == False as "certified
    blocked" and DISCHARGES the disc: the wall is then free to cross it and
    run_proof stops following it. Both are the unsound direction, so the two
    cases where the fiber cannot be evaluated at all -- no IK at the anchor,
    and a disc whose vanishing set names no characterised fiber -- must both
    come back ok=True (invariant 2). certify_fiber has always done this; this
    test pins door_fiber_validity to agree with it.

    Also checks the q4 fold branch, whose certification is EXACT (the whole
    theta-line is one configuration) rather than sampled.
    """
    import dataclasses
    from zzy_rss.doors import SigmaDisc, DOOR_FOLD, SIGMA_COORD, door_fiber_validity
    from zzy_rss.flood import live_discs
    from zzy_rss.scene import Scene

    cfg = DEFAULT
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)
    signs = (1, 1, -1)
    cfg_cert = dataclasses.replace(cfg, sigma_policy=dataclasses.replace(
        cfg.sigma_policy, enable_fiber_certification=True))

    checked = 0
    # (a) vanishing set names no characterised fiber -> must be left OPEN
    for kind, vanish in ((SIGMA_COORD, frozenset()),
                         ("kin", frozenset())):
        d = SigmaDisc(kind=kind, y=0.5, theta=0.0, radius=0.03,
                      radius_theta=0.2, vanishing=vanish)
        fib = door_fiber_validity(cfg, arm, weld, scene, d, signs)
        assert fib["ok"], f"{kind} disc was discharged: {fib['reason']}"
        live, disch = live_discs(cfg_cert, arm, weld, scene, signs, [d])
        assert len(live) == 1 and not disch, f"{kind} disc discharged by live_discs"
        checked += 1

    # (b) no IK at the anchor -> must be left OPEN
    d = SigmaDisc(kind=DOOR_FOLD, y=0.5, theta=0.0, radius=0.03,
                  radius_theta=0.2, vanishing=frozenset({1}),
                  exact_y=0.5, exact_theta=0.0)
    far = dataclasses.replace(cfg, weld=dataclasses.replace(
        cfg.weld, p0=(9.0, -0.26, 0.46), p1=(9.0, 0.26, 0.46)))
    fib = door_fiber_validity(far, arm, WeldLine(far.weld), scene, d, signs)
    assert fib["ok"], f"out-of-reach anchor was discharged: {fib['reason']}"
    checked += 1

    # (c) a q4 fold whose single configuration IS valid -> open, and EXACT
    fib = door_fiber_validity(cfg, arm, weld, scene, d, signs)
    assert "EXACT" in fib["reason"], f"fold not handled exactly: {fib['reason']}"
    if fib["ok"]:
        assert fib["valid_fraction"] == 1.0
        live, disch = live_discs(cfg_cert, arm, weld, scene, signs, [d])
        assert len(live) == 1 and not disch, "valid fold disc was discharged"
    checked += 1
    return checked


if __name__ == "__main__":
    n_ok, n_tried, worst_rt = test_round_trip()
    print(f"[ik] round-trip        : {n_ok}/{n_tried} branches solvable, "
          f"max pose error {worst_rt:.3e}")
    worst_sign = test_sign_consistency()
    print(f"[ik] sign consistency  : ok (max |q2|,|q4|,|q6| seen {worst_sign:.3f})")
    worst_geo = test_branch_geometry_identical()
    print(f"[ik] branch geometry   : identical across branches, max dev {worst_geo:.3e}")
    test_degeneracy_geometry()
    print("[ik] degeneracies      : q2=0 pins elbow, q4=0 full extension, "
          "gimbal arcs fix the tool")
    test_mirror_identities()
    print("[ik] mirror identities : s2/s4/s6 confirmed as representational mirrors")
    worst_cells = test_shared_page_build_matches()
    print(f"[ik] shared page build : identical to the per-page build "
          f"(worst {worst_cells} differing cells over 8 pages x 2 scenes)")
    worst_fast = test_fasteval_matches_page_build()
    print(f"[ik] fasteval           : identical to the page build "
          f"(worst {worst_fast} differing cells over 8 pages x 2 scenes)")
    n_doors = test_unevaluable_doors_are_left_open()
    print(f"[ik] door discharge     : {n_doors} unevaluable/fold door case(s) "
          f"left OPEN, never discharged (invariant 2)")
    print("[ik] ALL TESTS PASSED")
