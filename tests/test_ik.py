"""IK round-trip and branch-structure tests (Stage 0 acceptance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SIGN_BRANCHES, VALID, SrsArm, rot_y, rot_z  # noqa: E402
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


def test_shared_build_handles_page_dependent_geometry() -> int:
    """The eight-pages-for-one trick must NOTICE when it no longer applies.

    One collision check per cell serves all eight pages only because the
    mirrors leave the arm's skeleton alone -- the upper-arm, forearm and tool
    capsules move 0.0 mm between pages. An OFF-AXIS capsule breaks that: the
    hubs hang off frames 2 and 3, which do rotate, so flipping s2 moves the
    upper-arm hub by 210 mm and flipping s4 moves the elbow hub by 180 mm.

    Measured with a 5 cm box parked on the upper-arm hub's position: the eight
    pages' valid fractions spread over 11 percentage points, and reusing the
    reference branch's collision report gets 26 of 961 cells wrong on each of
    the four s2 = - pages. build_all_pages_shared must therefore fall back to
    one collision check per page, and fasteval -- whose whole design caches one
    set of capsule points per cell -- must refuse the configuration outright.
    """
    import dataclasses

    from zzy_rss import pagemap as PM
    from zzy_rss.fasteval import Precomputed
    from zzy_rss.params import BoxObstacle, PageMapParams, SceneParams
    from zzy_rss.scene import Scene

    cfg = dataclasses.replace(
        DEFAULT,
        arm=dataclasses.replace(DEFAULT.arm, use_off_axis_upper_arm=True,
                                use_off_axis_elbow=True),
        pagemap=PageMapParams(n_y=31, n_theta=31),
        scene=SceneParams(gap_open=True, obstacles=(
            BoxObstacle(center=(0.198819, 0.088679, 0.191399),
                        half_extent=(0.025, 0.025, 0.025), label="hub"),)))
    assert PM.geometry_is_page_dependent(cfg), "off-axis geometry must be flagged"
    arm = SrsArm(cfg.arm)
    weld = WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    truth = PM.build_all_pages(cfg, arm, weld, scene, with_kin=False,
                               verbose=False, shared=False)
    # the pages really do differ by collision here, or the test proves nothing
    fracs = [float((truth[n].tag == VALID).mean()) for n in truth]
    assert max(fracs) - min(fracs) > 0.05, (
        f"pages do not differ enough to be a test: {fracs}")

    shared = PM.build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                       verbose=False)
    worst = max(int(np.sum(truth[n].tag != shared[n].tag)) for n in truth)
    assert worst == 0, f"shared build differs in {worst} cells"

    # and the old, wrong behaviour must really have been wrong
    original = PM.geometry_is_page_dependent
    try:
        PM.geometry_is_page_dependent = lambda _cfg: False
        naive = PM.build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                          verbose=False)
    finally:
        PM.geometry_is_page_dependent = original
    naive_worst = max(int(np.sum(truth[n].tag != naive[n].tag)) for n in truth)
    assert naive_worst > 0, (
        "reusing one collision report should be WRONG here; if it is not, this "
        "test no longer guards anything")

    try:
        Precomputed(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("fasteval must refuse page-dependent geometry")
    return naive_worst


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
    naive = test_shared_build_handles_page_dependent_geometry()
    print(f"[ik] page-dep geometry : shared build falls back correctly; "
          f"reusing one report would have been wrong in {naive} cells")
    print("[ik] ALL TESTS PASSED")
