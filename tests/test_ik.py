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


def test_gamma_search_modes_agree() -> tuple[int, int, float]:
    """`gamma_search` must be a COST choice, never a modelling one.

    "bisect" probes rungs 1, 2, 4, ... of IJRR's gamma ladder and then
    bisects, which is ~90x fewer SVM fits than walking all 3990 rungs. That
    is only legitimate if it lands on the same rung as "linear". Check it on
    REAL page data -- the start region of a page versus the rest of it --
    rather than on synthetic clouds, because what makes this hard is the
    geometry of the valid set.

    gamma_max is lowered here so the linear scan is affordable in a test; the
    search logic does not depend on it.
    """
    import dataclasses
    from zzy_rss.flood import (invalid_points, label_regions,
                               other_valid_points, region_points, thin)
    from zzy_rss.pagemap import build_all_pages_shared
    from zzy_rss.params import (BoxObstacle, PageMapParams, SceneParams,
                                WeldParams)
    from zzy_rss.scene import Scene
    from zzy_rss.wall import train_wall, virtual_boundary_samples

    a = np.radians(45.0)
    weld_p = WeldParams(p0=(0.44, -0.34, 0.22), p1=(0.44, 0.34, 0.22),
                        approach=(float(np.sin(a)), 0.0, float(-np.cos(a))),
                        reference=(float(np.cos(a)), 0.0, float(np.sin(a))))
    obstacles = (BoxObstacle(center=(0.18, 0.30, 0.34),
                             half_extent=(0.07, 0.07, 0.07), label="cut"),
                 BoxObstacle(center=(0.18, -0.06, 0.42),
                             half_extent=(0.05, 0.05, 0.05), label="split"))
    n_cases = 0
    worst = 0.0
    for obs, pages in (((), ("+-+",)), (obstacles, ("+-+", "+--"))):
        cfg = dataclasses.replace(
            DEFAULT, weld=weld_p, pagemap=PageMapParams(n_y=41, n_theta=41),
            scene=SceneParams(gap_open=True, obstacles=obs))
        arm = SrsArm(cfg.arm)
        wl = WeldLine(cfg.weld)
        scene = Scene(cfg.arm, cfg.scene)
        built = build_all_pages_shared(cfg, arm, wl, scene, with_kin=False,
                                       verbose=False)
        for name in pages:
            pm = built[name]
            lab, regs = label_regions(cfg, pm)
            if not regs:
                continue
            k = max(regs, key=lambda r: r.n_cells).index
            rng = np.random.default_rng(0)
            seed = thin(region_points(pm, lab, k), 300, rng)
            rest = np.vstack([
                thin(other_valid_points(pm, lab, k, regs, "goal_only"),
                     300, rng),
                virtual_boundary_samples(cfg, 60)])
            if not len(seed) or not len(rest):
                continue
            got = {}
            for mode in ("linear", "bisect"):
                c = dataclasses.replace(cfg, wall=dataclasses.replace(
                    cfg.wall, gamma_search=mode, gamma_max=80.0,
                    gamma_warn=1e9))
                w = train_wall(c, seed, rest)
                got[mode] = (w.gamma, w.gamma_capped, w.accuracy)
            gl, cl, al = got["linear"]
            gb, cb, ab = got["bisect"]
            assert cl == cb, (f"page {name}: capped differs "
                              f"(linear {cl}, bisect {cb})")
            assert abs(gl - gb) < 1e-9, (
                f"page {name}: gamma differs (linear {gl}, bisect {gb}) -- "
                f"bisect is only valid if it lands on the same rung")
            assert abs(al - ab) < 1e-12
            n_cases += 1
            worst = max(worst, abs(gl - gb))
    assert n_cases >= 3, f"only {n_cases} cases exercised"
    return n_cases, 2, worst


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
    n_g, n_sc, worst_g = test_gamma_search_modes_agree()
    print(f"[ik] gamma search       : bisect == linear on {n_g} real page "
          f"regions over {n_sc} scenes (max gamma diff {worst_g:.1e})")
    print("[ik] ALL TESTS PASSED")
