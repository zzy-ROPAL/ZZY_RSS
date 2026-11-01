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


# --------------------------------------------------------------------------
# EXP 5: the arc seam, the interior fold door, and the cylinder page
# --------------------------------------------------------------------------


def _arc_cfg(res: int = 41):
    """The EXP 5 placement, rebuilt here so the test does not import a script.

    The fixture's x is SOLVED so that max_y |W(y) - S| == reach_max exactly;
    that double root is what puts a q4 fold door in the task's INTERIOR, which
    invariant 22 says a straight seam can never have.
    """
    import dataclasses
    from scipy.optimize import brentq
    from zzy_rss.params import Config, PageMapParams, SceneParams, WeldParams

    arm_p = Config().arm
    shoulder = np.array([0.0, 0.0, arm_p.d_bs])
    cz, radius, tilt, phi0, phi1 = 0.30, 0.20, 30.0, -0.80, 1.00

    def wp(cx, a, b):
        t = np.radians(tilt)
        return WeldParams(
            p0=(cx + radius, 0.0, cz), p1=(cx + radius, 0.0, cz),
            approach=(float(np.sin(t)), 0.0, -float(np.cos(t))),
            reference=(float(np.cos(t)), 0.0, float(np.sin(t))),
            curve="arc", arc_center=(cx, 0.0, cz), arc_radius=radius,
            arc_normal=(0.0, 0.0, 1.0), arc_ref=(1.0, 0.0, 0.0),
            arc_phi0=a, arc_phi1=b)

    def residual(cx):
        w = WeldLine(wp(cx, 0.0, 1.0))
        g = shoulder + arm_p.d_wt * (w.rot @ np.array([0.0, 0.0, 1.0]))
        rho = float(np.linalg.norm((g - np.array([cx, 0.0, cz]))[:2]))
        return float(np.hypot(g[2] - cz, rho + radius)) - arm_p.reach_max

    cx = brentq(residual, 0.05, 1.2, xtol=1e-15, rtol=8.9e-16)
    base = Config()
    cfg = dataclasses.replace(
        base, weld=wp(cx, phi0, phi1),
        pagemap=PageMapParams(n_y=res, n_theta=res, theta_periodic=True),
        scene=SceneParams(gap_open=True, obstacles=()))
    y_star = (0.0 - phi0) / (phi1 - phi0)
    return cfg, y_star, float(residual(cx))


def test_arc_weld_round_trip() -> float:
    """FK(IK(.)) on the ARC seam, and the arc's own pose/tangent consistency."""
    from zzy_rss.params import DEFAULT
    cfg, _y_star, _res = _arc_cfg()
    arm = SrsArm(cfg.arm)
    weld = WeldLine(cfg.weld)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        y = float(rng.uniform(0.0, 1.0))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        # the seam point really is on the arc
        d = float(np.linalg.norm(pos - weld.center))
        assert abs(d - weld.radius) < 1e-12, f"off the arc by {d - weld.radius:.3e}"
        # the tangent really is the derivative of the position
        h = 1e-6
        fd = (weld.position(y + h) - weld.position(y - h)) / (2 * h)
        assert np.max(np.abs(fd - weld.tangent(y))) < 1e-6
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            if not res.ok:
                continue
            p_fk, r_fk = arm.fk(res.q)
            err = max(float(np.linalg.norm(p_fk - pos)),
                      float(np.linalg.norm(r_fk - rot)))
            worst = max(worst, err)
            assert err < 1e-8, f"arc round-trip failed: {err:.3e}"
    # a straight seam must be untouched by the generalisation
    wl = WeldLine(DEFAULT.weld)
    assert np.max(np.abs(wl.position(0.37)
                         - (wl.p0 + 0.37 * (wl.p1 - wl.p0)))) < 1e-15
    return worst


def test_door_rootfinder_matches_quadratic() -> float:
    """The numeric door finder must reproduce the closed form on a straight seam.

    The q2 and q6 conditions used to be solved as quadratics in y, which was
    only legitimate because a straight seam with a constant tool pose makes
    W(y) affine. doors.py now solves them numerically so that an ARC seam
    works; on a straight seam the two must agree to the last bit, or EXP 1-4
    have silently moved.
    """
    import dataclasses
    from zzy_rss.doors import _quadratic_roots_on_unit, _roots_on_unit
    from zzy_rss.params import Config, WeldParams

    worst = 0.0
    e_z = np.array([0.0, 0.0, 1.0])
    for tilt, x, z in ((0.0, 0.28, 0.46), (15.0, 0.32, 0.40),
                       (30.0, 0.34, 0.48), (45.0, 0.40, 0.22)):
        a = np.radians(tilt)
        wparams = WeldParams(
            p0=(x, -0.26, z), p1=(x, 0.26, z),
            approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
            reference=(float(np.cos(a)), 0.0, float(np.sin(a))))
        cfg = dataclasses.replace(Config(), weld=wparams)
        arm = SrsArm(cfg.arm)
        weld = WeldLine(wparams)
        p = arm.p
        taxis = weld.rot @ e_z
        w0 = np.asarray(wparams.p0) - p.d_wt * taxis
        wd = np.asarray(wparams.p1) - np.asarray(wparams.p0)
        for target, base, req, cond in (
            (arm.shoulder + p.d_se * e_z, w0, p.d_ew,
             lambda y, t=arm.shoulder + p.d_se * e_z: float(np.linalg.norm(
                 weld.wrist_center(y, p.d_wt) - t)) - p.d_ew),
            (arm.shoulder, w0 - p.d_ew * taxis, p.d_se,
             lambda y: float(np.linalg.norm(
                 weld.wrist_center(y, p.d_wt)
                 - p.d_ew * (weld.pose(y)[1] @ e_z) - arm.shoulder)) - p.d_se),
        ):
            delta = base - target
            quad = sorted(min(max(v, 0.0), 1.0) for v in _quadratic_roots_on_unit(
                float(np.dot(wd, wd)), 2.0 * float(np.dot(delta, wd)),
                float(np.dot(delta, delta)) - req ** 2))
            num = _roots_on_unit(cond)
            assert len(quad) == len(num), (
                f"tilt {tilt}: closed form found {len(quad)} roots, numeric "
                f"found {len(num)}")
            for u, v in zip(quad, num):
                worst = max(worst, abs(u - v))
    assert worst < 1e-12, f"door root finders disagree by {worst:.3e}"
    return worst


def test_fold_door_is_one_posture() -> tuple:
    """The whole theta line at a fold is ONE posture, and so is the q3/q5 fiber.

    This is what makes a line door's certificate exact rather than sampled
    (todo 5): collision is one query for the entire line, and joint limits are
    a line meeting a box. If this assertion ever fails, certify_fold_door is no
    longer entitled to call itself exact.
    """
    from zzy_rss.doors import certify_fold_door, find_fold_doors
    from zzy_rss.pagemap import build_all_pages_shared
    from zzy_rss.scene import Scene

    cfg, y_star, residual = _arc_cfg(res=31)
    assert abs(residual) <= cfg.sigma.fold_tangency_tol, (
        f"the placement is not tangential: residual {residual:.3e}")
    arm = SrsArm(cfg.arm)
    weld = WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)
    pages = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
    pm = pages["++-"]
    discs = find_fold_doors(cfg, pm, arm, weld)
    assert discs, "no fold door found on the arc placement"
    ys = sorted({round(d.y, 6) for d in discs})
    assert len(ys) == 1 and abs(ys[0] - y_star) < 2e-3, (
        f"fold doors at {ys}, expected one at {y_star:.6f}")
    assert all(d.fold for d in discs), "fold discs lost their `fold` flag"

    # q4 really is 0 there
    r = arm.ik_branch(*weld.pose(y_star), 0.0, (1, 1, -1))
    assert r.ok and abs(r.q[3]) < 1e-5, f"q4 at the fold is {r.q[3]:.3e}"

    d0 = discs[0]
    d0 = type(d0)(**{**d0.__dict__, "pages": ("++-", "+--")})
    cert = certify_fold_door(cfg, arm, weld, scene, d0, pages=("++-", "+--"))
    assert cert.exact, "certificate not exact with symmetric capsules"
    assert cert.posture_drift_m < 1e-12, (
        f"the theta line is NOT one posture: drift {cert.posture_drift_m:.3e} m")
    assert cert.open, "the fold door should be open on this placement"
    for name in ("++-", "+--"):
        assert cert.pages[name]["within_limits"], f"page {name} out of limits"
    # and the IK representative alone would have said NO -- invariant 3
    assert arm.limit_margin(r.q) < 0.0, (
        "the IK representative at the fold is inside the limits here, so this "
        "placement no longer demonstrates invariant 3")

    # a STRAIGHT seam must have no interior fold at all (invariant 22)
    from zzy_rss.params import DEFAULT
    arm2 = SrsArm(DEFAULT.arm)
    weld2 = WeldLine(DEFAULT.weld)
    pages2 = build_all_pages_shared(DEFAULT, arm2, weld2,
                                    Scene(DEFAULT.arm, DEFAULT.scene),
                                    with_kin=False, verbose=False)
    straight = find_fold_doors(DEFAULT, pages2["++-"], arm2, weld2)
    assert not straight, (
        f"{len(straight)} fold discs on a STRAIGHT seam, which invariant 22 "
        f"says is impossible")
    return cert.posture_drift_m, float(np.degrees(
        cert.pages["++-"]["best_margin_rad"]))


def test_cylinder_wall_closes() -> tuple:
    """On a cylinder page the learned manifold is 2 pi periodic and WRAPS.

    Three things are asserted, each of which was a real failure before:
      * the decision function is exactly periodic in theta, so it is a
        function on the cylinder and the certificate's "every path crosses
        f = 0" argument applies at all;
      * its analytic gradient matches finite differences through the
        (cos, sin) embedding;
      * a wall that wraps the cylinder is recognised as CLOSED. Reading
        matplotlib's compound contour path straight off `.vertices` spliced
        the two wrapping loops into one arc with a 0.876 gap in y and the
        wall was reported open.
    """
    import dataclasses
    from zzy_rss.params import Config, PageMapParams
    from zzy_rss.wall import reconstruct_wall, train_wall

    rng = np.random.default_rng(5)
    seed = np.column_stack([rng.uniform(0.0, 0.45, 300),
                            rng.uniform(-np.pi, np.pi, 300)])
    rest = np.vstack([
        np.column_stack([rng.uniform(0.75, 1.0, 300),
                         rng.uniform(-np.pi, np.pi, 300)]),
        np.column_stack([np.full(60, -0.075), np.linspace(-np.pi, np.pi, 60)]),
        np.column_stack([np.full(60, 1.075), np.linspace(-np.pi, np.pi, 60)]),
    ])
    cfg = dataclasses.replace(
        Config(), pagemap=PageMapParams(n_y=41, n_theta=41, theta_periodic=True))
    w = train_wall(cfg, seed, rest)
    assert w.periodic and w.accuracy >= 1.0
    probe = np.array([[0.6, 0.3], [0.2, -1.1], [0.9, 2.0]])
    a = w.decision(probe)
    b = w.decision(probe + np.array([0.0, 2.0 * np.pi]))
    assert np.max(np.abs(a - b)) < 1e-12, (
        f"the wall is not periodic in theta: {np.max(np.abs(a - b)):.3e}")
    h = 1e-6
    for pt in probe:
        g = w.decision_grad(pt)
        fd = np.array([
            (w.decision(pt + [h, 0])[0] - w.decision(pt - [h, 0])[0]) / (2 * h),
            (w.decision(pt + [0, h])[0] - w.decision(pt - [0, h])[0]) / (2 * h)])
        assert np.max(np.abs(g - fd)) < 1e-5, (
            f"gradient through the embedding is wrong: {np.max(np.abs(g - fd)):.3e}")
    poly = reconstruct_wall(cfg, w)
    assert poly.closed, f"cylinder wall not closed: {poly.note!r}"
    assert len(poly.loops) == 2 and all(k == "wrapping" for k in poly.loop_kinds), (
        f"expected two wrapping loops, got {poly.loop_kinds}")
    return len(poly.loops), float(np.max(np.abs(a - b)))


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
    worst_arc = test_arc_weld_round_trip()
    print(f"[ik] arc seam           : FK(IK) round-trip max error "
          f"{worst_arc:.3e}; straight seam unchanged")
    worst_root = test_door_rootfinder_matches_quadratic()
    print(f"[ik] door root finder   : numeric == closed form on straight seams "
          f"(worst {worst_root:.3e})")
    drift, margin = test_fold_door_is_one_posture()
    print(f"[ik] fold (line) door   : whole theta line + q3/q5 fiber is ONE "
          f"posture (drift {drift:.2e} m), open with {margin:.2f} deg margin; "
          f"straight seam has none")
    nloops, per = test_cylinder_wall_closes()
    print(f"[ik] cylinder wall      : decision exactly periodic ({per:.1e}), "
          f"{nloops} WRAPPING loops, closed")
    print("[ik] ALL TESTS PASSED")
