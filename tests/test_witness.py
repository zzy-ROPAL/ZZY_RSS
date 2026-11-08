"""Exact door crossability (item #0) and witness-path verification (item #2).

Two things are pinned here:

  * the CLOSED FORM must agree with a brute-force walk of the fiber. The
    closed form is only worth having if it is exactly the thing the sweep was
    approximating, so the test walks the fiber joint by joint and compares.

  * the VERIFIER must reject tampered paths. A verifier that only ever sees
    good input proves nothing, so each check is fed a path that violates it.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import (  # noqa: E402
    build_sigma, certify_fiber, door_passable, fiber_pair,
)
from zzy_rss.pagemap import build_all_pages_shared  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, CrossingParams, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402
from zzy_rss.witness import (  # noqa: E402
    LEG_PAGE, Leg, WitnessPath, exact_clearance, plan_in_page, verify_witness,
)

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")


def _weld(tilt_deg: float, x: float, z: float, half_y: float) -> WeldParams:
    a = np.radians(tilt_deg)
    return WeldParams(p0=(x, -half_y, z), p1=(x, half_y, z),
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def _scene(tilt, x, z, half_y, obstacles, res=61, **kw):
    base = Config()
    cfg = dataclasses.replace(
        base, weld=_weld(tilt, x, z, half_y),
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=tuple(obstacles)), **kw)
    return cfg, SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)


def _doors(cfg, arm, weld, scene, page):
    pms = {n: p for n, p in build_all_pages_shared(
        cfg, arm, weld, scene, with_kin=False, verbose=False).items() if n in PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    return [d for d in sig.get(page, []) if d.vanishing]


def _walk_fiber(arm, weld, disc, signs, n=4001):
    """Brute force: the largest |delta| reachable from 0 without leaving limits."""
    pair = fiber_pair(disc)
    res = arm.ik_branch(*weld.pose(disc.anchor[0]), disc.anchor[1], signs)
    if pair is None or not res.ok:
        return None
    i, k = pair
    out = {}
    for sgn in (+1, -1):
        last = None
        for t in np.linspace(0.0, sgn * np.pi, n):
            q = res.q.copy()
            q[i] += t
            q[k] -= t
            if not arm.within_limits(q):
                break
            last = float(t)
        out[sgn] = last
    return out


# --------------------------------------------------------------------------


def test_closed_form_matches_brute_force() -> list:
    """The exact interval must agree with a joint-by-joint walk of the fiber."""
    cases = [
        ("EXP 3 q2 door", dict(tilt=15.0, x=0.32, z=0.40, half_y=0.26,
                               obstacles=_zgap_y()), "++-"),
        ("EXP 4 q6 door", dict(tilt=45.0, x=0.40, z=0.22, half_y=0.34,
                               obstacles=(BoxObstacle(
                                   center=(0.095, 0.29, 0.377),
                                   half_extent=(0.07, 0.07, 0.07),
                                   label="e4"),)), "-++"),
    ]
    out = []
    for label, kw, page in cases:
        cfg, arm, weld, scene = _scene(**kw)
        signs = tuple(1 if c == "+" else -1 for c in page)
        for d in _doors(cfg, arm, weld, scene, page):
            cert = certify_fiber(cfg, arm, weld, scene, d, signs)
            if not cert.exact or cert.empty:
                continue
            walk = _walk_fiber(arm, weld, d, signs)
            if walk is None or walk[+1] is None or walk[-1] is None:
                continue
            # The walk stops one sample before the true boundary, so it must
            # land just inside the closed-form interval, never outside it.
            step = np.pi / 4000.0
            assert walk[+1] <= cert.hi + 1e-9, (
                f"{label}: walk reached {walk[+1]:.6f} > closed-form hi {cert.hi:.6f}")
            assert walk[-1] >= cert.lo - 1e-9, (
                f"{label}: walk reached {walk[-1]:.6f} < closed-form lo {cert.lo:.6f}")
            assert cert.hi - walk[+1] <= step + 1e-9, (
                f"{label}: closed-form hi {cert.hi:.6f} overshoots the walk "
                f"{walk[+1]:.6f} by more than one sample")
            assert walk[-1] - cert.lo <= step + 1e-9, (
                f"{label}: closed-form lo {cert.lo:.6f} overshoots the walk "
                f"{walk[-1]:.6f} by more than one sample")
            out.append((label, d.anchor[0], cert.lo, cert.hi, cert.crossable))
    assert out, "no doors exercised"
    return out


def test_known_doors_are_not_crossable() -> list:
    """EXP 3's q2 door and EXP 4's q6 door both fail the connected-interval test.

    EXP 3 misses by the URDF's 0.02 rad safety offset. EXP 4 was documented as
    "physically crossable, margin +14.46 deg" -- an ENDPOINT check on the
    2*pi-wrapped mirror representative, which is exactly what invariant 21
    says is invalid: the traversal takes q7 to -316.62 deg, and mirror_q wraps
    that to +43.38 deg, inside the limits, but the joint cannot turn there.
    """
    out = []
    cfg, arm, weld, scene = _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    for d in _doors(cfg, arm, weld, scene, "++-"):
        cert = certify_fiber(cfg, arm, weld, scene, d, (1, 1, -1))
        assert not cert.crossable, f"EXP 3 door y={d.anchor[0]:.4f} became crossable"
        out.append(("EXP 3", d.anchor[0], float(np.degrees(cert.margin))))

    cfg, arm, weld, scene = _scene(
        45.0, 0.40, 0.22, 0.34,
        (BoxObstacle(center=(0.095, 0.29, 0.377),
                     half_extent=(0.07, 0.07, 0.07), label="e4"),))
    for d in _doors(cfg, arm, weld, scene, "-++"):
        cert = certify_fiber(cfg, arm, weld, scene, d, (-1, 1, 1))
        assert not cert.crossable, f"EXP 4 door y={d.anchor[0]:.4f} became crossable"
        out.append(("EXP 4", d.anchor[0], float(np.degrees(cert.margin))))
    return out


def test_null_motion_is_verified_not_assumed() -> float:
    """The closed form is only claimed where the fiber really moves nothing."""
    cfg, arm, weld, scene = _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    worst = 0.0
    for d in _doors(cfg, arm, weld, scene, "++-"):
        cert = certify_fiber(cfg, arm, weld, scene, d, (1, 1, -1))
        assert cert.exact, f"null motion not verified at y={d.anchor[0]:.4f}"
        worst = max(worst, cert.drift_m)
    assert worst <= cfg.crossing.null_motion_tol_m
    return worst


def test_self_motion_is_one_dimensional() -> int:
    """Pin the precondition the closed form rests on.

    certify_fiber describes the 1-D counter-rotating slice. That is the WHOLE
    fiber only when the self-motion manifold is 1-D -- a 7-DOF arm with the
    full tool pose fixed. Raising the TASK dimension keeps this (a 2-D surface
    task still leaves exactly the arm angle free, so the page grows to
    (u, v, theta) and the fiber stays 1-D). Raising the REDUNDANCY breaks it,
    and breaks it in the dangerous direction: "no connected delta interval"
    would stop proving the pages cannot be joined, yet would still discharge
    the disc and skip the door -- a false infeasibility proof.

    If this ever reads other than 1, certify_fiber must return exact=False,
    which is what the guard inside it does.
    """
    cfg, arm, weld, scene = _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    rng = np.random.default_rng(3)
    n_checked = 0
    for _ in range(40):
        y = float(rng.uniform(0.05, 0.95))
        th = float(rng.uniform(-1.0, 1.0))
        res = arm.ik_branch(*weld.pose(y), th, (1, 1, -1))
        if not res.ok:
            continue
        n_self = int(res.q.size - np.linalg.matrix_rank(arm.jacobian(res.q),
                                                        tol=1e-6))
        assert n_self == 1, (
            f"self-motion is {n_self}-dimensional at y={y:.3f}, theta={th:.3f}; "
            f"the closed-form fiber slice is no longer the whole fiber")
        n_checked += 1
    assert n_checked > 10
    return n_checked


def test_unknown_never_discharges() -> None:
    """A door we could not evaluate must stay LIVE, not be certified blocked.

    The old code returned ok=False for "no IK at door" and "fiber not
    characterised" alike, and live_discs read ok=False as CERTIFIED CLOSED --
    so the wall was free to route through a door nobody had looked at.
    """
    cfg, arm, weld, scene = _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    doors = _doors(cfg, arm, weld, scene, "++-")
    assert doors
    d = doors[0]
    # A disc anchored far off the weld has no IK: the correct answer is
    # "unknown", never "certified blocked".
    bogus = dataclasses.replace(d, exact_y=5.0, exact_theta=0.0, y=5.0)
    verdict = door_passable(cfg, arm, weld, scene, bogus, (1, 1, -1))
    assert not verdict["certified_blocked"], (
        "a door with no IK was reported CERTIFIED BLOCKED; that is the bug "
        "that let walls cross unevaluated doors")
    assert verdict["unknown"] and verdict["passable"]


def test_fold_doors_are_characterised() -> int:
    """q4 fold doors used to fall through to 'fiber not characterised'."""
    n = 0
    for idx, joints in ((0, (0, 2)), (1, (2, 4)), (2, (4, 6))):
        disc = _fake_disc(idx)
        assert fiber_pair(disc) == joints, (
            f"sign index {idx} should counter-rotate joints {joints}")
        n += 1
    return n


def test_spelling_offset_is_limit_checked() -> None:
    """A carried 2*pi offset must gate limits, not just shift the numbers.

    After a crossing the robot holds the traversed endpoint, which can differ
    from the partner page's IK representative by a whole turn on one joint.
    Planning the continuation in that spelling is what makes the legs meet in
    Q -- but only if the shifted values are the ones limit-checked. A turn
    added to a joint already near its limit must make the plan fail.
    """
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, ())
    pm = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                verbose=False)["++-"]
    ij = np.argwhere(pm.valid_mask)
    ij = ij[np.argsort(ij[:, 0])]
    start = (float(pm.y_grid[ij[0, 0]]), float(pm.theta_grid[ij[0, 1]]))

    base = plan_in_page(cfg, arm, weld, scene, (1, 1, -1), start,
                        goal_y=cfg.flood.goal_y)
    assert base is not None, "baseline plan must exist"

    # q1's limit is +/-178.85 deg, so a whole extra turn cannot be inside it.
    off = np.zeros(7)
    off[0] = 2.0 * np.pi
    shifted = plan_in_page(cfg, arm, weld, scene, (1, 1, -1), start,
                           goal_y=cfg.flood.goal_y, q_offset=off)
    assert shifted is None, (
        "a 2*pi offset on q1 was accepted; the carried spelling is not being "
        "limit-checked, which is exactly the mirror_q trap of invariant 21")


def test_verifier_rejects_tampered_paths() -> int:
    """Each check must actually fire. A verifier only fed good input is useless."""
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, ())
    cfg = dataclasses.replace(cfg, crossing=CrossingParams())
    signs = (1, 1, -1)
    pm = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                verbose=False)["++-"]
    ij = np.argwhere(pm.valid_mask)
    ij = ij[np.argsort(ij[:, 0])]          # lowest y first, like the real start
    start = (float(pm.y_grid[ij[0, 0]]), float(pm.theta_grid[ij[0, 1]]))
    leg = plan_in_page(cfg, arm, weld, scene, signs, start,
                       goal_y=cfg.flood.goal_y)
    assert leg is not None, "no baseline path to tamper with"
    good = WitnessPath(legs=[leg])
    assert verify_witness(cfg, arm, weld, scene, good).ok, "baseline must verify"

    fired = 0

    # (a) joint-limit violation
    bad = Leg(kind=LEG_PAGE, page=signs, q=leg.q.copy(), y=leg.y.copy(),
              waypoints=leg.waypoints)
    bad.q[len(bad.q) // 2, 0] = 4.0
    r = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[bad]))
    assert not r.ok and any("joint limit" in f[1] for f in r.failures)
    fired += 1

    # (b) off the constraint manifold
    bad = Leg(kind=LEG_PAGE, page=signs, q=leg.q.copy(), y=leg.y.copy(),
              waypoints=leg.waypoints)
    bad.q[len(bad.q) // 2, 3] += 0.05
    r = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[bad]))
    assert not r.ok and any("constraint" in f[1] for f in r.failures)
    fired += 1

    # (c) a jump: two legs that do not meet in Q. This is the 2*pi-wrap trap
    #     -- the endpoint can be in limits while the joint cannot get there.
    half = len(leg.q) // 2
    l1 = Leg(kind=LEG_PAGE, page=signs, q=leg.q[:half].copy(),
             y=leg.y[:half].copy())
    l2 = Leg(kind=LEG_PAGE, page=signs, q=leg.q[half + 8:].copy(),
             y=leg.y[half + 8:].copy())
    r = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[l1, l2]))
    assert not r.ok and any("do not meet in Q" in f[1] for f in r.failures)
    fired += 1

    # (d) a step larger than step_max inside one leg
    keep = np.arange(0, len(leg.q), 12)
    bad = Leg(kind=LEG_PAGE, page=signs, q=leg.q[keep].copy(),
              y=leg.y[keep].copy())
    r = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[bad]))
    assert not r.ok and any("step" in f[1] for f in r.failures)
    fired += 1

    # (e) ends short of the goal
    bad = Leg(kind=LEG_PAGE, page=signs, q=leg.q[:half].copy(),
              y=leg.y[:half].copy())
    r = verify_witness(cfg, arm, weld, scene, WitnessPath(legs=[bad]),
                       goal_y=cfg.flood.goal_y)
    assert not r.ok and any("short of goal" in f[1] for f in r.failures)
    fired += 1
    return fired


def test_exact_clearance_is_not_optimistic() -> float:
    """Scene.check samples the capsule axis, so its clearance is an UPPER bound.

    Harmless for wall verification; wrong-signed for a swept path segment,
    where an optimistic clearance would certify a segment that collides.
    """
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, _zgap())
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(60):
        y = float(rng.uniform(0.0, 1.0))
        th = float(rng.uniform(-1.0, 1.0))
        res = arm.ik_branch(*weld.pose(y), th, (1, 1, -1))
        if not res.ok:
            continue
        frames = arm.frames(res.q)
        sampled = scene.check(frames).clearance
        exact, _ = exact_clearance(scene, frames)
        # 1e-9 m, not machine epsilon: segment_box_distance converges by
        # ternary refinement, so its residual is the floor here, not float64.
        assert exact <= sampled + 1e-9, (
            f"exact clearance {exact:.9f} exceeded the sampled {sampled:.9f}; "
            f"the sampled value is supposed to be an UPPER bound")
        worst = max(worst, sampled - exact)
    return worst


# -- fixtures ---------------------------------------------------------------


def _zgap():
    return (BoxObstacle(center=(0.14, 0.0, 0.38), half_extent=(0.06,) * 3,
                        label="low"),
            BoxObstacle(center=(0.14, 0.0, 0.66), half_extent=(0.06,) * 3,
                        label="high"))


def _zgap_y():
    return tuple(dataclasses.replace(b, center=(b.center[0], b.center[1] + 0.10,
                                                b.center[2]))
                 for b in _zgap())


def _fake_disc(idx):
    from zzy_rss.doors import DOOR_INTERIOR, SigmaDisc
    return SigmaDisc(kind=DOOR_INTERIOR, y=0.5, theta=0.0, radius=0.02,
                     radius_theta=0.02, vanishing=frozenset({idx}))

def test_roadmap_two_class_distance_converges() -> list:
    """De-gridding is blocked only by the WRONG rest class (findings 38).

    The handoff predicted that dropping the grid would push the required
    kernel width from ~178 to ~679, because a grid cannot put two
    opposite-class points closer than one cell while a roadmap can sample
    arbitrarily close. That is true for rest_mode "all_valid" and false for
    "goal_only".

    With "all_valid" the rest class contains small dead components that sit a
    hair from the start component, and the hair gets thinner without bound as
    sampling densifies -- exactly invariant 15's complaint, now measured as a
    DIVERGENCE rather than a constant factor. With "goal_only" the two classes
    are separated by the real obstacle, whose thickness is a fixed geometric
    quantity, so d_min converges and gamma stays bounded.
    """
    from zzy_rss.doors import build_sigma
    from zzy_rss.source import roadmap_sources

    cfg, arm, weld, scene = _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    pms = {n: p for n, p in build_all_pages_shared(
        cfg, arm, weld, scene, with_kin=False, verbose=False).items()
        if n in PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    sc = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])

    rows = []
    for n in (1500, 4000):
        src = roadmap_sources(cfg, arm, weld, scene, pms, sig,
                              n_samples=n, verbose=False)["++-"]
        cand = [r for r in src.regions() if r.reaches_start and not r.reaches_goal] \
            or [r for r in src.regions() if r.reaches_start]
        k = sorted(cand, key=lambda r: -r.n_cells)[0].index
        seed = src.seed_pool(k)
        out = {}
        for mode in ("goal_only", "all_valid"):
            rest = src.rest_pool(k, mode)
            out[mode] = float(np.linalg.norm(
                seed[:, None, :] * sc - rest[None, :, :] * sc, axis=2).min())
        rows.append((n, out["goal_only"], out["all_valid"]))

    (n0, g0, a0), (n1, g1, a1) = rows
    assert a1 < a0, ("all_valid d_min did not shrink with density; the "
                     "divergence this test documents is gone")
    assert g1 > a1 * 2.0, (
        f"goal_only ({g1:.5f}) is no longer well clear of all_valid "
        f"({a1:.5f}); the rest class is not what keeps gamma bounded")
    assert g1 > 0.04, (
        f"goal_only d_min collapsed to {g1:.5f}; eta-separation is now "
        f"required and cfg.wall.eta_separation must be switched on")
    return rows


if __name__ == "__main__":
    rows = test_closed_form_matches_brute_force()
    print(f"[witness] closed form     : matches a 4001-step fiber walk on "
          f"{len(rows)} door(s), to within one sample")
    margins = test_known_doors_are_not_crossable()
    print("[witness] known doors     : none crossable --")
    for label, y, deg in margins:
        print(f"              {label} door y={y:.4f}: short of +/-pi by "
              f"{abs(deg):.4f} deg")
    drift = test_null_motion_is_verified_not_assumed()
    print(f"[witness] null motion     : verified, worst drift {drift * 1e3:.2e} mm")
    n_sm = test_self_motion_is_one_dimensional()
    print(f"[witness] redundancy      : self-motion is 1-D at all {n_sm} sampled "
          f"configurations (the closed form's precondition)")
    test_unknown_never_discharges()
    print("[witness] unknown doors   : stay live (never 'certified blocked')")
    n = test_fold_doors_are_characterised()
    print(f"[witness] fiber pairs     : all {n} sign indices characterised "
          f"(q4 fold doors included)")
    test_spelling_offset_is_limit_checked()
    print("[witness] carried spelling: a 2*pi offset is limit-checked, not just "
          "shifted")
    fired = test_verifier_rejects_tampered_paths()
    print(f"[witness] verifier        : rejects all {fired} tampered paths "
          f"(limits, constraint, junction, step, goal)")
    rows = test_roadmap_two_class_distance_converges()
    print("[witness] de-gridding     : roadmap two-class d_min --")
    for n, g, a in rows:
        print(f"              n={n:<6} goal_only {g:.5f} (gamma {1/g**2:>5.0f})  "
              f"all_valid {a:.5f} (gamma {1/a**2:>5.0f})")
    gap = test_exact_clearance_is_not_optimistic()
    print(f"[witness] exact clearance : never above the sampled value "
          f"(worst optimism {gap * 1e3:.3f} mm)")
    print("[witness] ALL TESTS PASSED")
