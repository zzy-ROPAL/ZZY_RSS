"""The chart singularity function sigma and the interior door search
(docs/recursive_proof_design.md, steps 1 and 2a).

What is pinned:

  * SIGMA IS THE CHART'S SINGULARITY. |q2| and |q6| from sigma equal the
    IK's on all eight pages, and at the closed-form q2 / q6 door points the
    6x7 Jacobian is FULL RANK -- a task-Jacobian test would miss them.
  * THE BOUNDS ARE GUARANTEED. |sigma_i(z) - sigma_i(c)| <= L_i |z - c| on
    random balls, never violated.
  * THE INTERIOR SEARCH IS COMPLETE. Without any door list it finds a door
    region around every closed-form door curve point of four patches, and no
    region without a door in it. A missed door is the one way to a false
    INFEASIBLE, so completeness is checked against independent code (doors3).
  * THE WALL DECIDES WHAT IS INSIDE. A closed wall far from every door
    encloses no door; one around a door point encloses it.
  * TEST (ii) ON FACETS. A triangle through a door curve is rejected, one
    far away passes; check_simplex reports the sigma reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_3d import PATCHES  # noqa: E402
from search_scene3 import surface  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.facets import FacetOracle, check_simplex, sigma_clear_simplex  # noqa: E402
from zzy_rss.interior import (CROSS, IN, OUT, find_doors,  # noqa: E402
                              polytope_region)
from zzy_rss.params import Config, Page3Params  # noqa: E402
from zzy_rss.wall import Polytope  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
P3 = Page3Params()
SCENES = dict(PATCHES)
SCENES["exp3-new"] = surface(0.40, 0.21, 45, 305)      # the q6-door EXP 3 (findings 45)


def _chart(name):
    surf = WeldSurface(SCENES[name])
    return surf, SrsChart.for_surface(ARM, surf, P3)


def _on_patch(c):
    return c.pts[np.all((c.pts[:, :2] >= 0.0) & (c.pts[:, :2] <= 1.0), axis=1)]


def test_sigma_is_the_chart_singularity() -> dict:
    worst, n = 0.0, 0
    for name in ("exp2", "exp3-new"):
        surf, ch = _chart(name)
        Z = np.random.default_rng(0).uniform(ch.lo, ch.hi, (1500, 3))
        S = ch.sigma(Z)
        for z, s in zip(Z, S):
            for sg in SIGN_BRANCHES:
                r = ARM.ik_branch(*surf.pose(z[0], z[1]), z[2], sg)
                if not r.ok:
                    continue
                n += 1
                worst = max(worst, abs(abs(r.q[1]) - s[0]), abs(abs(r.q[5]) - s[1]))
    assert n > 1000 and worst < 1e-10, (n, worst)

    # At the closed-form doors sigma vanishes while the Jacobian is full rank.
    jac, sig_max, m = [], 0.0, 0
    for name in ("exp2", "exp3-new"):
        surf, ch = _chart(name)
        for c in door_curves(CFG, ARM, surf, P3):
            pts = _on_patch(c)[::7]
            k = 0 if c.joint == 2 else 1
            sig_max = max(sig_max, float(np.max(ch.sigma(pts)[:, k])))
            for z in pts:
                r = ARM.ik_branch(*surf.pose(z[0], z[1]), z[2], (1, 1, 1))
                if r.ok:
                    m += 1
                    jac.append(float(np.linalg.svd(ARM.jacobian(r.q), compute_uv=False)[-1]))
    assert m > 10 and sig_max < 1e-6, (m, sig_max)
    # A true singularity (q4 = 0) sits at ~1e-10; the doors are nowhere near.
    jac = np.asarray(jac)
    assert np.median(jac) > 0.05 and jac.min() > 1e-4, (np.median(jac), jac.min())
    return dict(n=n, worst=worst, door_pts=m, sigma_at_doors=sig_max,
                jac_min=float(jac.min()), jac_med=float(np.median(jac)))


def test_bounds_are_guaranteed() -> dict:
    pairs = viol = 0
    for name in ("exp2", "exp3", "exp3-new"):
        _, ch = _chart(name)
        rng = np.random.default_rng(1)
        for _ in range(300):
            c = rng.uniform(ch.lo, ch.hi)
            R = 10 ** rng.uniform(-3, -0.7)
            L = ch.lipschitz(c[None], np.array([R]))[0]
            sc = ch.sigma(c[None])[0]
            d = rng.normal(size=(150, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            Z = c[None] + d * (rng.uniform(0, 1, (150, 1)) ** (1 / 3) * R) / ch.scale
            s = ch.sigma(Z)
            dist = np.linalg.norm((Z - c) * ch.scale, axis=1)
            for k in range(4):
                if not (np.isfinite(L[k]) and np.isfinite(sc[k])):
                    continue
                ok = np.isfinite(s[:, k])
                pairs += int(ok.sum())
                viol += int(np.sum(np.abs(s[ok, k] - sc[k]) > L[k] * dist[ok] + 1e-12))
    assert pairs > 100_000 and viol == 0, (pairs, viol)
    return dict(pairs=pairs)


def test_interior_search_is_complete() -> dict:
    out = {}
    for name in SCENES:
        surf, ch = _chart(name)
        res = find_doors(ch, tol=0.01)
        assert res.complete
        curves = door_curves(CFG, ARM, surf, P3)
        pts = [_on_patch(c) for c in curves]
        # Completeness: every closed-form door point lies in a reported region.
        for P in pts:
            hit = np.zeros(len(P), dtype=bool)
            for d in res.doors:
                hit |= d.contains(P)
            assert hit.all(), f"{name}: {int((~hit).sum())} closed-form door point(s) missed"
        # No spurious region: each contains a closed-form door point.
        allp = np.vstack(pts) if pts else np.zeros((0, 3))
        for d in res.doors:
            assert len(allp) and d.contains(allp).any(), f"{name}: region {d.label()} has no door"
        out[name] = (len(curves), [d.label() for d in res.doors], res.stats["evaluated"])
    return out


def _sphere(centre_scaled, radius, scale, levels=2) -> Polytope:
    """A closed triangle mesh around a point, in unscaled page coordinates."""
    V = [np.array(v, dtype=float) for v in
         ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    T = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(levels):
        cache, T2 = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in T:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            T2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        T = T2
    Vs = (np.asarray(centre_scaled)[None, :] + radius * np.asarray(V)) / scale
    return Polytope(closed=True, method="tdc", vertices=Vs, simplices=np.asarray(T))


def test_the_wall_decides_what_is_inside() -> dict:
    surf, ch = _chart("exp2")
    curves = door_curves(CFG, ARM, surf, P3)
    door = _on_patch(curves[0])[len(_on_patch(curves[0])) // 2]
    rng = np.random.default_rng(2)
    # A point far from every door (by sigma's own certificate).
    for _ in range(1000):
        far = rng.uniform(ch.lo + 0.2, ch.hi - 0.2)
        if ch.ball_test(far[None], np.array([0.12])).clear[0]:
            break
    else:
        raise AssertionError("no door-free ball found")
    out = {}
    for tag, centre, want in (("far", far, 0), ("door", door, 1)):
        poly = _sphere(centre * ch.scale, 0.08, ch.scale)
        region = polytope_region(poly, ch.scale)
        # IN / OUT agree with the plain winding-number test, CROSS is near.
        Z = (centre * ch.scale + rng.uniform(-0.15, 0.15, (300, 3))) / ch.scale
        code = region(Z, np.zeros(len(Z)))
        dist = np.linalg.norm(Z * ch.scale - centre * ch.scale, axis=1)
        assert np.all(code[dist < 0.06] == IN) and np.all(code[dist > 0.09] == OUT)
        code_r = region(Z, np.full(len(Z), 0.02))
        assert np.all(code_r[np.abs(dist - 0.08) < 0.012] == CROSS)
        res = find_doors(ch, region=region, tol=0.005)
        assert res.complete
        n = len(res.doors)
        assert (n == 0) if want == 0 else (n >= 1 and any(d.contains(door[None])[0]
                                                         for d in res.doors)), (tag, n)
        out[tag] = n
    return out


def test_facet_sigma() -> dict:
    surf, ch = _chart("exp2")
    door = _on_patch(door_curves(CFG, ARM, surf, P3)[0])
    p = door[len(door) // 2]
    s = ch.scale
    tri_hit = np.array([p + np.array([0.05, 0.0, 0.0]) / s, p + np.array([-0.05, 0.03, 0.0]) / s,
                        p + np.array([-0.05, -0.03, 0.02]) / s])
    rng = np.random.default_rng(3)
    for _ in range(1000):
        c = rng.uniform(ch.lo + 0.1, ch.hi - 0.1)
        if ch.ball_test(c[None], np.array([0.1])).clear[0]:
            break
    tri_far = np.array([c, c + np.array([0.04, 0.0, 0.0]) / s, c + np.array([0.0, 0.04, 0.0]) / s])
    # Everything invalid with deep penetration: only test (ii) can fail a facet.
    oracle = FacetOracle(scale=s, guard=lambda v: None, virtual=lambda c: False,
                         valid=lambda c: False, depth=lambda c, r: (1e9, "test"),
                         sigma_many=ch.clear_many, sigma_max_depth=6)
    ok_hit, why_hit, _ = sigma_clear_simplex(oracle, tri_hit)
    ok_far, _, d_far = sigma_clear_simplex(oracle, tri_far)
    assert not ok_hit and "sigma" in why_hit
    assert ok_far
    r_hit = check_simplex(oracle, tri_hit, max_depth=5)
    r_far = check_simplex(oracle, tri_far, max_depth=5)
    assert not r_hit.ok and "sigma" in r_hit.reason and r_far.ok
    return dict(hit=why_hit, far_depth=d_far)


if __name__ == "__main__":
    r = test_sigma_is_the_chart_singularity()
    print(f"[chart] sigma = chart singularity: {r['n']} IK solutions, max error "
          f"{r['worst']:.1e}; at {r['door_pts']} closed-form door points sigma <= "
          f"{r['sigma_at_doors']:.1e} while the Jacobian's smallest singular value has "
          f"median {r['jac_med']:.3f} (min {r['jac_min']:.4f}; a real fold is ~1e-10)")
    r = test_bounds_are_guaranteed()
    print(f"[chart] bounds guaranteed: {r['pairs']} random pairs, 0 violations")
    for k, (nc, labels, cells) in test_interior_search_is_complete().items():
        print(f"[chart] interior search {k:9s}: {nc} closed-form curve(s) -> regions "
              f"{labels}, every door point covered, no spurious region ({cells} cells)")
    r = test_the_wall_decides_what_is_inside()
    print(f"[chart] wall interior: far sphere encloses {r['far']} door region(s), "
          f"sphere at a door encloses {r['door']}")
    r = test_facet_sigma()
    print(f"[chart] facet test (ii): through a door -> '{r['hit']}'; far facet clear "
          f"at sigma depth {r['far_depth']}")
    print("[chart] ALL TESTS PASSED")
