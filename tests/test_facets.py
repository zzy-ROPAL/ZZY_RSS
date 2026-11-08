"""IJRR Algorithm 3 on triangle facets (findings 41).

The robot's page is still 2-D, so a d = 3 wall is checked here against a
SYNTHETIC page whose answer is known exactly: C_obs is the spherical shell
0.4 <= |x| <= 0.8, valid space is inside 0.4 and outside 0.8, and the true
penetration depth of a shell point is min(|x| - 0.4, 0.8 - |x|).

What is pinned:

  * GEOMETRY. The enclosing ball is minimal (against a brute-force minimax)
    and contains the triangle; a midpoint split halves it exactly.
  * ALGORITHM 3 ON TRIANGLES. A TDC sphere inside the shell passes every
    facet; one that sits in valid space, or pokes out of the shell, or is too
    coarse for its depth, fails -- each with the reason the algorithm gives.
  * THE DOOR GUARD. A door curve lying on the wall rejects exactly the
    triangles it touches, before any depth is computed; the exact
    segment-triangle distance it rests on agrees with dense sampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.facets import (  # noqa: E402
    FacetOracle, SigmaTube, enclosing_ball,
    segment_triangle_distance, subdivide, triangle_hits_tube, tube_guard,
)
from zzy_rss.params import Config  # noqa: E402
from zzy_rss.tangential import reconstruct  # noqa: E402
from zzy_rss.wall import Polytope, check_polytope  # noqa: E402

R_IN, R_OUT = 0.4, 0.8


def _shell_oracle(guard=None, counter=None):
    def depth(c):
        if counter is not None:
            counter[0] += 1
        r = float(np.linalg.norm(c))
        return min(r - R_IN, R_OUT - r), "exact shell depth"

    return FacetOracle(
        scale=np.ones(3),
        guard=guard or (lambda v: None),
        virtual=lambda c: False,
        valid=lambda c: not (R_IN <= float(np.linalg.norm(c)) <= R_OUT),
        depth=depth,
    )


def _sphere_wall(r, eta, seed=1):
    def f(x):
        return r * r - np.sum(x * x, axis=1), -2.0 * x
    rng = np.random.default_rng(seed)
    rec = reconstruct(f, rng.uniform(-1, 1, (300, 3)), eta,
                      -np.ones(3), np.ones(3), rng=rng)
    assert rec.closed, rec.note
    return Polytope(closed=True, n_segments=len(rec.simplices), method="tdc",
                    vertices=rec.points, simplices=rec.simplices)


def _ellipsoid_wall(axes, eta, seed=2):
    ax = np.asarray(axes, dtype=float)

    def f(x):
        return 1.0 - np.sum((x / ax) ** 2, axis=1), -2.0 * x / ax ** 2
    rng = np.random.default_rng(seed)
    rec = reconstruct(f, rng.uniform(-1, 1, (300, 3)), eta,
                      -np.ones(3), np.ones(3), rng=rng)
    assert rec.closed, rec.note
    return Polytope(closed=True, n_segments=len(rec.simplices), method="tdc",
                    vertices=rec.points, simplices=rec.simplices)


# --------------------------------------------------------------------------


def test_enclosing_ball_is_minimal() -> float:
    """Against brute force: the minimax centre over a fine search."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    worst = 0.0
    n_obtuse = 0
    for _ in range(60):
        t = rng.normal(size=(3, 3))
        c, r = enclosing_ball(t)
        assert np.all(np.linalg.norm(t - c, axis=1) <= r + 1e-12), "vertex outside"
        best = min((minimize(lambda z: np.max(np.linalg.norm(t - z, axis=1)), x0,
                             method="Nelder-Mead",
                             options={"xatol": 1e-12, "fatol": 1e-12,
                                      "maxiter": 4000}).fun
                    for x0 in (t.mean(0), t[0], t[1], t[2])))
        assert r <= best + 1e-7, f"ball radius {r} exceeds the minimax {best}"
        worst = max(worst, r - best)
        a, b, cc = t
        sides = sorted([np.sum((b - cc) ** 2), np.sum((cc - a) ** 2), np.sum((a - b) ** 2)])
        n_obtuse += sides[2] >= sides[0] + sides[1]
    assert n_obtuse > 8, "the obtuse branch was not exercised"
    return worst


def test_subdivision_halves_the_ball() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        t = rng.normal(size=(3, 3))
        _, r = enclosing_ball(t)
        kids = subdivide(t)
        area = lambda x: 0.5 * np.linalg.norm(np.cross(x[1] - x[0], x[2] - x[0]))  # noqa: E731
        assert abs(sum(area(k) for k in kids) - area(t)) < 1e-12
        for k in kids:
            assert abs(enclosing_ball(k)[1] - 0.5 * r) < 1e-12


def test_wall_inside_the_shell_passes() -> dict:
    """The middle of the shell is 0.2 deep and the facets are ~0.06 wide:
    every one passes at depth 0."""
    poly = _sphere_wall(0.6, 0.06)
    calls = [0]
    rep = check_polytope(Config(), None, None, None, (1, 1, 1), poly, [],
                         np.zeros((0, 3)), verbose=False,
                         oracle=_shell_oracle(counter=calls))
    assert rep.ok and rep.n_passed == rep.n_facets == len(poly.simplices)
    assert rep.max_recursion == 0
    return dict(facets=rep.n_facets, depth_calls=calls[0])


def test_shallow_wall_needs_subdivision() -> dict:
    """A wall 0.03 inside C_obs with facets of radius up to ~0.04: some
    facets fail pd >= r at depth 0 and must pass after splitting."""
    poly = _sphere_wall(0.43, 0.05)
    rep = check_polytope(Config(), None, None, None, (1, 1, 1), poly, [],
                         np.zeros((0, 3)), verbose=False, oracle=_shell_oracle())
    assert rep.ok, rep.failures[:3]
    assert rep.max_recursion >= 1, "nothing needed subdividing; test too easy"
    return dict(facets=rep.n_facets, max_recursion=rep.max_recursion)


def test_bad_walls_fail_for_the_right_reason() -> dict:
    out = {}
    cfg = Config()
    # Entirely in valid space.
    rep = check_polytope(cfg, None, None, None, (1, 1, 1), _sphere_wall(0.3, 0.05),
                         [], np.zeros((0, 3)), verbose=False, oracle=_shell_oracle())
    assert not rep.ok and rep.n_passed == 0
    assert all("VALID" in why for _, why in rep.failures)
    out["in valid space"] = rep.failures[0][1]
    # Pokes out of the shell along x only.
    rep = check_polytope(cfg, None, None, None, (1, 1, 1),
                         _ellipsoid_wall((0.95, 0.6, 0.6), 0.08), [],
                         np.zeros((0, 3)), verbose=False, oracle=_shell_oracle())
    assert not rep.ok and 0 < rep.n_passed < rep.n_facets
    bad = np.array([c for c, _ in rep.failure_points])
    assert np.all(np.linalg.norm(bad, axis=1) > R_OUT - 0.06), \
        "a failure away from where the wall leaves C_obs"
    out["pokes out"] = f"{rep.n_facets - rep.n_passed}/{rep.n_facets} failed"
    # Grazing the inner boundary by 0.001: it must fail, and at a BOUNDED
    # cost. With the segment cap (depth 8, 4^8 leaves per triangle) this case
    # took 1.35 M depth queries; facet_max_depth_tri caps every facet at
    # 1 + 4 + ... + 4^5 = 1365 nodes.
    calls = [0]
    poly = _sphere_wall(0.401, 0.05)
    rep = check_polytope(cfg, None, None, None, (1, 1, 1), poly,
                         [], np.zeros((0, 3)), verbose=False,
                         oracle=_shell_oracle(counter=calls))
    assert not rep.ok
    assert rep.max_recursion <= cfg.wall.facet_max_depth_tri
    assert calls[0] <= 1365 * rep.n_facets
    reasons = sorted({why.split(";")[0] for _, why in rep.failures})
    assert "depth limit" in reasons, reasons
    out["grazing"] = f"{reasons}, {calls[0]} depth queries"
    return out


def test_segment_triangle_distance_is_exact() -> float:
    rng = np.random.default_rng(4)
    worst = 0.0
    u = np.linspace(0, 1, 61)
    s_, t_ = np.meshgrid(u, u, indexing="ij")
    keep = s_ + t_ <= 1.0
    s_, t_ = s_[keep], t_[keep]
    for _ in range(100):
        tri = rng.normal(size=(3, 3))
        p0, p1 = rng.normal(size=3), rng.normal(size=3)
        d = segment_triangle_distance(p0, p1, tri)
        P = tri[0] + s_[:, None] * (tri[1] - tri[0]) + t_[:, None] * (tri[2] - tri[0])
        Q = p0 + np.linspace(0, 1, 201)[:, None] * (p1 - p0)
        brute = np.min(np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2))
        assert d <= brute + 1e-12, "exact distance above a sampled one"
        worst = max(worst, brute - d)
    assert worst < 0.05, f"exact and sampled distances disagree by {worst}"
    # Piercing segment.
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert segment_triangle_distance(np.array([.2, .2, -1.]), np.array([.2, .2, 1.]), tri) == 0.0
    return worst


def test_door_tube_rejects_exactly_the_triangles_it_touches() -> dict:
    """A door curve ON the wall (the equator of the r = 0.6 sphere). Every
    triangle within the tube's ellipsoidal radius is rejected before any
    depth is computed; triangles away from it pass; a door well inside the
    valid core rejects nothing."""
    poly = _sphere_wall(0.6, 0.06)
    a = np.linspace(0, 2 * np.pi, 121)
    equator = np.column_stack([0.6 * np.cos(a), 0.6 * np.sin(a), np.zeros_like(a)])
    tube = SigmaTube(polyline=equator, radii=np.array([0.02, 0.02, 0.05]))
    calls = [0]
    rep = check_polytope(Config(), None, None, None, (1, 1, 1), poly, [],
                         np.zeros((0, 3)), verbose=False,
                         oracle=_shell_oracle(guard=tube_guard([tube]), counter=calls))
    assert not rep.ok
    hit = {i for i, why in rep.failures}
    assert all("Sigma tube" in why for _, why in rep.failures)
    for i, s_ in enumerate(poly.simplices):
        assert (i in hit) == triangle_hits_tube(poly.vertices[s_], tube)
    # The rejected triangles straddle z = 0 within the tube's z radius.
    z = poly.vertices[poly.simplices[sorted(hit)]][:, :, 2]
    assert np.all(np.min(np.abs(z), axis=1) <= 0.05 + 0.06)
    inner = SigmaTube(polyline=equator * (0.2 / 0.6), radii=np.array([0.02] * 3))
    rep2 = check_polytope(Config(), None, None, None, (1, 1, 1), poly, [],
                          np.zeros((0, 3)), verbose=False,
                          oracle=_shell_oracle(guard=tube_guard([inner])))
    assert rep2.ok
    return dict(rejected=len(hit), facets=rep.n_facets,
                depth_calls=calls[0])


if __name__ == "__main__":
    print(f"[facets] enclosing ball : minimal on 60 triangles "
          f"(worst excess {test_enclosing_ball_is_minimal():.1e})")
    test_subdivision_halves_the_ball()
    print("[facets] subdivision    : 4 children, same area, exactly half the radius")
    r = test_wall_inside_the_shell_passes()
    print(f"[facets] mid-shell wall : {r['facets']}/{r['facets']} triangles pass at "
          f"depth 0 ({r['depth_calls']} depth queries)")
    r = test_shallow_wall_needs_subdivision()
    print(f"[facets] shallow wall   : {r['facets']}/{r['facets']} pass, needing "
          f"subdivision to depth {r['max_recursion']}")
    for k, v in test_bad_walls_fail_for_the_right_reason().items():
        print(f"[facets] bad wall       : {k:15s} -> {v}")
    print(f"[facets] seg-tri dist   : exact, never above dense sampling "
          f"(gap {test_segment_triangle_distance_is_exact():.1e})")
    r = test_door_tube_rejects_exactly_the_triangles_it_touches()
    print(f"[facets] door tube      : rejects {r['rejected']}/{r['facets']} triangles, "
          f"exactly those it touches")
    print("[facets] ALL TESTS PASSED")
