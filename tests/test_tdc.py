"""IJRR 4.3 reconstruction by the tangential Delaunay complex (findings 39).

What is pinned here:

  * KNOWN TOPOLOGY. On implicit shapes whose answer is known, the complex
    must come out closed with the right number of loops (d = 2) or the right
    Euler characteristic (sphere 2, torus 0 at d = 3), with every vertex on
    the manifold.
  * IT REFUSES. A manifold that leaves the window is not closed and must be
    reported open; a piece of wall the net never reached must fail the
    sidedness check. A reconstruction that cannot refuse proves nothing.
  * NO GRID. The SVM wall is reconstructed with the legacy contour tracer
    trapped, so any grid evaluation would raise.
  * DIMENSION GUARDS. A d = 3 polytope is never passed by the segment-only
    facet check, and a periodic page is refused rather than flattened.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import zzy_rss.wall as W  # noqa: E402
from zzy_rss.params import Config  # noqa: E402
from zzy_rss.tangential import inside, reconstruct  # noqa: E402


# --------------------------------------------------------------------------
# implicit test shapes: f > 0 inside, gradient in closed form
# --------------------------------------------------------------------------


def _ball(r, c):
    c = np.asarray(c, dtype=float)

    def f(x):
        y = x - c
        return r * r - np.sum(y * y, axis=1), -2.0 * y
    return f


def _union(*fs):
    def f(x):
        vals = [ff(x) for ff in fs]
        V = np.stack([v for v, _ in vals])
        G = np.stack([g for _, g in vals])
        i = np.argmax(V, axis=0)
        n = np.arange(len(x))
        return V[i, n], G[i, n]
    return f


def _minus(fa, fb):
    """Inside a and outside b."""
    def f(x):
        va, ga = fa(x)
        vb, gb = fb(x)
        pick_a = va < -vb
        return np.where(pick_a, va, -vb), np.where(pick_a[:, None], ga, -gb)
    return f


def _torus(R=0.6, r=0.25):
    def f(x):
        q = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        qs = np.maximum(q, 1e-12)
        dq = -2.0 * (q - R)
        g = np.column_stack([dq * x[:, 0] / qs, dq * x[:, 1] / qs, -2.0 * x[:, 2]])
        return r * r - ((q - R) ** 2 + x[:, 2] ** 2), g
    return f


def _half_space(d):
    def f(x):
        g = np.zeros_like(x)
        g[:, -1] = 1.0
        return x[:, -1] - 0.1, g
    return f


def _run(f, d, eta, seeds=None, side=None, rng_seed=1, **kw):
    rng = np.random.default_rng(rng_seed)
    lo, hi = -np.ones(d), np.ones(d)
    if seeds is None:
        seeds = rng.uniform(-1.0, 1.0, (300, d))
    if side is None:
        side = rng.uniform(-1.0, 1.0, (400, d))
    return reconstruct(f, seeds, eta, lo, hi, side_points=side, rng=rng, **kw)


# --------------------------------------------------------------------------


def test_known_topology() -> list:
    cases = [
        ("circle", _ball(0.5, [0, 0]), 2, 0.03, dict(loops=1)),
        ("two circles", _union(_ball(0.3, [-0.5, 0]), _ball(0.3, [0.5, 0])),
         2, 0.03, dict(loops=2)),
        ("annulus", _minus(_ball(0.8, [0, 0]), _ball(0.4, [0, 0])),
         2, 0.03, dict(loops=2)),
        ("sphere", _ball(0.6, [0, 0, 0]), 3, 0.06, dict(euler=2)),
        ("torus", _torus(), 3, 0.05, dict(euler=0)),
    ]
    rows = []
    for name, f, d, eta, want in cases:
        rec = _run(f, d, eta)
        assert rec.closed, f"{name}: not closed ({rec.note})"
        if "loops" in want:
            assert len(rec.loops) == want["loops"], (
                f"{name}: {len(rec.loops)} loops, expected {want['loops']}")
        if "euler" in want:
            assert rec.stats["euler"] == want["euler"], (
                f"{name}: Euler characteristic {rec.stats['euler']}, "
                f"expected {want['euler']}")
        fv, _ = f(rec.points)
        assert float(np.max(np.abs(fv))) < 1e-6, f"{name}: vertex off the manifold"
        assert rec.stats["side_wrong"] == 0 and rec.stats["side_checked"] > 300
        rows.append((name, len(rec.points), len(rec.simplices),
                     rec.stats.get("euler"), rec.stats["fix_rounds"]))
    return rows


def test_inside_matches_the_shape() -> int:
    """The inside test the sidedness check relies on, against the truth."""
    rng = np.random.default_rng(7)
    n = 0
    for f, d, eta in ((_minus(_ball(0.8, [0, 0]), _ball(0.4, [0, 0])), 2, 0.02),
                      (_ball(0.6, [0, 0, 0]), 3, 0.05),
                      (_torus(), 3, 0.04)):
        rec = _run(f, d, eta)
        assert rec.closed, rec.note
        q = rng.uniform(-1.0, 1.0, (2000, d))
        fq, gq = f(q)
        # Away from the surface by more than the chord sagitta.
        far = np.abs(fq) / np.linalg.norm(gq, axis=1) > 2 * eta
        got = inside(q[far], rec.points, rec.simplices, d - 1)
        assert np.array_equal(got, fq[far] > 0), "inside test disagrees with f"
        n += int(far.sum())
    return n


def test_open_manifold_is_refused() -> list:
    out = []
    for d, eta in ((2, 0.03), (3, 0.08)):
        rec = _run(_half_space(d), d, eta)
        assert not rec.closed, f"d={d}: a manifold leaving the window was closed"
        out.append((d, rec.note))
    return out


def test_missed_component_fails_sidedness() -> str:
    """Seed only ONE of two circles: the net never reaches the other, the
    complex is a perfectly closed single loop, and only the sidedness check
    can tell that a piece of wall is missing. It must."""
    f = _union(_ball(0.3, [-0.5, 0]), _ball(0.3, [0.5, 0]))
    rng = np.random.default_rng(3)
    # Seeds just off the far side of the LEFT circle. (Seeds near a centre
    # are no good: the gradient vanishes there and Newton overshoots onto
    # whichever circle it likes.)
    a = rng.uniform(np.radians(120), np.radians(240), 20)
    seeds = np.column_stack([-0.5 + 0.32 * np.cos(a), 0.32 * np.sin(a)])
    side = np.vstack([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.9]])
    rec = _run(f, 2, 0.03, seeds=seeds, side=side)
    assert len(rec.loops) == 0 or not rec.closed, "missing circle went unnoticed"
    assert "sidedness" in rec.note, rec.note
    return rec.note


def test_inconsistencies_are_fixed_by_weights() -> tuple:
    """The torus produces inconsistent stars; the weight perturbation is what
    makes it closed. Without it the complex must NOT come out closed."""
    raw = _run(_torus(), 3, 0.05, fix_rounds=0)
    fixed = _run(_torus(), 3, 0.05)
    assert raw.stats["n_inconsistent"] > 0, "no inconsistency to fix"
    assert not raw.closed, "inconsistent complex reported closed"
    assert fixed.closed and fixed.stats["n_inconsistent"] == 0
    return raw.stats["n_inconsistent"], fixed.stats["fix_rounds"]


def _toy_wall():
    """An SVM wall on the real page window: a start blob vs the Fig. 5 ring."""
    cfg = Config()
    rng = np.random.default_rng(5)
    seed = np.column_stack([rng.uniform(0.0, 0.35, 150), rng.uniform(-1.2, 0.8, 150)])
    goal = np.column_stack([rng.uniform(0.6, 1.0, 150), rng.uniform(-3.0, 3.0, 150)])
    rest = np.vstack([goal, W.virtual_boundary_samples(cfg, 80)])
    return cfg, W.train_wall(cfg, seed, rest), seed, rest


def test_svm_wall_without_any_grid() -> dict:
    cfg, wall, seed, rest = _toy_wall()

    def trap(*_a, **_k):
        raise AssertionError("grid contour tracer called")

    saved = W.reconstruct_wall_contour
    W.reconstruct_wall_contour = trap
    try:
        poly = W.reconstruct_wall(cfg, wall, seed_pts=seed, rest_pts=rest)
    finally:
        W.reconstruct_wall_contour = saved
    assert poly.method == "tdc"
    assert poly.closed and poly.loops, poly.note
    f = wall.decision(poly.vertices)
    assert float(np.max(np.abs(f))) < 1e-5, "a vertex is off the wall"
    assert poly.stats["side_checked"] > 300 and poly.stats["side_wrong"] == 0
    for loop in poly.loops:
        assert np.allclose(loop[0], loop[-1]), "loop not repeated-closed"
    return dict(loops=len(poly.loops), segments=poly.n_segments,
                gamma=wall.gamma, eta=poly.stats["eta"])


def test_periodic_page_is_refused() -> str:
    cfg, wall, seed, rest = _toy_wall()
    cfg = dataclasses.replace(cfg, pagemap=dataclasses.replace(
        cfg.pagemap, theta_periodic=True))
    poly = W.reconstruct_wall(cfg, wall, seed_pts=seed, rest_pts=rest)
    assert not poly.closed and "periodic" in poly.note
    return poly.note


def test_d3_facets_are_never_passed() -> str:
    poly = W.Polytope(closed=True, n_segments=4, method="tdc",
                      vertices=np.eye(4, 3), simplices=np.array([[0, 1, 2]]))
    rep = W.check_polytope(Config(), None, None, None, (1, 1, 1), poly, [],
                           np.zeros((0, 2)), verbose=False)
    assert not rep.ok and rep.n_passed == 0
    return rep.failures[0][1]


if __name__ == "__main__":
    for name, n, m, chi, fr in test_known_topology():
        print(f"[tdc] topology   : {name:11s} {n:5d} vertices {m:5d} facets"
              + (f"  Euler {chi}" if chi is not None and name in ("sphere", "torus")
                 else "") + (f"  ({fr} weight round(s))" if fr else ""))
    print(f"[tdc] inside     : agrees with f on {test_inside_matches_the_shape()} "
          f"query points")
    for d, note in test_open_manifold_is_refused():
        print(f"[tdc] open d={d}   : refused -- {note}")
    print(f"[tdc] missed wall: refused -- {test_missed_component_fails_sidedness()}")
    raw, rounds = test_inconsistencies_are_fixed_by_weights()
    print(f"[tdc] weights    : {raw} inconsistent simplices without, closed "
          f"after {rounds} round(s) with")
    r = test_svm_wall_without_any_grid()
    print(f"[tdc] SVM wall   : {r['loops']} loop(s), {r['segments']} facets, "
          f"gamma {r['gamma']:.1f}, eta {r['eta']:.3f}, contour trapped")
    print(f"[tdc] periodic   : refused -- {test_periodic_page_is_refused()}")
    print(f"[tdc] d=3 facets : refused -- {test_d3_facets_are_never_passed()}")
    print("[tdc] ALL TESTS PASSED")
