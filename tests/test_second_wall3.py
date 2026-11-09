"""Exit 2 at d = 3: when does a second wall exclude a door curve? (findings 47)

The verdict a second wall can add -- "this door is unreachable, skip it" --
is the one direction that can make INFEASIBLE false, so the test that
decides it (page3.tube_excluded) is pinned here on a closed TDC sphere whose
answer is known exactly:

  * a curve far outside the sphere is excluded;
  * a curve through the sphere, one wholly inside it, and one outside but
    within its tube radius of the surface are all REFUSED -- the last is the
    case a centre probe alone would wave through;
  * only the ON-PATCH part of a curve counts (off the patch there is no
    door); a curve with no on-patch part is trivially excluded;
  * a curve with several on-patch pieces is excluded only if every piece is.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.facets import SigmaTube  # noqa: E402
from zzy_rss.page3 import tube_excluded  # noqa: E402
from zzy_rss.tangential import reconstruct  # noqa: E402

LO, HI, SCALE = np.zeros(3), np.ones(3), np.ones(3)
C, R = np.array([0.5, 0.5, 0.5]), 0.25
RADII = np.array([0.02, 0.02, 0.02])


def _sphere():
    def f(x):
        d = x - C
        return R * R - np.sum(d * d, axis=1), -2.0 * d
    rng = np.random.default_rng(7)
    rec = reconstruct(f, rng.uniform(0, 1, (300, 3)), 0.04, LO - 0.1, HI + 0.1, rng=rng)
    assert rec.closed, rec.note
    return rec.points, np.asarray(rec.simplices)


def _curve(a, b, n=40):
    pts = np.linspace(np.asarray(a, float), np.asarray(b, float), n)
    return SimpleNamespace(pts=pts, tube=SigmaTube(polyline=pts, radii=RADII))


def _excl(V, S, c):
    return tube_excluded(V, S, c, LO, HI, SCALE)


def test_cases() -> dict:
    V, S = _sphere()
    out = {}
    # Far outside: excluded.
    ok, why = _excl(V, S, _curve([0.05, 0.05, 0.1], [0.95, 0.05, 0.1]))
    assert ok, why
    out["outside"] = why
    # Through the sphere: refused (a triangle touches the tube).
    ok, why = _excl(V, S, _curve([0.1, 0.5, 0.5], [0.9, 0.5, 0.5]))
    assert not ok and "touches" in why, why
    out["through"] = why
    # Wholly inside, clear of the surface: refused (inside).
    ok, why = _excl(V, S, _curve([0.45, 0.5, 0.5], [0.55, 0.5, 0.5]))
    assert not ok and "inside" in why, why
    out["inside"] = why
    # Outside, but the tube grazes the surface: the polyline itself never
    # enters, a probe at its start is outside -- and it is still refused.
    g = 0.5 + R + 0.5 * RADII[0]
    ok, why = _excl(V, S, _curve([g, 0.2, 0.5], [g, 0.8, 0.5]))
    assert not ok and "touches" in why, why
    out["grazing"] = why
    # Off the patch entirely (u < 0): no door, trivially excluded.
    ok, why = _excl(V, S, _curve([-0.3, 0.1, 0.5], [-0.3, 0.9, 0.5]))
    assert ok and "no part" in why, why
    out["off_patch"] = why
    # Only the on-patch part counts: the curve dives off the patch (v > 1)
    # and the part that would run through the sphere's u-range there is cut.
    pts = np.array([[0.05, 0.9, 0.5], [0.05, 1.3, 0.5], [0.95, 1.3, 0.5], [0.95, 0.9, 0.5]])
    pl = np.vstack([np.linspace(pts[i], pts[i + 1], 20) for i in range(3)])
    ok, why = _excl(V, S, SimpleNamespace(pts=pl, tube=SigmaTube(polyline=pl, radii=RADII)))
    assert ok and "2 on-patch piece" in why, why
    out["two_pieces"] = why
    # Two pieces, the second one inside: refused.
    pts = np.array([[0.05, 0.9, 0.5], [0.05, 1.3, 0.5], [0.5, 1.3, 0.5], [0.5, 0.5, 0.5]])
    pl = np.vstack([np.linspace(pts[i], pts[i + 1], 20) for i in range(3)])
    ok, why = _excl(V, S, SimpleNamespace(pts=pl, tube=SigmaTube(polyline=pl, radii=RADII)))
    assert not ok, why
    out["second_piece_in"] = why
    return out


if __name__ == "__main__":
    for k, v in test_cases().items():
        print(f"[second-wall-3d] {k:16s}: {v}")
    print("[second-wall-3d] all passed")
