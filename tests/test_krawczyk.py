"""Method C (krawczyk.py): the general door certificate -- sound, but small cells only.

What is pinned:

  * INTERVAL ARITHMETIC ENCLOSES. Random interval operands, random points
    inside them: every exact result lies in the computed interval (+, -, *,
    /, sqr, sqrt, sin, cos, sums of products).
  * INTERVAL FK ENCLOSES the arm's forward kinematics: every posture sampled
    in a joint box lies inside the boxed elbow, wrist, tool and rotation.
  * THE CERTIFICATE IS SOUND: no cell containing a closed-form door point is
    ever certified, at any size.
  * ...AND IT IS ONLY PRACTICAL FOR SMALL CELLS: tiny cells (half-width 1e-4
    scaled) are almost all certified; cells of the door search's size (1e-2)
    are not -- the Newton-Kantorovich limit (radius ~ sigma_min^2), recorded
    so a future change that improves it shows up here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_3d import PATCHES  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.interval import Interval, imatmul  # noqa: E402
from zzy_rss.krawczyk import KrawczykCertifier, SerialChain, interval_fk  # noqa: E402
from zzy_rss.params import Config, Page3Params  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
P3 = Page3Params()


def test_interval_arithmetic_encloses() -> dict:
    rng = np.random.default_rng(0)
    n, k = 2000, 30
    c1, c2 = rng.normal(size=(2, n)) * 3
    r1, r2 = 10 ** rng.uniform(-6, 0, (2, n))
    a, b = Interval.around(c1, r1), Interval.around(c2, r2)
    pos = Interval.around(np.abs(c2) + 2 * r2 + 0.1, r2)
    ops = {"add": (a + b, lambda x, y: x + y), "sub": (a - b, lambda x, y: x - y),
           "mul": (a * b, lambda x, y: x * y), "sqr": (a.sqr(), lambda x, y: x * x),
           "sin": (a.sin(), lambda x, y: np.sin(x)), "cos": (a.cos(), lambda x, y: np.cos(x)),
           "div": (a / pos, None), "sqrt": (pos.sqrt(), None)}
    bad = 0
    for _ in range(k):
        x = c1 + rng.uniform(-1, 1, n) * r1
        y = c2 + rng.uniform(-1, 1, n) * r2
        yp = pos.lo + rng.uniform(0, 1, n) * (pos.hi - pos.lo)
        for name, (I, f) in ops.items():
            v = (x / yp) if name == "div" else (np.sqrt(yp) if name == "sqrt" else f(x, y))
            bad += int(np.sum(~I.contains(v)))
    A = Interval.around(rng.normal(size=(50, 3, 3)), 1e-3)
    B = Interval.around(rng.normal(size=(50, 3, 3)), 1e-3)
    M = imatmul(A, B)
    for _ in range(k):
        a_ = A.lo + rng.uniform(0, 1, A.shape) * (A.hi - A.lo)
        b_ = B.lo + rng.uniform(0, 1, B.shape) * (B.hi - B.lo)
        bad += int(np.sum(~M.contains(a_ @ b_)))
    assert bad == 0, bad
    return dict(checks=k * (n * len(ops) + 50 * 9))


def test_interval_fk_encloses() -> dict:
    rng = np.random.default_rng(1)
    qc = rng.uniform(-2, 2, (200, 7))
    r = 10 ** rng.uniform(-4, -1, (200, 7))
    fk = interval_fk(SerialChain.srs(ARM), Interval.around(qc, r))
    bad = n = 0
    for k in range(200):
        for _ in range(20):
            f = ARM.frames(qc[k] + rng.uniform(-1, 1, 7) * r[k])
            for name in ("elbow", "wrist", "tool"):
                bad += int(not np.all(fk[name][k].contains(f[name])))
            bad += int(not np.all(fk["R"][k].contains(f["R07"])))
            n += 4
    assert bad == 0, bad
    return dict(checks=n)


def test_certificate_is_sound_and_small() -> dict:
    surf = WeldSurface(PATCHES["exp2"])
    ch = SrsChart.for_surface(ARM, surf, P3)
    kc = KrawczykCertifier(ch)
    rng = np.random.default_rng(2)
    pts = np.vstack([c.pts for c in door_curves(CFG, ARM, surf, P3)])
    pts = pts[np.all((pts[:, :2] >= 0) & (pts[:, :2] <= 1), axis=1)][::4]
    for h in (1e-4, 1e-3, 1e-2, 5e-2):
        H = np.full(pts.shape, h) / ch.scale
        C = pts + rng.uniform(-0.9, 0.9, pts.shape) * H
        assert not kc.krawczyk(C, H).ok.any(), f"a cell holding a door point was certified (h={h})"
    C = rng.uniform(ch.lo, ch.hi, (400, 3))
    rate = {}
    for h in (1e-4, 1e-2):
        r = kc.krawczyk(C, np.full((400, 3), h) / ch.scale)
        have = np.array([w != "no IK at the centre" for w in r.why])
        rate[h] = float(r.ok[have].mean())
    assert rate[1e-4] > 0.9 and rate[1e-2] < 0.05, rate
    return dict(door_points=len(pts), rate=rate)


if __name__ == "__main__":
    r = test_interval_arithmetic_encloses()
    print(f"[krawczyk] interval arithmetic: {r['checks']} sampled results, all enclosed")
    r = test_interval_fk_encloses()
    print(f"[krawczyk] interval FK: {r['checks']} sampled frames, all enclosed")
    r = test_certificate_is_sound_and_small()
    print(f"[krawczyk] sound: no cell holding one of {r['door_points']} door points certified "
          f"(half-widths 1e-4..5e-2); certified share: {r['rate'][1e-4]:.0%} of cells at 1e-4, "
          f"{r['rate'][1e-2]:.0%} at 1e-2 (the search's cell size)")
    print("[krawczyk] ALL TESTS PASSED")
