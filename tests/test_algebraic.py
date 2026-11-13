"""The algebraic route (algebraic.py): all doors of a slice by homotopy continuation.

What is pinned:

  * EVERY PATH IS TRACKED: no path fails, on four patches.
  * EXACT AGREEMENT WITH THE CLOSED FORM, slice by slice: on every weld slice
    the homotopy returns the same q2 / q6 door points (u, theta) as the
    independent closed-form finder (doors.find_interior_doors_analytic, a
    quadratic in u), same count, to 1e-9. Completeness is checked, not
    assumed: a door the homotopy missed would show up as a count mismatch.
  * THEY ARE DOORS: sigma ~ 0 at every point, and every point lies inside a
    certified sigma region (doorset.py) -- the two complete methods agree.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_3d import PATCHES  # noqa: E402
from search_scene3 import surface  # noqa: E402
from zzy_rss.algebraic import doors_on_slice  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.doors import find_interior_doors_analytic  # noqa: E402
from zzy_rss.doorset import find_doorset  # noqa: E402
from zzy_rss.params import Config, Page3Params  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
P3 = Page3Params()
SCENES = dict(PATCHES)
SCENES["exp3-new"] = surface(0.40, 0.21, 45, 305)


def _closed_form(surf, v):
    line = surf.line(v)
    cfg = dataclasses.replace(CFG, weld=line.p)
    out = []
    for si, kind in ((0, "q2"), (2, "q6")):
        for d in find_interior_doors_analytic(cfg, ARM, line, None, si, radius=1e-3,
                                              r_th=1e-3, name="x"):
            if 0.0 < d.exact_y < 1.0:
                out.append((kind, d.exact_y, d.exact_theta))
    return sorted(out)


def test_matches_closed_form_and_sigma() -> dict:
    out = {}
    for name, sp in SCENES.items():
        surf = WeldSurface(sp)
        ch = SrsChart.for_surface(ARM, surf, P3)
        regions = find_doorset(ch, P3.tube_radius).doors
        n_pts = failed = 0
        worst = 0.0
        for v in np.linspace(0.0, 1.0, 21):
            sl = doors_on_slice(ch, float(v))
            failed += sl.failed
            alg = sorted((p["kind"], p["u"], p["theta"]) for p in sl.points
                         if p["spelling"] in ("q2=0", "q6=0") and 0.0 < p["u"] < 1.0)
            cf = _closed_form(surf, float(v))
            assert len(alg) == len(cf), f"{name} v={v}: homotopy {len(alg)} vs closed form {len(cf)}"
            for (k1, u1, t1), (k2, u2, t2) in zip(alg, cf):
                assert k1 == k2
                worst = max(worst, abs(u1 - u2), abs(np.angle(np.exp(1j * (t1 - t2)))))
            Z = np.array([[u, v, th] for _, u, th in alg]).reshape(-1, 3)
            if len(Z):
                s = ch.sigma(Z)
                assert float(np.max(np.minimum(s[:, 0], s[:, 1]))) < 1e-6
                inside = np.zeros(len(Z), dtype=bool)
                for d in regions:
                    inside |= d.region.contains(Z)
                assert inside.all(), f"{name}: a homotopy door point outside every sigma region"
            n_pts += len(alg)
        assert failed == 0 and worst < 1e-9, (failed, worst)
        out[name] = (n_pts, worst)
    return out


if __name__ == "__main__":
    for k, (n, w) in test_matches_closed_form_and_sigma().items():
        print(f"[algebraic] {k:9s}: {n} door points on 21 slices, same as the closed form "
              f"(worst difference {w:.1e}); sigma ~ 0 and inside certified regions")
    print("[algebraic] ALL TESTS PASSED")
