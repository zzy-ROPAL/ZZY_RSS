"""The algebraic route in JOINT variables (algebraic_joint.py): no S-R-S structure.

The door systems are written for a generic serial revolute chain (c_i, s_i
per joint, the task, s_label = 0, a spin fix). Their generic solution sets
(64 each, found by monodromy) are cached in tests/data/joint_monodromy.npz by
scripts/joint_monodromy.py. What is pinned:

  * THE CACHE IS GENUINE: every cached root satisfies F(., P0) = 0 and the
    roots are distinct -- a stale or edited cache fails here, not later.
  * EXACT AGREEMENT WITH THE CLOSED FORM: tracking the 64 roots to a real
    weld slice gives the same door points (u, theta) as the independent
    closed-form finder (a quadratic in u), same count -- for a q2 slice and a
    q6 slice. A missed door would be a count mismatch.
  * THEY ARE REAL DOORS: each returned configuration has q_label = 0 and puts
    the tool on the slice with the task's orientation (FK, not the system).

About 4 minutes (two slices x 64 paths).
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
from zzy_rss.algebraic_joint import (DoorSystem, MonodromyResult, real_doors,  # noqa: E402
                                     solve_instance, task_instance)
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import arm_angle  # noqa: E402
from zzy_rss.doors import find_interior_doors_analytic  # noqa: E402
from zzy_rss.krawczyk import SerialChain  # noqa: E402
from zzy_rss.params import Config  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
CHAIN = SerialChain.srs(ARM)
CACHE = ROOT / "tests/data/joint_monodromy.npz"
SYSTEMS = {"q2": (1, 0, 0), "q6": (5, 4, 2)}      # label, spin, closed-form sign index
CASES = (("q2", "exp2", 0.5), ("q6", "exp4", 0.5))


def _load(name):
    z = np.load(CACHE)
    label, spin, _ = SYSTEMS[name]
    base = {k: z[f"{name}_P_{k}"] for k in ("p0", "A", "Rt", "ab")}
    return DoorSystem(CHAIN, label, spin), MonodromyResult(base, list(z[f"{name}_sols"]), 0)


def test_cache_is_genuine() -> dict:
    out = {}
    for name in SYSTEMS:
        sys_, m = _load(name)
        res = max(float(np.linalg.norm(sys_.F(x, m.base))) for x in m.solutions)
        X = np.array(m.solutions)
        gap = min(np.linalg.norm(X[i] - X[j]) for i in range(len(X)) for j in range(i))
        assert res < 1e-8 and gap > 1e-4, (name, res, gap)
        assert len(m.solutions) == 64, (name, len(m.solutions))
        out[name] = (len(m.solutions), res)
    return out


def _closed_form(surf, v, si):
    line = surf.line(v)
    cfg = dataclasses.replace(CFG, weld=line.p)
    return sorted((d.exact_y, d.exact_theta) for d in find_interior_doors_analytic(
        cfg, ARM, line, None, si, radius=1e-3, r_th=1e-3, name="x") if 0.0 < d.exact_y < 1.0)


def test_matches_closed_form() -> dict:
    out = {}
    for name, scene, v in CASES:
        sys_, m = _load(name)
        surf = WeldSurface(PATCHES[scene])
        P = task_instance(surf, v)
        confs = real_doors(sys_, solve_instance(sys_, m, P), P)
        pts = []
        for u, q in confs:
            # really a door configuration on the slice, checked by FK
            assert abs(q[sys_.label]) < 1e-8
            T, R = ARM.fk(q)            # the arm's own FK, not the system's
            Tt, Rt = surf.pose(u, v)
            assert np.linalg.norm(T - Tt) < 1e-9 and np.linalg.norm(R - Rt) < 1e-9
            th = arm_angle(ARM, q)
            if not any(abs(u - a) < 1e-7 and abs(np.angle(np.exp(1j * (th - b)))) < 1e-6
                       for a, b in pts):
                pts.append((u, th))
        pts.sort()
        cf = _closed_form(surf, v, SYSTEMS[name][2])
        assert len(pts) == len(cf), (name, pts, cf)
        worst = max(max(abs(a - c), abs(np.angle(np.exp(1j * (b - d)))))
                    for (a, b), (c, d) in zip(pts, cf))
        assert worst < 1e-8, (name, worst)
        out[name] = (scene, v, len(pts), len(confs), worst)
    return out


if __name__ == "__main__":
    for k, (n, r) in test_cache_is_genuine().items():
        print(f"[algebraic-joint] {k}: {n} cached generic roots, max residual {r:.1e}")
    for k, (sc, v, n, nc, w) in test_matches_closed_form().items():
        print(f"[algebraic-joint] {k} on {sc} v={v}: {n} door points ({nc} joint spellings), "
              f"same as the closed form (worst difference {w:.1e})")
    print("[algebraic-joint] ALL TESTS PASSED")
