"""The algebraic route in JOINT variables (algebraic_joint.py): no S-R-S structure.

The door systems are written for a generic serial revolute chain (c_i, s_i per
joint, the task, the door as a linear form, a spin fix). Their WITNESS SETS --
the solutions of a generic instance whose door equation is replaced by a
generic hyperplane -- are cached in tests/data/joint_witness.npz by
scripts/joint_monodromy.py. What is pinned:

  * THE CACHE IS GENUINE: every witness point satisfies F(., base) = 0 and the
    points are distinct -- a stale or edited cache fails here.
  * COMPLETENESS IS CERTIFIED, NOT GUESSED: the trace test passes on the
    cached set and FAILS when points are removed (the old stopping rule
    stopped at 32 of 64 on one seed).
  * EXACT AGREEMENT WITH THE CLOSED FORM: tracking the witness set to a real
    slice gives the same door points (u, theta) as the closed-form finder,
    same count, for a q2 slice and a q6 slice.
  * EVERY DOOR POINT IS PROVEN: Krawczyk (interval arithmetic) certifies a
    unique exact root in a 1e-10 box around each, and the arm's own FK puts
    it on the slice with the task orientation. A box without a root is
    never certified.
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
from zzy_rss.algebraic_joint import (DoorSystem, MonodromyResult, certify_root,  # noqa: E402
                                     real_doors, solve_instance, task_instance, trace_test)
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import arm_angle  # noqa: E402
from zzy_rss.doors import find_interior_doors_analytic  # noqa: E402
from zzy_rss.krawczyk import SerialChain  # noqa: E402
from zzy_rss.params import Config  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
CHAIN = SerialChain.srs(ARM)
CACHE = ROOT / "tests/data/joint_witness.npz"
SYSTEMS = {"q2": (1, 0, 0), "q6": (5, 4, 2)}      # label, spin, closed-form sign index
CASES = (("q2", "exp2", 0.5), ("q6", "exp4", 0.5))
KEYS = ("p0", "A", "Rt", "ab", "r0", "nu", "rd")


def _load(name):
    z = np.load(CACHE)
    label, spin, _ = SYSTEMS[name]
    base = {k: z[f"{name}_P_{k}"] for k in KEYS}
    return DoorSystem(CHAIN, label, spin), MonodromyResult(base, list(z[f"{name}_sols"]), 0)


def test_cache_is_genuine() -> dict:
    out = {}
    for name in SYSTEMS:
        sys_, m = _load(name)
        X = np.array(m.solutions)
        res = float(np.max(np.linalg.norm(sys_.Fb(X, m.base), axis=1)))
        d = np.linalg.norm(X[:, None] - X[None], axis=2) + np.eye(len(X)) * 1e9
        assert res < 1e-8 and d.min() > 1e-4, (name, res, d.min())
        out[name] = (len(X), res)
    return out


def test_trace_certifies_completeness() -> dict:
    out = {}
    for name in SYSTEMS:
        sys_, m = _load(name)
        full = trace_test(sys_, m)
        assert full.ok, (name, full)
        cut = MonodromyResult(m.base, m.solutions[:-3], 0)
        part = trace_test(sys_, cut)
        assert not part.ok and part.residual > 1e-3, (name, part)
        out[name] = (len(m.solutions), full.residual, part.residual)
    return out


def _closed_form(surf, v, si):
    line = surf.line(v)
    cfg = dataclasses.replace(CFG, weld=line.p)
    return sorted((d.exact_y, d.exact_theta) for d in find_interior_doors_analytic(
        cfg, ARM, line, None, si, radius=1e-3, r_th=1e-3, name="x") if 0.0 < d.exact_y < 1.0)


def test_slices_match_and_are_certified() -> dict:
    out = {}
    for name, scene, v in CASES:
        sys_, m = _load(name)
        surf = WeldSurface(PATCHES[scene])
        P = task_instance(surf, v)
        confs = real_doors(sys_, solve_instance(sys_, m, P), P)
        pts = []
        for u, q in confs:
            x = np.concatenate([np.cos(q), np.sin(q), [u]])
            cert = certify_root(sys_, x, P)
            assert cert.certified, (name, u)
            q = np.arctan2(cert.center[sys_.n:2 * sys_.n], cert.center[:sys_.n])
            assert abs(q[sys_.label]) < 1e-9
            T, R = ARM.fk(q)                # the arm's own FK, not the system's
            Tt, Rt = surf.pose(cert.center[-1], v)
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
        # soundness: a box shifted off the root holds no root -> never certified
        u, q = confs[0]
        x = np.concatenate([np.cos(q), np.sin(q), [u]]) + 1e-6
        assert not certify_root(sys_, x, P, polish=False, tries=1).certified
        out[name] = (scene, v, len(pts), len(confs), worst)
    return out


if __name__ == "__main__":
    for k, (n, r) in test_cache_is_genuine().items():
        print(f"[algebraic-joint] {k}: {n} cached witness points, max residual {r:.1e}")
    for k, (n, rf, rp) in test_trace_certifies_completeness().items():
        print(f"[algebraic-joint] {k}: trace test COMPLETE on {n} points (residual {rf:.1e}); "
              f"3 points removed -> fails (residual {rp:.1e})")
    for k, (sc, v, n, nc, w) in test_slices_match_and_are_certified().items():
        print(f"[algebraic-joint] {k} on {sc} v={v}: {n} door points ({nc} joint spellings), "
              f"same as the closed form (worst {w:.1e}); every one Krawczyk-certified")
    print("[algebraic-joint] ALL TESTS PASSED")
