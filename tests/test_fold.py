#!/usr/bin/env python3
"""The q4 fold as a door (Page3Params.fold_doors, coinduction).

    python3 tests/test_fold.py           # bounds + door search (~1 min)
    python3 tests/test_fold.py --e2e     # also the end-to-end proof on the fold patch

The fold patch is EXP 2's patch stretched to y = 0.70: past y ~ 0.5 the wrist
is beyond full reach, so the fold (q4 = 0) crosses the patch at u ~ 0.8.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_3d import PATCHES  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import I_Q2, I_Q4, I_Q6, SrsChart  # noqa: E402
from zzy_rss.doorset import door_pages, find_doorset  # noqa: E402
from zzy_rss.params import Config, Page3Params, SurfaceParams  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
P3 = Page3Params()
FOLD = SurfaceParams(p0=(0.28, -0.26, 0.41), p1=(0.28, 0.70, 0.41))


def _chart(sp=FOLD) -> SrsChart:
    return SrsChart.for_surface(ARM, WeldSurface(sp), P3)


def _ball_points(ch, C, R, rng):
    d = rng.normal(size=C.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return C + d * (rng.uniform(0, 1, (len(C), 1)) ** (1 / 3) * R[:, None]) / ch.scale


def test_fold_modulus_is_guaranteed() -> dict:
    """|sigma_k(z) - sigma_k(c)| <= fold_modulus over balls at and across the fold."""
    ch = _chart()
    rng = np.random.default_rng(0)
    n = viol = 0
    worst = 0.0
    for R0 in (0.002, 0.005, 0.01, 0.02, 0.05):
        C = np.column_stack([rng.uniform(0.6, 0.9, 20000), rng.uniform(0, 1, 20000),
                             rng.uniform(-np.pi, np.pi, 20000)])
        R = np.full(len(C), R0)
        w = ch.fold_modulus(C, R)
        sc = ch.sigma(C)
        for _ in range(5):
            sz = ch.sigma(_ball_points(ch, C, R, rng))
            for j, k in enumerate((I_Q2, I_Q6)):
                m = np.isfinite(sc[:, k]) & np.isfinite(sz[:, k]) & np.isfinite(w[:, j])
                diff = np.abs(sz[m, k] - sc[m, k])
                n += int(m.sum())
                viol += int(np.sum(diff > w[m, j] + 1e-12))
                if m.any():
                    worst = max(worst, float(np.max(diff / np.maximum(w[m, j], 1e-300))))
    assert n > 100_000 and viol == 0, (n, viol)
    return dict(pairs=n, violations=viol, worst_ratio=round(worst, 3))


def test_reach_lower_is_a_lower_bound() -> dict:
    out = {}
    for name, sp in (("fold", FOLD), ("exp2", PATCHES["exp2"])):
        ch = _chart(sp)
        rng = np.random.default_rng(1)
        C = np.column_stack([rng.uniform(-0.1, 1.1, 5000), rng.uniform(-0.1, 1.1, 5000),
                             rng.uniform(-3, 3, 5000)])
        R = rng.uniform(0.001, 0.1, 5000)
        lb = ch.reach_lower(C, R)
        worst = np.inf
        for _ in range(200):
            L = np.linalg.norm(ch.wrist(_ball_points(ch, C, R, rng)) - ch.S, axis=1)
            worst = min(worst, float(np.min(L - lb)))
        assert worst >= -1e-12, (name, worst)
        out[name] = f"min(L - bound) = {worst:.2e}"
    return out


def test_fold_is_a_door() -> dict:
    """The fold region is ONE door that flips s4 only (the q2 / q6 halo is
    excluded by the fold modulus) and joins two pages; the old rule keeps it
    unsupported."""
    ch = _chart()
    t = time.time()
    ds = find_doorset(ch, P3.tube_radius)
    dt = time.time() - t
    assert ds.complete
    folds = [d for d in ds.doors if 1 in d.flips]
    assert len(folds) == 1, [d.label() for d in ds.doors]
    f = folds[0]
    assert f.flips == (1,) and f.supported and f.sign_index is None, f.label()
    assert door_pages((1, 1, -1), f.flips) == frozenset({(1, 1, -1), (1, -1, -1)})
    P = f.points
    assert len(P) > 50
    slack = ch.sigma(P)[:, I_Q4]
    assert float(np.max(slack)) < 1e-6, float(np.max(slack))
    old = find_doorset(ch, P3.tube_radius, fold_doors=False)
    assert not [d for d in old.doors if "q4" in d.region.kinds and d.supported]
    # The existing patches have no fold: same regions as before.
    ex = find_doorset(_chart(PATCHES["exp2"]), P3.tube_radius)
    assert [d.region.kinds for d in ex.doors] == [("q2",), ("q2",)]
    return dict(doors=[d.label() for d in ds.doors], fold_points=len(P),
                fold_cells=len(f.region.centres), seconds=round(dt, 1))


def test_end_to_end() -> dict:
    """The fold patch has no goal within reach: INFEASIBLE, through the fold
    door (old rule: INCONCLUSIVE, the theta cut is touched at the fold)."""
    from zzy_rss.arm import SrsArm as _A
    from zzy_rss.params import PageMapParams, SceneParams
    from zzy_rss.proof import INFEASIBLE, prove_task
    from zzy_rss.scene import Scene
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=()))
    cfg = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall, gamma_search="gallop"))
    p3 = dataclasses.replace(P3, n_samples=6000, alg4_iters=8)
    t = time.time()
    res = prove_task(cfg, _A(cfg.arm), WeldSurface(FOLD), Scene(cfg.arm, cfg.scene),
                     (1, 1, -1), p3, verbose=False)
    assert res.verdict == INFEASIBLE, (res.verdict, res.reason)
    return dict(verdict=res.verdict, reason=res.reason, seconds=round(time.time() - t, 1))


if __name__ == "__main__":
    tests = [test_fold_modulus_is_guaranteed, test_reach_lower_is_a_lower_bound,
             test_fold_is_a_door]
    if "--e2e" in sys.argv:
        tests.append(test_end_to_end)
    for fn in tests:
        print(f"[fold] {fn.__name__}: {fn()}")
    print("[fold] ALL TESTS PASSED")
