"""Fail-fast budgets never change a verdict, only how soon INCONCLUSIVE comes
(findings 44).

What is pinned:

  * Algorithm 1 on inseparable data stops at gamma_stop instead of scanning
    0.1 at a time to gamma_max; outside any budget it is unchanged.
  * A wall stopped by its budget is a FAILED wall: its region stays in the
    certificate with ok=False, and the run is INCONCLUSIVE -- never
    INFEASIBLE (2-D run_proof and 3-D prove3 alike).
  * A run that exhausts its own budget is INCONCLUSIVE "stopped by budget".
  * Escalation re-runs the whole proof; a rung that was stopped is followed
    by a fresh, larger rung, and the first conclusive rung is the answer.
  * Structural INCONCLUSIVE reasons are not retried.
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
sys.path.insert(0, str(ROOT / "tests"))

import run_3d as R  # noqa: E402
from test_witness import _scene, _zgap  # noqa: E402
from zzy_rss import budget as B  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.flood import INCONCLUSIVE, INFEASIBLE, prove_line_escalating, run_proof  # noqa: E402
from zzy_rss.page3 import prove3  # noqa: E402
from zzy_rss.params import (BudgetParams, Config, Page3Params, PageMapParams,  # noqa: E402
                            SceneParams)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.source import bootstrap_roadmap  # noqa: E402
from zzy_rss.wall import train_wall  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402


def _exp2(budget: BudgetParams):
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, _zgap())
    return dataclasses.replace(cfg, budget=budget), arm, weld, scene


def test_gamma_stop() -> dict:
    """Two classes sharing points are inseparable at every gamma."""
    cfg = Config()
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(40, 2))
    seed, rest = np.vstack([a, a[:5]]), np.vstack([rng.uniform(0, 1, size=(40, 2)), a[:5]])
    bp = BudgetParams(gamma_stop=3.0)
    t0 = time.time()
    try:
        with B.wall_attempt(bp, "synthetic"):
            train_wall(cfg, seed, rest)
        raise AssertionError("inseparable data passed Algorithm 1")
    except B.BudgetExceeded as e:
        assert e.scope == "wall" and e.gamma > 3.0 and B.STOP in e.reason, e.reason
        t_stop, g_stop = time.time() - t0, e.gamma
    # Outside any budget the scan is the unbudgeted one: to gamma_max.
    small = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall, gamma_max=6.0))
    w = train_wall(small, seed, rest)
    assert w.gamma_capped and not B.active()
    return dict(stop_s=t_stop, gamma=g_stop)


def test_stopped_wall_is_failed_not_dropped_2d() -> dict:
    # gamma_init is 1.0, so gamma_stop 0.5 stops the very first fit.
    cfg, arm, weld, scene = _exp2(BudgetParams(gamma_stop=0.5))
    srcs, sig = bootstrap_roadmap(cfg, arm, weld, scene, n_samples=1200)
    seeds = [("++-", r.index) for r in srcs["++-"].regions() if r.reaches_start]
    res = run_proof(cfg, arm, weld, scene, None, sig, sources=srcs, seeds=seeds,
                    verbose=False)
    assert res.verdict == INCONCLUSIVE, res.verdict
    stopped = [c for c in res.certificate if c["note"].startswith(B.STOP)]
    assert stopped and all(not c["ok"] for c in stopped), res.certificate
    # The same roadmap without the stop is a proof: the stop is the ONLY
    # thing between this run and INFEASIBLE.
    ok = run_proof(dataclasses.replace(cfg, budget=BudgetParams()), arm, weld, scene,
                   None, sig, sources=srcs, seeds=seeds, verbose=False)
    assert ok.verdict == INFEASIBLE, ok.reason
    return dict(stopped=len(stopped), note=stopped[0]["note"])


def test_run_budget_2d() -> str:
    cfg, arm, weld, scene = _exp2(BudgetParams(run_seconds=0.0))
    srcs, sig = bootstrap_roadmap(cfg, arm, weld, scene, n_samples=600)
    seeds = [("++-", r.index) for r in srcs["++-"].regions() if r.reaches_start]
    res = run_proof(cfg, arm, weld, scene, None, sig, sources=srcs, seeds=seeds,
                    verbose=False)
    assert res.verdict == INCONCLUSIVE and res.reason.startswith(B.STOP), res.reason
    assert not B.active()
    return res.reason


def test_stopped_wall_3d() -> str:
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=R.ZGAP),
                              budget=BudgetParams(gamma_stop=0.5))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(R.PATCHES["exp2"])
    res = prove3(cfg, arm, surf, scene, (1, 1, -1), Page3Params(), verbose=False)
    assert res.verdict == INCONCLUSIVE and res.reason.startswith(B.STOP), res.reason
    assert res.walls and not res.walls[0]["ok"] and "budget_stop" in res.stats
    return res.reason


def test_escalation_2d() -> list:
    """Rung 0 is stopped (gamma_stop 0.5 < gamma_init); each next rung has
    twice the budget, three times the samples and a new seed. Every rung but
    the last is a budget stop, and the last proves EXP 2."""
    cfg, arm, weld, scene = _exp2(BudgetParams(gamma_stop=0.5, escalate_rungs=4))
    res, log = prove_line_escalating(cfg, arm, weld, scene, n_samples=500)
    assert res.verdict == INFEASIBLE and len(log) >= 2, log
    for a, b in zip(log, log[1:]):
        assert a["verdict"] == INCONCLUSIVE and B.STOP in a["reason"], a
        assert b["samples"] == 3 * a["samples"] and b["seed"] != a["seed"]
        assert b["gamma_stop"] == 2 * a["gamma_stop"]
    return log


def test_gallop_gamma_search() -> dict:
    """findings 45: gamma by galloping + bisection lands within gamma_step of
    IJRR's linear scan on data the scan separates, with far fewer SVM fits."""
    import sklearn.svm as skl
    rng = np.random.default_rng(3)
    X = rng.uniform([0.0, -1.0], [1.0, 1.0], size=(900, 2))
    f = X[:, 1] - 0.5 * np.sin(6.0 * X[:, 0])
    seed, rest = X[f > 0.06], X[f < -0.06]
    out = {}
    for mode in ("linear", "gallop"):
        cfg = Config()
        cfg = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall, gamma_search=mode))
        n = [0]
        orig = skl.SVC.fit

        def fit(self, *a, **k):
            n[0] += 1
            return orig(self, *a, **k)
        skl.SVC.fit = fit
        try:
            w = train_wall(cfg, seed, rest)
        finally:
            skl.SVC.fit = orig
        assert w.accuracy == 1.0 and not w.gamma_capped
        out[mode] = (w.gamma, n[0])
    (gl, nl), (gg, ng) = out["linear"], out["gallop"]
    assert abs(gl - gg) <= Config().wall.gamma_step + 1e-9, out
    assert ng < nl, out
    return dict(linear=out["linear"], gallop=out["gallop"])


def test_structural_not_retried() -> None:
    assert not B.retryable(INCONCLUSIVE, "bounded theta not justified: 3 valid")
    assert not B.retryable(INCONCLUSIVE, "live q2 door curve is INSIDE the wall")
    assert B.retryable(INCONCLUSIVE, "stopped by budget: ++-/1: ...")
    assert B.retryable(INCONCLUSIVE, "no verified wall (12/14 facets; ...)")
    assert not B.retryable(INFEASIBLE, "")


if __name__ == "__main__":
    r = test_gamma_stop()
    print(f"[budget] gamma stop    : inseparable data stopped at gamma {r['gamma']:.1f} "
          f"in {r['stop_s']:.2f} s")
    r = test_stopped_wall_is_failed_not_dropped_2d()
    print(f"[budget] 2-D wall stop : {r['stopped']} region(s) kept as FAILED walls "
          f"-> INCONCLUSIVE; unbudgeted same roadmap -> INFEASIBLE")
    print(f"[budget] 2-D run stop  : {test_run_budget_2d()}")
    print(f"[budget] 3-D wall stop : {test_stopped_wall_3d()}")
    for row in test_escalation_2d():
        print(f"[budget] escalation    : rung {row['rung']} {row['samples']} samples "
              f"seed {row['seed']} gamma_stop {row['gamma_stop']:g} -> {row['verdict']} "
              f"({row['seconds']} s)")
    r = test_gallop_gamma_search()
    print(f"[budget] gamma search  : linear gamma {r['linear'][0]:.2f} in {r['linear'][1]} fits, "
          f"gallop {r['gallop'][0]:.2f} in {r['gallop'][1]} fits")
    test_structural_not_retried()
    print("[budget] structural    : not retried")
    print("[budget] ALL TESTS PASSED")
