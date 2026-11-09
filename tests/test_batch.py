"""Batched validity is the scalar validity, exactly (findings 44).

batch.py classifies N states in one numpy pass. It may only be used in a
proof if it answers what the scalar path answers, state for state -- a
single disagreement could turn a facet check or a roadmap edge. Pinned here:

  * classify_line_batch / classify_surface_batch give the scalar TAG (no-IK /
    limit / collision / valid) on every one of thousands of random states,
    on lines and on a patch, with and without obstacles, including states
    outside the page box;
  * classify_all_pages_line_batch gives classify_all_pages' eight verdicts;
  * pd_march_batch returns pd_march's value (so the same pd >= r verdicts);
  * check_many returns check_simplex's verdict and failure reason facet by
    facet;
  * Page3 and RoadmapSource built batched and scalar are the same roadmap.
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
from test_witness import _scene, _zgap, _zgap_y  # noqa: E402
from zzy_rss.arm import SIGN_BRANCHES, SrsArm  # noqa: E402
from zzy_rss.batch import (classify_all_pages_line_batch, classify_line_batch,  # noqa: E402
                           classify_surface_batch, pd_march_batch)
from zzy_rss.facets import check_many, check_simplex  # noqa: E402
from zzy_rss.page3 import Page3, classify3, robot_oracle3  # noqa: E402
from zzy_rss.params import Config, Page3Params, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.prm import classify  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.source import RoadmapSource, classify_all_pages  # noqa: E402
from zzy_rss.wall import pd_march  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

SIGNS = (1, 1, -1)


def _lines():
    yield "EXP 2 line", _scene(0.0, 0.28, 0.46, 0.26, _zgap())
    yield "EXP 3 line", _scene(15.0, 0.32, 0.40, 0.26, _zgap_y())
    yield "free line", _scene(30.0, 0.28, 0.46, 0.26, ())


def _patch(batch=True):
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=R.ZGAP))
    cfg = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner,
                                                               batch_classify=batch))
    return cfg, SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene), WeldSurface(R.PATCHES["exp2"])


def test_line_tags_equal() -> dict:
    rng = np.random.default_rng(1)
    out = {}
    for name, (cfg, arm, weld, scene) in _lines():
        P = np.column_stack([rng.uniform(-0.1, 1.1, 3000), rng.uniform(-np.pi, np.pi, 3000)])
        for sg in SIGN_BRANCHES:
            t0 = time.perf_counter()
            tag_s = np.array([classify(cfg, arm, weld, scene, float(y), float(t), sg)[1]
                              for y, t in P])
            ts = time.perf_counter() - t0
            t0 = time.perf_counter()
            _ok, tag_b, _q = classify_line_batch(cfg, arm, weld, scene, P, sg)
            tb = time.perf_counter() - t0
            bad = int((tag_s != tag_b).sum())
            assert bad == 0, f"{name} {sg}: {bad} tag disagreements"
        out[name] = dict(n=len(P) * 8, speedup=ts / tb, kinds=len(set(tag_s.tolist())))
    return out


def test_surface_tags_equal() -> dict:
    cfg, arm, scene, surf = _patch()
    rng = np.random.default_rng(2)
    P = rng.uniform([-0.1, -0.1, -np.pi], [1.1, 1.1, np.pi], size=(4000, 3))
    n_valid = 0
    for sg in SIGN_BRANCHES:
        ok_s = np.array([classify3(cfg, arm, surf, scene, x, sg) for x in P])
        ok_b = classify_surface_batch(cfg, arm, surf, scene, P, sg)[0]
        bad = int((ok_s != ok_b).sum())
        assert bad == 0, f"patch {sg}: {bad} disagreements"
        n_valid += int(ok_s.sum())
    assert 0 < n_valid < 8 * len(P)
    return dict(n=8 * len(P), valid=n_valid)


def test_all_pages_equal() -> dict:
    rng = np.random.default_rng(3)
    n = 0
    for name, (cfg, arm, weld, scene) in _lines():
        P = np.column_stack([rng.uniform(0, 1, 1500), rng.uniform(-np.pi, np.pi, 1500)])
        vb = classify_all_pages_line_batch(cfg, arm, weld, scene, P)
        for i, (y, t) in enumerate(P):
            vs = classify_all_pages(cfg, arm, weld, scene, float(y), float(t))
            for sg in SIGN_BRANCHES:
                assert bool(vs[sg][0]) == bool(vb[sg][i]), (name, sg, y, t)
        n += 8 * len(P)
    return dict(n=n)


def test_pd_march_equal() -> dict:
    cfg, arm, scene, surf = _patch()
    page = Page3(cfg, arm, surf, scene, SIGNS, Page3Params(), None, n_samples=1500)
    rng = np.random.default_rng(4)
    P = rng.uniform(page.lo, page.hi, size=(600, 3))
    P = P[~page.valid_many(P)][:150]
    caps = rng.choice([0.02, 0.05, 0.2, np.inf], size=len(P))
    kw = dict(step=float(cfg.planner.edge_resolution), attempts=int(cfg.wall.pd_seed_attempts),
              bisect_iters=int(cfg.wall.pd_bisect_iters))
    pb = pd_march_batch(P, page.valid, page.valid_many, page.scale, caps=caps, **kw)
    ps = np.array([pd_march(x, page.valid, page.valid_fn, page.scale, cap=c, **kw)[0]
                   for x, c in zip(P, caps)])
    err = float(np.max(np.abs(pb - ps)))
    assert err <= 1e-12, f"pd differs by {err}"
    return dict(n=len(P), max_err=err)


def test_check_many_equal() -> dict:
    """Random small triangles across the obstacle band: a mix of passes at
    depth 0, passes after subdivision, VALID centres and depth limits."""
    cfg, arm, scene, surf = _patch()
    page = Page3(cfg, arm, surf, scene, SIGNS, Page3Params(), None, n_samples=1500)
    oracle = robot_oracle3(cfg, page, [])
    rng = np.random.default_rng(5)
    base = rng.uniform(page.lo, page.hi, size=(300, 3))
    tris = [b + rng.normal(scale=[0.02, 0.02, 0.05], size=(3, 3)) for b in base]
    depth = int(cfg.wall.facet_max_depth_tri)
    t0 = time.perf_counter()
    rs = [check_simplex(oracle, v, depth) for v in tris]
    ts = time.perf_counter() - t0
    t0 = time.perf_counter()
    rb = check_many(oracle, tris, depth)
    tb = time.perf_counter() - t0
    kinds = {}
    for a, b in zip(rs, rb):
        assert a.ok == b.ok, (a, b)
        if not a.ok:
            assert a.reason == b.reason, (a.reason, b.reason)
        key = "pass" if a.ok else a.reason.split(";")[0]
        kinds[key] = kinds.get(key, 0) + 1
    assert kinds.get("pass") and len(kinds) > 1, kinds
    return dict(n=len(tris), kinds=kinds, speedup=ts / tb)


def test_roadmaps_equal() -> dict:
    cfg, arm, scene, surf = _patch(batch=True)
    cfg_s = dataclasses.replace(cfg, planner=dataclasses.replace(cfg.planner,
                                                                 batch_classify=False))
    p3 = Page3Params()
    t0 = time.perf_counter()
    pb = Page3(cfg, arm, surf, scene, SIGNS, p3, None, n_samples=800)
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps = Page3(cfg_s, arm, surf, scene, SIGNS, p3, None, n_samples=800)
    ts = time.perf_counter() - t0
    assert np.array_equal(pb.valid, ps.valid) and np.array_equal(pb.comp, ps.comp)
    cfg2, arm2, weld2, scene2 = _scene(0.0, 0.28, 0.46, 0.26, _zgap())
    cfg2s = dataclasses.replace(cfg2, planner=dataclasses.replace(cfg2.planner,
                                                                  batch_classify=False))
    rb = RoadmapSource(cfg2, arm2, weld2, scene2, SIGNS, "++-", n_samples=600,
                       rng=np.random.default_rng(7))
    rs = RoadmapSource(cfg2s, arm2, weld2, scene2, SIGNS, "++-", n_samples=600,
                       rng=np.random.default_rng(7))
    assert np.array_equal(rb.valid, rs.valid) and np.array_equal(rb.comp, rs.comp)
    return dict(page3_speedup=ts / tb, page3_vertices=len(pb.valid),
                line_vertices=len(rb.valid))


if __name__ == "__main__":
    for k, v in test_line_tags_equal().items():
        print(f"[batch] {k:11s}: {v['n']} tags equal ({v['kinds']} kinds), "
              f"{v['speedup']:.0f}x")
    r = test_surface_tags_equal()
    print(f"[batch] patch      : {r['n']} verdicts equal ({r['valid']} valid)")
    r = test_all_pages_equal()
    print(f"[batch] eight pages: {r['n']} verdicts equal")
    r = test_pd_march_equal()
    print(f"[batch] pd march   : {r['n']} depths, max |batch - scalar| = {r['max_err']:.1e}")
    r = test_check_many_equal()
    print(f"[batch] check_many : {r['n']} triangles, same verdict and reason "
          f"{r['kinds']}, {r['speedup']:.1f}x")
    r = test_roadmaps_equal()
    print(f"[batch] roadmaps   : Page3 ({r['page3_vertices']} vertices) and "
          f"RoadmapSource ({r['line_vertices']}) identical, Page3 {r['page3_speedup']:.1f}x")
    print("[batch] ALL TESTS PASSED")
