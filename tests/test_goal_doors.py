"""The goal and the doors as GEOMETRIC objects of the wall (findings 47).

What is pinned:

  * The goal check is exact: a d = 2 facet is a segment, so the wall's
    largest y is attained at a vertex; the feedback for a wall that reaches
    the goal band is its vertices projected ONTO the band, in the rest class.
  * A verified 2-D wall keeps the WHOLE goal band outside (y_max below
    goal_y - goal_y_tol - goal_margin), and the rest class carries goal-line
    samples whether or not they are valid.
  * Leaving the goal-reaching part of the start region unsampled never
    produces INFEASIBLE.
  * A door next to the start that an obstacle seals off is EXCLUDED, not
    followed: on EXP 2 with every door kept live, the old proximity rule put
    both start-page doors in the seed class and no wall could be built
    (INCONCLUSIVE); "exclude_first" proves INFEASIBLE with both outside.
  * A door curve that leaves the patch and comes back is two pieces, and
    each is probed on its own.
  * (--slow) The 3-D two-wall scene: a followed door goes WHOLE into the
    partner page's seed class, so the door pocket there is inside the same
    wall and the own-page pocket is never visited -- 2 walls, not 4.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_witness import _scene, _zgap  # noqa: E402
from zzy_rss.flood import (INFEASIBLE, goal_feedback_points, goal_line_samples,  # noqa: E402
                           run_proof, wall_y_max)
from zzy_rss.params import BudgetParams  # noqa: E402
from zzy_rss.source import bootstrap_roadmap  # noqa: E402
from zzy_rss.wall import Polytope  # noqa: E402


def test_goal_check_is_exact_at_vertices() -> dict:
    loop = np.array([[0.2, -0.5], [0.7, -0.5], [1.04, 0.0], [0.7, 0.5],
                     [0.2, 0.5], [0.2, -0.5]])
    poly = Polytope(loops=[loop], closed=True)
    # y is linear on every segment, so the max over the wall is at a vertex.
    t = np.linspace(0.0, 1.0, 101)[:, None]
    dense = np.vstack([a + t * (b - a) for a, b in zip(loop[:-1], loop[1:])])
    assert abs(wall_y_max(poly) - dense[:, 0].max()) < 1e-12
    feed = goal_feedback_points(poly, 0.98, 0.98)
    # only the vertex past the limit, moved onto the band at its own theta
    assert feed.shape == (1, 2) and np.allclose(feed, [[0.98, 0.0]]), feed
    return dict(y_max=wall_y_max(poly), feed=feed.tolist())


def _exp2(**flood):
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, _zgap())
    cfg = dataclasses.replace(cfg, budget=BudgetParams(wall_seconds=120.0, run_seconds=600.0),
                              flood=dataclasses.replace(cfg.flood, **flood))
    return cfg, arm, weld, scene


def test_verified_wall_keeps_the_goal_band_outside() -> dict:
    cfg, arm, weld, scene = _exp2()
    g = goal_line_samples(cfg)
    assert len(g) == cfg.flood.goal_line_samples
    assert np.allclose(g[:, 0], cfg.flood.goal_y - cfg.flood.goal_y_tol)
    srcs, sig = bootstrap_roadmap(cfg, arm, weld, scene, n_samples=1500)
    seeds = [("++-", r.index) for r in srcs["++-"].regions() if r.reaches_start]
    res = run_proof(cfg, arm, weld, scene, None, sig, sources=srcs, seeds=seeds,
                    verbose=False)
    assert res.verdict == INFEASIBLE, res.reason
    limit = cfg.flood.goal_y - cfg.flood.goal_y_tol - cfg.flood.goal_margin
    for w in res.certificate:
        assert w["ok"] and w["y_max"] is not None and w["y_max"] < limit, w
    return dict(walls=[(w["region"], f"{w['n_passed']}/{w['n_facets']}",
                        round(w["y_max"], 4)) for w in res.certificate])


def test_unsampled_goal_tail_is_never_infeasible() -> str:
    """EXP 1 (feasible) on its single-region page, with every roadmap vertex
    past y = 0.9 deleted: by its labels the start region no longer reaches
    the goal. Whatever happens, the answer must not be INFEASIBLE -- the
    robot really can reach the goal."""
    cfg, arm, weld, scene = _scene(0.0, 0.28, 0.46, 0.26, ())
    cfg = dataclasses.replace(cfg, budget=BudgetParams(wall_seconds=60.0, run_seconds=300.0))
    srcs, _sig = bootstrap_roadmap(cfg, arm, weld, scene, n_samples=1500)
    src = srcs["+--"]
    src.valid = src.valid[src.valid[:, 0] <= 0.9]
    src._tree = None
    src._tol = None
    src.comp = src._components()
    src._regions = src._build_regions()
    seeds = [("+--", r.index) for r in src.regions() if r.reaches_start]
    assert seeds and not any(r.reaches_goal for r in src.regions())
    res = run_proof(cfg, arm, weld, scene, None, {}, sources={"+--": src}, seeds=seeds,
                    verbose=False)
    assert res.verdict != INFEASIBLE, res.certificate
    return f"{res.verdict} ({res.reason[:70]})"


def test_sealed_doors_are_excluded_not_followed() -> dict:
    cfg, arm, weld, scene = _exp2(door_policy="exclude_first")
    cfg = dataclasses.replace(
        cfg, crossing=dataclasses.replace(cfg.crossing, exact=False),
        sigma_policy=dataclasses.replace(cfg.sigma_policy, enable_fiber_certification=False))
    srcs, sig = bootstrap_roadmap(cfg, arm, weld, scene, n_samples=1500)
    live = [d for d in sig["++-"] if d.vanishing]
    assert len(live) == 2, len(live)
    seeds = [("++-", r.index) for r in srcs["++-"].regions() if r.reaches_start]
    res = run_proof(cfg, arm, weld, scene, None, sig, sources=srcs, seeds=seeds,
                    verbose=False)
    assert res.verdict == INFEASIBLE, res.reason
    (w,) = res.certificate
    assert not w["seed_doors"] and len(w["excluded_doors"]) == 2, w
    start_page = [d for d in res.door_decisions if d["page"] == "++-"]
    assert start_page and all(not d["followed"] and d["exit"] == "1-wall1"
                              for d in start_page), start_page
    return dict(facets=f"{w['n_passed']}/{w['n_facets']}", excluded=len(w["excluded_doors"]))


def test_each_door_piece_is_probed() -> list:
    from types import SimpleNamespace

    from zzy_rss.facets import SigmaTube
    from zzy_rss.page3 import door_pieces
    # A closed curve that crosses the patch boundary twice (out at u > 1 and
    # at u < 0): two on-patch pieces.
    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    pts = np.column_stack([0.5 + 0.6 * np.cos(t), 0.5 + 0.3 * np.sin(t), 0.2 * np.sin(t)])
    curve = SimpleNamespace(joint=6, sign_index=2, tube=SigmaTube(
        polyline=np.vstack([pts, pts[:1]]), radii=np.array([0.02, 0.02, 0.06]), kind="q6"))
    page = SimpleNamespace(lo=np.array([0.0, 0.0, -np.pi]), hi=np.array([1.0, 1.0, np.pi]))
    pcs = door_pieces(page, [(3, curve)])
    assert len(pcs) == 2 and len({pc.key for pc in pcs}) == 2, [pc.key for pc in pcs]
    for pc in pcs:
        p = pc.probe()
        assert np.min(np.linalg.norm(pc.polyline - p, axis=1)) < 1e-12
        assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
    return [pc.key for pc in pcs]


def test_two_wall_3d() -> dict:
    """The 3-D two-wall scene of findings 46 (slow: ~75 s)."""
    from search_scene3 import surface

    from zzy_rss.arm import SrsArm
    from zzy_rss.flood3 import run_flood3
    from zzy_rss.params import BoxObstacle, Config, Page3Params, PageMapParams, SceneParams
    from zzy_rss.scene import Scene
    from zzy_rss.weld import WeldSurface
    boxes = ((0.2, 0.0, 0.34, 0.06, 0.06, 0.06), (0.3, 0.2, 0.3, 0.06, 0.06, 0.06))
    obs = tuple(BoxObstacle(center=b[:3], half_extent=b[3:], label=f"box{i}")
                for i, b in enumerate(boxes))
    cfg = dataclasses.replace(Config(), pagemap=PageMapParams(n_y=61, n_theta=61),
                              scene=SceneParams(gap_open=True, obstacles=obs))
    cfg = dataclasses.replace(cfg, budget=dataclasses.replace(cfg.budget, wall_seconds=1200.0),
                              wall=dataclasses.replace(cfg.wall, gamma_search="gallop"))
    arm, scene = SrsArm(cfg.arm), Scene(cfg.arm, cfg.scene)
    surf = WeldSurface(surface(0.40, 0.21, 45.0, 305.0))
    p3 = dataclasses.replace(Page3Params(), alg4_iters=8)
    res = run_flood3(cfg, arm, surf, scene, (1, 1, 1), p3, verbose=False)
    assert res.verdict == INFEASIBLE, res.reason
    assert len(res.certificate) == 2, res.certificate
    assert all(w["ok"] and w["u_max"] < 1.0 - p3.goal_tol for w in res.certificate)
    skipped = [r for r in res.regions if "skipped" in r]
    assert skipped, res.regions
    return dict(walls=[(w["page"], w["facets"]) for w in res.certificate],
                skipped=[(r["page"], r["comp"]) for r in skipped],
                seconds=res.stats.get("seconds"))


if __name__ == "__main__":
    r = test_goal_check_is_exact_at_vertices()
    print(f"[goal] exact check   : y_max {r['y_max']:.3f} at a vertex; feedback {r['feed']}")
    r = test_verified_wall_keeps_the_goal_band_outside()
    print(f"[goal] EXP 2 wall    : {r['walls']} (region, facets, y_max) -- band outside")
    print(f"[goal] unsampled tail: {test_unsampled_goal_tail_is_never_infeasible()}")
    r = test_sealed_doors_are_excluded_not_followed()
    print(f"[doors] sealed doors : INFEASIBLE, {r['facets']} facets, {r['excluded']} doors "
          f"excluded (outside the wall), none followed")
    print(f"[doors] door pieces  : {test_each_door_piece_is_probed()} probed separately")
    if "--slow" in sys.argv:
        r = test_two_wall_3d()
        print(f"[doors] 3-D two walls: {r['walls']}, skipped {r['skipped']} "
              f"({r['seconds']} s)")
    print("[goal-doors] ALL TESTS PASSED")
