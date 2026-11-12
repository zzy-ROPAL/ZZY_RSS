"""The recursive proof (docs/recursive_proof_design.md, steps 2b and 3).

What is pinned:

  * THE RECURSION'S LOGIC, on a scripted check(): one wall with no door is a
    leaf; doors inside are cut with AND across doors and OR within a door;
    a door already on the branch is a cycle and is NEVER assumed proven;
    an uncut door leaves the whole node NOT PROVEN; only PROVEN results are
    cached.
  * DOORS WITHOUT A DOOR LIST: the generic refinement (doorset.py) puts a
    curve on every closed-form door point of two patches, and every point it
    returns has sigma ~ 0.
  * TEST (ii) EXEMPTS ONLY DISCHARGED DOORS: a facet ball on a door fails
    while the door is live and passes once it is discharged; a ball far from
    every door passes either way.
  * END TO END, no door list: 3-D EXP 2 without obstacles is FEASIBLE (a
    verified path) and with ZGAP is INFEASIBLE (a one-leaf proof tree whose
    wall passes every facet), as today's pipeline says (findings 42).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import zzy_rss.proof as P  # noqa: E402
from run_3d import PATCHES  # noqa: E402
from search_scene3 import surface  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.chart import SrsChart  # noqa: E402
from zzy_rss.check import (CONNECTED, FAILED, WALLED, CheckResult, DoorNode,  # noqa: E402
                           GoalSet, PageWall, StartSet, World)
from zzy_rss.doors3 import door_curves  # noqa: E402
from zzy_rss.doorset import find_doorset  # noqa: E402
from zzy_rss.params import Config, Page3Params  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldSurface  # noqa: E402

CFG = Config()
ARM = SrsArm(CFG.arm)
P3 = Page3Params()
S0 = StartSet((1, 1, 1))
G = GoalSet()


def _door(k):
    return DoorNode(door=k, pages=frozenset({(1, 1, 1), (1, 1, -1)}))


def _run(script: dict, A=S0, B=G):
    """inf_proof on a scripted check: script[(A, B)] = (kind, [door ids])."""
    calls = []

    def fake_check(world, a, b, verbose=True):
        calls.append((a, b))
        kind, doors = script.get((a, b), (FAILED, []))
        walls = [PageWall((1, 1, 1), kind == WALLED, info=dict(facets="1/1"))] \
            if kind == WALLED else []
        return CheckResult(kind, "scripted", walls=walls, doors=[_door(k) for k in doors])

    real = P.check
    P.check = fake_check
    try:
        pr = P.Prover(world=None, max_checks=50, verbose=False)
        node = pr.inf_proof(A, B)
    finally:
        P.check = real
    return node, calls, pr


def test_recursion_logic() -> dict:
    d1, d2 = _door(1), _door(2)
    # A leaf.
    n, _, _ = _run({(S0, G): (WALLED, [])})
    assert n.status == P.PROVEN and not n.children
    # One door, cut on the start side.
    n, calls, _ = _run({(S0, G): (WALLED, [1]), (S0, d1): (WALLED, [])})
    assert n.status == P.PROVEN and n.children[0]["cut"].startswith("start")
    assert (d1, G) not in calls, "the door->goal half must not be tried once start->door is cut"
    # One door, start reaches it, cut on the goal side.
    n, _, _ = _run({(S0, G): (WALLED, [1]), (S0, d1): (CONNECTED, []), (d1, G): (WALLED, [])})
    assert n.status == P.PROVEN and n.children[0]["cut"].startswith("door")
    # AND across doors: the second door uncut -> not proven.
    n, _, _ = _run({(S0, G): (WALLED, [1, 2]), (S0, d1): (WALLED, []),
                    (S0, d2): (CONNECTED, []), (d2, G): (FAILED, [])})
    assert n.status == P.NOT_PROVEN and "D2" in n.reason
    # Cycle: cutting d1 -> goal finds d2 inside, and d2's halves need d1 again.
    n, _, _ = _run({(S0, G): (WALLED, [1]), (S0, d1): (CONNECTED, []),
                    (d1, G): (WALLED, [2]), (d1, d2): (CONNECTED, []),
                    (d2, G): (WALLED, [1])})
    assert n.status == P.NOT_PROVEN, "a cycle must never be read as proven"
    # A wall that failed for another reason is not a split point.
    n, _, _ = _run({(S0, G): (FAILED, [1])})
    assert n.status == P.NOT_PROVEN and not n.children
    # Cache: PROVEN results only.
    _, _, pr = _run({(S0, G): (WALLED, [1]), (S0, d1): (WALLED, [])})
    assert all(v.status == P.PROVEN for v in pr.cache.values()) and (S0, d1) in pr.cache
    return dict(cases=6)


def test_doors_without_a_door_list() -> dict:
    out = {}
    for name, sp in (("exp2", PATCHES["exp2"]), ("exp3-new", surface(0.40, 0.21, 45, 305))):
        surf = WeldSurface(sp)
        ch = SrsChart.for_surface(ARM, surf, P3)
        ds = find_doorset(ch, P3.tube_radius)
        assert ds.complete
        pts = np.vstack([d.points for d in ds.doors if d.supported])
        sig = ch.sigma(pts)
        assert np.nanmin(np.minimum(sig[:, 0], sig[:, 1])) >= 0
        assert float(np.max(np.minimum(sig[:, 0], sig[:, 1]))) < 1e-6
        worst = 0.0
        for c in door_curves(CFG, ARM, surf, P3):
            on = c.pts[np.all((c.pts[:, :2] >= 0) & (c.pts[:, :2] <= 1), axis=1)]
            d = np.min(np.linalg.norm((on[:, None, :] - pts[None]) * ch.scale, axis=2), axis=1)
            worst = max(worst, float(d.max()))
        assert worst < 0.03, f"{name}: a closed-form door point is {worst:.3f} from every curve"
        out[name] = (len(ds.doors), [len(d.curves) for d in ds.doors], round(worst, 4))
    return out


def test_sigma_test_exempts_only_discharged_doors() -> dict:
    surf = WeldSurface(PATCHES["exp2"])
    w = World(CFG, ARM, surf, Scene(CFG.arm, CFG.scene), (1, 1, -1), P3, start_theta=0.0,
              verbose=False)
    door = w.doorset.doors[0]
    p = door.curves[0].pts[len(door.curves[0].pts) // 2]
    rng = np.random.default_rng(4)
    for _ in range(2000):
        far = rng.uniform(w.chart.lo + 0.1, w.chart.hi - 0.1)
        if w.chart.ball_test(far[None], np.array([0.05])).clear[0]:
            break
    signs = (1, 1, -1)
    for k in range(len(w.doorset.doors)):
        w._discharge[(signs, k)] = (False, "test: live")
    live = w.sigma_clear(signs)(np.array([p, far]), np.array([0.01, 0.05]))
    for k in range(len(w.doorset.doors)):
        w._discharge[(signs, k)] = (True, "test: discharged")
    w._live_index.clear()
    disch = w.sigma_clear(signs)(np.array([p, far]), np.array([0.01, 0.05]))
    assert not live[0] and live[1], live
    assert disch[0] and disch[1], disch
    return dict(on_door_live=bool(live[0]), on_door_discharged=bool(disch[0]))


def test_end_to_end() -> dict:
    import dataclasses
    from run_3d import ZGAP
    from zzy_rss.params import SceneParams
    out = {}
    for name, obs, want in (("none", (), P.FEASIBLE), ("zgap", ZGAP, P.INFEASIBLE)):
        cfg = dataclasses.replace(CFG, scene=SceneParams(gap_open=True, obstacles=obs))
        cfg = dataclasses.replace(cfg, wall=dataclasses.replace(cfg.wall, gamma_search="gallop"))
        res = P.prove_task(cfg, ARM, WeldSurface(PATCHES["exp2"]), Scene(cfg.arm, cfg.scene),
                           (1, 1, -1), dataclasses.replace(P3, alg4_iters=8), verbose=False)
        assert res.verdict == want, (name, res.verdict, res.reason)
        if want == P.INFEASIBLE:
            assert res.tree.status == P.PROVEN and all(
                w["facets"].split("/")[0] == w["facets"].split("/")[1] for w in res.tree.walls)
            out[name] = (res.verdict, res.tree.walls[0]["facets"], res.stats["seconds"])
        else:
            out[name] = (res.verdict, res.reason, res.stats["seconds"])
    return out


if __name__ == "__main__":
    r = test_recursion_logic()
    print(f"[recursive] logic: {r['cases']} scripted cases -- leaf, start-side cut, door-side "
          f"cut, AND across doors, cycle never proven, failure is not a split; cache holds "
          f"PROVEN only")
    for k, v in test_doors_without_a_door_list().items():
        print(f"[recursive] doors from sigma {k:8s}: {v[0]} region(s), curves {v[1]}, every "
              f"closed-form door point within {v[2]} of a curve")
    r = test_sigma_test_exempts_only_discharged_doors()
    print(f"[recursive] test (ii): ball on a door passes live={r['on_door_live']}, "
          f"discharged={r['on_door_discharged']}; far ball passes both")
    for k, v in test_end_to_end().items():
        print(f"[recursive] end to end exp2/{k}: {v[0]} -- {v[1]} ({v[2]} s)")
    print("[recursive] ALL TESTS PASSED")
