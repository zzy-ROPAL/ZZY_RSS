"""run_proof driven by a PRM instead of the Stage 1 grid.

The proof side of IJRR never builds a grid: it reads the planner's own
samples. This runs the SAME flood.run_proof -- same walls, same Algorithm
1-4, same certificate -- with flood.RoadmapPageSource in place of
GridPageSource, so the page/door layer is sampled too.

What still touches a grid: Sigma. The ANALYTIC door finders need only the
weld and a disc radius, but find_coord_discs / find_kin_discs read
pm.near_axis and pm.kin_measure. Both were measured to produce zero discs on
EXP 3 and EXP 4 (|u.z| peaks at 0.626 against a 0.999999 threshold;
det(J J^T) bottoms out at 2.3e-5 against 1e-7), so the page maps built here
are for Sigma alone and the PROOF does not read them.

    python3 -u scripts/run_proof_sampled.py --scene exp3 --samples 1500
"""
from __future__ import annotations
import argparse, dataclasses, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zzy_rss.params import Config, WeldParams, PageMapParams, SceneParams, BoxObstacle
from zzy_rss.arm import SrsArm
from zzy_rss.weld import WeldLine
from zzy_rss.scene import Scene
from zzy_rss.pagemap import build_all_pages_shared
from zzy_rss.doors import build_sigma, door_fiber_validity, door_anchor_vertices
from zzy_rss.prm import build_prm
from zzy_rss.flood import RoadmapPageSource, run_proof, write_certificate
from zzy_rss.wall import edge_anchor_samples
from zzy_rss.prm import _edge_valid
import run_three as R

SCENES = {
    "exp3": dict(tilt=15.0, x=0.32, z=0.40, half_y=0.26, boxes=None,
                 start_page="++-", seed_region=None),
    "exp4": dict(tilt=45.0, x=0.44, z=0.22, half_y=0.34,
                 boxes=(((0.18, 0.30, 0.34), (0.07,)*3, "cut"),
                        ((0.18, -0.06, 0.42), (0.05,)*3, "split")),
                 start_page="+-+", seed_region=None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="exp3", choices=tuple(SCENES))
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--res", type=int, default=61, help="for Sigma only")
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--n-train", dest="n_train", type=int, default=450)
    ap.add_argument("--feedback-margin", dest="fb", type=float, default=0.019)
    ap.add_argument("--door-routing", dest="routing", default="connectivity",
                    choices=("connectivity", "wall_side"))
    ap.add_argument("--no-door-seeds", dest="seeds", action="store_false",
                    default=True)
    ap.add_argument("--no-edge-anchors", dest="edge_anchors",
                    action="store_false", default=True,
                    help="drop the exact y=0 / y=1 training anchors "
                         "(findings 32); the control for that fix")
    ap.add_argument("--gamma-max", dest="gamma_max", type=float, default=400.0,
                    help="invariant 14: gamma is a cost/robustness signal, not "
                         "a modelling constraint -- closedness and the facet "
                         "checks are what police fragmentation. A random "
                         "roadmap puts the two classes ~2x closer than the "
                         "grid does, so it needs a higher cap.")
    ap.add_argument("--out", default="out/proof_sampled")
    args = ap.parse_args()
    sc = SCENES[args.scene]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    T, timings = time.perf_counter(), {}

    def lap(tag, t0):
        timings[tag] = round(time.perf_counter() - t0, 2)
        print(f"[time] {tag}: {timings[tag]} s", flush=True)
        return time.perf_counter()

    base = Config()
    obs = tuple(BoxObstacle(center=tuple(c), half_extent=tuple(h), label=l)
                for c, h, l in sc["boxes"]) if sc["boxes"] else R.OBS_OPEN
    cfg = dataclasses.replace(
        base, weld=R.weld_for(sc["tilt"], x=sc["x"], z=sc["z"],
                              half_y=sc["half_y"]),
        pagemap=PageMapParams(n_y=args.res, n_theta=args.res),
        scene=SceneParams(gap_open=True, obstacles=obs),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        planner=dataclasses.replace(base.planner, n_samples=args.samples,
                                    k_neighbors=args.k),
        flood=dataclasses.replace(base.flood, door_routing=args.routing),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz",
                                 rest_mode="goal_only", label_mode="metric",
                                 feedback_margin=args.fb,
                                 gamma_max=args.gamma_max))
    arm, weld, scene = SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)
    print(f"=== run_proof on a PRM -- scene {args.scene}, "
          f"{args.samples} samples/page, routing {args.routing} ===", flush=True)

    t = time.perf_counter()
    pms = {n: p for n, p in build_all_pages_shared(
        cfg, arm, weld, scene, with_kin=False, verbose=False).items()
        if n in R.PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blk = 0
    for n in R.PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sig.get(n, []):
            if d.vanishing:
                ok = door_fiber_validity(cfg, arm, weld, scene, d, sg)["ok"]
                n_open += ok; n_blk += (not ok)
    print(f"[sigma] doors {n_open} open / {n_blk} blocked "
          f"(page maps are for Sigma only)", flush=True)
    t = lap("sigma", t)

    prm = build_prm(cfg, arm, weld, scene, verbose=False,
                    sigma=sig if args.seeds else None)
    t = lap("prm", t)
    sources = {}
    for n in R.PAGES:
        ps = prm[n]
        ea = (edge_anchor_samples(cfg, arm, weld, scene, ps.signs)
              if args.edge_anchors else None)
        src = RoadmapPageSource(
            cfg, n, ps.signs, ps.valid, ps.component, ps.invalid,
            edge_anchors=ea,
            connects=(lambda a, b, sg=ps.signs:
                      _edge_valid(cfg, arm, weld, scene, a, b, sg)))
        sources[n] = src
        anc = len(door_anchor_vertices(cfg, sig, n)) if args.seeds else 0
        print(f"  page {n}: {len(ps.valid):4d} valid / {len(ps.invalid):4d} obs, "
              f"{len(src.regions())} component(s)"
              + (f", {anc} door-anchor seeds offered" if anc else "")
              + (f", {len(src.anchors)} edge anchors kept / "
                 f"{src.n_anchors_dropped} unconnected"
                 if len(src.anchors) or src.n_anchors_dropped else ""),
              flush=True)

    sp = sources[sc["start_page"]]
    cand = [r for r in sp.regions() if r.reaches_start and not r.reaches_goal]
    if not cand:
        print(f"\n[!] no start-touching component on {sc['start_page']} misses "
              f"the goal -- the roadmap already spans it; nothing to prove.")
        return
    k0 = sorted(cand, key=lambda r: -r.n_cells)[0].index
    print(f"\n[seed] ({sc['start_page']}, {k0}): "
          f"{dict((r.index, r.n_cells) for r in sp.regions())}", flush=True)

    res = run_proof(cfg, arm, weld, scene, sources, sig, verbose=True,
                    alg4_iters=args.iters, seeds=[(sc["start_page"], k0)],
                    n_train_max=args.n_train, n_invalid_max=150)
    t = lap("run_proof", t)
    print(f"\n[P] proven={res.proven} -- {res.reason}", flush=True)
    print(f"[P] goal_reached_via={res.goal_reached_via!r}", flush=True)
    print(f"[P] regions visited {len(res.regions_visited)}, walls "
          f"{len(res.certificate)}, doors followed {len(res.doors_followed)}",
          flush=True)
    for c in res.certificate:
        print(f"[P]   wall {c['page']}/{c['region']}: "
              f"{c['n_passed']}/{c['n_facets']} facets, closed={c['closed']}, "
              f"gamma={c['gamma']:.1f}, ok={c['ok']}", flush=True)
    for d in res.doors_followed:
        print(f"[P]   door {d['from']} -> {d['to']} at y={d['y']:.4f}", flush=True)
    lap("total", T)
    write_certificate(res, sig, out / f"{args.scene}_sampled_certificate.json")
    json.dump({"scene": args.scene, "proven": bool(res.proven),
               "reason": res.reason, "goal_reached_via": res.goal_reached_via,
               "certificate": res.certificate,
               "doors_followed": res.doors_followed,
               "timings_seconds": timings,
               "config": {"samples": args.samples, "routing": args.routing,
                          "door_seeds": args.seeds}},
              open(out / f"{args.scene}_sampled_report.json", "w"), indent=2)
    print(f"\n[out] {out}/{args.scene}_sampled_report.json", flush=True)


if __name__ == "__main__":
    main()
