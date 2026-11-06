"""EXP 4 -- FEASIBLE ONLY AFTER CHANGING PAGES.

The dual of EXP 3. The start posture's page cannot reach the goal; the robot
crosses a q6 door to a partner page where it can. Expected verdict is
therefore proven=False with goal_reached_via on the PARTNER page.

Scene is the one already found and verified in docs/exp4_brief.md (findings
36) -- do not re-search it. Straight weld, so no new kinematics are needed.

    python3 -u scripts/run_exp4.py --door-routing wall_side 2>&1 | tee exp4.log
"""
from __future__ import annotations
import argparse, dataclasses, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zzy_rss.params import (Config, WeldParams, PageMapParams, SceneParams,
                            BoxObstacle)
from zzy_rss.arm import SrsArm
from zzy_rss.weld import WeldLine
from zzy_rss.scene import Scene
from zzy_rss.pagemap import build_all_pages_shared
from zzy_rss.doors import build_sigma, door_fiber_validity
from zzy_rss.flood import label_regions, run_proof, write_certificate
import run_three as R

A = np.radians(45.0)
WELD = WeldParams(p0=(0.40, -0.34, 0.22), p1=(0.40, 0.34, 0.22),
                  approach=(float(np.sin(A)), 0.0, -float(np.cos(A))),
                  reference=(float(np.cos(A)), 0.0, float(np.sin(A))))
OBS = (BoxObstacle(center=(0.095, 0.29, 0.377),
                   half_extent=(0.07, 0.07, 0.07), label="exp4"),)
START_PAGE, START_REGION, PARTNER_PAGE = "-++", 1, "-+-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--n-train", dest="n_train", type=int, default=450)
    ap.add_argument("--door-routing", dest="door_routing",
                    default="wall_side", choices=("connectivity", "wall_side"))
    ap.add_argument("--feedback-margin", dest="feedback_margin", type=float,
                    default=0.0,
                    help="Algorithm 4 / Algorithm 2 feed-back guard (findings "
                         "32). The grid pipeline keeps IJRR's 0; the sampled "
                         "one needs 0.019. A fed-back point closer than this "
                         "to the class it was NOT given is dropped instead of "
                         "being allowed to make the training data "
                         "inseparable.")
    ap.add_argument("--out", default="out/exp4")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    timings = {}

    def lap(tag, t0):
        timings[tag] = round(time.perf_counter() - t0, 2)
        print(f"[time] {tag}: {timings[tag]} s", flush=True)
        return time.perf_counter()

    base = Config()
    cfg = dataclasses.replace(
        base, weld=WELD,
        pagemap=PageMapParams(n_y=args.res, n_theta=args.res),
        scene=SceneParams(gap_open=True, obstacles=OBS),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
        flood=dataclasses.replace(base.flood, door_routing=args.door_routing),
        wall=dataclasses.replace(base.wall, pd_mode="lipschitz",
                                 label_mode="metric", rest_mode="goal_only",
                                 feedback_margin=args.feedback_margin))
    arm, weld, scene = SrsArm(cfg.arm), WeldLine(cfg.weld), Scene(cfg.arm, cfg.scene)
    print(f"=== EXP 4 : page {START_PAGE} is sealed, the q6 door leads to "
          f"{PARTNER_PAGE} where the goal IS reachable ===", flush=True)
    print(f"[cfg] res={args.res} iters={args.iters} n_train={args.n_train} "
          f"door_routing={args.door_routing} "
          f"feedback_margin={args.feedback_margin}", flush=True)

    t_all = t = time.perf_counter()
    built = build_all_pages_shared(cfg, arm, weld, scene, with_kin=False,
                                   verbose=False)
    pms = {n: built[n] for n in R.PAGES}
    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blk = 0
    for n in R.PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        for d in sig.get(n, []):
            if d.vanishing:
                ok = door_fiber_validity(cfg, arm, weld, scene, d, sg)["ok"]
                d.__dict__["_state"] = "open" if ok else "blocked"
                n_open += ok; n_blk += (not ok)
    print(f"[4] doors {n_open} open / {n_blk} blocked", flush=True)
    t = lap("exp4_pages_and_doors", t)

    R.theta_seam_check(pms, " exp4")
    for n in (START_PAGE, PARTNER_PAGE):
        _, regs = label_regions(cfg, pms[n])
        for r in regs:
            print(f"[4] {n}/{r.index}: {r.n_cells} cells, "
                  f"y in [{r.y_min:.3f}, {r.y_max:.3f}], "
                  f"start={r.reaches_start} goal={r.reaches_goal}", flush=True)

    print(f"\n[4] --- run_proof, seeds=[({START_PAGE}, {START_REGION})] ---",
          flush=True)
    res = run_proof(cfg, arm, weld, scene, pms, sig, verbose=True,
                    alg4_iters=args.iters, seeds=[(START_PAGE, START_REGION)],
                    n_train_max=args.n_train, n_invalid_max=150)
    t = lap("exp4_run_proof", t)
    print(f"\n[4] proven={res.proven} -- {res.reason}", flush=True)
    print(f"[4] goal_reached_via = {res.goal_reached_via}", flush=True)
    print(f"[4] regions visited {len(res.regions_visited)}, "
          f"walls {len(res.certificate)}, doors followed "
          f"{len(res.doors_followed)}", flush=True)
    for c in res.certificate:
        print(f"[4]   wall {c['page']}/{c['region']}: "
              f"{c['n_passed']}/{c['n_facets']} facets, closed={c['closed']}, "
              f"gamma={c['gamma']:.1f}, ok={c['ok']}", flush=True)
    for d in res.doors_followed:
        print(f"[4]   door {d['from']} -> {d['to']} at y={d['y']:.4f}, "
              f"th={d['theta']:+.4f}", flush=True)
    write_certificate(res, sig, out / "exp4_certificate.json")

    R.START_PAGE, R.PARTNER_PAGE = START_PAGE, PARTNER_PAGE
    R.fig8(pms, sig, {}, "EXP 4 -- the start page is sealed; the q6 door "
           "leads to a page that REACHES the goal",
           "proven=False is the CORRECT answer here: the task is feasible, "
           "but only by changing pages.", out / "exp4_pages.png")
    t = lap("exp4_figures", t)
    lap("exp4_total", t_all)
    json.dump({"proven": bool(res.proven), "reason": res.reason,
               "goal_reached_via": res.goal_reached_via,
               "doors_open": n_open, "doors_blocked": n_blk,
               "certificate": res.certificate,
               "doors_followed": res.doors_followed,
               "timings_seconds": timings,
               "config": {"res": args.res, "iters": args.iters,
                          "n_train": args.n_train,
                          "door_routing": args.door_routing}},
              open(out / "exp4_report.json", "w"), indent=2)
    print(f"\n[out] {out}/exp4_report.json", flush=True)


if __name__ == "__main__":
    main()
