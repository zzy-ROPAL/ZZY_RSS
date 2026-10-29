"""E4 -- forced sign change on the real Schunk LWA 4D.

Goal: a scene in which finishing the weld REQUIRES changing a branch sign
mid-task, with no invented geometry.

Why this works on the real arm. All eight sign branches place shoulder, elbow,
wrist and tool identically, so an obstacle shrinks the collision-invalid set
of every page by the same amount. The pages differ only through joint limits,
and on the real LWA 4D q7's limit (+/-168.45 deg) is strictly below pi, which
makes the s6 mirror the only sign with a substantial asymmetry: at the q6-door
placement the s6=+ pages sit near 84% valid and the s6=- pages near 92%.

So growing one obstacle in the elbow's swing disconnects the s6=+ page first.
Once it does, the q6 door -- which crosses exactly s6 -- becomes the only
route from that page to the goal, and a sign change is forced.

The obstacle is a bar parallel to the weld placed in the elbow's swept volume
(elbow sweeps x in [0.11, 0.33], z in [0.07, 0.53]); the elbow must pass it on
one side or the other.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import (  # noqa: E402
    discs_in_region, label_regions, region_points,
)
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.params import (  # noqa: E402
    BoxObstacle, Config, PageMapParams, SceneParams, WeldParams,
)
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402

PAGES = ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---")
# s6-paired: the two entries differ only in the last sign.
S6_PAIRS = (("+++", "++-"), ("+-+", "+--"), ("-++", "-+-"), ("--+", "---"))

TILT_DEG = 60.0
WELD_P0 = (0.50, -0.26, 0.30)
WELD_P1 = (0.50, 0.26, 0.30)


def q6_weld() -> WeldParams:
    """The placement that admits a q6 door on the real LWA 4D."""
    a = np.radians(TILT_DEG)
    return WeldParams(p0=WELD_P0, p1=WELD_P1,
                      approach=(float(np.sin(a)), 0.0, -float(np.cos(a))),
                      reference=(float(np.cos(a)), 0.0, float(np.sin(a))))


def make_cfg(res: int, obstacles: tuple) -> Config:
    base = Config()
    return dataclasses.replace(
        base, weld=q6_weld(), pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams(gap_open=True, obstacles=obstacles),
        sigma_policy=dataclasses.replace(base.sigma_policy,
                                         enable_fiber_certification=True),
    )


def analyse(res: int, obstacles: tuple, want_doors: bool = True) -> dict:
    cfg = make_cfg(res, obstacles)
    arm, weld = SrsArm(cfg.arm), WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    pms, span = {}, {}
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        pm = build_page_map(cfg, arm, weld, scene, sg, with_kin=False)
        lab, regs = label_regions(cfg, pm)
        pms[n] = (pm, lab, regs)
        span[n] = [r.index for r in regs if r.reaches_start and r.reaches_goal]

    asym = [(a, b) for a, b in S6_PAIRS if bool(span[a]) != bool(span[b])]
    out = {
        "valid": {n: pms[n][0].stats["frac_valid"] for n in PAGES},
        "span": {n: bool(span[n]) for n in PAGES},
        "asym": asym, "edges": 0, "open_q6": 0, "forced": [],
    }
    if not want_doors:
        return out

    sigma = build_sigma(cfg, arm, weld, {n: v[0] for n, v in pms.items()},
                        verbose=False)
    edges = {}
    for n in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in n)
        pm, lab, regs = pms[n]
        tol = float(np.hypot(pm.y_grid[1] - pm.y_grid[0],
                             pm.theta_grid[1] - pm.theta_grid[0]))
        for d in sigma.get(n, []):
            if not d.vanishing:
                continue
            fib = door_fiber_validity(cfg, arm, weld, scene, d, sg, n=181)
            if not fib["ok"]:
                continue
            if 2 in d.vanishing:
                out["open_q6"] += 1
            ay, at = d.anchor
            for r in regs:
                pts = region_points(pm, lab, r.index)
                if d not in discs_in_region(sigma[n], pts, tol):
                    continue
                for o in d.pages:
                    if o == n or o not in pms:
                        continue
                    opm, olab, _ = pms[o]
                    i = int(np.argmin(np.abs(opm.y_grid - ay)))
                    j = int(np.argmin(np.abs(opm.theta_grid - at)))
                    rad = max(1, int(np.ceil(d.radius /
                                             (opm.y_grid[1] - opm.y_grid[0]))))
                    sub = olab[max(0, i - rad): i + rad + 1,
                               max(0, j - rad): j + rad + 1]
                    for kk in sorted({int(v) for v in np.unique(sub) if v > 0}):
                        edges.setdefault((n, r.index), set()).add(
                            ((o, kk), round(ay, 4), tuple(sorted(d.vanishing))))
                        edges.setdefault((o, kk), set()).add(
                            ((n, r.index), round(ay, 4), tuple(sorted(d.vanishing))))
    out["edges"] = sum(len(v) for v in edges.values()) // 2

    goals = {(n, r.index) for n in PAGES for r in pms[n][2] if r.reaches_goal}
    for s in PAGES:
        if span[s]:
            continue
        starts = [(s, r.index) for r in pms[s][2] if r.reaches_start]
        prev = {x: None for x in starts}
        q, hit = deque(starts), None
        while q:
            x = q.popleft()
            if x in goals:
                hit = x
                break
            for nxt, ay, van in edges.get(x, ()):
                if nxt not in prev:
                    prev[nxt] = (x, ay, van)
                    q.append(nxt)
        if hit is None:
            continue
        path, cur = [], hit
        while prev[cur] is not None:
            a_, ay, van = prev[cur]
            path.append((a_, cur, ay, van))
            cur = a_
        out["forced"].append({"start_page": s, "goal_at": f"{hit[0]}/{hit[1]}",
                              "path": list(reversed(path))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=41)
    ap.add_argument("--out", default="out/e4")
    args = ap.parse_args()

    print("E4: forced sign change on the real Schunk LWA 4D")
    print(f"weld {WELD_P0} -> {WELD_P1}, torch tilt {TILT_DEG:.0f} deg, "
          f"grid {args.res}x{args.res}")
    print("obstacle: bar parallel to the weld in the elbow's swept volume\n")

    print(f"{'bar z':>7}{'half z':>8}{'half y':>8}  "
          f"{'s6=+ valid':>11}{'s6=- valid':>11}{'q6 open':>9}  asymmetry / verdict")
    print("-" * 94)
    rows, hits = [], []
    # The q6 doors sit at normalised y = 0.254 / 0.746, i.e. world y = -+0.128.
    # A bar spanning the whole weld closes them; a narrower one centred at
    # mid-weld can break spanning while leaving the doors clear.
    for cz in (0.42, 0.46, 0.50):
        for hz in (0.060, 0.080, 0.100):
            for hy in (0.060, 0.090, 0.120):
                obs = (BoxObstacle(center=(0.215, 0.0, cz),
                                   half_extent=(0.085, hy, hz),
                                   label="elbow_bar"),)
                r = analyse(args.res, obs)
                vp = np.mean([r["valid"][n] for n in ("+++", "+-+", "-++", "--+")])
                vm = np.mean([r["valid"][n] for n in ("++-", "+--", "-+-", "---")])
                tag = ""
                if r["forced"]:
                    tag = f"*** FORCED from {[f['start_page'] for f in r['forced']]} ***"
                elif r["asym"] and r["open_q6"]:
                    tag = f"asym {len(r['asym'])} pair(s), {r['open_q6']} q6 open, no route"
                elif r["asym"]:
                    tag = f"asym {len(r['asym'])} pair(s) but q6 doors shut"
                elif r["open_q6"]:
                    tag = f"{r['open_q6']} q6 open, no asymmetry"
                print(f"{cz:>7.2f}{hz:>8.3f}{hy:>8.3f}"
                      f"{vp * 100:>10.1f}%{vm * 100:>10.1f}%{r['open_q6']:>9}  {tag}")
                rows.append({"cz": cz, "hz": hz, "hy": hy,
                             **{k: r[k] for k in ("span", "asym", "open_q6", "edges")},
                             "forced": r["forced"]})
                if r["forced"]:
                    hits.append((cz, hz, r))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e4_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nscenes forcing a sign change: {len(hits)}")
    for cz, hz, r in hits:
        for f in r["forced"]:
            hops = " -> ".join(f"{a[0]}/{a[1]}" for a, b, ay, van in f["path"])
            last = f["path"][-1][1]
            doors = ", ".join(f"y={ay} ({'+'.join('s%d' % (v * 2 + 2) for v in van)})"
                              for a, b, ay, van in f["path"])
            print(f"   bar z={cz} hz={hz}: start {f['start_page']} -> "
                  f"{hops} -> {last[0]}/{last[1]}   via {doors}")
    print(f"\n[out] wrote {out / 'e4_sweep.json'}")


if __name__ == "__main__":
    main()
