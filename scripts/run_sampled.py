"""EXP 1 and EXP 2 with a SAMPLED roadmap driving the proof.

The grid is demoted to a rendering device. Nothing the proof depends on
comes from it.

Why: IJRR never builds a grid. Li samples a PRM and reads the training
classes off its connected components (makeTrainingDataFromGraph). Our 61x61
page maps were our own addition for Y+, and they are 55% of the runtime and
cost 61^(page dim), which does not survive a higher-dimensional task. See
findings 25.

What drives the proof here:
  * uniform samples in (y, theta) per page; VALID ones become roadmap
    vertices, INVALID ones are kept as P_obs for Algorithm 2
  * k-nearest-neighbour edges, each validated by a checkMotion-style scan
  * connected components give the two training classes, Li's way: the
    component(s) reaching the goal are one class, every other vertex is the
    other
  * Algorithms 1-4 then run unchanged on those classes

What the grid is still used for: colouring the obstacle region behind the
scatter in the figure, exactly as Li's Figure 7 fills its 2-D maze from the
scene image. We have no analytic description of the invalid set on a page,
so a filled background has to be evaluated somewhere; doing it for the
picture only keeps it out of the algorithm.

Figure follows IJRR Figure 7/8: obstacle region, start and goal, graph
components with the goal, other vertices, learned manifold.

Usage:
  python3 scripts/run_sampled.py --samples 1200 --out out/sampled
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from zzy_rss.doors import build_sigma, door_fiber_validity  # noqa: E402
from zzy_rss.flood import find_inf_proof_region, live_discs  # noqa: E402
from zzy_rss.pagemap import build_page_map  # noqa: E402
from zzy_rss.prm import classify  # noqa: E402
from zzy_rss.wall import (  # noqa: E402
    check_polytope, reconstruct_wall, sample_wall, sigma_disc_samples,
    train_wall, virtual_boundary_samples,
)

import run_three as rt  # noqa: E402

PAGES = rt.PAGES
START_PAGE, START_SIGNS = rt.START_PAGE, rt.START_SIGNS
GOAL_TOL = 0.02
OBST_BLUE = "#aed6f1"


# --------------------------------------------------------------------------
# the roadmap -- this is what the proof runs on
# --------------------------------------------------------------------------


def sample_page(cfg, arm, weld, scene, signs, n, rng):
    """Uniform samples in (y, theta); split into valid vertices and P_obs."""
    y = rng.uniform(0.0, 1.0, n)
    th = rng.uniform(-np.pi, np.pi, n)
    pts = np.column_stack([y, th])
    ok = np.zeros(n, dtype=bool)
    for i, (yy, tt) in enumerate(pts):
        ok[i] = classify(cfg, arm, weld, scene, float(yy), float(tt), signs)[0]
    return pts[ok], pts[~ok]


def motion_valid(cfg, arm, weld, scene, signs, a, b, n_steps=6):
    """checkMotion: every interpolated sample on the segment must be valid."""
    for t in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        p = a + t * (b - a)
        if not classify(cfg, arm, weld, scene, float(p[0]), float(p[1]),
                        signs)[0]:
            return False
    return True


def build_roadmap(cfg, arm, weld, scene, signs, verts, k=8):
    """k-nearest-neighbour PRM over the valid samples. Returns labels, n_comp."""
    n = len(verts)
    if n == 0:
        return np.zeros(0, dtype=int), 0, []
    scale = np.array([cfg.wall.scale_y, cfg.wall.scale_theta])
    tree = cKDTree(verts * scale)
    kk = min(k + 1, n)
    _d, idx = tree.query(verts * scale, k=kk)
    idx = np.atleast_2d(idx)
    rows, cols, edges = [], [], []
    for i in range(n):
        for j in idx[i][1:]:
            j = int(j)
            if j <= i:
                continue
            if motion_valid(cfg, arm, weld, scene, signs, verts[i], verts[j]):
                rows.append(i); cols.append(j); edges.append((i, j))
    if rows:
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        g = coo_matrix((n, n))
    ncomp, lab = connected_components(g, directed=False)
    return lab, ncomp, edges


def path_from_roadmap(verts, edges, start_idx, to_goal=True):
    """BFS on the PRM itself -- the motion comes from the roadmap, not a grid."""
    if start_idx is None or not len(verts):
        return None
    adj = {}
    for i, j in edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    from collections import deque
    prev = {start_idx: None}
    q = deque([start_idx])
    best = start_idx
    while q:
        u = q.popleft()
        if verts[u][0] > verts[best][0]:
            best = u
        if to_goal and verts[u][0] >= 1.0 - GOAL_TOL:
            best = u
            break
        for w in adj.get(u, ()):
            if w not in prev:
                prev[w] = u
                q.append(w)
    out, cur = [], best
    while cur is not None:
        out.append(verts[cur])
        cur = prev[cur]
    out.reverse()
    return np.array(out) if len(out) > 1 else None


def resample_path(path, n=61):
    """Even resampling so the animation moves at a steady rate."""
    if path is None or len(path) < 2:
        return path
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if d[-1] <= 0:
        return path
    t = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(t, d, path[:, 0]),
                            np.interp(t, d, path[:, 1])])


def goal_classes(verts, lab):
    """Li's split: components touching the goal vs every other vertex."""
    if len(verts) == 0:
        return np.zeros(0, dtype=bool)
    goal_comps = set(lab[verts[:, 0] >= 1.0 - GOAL_TOL])
    return np.array([c in goal_comps for c in lab])


# --------------------------------------------------------------------------
# figure -- the grid appears HERE and nowhere else
# --------------------------------------------------------------------------


def page_background(cfg, arm, weld, scene, signs, res):
    """Validity on a grid, used only to colour the obstacle region."""
    pm = build_page_map(cfg, arm, weld, scene, signs, with_kin=False)
    return pm


def fig8_sampled(cfg, data, title, out_path, res_note):
    fig, axes = plt.subplots(2, 4, figsize=(21, 10.2))
    for ax, name in zip(axes.ravel(), PAGES):
        d = data[name]
        pm = d["shade"]
        ext = [0, 1, pm.theta_grid[0], pm.theta_grid[-1]]
        # obstacle region (grid, for display only)
        ax.imshow(np.where(pm.valid_mask.T, 0.0, 1.0), origin="lower",
                  aspect="auto", extent=ext,
                  cmap=ListedColormap(["white", OBST_BLUE]), vmin=0, vmax=1,
                  interpolation="nearest")
        # roadmap vertices
        v, ing = d["verts"], d["in_goal"]
        if len(v):
            if (~ing).any():
                ax.plot(v[~ing, 0], v[~ing, 1], "x", color="#d62728", ms=3.4,
                        mew=0.9, linestyle="none")
            if ing.any():
                ax.plot(v[ing, 0], v[ing, 1], "s", mfc="none", mec="#1f4eb8",
                        ms=3.4, mew=0.9, linestyle="none")
        # learned manifold
        for loop in d.get("wall", []):
            ax.plot(loop[:, 0], loop[:, 1], color="black", lw=2.2)
        # doors
        for disc in d.get("discs", []):
            ay, at = disc.anchor
            ax.plot([ay], [at], "o" if disc.__dict__.get("_state") == "open"
                    else "X", color="#1f4eb8" if
                    disc.__dict__.get("_state") == "open" else "#d62728",
                    ms=7, mfc="none" if disc.__dict__.get("_state") == "open"
                    else None, mew=1.6)
        if name == START_PAGE and d.get("start_xy") is not None:
            sx, sy = d["start_xy"]
            ax.plot([sx], [sy], "D", color="#6a1b9a", ms=10,
                    mec="white", mew=1.4)
        gv = v[ing] if len(v) else np.zeros((0, 2))
        gv = gv[gv[:, 0] >= 1.0 - GOAL_TOL] if len(gv) else gv
        if len(gv):
            ax.plot(gv[:, 0], gv[:, 1], "*", color="#6a1b9a", ms=13,
                    mec="white", mew=1.0, linestyle="none")
        tag = "  <- START PAGE" if name == START_PAGE else ""
        ax.set_title(f"({name})  {len(v)} vertices, "
                     f"{d['ncomp']} component(s){tag}",
                     fontsize=10,
                     fontweight="bold" if name == START_PAGE else "normal")
        ax.set_xlabel("y  (weld coordinate)")
        ax.set_ylabel(r"$\theta$  (arm angle)")
        ax.set_xlim(0, 1)
        ax.set_ylim(pm.theta_grid[0], pm.theta_grid[-1])

    handles = [
        Line2D([], [], marker="*", color="#6a1b9a", ls="none", ms=12,
               label="Goal and start conf"),
        Line2D([], [], marker="D", color="#6a1b9a", ls="none", ms=8, label=""),
        mpatches.Patch(facecolor=OBST_BLUE, label="Obstacle region"),
        Line2D([], [], marker="s", mfc="none", mec="#1f4eb8", ls="none",
               ms=8, label="Graph components with goal"),
        Line2D([], [], marker="x", color="#d62728", ls="none", ms=8,
               label="Other vertices"),
        Line2D([], [], color="black", lw=2.2, label="Learned manifold"),
        Line2D([], [], marker="X", color="#d62728", ls="none", ms=8,
               label=r"$\Sigma$ door BLOCKED"),
    ]
    fig.legend(handles=[h for h in handles if h.get_label()],
               loc="lower center", ncol=6, frameon=True, fontsize=11)
    fig.suptitle(title + "\n" + res_note, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    print(f"[fig] {out_path}", flush=True)


# --------------------------------------------------------------------------


def run_case(tag, tilt, obstacles, args, out):
    print(f"\n=== {tag} ===", flush=True)
    t0 = time.time()
    cfg, arm, weld, scene = rt.make(tilt, obstacles, args.res)
    rng = np.random.default_rng(1)

    data, pms = {}, {}
    for name in PAGES:
        signs = tuple(1 if c == "+" else -1 for c in name)
        verts, obs = sample_page(cfg, arm, weld, scene, signs, args.samples, rng)
        lab, ncomp, edges = build_roadmap(cfg, arm, weld, scene, signs, verts,
                                          k=args.k)
        ing = goal_classes(verts, lab)
        pm = page_background(cfg, arm, weld, scene, signs, args.res)
        pms[name] = pm
        # a finer grid for the picture only: at 61 cells imshow paints a
        # whole 0.0167 x 0.105 block one colour, which swallows valid
        # samples sitting in thin slivers and makes start/goal look as if
        # they are inside an obstacle.
        shade = build_page_map(
            dataclasses.replace(
                cfg, pagemap=dataclasses.replace(
                    cfg.pagemap, n_y=args.fig_res, n_theta=args.fig_res)),
            arm, weld, scene, signs, with_kin=False)
        data[name] = dict(verts=verts, obs=obs, lab=lab, ncomp=ncomp,
                          in_goal=ing, pm=pm, shade=shade, start_theta=0.0,
                          start_xy=None, edges=edges)
        print(f"  page {name}: {len(verts)} valid / {args.samples} sampled, "
              f"{ncomp} component(s), {int(ing.sum())} reach the goal",
              flush=True)

    sig = build_sigma(cfg, arm, weld, pms, verbose=False)
    n_open = n_blk = 0
    for name in PAGES:
        sg = tuple(1 if c == "+" else -1 for c in name)
        for d in sig.get(name, []):
            if d.vanishing:
                ok = door_fiber_validity(cfg, arm, weld, scene, d, sg)["ok"]
                d.__dict__["_state"] = "open" if ok else "blocked"
                n_open += ok; n_blk += (not ok)
        data[name]["discs"] = sig.get(name, [])
    print(f"  doors {n_open} open / {n_blk} blocked", flush=True)

    # start vertex: page START_PAGE, y = 0
    sp = data[START_PAGE]
    sv = sp["verts"]
    near = sv[sv[:, 0] < 0.08] if len(sv) else np.zeros((0, 2))
    if len(near):
        si0 = int(np.argmin(np.abs(near[:, 1] + 1.57)))
        start_xy = near[si0]
    else:
        start_xy = np.array([0.0, 0.0])
    start_theta = float(start_xy[1])
    sp["start_theta"] = start_theta
    sp["start_xy"] = (float(start_xy[0]), float(start_xy[1]))
    assert classify(cfg, arm, weld, scene, float(start_xy[0]),
                    float(start_xy[1]), START_SIGNS)[0], \
        "start marker must be a VALID configuration"

    start_comp = None
    if len(sv):
        i = int(np.argmin(np.linalg.norm(
            sv - start_xy, axis=1)))
        start_comp = int(sp["lab"][i])
    reaches = bool(sp["in_goal"][sp["lab"] == start_comp].any()) \
        if start_comp is not None else False
    print(f"  start component {start_comp} reaches the goal: {reaches}",
          flush=True)

    result = dict(tag=tag, tilt=tilt, doors_open=n_open, doors_blocked=n_blk,
                  start_reaches_goal=reaches,
                  vertices={n: int(len(data[n]["verts"])) for n in PAGES},
                  components={n: int(data[n]["ncomp"]) for n in PAGES})

    if not reaches:
        # ---- the proof, on the SAMPLED classes (Li's split) --------------
        sg = START_SIGNS
        v, ing = sp["verts"], sp["in_goal"]
        rest = v[~ing]                      # includes the start component
        goal_cls = v[ing]
        if len(goal_cls) == 0:              # nothing reaches the goal
            goal_cls = virtual_boundary_samples(cfg, 80)
        live, disch = live_discs(cfg, arm, weld, scene, sg, sig[START_PAGE])
        tol = 1.0 / args.res
        se, re_ = sigma_disc_samples(live, rest, tol)
        seed = rest if not len(se) else np.vstack([rest, se])
        other = np.vstack([goal_cls, virtual_boundary_samples(cfg, 80)])
        if len(re_):
            other = np.vstack([other, re_])
        inv = sp["obs"]
        if len(inv) > 200:
            inv = inv[rng.choice(len(inv), 200, replace=False)]

        w0 = train_wall(cfg, seed, other)
        print(f"  alg1: gamma={w0.gamma:.2f} acc={w0.accuracy:.4f} "
              f"({len(seed)} start-side / {len(other)} goal-side)", flush=True)
        # IJRR Algorithm 4: Alg 2 + reconstruction + Alg 3 in a feedback loop,
        # retraining on any facet centre that turns out valid.
        poly, rep, wall, iters, rounds = find_inf_proof_region(
            cfg, arm, weld, scene, sg, seed, other, inv, live,
            max_iters=args.iters, verbose=True)
        data[START_PAGE]["wall"] = poly.loops
        print(f"  alg4: {iters} iteration(s), alg2 {rounds} round(s), "
              f"gamma={wall.gamma:.2f}", flush=True)
        print(f"  WALL: {rep.n_passed}/{rep.n_facets} facets, "
              f"closed={poly.closed} -> "
              f"{'PROOF COMPLETE' if rep.ok and poly.closed else 'incomplete'}",
              flush=True)
        result.update(gamma=float(wall.gamma), alg2_rounds=int(rounds),
                      facets=[int(rep.n_passed), int(rep.n_facets)],
                      closed=bool(poly.closed),
                      proof=bool(rep.ok and poly.closed))

    # ---- the motion, taken from the ROADMAP ------------------------------
    si = None
    if len(sv):
        si = int(np.argmin(np.linalg.norm(sv - start_xy, axis=1)))
    raw = path_from_roadmap(sv, sp["edges"], si, to_goal=True)
    path = resample_path(raw, 61)
    if path is not None:
        q_list, keep = rt.qs_from_path(arm, weld, path, START_SIGNS)
        stall = float(path[-1, 0])
        cap = (lambda i, p=path, k=keep, t=tag:
               f"{t.split(' - ')[0]}   weld {100*p[k[i],0]:.0f}% "
               f"({'reached y=1.00' if p[-1,0] >= 1.0 - GOAL_TOL else 'blocked at y=%.2f' % p[-1,0]})")
        rt.render([dict(cfg=cfg, arm=arm, weld=weld, scene=scene, pm=sp["pm"],
                        q_list=q_list, path=path, upto_list=keep,
                        wall=data[START_PAGE].get("wall"),
                        discs=sig[START_PAGE], page=START_PAGE, caption=cap)],
                  out / f"{tag.split()[0].lower()}_sampled.mp4")
        result["path_waypoints"] = int(len(path))
        result["path_reaches"] = stall
        print(f"  roadmap path: {len(raw)} vertices, reaches y={stall:.2f}",
              flush=True)

    note = (f"roadmap: {args.samples} samples/page, k={args.k} neighbours, "
            f"checkMotion on every edge.  Every plotted vertex is a VALID "
            f"configuration by construction; the grid ({args.fig_res}x"
            f"{args.fig_res}) only shades the obstacle region.")
    fig8_sampled(cfg, data, tag, out / f"{result['tag'].split()[0].lower()}"
                 f"_sampled_pages.png", note)
    result["seconds"] = round(time.time() - t0, 1)
    print(f"  [{time.time()-t0:.1f}s]", flush=True)
    return result, data, cfg, arm, weld, scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--res", type=int, default=61)
    ap.add_argument("--fig-res", dest="fig_res", type=int, default=241,
                    help="grid resolution used ONLY to shade the figure")
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--out", default="out/sampled")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    report = {}
    r1, d1, cfg1, arm1, weld1, scene1 = run_case(
        "EXP1 no obstacle - the robot FINISHES the weld", 0.0, None, args, out)
    report["exp1"] = r1
    r2, d2, cfg2, arm2, weld2, scene2 = run_case(
        "EXP2 two-box obstacle - INFEASIBLE, proof in one page", 0.0,
        rt.ZGAP, args, out)
    report["exp2"] = r2

    json.dump(report, open(out / "sampled_report.json", "w"), indent=1)
    print(f"\n[out] {out / 'sampled_report.json'}")


if __name__ == "__main__":
    main()
