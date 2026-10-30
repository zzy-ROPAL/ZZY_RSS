"""Stage 0-2 experiment: page census, Sigma detection, page-map figures.

This is the decision gate for the project: if too few pages are non-empty or
no door is load-bearing, the headline scene must be redesigned before any of
the Stage 3 wall machinery is worth building.
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

from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.doors import build_sigma, certify_fiber  # noqa: E402
from zzy_rss.pagemap import build_all_pages, page_census  # noqa: E402
from zzy_rss.params import ArmParams, Config, PageMapParams, SceneParams  # noqa: E402
from zzy_rss.scene import Scene  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def make_config(res: int, gap_open: bool, off_axis: bool, fiber: bool) -> Config:
    base = Config()
    arm = dataclasses.replace(base.arm, use_off_axis_upper_arm=off_axis)
    policy = dataclasses.replace(base.sigma_policy, enable_fiber_certification=fiber)
    return dataclasses.replace(
        base,
        arm=arm,
        pagemap=PageMapParams(n_y=res, n_theta=res),
        scene=SceneParams.worked_illustration(gap_open),
        sigma_policy=policy,
    )


def run(res: int, gap_open: bool, off_axis: bool, fiber: bool, with_kin: bool,
        out_dir: Path, tag: str) -> dict:
    cfg = make_config(res, gap_open, off_axis, fiber)
    arm = SrsArm(cfg.arm)
    weld = WeldLine(cfg.weld)
    scene = Scene(cfg.arm, cfg.scene)

    box = cfg.scene.obstacles[0]
    print(f"\n{'=' * 78}")
    print(f"[run] tag={tag}  grid={res}x{res}  gap_open={gap_open}  "
          f"off_axis_upper_arm={off_axis}")
    print(f"[run] box center={box.center} half={box.half_extent}")
    print(f"[run] weld {cfg.weld.p0} -> {cfg.weld.p1}, reach=[{cfg.arm.reach_min:.3f},"
          f" {cfg.arm.reach_max:.3f}]")
    print(f"{'=' * 78}")

    t0 = time.time()
    pages = build_all_pages(cfg, arm, weld, scene, with_kin=with_kin)
    t_stage1 = time.time() - t0

    census = page_census(pages)
    print(f"\n[census] {census['n_pages_non_empty']}/{census['n_pages_total']} "
          f"pages non-empty: {census['non_empty']}")
    if census["empty"]:
        print(f"[census] empty pages: {census['empty']}")

    t0 = time.time()
    sigma = build_sigma(cfg, arm, weld, pages)
    t_stage2 = time.time() - t0

    n_doors = sum(
        1 for discs in sigma.values() for d in discs if "interior" in d.kind or "fold" in d.kind
    )
    print(f"\n[stage2] total door discs across all pages: {n_doors}")

    # Report the doors that connect two non-empty pages: only those can carry
    # connectivity the proof has to account for.
    live = set(census["non_empty"])
    load_bearing = []
    for name, discs in sigma.items():
        if name not in live:
            continue
        for d in discs:
            if not d.vanishing:
                continue
            others = [p for p in d.pages if p != name and p in live]
            if others:
                load_bearing.append((name, d, others))
    print(f"[stage2] doors joining two NON-EMPTY pages: {len(load_bearing)}")
    for name, d, others in load_bearing[:12]:
        print(f"         page {name} -> {others}  at (y={d.y:.4f}, th={d.theta:+.4f}) "
              f"kind={d.kind}")

    # Stage 5 refinement, only if explicitly switched on.
    fiber_reports = []
    if fiber:
        print("\n[stage5-fiber] certifying door fibers (OPTIONAL; not needed for "
              "soundness)")
        seen = set()
        for name, d, _others in load_bearing:
            key = (name, round(d.y, 4), round(d.theta, 4))
            if key in seen:
                continue
            seen.add(key)
            rep = certify_fiber(cfg, arm, weld, scene, d,
                                tuple(1 if c == "+" else -1 for c in name))
            if rep is not None:
                fiber_reports.append((name, rep))
                print(f"[stage5-fiber] page {name} door (y={rep.y:.4f}, "
                      f"th={rep.theta:+.4f}): {rep.reason}")

    print(f"\n[timing] stage1 {t_stage1:.1f}s   stage2 {t_stage2:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pages(pages, sigma, out_dir / f"pages_{tag}.png", tag, cfg)

    return {
        "tag": tag,
        "grid": res,
        "gap_open": gap_open,
        "off_axis_upper_arm": off_axis,
        "census": census,
        "page_stats": {n: p.stats for n, p in pages.items()},
        "n_door_discs": n_doors,
        "n_load_bearing_doors": len(load_bearing),
        "fiber": [
            {"page": n, "y": r.y, "theta": r.theta, "traversable": r.traversable,
             "blocked_fraction": r.blocked_fraction, "reason": r.reason}
            for n, r in fiber_reports
        ],
        "timing": {"stage1_s": t_stage1, "stage2_s": t_stage2},
    }


def plot_pages(pages: dict, sigma: dict, path: Path, tag: str, cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # valid / no-IK / joint-limit / collision
    cmap = ListedColormap(["#2a9d8f", "#e9e9e9", "#e9c46a", "#e76f51"])
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for ax, (name, pm) in zip(axes.ravel(), pages.items()):
        ax.imshow(
            pm.tag.T, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
            extent=[pm.y_grid[0], pm.y_grid[-1], pm.theta_grid[0], pm.theta_grid[-1]],
        )
        for d in sigma.get(name, []):
            if "interior" in d.kind or "fold" in d.kind:
                ax.plot(d.y, d.theta, "o", ms=7, mfc="none", mec="#1d3557", mew=2.0)
        ax.set_title(f"page ({name})   valid {pm.stats['frac_valid'] * 100:.1f}%",
                     fontsize=11)
        ax.set_xlabel("y  (weld coordinate)")
        ax.set_ylabel(r"$\theta$  (arm angle)")
    handles = [
        mpatches.Patch(color="#2a9d8f", label="valid"),
        mpatches.Patch(color="#e9e9e9", label="invalid: no IK"),
        mpatches.Patch(color="#e9c46a", label="invalid: joint limit"),
        mpatches.Patch(color="#e76f51", label="invalid: collision"),
        plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#1d3557",
                   mew=2.0, ms=7, label=r"$\Sigma$ door"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Stage 1 page maps  —  {tag}", fontsize=14)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=121)
    ap.add_argument("--off-axis", action="store_true")
    ap.add_argument("--fiber", action="store_true")
    ap.add_argument("--with-kin", action="store_true")
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    results = []
    for gap_open in (True, False):
        suffix = "gap_open" if gap_open else "gap_closed"
        tag = f"{suffix}{'_offaxis' if args.off_axis else ''}"
        results.append(
            run(args.res, gap_open, args.off_axis, args.fiber, args.with_kin,
                out_dir, tag)
        )

    path = out_dir / "stage12_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\n[out] wrote {path}")


if __name__ == "__main__":
    main()
