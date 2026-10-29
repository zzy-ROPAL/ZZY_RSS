"""Stage 1 -- page maps.

One page per sign branch (s2, s4, s6). Each page is a dense grid over
(y, theta) classified as valid or invalid, and every invalid cell carries the
reason (no-IK / joint-limit / collision) because the reason selects which
penetration-depth routine the facet check will call.

The grid is ground truth and intuition only; it is never part of the proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arm import (
    INVALID_COLLISION,
    INVALID_JOINT_LIMIT,
    INVALID_NO_IK,
    SIGN_BRANCHES,
    TAG_NAMES,
    VALID,
    SrsArm,
)
from .params import Config
from .scene import Scene
from .weld import WeldLine


def page_name(signs: tuple[int, int, int]) -> str:
    return "".join("+" if s > 0 else "-" for s in signs)


@dataclass
class PageMap:
    """Classified grid for a single sign branch."""

    signs: tuple[int, int, int]
    y_grid: np.ndarray
    theta_grid: np.ndarray
    tag: np.ndarray            # int array, VALID / INVALID_*
    q2: np.ndarray             # NaN where no IK
    q4: np.ndarray
    q6: np.ndarray
    clearance: np.ndarray      # capsule clearance, NaN where no IK
    limit_margin: np.ndarray   # slack to nearest joint limit, NaN where no IK
    kin_measure: np.ndarray    # det(J J^T), NaN where no IK
    near_full: np.ndarray      # bool, elbow circle collapsed
    near_axis: np.ndarray      # bool, arm-angle reference degenerate
    stats: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return page_name(self.signs)

    @property
    def valid_mask(self) -> np.ndarray:
        return self.tag == VALID

    @property
    def is_empty(self) -> bool:
        return not bool(np.any(self.valid_mask))


def build_page_map(
    cfg: Config,
    arm: SrsArm,
    weld: WeldLine,
    scene: Scene,
    signs: tuple[int, int, int],
    with_kin: bool = True,
) -> PageMap:
    """Classify one page on the dense (y, theta) grid."""
    pm = cfg.pagemap
    ys = np.linspace(pm.y_min, pm.y_max, pm.n_y)
    thetas = np.linspace(pm.theta_min, pm.theta_max, pm.n_theta)

    shape = (pm.n_y, pm.n_theta)
    tag = np.full(shape, INVALID_NO_IK, dtype=np.int8)
    q2 = np.full(shape, np.nan)
    q4 = np.full(shape, np.nan)
    q6 = np.full(shape, np.nan)
    clearance = np.full(shape, np.nan)
    limit_margin = np.full(shape, np.nan)
    kin_measure = np.full(shape, np.nan)
    near_full = np.zeros(shape, dtype=bool)
    near_axis = np.zeros(shape, dtype=bool)

    # Tool pose depends only on y, so cache it per row.
    poses = [weld.pose(float(y)) for y in ys]

    for i, y in enumerate(ys):
        pos, rot = poses[i]
        for j, theta in enumerate(thetas):
            res = arm.ik_branch(
                pos,
                rot,
                float(theta),
                signs,
                coord_axis_tol=pm.coord_axis_tol,
                coord_radius_tol=pm.coord_radius_tol,
            )
            if not res.ok:
                tag[i, j] = INVALID_NO_IK
                continue
            q = res.q
            q2[i, j] = q[1]
            q4[i, j] = q[3]
            q6[i, j] = q[5]
            near_full[i, j] = res.near_full_extension
            near_axis[i, j] = res.near_axis_align

            margin = arm.limit_margin(q)
            limit_margin[i, j] = margin
            if margin < 0.0:
                tag[i, j] = INVALID_JOINT_LIMIT
                continue

            frames = arm.frames(q)
            report = scene.check(frames)
            clearance[i, j] = report.clearance
            if report.in_collision:
                tag[i, j] = INVALID_COLLISION
                continue

            tag[i, j] = VALID
            if with_kin:
                kin_measure[i, j] = arm.kin_singularity_measure(q)

    total = tag.size
    stats = {
        "page": page_name(signs),
        "n_cells": int(total),
        "n_valid": int(np.sum(tag == VALID)),
        "n_no_ik": int(np.sum(tag == INVALID_NO_IK)),
        "n_joint_limit": int(np.sum(tag == INVALID_JOINT_LIMIT)),
        "n_collision": int(np.sum(tag == INVALID_COLLISION)),
    }
    stats["frac_valid"] = stats["n_valid"] / total

    return PageMap(
        signs=signs,
        y_grid=ys,
        theta_grid=thetas,
        tag=tag,
        q2=q2,
        q4=q4,
        q6=q6,
        clearance=clearance,
        limit_margin=limit_margin,
        kin_measure=kin_measure,
        near_full=near_full,
        near_axis=near_axis,
        stats=stats,
    )


def build_all_pages(
    cfg: Config,
    arm: SrsArm,
    weld: WeldLine,
    scene: Scene,
    with_kin: bool = True,
    verbose: bool = True,
) -> dict:
    """Stage 1 for every sign branch. Returns {page_name: PageMap}."""
    pages = {}
    for signs in SIGN_BRANCHES:
        pmap = build_page_map(cfg, arm, weld, scene, signs, with_kin=with_kin)
        pages[pmap.name] = pmap
        if verbose:
            s = pmap.stats
            print(
                f"[stage1] page {s['page']}  valid {s['frac_valid'] * 100:6.2f}%  "
                f"no-IK {s['n_no_ik'] / s['n_cells'] * 100:6.2f}%  "
                f"limit {s['n_joint_limit'] / s['n_cells'] * 100:6.2f}%  "
                f"coll {s['n_collision'] / s['n_cells'] * 100:6.2f}%"
                f"{'   [EMPTY]' if pmap.is_empty else ''}"
            )
    return pages


def page_census(pages: dict) -> dict:
    """The number this project needs before any theory: how many pages live."""
    non_empty = [name for name, pm in pages.items() if not pm.is_empty]
    return {
        "n_pages_total": len(pages),
        "n_pages_non_empty": len(non_empty),
        "non_empty": non_empty,
        "empty": [name for name, pm in pages.items() if pm.is_empty],
    }
