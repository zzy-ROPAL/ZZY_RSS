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
    shared: bool = True,
) -> dict:
    """Stage 1 for every sign branch. Returns {page_name: PageMap}.

    shared=True routes to build_all_pages_shared, which is EXACT (asserted
    cell by cell by tests/test_ik.py::test_shared_page_build_matches) and
    about 4.5x faster. shared=False is the original one-page-at-a-time build,
    kept as the reference the test compares against.
    """
    if shared:
        return build_all_pages_shared(cfg, arm, weld, scene,
                                      with_kin=with_kin, verbose=verbose)
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


# --------------------------------------------------------------------------
# Stage 1, eight pages for the price of one (findings 31 / 33)
# --------------------------------------------------------------------------

# s2, s4 and s6 each act on the joint vector as the same representational
# mirror, asserted exactly (1e-9) by tests/test_ik.py::test_mirror_identities:
# the MIDDLE joint negates and the two OUTER joints shift by pi.
#
#   s2 : q1 -> q1 +- pi,  q2 -> -q2,  q3 -> q3 +- pi
#   s4 : q3 -> q3 +- pi,  q4 -> -q4,  q5 -> q5 +- pi
#   s6 : q5 -> q5 +- pi,  q6 -> -q6,  q7 -> q7 +- pi
#
# (0-based joint indices below.)
MIRROR_JOINTS = {0: (0, 1, 2), 1: (2, 3, 4), 2: (4, 5, 6)}


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]. q +- pi is one angle once wrapped, so the sign of
    the shift in the mirror identity is not a choice."""
    return -((-np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi)


def mirror_q(q: np.ndarray, sign_index: int) -> np.ndarray:
    """Apply the s2/s4/s6 mirror to one joint vector. Exact, no IK."""
    i, j, k = MIRROR_JOINTS[sign_index]
    out = np.array(q, dtype=float, copy=True)
    out[j] = -out[j]
    out[i] = _wrap(out[i] + np.pi)
    out[k] = _wrap(out[k] + np.pi)
    return out


def q_for_branch(q_ref: np.ndarray, signs_ref: tuple, signs: tuple) -> np.ndarray:
    """The joint vector of the SAME arm placement written in another branch.

    All eight branches place shoulder, elbow, wrist and tool identically
    (findings 31, asserted to 3e-16 by test_branch_geometry_identical); they
    are eight coordinate spellings of one posture. So one IK call plus the
    mirror identities gives all eight, and collision -- which depends on the
    posture, not the spelling -- need only be evaluated once.
    """
    q = np.array(q_ref, dtype=float, copy=True)
    for idx in range(3):
        if signs[idx] != signs_ref[idx]:
            q = mirror_q(q, idx)
    return q


def geometry_is_page_dependent(cfg: Config) -> bool:
    """Does any collision capsule MOVE when a sign is flipped?

    The mirrors leave the arm's SKELETON alone -- shoulder, elbow, wrist and
    tool are the same points on all eight pages, which is what makes one
    collision check serve all of them. Measured: the upper-arm, forearm and
    tool capsules move 0.0 mm across the eight pages.

    An OFF-AXIS capsule is different. The hubs hang off frames 2 and 3, which
    DO rotate between pages, so flipping s2 moves the upper-arm hub by 210 mm
    and flipping s4 moves the elbow hub by 180 mm. With either enabled,
    collision is no longer a property of the cell alone and the shared build
    must check each page separately.

    This is the general statement, and it is worth being precise about: page
    independence of collision is a property of the COLLISION MODEL, not of the
    kinematics. It holds for geometry that is rotationally symmetric about each
    link's own joint-to-joint axis -- capsules on the S-E-W-tool skeleton. It
    will NOT hold for real STL meshes (todo 4), which are not symmetric about
    those axes. The IK sharing survives either way, since the mirror
    identities are exact; only the collision sharing is lost.
    """
    p = cfg.arm
    return bool(getattr(p, "use_off_axis_upper_arm", False)
                or getattr(p, "use_off_axis_elbow", False))


def build_all_pages_shared(
    cfg: Config,
    arm: SrsArm,
    weld: WeldLine,
    scene: Scene,
    with_kin: bool = False,
    verbose: bool = True,
) -> dict:
    """All eight pages from ONE IK and ONE collision check per grid cell.

    build_all_pages does 8 x n_cells of (IK + FK + collision). This does
    1 x n_cells of (IK + FK + collision) plus 8 x n_cells of a joint-limit
    comparison, because:

      * IK EXISTENCE depends only on |W(y) - S| lying inside the annulus,
        which is a property of y alone -- no sign appears in it;
      * COLLISION is bit-for-bit page-independent (findings 31: all eight
        pages differ in 0 of 3721 cells when limits are ignored);
      * only the JOINT-LIMIT test can tell the pages apart, and the mirror
        identities give the eight joint vectors in closed form.

    The result is asserted equal to build_all_pages cell by cell -- see
    tests/test_ik.py::test_shared_page_build_matches.

    with_kin still costs one finite-difference Jacobian per cell per page, so
    it is OFF by default here; the callers that need Sigma_kin ask for it.
    """
    pm_p = cfg.pagemap
    ys = np.linspace(pm_p.y_min, pm_p.y_max, pm_p.n_y)
    thetas = np.linspace(pm_p.theta_min, pm_p.theta_max, pm_p.n_theta)
    shape = (pm_p.n_y, pm_p.n_theta)

    ref = SIGN_BRANCHES[0]
    # One collision check per cell is only valid when no capsule moves between
    # pages. With off-axis geometry it does, so fall back to one check per page
    # -- still sharing the IK, which the mirror identities make exact.
    page_dep = geometry_is_page_dependent(cfg)
    if page_dep and verbose:
        print("[stage1] off-axis geometry is on: collision is page-DEPENDENT, "
              "checking each page separately (the IK is still shared)")
    fields = {}
    for signs in SIGN_BRANCHES:
        fields[signs] = dict(
            tag=np.full(shape, INVALID_NO_IK, dtype=np.int8),
            q2=np.full(shape, np.nan), q4=np.full(shape, np.nan),
            q6=np.full(shape, np.nan),
            clearance=np.full(shape, np.nan),
            limit_margin=np.full(shape, np.nan),
            kin_measure=np.full(shape, np.nan),
            near_full=np.zeros(shape, dtype=bool),
            near_axis=np.zeros(shape, dtype=bool),
        )

    poses = [weld.pose(float(y)) for y in ys]
    for i, _y in enumerate(ys):
        pos, rot = poses[i]
        for j, theta in enumerate(thetas):
            res = arm.ik_branch(pos, rot, float(theta), ref,
                                coord_axis_tol=pm_p.coord_axis_tol,
                                coord_radius_tol=pm_p.coord_radius_tol)
            if not res.ok:
                continue    # no-IK is page-independent; tags already say so

            # One posture -> one collision report, shared by all eight pages
            # (only when no capsule moves between pages; see page_dep above).
            report = None if page_dep else scene.check(arm.frames(res.q))

            for signs in SIGN_BRANCHES:
                q = res.q if signs == ref else q_for_branch(res.q, ref, signs)
                f = fields[signs]
                f["q2"][i, j], f["q4"][i, j], f["q6"][i, j] = q[1], q[3], q[5]
                f["near_full"][i, j] = res.near_full_extension
                f["near_axis"][i, j] = res.near_axis_align
                margin = arm.limit_margin(q)
                f["limit_margin"][i, j] = margin
                if margin < 0.0:
                    f["tag"][i, j] = INVALID_JOINT_LIMIT
                    continue
                rep = report if report is not None else scene.check(arm.frames(q))
                f["clearance"][i, j] = rep.clearance
                if rep.in_collision:
                    f["tag"][i, j] = INVALID_COLLISION
                    continue
                f["tag"][i, j] = VALID
                if with_kin:
                    f["kin_measure"][i, j] = arm.kin_singularity_measure(q)

    pages = {}
    for signs in SIGN_BRANCHES:
        f = fields[signs]
        tag = f["tag"]
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
        pmap = PageMap(signs=signs, y_grid=ys, theta_grid=thetas, stats=stats,
                       **f)
        pages[pmap.name] = pmap
        if verbose:
            s = pmap.stats
            print(f"[stage1] page {s['page']}  valid {s['frac_valid']*100:6.2f}%  "
                  f"no-IK {s['n_no_ik']/s['n_cells']*100:6.2f}%  "
                  f"limit {s['n_joint_limit']/s['n_cells']*100:6.2f}%  "
                  f"coll {s['n_collision']/s['n_cells']*100:6.2f}%"
                  f"{'   [EMPTY]' if pmap.is_empty else ''}")
    return pages
