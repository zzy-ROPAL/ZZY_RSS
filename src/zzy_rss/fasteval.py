"""Fast scene evaluation for scene SEARCH, not for proofs.

A scene search asks the same question thousands of times: given this weld
placement, what does the page map look like with THIS obstacle? Rebuilding the
page map each time re-runs the inverse kinematics and the forward kinematics,
which is 97% of the work and none of it depends on the obstacle.

Three facts make the split possible, all of them established measurements:

  * IK EXISTENCE depends only on |W(y) - S| lying inside the reachable
    annulus, which is a property of y alone -- no sign and no obstacle appears
    in it.
  * All eight sign branches place the arm IDENTICALLY (findings 31, asserted
    to 3e-16 by tests/test_ik.py::test_branch_geometry_identical), so the
    capsule geometry of a cell is page-independent, and the mirror identities
    give the other seven joint vectors in closed form.
  * COLLISION therefore depends on the obstacle and the cell, never on the
    page.

So `Precomputed` does the kinematics once per weld placement and caches the
capsule sample points; after that each candidate obstacle costs one vectorised
distance evaluation. Measured at res 41: 0.37 s to precompute, 12.9 ms per
obstacle, against ~400 ms to rebuild two page maps from scratch.

WHAT THIS IS NOT FOR. The page maps it produces are for SEARCH ONLY. Grid
connectivity is not an infeasibility criterion (invariant 8) and must not be
used as one; a candidate found here has to be re-checked at two or more
resolutions (invariant 23 -- a single-resolution screen produced 13 false
positives in findings 34e) and then put through the real `flood.run_proof`.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .arm import (
    INVALID_COLLISION,
    INVALID_JOINT_LIMIT,
    INVALID_NO_IK,
    SIGN_BRANCHES,
    VALID,
    SrsArm,
)
from .pagemap import PageMap, page_name, q_for_branch
from .params import BoxObstacle, Config
from .scene import Scene
from .weld import WeldLine

REF_BRANCH = SIGN_BRANCHES[0]


class Precomputed:
    """Everything about a weld placement that no obstacle can change."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.arm = SrsArm(cfg.arm)
        self.weld = WeldLine(cfg.weld)
        # An EMPTY scene: this object supplies collision itself, per candidate.
        self.scene = Scene(cfg.arm, dataclasses.replace(cfg.scene, obstacles=()))

        pm = cfg.pagemap
        self.y_grid = np.linspace(pm.y_min, pm.y_max, pm.n_y)
        self.theta_grid = np.linspace(pm.theta_min, pm.theta_max, pm.n_theta)
        self.shape = (pm.n_y, pm.n_theta)

        self.ok = np.zeros(self.shape, dtype=bool)
        self.limit_ok = {page_name(s): np.zeros(self.shape, dtype=bool)
                         for s in SIGN_BRANCHES}
        self.elbow = np.full(self.shape + (3,), np.nan)
        self.q_ref = np.full(self.shape + (7,), np.nan)

        ts = np.linspace(0.0, 1.0, self.scene.n_coarse)[None, :, None]
        pts, idx, self.radii = [], [], None
        for i, y in enumerate(self.y_grid):
            pos, rot = self.weld.pose(float(y))
            for j, th in enumerate(self.theta_grid):
                res = self.arm.ik_branch(
                    pos, rot, float(th), REF_BRANCH,
                    coord_axis_tol=pm.coord_axis_tol,
                    coord_radius_tol=pm.coord_radius_tol)
                if not res.ok:
                    continue
                self.ok[i, j] = True
                self.q_ref[i, j] = res.q
                for s in SIGN_BRANCHES:
                    q = res.q if s == REF_BRANCH else q_for_branch(res.q, REF_BRANCH, s)
                    self.limit_ok[page_name(s)][i, j] = self.arm.limit_margin(q) >= 0.0
                frames = self.arm.frames(res.q)
                self.elbow[i, j] = frames["elbow"]
                caps = self.scene.capsules(frames)
                a = np.array([c[0] for c in caps])
                b = np.array([c[1] for c in caps])
                pts.append(a[:, None, :] + ts * (b - a)[:, None, :])
                idx.append((i, j))
                if self.radii is None:
                    self.radii = np.array([c[2] for c in caps])
        # (n_reachable, n_capsules, n_samples, 3)
        self.pts = np.asarray(pts) if pts else np.zeros((0, 0, 0, 3))
        self.idx = np.asarray(idx) if idx else np.zeros((0, 2), dtype=int)

    # -- collision ---------------------------------------------------------

    @staticmethod
    def _boxes(obstacles) -> list:
        out = []
        for o in obstacles:
            if isinstance(o, BoxObstacle):
                out.append((np.asarray(o.center, float),
                            np.asarray(o.half_extent, float)))
            else:
                out.append((np.asarray(o[0], float), np.asarray(o[1], float)))
        return out

    def collide(self, obstacles) -> np.ndarray:
        """Collision mask over the grid. Matches Scene.check's sampled test."""
        out = np.zeros(self.shape, dtype=bool)
        if not len(self.pts):
            return out
        hit = np.zeros(len(self.pts), dtype=bool)
        for center, half in self._boxes(obstacles):
            delta = np.abs(self.pts - center) - half
            dist = np.linalg.norm(np.maximum(delta, 0.0), axis=3)
            hit |= ((dist.min(axis=2) - self.radii) < 0.0).any(axis=1)
        out[self.idx[:, 0], self.idx[:, 1]] = hit
        return out

    def hits_cell(self, obstacles, i: int, j: int) -> bool:
        """Does any obstacle touch the arm at this one cell? True if no IK."""
        k = np.flatnonzero((self.idx[:, 0] == i) & (self.idx[:, 1] == j))
        if not len(k):
            return True
        for center, half in self._boxes(obstacles):
            delta = np.abs(self.pts[k[0]] - center) - half
            dist = np.linalg.norm(np.maximum(delta, 0.0), axis=2)
            if ((dist.min(axis=1) - self.radii) < 0.0).any():
                return True
        return False

    # -- page maps ---------------------------------------------------------

    def page_map(self, page: str, collision: np.ndarray) -> PageMap:
        """A PageMap with the same tags build_page_map would produce.

        Only `tag`, `y_grid`, `theta_grid` and `signs` are filled: those are
        what label_regions and the door finders read. The per-cell diagnostics
        are left NaN, so do NOT hand this to the wall or the facet checks.
        """
        tag = np.full(self.shape, INVALID_NO_IK, dtype=np.int8)
        lim = self.limit_ok[page]
        tag[self.ok & ~lim] = INVALID_JOINT_LIMIT
        tag[self.ok & lim & collision] = INVALID_COLLISION
        tag[self.ok & lim & ~collision] = VALID
        nan = np.full(self.shape, np.nan)
        return PageMap(
            signs=tuple(1 if c == "+" else -1 for c in page),
            y_grid=self.y_grid, theta_grid=self.theta_grid, tag=tag,
            q2=nan, q4=nan, q6=nan, clearance=nan, limit_margin=nan,
            kin_measure=nan,
            near_full=np.zeros(self.shape, dtype=bool),
            near_axis=np.zeros(self.shape, dtype=bool),
            stats={"page": page, "n_cells": int(tag.size),
                   "n_valid": int(np.sum(tag == VALID)),
                   "frac_valid": float(np.mean(tag == VALID))},
        )

    def all_pages(self, collision: np.ndarray) -> dict:
        return {page_name(s): self.page_map(page_name(s), collision)
                for s in SIGN_BRANCHES}

    # -- the asymmetry band, which is what scene searches are really after --

    def asymmetry_band(self, page: str, partner: str,
                       collision: np.ndarray | None = None) -> np.ndarray:
        """Cells valid on `partner` but limit-invalid on `page`.

        This is the ONLY thing one page of an s-pair has that the other does
        not, because collision is page-independent. Any scene where `page`
        cannot reach the goal and `partner` can must route the partner's
        corridor through this band (findings 34/35).
        """
        coll = np.zeros(self.shape, bool) if collision is None else collision
        return (self.ok & ~coll & self.limit_ok[partner]
                & ~self.limit_ok[page])

    def band_thickness(self, page: str, partner: str) -> dict:
        """How separable the band is from the cells an obstacle would remove.

        For each band cell, the largest link-point displacement to its nearest
        posture that is valid on `page`. A band cell closer than the thinnest
        capsule radius CANNOT be spared by an obstacle that removes its
        neighbour -- the two are the same arm in space. Measured on three
        placements (findings 35): median 17.33 mm against a 60.0 mm radius,
        with 0 to 4 of ~330 cells separable, and those isolated and sitting at
        the weld's endpoints. Use this as a cheap pre-filter: a placement whose
        separable cells do not form a connected run cannot support EXP 4.
        """
        band = self.asymmetry_band(page, partner)
        keep = self.ok & self.limit_ok[page]
        link = self.pts[:, :, [0, -1], :].reshape(len(self.pts), -1)
        in_band = np.zeros(len(self.pts), bool)
        in_keep = np.zeros(len(self.pts), bool)
        for n, (i, j) in enumerate(self.idx):
            in_band[n] = band[i, j]
            in_keep[n] = keep[i, j]
        PB, PA = link[in_band], link[in_keep]
        r_min = float(min(self.cfg.arm.r_upper_arm, self.cfg.arm.r_forearm))
        if not len(PB) or not len(PA):
            return {"n_band": int(len(PB)), "n_keep": int(len(PA)),
                    "r_min": r_min, "nearest": np.zeros(0), "n_separable": 0}
        d = np.abs(PB[:, None, :] - PA[None, :, :]).reshape(
            len(PB), len(PA), -1, 3)
        nearest = np.linalg.norm(d, axis=3).max(axis=2).min(axis=1)
        return {"n_band": int(len(PB)), "n_keep": int(len(PA)), "r_min": r_min,
                "nearest": nearest, "n_separable": int((nearest > r_min).sum())}
