"""The 1D weld task: a parametric seam with a prescribed tool orientation.

The task set is the manifold M of [IROS]: the tool tip must lie on the weld
seam and the tool orientation is fully specified, so the allowed pose set has
dimension k = 1 and the valid set in Q has dimension n - (6 - k) = 7 - 5 = 2,
parameterised by (y, psi).

Two seam shapes, selected by WeldParams.curve:

  "line"  p(y) = p0 + y (p1 - p0), orientation constant.  EXP 1-4 use this.

  "arc"   p(y) on a circular arc, which is what EXP 5 needs and why this file
          is no longer a straight line. The reason is invariant 22: with a
          straight seam and a constant tool pose the wrist centre

              W(y) = p0 + y (p1 - p0) - d_wt (R e_z)

          is AFFINE in y, so |W(y) - S| is CONVEX, so {y : |W - S| <= reach_max}
          is a single interval and |W - S| = reach_max -- the q4 fold, where the
          elbow circle collapses and the whole theta line degenerates to one
          posture -- can only occur at an endpoint of that interval, i.e. where
          the task leaves the workspace. Measured over 2280 straight placements:
          0 non-convex. A q4 fold door therefore cannot exist in the interior of
          a straight-seam task, and no amount of searching changes that.

          An arc breaks the affinity. On a circle of centre C and radius R in a
          plane with unit normal n,

              |W(phi) - S|^2 = A - B cos(phi - phi_near),   B = 2 R rho >= 0

          with rho the in-plane distance from C to the projection of S + t. So
          |W - S| rises monotonically from the near point to the FAR point of
          the arc and falls again: it has an interior maximum. Placing the seam
          so that this maximum equals reach_max exactly puts a fold door in the
          middle of the task with both sides reachable. That placement is not a
          contrivance -- it is the limiting workcell layout in which the arm
          just reaches the farthest point of the seam.

Everything downstream reads the task through pose() and wrist_center() only, so
the seam shape is a local change. doors.py is the one module that also needed
work: its door conditions were solved as quadratics in y, which relied on the
affinity, and are now solved by numerical root finding.
"""

from __future__ import annotations

import numpy as np

from .params import WeldParams

CURVE_LINE = "line"
CURVE_ARC = "arc"

POSE_CONSTANT = "constant"
POSE_RADIAL = "radial"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("zero-length direction in the weld definition")
    return v / n


def _frame_from_axes(z_axis, reference) -> np.ndarray:
    """Right-handed rotation whose z column is `z_axis`, roll fixed by `reference`."""
    z = _unit(z_axis)
    ref = np.asarray(reference, dtype=float)
    x = ref - float(np.dot(ref, z)) * z
    if float(np.linalg.norm(x)) < 1e-12:
        # reference parallel to z: pick any perpendicular, deterministically
        alt = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(alt, z))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        x = alt - float(np.dot(alt, z)) * z
    x = _unit(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


class WeldLine:
    """Maps the weld coordinate y in [0, 1] to a tool pose.

    The class name is kept because it is the task object the whole pipeline
    imports; the seam itself may be a line or an arc.
    """

    def __init__(self, p: WeldParams):
        self.p = p
        self.curve = getattr(p, "curve", CURVE_LINE)
        self.pose_mode = getattr(p, "pose_mode", POSE_CONSTANT)

        self.p0 = np.asarray(p.p0, dtype=float)
        self.p1 = np.asarray(p.p1, dtype=float)

        # Tool z-axis points along `approach`; `reference` fixes the roll.
        self.rot = _frame_from_axes(p.approach, p.reference)

        if self.curve == CURVE_ARC:
            self.center = np.asarray(p.arc_center, dtype=float)
            self.radius = float(p.arc_radius)
            self.normal = _unit(p.arc_normal)
            ref = np.asarray(p.arc_ref, dtype=float)
            e1 = ref - float(np.dot(ref, self.normal)) * self.normal
            self.e1 = _unit(e1)
            self.e2 = np.cross(self.normal, self.e1)
            self.phi0 = float(p.arc_phi0)
            self.phi1 = float(p.arc_phi1)
            self.dphi = self.phi1 - self.phi0
            self.direction = self.position(1.0) - self.position(0.0)
            self.length = float(np.linalg.norm(self.direction))
        elif self.curve == CURVE_LINE:
            self.direction = self.p1 - self.p0
            self.length = float(np.linalg.norm(self.direction))
        else:
            raise ValueError(f"unknown weld curve type {self.curve!r}")

    # -- the seam ----------------------------------------------------------

    def phi(self, y: float) -> float:
        return self.phi0 + float(y) * self.dphi

    def position(self, y: float) -> np.ndarray:
        if self.curve == CURVE_ARC:
            a = self.phi(y)
            return (self.center
                    + self.radius * (np.cos(a) * self.e1 + np.sin(a) * self.e2))
        return self.p0 + float(y) * self.direction

    def tangent(self, y: float) -> np.ndarray:
        """dp/dy. Constant for a line; rotates along an arc."""
        if self.curve == CURVE_ARC:
            a = self.phi(y)
            return self.radius * self.dphi * (-np.sin(a) * self.e1
                                              + np.cos(a) * self.e2)
        return self.direction

    def orientation(self, y: float) -> np.ndarray:
        """Tool rotation at y.

        "constant" holds the torch fixed in the world (EXP 1-4, and EXP 5 as
        run: it keeps the only non-affinity in the seam itself, so the fold
        door's cause is unambiguous).

        "radial" aims the torch at the arc's axis, which is what a real ring
        weld does; it rotates with the seam and so adds a second source of
        non-affinity. Available, not used by the reported experiments.
        """
        if self.curve == CURVE_ARC and self.pose_mode == POSE_RADIAL:
            a = self.phi(y)
            outward = np.cos(a) * self.e1 + np.sin(a) * self.e2
            # Keep the torch's tilt away from the surface normal, but swing its
            # azimuth with the seam: build the frame from the stored approach
            # expressed in the local (outward, normal) basis.
            t_local = self.rot @ np.array([0.0, 0.0, 1.0])
            # Decompose the stored approach in the global frame used at phi0.
            a0 = self.phi0
            out0 = np.cos(a0) * self.e1 + np.sin(a0) * self.e2
            tan0 = -np.sin(a0) * self.e1 + np.cos(a0) * self.e2
            c_out = float(np.dot(t_local, out0))
            c_tan = float(np.dot(t_local, tan0))
            c_n = float(np.dot(t_local, self.normal))
            tangent = -np.sin(a) * self.e1 + np.cos(a) * self.e2
            z_axis = c_out * outward + c_tan * tangent + c_n * self.normal
            return _frame_from_axes(z_axis, tangent)
        return self.rot

    def pose(self, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose at weld coordinate y. The single entry point downstream."""
        return self.position(y), self.orientation(y)

    def wrist_center(self, y: float, d_wt: float) -> np.ndarray:
        """Wrist centre implied by the pose."""
        pos, rot = self.pose(y)
        return pos - d_wt * (rot @ np.array([0.0, 0.0, 1.0]))

    # -- Lipschitz data used by the penetration-depth routines -------------

    @property
    def speed_max(self) -> float:
        """Bound on |dW/dy|, i.e. how fast the wrist centre moves in y.

        wall._no_ik_margin converts an out-of-reach excess in metres into a
        distance in y with this. For a line it is |p1 - p0| (the old
        `length`); for an arc with a constant tool pose it is R |dphi|, which
        is also constant, so the bound stays tight rather than conservative.
        """
        if self.curve == CURVE_ARC and self.pose_mode == POSE_CONSTANT:
            return abs(self.radius * self.dphi)
        if self.curve == CURVE_LINE:
            return self.length
        # Radial pose: sample the derivative; cheap and only used as a bound.
        ys = np.linspace(0.0, 1.0, 201)
        h = 1e-6
        best = 0.0
        for y in ys:
            a = self.wrist_center(min(1.0, y + h), self.p.__dict__.get("d_wt", 0.0))
            b = self.wrist_center(max(0.0, y - h), self.p.__dict__.get("d_wt", 0.0))
            best = max(best, float(np.linalg.norm(a - b)) / (2.0 * h))
        return max(best, 1e-9)

    def polyline(self, n: int = 121) -> np.ndarray:
        """The seam as a polyline, for plotting."""
        return np.array([self.position(float(y)) for y in np.linspace(0.0, 1.0, n)])
