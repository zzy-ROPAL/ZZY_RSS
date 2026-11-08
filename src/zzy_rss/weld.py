"""The 1D weld task: a straight line with constant tool orientation.

The task set is the manifold M of [IROS]: the tool tip must lie on the weld
line and the tool orientation is fully specified, so the allowed pose set has
dimension k = 1 and the valid set in Q has dimension n - (6 - k) = 7 - 5 = 2,
parameterised by (y, psi).
"""

from __future__ import annotations

import numpy as np

from .params import SurfaceParams, WeldParams


class WeldLine:
    """Maps the weld coordinate y in [0, 1] to a tool pose."""

    def __init__(self, p: WeldParams):
        self.p = p
        self.p0 = np.asarray(p.p0, dtype=float)
        self.p1 = np.asarray(p.p1, dtype=float)

        # Tool z-axis points along `approach`; `reference` fixes the roll.
        z_axis = np.asarray(p.approach, dtype=float)
        z_axis = z_axis / np.linalg.norm(z_axis)
        ref = np.asarray(p.reference, dtype=float)
        x_axis = ref - float(np.dot(ref, z_axis)) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        self.rot = np.column_stack([x_axis, y_axis, z_axis])

        self.direction = self.p1 - self.p0
        self.length = float(np.linalg.norm(self.direction))

    def position(self, y: float) -> np.ndarray:
        return self.p0 + y * self.direction

    def pose(self, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose at weld coordinate y. Orientation is constant."""
        return self.position(y), self.rot

    def wrist_center(self, y: float, d_wt: float) -> np.ndarray:
        """Wrist centre implied by the pose; traces a line since rot is fixed."""
        return self.position(y) - d_wt * (self.rot @ np.array([0.0, 0.0, 1.0]))


class WeldSurface:
    """Maps (u, v) in [0, 1]^2 to a tool pose on a flat weld patch.

    The task set now has dimension k = 2, so the valid set in Q has dimension
    7 - (6 - 2) = 3, parameterised by (u, v, theta): one arm angle on top of
    the two task coordinates, exactly as [IROS] d = 3 requires. The self-
    motion fiber at a door is still 1-D, because the pose is still fully
    specified at each (u, v).

    Each v-slice is a straight WeldLine (`line(v)`), so everything the 2-D
    pipeline knows about a line -- IK, door geometry, fiber certificates --
    applies slice by slice.
    """

    def __init__(self, p: SurfaceParams):
        self.p = p
        self.p0 = np.asarray(p.p0, dtype=float)
        self.p1 = np.asarray(p.p1, dtype=float)
        self.du = self.p1 - self.p0
        self.dv = np.asarray(p.sweep, dtype=float)
        self._line0 = WeldLine(WeldParams(p0=p.p0, p1=p.p1, approach=p.approach,
                                          reference=p.reference))
        self.rot = self._line0.rot
        self.tool_axis = self.rot @ np.array([0.0, 0.0, 1.0])

    def position(self, u: float, v: float) -> np.ndarray:
        return self.p0 + u * self.du + v * self.dv

    def pose(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        return self.position(u, v), self.rot

    def wrist_center(self, u: float, v: float, d_wt: float) -> np.ndarray:
        return self.position(u, v) - d_wt * self.tool_axis

    def line(self, v: float) -> WeldLine:
        """The straight weld at this v: y on the line is u on the patch."""
        off = v * self.dv
        return WeldLine(WeldParams(p0=tuple(self.p0 + off), p1=tuple(self.p1 + off),
                                   approach=self.p.approach,
                                   reference=self.p.reference))
