"""The 1D weld task: a straight line with constant tool orientation.

The task set is the manifold M of [IROS]: the tool tip must lie on the weld
line and the tool orientation is fully specified, so the allowed pose set has
dimension k = 1 and the valid set in Q has dimension n - (6 - k) = 7 - 5 = 2,
parameterised by (y, psi).
"""

from __future__ import annotations

import numpy as np

from .params import WeldParams


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
