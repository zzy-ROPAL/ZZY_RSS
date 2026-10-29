"""Forward and analytic inverse kinematics for a 3R-1R-3R (S-R-S) arm.

Convention
----------
Shoulder point S = (0, 0, d_bs). Joint axes, each expressed in the frame of
the preceding joint:

    q1 about z, q2 about y, q3 about z      (shoulder, spherical)
    q4 about y                              (elbow)
    q5 about z, q6 about y, q7 about z      (wrist, spherical)

Upper arm runs from S along the frame-2 z-axis for d_se; the forearm runs from
the elbow along the frame-4 z-axis for d_ew; the tool runs from the wrist along
the frame-7 z-axis for d_wt.

Branch structure
----------------
Given a tool pose and an arm angle psi, the IK has eight solutions indexed by
(s2, s4, s6) = (sign q2, sign q4, sign q6):

    s2:  (q1, q2, q3)    -> (q1 + pi, -q2, q3 + pi)
    s4:  (q3, q4, q5)    -> (q3 + pi, -q4, q5 + pi)
    s6:  (q5, q6, q7)    -> (q5 + pi, -q6, q7 + pi)

All eight place the shoulder, elbow, wrist and tool identically in space. They
differ only in joint values and in the roll of each link about its own axis,
so they are distinguished by joint limits and by any off-axis link geometry.

Degeneracies (these are the Sigma set of the proof):
    q2 = 0  -> upper arm along the joint-1 axis, q1 and q3 counter-rotate
    q4 = 0  -> full extension, elbow circle collapses, psi undefined, q3 free
    q6 = 0  -> wrist gimbal, q5 and q7 counter-rotate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import ArmParams

# Invalidity tags. The tag selects which penetration-depth routine the facet
# check calls, so it is part of the proof pipeline, not diagnostics.
VALID = 0
INVALID_NO_IK = 1
INVALID_JOINT_LIMIT = 2
INVALID_COLLISION = 3

TAG_NAMES = {
    VALID: "valid",
    INVALID_NO_IK: "no-IK",
    INVALID_JOINT_LIMIT: "joint-limit",
    INVALID_COLLISION: "collision",
}

SIGN_BRANCHES = tuple(
    (s2, s4, s6) for s2 in (+1, -1) for s4 in (+1, -1) for s6 in (+1, -1)
)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class IkResult:
    """One IK branch evaluation.

    `ok` is False when no configuration exists for this (pose, psi, signs);
    `reason` then carries the geometric cause.
    """

    ok: bool
    q: np.ndarray | None = None
    reason: str = ""
    # Degeneracy flags, used by the Sigma detector.
    near_full_extension: bool = False
    near_axis_align: bool = False
    elbow_circle_radius: float = 0.0
    cos_q4: float = 0.0


class SrsArm:
    """Analytic kinematics for the S-R-S arm described in the module docstring."""

    def __init__(self, p: ArmParams):
        self.p = p
        self.shoulder = np.array([0.0, 0.0, p.d_bs])
        self.q_min = np.asarray(p.q_min, dtype=float)
        self.q_max = np.asarray(p.q_max, dtype=float)

    # -- forward kinematics ------------------------------------------------

    def frames(self, q: np.ndarray) -> dict:
        """Joint-frame origins and orientations needed by FK and collision."""
        p = self.p
        r02 = rot_z(q[0]) @ rot_y(q[1])
        r03 = r02 @ rot_z(q[2])
        r04 = r03 @ rot_y(q[3])
        r07 = r04 @ rot_z(q[4]) @ rot_y(q[5]) @ rot_z(q[6])

        e_z = np.array([0.0, 0.0, 1.0])
        elbow = self.shoulder + p.d_se * (r02 @ e_z)
        wrist = elbow + p.d_ew * (r04 @ e_z)
        tool = wrist + p.d_wt * (r07 @ e_z)
        return {
            "shoulder": self.shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "tool": tool,
            "R02": r02,
            "R03": r03,
            "R04": r04,
            "R07": r07,
        }

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose as (position, rotation)."""
        f = self.frames(q)
        return f["tool"], f["R07"]

    # -- inverse kinematics ------------------------------------------------

    def ik_branch(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        psi: float,
        signs: tuple[int, int, int],
        coord_axis_tol: float = 1e-6,
        coord_radius_tol: float = 1e-6,
    ) -> IkResult:
        """Solve one sign branch for a tool pose and arm angle.

        Returns IkResult.ok == False when the pose is out of reach for this
        branch, which is the `no-IK` invalidity of the page map.
        """
        p = self.p
        s2, s4, s6 = signs
        e_z = np.array([0.0, 0.0, 1.0])

        # 1. Wrist centre from the tool pose.
        wrist = pos - p.d_wt * (rot @ e_z)
        x_sw = wrist - self.shoulder
        length = float(np.linalg.norm(x_sw))

        # 2. Elbow bend from the law of cosines. |q4| = 0 is full extension.
        if length < 1e-12:
            return IkResult(False, reason="wrist at shoulder")
        cos_q4 = (length**2 - p.d_se**2 - p.d_ew**2) / (2.0 * p.d_se * p.d_ew)
        if cos_q4 > 1.0 or cos_q4 < -1.0:
            return IkResult(
                False,
                reason="out of reach" if cos_q4 > 1.0 else "inside dead zone",
                cos_q4=cos_q4,
            )
        q4 = s4 * float(np.arccos(np.clip(cos_q4, -1.0, 1.0)))

        # 3. Elbow on its self-motion circle about the shoulder-wrist axis.
        u = x_sw / length
        h = (p.d_se**2 - p.d_ew**2 + length**2) / (2.0 * length)
        r_sq = p.d_se**2 - h**2
        radius = float(np.sqrt(max(0.0, r_sq)))

        # Arm-angle reference plane: fix it with world z. The reference breaks
        # down when the shoulder-wrist axis aligns with z (Sigma_coord).
        axis_align = abs(float(np.dot(u, e_z)))
        near_axis = axis_align > 1.0 - coord_axis_tol
        ref = np.array([1.0, 0.0, 0.0]) if near_axis else e_z
        v0 = ref - float(np.dot(ref, u)) * u
        n_v0 = float(np.linalg.norm(v0))
        if n_v0 < 1e-12:
            return IkResult(False, reason="degenerate arm-angle reference")
        v0 = v0 / n_v0
        w0 = np.cross(u, v0)

        elbow = self.shoulder + h * u + radius * (np.cos(psi) * v0 + np.sin(psi) * w0)

        # 4. Shoulder angles from the upper-arm direction. q2 = 0 means the
        #    upper arm lies on the joint-1 axis, where q1 is undefined.
        d_u = elbow - self.shoulder
        n_du = float(np.linalg.norm(d_u))
        if n_du < 1e-12:
            return IkResult(False, reason="elbow at shoulder")
        d_u = d_u / n_du
        q2 = s2 * float(np.arccos(np.clip(d_u[2], -1.0, 1.0)))
        if s2 >= 0:
            q1 = float(np.arctan2(d_u[1], d_u[0]))
        else:
            q1 = float(np.arctan2(-d_u[1], -d_u[0]))

        # 5. q3 orients the elbow axis perpendicular to the arm plane. At full
        #    extension the plane is undefined, so q3 is free; pick 0.
        d_f = wrist - elbow
        n_df = float(np.linalg.norm(d_f))
        near_full = radius < coord_radius_tol or abs(cos_q4) > 1.0 - 1e-12
        if n_df < 1e-12:
            return IkResult(False, reason="wrist at elbow")
        d_f = d_f / n_df
        cross = np.cross(d_u, d_f)
        n_cross = float(np.linalg.norm(cross))
        if n_cross < 1e-9:
            q3 = 0.0
            near_full = True
        else:
            n_target = s4 * (cross / n_cross)
            r02 = rot_z(q1) @ rot_y(q2)
            m = r02.T @ n_target
            q3 = float(np.arctan2(-m[0], m[1]))

        # 6. Wrist angles from the residual rotation. q6 = 0 is the gimbal.
        r04 = rot_z(q1) @ rot_y(q2) @ rot_z(q3) @ rot_y(q4)
        r47 = r04.T @ rot
        c6 = float(np.clip(r47[2, 2], -1.0, 1.0))
        q6 = s6 * float(np.arccos(c6))
        if s6 >= 0:
            q5 = float(np.arctan2(r47[1, 2], r47[0, 2]))
            q7 = float(np.arctan2(r47[2, 1], -r47[2, 0]))
        else:
            q5 = float(np.arctan2(-r47[1, 2], -r47[0, 2]))
            q7 = float(np.arctan2(-r47[2, 1], r47[2, 0]))

        q = np.array([q1, q2, q3, q4, q5, q6, q7], dtype=float)
        return IkResult(
            True,
            q=q,
            near_full_extension=bool(near_full),
            near_axis_align=bool(near_axis),
            elbow_circle_radius=radius,
            cos_q4=float(cos_q4),
        )

    # -- joint limits ------------------------------------------------------

    def within_limits(self, q: np.ndarray) -> bool:
        return bool(np.all(q >= self.q_min) and np.all(q <= self.q_max))

    def limit_margin(self, q: np.ndarray) -> float:
        """Signed slack to the nearest joint limit; negative means violated."""
        return float(np.min(np.minimum(q - self.q_min, self.q_max - q)))

    # -- differential ------------------------------------------------------

    def jacobian(self, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """6x7 geometric Jacobian by central differences on the tool pose.

        Finite differences are adequate here: the Jacobian is used only to
        locate Sigma_kin by the vanishing of det(J J^T).
        """
        pos0, rot0 = self.fk(q)
        jac = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            p_plus, r_plus = self.fk(q + dq)
            p_minus, r_minus = self.fk(q - dq)
            jac[0:3, i] = (p_plus - p_minus) / (2.0 * eps)
            d_rot = (r_plus - r_minus) / (2.0 * eps) @ rot0.T
            jac[3:6, i] = np.array([d_rot[2, 1], d_rot[0, 2], d_rot[1, 0]])
        return jac

    def kin_singularity_measure(self, q: np.ndarray) -> float:
        jac = self.jacobian(q)
        return float(np.linalg.det(jac @ jac.T))
