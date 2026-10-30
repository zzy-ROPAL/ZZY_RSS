"""IK round-trip and branch-structure tests (Stage 0 acceptance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zzy_rss.arm import SIGN_BRANCHES, SrsArm, rot_y, rot_z  # noqa: E402
from zzy_rss.params import DEFAULT  # noqa: E402
from zzy_rss.weld import WeldLine  # noqa: E402


def test_round_trip() -> tuple[int, int, float]:
    """FK(IK(pose, psi, signs)) must reproduce the pose on every branch."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(7)

    n_ok = 0
    n_tried = 0
    worst = 0.0
    for _ in range(400):
        y = float(rng.uniform(0.0, 1.0))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            n_tried += 1
            if not res.ok:
                continue
            pos_fk, rot_fk = arm.fk(res.q)
            e_pos = float(np.linalg.norm(pos_fk - pos))
            e_rot = float(np.linalg.norm(rot_fk - rot))
            err = max(e_pos, e_rot)
            worst = max(worst, err)
            assert err < 1e-8, f"round-trip failed: {err:.3e} at y={y} psi={psi} {signs}"
            n_ok += 1
    return n_ok, n_tried, worst


def test_sign_consistency() -> float:
    """The requested signs must match the signs of the returned joint angles."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        y = float(rng.uniform(0.05, 0.95))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            if not res.ok:
                continue
            q = res.q
            for idx, want in zip((1, 3, 5), signs):
                got = q[idx]
                if abs(got) < 1e-7:
                    continue
                assert np.sign(got) == want, (
                    f"sign mismatch at q{idx + 1}: wanted {want}, got {got:.4f}"
                )
            worst = max(worst, abs(q[1]), abs(q[3]), abs(q[5]))
    return worst


def test_branch_geometry_identical() -> float:
    """All eight branches must place shoulder, elbow, wrist and tool alike."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(100):
        y = float(rng.uniform(0.05, 0.95))
        psi = float(rng.uniform(-np.pi, np.pi))
        pos, rot = weld.pose(y)
        pts = []
        for signs in SIGN_BRANCHES:
            res = arm.ik_branch(pos, rot, psi, signs)
            if res.ok:
                f = arm.frames(res.q)
                pts.append(np.concatenate([f["elbow"], f["wrist"], f["tool"]]))
        if len(pts) > 1:
            ref = pts[0]
            for other in pts[1:]:
                worst = max(worst, float(np.linalg.norm(other - ref)))
    assert worst < 1e-8, f"branches place geometry differently: {worst:.3e}"
    return worst


def test_degeneracy_geometry() -> None:
    """q2 = 0 must pin the elbow; q4 = 0 must be full extension."""
    arm = SrsArm(DEFAULT.arm)
    p = DEFAULT.arm

    # q2 = 0: elbow sits on the joint-1 axis at height d_bs + d_se,
    # independently of q1 and q3.
    for q1 in (-1.0, 0.0, 2.0):
        for q3 in (-2.0, 0.5, 3.0):
            q = np.array([q1, 0.0, q3, 1.0, 0.2, 0.7, -0.3])
            f = arm.frames(q)
            expected = np.array([0.0, 0.0, p.d_bs + p.d_se])
            assert np.linalg.norm(f["elbow"] - expected) < 1e-12

    # q2 = 0: the gimbal arc leaves the tool pose fixed when q1 + q3 is held.
    base = np.array([0.4, 0.0, -0.9, 1.1, 0.3, 0.8, -0.2])
    pos0, rot0 = arm.fk(base)
    for delta in (0.3, -0.7, 1.9):
        q = base.copy()
        q[0] += delta
        q[2] -= delta
        pos1, rot1 = arm.fk(q)
        assert np.linalg.norm(pos1 - pos0) < 1e-12, "gimbal arc moved the tool"
        assert np.linalg.norm(rot1 - rot0) < 1e-12, "gimbal arc rotated the tool"

    # q4 = 0: shoulder-to-wrist distance equals full reach.
    q = np.array([0.3, 0.6, -0.4, 0.0, 0.1, 0.2, 0.3])
    f = arm.frames(q)
    assert abs(np.linalg.norm(f["wrist"] - f["shoulder"]) - p.reach_max) < 1e-12

    # q6 = 0: the wrist gimbal leaves the tool pose fixed when q5 + q7 is held.
    base = np.array([0.2, 0.9, -0.5, 1.0, 0.6, 0.0, -0.4])
    pos0, rot0 = arm.fk(base)
    for delta in (0.4, -1.1):
        q = base.copy()
        q[4] += delta
        q[6] -= delta
        pos1, rot1 = arm.fk(q)
        assert np.linalg.norm(pos1 - pos0) < 1e-12, "wrist gimbal moved the tool"
        assert np.linalg.norm(rot1 - rot0) < 1e-12, "wrist gimbal rotated the tool"


def test_mirror_identities() -> None:
    """s2/s4/s6 act as the stated representational mirrors."""
    arm = SrsArm(DEFAULT.arm)
    weld = WeldLine(DEFAULT.weld)
    pos, rot = weld.pose(0.5)
    psi = 0.8

    pairs = [
        ((+1, +1, +1), (-1, +1, +1), (0, 1, 2)),
        ((+1, +1, +1), (+1, -1, +1), (2, 3, 4)),
        ((+1, +1, +1), (+1, +1, -1), (4, 5, 6)),
    ]
    for a, b, idx in pairs:
        ra = arm.ik_branch(pos, rot, psi, a)
        rb = arm.ik_branch(pos, rot, psi, b)
        if not (ra.ok and rb.ok):
            continue
        i, j, k = idx
        # Middle angle negates; the outer two shift by pi (mod 2 pi).
        assert abs(rb.q[j] + ra.q[j]) < 1e-9, f"middle angle not negated for {b}"
        for m in (i, k):
            d = (rb.q[m] - ra.q[m]) % (2.0 * np.pi)
            assert min(abs(d - np.pi), abs(d + np.pi - 2 * np.pi)) < 1e-9, (
                f"outer angle q{m + 1} not shifted by pi for {b}: delta={d:.6f}"
            )


if __name__ == "__main__":
    n_ok, n_tried, worst_rt = test_round_trip()
    print(f"[ik] round-trip        : {n_ok}/{n_tried} branches solvable, "
          f"max pose error {worst_rt:.3e}")
    worst_sign = test_sign_consistency()
    print(f"[ik] sign consistency  : ok (max |q2|,|q4|,|q6| seen {worst_sign:.3f})")
    worst_geo = test_branch_geometry_identical()
    print(f"[ik] branch geometry   : identical across branches, max dev {worst_geo:.3e}")
    test_degeneracy_geometry()
    print("[ik] degeneracies      : q2=0 pins elbow, q4=0 full extension, "
          "gimbal arcs fix the tool")
    test_mirror_identities()
    print("[ik] mirror identities : s2/s4/s6 confirmed as representational mirrors")
    print("[ik] ALL TESTS PASSED")
