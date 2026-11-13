"""Solve the generic joint-variable door systems by monodromy and cache them.

  python3 scripts/joint_monodromy.py --out tests/data/joint_monodromy.npz

The cache holds, for the q2 and q6 door systems of the LWA 4D, the generic
base instance P0 (generic task AND generic hyperplane nu.x = rd in place of
the door equation) and its witness points, certified complete by the trace
test. tests/test_algebraic_joint.py
re-checks each cached solution against F(., P0) before using it, so the cache
cannot smuggle in a wrong root; it only saves the monodromy time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zzy_rss.algebraic_joint import DoorSystem, TrackStats, monodromy_certified  # noqa: E402
from zzy_rss.arm import SrsArm  # noqa: E402
from zzy_rss.krawczyk import SerialChain  # noqa: E402
from zzy_rss.params import Config  # noqa: E402

SYSTEMS = (("q2", 1, 0), ("q6", 5, 4))     # name, label joint, spin joint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests/data/joint_monodromy.npz"))
    args = ap.parse_args()
    chain = SerialChain.srs(SrsArm(Config().arm))
    data = {}
    for name, label, spin in SYSTEMS:
        t0 = time.time()
        st = TrackStats()
        m = monodromy_certified(DoorSystem(chain, label, spin), verbose=True, stats=st)
        print(f"[joint-mono] {name}: {len(m.solutions)} witness points, {m.loops} loops, "
              f"trace residual {m.trace.residual:.1e} ({'COMPLETE' if m.trace.ok else 'NOT certified'}), "
              f"{st.paths} paths tracked, {time.time() - t0:.0f} s", flush=True)
        data[f"{name}_sols"] = np.array(m.solutions)
        for k, val in m.base.items():
            data[f"{name}_P_{k}"] = np.asarray(val)
        data[f"{name}_history"] = np.array(m.history)
        data[f"{name}_trace"] = np.array([m.trace.residual, float(m.trace.ok)])
    np.savez(args.out, **data)
    print(f"[joint-mono] saved {args.out}")


if __name__ == "__main__":
    main()
