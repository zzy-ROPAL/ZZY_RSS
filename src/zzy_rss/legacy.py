"""The old pipeline is legacy (2026-11-15).

The main method is the recursive prover (scripts/run_recursive.py:
check.py + proof.py), whose doors come only from the sigma search with the
guaranteed bound L (chart.py, interior.py, doorset.py -- "method 2"), and
whose door discharges are all certified (collision or joint limits).

The old flood pipeline (flood.py, flood3.py, page3.py, source.py and the
scripts built on them) finds doors with the closed-form formulas in
doors.py / doors3.py and discharges them on sampled evidence. It still runs
-- nothing is deleted -- but its results are not used for new claims. The
closed-form finders stay as the independent answer key in the tests.
"""

from __future__ import annotations

import os
import sys

MAIN = "scripts/run_recursive.py"


def notice(script: str) -> None:
    """One line on stderr when an old-pipeline script starts.
    ZZY_LEGACY_QUIET=1 silences it."""
    if os.environ.get("ZZY_LEGACY_QUIET"):
        return
    name = os.path.basename(script)
    print(f"[legacy] {name} runs the OLD flood pipeline (closed-form doors, evidence "
          f"discharge). The main method is {MAIN} (sigma + L doors, certified "
          f"discharge).", file=sys.stderr)
