"""Fail-fast budgets and escalation (findings 44).

The proof machinery has two ways to spend an hour saying nothing: Algorithm 1
scanning gamma upward 0.1 at a time to gamma_max = 400 on classes that are not
separable (one SVM fit per step), and Algorithm 4 halving the net spacing on
a wall that never closes (each halving quadruples the points at d = 3). Both
end in INCONCLUSIVE anyway. This module stops them early and spends the time
on a better-posed attempt instead.

    with run_budget(cfg.budget, "prove3"):          # a whole proof run
        with wall_attempt(cfg.budget, "++-/1"):      # one region's wall
            ... check_gamma(g) / check_time() / note_unclosed() ...

The hooks raise BudgetExceeded. Where it is caught decides what it means, and
there is exactly one rule: a stop is a FAILED attempt, never a skipped one.

  * a wall attempt that is stopped is reported as a wall that failed
    verification ("stopped by budget: ..."), with its region still in the
    certificate -- so the run can only be INCONCLUSIVE;
  * a run that is stopped returns INCONCLUSIVE with that reason;
  * escalate() re-runs the whole proof with more samples, a new seed and a
    larger budget, and returns the first conclusive verdict. Each rung is an
    independent proof (its own roadmap, walls and checks), so a conclusive
    rung is exactly as trustworthy as an unbudgeted run.

With no active budget (BudgetParams.enabled = False, or code outside any
`with`), every hook is a no-op.
"""

from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .params import BudgetParams

STOP = "stopped by budget"

# INCONCLUSIVE reasons that more samples cannot change: the task itself is
# outside what the prover supports, or the answer needs another mechanism.
STRUCTURAL = ("unsupported singular set", "no valid start posture",
              "bounded theta not justified", "live q")


class BudgetExceeded(Exception):
    """A budget ran out. `scope` is "wall" (one region's wall attempt) or
    "run" (the whole proof, or the escalation total)."""

    def __init__(self, scope: str, label: str, why: str, gamma: float | None = None):
        super().__init__(f"{STOP}: {label}: {why}")
        self.scope, self.label, self.why, self.gamma = scope, label, why, gamma

    @property
    def reason(self) -> str:
        return str(self)


@dataclass
class _Frame:
    scope: str
    label: str
    seconds: float
    t0: float = field(default_factory=time.time)
    gamma_stop: float = float("inf")
    max_unclosed: int = 0
    unclosed: int = 0
    gamma: float | None = None


_STACK: list = []


def active() -> bool:
    return bool(_STACK)


@contextmanager
def run_budget(bp: BudgetParams | None, label: str):
    """Time budget for a whole proof run. Nests: an escalation total and a
    rung's own run budget are both checked."""
    if bp is None or not bp.enabled:
        yield None
        return
    f = _Frame("run", label, float(bp.run_seconds))
    _STACK.append(f)
    try:
        yield f
    finally:
        _STACK.remove(f)


@contextmanager
def wall_attempt(bp: BudgetParams | None, label: str):
    """Budget for one region's wall (Algorithms 1-4)."""
    if bp is None or not bp.enabled:
        yield None
        return
    f = _Frame("wall", label, float(bp.wall_seconds), gamma_stop=float(bp.gamma_stop),
               max_unclosed=int(bp.max_unclosed))
    _STACK.append(f)
    try:
        yield f
    finally:
        _STACK.remove(f)


def _wall() -> _Frame | None:
    for f in reversed(_STACK):
        if f.scope == "wall":
            return f
    return None


def check_time() -> None:
    """Raise if any active budget's clock has run out (outermost first, so
    an exhausted run is reported as the run, not as the wall inside it)."""
    now = time.time()
    for f in _STACK:
        if now - f.t0 > f.seconds:
            raise BudgetExceeded(f.scope, f.label,
                                 f"{f.scope} time {now - f.t0:.0f} s > {f.seconds:g} s",
                                 gamma=_wall().gamma if _wall() else None)


def check_gamma(gamma: float) -> None:
    """Algorithm 1 is about to fit at this gamma. Past gamma_stop the
    classes are effectively not separable at a usable kernel width."""
    f = _wall()
    if f is None:
        return
    f.gamma = float(gamma)
    if gamma > f.gamma_stop:
        raise BudgetExceeded("wall", f.label,
                             f"Algorithm 1 needs gamma > {f.gamma_stop:g}", gamma=gamma)
    check_time()


def note_unclosed(note: str = "") -> None:
    f = _wall()
    if f is None:
        return
    f.unclosed += 1
    if f.max_unclosed and f.unclosed >= f.max_unclosed:
        raise BudgetExceeded("wall", f.label,
                             f"{f.unclosed} reconstructions in a row did not close"
                             + (f" ({note})" if note else ""), gamma=f.gamma)
    check_time()


def note_closed() -> None:
    f = _wall()
    if f is not None:
        f.unclosed = 0


# --------------------------------------------------------------------------
# escalation
# --------------------------------------------------------------------------


def rung_params(bp: BudgetParams, rung: int, total_left: float) -> BudgetParams:
    """The budget of escalation rung `rung` (0-based): per-wall budgets grow
    by escalate_budget_factor per rung; the run budget never exceeds what
    is left of the escalation total."""
    g = float(bp.escalate_budget_factor) ** rung
    return dataclasses.replace(bp, gamma_stop=bp.gamma_stop * g,
                               wall_seconds=bp.wall_seconds * g,
                               run_seconds=min(bp.run_seconds * g, max(1.0, total_left)))


def retryable(verdict: str, reason: str) -> bool:
    if verdict != "INCONCLUSIVE":
        return False
    return not any(reason.startswith(s) for s in STRUCTURAL)


def escalate(attempt, bp: BudgetParams, base_samples: int, base_seed: int,
             verbose: bool = True):
    """Run `attempt(n_samples, seed, budget) -> result` (result has .verdict,
    .reason, .stats) on a ladder of rungs until one is conclusive.

    Returns (result, log): the first FEASIBLE / INFEASIBLE result, or the
    last INCONCLUSIVE one. `log` lists every rung, and is also put in
    result.stats["escalation"]. Without an enabled budget this is a single
    plain attempt."""
    if not bp.enabled:
        res = attempt(int(base_samples), int(base_seed), bp)
        return res, []
    t0 = time.time()
    log = []
    res = None
    for rung in range(int(bp.escalate_rungs)):
        left = float(bp.escalate_total_seconds) - (time.time() - t0)
        if left <= 0:
            break
        n = int(round(base_samples * float(bp.escalate_factor) ** rung))
        seed = int(base_seed + rung * int(bp.escalate_seed_step))
        bpr = rung_params(bp, rung, left)
        t = time.time()
        res = attempt(n, seed, bpr)
        row = dict(rung=rung, samples=n, seed=seed, gamma_stop=bpr.gamma_stop,
                   wall_seconds=bpr.wall_seconds, verdict=res.verdict,
                   reason=res.reason, seconds=round(time.time() - t, 1))
        log.append(row)
        if verbose:
            print(f"[escalate] rung {rung}: {n} samples, seed {seed}, gamma_stop "
                  f"{bpr.gamma_stop:g} -> {res.verdict} ({row['seconds']} s): {res.reason}")
        if not retryable(res.verdict, res.reason):
            break
    if res is not None and isinstance(getattr(res, "stats", None), dict):
        res.stats["escalation"] = log
    return res, log
