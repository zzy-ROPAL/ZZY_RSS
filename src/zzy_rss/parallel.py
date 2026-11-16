"""Run independent pieces of work in forked worker processes.

Used where the proof spends its time on many independent, identical
computations: the winding-number self-check of a rebuilt wall
(tangential.inside, split by query point) and the facets that need a full
Algorithm-3 subdivision (facets.check_many, split by facet). Each piece is
computed exactly as in one process -- the split is over independent items,
never inside one item's arithmetic -- so the results are bit-identical.

The payload (oracles hold closures, which do not pickle) is put in a module
global BEFORE the pool forks, so the workers inherit it; only item indices
and results cross the process boundary. Without fork (or with one worker) the
work runs in this process.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_WORKERS = 1
_SHARED = None


def set_workers(n: int) -> None:
    """How many worker processes the proof may use (Page3Params.parallel_workers)."""
    global _WORKERS
    _WORKERS = max(1, int(n))


def workers() -> int:
    return _WORKERS


def _call(args):
    fn, item = args
    return fn(_SHARED, item)


def fork_map(fn, items: list, shared, min_items: int = 2) -> list:
    """[fn(shared, item) for item in items], in up to `workers()` processes.
    fn must be a module-level function (it is pickled by name)."""
    global _SHARED
    items = list(items)
    n = min(_WORKERS, len(items))
    if n <= 1 or len(items) < min_items:
        return [fn(shared, it) for it in items]
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return [fn(shared, it) for it in items]
    _SHARED = shared
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
            return list(ex.map(_call, [(fn, it) for it in items]))
    finally:
        _SHARED = None
