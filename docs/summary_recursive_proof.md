# Summary: the recursive infeasibility proof and everything changed with it

Developed on branch `claude/focused-ramanujan-w0714i` (merged into `main` on 2026-11-17).
Detailed records (in Chinese): `docs/recursive_proof_design.md` §10.1–10.9 and
`docs/code_changes_recursive.md` §1–6.9.

## 1. The recursive proof (the main method)

| What | Where |
|---|---|
| Design: the 8 pages glued at doors; recursive `inf_proof(A, B)` (AND across doors, OR within a door); the three checks — (i) facets in the blocked region, (ii) facets clear of doors, (iii) doors inside the wall become split points | `docs/recursive_proof_design.md` |
| σ (the page-coordinate singularity) with a proven speed limit L per box | `src/zzy_rss/chart.py` |
| Box search that proves where doors cannot be; door regions | `src/zzy_rss/interior.py` |
| Door curves refined inside the regions (no door list, no closed form) | `src/zzy_rss/doorset.py` |
| `check(A, B)` → CONNECTED / WALLED / FAILED; per-page walls trained and verified | `src/zzy_rss/check.py` |
| Recursive prover, proof tree, path search with verification, certificates | `src/zzy_rss/proof.py` |
| Runner for 4 cases: `exp2-none`, `exp2-zgap`, `exp3-new`, `two-wall` | `scripts/run_recursive.py` |
| Tests | `tests/test_chart.py`, `tests/test_recursive.py` |

The recursive prover reproduced every earlier verdict. The two-page experiment became the first
proof that splits at a door.

## 2. Door finding: six methods tried, Method 2 kept

| Method | What it was | Outcome |
|---|---|---|
| **2: σ + hand-derived L** | Box search with a proven bound | **Kept, the only door finder.** Bound tightened ~2.9× ("frame" bound); 540k random pairs, 0 violations |
| General σ (7×7 Jacobian) | Smallest singular value | Finds doors, no proven bound; kept only as a test |
| 4: Krawczyk / interval arithmetic | General and proven | Impractical (certifies only tiny boxes). **Deleted** |
| 5: algebraic, arm-geometry variables | 4 homotopy paths per slice | Matched the closed form exactly, but only for this arm. **Deleted** |
| 6: algebraic, joint variables | Monodromy + homotopy; later exact Jacobian, batched tracking, trace test, per-point Krawczyk | Matched the closed form on 12/12 slices; the trace test never passed. **Deleted** |

Deleted code is recoverable from commit `7c00d48`; findings remain in the design doc §10.4–10.6.

## 3. Dropping doors: certificates only

- **Certified collision:** every posture in a door region collides, checked box by box; holds on
  all pages.
- **Certified joint limits (new):** proven speed limits for all 7 joint angles (670k random pairs,
  0 violations). A door is dropped if a joint that does not move along its free spin is past its
  limit everywhere in the region, checked page by page. exp2-none's q2 doors are now dropped by
  proof in 0.07 s (the old evidence check took ~31 s).
- **The old sampled-evidence check is off by default** (`Page3Params.discharge_evidence=False`).

## 4. Method 2 as the main method (soft switch-over)

- `scripts/run_recursive.py` is the documented main entry point.
- The 9 old-pipeline scripts print a `[legacy]` notice (`src/zzy_rss/legacy.py`) but still run;
  nothing was deleted.
- The closed-form door formulas remain as the independent answer key in the tests.

## 5. Wall reuse

Before training a wall for a claim, the prover first runs the usual final check on the verified
walls already built on that page and reuses one that passes. The two-page experiment now trains
**one wall per page** (2 trained + 1 reused, instead of 3). Switch: `Page3Params.reuse_walls`.

## 6. Speed-ups (all with identical results)

| Change | Effect |
|---|---|
| Batched edge checking while building pages (one call per halving level) | Page building ~3× faster; maps identical |
| Pages built in parallel (`Page3Params.parallel_pages=2`) | The pages are built at the same time |
| Wall self-check and hard facet checks in parallel (`src/zzy_rss/parallel.py`, `parallel_workers=4`) | Self-check bit-identical, ~3.3× faster |
| Batched path verification (`WitnessParams.batch_verify`) | Bit-identical, 48× faster |

## 7. Figures and outputs

| Output | What it shows |
|---|---|
| `docs/fig_sigma_doors.png` | σ and the door regions |
| `docs/fig_facet_threshold.png` | Why the facet threshold must scale with facet size |
| `scripts/fig_eight_pages.py` → `docs/fig_eight_pages_*.png` | All 8 pages on one slice with door regions, start, goal, walls (reused ones dashed) and the path; plus a 3-D view of each wall (`*_wall3d.png`). One "invalid" colour by default; `--split-invalid` restores per-reason colours |
| `run_recursive.py --save-geometry` | Saves wall meshes and the path (also fixed walls being dropped on failed door splits) |

## 8. Results now

| Experiment | Verdict | First recursive version | Now |
|---|---|---|---|
| Two-page (`two-wall`) | INFEASIBLE, 2 walls + 1 reuse | 116.3 s (3 walls) | **35.4 s** |
| EXP 3 (`exp3-new`, `--also-prove`) | FEASIBLE; start page walled 2520/2520 | 96.4 s | **24.5 s** |
| exp2-zgap | INFEASIBLE, 1 wall 2006/2006 | 9.9 s | **6.2 s** |
| exp2-none | FEASIBLE | 3.7 s | **2.2 s** |

Tests passing: `test_chart`, `test_recursive`, `test_witness` (main method), and the legacy tests
affected by shared code — `test_facets`, `test_tdc`, `test_batch`, `test_page3`, `test_flood3`,
`test_crossing`.

## 9. Still open

- The path is checked configuration by configuration, not the motion between them.
- Collision uses capsules, not the real link meshes.
- σ, L and the joint bounds are specific to the S-R-S arm with a flat task.
- The prover handles only 3-D (weld-patch) experiments; 2-D weld-line experiments need the legacy
  pipeline.
- About 10 s of each two-page run (setup and final checks) has not been optimised.
