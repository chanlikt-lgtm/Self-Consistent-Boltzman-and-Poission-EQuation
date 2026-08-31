# From-scratch recovery campaign — formal summary (frozen 2026-08-23; audit-closed 2026-08-23 recheck)

## Final campaign classification: YELLOW (audit-closed)
The corrected plateau bookkeeping certifies all three controlled nodes on genuine four-real-iteration
windows. The independently reproduced trio remains significantly non-monotone,
3.0756→2.1362→2.3874×10²⁵, with S_rev=736.9 for Gᵢᵢ,max and 524.1 for G_tot; Richardson extrapolation
is therefore suppressed and spatial convergence is not demonstrated. Two nodes reproduce within the
declared 2% archival tolerance, while the accepted 71×58 value differs by +2.86%. The resulting
classification is legitimately **YELLOW**: the scientific verdict reproduces, but one accepted node lies
outside the archival numerical tolerance.

## History preserved (do not overwrite)
The **original** 71×58 acceptance was a **defective one-iteration plateau window** (tol_phi early-stop →
last-4 over a single sample → spread=0 false PASS), classified NEEDS_RECHECK and superseded. The earlier
auto-generated YELLOW was defective-bookkeeping output. The **later recheck** (run_id
`recheck_71x58_20260823T161930_49efe4f5`, parent checkpoint `e1ceb7a8`, all physics/mesh/solver/
thresholds unchanged) produced **four genuine cumulative samples** (abs-iters 359–362) and certified the
node. The corrected report YELLOW is now explicitly labeled "re-derived after plateau-bookkeeping repair".

## Node evidence (post-recheck)
| node | Gᵢᵢ,max (×10²⁵) | certified cumulative-window spread (Gᵢᵢ / G_tot) | window (abs-iters) | audit status | vs archival |
|---|---|---|---|---|---|
| 71×58  | 3.0756 | 0.00 / 0.00 (4 identical real samples) | 359–362 | **CERTIFIED** (was NEEDS_RECHECK; defective n=1 PASS superseded) | +2.86% (outside 2%) |
| 100×70 | 2.1362 | 1.88×10⁻⁴ / 1.78×10⁻⁴ | 57–60 | **VALID PLATEAU** | +0.06% |
| 141×85 | 2.3874 | 2.30×10⁻⁴ / 2.42×10⁻⁴ | 57–60 | **VALID PLATEAU** | −1.06% |

71×58 spread=0.00 is now legitimate: four **distinct** warm-started re-solves each re-confirming the same
fixed point (R_φ=3.94×10⁻⁵ floored) — not a single-sample artifact. Richardson fit re-run on the three
accepted npz reproduces the non-monotone reversal (S_rev: Gᵢᵢ 736.9, G_tot 524.1). Archival auto-report
`report_run_latest` (defective YELLOW, 2026-08-23 13:37) preserved as
`archival_backup/report_run_latest.DEFECTIVE_yellow.2026-08-23T1337.pdf`; superseded, not deleted.

## Narrow 71×58 audit-verification rerun — COMPLETED 2026-08-23
- Ran **only 71×58** (100×70 & 141×85 certified-and-skipped via NPZ terminal reconstruction — no recompute).
- **run_id** `recheck_71x58_20260823T161930_49efe4f5`; parent checkpoint sha `e1ceb7a8`.
- **Corrected rolling cumulative-four-real-sample bookkeeping** now in `src/plateau_gate.py` (authoritative)
  and wired into `src/run_richardson_trio.py`; `plateau_evaluable=false` until ≥4 genuine samples;
  `tol_phi` decoupled; buffer-integrity (strictly-increasing abs-iter) fail-closed; `buffer_origin` recorded.
- Every physics/mesh/solver parameter and numerical threshold UNCHANGED (only the acceptance measurement
  was repaired). MAX_ROUNDS raised 8→16 (pure safety cap, not a scientific gate).
- The node early-stopped 1 iter/chunk (converged); the corrected gate correctly held `not-evaluable` for
  n=1,2,3 and certified at n=4 (abs-iters 359–362), Gᵢᵢ=3.0756×10²⁵, spread 0.00/0.00, R_φ floored.

## Outcome: #2 (passes but outside 2% tol) → legitimate YELLOW
The recheck **passed** (genuine 4-sample plateau) but 71×58 remains **+2.86%** vs archival (outside 2%),
while 100×70 (+0.06%) and 141×85 (−1.06%) are within. `richardson_fit` re-run on the three accepted npz
reproduces the significant non-monotone reversal → **YELLOW** is now legitimately re-derived.

## Lineage after the recheck (100×70 dependency) — RESOLVED
Rechecked G₁ = 3.0756×10²⁵, **unchanged** from the pre-recheck value (the fix corrected the measurement,
not the physics). G₁ did not move materially → **no sensitivity assessment needed**; the independently-
plateaued 100×70 (G₂) is NOT regenerated.

## Audit artifacts (immutable chain)
- `data/she2d_richardson_71x58.recheck_manifest.json` (sha 29809996…) — run_id, parent, unchanged params.
- `data/71x58_recheck_verdict.json` (sha 7a3bcedd…) — plateau evidence, deviations, YELLOW verdict.
- `archival_backup/she2d_richardson_71x58.defective_pass.parent.npz` (sha e1ceb7a8…) — defective PASS preserved.
- `archival_backup/report_run_latest.DEFECTIVE_yellow.2026-08-23T1337.pdf/.tex` — defective auto-report preserved.

## Still pending (post-reconciliation)
Native `.bat` smoke test (the monitored batch already exercises the corrected harness end-to-end with all
three nodes `ALREADY PASSED … buffer_origin=persisted_sidecar`) + the design-spec v0.2.9 additions
(plateau-measurement-integrity contract, NEEDS_RECHECK state, audit-correction pattern).
