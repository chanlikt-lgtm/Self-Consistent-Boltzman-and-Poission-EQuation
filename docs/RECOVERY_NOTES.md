# Rev 6 trio — recovery campaign notes (2026-08-22)

## What happened
The from-scratch run (`run_from_scratch.bat`) aborted at Stage 3 when the **cold-started** first node
71×58 did not reach the observable-plateau gate within `MAX_ROUNDS=4` (240 iters). This was a
**harness-limit failure, not a scientific one**: at 240 iters 71×58 was already at Gᵢᵢ=2.9846e25
(0.2% from archival 2.990e25), G_tot and Tₑ matching, R_φ=6.99e-4 floored — only the last-4 spreads
(1.24e-3/1.02e-3) had not yet collapsed below 3e-4. `MAX_ROUNDS=4` was implicitly tuned to a
warm-start initialization; the archival 71×58 was warm-started and reached the gate in ~68 iters.

## Recovery applied (running as of 07:14)
- `MAX_ROUNDS` 4→8 (hard cap retained). **Scientific gates UNCHANGED** (spread≤3e-4 both, R_φ≤1.5e-3).
- Idempotent, provenance-checked resume in `run_richardson_trio.py`: same-node checkpoint →
  prev-node warm-start → cold; fail-closed on provenance mismatch; skip passed / resume incomplete.
- 71×58 resumes from its own 240-iter checkpoint (chunks_done=4/8, cum_iters preserved).
- Archival 100×70/141×85 deliberately excluded so they regenerate (moved to `archival_backup/`).
- `run_from_scratch.bat`: added stage-level skip (Stage 1/2 via sentinel figures).

## Post-campaign TODOs (do NOT hot-patch the running driver/harness)
1. **campaign_id / run_id lineage.** Stamp `campaign_id` + `run_id` into every NPZ, sidecar, and
   manifest. The existing three identifiers answer *same-definition* (config_sha256), *same-code*
   (git_commit), *same-energy-grid* (H_hash) — none answers *"produced by THIS campaign?"*. This
   run exposed that the new harness could not distinguish tonight's would-be 100×70/141×85 from older
   archival states (identical config/H-grid) and marked them SKIP-eligible. With a campaign/run UUID,
   policy can explicitly choose *scientifically-reusable checkpoint* vs *fresh artifact required for
   this campaign* without relying on filenames/timestamps. (Touches `coupled_drain_richardson.py` +
   `run_richardson_trio.py`.)
2. **Sidecar carry-forward of the `recovery` block.** `save_progress` currently overwrites the
   sidecar each chunk; it should merge/carry a `recovery{bootstrapped, source_checkpoint,
   reconstructed_cum_iters, original_harness_had_sidecar, original/resumed MAX_ROUNDS}` block. For
   this run that provenance lives in `data/she2d_richardson_71x58.recovery.json` (separate file).
3. **Validate the revised `.bat` from its native Windows launch.** This recovery validated the
   *scientific* path (Stages 3–6 run directly as a Python chain, because launching the `.bat` from
   Git-Bash/`cmd` hit a cwd/path quirk). It did NOT validate the updated one-command batch harness
   end-to-end. Run one clean PowerShell/cmd smoke test of `run_from_scratch.bat` (with the stage-skip)
   after the campaign.
4. Move stale `*_stage1.npz` out of `data/` too (same glob-safety rationale as the archival move).
5. **Degenerate-plateau gate hole (exposed by resumed 71x58, 2026-08-22 10:18).** When the driver's
   nominal `tol_phi=1e-4` early-stop fires, an outer chunk can run a SINGLE iteration; the last-4
   spread over one value is trivially 0.00, which passes the observable-plateau gate. The last REAL
   last-4 G_tot spread for 71x58 was 3.65e-4 (marginally ABOVE the 3e-4 gate), so the formal pass came
   via a degenerate 1-iter chunk. The node was substantively converged (R_phi=3.94e-5, Gii=3.076e25
   stable), so the SCIENCE is intact, but the gate must not accept a degenerate spread. Fix options:
   (a) require the last chunk to contribute >= 4 real outer iterations before the last-4 spread counts;
   (b) lower the driver `tol_phi` below the observable-gate requirement so it cannot early-stop before
   the plateau is genuinely measured; (c) compute the spread over a fixed trailing window that spans
   >= 4 distinct iterations even across chunk boundaries. Also: the "slow 186s/iter" WARN on a 1-iter
   chunk is a false positive (setup overhead / 1 iteration), not a real slowdown.
6. **71x58 fresh value 3.076e25 vs archival 2.990e25 (~+2.9%)** is just outside the 2% archival
   reproducibility tolerance. BUT the 2.9% is only relevant AFTER the node has a valid accepted
   plateau state (see below); it does not by itself imply YELLOW.

## 71x58 acceptance status: NEEDS_RECHECK (not PASS) -- CORRECTED 2026-08-22
The recorded gate=PASS for 71x58 is INVALID. Solver-completed != Rev-6 plateau gate validly measured.
Verified facts:
- The npz holds only the degenerate single sample (last4_* = 1 value, n_iter=1, spread=0.0); chunk 6's
  1-iteration save OVERWROTE chunk 5's real last-4 array.
- The metrics CSV is per-CHUNK, not per-iteration; the 4 genuine trailing samples are stored nowhere.
- Chunk 5's last genuine window had G_tot spread = 3.653e-4 > 3e-4, i.e. it FAILED the gate; the 1-iter
  chunk 6 (spread 0.0) produced a trivial pass that masked this.
- Physical state is well converged (R_phi=3.94e-5, Gii=3.076e25 stable), but the strict observable gate
  is UNVERIFIED, not passed.

### Repair (post-campaign, ordered; live campaign untouched)
(a) Reconstruct the true cumulative last-4 window from stored per-iteration history -- INFEASIBLE here
    (per-iteration history not persisted; chunk-5 last-4 overwritten; chunk-5 summary itself failed).
(b) => Rerun ONLY 71x58 from its final checkpoint after fixing the measurement logic. Keep every
    physical/numerical parameter and threshold UNCHANGED. This is an audit-verification rerun, NOT a
    change to the scientific method.
(c) Do NOT use the 1-iteration zero-spread result as archival evidence.

### Provenance rules for the correction (locked)
- **Do NOT overwrite/"correct" the original PASS metrics row** (richardson_trio_metrics.csv 71x58 chunk 6,
  gate_pass=True). Preserve it as the historical output of the defective bookkeeping so the DEFECT is
  reproducible. The superseding record is external:
  `data/she2d_richardson_71x58.audit_correction.json`
  (original_status=PASS, audit_status=NEEDS_RECHECK, reason=degenerate_plateau_window,
  source_artifact_sha256=e1ceb7a8236fc92f3c29669337c75a188a4a6060771563fbc19a4d223786d842).
- **The (b) verification rerun gets a NEW run_id**, linked to the existing checkpoint as its PARENT:
  `original recovered run -> checkpoint -> audit-verification rerun`. It must NOT look like a
  continuation of an already-valid accepted node. Physical/numerical inputs + gate byte-identical;
  only the bookkeeping implementation changes.

### Permanent fix (measurement bookkeeping only -- NOT a science change)
- Authoritative definition: **plateau window = last four ACTUAL iterations of the cumulative node
  trajectory, independent of chunk boundaries.**
- The continuation layer must persist a ROLLING BUFFER of real per-iteration (Gii, Gtot, R_phi) samples
  across solver invocations, and set `plateau_evaluable=false` until >=4 real samples exist. Only then
  evaluate the spread gate.
- `tol_phi` remains the SOLVER's own termination criterion; do NOT couple it numerically to the
  observable-spread threshold (two different convergence measures). (The driver should also persist
  per-iteration history so option (a) is possible in future -- the current per-chunk-only + npz-overwrite
  design destroyed the evidence.)

### Report state machine consequence
YELLOW is NOT yet justified. YELLOW requires valid plateau measurements + qualitative verdict reproduces
+ a value missing the 2% tolerance. Until 71x58 has a VALID accepted plateau (via reconstruction or the
(b) rerun), the study is audit-incomplete for archival acceptance -> effectively RED or a dedicated
`NEEDS_RECHECK` state. The 2.9% discrepancy becomes relevant only after a valid accepted state exists.

### 100x70 / 141x85 during the live run
Same distinction: solver/node completed != Rev-6 plateau gate validly measured. If either finishes via a
1-3 iteration continuation (tol_phi early-stop), let the campaign mechanically continue but treat the
plateau as NOT-yet-validly-measured until the cumulative 4-real-sample window is checked offline. Do NOT
pre-judge 100x70 as "should return to 2.135e25": after chunk 1 it is plainly unconverged (spreads ~3e-3,
Gii ~1.95e25); let the valid convergence criterion decide between "same fixed point via a different
transient" and "settles materially different".

## Report wording distinction (for the eventual write-up)
Distinguish: *"Recovered scientific run completed using the same Stage 3–6 Python commands"* from
*"the updated one-command Windows batch harness has itself been validated"*. Only the former is true
until the `.bat` smoke test in TODO 3 is done.

## Live campaign status + policy (2026-08-22 18:36)
Node states (NO GREEN/YELLOW/RED language until data decide):
- 71x58  = NEEDS_RECHECK (degenerate 1-iter plateau window; see audit_correction.json).
- 100x70 = RUNNING / NOT PLATEAUED (chunk 2/8; Gii 1.967->1.949->1.932e25 coherent downward drift;
  both spreads ~3e-3, ~10x above the 3e-4 gate; R_phi in band but node still evolving materially --
  another demonstration that R_phi cannot substitute for the observable plateau gate).
- 141x85 = NOT REGENERATED.

Clean status wording (use this): "Controlled trio reproduction remains unresolved; the fresh middle
node is following a materially different transient and has not reached the observable plateau gate."

Confirmed policy WHILE THE CAMPAIGN EXECUTES (user-approved): no solver change; no tolerance change; no
cap change. Do NOT compare 1.932e25 to the 2% archival tolerance (not an accepted node). WORDING: the
harness loops range(chunks_done, MAX_ROUNDS), so chunks are 0-indexed (0..7) and MAX_ROUNDS=8 means a
HARD CAP AFTER EIGHT CHUNKS (use this phrasing, not "chunk 8", to avoid 0/1-based ambiguity). If 100x70
reaches the hard cap (eight chunks) without a valid four-REAL-iteration plateau -> classify
INCOMPLETE / NEEDS_CONTINUATION, NOT reproduced. If any late continuation terminates after <4 real iterations -> apply
plateau_evaluable=false (same rule as 71x58), never accept a degenerate spread. Cap risk is real
(chunk drift eased only 0.94%->0.83%); do not assume convergence within 8 chunks.

Observation to record from chunks 3-5: does the chunk-drift stay NEGATIVE while its magnitude
collapses (-> convergence toward a lower fixed-point value), or does it REVERSE sign (-> longer
transient)? OBSERVED (chunks 0-6): down through chunk 2 (min 1.932e25), then UP through chunks 3-6 to
2.129e25 -> ONE turning point = **non-monotone transient with a sign reversal**. Do NOT call it
"oscillatory" unless the sign reverses AGAIN (repeated reversals). Chunk 6 shows the up-relocation
DAMPING (drift 5.69%->1.05%; spreads shrinking but still ~5.1e-4 > 3e-4).

### Chunk-to-chunk drift is DIAGNOSTIC, not part of the gate (clarified)
The formal Rev-6 acceptance gate is UNCHANGED and drift is NOT in it: accept a node iff there are
>=4 GENUINE final iterations AND s(Gii)<=3e-4 AND s(Gtot)<=3e-4 AND R_phi<=1.5e-3. A large
chunk-to-chunk move followed by four genuinely stable final iterations CAN validly pass; a tiny drift
does NOT rescue spreads that remain above threshold. Early termination from tol_phi is NOT itself
defective: if the driver stops after >=4 genuine iterations (e.g. 8), the last-four window is valid.
The defect arises ONLY when <4 real samples exist and the implementation treats the shortened array
as a legitimate "last four".
Chunk-7 audit outcomes (REFINED -- cumulative last-4 across chunk boundaries; chunk-6 preservation
makes a short chunk 7 reconstructable, NOT auto-NEEDS_RECHECK):
- >=4 real iters; cumulative final-4 available; BOTH spreads <=3e-4; R_phi in band -> VALID PLATEAU.
- >=4 real iters; cumulative final-4 available; either spread fails -> INCOMPLETE / NEEDS_CONTINUATION.
- 1-3 real iters, BUT chunk-6 + chunk-7 reconstruct 4 cumulative real iters -> evaluate that
  reconstructed 4-point window NORMALLY (last (4-n) of chunk-6 + all n of chunk-7).
- <4 cumulative real samples retrievable / provenance ambiguous -> plateau_evaluable=false / NEEDS_RECHECK.
IMPORTANT: the harness's OWN reported spread from a 1-3-iteration chunk is INVALID; only the OFFLINE
cumulative reconstruction is admissible. Reconstruction recipe: cumulative_last4 =
{last (4-n) samples of chunk-6 last4_*} ++ {all n real samples of chunk-7 last4_*} (in iteration
order). Use the EXACT Rev-6 gate formula on the four RAW cumulative samples (NOT per-chunk summaries):
  spread = (max4 - min4) / mean4   for each of G_ii and G_tot,
applied identically to how coupled_drain_richardson.py computes it. Persist alongside the reconstructed
result, for independent checkability: the four raw values (G_ii, G_tot, R_phi), their absolute
iteration indices, and the sha256 of BOTH source NPZs (chunk-6 preserved + chunk-7). A short chunk 7
(1-3 new iters) can legitimately yield a VALID PLATEAU if the preserved chunk-6 tail supplies the
missing samples and provenance is unambiguous.
Emit the reconstructed decision as a SEPARATE audit artifact
(data/100x70_chunk7_plateau_reconstruction.json) -- do NOT modify either source NPZ. That file records:
the 4 raw cumulative samples (Gii, Gtot, R_phi), their absolute iteration indices, both source NPZ
sha256, the two computed spreads, R_phi-band check, and the resulting audit_status. Source solver
outputs stay immutable; the derived acceptance decision is fully reproducible from that artifact.
Evaluation order: preserve chunk-7 artifact -> read n_iter -> (n>=4: use its genuine final four) OR
(n=1-3: append real samples to preserved chunk-6 tail, select final four cumulative) -> exact Rev-6
spreads -> unchanged gate.
Audit-artifact schema (v1.0, user-frozen): {artifact_type:"plateau_reconstruction", schema_version,
node, source_npz:[{role:"previous_chunk_tail",path,sha256},{role:"terminal_chunk",path,sha256}],
samples:[{absolute_iteration,Gii_max,Gtot,Rphi_V}x4], formula:"(max(x)-min(x))/mean(x)", Gii_spread,
Gtot_spread, max_Rphi_V, thresholds:{observable_spread:3e-4,max_Rphi_V:1.5e-3}, plateau_evaluable,
plateau_status:"PASS|FAIL", note}. TWO REQUIREMENTS: (1) plateau_status is derived ONLY from the
unchanged Rev-6 thresholds; (2) the JSON itself gets a SHA-256 and is referenced from the final
manifest/report so the audit decision has its OWN identity. Source NPZs remain immutable historical
facts; the corrected plateau decision is a separate reproducible interpretation layer.
ACTION when chunk 7 lands: (1) PRESERVE chunk-7 npz (esp. last4_Gii_max/last4_G_tot/last4_R_phi + n_iter)
before any future continuation can overwrite it; (2) if n_iter>=4 evaluate normally; (3) if 1<=n_iter<=3
do the offline cumulative reconstruction with the preserved chunk-6 tail BEFORE any classification;
(4) do NOT classify NEEDS_RECHECK until reconstruction is shown impossible/ambiguous.
Chunk-6 genuine last-4 evidence preserved:
archival_backup/she2d_richardson_100x70.chunk6_preserved.npz (sha256 5657b44b...).

## 100x70 ACCEPTED (2026-08-23 08:18) + lineage dependency
100x70 = VALID PLATEAU under the UNCHANGED Rev-6 gate, NO bookkeeping caveat: chunk 7 ran the full 60
iterations (all 4 terminal samples real), s(Gii)=1.88e-4, s(Gtot)=1.78e-4, R_phi=2.57e-4 V. Accepted
Gii=2.1362e25. Preserved: archival_backup/she2d_richardson_100x70.accepted_plateau.npz (sha 9e829780).
WORDING: the non-monotone transient (dip to 1.932e25 -> climb to 2.1362e25) resolved to a state
CONSISTENT WITH the archival fixed point (2.135e25); do NOT assert identity of the fixed point before
the whole reproduction/audit closes, and do NOT invoke the 2% parity criterion yet.
ENGINEERING: acceptance occurred on chunk index 7 -- the node used the ENTIRE eight-chunk allowance.
Treat MAX_ROUNDS=8 as a safety cap that HAPPENED to be sufficient, NOT a newly-established optimal
production limit; do NOT tune it downward based on this run.

### Lineage dependency (record explicitly)
100x70 was warm-started from she2d_richardson_71x58.npz AT THE PRE-RECHECK 71x58 state
(cum 358 iters, Gii=3.076e25, sha256 e1ceb7a8...=the degenerate-accepted seed). The post-campaign
71x58 audit-verification rerun may advance 71x58 by a few real iterations, so its final ACCEPTED state
may differ slightly from the exact checkpoint that seeded 100x70. This does NOT automatically
invalidate 100x70: its own 60-iteration terminal window is valid and independently satisfied the
observable gate. RULE: if the 71x58 recheck produces a MATERIALLY different state, THEN decide whether
a 100x70 sensitivity rerun is warranted; do NOT automatically regenerate 100x70 merely because the
bookkeeping recheck advanced 71x58 by a few iterations.

### 141x85 (running): same evidence-preservation discipline
Preserve the last genuine FULL-chunk state before any continuation that might terminate early
(tol_phi early-stop); if a terminal chunk runs <4 real iters, use the cumulative-window reconstruction
protocol + separate audit artifact (as specified above). No solver/gate/cap change.

## CAMPAIGN COMPLETE (2026-08-23 13:37) -- auto-YELLOW banner is INVALID/PREMATURE
Chain finished (exit 0). Archive: runs/2026-08-23_133710_resume/. Fresh trio node states:
- 71x58  = NEEDS_RECHECK. npz still holds the DEGENERATE window (n_iter=1, spread=0.00/0.00, Gii=3.0756e25).
- 100x70 = VALID PLATEAU. n_iter=60, spread 1.88e-4/1.78e-4, Gii=2.1362e25. (accepted)
- 141x85 = VALID PLATEAU. n_iter=60, spread 2.30e-4/2.42e-4, Gii=2.3874e25. (accepted, passed on chunk 0)

Harness/richardson_fit/gen_run_report auto-computed: TRIO 3.076 -> 2.136 -> 2.387 e25, NON-MONOTONE,
S_rev=737, banner = **YELLOW** ("all scientific gates pass ... value(s) outside 2% archival tol").

**THIS YELLOW IS INVALID / SUPERSEDED.** gen_run_report read 71x58's npz and treated its DEGENERATE
spread=0.00 (n_iter=1) as a passing gate -- so "all scientific gates pass" is FALSE for 71x58. Per the
confirmed policy, NO GREEN/YELLOW/RED verdict may issue until 71x58 is validly rechecked. The honest
status is:
  AUDIT-INCOMPLETE -- two VALID PLATEAUS (100x70, 141x85) + one NEEDS_RECHECK (71x58);
  controlled-trio verdict WITHHELD. The auto-YELLOW banner in report_run_latest is superseded by this.
Preserve report_run_latest as the historical (defective-bookkeeping) output; do not overwrite it; the
superseding status lives here + in the 71x58 audit_correction.json.

Qualitative evidence (NOT a verdict/parity claim): the fresh sequence is down-then-up (non-monotone,
S_rev=737), reproducing the archival non-monotone SHAPE (archival S_rev=874). The two GENUINELY
accepted nodes match archival within 2% (100x70 2.1362 vs 2.135 = +0.06%; 141x85 2.3874 vs 2.413 =
-1.1%). 71x58 fresh value 3.076 vs archival 2.990 = +2.9% AND is NEEDS_RECHECK -- both its value and its
acceptance are unresolved, so even the YELLOW rationale ("value outside 2%") is provisional on the recheck.

NEXT (post-campaign, ordered): (b) 71x58 audit-verification rerun (new run_id, parent=checkpoint) with
fixed plateau bookkeeping (rolling 4-real-iteration window) -> validly accept 71x58 -> re-run
richardson_fit on the three accepted npz -> ONLY THEN issue the honest GREEN/YELLOW/RED verdict.
Then report reconciliation checklist below.

## Post-campaign REPORT reconciliation checklist (finalized, user-approved)
Apply these to `report/report_v6_full.tex` only after the campaign finishes and its artifacts +
backend audit support the verdict:
1. **Preserve the standalone ~1.8e25 (100x70, own per-mesh H-grid) as historical / pre-equalization
   evidence** that motivated the controlled experiment. Keep it, but label it clearly an
   *intermediate drain-refined result*, NOT part of the common-H controlled trio.
2. **The common-H trio 2.990 -> 2.135 -> 2.413 e25 is the authoritative controlled sequence** and owns
   the final spatial-convergence verdict.
3. Insert the canonical **"Superseded Rev-5 interpretation"** paragraph verbatim (user-supplied).
4. **Call the uniform 40x34->60x50->80x66 extrapolation apparent/heuristic and NON-admissible as
   Richardson** (not a common-ratio self-similar sequence -> the 3-point p-formula's common-ratio
   assumption is violated). If p~=4.6 and G_inf~=1.1e26 are printed at all, label them
   illustrative/apparent and state the exact heuristic used; OMITTING the numeric p and G_inf
   altogether is also acceptable -- the only historical claim needed is that Rev 5 suggested a scale
   of order 1e26, which Rev 6 superseded.
5. Prefer **"several-fold" (~5x, using the trio value 2.135e25)** over "order of magnitude" when
   comparing controlled drain-refined values against the old uniform scale/extrapolation.
6. Keep campaign language in **design tense** ("is designed to independently reproduce that verdict")
   until the fresh trio actually produces its verdict.
7. Only after completion, change "is designed to reproduce" -> "reproduced" **iff** the regenerated
   artifacts and backend audit actually support it. Let the recovered 71x58 pass/fail on the
   UNCHANGED acceptance criteria, then let the independently regenerated 100x70/141x85 determine what
   the final report is allowed to say.

## RECHECK CLOSED (2026-08-23) -- 71x58 CERTIFIED; auto-YELLOW now legitimately re-derived
Supersedes the "NEEDS_RECHECK" / "auto-YELLOW INVALID" entries above (history preserved, not rewritten).

The permanent fix landed as `src/plateau_gate.py` (authoritative; cumulative window over the last four
REAL iterations, spanning chunks; `plateau_evaluable=false` until 4 genuine samples; `tol_phi` decoupled;
production spread formula (max-min)/mean and thresholds 3e-4 / 1.5e-3 UNCHANGED; strictly-increasing
abs-iter buffer integrity fail-closed; `buffer_origin` recorded). Wired into `run_richardson_trio.py`;
21/21 unit tests + 7/7 harness-glue tests pass.

Narrow recheck (run_id `recheck_71x58_20260823T161930_49efe4f5`, parent `e1ceb7a8`, all physics/mesh/
solver/thresholds unchanged, MAX_ROUNDS 8->16 safety cap only): only 71x58 recomputed; 100x70/141x85
certified-and-skipped via NPZ terminal reconstruction. The converged node early-stopped 1 iter/chunk;
the corrected gate held `not-evaluable` for n=1,2,3 and certified at n=4 (abs-iters 359-362),
Gii=3.0756e25, spread 0.00/0.00, R_phi=3.94e-5 floored. This spread=0 is legitimate (four distinct
warm-started re-solves each re-confirming the same fixed point), NOT the earlier single-sample artifact.

`richardson_fit` on the three accepted npz: 3.0756 -> 2.1362 -> 2.3874 e25, NON-monotone, S_rev
(Gii)=736.9, (G_tot)=524.1 -> Richardson SUPPRESSED. Parity vs archival [2.990,2.135,2.413]e25:
71x58 +2.86% (OUTSIDE 2%), 100x70 +0.06%, 141x85 -1.06% (within). State machine => legitimate YELLOW:
scientific verdict reproduces, one accepted node outside numerical tolerance. `gen_run_report.py` now
sources certified cumulative-window evidence and labels the banner "re-derived after plateau-bookkeeping
repair"; the report reports scientific reproduction and numerical parity SEPARATELY.

Lineage: G1 unchanged (3.0756e25) -> the acceptance measurement changed, not the physical state ->
100x70 (warm-started from pre-recheck 71x58) does NOT require regeneration.

Immutable artifacts: data/she2d_richardson_71x58.recheck_manifest.json (sha 29809996), 71x58_recheck_
verdict.json (sha 7a3bcedd), archival_backup/she2d_richardson_71x58.defective_pass.parent.npz (e1ceb7a8),
archival_backup/report_run_latest.DEFECTIVE_yellow.2026-08-23T1337.pdf/.tex (defective report preserved).

Node states now: 71x58 = CERTIFIED (was NEEDS_RECHECK; defective n=1 PASS superseded);
100x70 = VALID PLATEAU; 141x85 = VALID PLATEAU. Final campaign classification = YELLOW.
Still pending: native .bat smoke test (batch already exercises the corrected harness end-to-end);
design-spec v0.2.9 (plateau-measurement-integrity contract + NEEDS_RECHECK state + audit-correction pattern).
