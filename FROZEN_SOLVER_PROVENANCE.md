# Frozen solver provenance — reported-results version

This records the exact solver version used to produce the reported Liang-1997 SHE-BTE results,
per the reproducibility protocol (freeze after the banked-speed port, before the final trio).

## Code
- Repo: `liang1997_she_bte_mosfet_repro`
- Branch: `richardson-trio-recheck-closeout`
- Commit: **7576ce0** ("Port banked solver speed changes (DD-skip + cleanups) from speedup sandbox")
- Parent (certified closeout): `d30b471`
- Tag: **`solver-frozen-2026-09-26`**
- Banked changes vs `d30b471`: DD-skip for valid warm starts (bit-identical); numba-accelerated
  assembly present but DORMANT (numba not installed -> byte-identical Python assembler runs);
  direct-solve cutoff 130000; Krylov x0 = OFF; preconditioner reuse = OFF; per-iter backend/reuse
  logging. Optimization campaign (fixed-envelope, outer-Anderson) rejected -> not included.

## Environment
- Python 3.13.15  (`C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe`)
- NumPy 2.5.1
- SciPy 1.18.0
- OS: Windows 11
- Assembly backend at runtime: **python** (numba absent; force with `SHE2D_DISABLE_NUMBA=1`)

## Runtime settings (Rev-6 controlled trio)
- Energy grid: dH = 0.0125 eV, eps_max = 3.02 eV, COMMON frozen H-span (identical across nodes)
- SHE linear tol: she_tol = 1e-8; fresh matrix + fresh split-ILU + cold Krylov every outer iteration
- Outer accelerator: Anderson depth = 6, beta = 0.50, step_cap = 5.0, restart = 6
- Chunking: chunk = 60 outer iters; MAX_ROUNDS = 16 (safety cap, NOT a scientific gate)
- Certified plateau gate (UNCHANGED): last-4 cumulative spread(Gii) <= 3e-4 AND spread(Gtot) <= 3e-4
  AND R_phi <= 1.5e-3  (WINDOW = 4)
- Bias: Vg = Vd = 3 V, Vs = Vb = 0, Phi_gate = -0.740 V (recalibrated, Vth = 0.614 V)
- Mesh law: ratio = 6.0, W = 0.09 um, X_C = 0.60 um; trio nodes 71x58, 100x70, 141x85

## Numerical reproducibility note
The one-way SHE solve has an inherent run-to-run non-determinism of ~4e-6 (relative) in the
tail-sensitive Gii, from non-deterministic sparse linear algebra amplified by the near-null-space
character of Gii; the ORIGINAL code reproduces the stored reference only to this same 4.35e-6 with
identical phi. The CONVERGED trio fixed point is nonetheless stable to ~7 sig figs because the outer
loop damps this per-solve noise (verified: 2026-09-02 and 2026-09-24 cold runs both gave
Gii = 3.0756/2.1362/2.3874e25). The 3e-4 certified gate sits far above this noise floor.

## Port regression (40x34, this frozen version)
Gii_max rel 4.35e-6, Te_max rel 6.9e-7, n_max rel 7.6e-10, conservation -1.3e-7, SHE residual
1.2e-11 vs the stored certified observables -> within the inherent solve noise floor. PASS.
