# SHE solver acceleration patch

This patch replaces the large-grid full 3-D shifted ILU with a physics-split multiplicative preconditioner:

- **E block:** all same-spatial-cell energy couplings, including acoustic, optical, impact-ionization redistribution, and the spatial diagonal.
- **S block:** the diagonal plus cross-cell couplings, i.e. the fixed-H real-space transport part.
- One preconditioner application performs **E^-1 -> S^-1 -> E^-1**, with a fresh residual between stages.
- The actual linear system remains unshifted. Any emergency diagonal shift is applied only to a preconditioner factor.

`solve(method="auto")` uses direct sparse solve below 130k DOF and the split ILU above it. `method="legacy_ilu"` keeps the previous full-graph ILU for regression comparison.

The default solve tolerance is tightened to `1e-10` so final audit runs retain near-nonnegativity and conservation after tiny numerical negatives are clipped. For intermediate coupled iterations, `tol=1e-8` can be used if speed matters more than the final conservation audit.

## Benchmarks in this sandbox

Single-threaded (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`):

### Exact 40x34 problem, 328,544 DOF

- Assembly: ~5.5 s
- E-factor setup: ~0.9 s
- S-factor setup: ~0.6 s
- LGMRES solve at `tol=1e-10`: ~8.6 s
- Outer LGMRES iterations: 6
- Raw scaled residual: `1.22e-11`
- Relative global number balance after clipping: `1.27e-7`
- `n_max = 1.05159609e20 cm^-3`
- `Te_max = 3232.90 K`
- `Gii_max = 1.35134e27 cm^-3 s^-1`
- `|v|_max = 9.73182e6 cm/s`

The old full shifted global ILU did not finish its **preconditioner setup** within 120 s on this same 328k matrix in the sandbox.

### Representative 60x50-size problem, 724,831 DOF

For performance testing only, the 40x34 converged potential was interpolated onto 60x50; therefore this is a solver-performance benchmark, not a replacement for the user's physical 60x50 result.

- Assembly: ~13.3 s
- E-factor setup: ~2.2-2.6 s
- S-factor setup: ~1.7 s
- LGMRES solve at `tol=1e-10`: ~21.6 s
- Outer LGMRES iterations: 6
- Raw scaled residual: `3.23e-11`

This directly targets the global-ILU bottleneck while preserving the exact matrix being solved.
