# Liang 1997 SHE-BTE reproduction — immutable closeout bundle

Audit-ready package for citing this reproduction later without reconstructing the git history.
Generated 2026-09-26. **Liang is frozen reference work from here on** — do not extend this campaign;
see the note at the end for the only sanctioned follow-up.

## Frozen solver
- Repo `liang1997_she_bte_mosfet_repro`, branch `richardson-trio-recheck-closeout`
- **Solver tag `solver-frozen-2026-09-26`** (provenance commit `f6a8f91`; code commit `7576ce0`)
- Audit commit (this bundle generated at): `457a021`
- Environment: see `ENVIRONMENT.txt` (Python 3.13.15 / NumPy 2.5.1 / SciPy 1.18.0, Windows 11)

## Contents
| file | what it is |
|---|---|
| `README_CLOSEOUT.md` | this file |
| `FROZEN_SOLVER_PROVENANCE.md` | frozen solver commit + env + runtime settings + noise note |
| `report_richardson_audit.pdf` | the grid-convergence / Richardson audit (2 pp) |
| `she2d_richardson_71x58.npz` | trio node 1 result (Gii_max 3.0756e25) |
| `she2d_richardson_100x70.npz` | trio node 2 result (Gii_max 2.1362e25) |
| `she2d_richardson_141x85.npz` | trio node 3 result (Gii_max 2.3874e25) |
| `ENVIRONMENT.txt` | version + runtime manifest |
| `SHA256SUMS.txt` | integrity hashes (verify: `sha256sum -c SHA256SUMS.txt`) |

## Provenance (verbatim)
The three trio NPZs were **produced by pre-freeze revision `d30b471`**, then validated as
**numerically equivalent to `solver-frozen-2026-09-26`** within the solver's own run-to-run noise
(~4e-6 relative on Gii) and well inside the 3e-4 certification gate (40x34 port regression:
Gii rel 4.35e-6). They are NOT relabelled as frozen-tag outputs.

## Scientific verdict (from the audit)
- All three drain-refined nodes are individually **converged, gate-PASS, conservation-respecting**
  fixed points (spread<=3e-4, R_phi<=1.5e-3, |bal|<=2e-4, SHE residual ~1e-8).
- Under sqrt(2) refinement the drain peak Gii is **non-monotone**:
  3.0756 -> 2.1362 -> 2.3874 e25 cm^-3 s^-1 (-30.5% then +11.8%).
- **Richardson extrapolation is NOT justified:** (Q1-Q2)/(Q2-Q3) = -3.74 has no real order p under
  the one-term model Q(h)=Q*+C h^p; S_rev=737>>1 shows the reversal is genuine, not solver noise. No
  continuum Gii / GCI is quoted (it would be meaningless from a sign-reversing sequence).
- **Two conclusions kept separate:** (a) the ABSOLUTE peak Gii is grid-limited (bounded
  [2.14, 3.08]e25, finest-mesh 2.39e25; not demonstrably asymptotic); (b) the QUALITATIVE physics is
  ROBUST — the non-monotone reversal itself, drain-localized avalanche, ~37% self-consistent
  feedback suppression, and the transport regime (n~1e20, |v|~1e7 cm/s, Te~3180 K).

## Only sanctioned follow-up
Any future continuum-Gii effort must be a **separate study** using a second mesh family
(**fixed y-mesh, refine drain-x**) to disentangle the anisotropic x/y discretization error and
re-test for a consistent order p. It is NOT an extension of this closed campaign.
