# Reproduction of Liang, Goldsman, Mayergoyz & Oldiges (1997)
**"2-D MOSFET Modeling Including Surface Effects and Impact Ionization by Self-Consistent
Solution of the Boltzmann, Poisson, and Hole-Continuity Equations"**, IEEE TED 44(2), 257-267.

Goal: **derive** the method from first principles and **fully reproduce all 2-D/3-D plots** (Figs. 2-8).

## Method summary
Deterministic first-order **Spherical-Harmonic Expansion (SHE)** of the electron BTE, coupled
self-consistently to 2-D Poisson and the hole-continuity equation, via a damped Gummel loop.
Independent variable in energy space is total energy H = eps - q*phi (H-transform).
Device: 0.35-um effective-channel LDD N-MOSFET, t_ox = 9.6 nm, x_j ~ 0.17 um.

## Key inputs (Table I, paper)
- D_ac  = 4.0 eV           (acoustic deformation potential)
- D_n   = 5.0e8 eV/cm      (optical/intervalley deformation potential)
- E_op  = 0.05 eV          (optical phonon energy)
- v_s   = 9.0e6 cm/s       (saturation velocity param)
- Gamma = 1.96 cm^2/V^2 s  (surface-roughness lumped param)
- n_scr = 1.45e15 1/cm^3   (screening density for ionized-impurity)
- D_sa  = 17.8 eV          (surface acoustic deformation potential)
- delta = 13.1             (quantum term param, surface roughness)
- eps_max = 3.02 eV, energy grid 0..3.02 eV, 2 bands of gamma(eps) [Fiegna/Brunetti]

## HONESTY NOTES
- Doping profile is RECONSTRUCTED (ref [25] Khalil 1994 not in hand): n+/n-/psub ~ 1e20/1e18/1e16.
  Any curve depending on exact doping (esp. Vth, subthreshold slope in Fig 8) is qualitative.
- Impact-ionization rate 1/tau_ii from ref [23] (Wu-Goldsman random-k); using a Keldysh/Cartier-type
  parameterization as stand-in unless the exact table is reconstructed. Flagged in code + report.

## Phases / status
- [x] P0  Environment + workspace
- [x] P1  Derivation doc (LaTeX): BTE -> SHE -> f0 balance eq -> SG discretization -> moments
        [DONE: report/report.tex -> report.pdf (Part I). Derivation validated internally:
         boxed f1 relation, conservative (x,eps) balance, H-transform form, detailed-balance
         optical operator, moment defs. Compiles clean (pdflatex x2).]
- [x] P2  Device module: mesh + reconstructed LDD doping  (src/device.py)  [VALIDATED: figures/device_doping.png]
- [x] P3  Nonlinear 2-D Poisson + drift-diffusion baseline (src/poisson.py, src/dd.py)
        [VALIDATED: poisson_equilibrium.png (built-in +0.586/-0.44V, quadratic Newton);
         dd_bias.png (inversion layer, pinch-off at Vg=Vd=3, Id~0.27 mA/um -- right order)]
- [x] P4  Band structure gamma(eps), DOS, group velocity (src/bands.py)
        [VALIDATED: Nc=2.785e19 vs Si 2.8e19 (0.5%); v(0.1eV)=3.1e7, v(1eV)=6.4e7 cm/s]
- [~] P5  First-order SHE core (src/she2d.py) -- FIRST WORKING SOLVE (commit 40d5929).
        (r,H) form; spatial diffusion per H-plane + optical/acoustic collision; contact
        Maxwellian injection on DD phi. Bugs fixed: singular matrix (pin orphans) + blow-up
        to 1e56 (Jacobi row-scaling: contact diag=1 vs interior ~1e23 -> cond 1e23).
        RESULT (coarse 32x26, E_op=2): n~1e20 inversion+pinch-off (Fig 3 OK); Te peaks at
        DRAIN to 3184K (Fig 7 dashed OK); G_ii~1e27 at drain, offset from Te peak (Fig 7
        solid OK, matches paper contour values 1-2.3e27). Hot-carrier physics VALIDATED.
        TODO: refine grid (E_op>=4), Fig 2 (distribution slices), Fig 4 (velocity), Fig 8
        (I-V), quantitative contour comparison, self-consistent SHE->Poisson loop.
- [ ] P6  Scattering operators (src/scattering.py)
- [ ] P7  Impact ionization + hole continuity (src/holes.py)
- [ ] P8  Self-consistent Gummel driver (src/solve.py)
- [ ] P9  Moment extraction (n, v, Te, G_ii) + ALL figures (src/figures_*.py)
- [ ] P10 LaTeX PDF report assembling derivation + reproduced figures + validation

## AUDIT 4 (full 28-page source-level audit of combined report) + PATCH INTEGRATION, 2026-08-15
Auditor ran the actual code (reproduced n=1.0516e20, Te=3235.95K, Gii=1.37085e27, orphans=0).
BLOCKER (acoustic Eq.6 conservation) RESOLVED: code implements conservative Z-weighted form
(-d/deps[ZA(d/deps+1/kT)f0]); only report Eq.6 dropped the Z (doc typo). Auditor then PROVIDED
A PATCH (she_reproduction_patched.zip + audit_patch.diff), reviewed + integrated (validate_patch.py
PASSES in-place: optical/acoustic conservation defects 2e-16, SHE number balance -1e-9).
Patch changes (7 files + coupled_she.py, validate_patch.py, PATCH_NOTES.md):
- scattering.py: documents conservative acoustic form; check_number_conservation(); impact_ionization_rate();
  tau1 relabeled combined-surrogate.
- she2d.py: TRUE absorbing eps_max cutoff (spatial+acoustic+optical, Dirichlet-to-zero); REAL impact-
  ionization collision operator (primary loss + 2 secondaries at (eps-Eth)/2, net +1 e per event);
  finite-volume face-flux velocity/current; terminal currents + conservation_diagnostics().
- dd.py: solve_holes() with G_ii pair-generation source.
- coupled_she.py: NEW damped self-consistent SHE<->hole<->Poisson loop (recomputes n from SHE each step).
- poisson.py: solve_fixed_charge()/solve_holes support. diagnostics.py: Vth=1.64 default.
  vth_extract.py: dVth/dPhi_gate sign +1. she2d_fig2.py: DOS-weighted (Zf0) tail fraction.
Superseded my partial manual patches (same fixes).
LARGE-GRID SOLVE FIX: II gain terms break M-matrix -> spilu singular on 40x34. Fixed via no-II
preconditioner (run_she_clean.py / grid_refine.py): ILU on the clean transport matrix (no II),
lgmres on the full system -> 0.9s, residual 1.5e-9, NO bulk artifacts. Commit 84e4065.
**PATCHED DD->SHE BASELINE: PASS (visual+numerical), auditor-confirmed. LOCKED REF = commit f016e3c.**
Numerical audit (audit_result.py, data/audit_result.log = regression benchmark):
  n_max 1.0516e20 @source; Te_max 3232.9K @drain(0.595,0.005) [0.10% vs 3236]; Gii_max 1.3513e27
  @drain(0.595,0.009) [1.4% vs 1.37e27]; |v|_max 9.732e6 cm/s (FV, below old postproc 1.23e7);
  Isrc=-1.7044e-4/Idrn=+1.7071e-4 A/um (0.16% mismatch); continuity residual -5.7e-6 (relative);
  0 orphans; f0 pre-clip min -3.3e-11 (46/328544 tiny negatives).
HONEST: Te-Gii peaks 5nm apart = WITHIN one cell (mesh ~20nm) -> co-located at this resolution,
NOT a resolved offset. Needs grid refinement to resolve Liang's offset.
II STATUS: coupled to electron BTE (in collision operator, affects f0), NOT yet self-consistent
with Poisson/holes.
NEXT (auditor plan, one change at a time, compare vs f016e3c): (1) GRID REFINE 60x50 dH=12.5meV
[IN PROGRESS] - check if Te/Gii peak separation persists; (2) then self-consistency (coupled_she.py
damped); (3) energy check dH=6.25meV (E_op=8). Then report revision + target-figure overlays.

## AUDIT (Rev 1 -> Rev 2), 2026-08-14
External audit verdict on report Rev 1: "promising Part I; NOT reproduction-grade yet."
8 findings, ALL addressed in report.tex Rev 2 (report.pdf, 8 pages):
- F1 "full momentum-space" overclaim -> "energy-resolved + first-order angular anisotropy"
- F2 missing specs -> Appendix A: all BCs (incl. gate oxide Robin BC), stats, mobility law+params,
     SRH, mesh, tolerances, damping, current-sign convention
- F3 junction 0.24 vs 0.17 -> DEFINED x_j = metallurgical crossing; RETUNED to 0.170um (device.py:
     xj_nplus=0.105, sig_dep=0.025). Vth calibration + taxonomy documented.
- F4 H-transform vs SG inconsistency -> device solve stated purely in (r,H); SG only stabilizes
     SPATIAL flux, no energy-space cross-flux.
- F5 Nc not auditable -> REAL FIX: Rev1's 2.785e19 was UNDER-CONVERGED (400-pt grid). Converged
     (4000-pt) Nc=2.857e19 at stated T=300K, matches auditor 2.86e19 to 0.1%; parabolic limit vs
     closed-form 0.09% (prefactor proven). bands.py updated.
- F6 II electron bookkeeping -> derived: primary lost + 2 carriers re-injected at (eps-eps_th)/2,
     net dn=dp=G_ii (charge conserved).
- F7 "validated" -> "sanity-checked"; benchmark table (expected vs computed); DD current honesty:
     long-channel 0.96 vs DD 0.25 mA/um, gap = velocity saturation (NOT "matching" as Rev1 said).
- F8 no refs -> References section (13 refs) + code/param provenance appendix.
New modules this round: scattering.py (optical operator, detailed-balance 2e-5),
she_bulk.py (bulk hot-electron validation: Te 282->4696K over 0-300kV/cm), diagnostics.py.

## Notes / decisions
- SHE will use the KINETIC-ENERGY first-order form (Gnudi/Ventura/Baccarani refs [19,20])
  rather than the paper's H-transform. Same physics/moments; far more transparent to
  discretize on a rectangular (x,y,eps) grid. Documented as a numerical-scheme deviation.
- DD Gummel converges linearly & slowly at high bias (~80 iters). Fine for operating point;
  may add acceleration for full I-V sweeps.
- Metallurgical S/D junction ~0.24um vs paper 0.17um; retune device.py geometry in calibration.
- Phi_gate=0.30V flatband offset is a calibration knob for V_th (doping is reconstructed).

## Figures to reproduce
- Fig 2(a,b,c): log f distribution vs (energy, transverse x) at y = 0.001, 0.2, 0.64 um
- Fig 3: electron concentration n(x,y), Vgs=Vds=3V
- Fig 4: electron average-velocity vector field
- Fig 5: electrostatic potential phi(x,y)
- Fig 6: hole concentration p(x,y)
- Fig 7: electron-temperature contours (dashed) + impact-ionization G_ii contours (solid)
- Fig 8(a): Id-Vgs subthreshold, Vds=0.25,0.5V (log)
- Fig 8(b): Id-Vds, Vgs=2.0,2.5,3.0V
