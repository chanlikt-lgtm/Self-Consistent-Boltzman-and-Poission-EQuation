# SHE reproduction audit patch

This patch addresses the code-level findings from the source audit of the supplied Liang/Goldsman reproduction.

## Patched

1. **DOS-weighted acoustic conservation documented and regression-tested**
   - `scattering.py` now states the implemented conservative operator explicitly:
     `-d/deps [ Z A (df0/deps + f0/kT) ]`.
   - Added `check_number_conservation()`.
   - Clarified that `tau1` is a calibrated combined surrogate, not a separately assembled Table-I acoustic/impurity/surface model.

2. **True absorbing upper kinetic-energy cutoff in the 2-D SHE assembly**
   - `eps=eps_max` is no longer an active unknown.
   - Spatial fixed-H crossings through the upper cutoff add a Dirichlet-to-zero boundary flux.
   - Acoustic Fokker-Planck flux through the upper cutoff adds a Scharfetter-Gummel Dirichlet-to-zero loss.
   - Optical absorption above the cutoff is retained as a loss even when the destination is outside the global H array.
   - Lower `eps=0` remains reflecting.

3. **Finite-volume velocity/current extraction**
   - Velocity is reconstructed from the same face-centered `ZD (f_L-f_R)/dx` flux used by the SHE operator.
   - Added source/drain/body terminal-current extraction from exact finite-volume contact links.
   - Added global particle-number balance diagnostics and cutoff-loss decomposition.

4. **Impact ionization is now an actual SHE collision process**
   - Added Keldysh-like `impact_ionization_rate()` in `scattering.py` using the disclosed `Eth=1.10 eV`, `P=2e13 1/s` surrogate.
   - Each event removes one primary electron and deposits two electrons at `eps'=(eps-Eth)/2`, linearly distributed onto neighboring H nodes.
   - The discrete redistribution has net electron gain +1 per event.
   - `Gii` remains available as a moment/diagnostic of the solved distribution.

5. **Hole-continuity generation source and coupled driver**
   - `dd.py` accepts an external pair-generation term and adds `solve_holes()`.
   - `coupled_she.py` performs a damped outer iteration:
     SHE -> hole continuity with `Gii` -> Poisson -> SHE.
   - The Poisson step uses effective quasi-Fermi potentials only as a robust approximate Newton/Gummel Jacobian; the electron density is recomputed from the SHE after each potential update.
   - `Poisson2D.solve_fixed_charge()` is included as a diagnostic raw-Picard step, but is not used by the coupled driver because it is poorly conditioned in strong inversion.

6. **Fig. 2 tail metric corrected**
   - `she2d_fig2.py` now reports the physical carrier-tail fraction using `Z f0` weighting.
   - The legacy unweighted `f0` metric is printed only for comparison.
   - The script now explicitly labels per-panel normalization and lateral smoothing as visualization-only operations.

7. **Small reproducibility fixes**
   - `diagnostics.py` uses the extracted `Vth=1.64 V` default, giving the intended ~0.285 mA/um analytical estimate.
   - `vth_extract.py` fixes the sign of `dVth/dPhi_gate` to approximately `+1 V/V`.
   - Added `validate_patch.py` regression/sanity checks.

## Validation performed

`python validate_patch.py` passes on a 12x10 device smoke test with 25-meV H spacing:

- optical detailed-balance residual: `2.064e-05`
- optical number-conservation defect: `2.244e-16`
- acoustic number-conservation defect: `7.625e-17`
- scaled SHE linear residual: `2.820e-16`
- global SHE number-balance residual (gross-flow normalized): `-5.031e-10`
- drain current: `1.469269e-04 A/um`
- peak `Te`: `1701.5 K`
- peak `Gii`: `4.751316e27 cm^-3 s^-1`
- DOS-weighted / legacy raw tail fraction at the smoke-test point: `1.934183e-02 / 1.919180e-03`
- corrected analytical DD current estimate: `2.851301e-04 A/um`

A 12x10, three-outer-iteration coupled smoke test was also stable: potential updates decreased from roughly 10.4 mV to 8.6 mV per damped iteration and the discrete SHE number balance stayed at ~1e-8 or better.

## Still requires production validation

- Run the full 40x34x~H grid with the patched physics to convergence. A larger 24x20 / 12.5-meV test exceeded the interactive execution time available during this patching session; this is a performance/validation limitation, not a demonstrated numerical failure.
- Perform real-space and H-grid convergence studies.
- Converge the full SHE/Poisson/hole outer loop and study damping sensitivity.
- Recalibrate the reconstructed device before claiming quantitative contour agreement.
- The momentum-relaxation model is still a calibrated surrogate rather than explicit Brooks-Herring + acoustic + surface-roughness rates.
- The band and impact-ionization models remain disclosed surrogates.
