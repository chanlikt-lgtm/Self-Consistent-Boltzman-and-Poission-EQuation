"""
grid_refine.py -- Controlled grid-refinement experiment against the locked 40x34 reference
(commit f016e3c). No physics changes: same models, same dH, same absorbing/II settings.
Uses the clean no-impact-ionization ILU preconditioner path.

Usage: python grid_refine.py Nx Ny [dHi_meV]
Reports the publication-grade audit table (units on balance terms, relative continuity
residual, precise % change vs the reference) and the Te / Gii peak coordinates so the
mesh-resolution question on their separation can be settled.
"""
import sys, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from she2d import SHE2D, get_phi

REF_TE, REF_GII = 3232.9, 1.3513e27   # 40x34 locked reference (f016e3c)


def _peak(field, x, y):
    j, i = np.unravel_index(int(np.argmax(field)), field.shape)
    return float(field[j, i]), float(x[i]), float(y[j]), j, i


def run(Nx, Ny, dHi_eV=0.0125):
    t_all = time.time()
    x, y, phi, ctype, neq = get_phi(Nx, Ny, 3.0, 3.0)
    dx_drain = float(np.diff(x)[np.argmin(np.abs(x - 0.6))]) * 1e3  # nm near drain
    dy_surf = float(y[1] - y[0]) * 1e3                              # nm at surface

    she = SHE2D(x, y, phi, ctype, neq, dHi_eV=dHi_eV,
                include_impact_ionization=True, absorbing_top=True)
    she.assemble()
    t0 = time.time()
    she.solve(tol=1e-10, method="auto")   # fast physics-split preconditioner
    tsolve = time.time() - t0
    ps = getattr(she, "preconditioner_stats", {})
    tbuild = float(ps.get("energy_setup_s", 0.0) + ps.get("space_setup_s", 0.0))
    f_raw = she.f_raw
    n, Te, Gii = she.moments()
    cons = she.conservation_diagnostics()

    Temax, tex, tey, tj, ti = _peak(Te, x, y)
    Gmax, gx, gy, gj, gi = _peak(Gii, x, y)
    vmag = np.sqrt(she.vx**2 + she.vy**2); vmax, vx_, vy_, _, _ = _peak(vmag, x, y)
    sep_um = float(np.hypot(tex - gx, tey - gy))
    sep_cells = abs(ti - gi) + abs(tj - gj)

    print("=" * 70)
    print("GRID-REFINEMENT AUDIT  %dx%d  dH=%.2f meV  (DOF=%d, ILU %.0fs, lgmres %.1fs)"
          % (Nx, Ny, dHi_eV * 1e3, she.Ndof, tbuild, tsolve))
    print("  mesh near drain: dx=%.1f nm  dy(surface)=%.1f nm" % (dx_drain, dy_surf))
    print("-" * 70)
    print("Te_max    %.1f K   at (%.3f, %.3f) um   (%.2f%% vs ref 3236 K)"
          % (Temax, tex, tey, 100 * (Temax - 3236.0) / 3236.0))
    print("Gii_max   %.4e /cm^3/s  at (%.3f, %.3f) um   (%.2f%% vs ref 1.37e27)"
          % (Gmax, gx, gy, 100 * (Gmax - 1.37e27) / 1.37e27))
    print("|v|_max   %.4e cm/s  at (%.3f, %.3f) um" % (vmax, vx_, vy_))
    print("n_max     %.4e cm^-3" % n.max())
    print("Te--Gii peak separation   %.3f um  = %d cell(s)  (mesh dx~%.1f nm near drain)"
          % (sep_um, sep_cells, dx_drain))
    print("SHE currents   I_src=%+.4e  I_drn=%+.4e A/um   mismatch %.2f%%"
          % (cons["source_current_A_per_um"], cons["drain_current_A_per_um"],
             100 * abs(abs(cons["source_current_A_per_um"]) - abs(cons["drain_current_A_per_um"]))
             / max(abs(cons["drain_current_A_per_um"]), 1e-30)))
    print("global continuity residual (relative, dimensionless)   %.3e"
          % cons["relative_number_balance"])
    print("  balance terms [1/(m s)]: contact_inj=%.3e  II_gen=%.3e  cutoff_loss=%.3e"
          % (cons["contact_particle_injection_per_m_s"], cons["ii_event_rate_per_m_s"],
             cons["cutoff_loss_per_m_s"]))
    print("orphans=%d   f0 raw min=%.3e max=%.3e   scaled residual=%.3e"
          % (she.n_orphans, float(np.min(f_raw)), float(np.max(f_raw)), she.scaled_residual_inf))
    print("total wall time %.0fs" % (time.time() - t_all))
    print("=" * 70)
    return dict(Nx=Nx, Ny=Ny, Temax=Temax, Gmax=Gmax, sep_um=sep_um, sep_cells=sep_cells)


if __name__ == "__main__":
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    Ny = int(sys.argv[2]) if len(sys.argv) > 2 else 34
    dHi = (float(sys.argv[3]) / 1000.0) if len(sys.argv) > 3 else 0.0125
    run(Nx, Ny, dHi)
