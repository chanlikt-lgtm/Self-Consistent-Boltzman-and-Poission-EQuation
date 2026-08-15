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
    A, b = she.A, she.b
    D = A.diagonal().copy(); D[np.abs(D) < 1e-300] = 1.0
    As = (sp.diags(1.0 / D) @ A).tocsc(); bs = b / D

    pre = SHE2D(x, y, phi, ctype, neq, dHi_eV=dHi_eV,
                include_impact_ionization=False, absorbing_top=True)
    pre.assemble()
    Ap = pre.A; Dp = Ap.diagonal().copy(); Dp[np.abs(Dp) < 1e-300] = 1.0
    Aps = (sp.diags(1.0 / Dp) @ Ap).tocsc()
    # robust ILU: finer/larger grids can give spilu a zero pivot even for the clean no-II
    # M-matrix (aggressive drop_tol); escalate a diagonal shift on the PRECONDITIONER only.
    t0 = time.time(); ilu = None; pshift = None
    for s in (0.0, 1e-3, 1e-2, 5e-2, 1e-1):
        try:
            Ash = Aps if s == 0.0 else (Aps + s * sp.eye(As.shape[0], format="csc")).tocsc()
            ilu = spla.spilu(Ash, drop_tol=1e-4, fill_factor=12); pshift = s; break
        except Exception as e:
            print("  no-II ILU shift=%.0e failed (%s)" % (s, e))
    if ilu is None:
        raise RuntimeError("no-II preconditioner ILU failed at all shifts")
    tbuild = time.time() - t0
    print("  no-II preconditioner ILU built with shift=%.0e in %.0fs" % (pshift, tbuild))
    M = spla.LinearOperator(As.shape, ilu.solve)
    t0 = time.time()
    f_raw, info = spla.lgmres(As, bs, M=M, rtol=1e-9, maxiter=1000)
    tsolve = time.time() - t0

    she.f_raw = np.nan_to_num(f_raw)
    she.f = np.maximum(she.f_raw, 0.0)
    she.info = info
    she.scaled_residual_inf = float(np.max(np.abs(As.dot(she.f) - bs)) / max(np.max(np.abs(bs)), 1.0))
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
