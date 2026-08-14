"""
audit_result.py -- Numerical audit of the clean 40x34 SHE solve, extracting the exact
peak values/locations, terminal currents, continuity residual, orphan count, and the
pre-clip distribution range that a source-level reviewer asked for.

Re-solves with the no-impact-ionization preconditioner (the clean production path) so the
full solver state (raw f before clipping, conservation diagnostics) is available.
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from she2d import SHE2D, get_phi


def loc(field, x, y):
    j, i = np.unravel_index(int(np.argmax(field)), field.shape)
    return float(field[j, i]), float(x[i]), float(y[j])


Nx, Ny = 40, 34
x, y, phi, ctype, neq = get_phi(Nx, Ny, 3.0, 3.0)

she = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.0125,
            include_impact_ionization=True, absorbing_top=True)
she.assemble()
A, b = she.A, she.b
D = A.diagonal().copy(); D[np.abs(D) < 1e-300] = 1.0
As = (sp.diags(1.0 / D) @ A).tocsc(); bs = b / D

pre = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.0125,
            include_impact_ionization=False, absorbing_top=True)
pre.assemble()
Ap = pre.A; Dp = Ap.diagonal().copy(); Dp[np.abs(Dp) < 1e-300] = 1.0
ilu = spla.spilu((sp.diags(1.0 / Dp) @ Ap).tocsc(), drop_tol=1e-3, fill_factor=10)
M = spla.LinearOperator(As.shape, ilu.solve)
f_raw, info = spla.lgmres(As, bs, M=M, rtol=1e-9, maxiter=800)

# raw (pre-clip) distribution stats
scale = max(np.max(np.abs(f_raw)), 1.0)
neg_min = float(np.min(f_raw))
neg_count = int(np.count_nonzero(f_raw < -1e-12 * scale))

she.f_raw = np.nan_to_num(f_raw)
she.f = np.maximum(she.f_raw, 0.0)
she.info = info
she.scaled_residual_inf = float(np.max(np.abs(As.dot(she.f) - bs)) / max(np.max(np.abs(bs)), 1.0))
n, Te, Gii = she.moments()
cons = she.conservation_diagnostics()

nmax, nx, nyy = loc(n, x, y)
Temax, tex, tey = loc(Te, x, y)
Gmax, gx, gy = loc(Gii, x, y)
vmag = np.sqrt(she.vx**2 + she.vy**2)
vmax, vx_, vy_ = loc(vmag, x, y)
sep = float(np.hypot(tex - gx, tey - gy))

print("=" * 66)
print("NUMERICAL AUDIT  --  clean 40x34 SHE (with II collision operator)")
print("=" * 66)
print("1. n_max            %.4e cm^-3   at (x=%.3f, y=%.3f) um" % (nmax, nx, nyy))
print("2. Te_max           %.1f K          at (x=%.3f, y=%.3f) um  [baseline 3236 K]" % (Temax, tex, tey))
print("3. Gii_max          %.4e /cm^3/s at (x=%.3f, y=%.3f) um  [baseline 1.37e27]" % (Gmax, gx, gy))
print("4. |v|_max          %.4e cm/s    at (x=%.3f, y=%.3f) um  [old postproc 1.23e7]" % (vmax, vx_, vy_))
print("5. Te--Gii peak separation                      %.3f um" % sep)
print("6. SHE currents     I_source = %+.4e A/um   I_drain = %+.4e A/um   I_body = %+.3e" %
      (cons["source_current_A_per_um"], cons["drain_current_A_per_um"], cons["body_current_A_per_um"]))
print("   |I_src|-|I_drn| mismatch                     %.2f %%" %
      (100 * abs(abs(cons["source_current_A_per_um"]) - abs(cons["drain_current_A_per_um"]))
       / max(abs(cons["drain_current_A_per_um"]), 1e-30)))
print("7. normalized global continuity residual        %.3e" % cons["relative_number_balance"])
print("   (contact injection %.3e + II gen %.3e - cutoff loss %.3e per m/s)" %
      (cons["contact_particle_injection_per_m_s"], cons["ii_event_rate_per_m_s"], cons["cutoff_loss_per_m_s"]))
print("8. orphan states pinned                         %d" % she.n_orphans)
print("9. f0 raw (pre-clip): min = %.3e   max = %.3e" % (neg_min, float(np.max(f_raw))))
print("   negative entries beyond -1e-12*scale         %d of %d  (%.2e of DOF)" %
      (neg_count, she.Ndof, neg_count / she.Ndof))
print("   scaled linear residual                       %.3e (lgmres info=%d)" % (she.scaled_residual_inf, info))
print("=" * 66)
