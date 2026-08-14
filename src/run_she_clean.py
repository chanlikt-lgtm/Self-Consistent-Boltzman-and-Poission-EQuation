"""
run_she_clean.py -- Full 40x34 SHE solve with a CLEAN preconditioner.

The impact-ionization gain terms break the M-matrix structure and make spilu singular on the
full grid, forcing a weak diagonal-shifted preconditioner that leaves bulk solve-noise.
Fix: build the ILU preconditioner from the transport+scattering+absorbing matrix WITHOUT
impact ionization (a well-conditioned M-matrix, same node set / size), then solve the FULL
system (with impact ionization) by LGMRES using that preconditioner. II is a weak
perturbation (only above Eth~1.1 eV), so this converges cleanly.
"""
import os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from she2d import SHE2D, get_phi

Nx, Ny = 40, 34
x, y, phi, ctype, neq = get_phi(Nx, Ny, 3.0, 3.0)
print("phi range %.3f..%.3f V" % (phi.min(), phi.max()))

# system WITH impact ionization (what we actually solve)
she = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.0125,
            include_impact_ionization=True, absorbing_top=True)
she.assemble()
A, b = she.A, she.b
D = A.diagonal().copy(); D[np.abs(D) < 1e-300] = 1.0
As = (sp.diags(1.0 / D) @ A).tocsc(); bs = b / D
print("assembled WITH-II: DOF=%d nnz=%d orphans=%d" % (she.Ndof, A.nnz, she.n_orphans))

# preconditioner from the NO-II matrix (clean M-matrix -> stable ILU)
pre = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.0125,
            include_impact_ionization=False, absorbing_top=True)
pre.assemble()
Ap = pre.A; Dp = Ap.diagonal().copy(); Dp[np.abs(Dp) < 1e-300] = 1.0
Aps = (sp.diags(1.0 / Dp) @ Ap).tocsc()
t0 = time.time()
ilu = spla.spilu(Aps, drop_tol=1e-3, fill_factor=10)
print("no-II ILU built in %.1fs" % (time.time() - t0))
M = spla.LinearOperator(As.shape, ilu.solve)

t0 = time.time()
f, info = spla.lgmres(As, bs, M=M, rtol=1e-9, maxiter=800)
print("lgmres info=%d in %.1fs" % (info, time.time() - t0))

# install solution and extract moments through the class machinery
she.f_raw = np.nan_to_num(f)
she.f = np.maximum(she.f_raw, 0.0)
she.info = info
she.scaled_residual_inf = float(np.max(np.abs(As.dot(she.f) - bs)) / max(np.max(np.abs(bs)), 1.0))
n, Te, Gii = she.moments()
print("scaled residual %.2e" % she.scaled_residual_inf)
print("n %.2e..%.2e  Te %.0f..%.0f K  |v|max %.2e cm/s  Gii max %.2e" %
      (n.min(), n.max(), Te.min(), Te.max(),
       np.sqrt(she.vx**2 + she.vy**2).max(), Gii.max()))
cons = she.conservation_diagnostics()
print("Isrc=%.3e Idrn=%.3e A/um  number-balance %.2e" %
      (cons["source_current_A_per_um"], cons["drain_current_A_per_um"],
       cons["relative_number_balance"]))

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
np.savez(os.path.join(data_dir, "she2d_result.npz"),
         x=x, y=y, phi=phi, n=n, Te=Te, Gii=Gii, vx=she.vx, vy=she.vy,
         F3d=she.F3d.astype(np.float32), H=she.H,
         Gamma_x_face=she.Gamma_x_face, Gamma_y_face=she.Gamma_y_face)
print("saved data/she2d_result.npz")
