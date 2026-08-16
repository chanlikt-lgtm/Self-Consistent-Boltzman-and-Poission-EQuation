"""
coupled_converge.py -- Continue the recalibrated coupled 60x50 solve toward the declared 2 mV
max|dphi| stopping criterion (audit request), then report whether observables are stable to a
tighter tolerance. Same physical case as she2d_coupled_result_60x50_Phg-0.74; only the number of
outer iterations (and the relaxation parameter) changes -- the model is untouched.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from coupled_she import solve_coupled

DAMP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
MAXOUTER = int(sys.argv[2]) if len(sys.argv) > 2 else 50

r = solve_coupled(Nx=60, Ny=50, Phi_gate=-0.74, max_outer=MAXOUTER,
                  tol_phi=2e-3, phi_damp=DAMP, verbose=True)

outdir = os.path.join(os.path.dirname(__file__), "..", "data")
np.savez(os.path.join(outdir, "she2d_coupled_result_60x50_Phg-0.74_conv.npz"),
         x=r["x"], y=r["y"], phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=r["Gii"],
         vx=r["vx"], vy=r["vy"], F3d=r["F3d"].astype(np.float32), H=r["H"],
         Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"])

h = r["history"]
print("\n=== convergence tail (last 6 outers) ===")
for rec in h[-6:]:
    print("it %2d  dphi=%.3e V  Temax=%.1f  Gmax=%.4e  Id=%.4e  nmax=%.4e" %
          (rec["iteration"], rec["max_dphi_V"], rec["Te_max_K"], rec["Gii_max_cm3s"],
           rec["drain_current_A_per_um"], rec["n_max_cm3"]))
final = h[-1]
print("\nreached 2mV?", final["max_dphi_V"] < 2e-3, " final dphi=%.3e V in %d outers" %
      (final["max_dphi_V"], len(h)))
# stability of observables over the last few iterations (relative spread)
import numpy as _np
for key in ("Te_max_K", "Gii_max_cm3s", "drain_current_A_per_um", "n_max_cm3"):
    vals = _np.array([rec[key] for rec in h[-4:]])
    spread = (vals.max() - vals.min()) / abs(vals.mean()) if vals.mean() else 0.0
    print("  %-22s last4 rel.spread = %.2e" % (key, spread))
print("saved data/she2d_coupled_result_60x50_Phg-0.74_conv.npz")
