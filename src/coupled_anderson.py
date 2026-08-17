"""
coupled_anderson.py -- Anderson-accelerated coupled SHE<->Poisson<->hole solve.

Goal: reach the nonlinear fixed point that the damped Picard iteration could not. In the 50-iteration
Picard study (coupled_converge.py) Gii fell monotonically with no plateau. Anderson acceleration
AA(depth) with mixing beta reuses a short residual history to drive the TRUE fixed-point residual
max|G(phi)-phi| -> 0; if a stable fixed point exists, Gii should then plateau.

Usage: python coupled_anderson.py [Nx Ny depth beta max_outer]
Reports the raw fixed-point residual and the Gii trajectory each iteration, and saves the result.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from coupled_she import solve_coupled

Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
Ny = int(sys.argv[2]) if len(sys.argv) > 2 else 34
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 6
beta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
maxo = int(sys.argv[5]) if len(sys.argv) > 5 else 40
freeze_H = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False

r = solve_coupled(Nx=Nx, Ny=Ny, Phi_gate=-0.74, accel="anderson",
                  aa_depth=depth, aa_beta=beta, max_outer=maxo, tol_phi=1e-4, verbose=True,
                  freeze_H=freeze_H)

h = r["history"]
G = np.array([rec["Gii_max_cm3s"] for rec in h])
res = np.array([rec["max_dphi_V"] for rec in h])
final = h[-1]
print("\n=== Anderson AA(%d) beta=%.2f  %dx%d (Phi_gate=-0.74)%s ===" %
      (depth, beta, Nx, Ny, "  [frozen H-grid]" if freeze_H else ""))
print("reached tol 1e-4?  %s   final raw residual = %.3e V  in %d iters"
      % (final["max_dphi_V"] < 1e-4, final["max_dphi_V"], len(h)))
print("residual: first=%.3e -> last=%.3e" % (res[0], res[-1]))
print("Gii_max : first=%.3e -> last=%.3e" % (G[0], G[-1]))
if len(G) >= 4:
    sp = (G[-4:].max() - G[-4:].min()) / G[-4:].mean()
    print("Gii last-4 relative spread = %.2e  (PLATEAU if small; Picard 50-iter stayed ~4e-2, still falling)" % sp)
print("Te_max=%.0f K  Id=%.3e A/um  n_max=%.3e" %
      (final["Te_max_K"], final["drain_current_A_per_um"], final["n_max_cm3"]))

outdir = os.path.join(os.path.dirname(__file__), "..", "data")
tag = "_frozenH" if freeze_H else ""
np.savez(os.path.join(outdir, "she2d_coupled_anderson_%dx%d%s.npz" % (Nx, Ny, tag)),
         x=r["x"], y=r["y"], phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=r["Gii"],
         vx=r["vx"], vy=r["vy"], F3d=r["F3d"].astype(np.float32), H=r["H"],
         Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"])
print("saved data/she2d_coupled_anderson_%dx%d%s.npz" % (Nx, Ny, tag))
print("consistency: saved Gii_max %.4e == trajectory-last %.4e (same iterate; residual %.3e V)"
      % (float(r["Gii"].max()), G[-1], final["max_dphi_V"]))
