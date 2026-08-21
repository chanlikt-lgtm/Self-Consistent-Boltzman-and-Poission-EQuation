"""
coupled_drain_refine.py -- 4th spatial point: DRAIN-LOCAL mesh refinement of the converged coupled
G_ii (Rev 5 follow-up). EVERYTHING except the real-space mesh is frozen: Vth=0.614 V
(Phi_gate=-0.74), frozen H-grid, Anderson AA(6) beta=0.5, dH=12.5 meV, she_tol=1e-8, bias
Vg=Vd=3 V, collision model. The x-mesh is graded to concentrate cells at the drain hotspot
(x~0.60), giving ~4x finer drain-direction spacing than uniform 80x66 at a similar total cell count
-- the efficient refinement since G_ii is localized in a few drain cells.

Records (per the plan): G_ii,max and its (x,y); T_e,max; n_max; final residual max|G(phi)-phi|;
last-4 G_ii plateau spread; and the INTEGRATED generation G_tot = \int G_ii dA (less sensitive to
whether the narrow hotspot lands on one cell or its neighbour).
Usage: python coupled_drain_refine.py [Nx Ny ratio maxouter]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams
from coupled_she import solve_coupled

Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 80
Ny = int(sys.argv[2]) if len(sys.argv) > 2 else 70
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
maxo = int(sys.argv[4]) if len(sys.argv) > 4 else 45
X_C, W = 0.60, 0.09                                   # drain hotspot centre / width [um]

p = DeviceParams(); Lx, Ly = p.Lx, p.Ly

# --- drain-graded x-mesh: node density 1 + (ratio-1) Gaussian centred on the drain hotspot ---
xf = np.linspace(0.0, Lx, 6000)
dens = 1.0 + (ratio - 1.0) * np.exp(-((xf - X_C) / W) ** 2)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xf))])
cum /= cum[-1]
x = np.interp(np.linspace(0.0, 1.0, Nx), cum, xf); x[0], x[-1] = 0.0, Lx
y = Ly * (np.linspace(0.0, 1.0, Ny) ** 1.8)           # surface-refined (same law as make_mesh)

dx_drain = float(np.diff(x)[np.argmin(np.abs(0.5 * (x[:-1] + x[1:]) - X_C))])
print("drain-graded mesh: Nx=%d Ny=%d  min dx=%.4f um @ drain  (uniform 80x66 dx=%.4f)  min dy=%.5f um"
      % (Nx, Ny, dx_drain, Lx / 79, float(np.diff(y).min())), flush=True)
print("  total cells=%d (uniform 80x66=5280); drain dx is %.1fx finer than uniform 80x66"
      % (Nx * Ny, (Lx / 79) / dx_drain), flush=True)

r = solve_coupled(x_mesh=x, y_mesh=y, Phi_gate=-0.74, accel="anderson", aa_depth=6, aa_beta=0.5,
                  freeze_H=True, max_outer=maxo, tol_phi=1e-4, verbose=True)

h = r["history"]; G = np.array([rec["Gii_max_cm3s"] for rec in h])
Gii, xg, yg = r["Gii"], r["x"], r["y"]
jg, ig = np.unravel_index(int(np.argmax(Gii)), Gii.shape)

# integrated generation G_tot = \int G_ii dA  (finite-volume cell areas, um^2 -> cm^2)
dxc = np.zeros(len(xg)); dxc[1:-1] = 0.5 * (xg[2:] - xg[:-2]); dxc[0] = 0.5 * (xg[1] - xg[0]); dxc[-1] = 0.5 * (xg[-1] - xg[-2])
dyc = np.zeros(len(yg)); dyc[1:-1] = 0.5 * (yg[2:] - yg[:-2]); dyc[0] = 0.5 * (yg[1] - yg[0]); dyc[-1] = 0.5 * (yg[-1] - yg[-2])
G_tot = float(np.sum(Gii * np.outer(dyc, dxc) * 1e-8))

sp = (G[-4:].max() - G[-4:].min()) / G[-4:].mean()
print("\n=== DRAIN-REFINED 4th point (frozen H, Anderson) ===")
print("Gii_max = %.3e at (x=%.3f, y=%.3f) um" % (Gii.max(), xg[ig], yg[jg]))
print("Te_max  = %.0f K   n_max = %.3e" % (r["Te"].max(), r["n"].max()))
print("final residual = %.3e V   last-4 Gii plateau spread = %.2e" % (h[-1]["max_dphi_V"], sp))
print("G_tot = int Gii dA = %.4e cm^-1 s^-1  (integrated generation)" % G_tot)
print("vs uniform 80x66 Gii_max=1.47e26:  Delta_mesh = %.1f%%" % (100 * abs(Gii.max() - 1.47e26) / Gii.max()))
np.savez(os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_anderson_drainrefined.npz"),
         x=xg, y=yg, phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=Gii, vx=r["vx"], vy=r["vy"],
         F3d=r["F3d"].astype(np.float32), H=r["H"], Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"])
print("saved data/she2d_coupled_anderson_drainrefined.npz")
