"""
coupled_drain_continue.py -- CONTINUE the drain-refined 4th point to a true plateau.

The first drain-refined run (coupled_drain_refine.py, 100x70) hit max_outer=40 with G_ii,max still
monotonically decreasing (~-2%/iter, last-4 spread 11%, residual 3.3 mV), so its 1.68e25 is an UPPER
bound, not a plateau. This warm-starts the SAME mesh from that run's saved (phi, p) and runs more
outer iterations so the drift can flatten. Everything else is byte-identical to the 4th run: frozen
H, Anderson AA(6) beta=0.5, dH=12.5 meV, she_tol=1e-8, Phi_gate=-0.74, Vg=Vd=3 V, same graded mesh.
Usage: python coupled_drain_continue.py [maxouter]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams
from coupled_she import solve_coupled

maxo = int(sys.argv[1]) if len(sys.argv) > 1 else 60

# Load the EXACT mesh and near-converged state from the 4th run's saved npz. Using the stored mesh
# (rather than rebuilding it) avoids any ratio/recipe drift -- the original run used ratio=6.
prev = np.load(os.path.join(os.path.dirname(__file__), "..", "data",
                            "she2d_coupled_anderson_drainrefined.npz"))
x, y = prev["x"], prev["y"]
print("continuing drain-refined %dx%d from saved plateau-tail (warm-start phi,p); maxouter=%d"
      % (len(x), len(y), maxo), flush=True)

r = solve_coupled(x_mesh=x, y_mesh=y, Phi_gate=-0.74, accel="anderson", aa_depth=6, aa_beta=0.5,
                  freeze_H=True, max_outer=maxo, tol_phi=1e-4, verbose=True,
                  phi_init=prev["phi"], p_init=prev["p"])

h = r["history"]; G = np.array([rec["Gii_max_cm3s"] for rec in h])
Gii, xg, yg = r["Gii"], r["x"], r["y"]
jg, ig = np.unravel_index(int(np.argmax(Gii)), Gii.shape)
dxc = np.zeros(len(xg)); dxc[1:-1] = 0.5 * (xg[2:] - xg[:-2]); dxc[0] = 0.5 * (xg[1] - xg[0]); dxc[-1] = 0.5 * (xg[-1] - xg[-2])
dyc = np.zeros(len(yg)); dyc[1:-1] = 0.5 * (yg[2:] - yg[:-2]); dyc[0] = 0.5 * (yg[1] - yg[0]); dyc[-1] = 0.5 * (yg[-1] - yg[-2])
G_tot = float(np.sum(Gii * np.outer(dyc, dxc) * 1e-8))
sp = (G[-4:].max() - G[-4:].min()) / G[-4:].mean()
print("\n=== DRAIN-REFINED 4th point CONTINUED (frozen H, Anderson, warm-start) ===")
print("Gii_max = %.3e at (x=%.3f, y=%.3f) um" % (Gii.max(), xg[ig], yg[jg]))
print("Te_max  = %.0f K   n_max = %.3e" % (r["Te"].max(), r["n"].max()))
print("final residual = %.3e V   last-4 Gii plateau spread = %.2e" % (h[-1]["max_dphi_V"], sp))
print("G_tot = int Gii dA = %.4e cm^-1 s^-1" % G_tot)
print("Gmax trajectory (this continuation):", " ".join("%.3e" % g for g in G))
np.savez(os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_anderson_drainrefined_cont.npz"),
         x=xg, y=yg, phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=Gii, vx=r["vx"], vy=r["vy"],
         F3d=r["F3d"].astype(np.float32), H=r["H"], Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"])
print("saved data/she2d_coupled_anderson_drainrefined_cont.npz")
