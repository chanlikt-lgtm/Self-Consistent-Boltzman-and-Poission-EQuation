"""
coupled_drain_refine5.py -- 5th spatial point: a FINER drain-graded mesh than the 4th (100x70),
to test whether the ~8x Gii drop from 80x66->drain-refined finally decelerates or keeps falling.

Everything except the real-space mesh is frozen (identical to the 4th run): Phi_gate=-0.74, frozen H,
Anderson AA(6) beta=0.5, dH=12.5 meV, she_tol=1e-8, Vg=Vd=3 V. Mesh: Nx=140 Ny=84 ratio=10 centred
on the Gii peak x~0.59, giving drain dx ~2x finer than the 4th grid. Warm-started by separable 1-D
interpolation of the converged 100x70 (phi, p) onto this mesh, so the loop finishes near the plateau.
Usage: python coupled_drain_refine5.py [Nx Ny ratio maxouter]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams
from coupled_she import solve_coupled

Nx    = int(sys.argv[1]) if len(sys.argv) > 1 else 140
Ny    = int(sys.argv[2]) if len(sys.argv) > 2 else 84
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
maxo  = int(sys.argv[4]) if len(sys.argv) > 4 else 55
X_C, W = 0.59, 0.06                                   # drain hotspot centre / width [um]

p = DeviceParams(); Lx, Ly = p.Lx, p.Ly
xf = np.linspace(0.0, Lx, 8000)
dens = 1.0 + (ratio - 1.0) * np.exp(-((xf - X_C) / W) ** 2)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xf))]); cum /= cum[-1]
x = np.interp(np.linspace(0.0, 1.0, Nx), cum, xf); x[0], x[-1] = 0.0, Lx
y = Ly * (np.linspace(0.0, 1.0, Ny) ** 1.8)

dx_drain = float(np.diff(x)[np.argmin(np.abs(0.5 * (x[:-1] + x[1:]) - X_C))])
print("5th drain-graded mesh: Nx=%d Ny=%d  min dx=%.5f um @ drain (4th grid dx=0.0027)  cells=%d"
      % (Nx, Ny, dx_drain, Nx * Ny), flush=True)

# --- warm-start: separable 1-D interp of the converged 4th-grid (phi,p) onto this mesh ---
prev = np.load(os.path.join(os.path.dirname(__file__), "..", "data",
                            "she2d_coupled_anderson_drainrefined_cont.npz"))
xo, yo = prev["x"], prev["y"]
def regrid(F):                                        # F shape (len(yo), len(xo)) -> (Ny, Nx)
    tmp = np.empty((len(yo), Nx))
    for j in range(len(yo)):
        tmp[j] = np.interp(x, xo, F[j])
    out = np.empty((Ny, Nx))
    for i in range(Nx):
        out[:, i] = np.interp(y, yo, tmp[:, i])
    return out
phi0 = regrid(prev["phi"])          # keep 2-D (Ny, Nx): solve_coupled works with 2-D phi/p
p0   = regrid(prev["p"])
print("warm-started by interpolating converged 100x70 (phi,p) onto %dx%d" % (Nx, Ny), flush=True)

r = solve_coupled(x_mesh=x, y_mesh=y, Phi_gate=-0.74, accel="anderson", aa_depth=6, aa_beta=0.5,
                  freeze_H=True, max_outer=maxo, tol_phi=1e-4, verbose=True,
                  phi_init=phi0, p_init=p0)

h = r["history"]; G = np.array([rec["Gii_max_cm3s"] for rec in h])
Gii, xg, yg = r["Gii"], r["x"], r["y"]
jg, ig = np.unravel_index(int(np.argmax(Gii)), Gii.shape)
dxc = np.zeros(len(xg)); dxc[1:-1] = 0.5 * (xg[2:] - xg[:-2]); dxc[0] = 0.5 * (xg[1] - xg[0]); dxc[-1] = 0.5 * (xg[-1] - xg[-2])
dyc = np.zeros(len(yg)); dyc[1:-1] = 0.5 * (yg[2:] - yg[:-2]); dyc[0] = 0.5 * (yg[1] - yg[0]); dyc[-1] = 0.5 * (yg[-1] - yg[-2])
G_tot = float(np.sum(Gii * np.outer(dyc, dxc) * 1e-8))
sp = (G[-4:].max() - G[-4:].min()) / G[-4:].mean()
print("\n=== DRAIN-REFINED 5th point (frozen H, Anderson, warm-start) ===")
print("Gii_max = %.3e at (x=%.3f, y=%.3f) um" % (Gii.max(), xg[ig], yg[jg]))
print("Te_max  = %.0f K   n_max = %.3e" % (r["Te"].max(), r["n"].max()))
print("final residual = %.3e V   last-4 Gii plateau spread = %.2e" % (h[-1]["max_dphi_V"], sp))
print("G_tot = int Gii dA = %.4e cm^-1 s^-1" % G_tot)
print("Gmax trajectory:", " ".join("%.3e" % g for g in G))
np.savez(os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_anderson_drainrefined5.npz"),
         x=xg, y=yg, phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=Gii, vx=r["vx"], vy=r["vy"],
         F3d=r["F3d"].astype(np.float32), H=r["H"], Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"])
print("saved data/she2d_coupled_anderson_drainrefined5.npz")
