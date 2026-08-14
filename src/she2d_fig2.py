"""
she2d_fig2.py -- Reproduce Liang Fig. 2: the electron distribution function f0(eps) along
three depths (0.001, 0.2, 0.64 um), as log10 f0 surfaces vs (transverse x, kinetic energy).
This is the quantity ONLY the Boltzmann solution provides.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

eV = 1.602176634e-19
D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_result.npz")
d = np.load(D)
x, y, phi, F3d, H = d["x"], d["y"], d["phi"], d["F3d"], d["H"]
Nx = len(x)

depths = [0.001, 0.20, 0.64]          # um (paper's three lines)
labels = ["(a) 0.001 um below surface (channel)",
          "(b) 0.20 um below surface (LDD)",
          "(c) 0.64 um below surface (substrate)"]

eps_axis = np.linspace(0.02, 3.0, 90)  # kinetic energy [eV]

fig = plt.figure(figsize=(16, 5.2))
for panel, (yl, lab) in enumerate(zip(depths, labels)):
    j = int(np.argmin(np.abs(y - yl)))
    Fmap = np.zeros((Nx, len(eps_axis)))
    for i in range(Nx):
        H_need = (eps_axis - phi[j, i]) * eV           # J
        Fmap[i, :] = np.interp(H_need, H, F3d[j, i, :], left=0.0, right=0.0)
    # normalize so the peak (low energy) is ~1; light smoothing in x tames grid noise
    Fmap = Fmap / (Fmap.max() + 1e-300)
    sm = Fmap.copy()
    sm[1:-1, :] = 0.25 * Fmap[:-2, :] + 0.5 * Fmap[1:-1, :] + 0.25 * Fmap[2:, :]
    logF = np.clip(np.log10(np.maximum(sm, 1e-28)), -28, 0.2)

    XX, EE = np.meshgrid(x, eps_axis, indexing="ij")
    ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
    ax.plot_surface(XX, EE, logF, cmap=cm.viridis, rstride=1, cstride=2,
                    linewidth=0, antialiased=True, vmin=-28, vmax=0)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("transverse x [um]", fontsize=8)
    ax.set_ylabel("kinetic energy [eV]", fontsize=8)
    ax.set_zlabel("log$_{10}$ f$_0$", fontsize=8)
    ax.set_zlim(-30, 2)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=7)

fig.suptitle("SHE electron distribution function f$_0$($\\varepsilon$)  (cf. Liang Fig. 2)  "
             "$V_{gs}=V_{ds}=3$ V", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_fig2_distribution.png")
fig.savefig(out, dpi=135)
print("wrote", out)

# quick numeric sanity: the drain-side channel line should have a hotter (fatter) tail than
# the deep-substrate line
for yl in depths:
    j = int(np.argmin(np.abs(y - yl)))
    i_drain = int(np.argmin(np.abs(x - 0.6)))
    col = F3d[j, i_drain, :]
    if col.max() > 0:
        # crude tail metric: fraction of f0 above eps_kin = 1 eV
        eps_k = H / eV + phi[j, i_drain]
        hot = col[(eps_k > 1.0)].sum() / (col.sum() + 1e-300)
        print("  y=%.3f um, x=0.6: hot-tail fraction (eps>1eV) = %.2e" % (yl, hot))
