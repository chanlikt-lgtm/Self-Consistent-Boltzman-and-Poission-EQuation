"""
she2d_figures.py -- SHE moment maps (Figs. 3, 4, 7 analogues) from data/she2d_result.npz.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_result.npz")
d = np.load(D)
x, y, phi, n, Te, Gii = d["x"], d["y"], d["phi"], d["n"], d["Te"], d["Gii"]
vx, vy = d["vx"], d["vy"]
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# Fig 3: electron concentration
a = ax[0, 0]
pcm = a.pcolormesh(x, y, np.log10(np.maximum(n, 1)), cmap="viridis", shading="gouraud", vmin=6, vmax=20)
a.invert_yaxis(); a.set_title("electron concentration  log$_{10}$ n [cm$^{-3}$]  (cf. Fig. 3)")
a.set_xlabel("transverse x [um]"); a.set_ylabel("depth y [um]"); fig.colorbar(pcm, ax=a)

# Fig 4: average-velocity vector field
a = ax[0, 1]
vmag = np.sqrt(vx**2 + vy**2)
s = 2
q = a.quiver(X[::s, ::s], Y[::s, ::s], vx[::s, ::s], vy[::s, ::s], vmag[::s, ::s],
             cmap="plasma", scale_units="xy", angles="xy", width=0.004)
a.invert_yaxis(); a.set_title("average electron velocity  [cm/s]  (cf. Fig. 4)")
a.set_xlabel("transverse x [um]"); a.set_ylabel("depth y [um]")
a.set_xlim(x.min(), x.max()); a.set_ylim(y.max(), y.min()); fig.colorbar(q, ax=a)

# Fig 7 dashed: electron temperature
a = ax[1, 0]
pcm = a.pcolormesh(x, y, Te, cmap="inferno", shading="gouraud")
a.invert_yaxis(); a.set_title("electron temperature  T$_e$ [K]  (cf. Fig. 7 dashed)")
a.set_xlabel("x [um]"); a.set_ylabel("y [um]")
try:
    cs = a.contour(x, y, Te, levels=[500, 1000, 2000, 3000, 4000], colors="cyan", linewidths=0.6)
    a.clabel(cs, fontsize=6, fmt="%d")
except Exception:
    pass
fig.colorbar(pcm, ax=a)

# Fig 7 solid: impact-ionization generation rate
a = ax[1, 1]
G = np.log10(np.maximum(Gii, 1e18))
pcm = a.pcolormesh(x, y, G, cmap="magma", shading="gouraud", vmin=20, vmax=27.5)
a.invert_yaxis(); a.set_title("impact-ionization G$_{ii}$  log$_{10}$[cm$^{-3}$s$^{-1}$]  (cf. Fig. 7 solid)")
a.set_xlabel("x [um]"); a.set_ylabel("y [um]"); fig.colorbar(pcm, ax=a)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_moments.png")
fig.savefig(out, dpi=135); print("wrote", out)
print("n %.2e..%.2e  Te %.0f..%.0f K  |v|max %.2e cm/s  Gii max %.2e" %
      (n.min(), n.max(), Te.min(), Te.max(), np.sqrt(vx**2+vy**2).max(), Gii.max()))
