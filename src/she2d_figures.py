"""
she2d_figures.py -- Render the SHE-derived moment maps (analogues of Liang Figs. 3, 4, 7)
from data/she2d_result.npz produced by she2d.py.
"""
import numpy as np
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_result.npz")
d = np.load(D)
x, y, phi, n, Te, Gii = d["x"], d["y"], d["phi"], d["n"], d["Te"], d["Gii"]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# Fig 3 analogue: electron concentration
a = ax[0, 0]
pcm = a.pcolormesh(x, y, np.log10(np.maximum(n, 1)), cmap="viridis", shading="auto", vmin=6, vmax=20)
a.invert_yaxis(); a.set_title("SHE electron concentration  log10 n  [cm$^{-3}$]  (cf. Fig. 3)")
a.set_xlabel("transverse x [um]"); a.set_ylabel("depth y [um]"); fig.colorbar(pcm, ax=a)

# potential for reference
a = ax[0, 1]
pcm = a.pcolormesh(x, y, phi, cmap="turbo", shading="auto")
a.invert_yaxis(); a.set_title("potential phi [V]  (from DD, drives SHE)  (cf. Fig. 5)")
a.set_xlabel("x [um]"); a.set_ylabel("y [um]"); fig.colorbar(pcm, ax=a)

# Fig 7 dashed analogue: electron temperature
a = ax[1, 0]
pcm = a.pcolormesh(x, y, Te, cmap="inferno", shading="auto")
a.invert_yaxis(); a.set_title("SHE electron temperature  T$_e$ [K]  (cf. Fig. 7 dashed)")
a.set_xlabel("x [um]"); a.set_ylabel("y [um]"); fig.colorbar(pcm, ax=a)
# contour overlay
try:
    cs = a.contour(x, y, Te, levels=[500, 1000, 2000, 3000, 4000, 5000], colors="w", linewidths=0.6)
    a.clabel(cs, fontsize=6, fmt="%d")
except Exception:
    pass

# Fig 7 solid analogue: impact-ionization generation rate
a = ax[1, 1]
G = np.log10(np.maximum(Gii, 1e18))
pcm = a.pcolormesh(x, y, G, cmap="magma", shading="auto")
a.invert_yaxis(); a.set_title("SHE impact-ionization G$_{ii}$  log10 [cm$^{-3}$s$^{-1}$]  (cf. Fig. 7 solid)")
a.set_xlabel("x [um]"); a.set_ylabel("y [um]"); fig.colorbar(pcm, ax=a)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_moments.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print("n: %.2e..%.2e  Te: %.0f..%.0f K  Gii max %.2e" % (n.min(), n.max(), Te.min(), Te.max(), Gii.max()))
