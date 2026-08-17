"""
she2d_liang_fig5_6.py -- Potential phi(x,y) [Liang Fig. 5] and hole concentration p(x,y)
[Liang Fig. 6] from the recalibrated coupled 60x50 solution, for the per-figure comparison.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_anderson_60x50.npz")  # converged Anderson result (Rev 5)
d = np.load(D)
x, y, phi, p = d["x"], d["y"], d["phi"], d["p"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

a = ax[0]
pcm = a.pcolormesh(x, y, phi, cmap="turbo", shading="gouraud")
cs = a.contour(x, y, phi, levels=np.linspace(0, 3.5, 8), colors="k", linewidths=0.5, alpha=0.6)
a.clabel(cs, fontsize=6, fmt="%.1f")
a.invert_yaxis(); a.set_title("$\\phi$(x,y) [V]  vs Liang Fig. 5"); fig.colorbar(pcm, ax=a)
a.set_xlabel("x [$\\mu$m]"); a.set_ylabel("y [$\\mu$m]")

a = ax[1]
pcm = a.pcolormesh(x, y, np.log10(np.maximum(p, 1.0)), cmap="magma", shading="gouraud",
                   vmin=6, vmax=17)
a.invert_yaxis(); a.set_title("hole $p$(x,y)  log$_{10}$[cm$^{-3}$]  vs Liang Fig. 6")
fig.colorbar(pcm, ax=a); a.set_xlabel("x [$\\mu$m]"); a.set_ylabel("y [$\\mu$m]")

fig.suptitle("Recalibrated coupled 60x50 ($V_{th}$=0.614 V): potential (Fig. 5) and holes (Fig. 6)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_liang_fig5_6.png")
fig.savefig(out, dpi=135); print("wrote", out)
print("phi swing %.2f..%.2f V ; p_max %.2e at (%.3f,%.3f)"
      % (phi.min(), phi.max(),
         p.max(), x[np.unravel_index(p.argmax(), p.shape)[1]],
         y[np.unravel_index(p.argmax(), p.shape)[0]]))
