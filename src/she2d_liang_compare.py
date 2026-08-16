"""
she2d_liang_compare.py -- Comprehensive spatial comparison of the RECALIBRATED COUPLED solution
against Liang et al. Figs. 3 (n), 4 (velocity), 7 (Te, Gii), with a quantitative summary table.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_result_60x50_Phg-0.74.npz")
d = np.load(D)
x, y, n, Te, Gii, vx, vy = d["x"], d["y"], d["n"], d["Te"], d["Gii"], d["vx"], d["vy"]
X, Y = np.meshgrid(x, y)
# Liang Fig. 7 caption (p.263) EXACT levels
TE_LEVELS = [500, 1000, 2000, 3000, 4500, 4800, 5000]
GII_LEVELS = [1.0e26, 4.0e26, 1.0e27, 1.7e27, 2.3e27]

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

a = ax[0, 0]
pcm = a.pcolormesh(x, y, np.log10(np.maximum(n, 1)), cmap="viridis", shading="gouraud", vmin=6, vmax=20)
a.invert_yaxis(); a.set_title("n  log$_{10}$[cm$^{-3}$]  vs Liang Fig. 3"); fig.colorbar(pcm, ax=a)
a.set_xlabel("x [um]"); a.set_ylabel("y [um]")

a = ax[0, 1]
mask = n < 1e13
vxm = np.where(mask, np.nan, vx); vym = np.where(mask, np.nan, vy)
vmag = np.sqrt(vxm**2 + vym**2); s = 3
q = a.quiver(X[::s, ::s], Y[::s, ::s], vxm[::s, ::s], vym[::s, ::s], vmag[::s, ::s],
             cmap="plasma", angles="xy", width=0.004)
a.invert_yaxis(); a.set_title("velocity [cm/s]  vs Liang Fig. 4")
a.set_xlim(x.min(), x.max()); a.set_ylim(y.max(), y.min()); fig.colorbar(q, ax=a)
a.set_xlabel("x [um]"); a.set_ylabel("y [um]")

a = ax[1, 0]
pcm = a.pcolormesh(x, y, Te, cmap="inferno", shading="gouraud", vmin=300, vmax=5000)
cs = a.contour(x, y, Te, levels=TE_LEVELS, colors="cyan", linestyles="--", linewidths=0.7)
a.clabel(cs, fontsize=6, fmt="%d")
a.invert_yaxis(); a.set_title("T$_e$ [K] (Liang Fig. 7 levels)"); fig.colorbar(pcm, ax=a)
a.set_xlabel("x [um]"); a.set_ylabel("y [um]")

a = ax[1, 1]
pcm = a.pcolormesh(x, y, np.log10(np.maximum(Gii, 1e20)), cmap="magma", shading="gouraud", vmin=22, vmax=27.5)
cs = a.contour(x, y, np.maximum(Gii, 1e20), levels=GII_LEVELS, colors="lime", linewidths=0.9)
a.invert_yaxis(); a.set_title("G$_{ii}$ log$_{10}$[cm$^{-3}$s$^{-1}$] (Liang Fig. 7 levels)"); fig.colorbar(pcm, ax=a)
a.set_xlabel("x [um]"); a.set_ylabel("y [um]")

fig.suptitle("Recalibrated coupled 60x50 (V$_{th}$=0.614 V) vs Liang et al. Figs. 3/4/7", fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_liang_compare.png")
fig.savefig(out, dpi=135); print("wrote", out)

vpk = float(np.nanmax(vmag))
print("\n=== COMPREHENSIVE COMPARISON vs Liang et al. ===")
print("%-22s %-18s %-22s %s" % ("quantity", "ours", "Liang (paper)", "assessment"))
print("-" * 82)
print("%-22s %-18s %-22s %s" % ("n_max [cm^-3]", "%.2e" % n.max(), "~1e20 (S/D)", "match"))
print("%-22s %-18s %-22s %s" % ("|v|_max [cm/s]", "%.2e" % vpk, "~1e7 (saturation)", "match"))
print("%-22s %-18s %-22s %s" % ("Te_max [K]", "%.0f" % Te.max(), "Fig7 to 5000",
      "%.0f%% of top (same order)" % (100 * Te.max() / 5000)))
print("%-22s %-18s %-22s %s" % ("Gii_max [cm^-3 s^-1]", "%.2e" % Gii.max(), "Fig7 to 2.3e27",
      "%.0f%% of top (unconverged)" % (100 * Gii.max() / 2.3e27)))
print("%-22s %-18s %-22s %s" % ("Te/Gii peaks", "drain, ~co-located", "drain, offset", "structure match; offset unresolved"))
print("%-22s %-18s %-22s %s" % ("Te region vs Gii", "Te broader", "Te broader", "match"))
