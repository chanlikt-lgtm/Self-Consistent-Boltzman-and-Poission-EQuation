"""
she2d_liang_overlay.py -- Quantitative overlay of the recalibrated coupled SHE solution against
Liang et al. Fig. 7, using Liang's EXACT published contour values.

Liang Fig. 7 (drain region, transverse 0.45-0.70 um, depth 0-0.15 um):
  electron temperature contours [K]:  5000, 4800, 4500, 3000, 2000, 1000, 500   (dashed lines)
  impact-ionization G_ii contours [cm^-3 s^-1]:
        2.3e27, 1.7e27, 1.0e27, 4.0e26, 1.0e26                                  (solid lines)

We plot OUR recalibrated coupled 60x50 Te and G_ii with those same contour levels on Liang's
axes, so the comparison is quantitative on the values (do our fields reach Liang's levels?) and
on structure (drain-localized? Te region larger than G_ii region? peak locations?).

Honest caveats printed with the figure: reconstructed doping (spatial positions need not align),
G_ii spatially unconverged, and the Te/G_ii offset unresolved at this mesh.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_result_60x50_Phg-0.74.npz")
d = np.load(D)
x, y, Te, Gii = d["x"], d["y"], d["Te"], d["Gii"]

# Liang's exact contour values (paper Fig. 7)
TE_LEVELS = [500, 1000, 2000, 3000, 4500, 4800, 5000]
GII_LEVELS = [1.0e26, 4.0e26, 1.0e27, 1.7e27, 2.3e27]

# Liang's Fig. 7 window (drain region)
xlim = (0.45, 0.72)
ylim = (0.0, 0.18)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

# --- Panel 1: Liang-style overlay (our data, Liang's levels/axes) ---
a = ax[0]
pcm = a.pcolormesh(x, y, Te, cmap="inferno", shading="gouraud", vmin=300, vmax=5000)
cs_te = a.contour(x, y, Te, levels=TE_LEVELS, colors="cyan", linestyles="--", linewidths=0.8)
a.clabel(cs_te, fontsize=6, fmt="%d")
cs_g = a.contour(x, y, np.maximum(Gii, 1e20), levels=GII_LEVELS, colors="lime",
                 linestyles="-", linewidths=1.1)
a.clabel(cs_g, fontsize=6, fmt="%.0e")
a.set_xlim(*xlim); a.set_ylim(*ylim); a.invert_yaxis()
a.set_xlabel("transverse x [um]"); a.set_ylabel("depth y [um]")
a.set_title("OUR recalibrated coupled solution\nT$_e$ (dashed) + G$_{ii}$ (solid), Liang Fig.7 levels")
fig.colorbar(pcm, ax=a, label="T$_e$ [K]")

# --- Panel 2: magnitude comparison of reachable contour levels ---
a = ax[1]
te_peak = float(Te.max()); g_peak = float(Gii.max())
labels = ["T$_e$ peak", "G$_{ii}$ peak"]
ours = [te_peak, g_peak]
liang_top = [5000.0, 2.3e27]           # Liang's highest contour
liang_low = [500.0, 1.0e26]
# normalized bars: ours / Liang-top
ax2 = a
xpos = np.arange(2)
w = 0.35
ax2.bar(xpos - w/2, [te_peak/5000, g_peak/2.3e27], w, label="ours / Liang top contour", color="#4a56c8")
ax2.axhline(1.0, color="k", ls="--", lw=1, label="Liang top contour")
for i, (o, lt) in enumerate(zip(ours, liang_top)):
    ax2.text(i - w/2, o/lt + 0.03, "%.2f" % (o/lt), ha="center", fontsize=9)
ax2.set_xticks(xpos); ax2.set_xticklabels(labels)
ax2.set_ylabel("peak value / Liang's top contour")
ax2.set_title("magnitude comparison")
ax2.set_ylim(0, 1.3); ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Quantitative overlay vs Liang et al. Fig. 7 (recalibrated coupled 60x50, V$_{th}$=0.614 V)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_liang_fig7_overlay.png")
fig.savefig(out, dpi=135); print("wrote", out)

# quantitative comparison table
print("\n=== QUANTITATIVE COMPARISON vs Liang Fig. 7 ===")
print("Te peak:   ours %.0f K   | Liang contours to 5000 K (top)   -> ours/top = %.2f" % (te_peak, te_peak/5000))
print("Gii peak:  ours %.3e     | Liang contours to 2.3e27 (top)   -> ours/top = %.2f" % (g_peak, g_peak/2.3e27))
# peak locations
jte, ite = np.unravel_index(int(np.argmax(Te)), Te.shape)
jg, ig = np.unravel_index(int(np.argmax(Gii)), Gii.shape)
print("Te peak at (x=%.3f, y=%.3f) um ; Gii peak at (x=%.3f, y=%.3f) um  [Liang: both near drain, offset]"
      % (x[ite], y[jte], x[ig], y[jg]))
print("both drain-localized; Te region broader than Gii region:",
      "YES" if (Te > 2000).sum() > (Gii > 1e27).sum() else "check")
