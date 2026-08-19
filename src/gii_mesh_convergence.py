"""
gii_mesh_convergence.py -- 3-grid mesh convergence of the CONVERGED coupled G_ii,max.

Grids 40x34, 60x50, 80x66 (uniform tensor refinement, frozen Rev-5 methodology: AA(6), beta=0.5,
dH=12.5 meV, Phi_gate=-0.74). Each grid is a converged Anderson fixed point (per-grid plateau,
last-4 spread <1%). Fits G_ii = G0 + C h^p (h = dx = Lx/Nx) as an INDICATIVE 3-point Richardson
extrapolation of the continuum value -- shaky on a hotspot-sensitive moment, shown with that caveat.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Lx = 0.80
Nx = np.array([40, 60, 80]); Ny = np.array([34, 50, 66])
G = np.array([7.439e26, 2.286e26, 1.049e26])    # converged Gii_max per grid (trajectory-last)
spread = np.array([0.33, 0.21, 0.64])           # last-4 % spread (per-grid plateau)
res_mV = np.array([1.47, 1.26, 0.83])           # final fixed-point residual floor [mV]
h = Lx / Nx                                       # drain-direction spacing [um]

# 3-point fit G = G0 + C h^p (exact through the three converged points)
Robs = (G[0] - G[1]) / (G[1] - G[2])
Rp = lambda p: (h[0]**p - h[1]**p) / (h[1]**p - h[2]**p)
p = brentq(lambda p: Rp(p) - Robs, 0.3, 8.0)
C = (G[0] - G[1]) / (h[0]**p - h[1]**p)
G0 = G[2] - C * h[2]**p

print("successive Gii drops:  %.1f%% (40x34->60x50),  %.1f%% (60x50->80x66)"
      % (100 * (G[1]/G[0] - 1), 100 * (G[2]/G[1] - 1)))
print("indicative 3-pt fit:  p=%.2f  C=%.3e  G0(continuum)=%.3e  (=%.1f%% of Liang top 2.3e27)"
      % (p, C, G0, 100 * G0 / 2.3e27))
for i in range(3):
    print("  %2dx%-2d  h=%.4f um  Gii=%.3e  (%.1f%% of Liang top)  plateau spread %.2f%%  res %.2f mV"
          % (Nx[i], Ny[i], h[i], G[i], 100 * G[i] / 2.3e27, spread[i], res_mV[i]))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7))
hh = np.linspace(0, h[0] * 1.06, 250)
ax[0].plot(hh, (G0 + C * hh**p) / 1e26, "-", color="gray", lw=1.3,
           label="indicative fit $G_0+Ch^p$ ($p$=%.1f)" % p)
ax[0].plot(h, G / 1e26, "o", ms=9, color="#1f4e9e", label="converged per grid (plateau)")
ax[0].plot(0, G0 / 1e26, "*", ms=16, color="crimson",
           label="extrapolated $G_0\\approx$%.2f$\\times10^{26}$" % (G0 / 1e26))
for i in range(3):
    ax[0].annotate("%d$\\times$%d" % (Nx[i], Ny[i]), (h[i], G[i] / 1e26),
                   textcoords="offset points", xytext=(7, 7), fontsize=8.5)
ax[0].set_xlabel("drain-direction spacing  $h=\\Delta x=L_x/N_x$  [$\\mu$m]")
ax[0].set_ylabel("converged $G_{ii,\\max}$  [$10^{26}$ cm$^{-3}$s$^{-1}$]")
ax[0].set_title("(a) each grid converges, but $G_{ii}$ still falls with $h$")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_xlim(left=-0.0008)

ax[1].bar([0, 1], [100 * (G[1]/G[0] - 1), 100 * (G[2]/G[1] - 1)], 0.5, color="#c0392b")
for i, v in enumerate([100 * (G[1]/G[0] - 1), 100 * (G[2]/G[1] - 1)]):
    ax[1].text(i, v - 4, "%.0f%%" % v, ha="center", color="white", fontsize=11, fontweight="bold")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["40$\\times$34\n$\\to$60$\\times$50",
                                                 "60$\\times$50\n$\\to$80$\\times$66"])
ax[1].set_ylabel("$G_{ii}$ change to next grid [%]")
ax[1].set_title("(b) decelerating ($-69\\%\\to-54\\%$) but not yet converged")
ax[1].grid(alpha=0.3, axis="y")

fig.suptitle("Mesh convergence of the converged coupled $G_{ii}$ (3 grids, frozen Rev-5 method): "
             "per-grid convergence demonstrated; continuum not yet reached", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), "..", "figures", "gii_mesh_convergence.png")
fig.savefig(out, dpi=150)
print("wrote", os.path.relpath(out, os.path.join(os.path.dirname(__file__), "..")))
