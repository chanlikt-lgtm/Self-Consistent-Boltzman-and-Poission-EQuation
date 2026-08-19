"""
she2d_paper_compare_figs.py -- Standalone single-quantity figures of the recalibrated coupled
60x50 solution, rendered to visually parallel Liang et al.'s Figs. 3-7 for a side-by-side
comparison: 3D wireframe surfaces for n (Fig.3), phi (Fig.5), p (Fig.6); a 2D velocity quiver
(Fig.4); and Te(dashed)+Gii(solid) contours on Liang's drain-region axes/levels (Fig.7).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

D = os.path.join(os.path.dirname(__file__), "..", "data", "she2d_coupled_anderson_80x66.npz")  # converged Anderson result, finest grid (Rev 5)
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
d = np.load(D)
x, y, n, phi, p = d["x"], d["y"], d["n"], d["phi"], d["p"]
Te, Gii, vx, vy = d["Te"], d["Gii"], d["vx"], d["vy"]
X, Y = np.meshgrid(x, y)

VIEW = dict(elev=24, azim=-66)


def wire(Z, zlabel, title, out, zunit=1.0):
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z / zunit, rstride=1, cstride=1, color="k", linewidth=0.35)
    ax.set_xlabel("Transverse Distance [$\\mu$m]", fontsize=9)
    ax.set_ylabel("Depth into Device [$\\mu$m]", fontsize=9)
    ax.set_zlabel(zlabel, fontsize=9)
    ax.view_init(**VIEW)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, out), dpi=140)
    plt.close(fig)
    print("wrote", out)


# Fig 3 -- electron concentration (units 1e19 cm^-3, peak ~10 = 1e20)
wire(n, "n [$10^{19}$cm$^{-3}$]", "This work: electron concentration (cf. Liang Fig. 3)",
     "our_fig3.png", zunit=1e19)

# Fig 5 -- electrostatic potential [V]
wire(phi, "Electric Potential [V]", "This work: potential (cf. Liang Fig. 5)", "our_fig5.png")

# Fig 6 -- hole concentration (units 1e15 cm^-3)
wire(p, "p [$10^{15}$cm$^{-3}$]", "This work: hole concentration (cf. Liang Fig. 6)",
     "our_fig6.png", zunit=1e15)

# Fig 4 -- average velocity quiver (masked where n<1e13), depth downward
fig, ax = plt.subplots(figsize=(6.2, 5.0))
mask = n < 1e13
vxm = np.where(mask, np.nan, vx); vym = np.where(mask, np.nan, vy)
s = 2
ax.quiver(X[::s, ::s], Y[::s, ::s], vxm[::s, ::s], vym[::s, ::s], color="k",
          angles="xy", scale=2.2e8, width=0.003)
ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.max(), 0.0)
ax.set_xlabel("Transverse Distance [$\\mu$m]   (Source | Gate | Drain)")
ax.set_ylabel("Depth into Device [$\\mu$m]")
ax.set_title("This work: average electron velocity (cf. Liang Fig. 4)", fontsize=10, pad=10)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "our_fig4.png"), dpi=140); plt.close(fig)
print("wrote our_fig4.png")

# Fig 7 -- Te (dashed) + Gii (solid) contours on Liang's drain-region axes and levels
TE = [500, 1000, 2000, 3000, 4500, 4800, 5000]
GII = [1.0e26, 4.0e26, 1.0e27, 1.7e27, 2.3e27]
fig, ax = plt.subplots(figsize=(6.2, 5.0))
cte = ax.contour(x, y, Te, levels=TE, colors="k", linestyles="--", linewidths=0.8)
cg = ax.contour(x, y, np.maximum(Gii, 1e20), levels=GII, colors="k", linestyles="-", linewidths=1.1)
ax.clabel(cte, fontsize=6, fmt="%d"); ax.clabel(cg, fontsize=6, fmt="%.0e")
ax.set_xlim(0.45, 0.70); ax.set_ylim(0.15, 0.0)
ax.set_xlabel("Transverse Distance [$\\mu$m]"); ax.set_ylabel("Depth into Device [$\\mu$m]")
ax.set_title("This work: T$_e$ (dashed) + G$_{ii}$ (solid) (cf. Liang Fig. 7)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "our_fig7.png"), dpi=140); plt.close(fig)
print("wrote our_fig7.png")
