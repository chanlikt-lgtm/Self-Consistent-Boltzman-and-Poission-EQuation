"""
she2d_converge_fig.py -- Convergence-study figure for the recalibrated coupled 60x50 loop
(coupled_converge.py, 50 outer iterations toward the 2 mV criterion). Parses the run log and the
final npz to show that n and Id stabilize while G_ii does NOT plateau (monotonic decrease), and
that the outer loop did not reach the 2 mV stopping criterion.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
log = open(os.path.join(ROOT, "data", "coupled_conv.log")).read()

pat = re.compile(r"outer\s+(\d+)\s+dphi=([\d.eE+-]+) V\s+nmax=([\d.eE+-]+)\s+pmax=([\d.eE+-]+)\s+"
                 r"Temax=([\d.]+) K\s+Gmax=([\d.eE+-]+)\s+Id=([\d.eE+-]+)")
it, dphi, nmax, Temax, Gmax, Id = [], [], [], [], [], []
for m in pat.finditer(log):
    it.append(int(m.group(1))); dphi.append(float(m.group(2))); nmax.append(float(m.group(3)))
    Temax.append(float(m.group(5))); Gmax.append(float(m.group(6))); Id.append(float(m.group(7)))
it = np.array(it); dphi = np.array(dphi); nmax = np.array(nmax)
Temax = np.array(Temax); Gmax = np.array(Gmax); Id = np.array(Id)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
a = ax[0, 0]
a.semilogy(it, dphi * 1e3, "o-", ms=3); a.axhline(2.0, color="r", ls="--", lw=1, label="2 mV criterion")
a.set_ylabel("max$|\\Delta\\phi|$ [mV]"); a.set_title("(a) potential update -- did NOT reach 2 mV"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
a = ax[0, 1]
a.plot(it, Gmax / 1e27, "o-", ms=3, color="crimson")
a.set_ylabel("$G_{ii,\\max}$ [$10^{27}$cm$^{-3}$s$^{-1}$]"); a.set_title("(b) $G_{ii}$ -- monotonic decrease, NO plateau"); a.grid(alpha=0.3)
a = ax[1, 0]
a.plot(it, Temax, "o-", ms=3, color="darkorange")
a.set_xlabel("outer iteration"); a.set_ylabel("$T_{e,\\max}$ [K]"); a.set_title("(c) $T_e$ -- slow drift down"); a.grid(alpha=0.3)
a = ax[1, 1]
a.plot(it, nmax / 1e20, "o-", ms=3, color="teal", label="$n_{\\max}$/$10^{20}$")
a.plot(it, Id / 1e-4, "s-", ms=3, color="navy", label="$I_d$/$10^{-4}$A/$\\mu$m")
a.set_xlabel("outer iteration"); a.set_title("(d) $n$, $I_d$ -- essentially stable"); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.suptitle("Damped Picard baseline (superseded): $G_{ii}$ still drifting after 50 outer "
             "iterations (recalibrated coupled 60$\\times$50, damp 0.35)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "she2d_converge_60x50.png"), dpi=135)
print("wrote figures/she2d_converge_60x50.png")

# final-field peaks from the conv npz
d = np.load(os.path.join(ROOT, "data", "she2d_coupled_result_60x50_Phg-0.74_conv.npz"))
x, y, n, Te, Gii, vx, vy, p, phi = d["x"], d["y"], d["n"], d["Te"], d["Gii"], d["vx"], d["vy"], d["p"], d["phi"]
mask = n < 1e13
vmag = np.sqrt(np.where(mask, np.nan, vx) ** 2 + np.where(mask, np.nan, vy) ** 2)
print("=== 50-iter converged-as-far-as-it-went snapshot (dphi=%.3f mV) ===" % (dphi[-1] * 1e3))
print("n_max   = %.3e" % n.max())
print("|v|_max = %.3e cm/s" % np.nanmax(vmag))
print("Te_max  = %.0f K  (%.0f%% of Liang top 5000K)" % (Te.max(), 100 * Te.max() / 5000))
print("Gii_max = %.3e  (%.0f%% of Liang top 2.3e27)  -- STILL DROPPING" % (Gii.max(), 100 * Gii.max() / 2.3e27))
print("Gii trajectory: it0=%.2e -> it49=%.2e  (x%.2f, monotone, no plateau)" % (Gmax[0], Gmax[-1], Gmax[-1] / Gmax[0]))
print("Id      = %.3e A/um" % Id[-1])
print("one-way 40x34 Gii=1.35e27 -> coupled-50iter %.2e = feedback %.0f%% (was quoted -37%%; still growing)"
      % (Gii.max(), 100 * (Gii.max() / 1.35e27 - 1)))
