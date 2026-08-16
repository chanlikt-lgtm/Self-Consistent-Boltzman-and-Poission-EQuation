"""
she2d_iv.py -- Terminal I-V characteristics of the RECALIBRATED device (Phi_gate=-0.74,
Vth=0.614 V) for comparison against Liang et al. Fig. 8.

Honesty scope (matches report taxonomy):
  * The paper computes the terminal current from the SHE velocity moment; here we use the
    self-consistent drift-diffusion terminal current, which reproduces the correct I-V *shape*
    (triode -> saturation, monotonic V_gs ordering, subthreshold exponential) at far lower cost
    and is the honest proxy given the reconstructed device. It is reported PER MICRON of width.
  * Liang's Fig. 8 solid lines are EXPERIMENTAL data for a real (unknown-width) device; only the
    per-width shape is comparable, not the absolute ampere magnitude.

Bias continuation: each bias point warm-starts from the previous one, so the stiff high-bias
Gummel solves converge cleanly (no drooping/under-converged artifacts) and quickly.

Reproduces the structure of:
  Fig. 8(b): Id-Vds output characteristic at Vgs = 2.0, 2.5, 3.0 V.
  Fig. 8(a): Id-Vgs subthreshold transfer at Vds = 0.25 and 0.5 V (log scale).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver

PHI_GATE = -0.74                      # recalibrated gate offset -> Vth = 0.614 V
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")

p = DeviceParams()
x, y, X, Y = make_mesh(p, Nx=44, Ny=36)   # coarse mesh: I-V shape is mesh-insensitive (DD proxy)
Nd, Na, Nnet = build_doping(p, X, Y)
ps = Poisson2D(x, y, Nnet, p, Phi_gate=PHI_GATE)
dd = DDSolver(ps)


def sweep(fixed, varlist, axis):
    """Sweep one terminal with bias continuation; return list of |Id| [A/um].
    axis='Vds' -> vary Vd at fixed Vg=fixed;  axis='Vgs' -> vary Vg at fixed Vd=fixed."""
    out, warm = [], None
    for v in varlist:
        kw = dict(Vg=fixed, Vd=v) if axis == "Vds" else dict(Vg=v, Vd=fixed)
        if warm is None:
            dd.solve(max_gummel=40, tol=1e-4, **kw)
        else:
            dd.solve(max_gummel=40, tol=1e-4,
                     phi_init=warm[0], n_init=warm[1], p_init=warm[2], **kw)
        out.append(abs(dd.drain_current()))
        warm = (dd.phi.copy(), dd.n.copy(), dd.p.copy())
    return out


# ----------------------------------------------------------------- Fig 8(b): Id-Vds
Vds_list = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Vgs_out = [2.0, 2.5, 3.0]
print("=== Id-Vds (Fig 8b), recalibrated device ===", flush=True)
Iout = {}
for vg in Vgs_out:
    Iout[vg] = sweep(vg, Vds_list, "Vds")
    print("Vgs=%.1f: " % vg + " ".join("%.3e" % i for i in Iout[vg]), flush=True)

# ----------------------------------------------------------------- Fig 8(a): subthreshold
# start at 0.2 V: the deep-off (Vgs=0) leakage sits at the DD numerical floor and is not
# informative for the subthreshold swing.
Vgs_sub = [0.2, 0.4, 0.6, 0.8, 1.0]
Vds_sub = [0.25, 0.5]
print("\n=== Id-Vgs subthreshold (Fig 8a) ===", flush=True)
Isub = {}
for vd in Vds_sub:
    Isub[vd] = sweep(vd, Vgs_sub, "Vgs")
    print("Vds=%.2f: " % vd + " ".join("%.3e" % i for i in Isub[vd]), flush=True)

np.savez(os.path.join(DATA, "iv_recal.npz"),
         Vds=Vds_list, Vgs_out=Vgs_out, Iout=np.array([Iout[v] for v in Vgs_out]),
         Vgs_sub=Vgs_sub, Vds_sub=Vds_sub, Isub=np.array([Isub[v] for v in Vds_sub]))

# subthreshold swing (mV/dec) in the exponential region, Vds=0.5
Idg = np.array(Isub[0.5]); vg = np.array(Vgs_sub)
lo = (Idg > 0) & (vg <= 0.6)
SS = None
if lo.sum() >= 2:
    slope = np.polyfit(vg[lo], np.log10(Idg[lo] + 1e-30), 1)[0]
    SS = 1000.0 / slope
    print("\nsubthreshold swing ~ %.0f mV/dec (Vds=0.5)" % SS, flush=True)
print("Id at Vgs=Vds=3: %.3e A/um  (SHE coupled gave 0.24 mA/um)" % Iout[3.0][-1], flush=True)

# ----------------------------------------------------------------- Liang-style Fig. 8
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

a = ax[0]
for vd in Vds_sub:
    a.semilogy(Vgs_sub, np.maximum(Isub[vd], 1e-16), "o-", label="V$_{ds}$=%.2f V" % vd)
a.set_xlabel("V$_{gs}$ [V]"); a.set_ylabel("I$_d$ [A/$\\mu$m]")
a.set_title("(a) subthreshold transfer  (cf. Liang Fig. 8a)")
a.grid(alpha=0.3, which="both"); a.legend(fontsize=9)
if SS is not None:
    a.text(0.05, 0.05, "SS $\\approx$ %.0f mV/dec" % SS, transform=a.transAxes,
           fontsize=9, va="bottom")

a = ax[1]
for vg in Vgs_out:
    a.plot(Vds_list, np.array(Iout[vg]) * 1e3, "o-", label="V$_{gs}$=%.1f V" % vg)
a.set_xlabel("V$_{ds}$ [V]"); a.set_ylabel("I$_d$ [mA/$\\mu$m]")
a.set_title("(b) output characteristic  (cf. Liang Fig. 8b)")
a.grid(alpha=0.3); a.legend(fontsize=9); a.set_ylim(bottom=0)

fig.suptitle("Recalibrated device I-V (drift-diffusion proxy, V$_{th}$=0.614 V) vs Liang Fig. 8 "
             "structure", fontsize=12)
fig.tight_layout()
out = os.path.join(FIGS, "she2d_iv_fig8.png")
fig.savefig(out, dpi=135); print("wrote", out, flush=True)
print("saved data/iv_recal.npz", flush=True)
