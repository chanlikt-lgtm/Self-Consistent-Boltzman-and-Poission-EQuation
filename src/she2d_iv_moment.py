"""
she2d_iv_moment.py -- Terminal I-V from the SHE/BTE VELOCITY MOMENT (finite-volume face-flux
current), the way Liang et al. obtain Fig. 8 -- NOT from a mobility-based drift-diffusion model.

For each bias the DD solve supplies ONLY the electrostatic potential phi; the terminal current is
then read from the SHE face fluxes (conservation_diagnostics -> drain_current_A_per_um), i.e. the
velocity moment of the distribution. This closes the Fig. 8 methodological gap: the reported
current genuinely comes from the SHE, and is compared against the DD proxy at the same bias.

Scope/honesty: one-way SHE on the DD potential (not the full coupled loop) for tractability, on the
recalibrated device (Phi_gate=-0.74, Vth=0.614 V), 40x34. Bias continuation warm-starts the DD phi
across each family. Audit stop rule: the Vgs=3.0 output family + subthreshold are computed first
(enough to show the current is from the SHE face flux), then Vgs=2.5, 2.0; results saved
incrementally to iv_moment_partial.npz.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver
from she2d import SHE2D

PHI_GATE = -0.74
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")

p = DeviceParams()
x, y, X, Y = make_mesh(p, Nx=40, Ny=34)
Nd, Na, Nnet = build_doping(p, X, Y)
ps = Poisson2D(x, y, Nnet, p, Phi_gate=PHI_GATE)
dd = DDSolver(ps)


def she_current(Vg, Vd, warm):
    """DD -> phi (warm-started); SHE -> face-flux terminal current. Returns (Id_she, Id_dd, mism, warm)."""
    kw = dict(Vg=Vg, Vd=Vd, max_gummel=60, tol=1e-5)
    if warm is not None:
        kw.update(phi_init=warm[0], n_init=warm[1], p_init=warm[2])
    dd.solve(**kw)
    Id_dd = abs(dd.drain_current())
    she = SHE2D(x, y, dd.phi, ps.contact_type, dd.n_eq, dHi_eV=0.0125,
                include_impact_ionization=True, absorbing_top=True)
    she.assemble(); she.solve(tol=1e-8)
    c = she.conservation_diagnostics()
    Is, Idn = c["source_current_A_per_um"], c["drain_current_A_per_um"]
    mism = abs(Is + Idn) / (0.5 * (abs(Is) + abs(Idn)) + 1e-30)
    warm = (dd.phi.copy(), dd.n.copy(), dd.p.copy())
    return abs(Idn), Id_dd, mism, warm


results = {}


def out_family(Vg, Vds_list):
    warm, rows = None, []
    for vd in Vds_list:
        if vd == 0.0:
            rows.append((0.0, 0.0, 0.0, 0.0)); continue
        t0 = time.time()
        Ishe, Idd, mism, warm = she_current(Vg, vd, warm)
        rows.append((vd, Ishe, Idd, mism))
        print("  Vgs=%.1f Vds=%.2f: Id_SHE=%.3e  Id_DD=%.3e  (mism=%.1e, %.0fs)"
              % (Vg, vd, Ishe, Idd, mism, time.time() - t0), flush=True)
    return rows


def save():
    np.savez(os.path.join(DATA, "iv_moment_partial.npz"),
             **{k: np.array(v) for k, v in results.items()})


Vds_out = [0.0, 0.5, 1.0, 2.0, 3.0]

print("=== SHE-moment output family Vgs=3.0 (primary) ===", flush=True)
results["out_3.0"] = out_family(3.0, Vds_out); save()

print("=== SHE-moment subthreshold Vds=0.5 ===", flush=True)
warm, rows = None, []
for vg in [0.6, 0.8, 1.0, 1.2]:
    t0 = time.time()
    Ishe, Idd, mism, warm = she_current(vg, 0.5, warm)
    rows.append((vg, Ishe, Idd, mism))
    print("  Vds=0.5 Vgs=%.2f: Id_SHE=%.3e  Id_DD=%.3e  (%.0fs)"
          % (vg, Ishe, Idd, time.time() - t0), flush=True)
results["sub_0.5"] = rows; save()

for vg in [2.5, 2.0]:
    print("=== SHE-moment output family Vgs=%.1f ===" % vg, flush=True)
    results["out_%.1f" % vg] = out_family(vg, Vds_out); save()

np.savez(os.path.join(DATA, "iv_moment.npz"), **{k: np.array(v) for k, v in results.items()})

# ---- plot: SHE face-flux current (solid) vs DD proxy (dashed) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
a = ax[1]
for vg in (2.0, 2.5, 3.0):
    key = "out_%.1f" % vg
    if key not in results:
        continue
    r = np.array(results[key])
    a.plot(r[:, 0], r[:, 1] * 1e3, "o-", label="SHE $V_{gs}$=%.1f" % vg)
    a.plot(r[:, 0], r[:, 2] * 1e3, "x--", alpha=0.5)
a.set_xlabel("V$_{ds}$ [V]"); a.set_ylabel("I$_d$ [mA/$\\mu$m]")
a.set_title("(b) output: SHE face-flux (solid) vs DD (dashed)")
a.grid(alpha=0.3); a.legend(fontsize=8); a.set_ylim(bottom=0)

a = ax[0]
if "sub_0.5" in results:
    r = np.array(results["sub_0.5"])
    a.semilogy(r[:, 0], np.maximum(r[:, 1], 1e-16), "o-", label="SHE face flux")
    a.semilogy(r[:, 0], np.maximum(r[:, 2], 1e-16), "x--", label="DD proxy")
a.set_xlabel("V$_{gs}$ [V]"); a.set_ylabel("I$_d$ [A/$\\mu$m]")
a.set_title("(a) subthreshold ($V_{ds}$=0.5), SHE vs DD")
a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

fig.suptitle("Terminal I-V from the SHE velocity moment (Liang's method) vs the DD proxy "
             "-- recalibrated device, 40$\\times$34", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "she2d_iv_moment.png"), dpi=135)
print("wrote figures/she2d_iv_moment.png ; saved data/iv_moment.npz", flush=True)
