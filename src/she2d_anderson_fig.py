"""
she2d_anderson_fig.py -- Rev 5 evidence figure. Anderson acceleration of the coupled loop vs the
damped-Picard baseline, plus the spatial (grid) sequence of the converged fixed-point G_ii.

Two findings, made visually explicit:
  1) NONLINEAR-LOOP convergence: Anderson (frozen-H) drives the true fixed-point residual
     max|G(phi)-phi| toward a floor and the observables PLATEAU, where the damped Picard left
     G_ii in monotone free-fall (no plateau).
  2) SPATIAL convergence: the converged G_ii is still strongly grid-dependent
     (40x34 -> 60x50 -> 80x66), i.e. the mesh non-convergence is a SEPARATE, still-open issue.

Parses the run logs in data/ (robust to iteration count); writes figures/she2d_anderson_converge.png.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
LINE = re.compile(r"outer\s+(\d+)\s+dphi=([\d.eE+-]+) V\s+(?:NH=\d+\s+)?nmax=([\d.eE+-]+)\s+"
                  r"pmax=[\d.eE+-]+\s+Temax=([\d.]+) K\s+Gmax=([\d.eE+-]+)\s+Id=([\d.eE+-]+)")


def parse(fname):
    p = os.path.join(ROOT, "data", fname)
    if not os.path.exists(p):
        return None
    it, dphi, n, Te, G, Id = [], [], [], [], [], []
    for m in LINE.finditer(open(p).read()):
        it.append(int(m.group(1))); dphi.append(float(m.group(2))); n.append(float(m.group(3)))
        Te.append(float(m.group(4))); G.append(float(m.group(5))); Id.append(float(m.group(6)))
    if not it:
        return None
    return dict(it=np.array(it), dphi=np.array(dphi), n=np.array(n),
                Te=np.array(Te), G=np.array(G), Id=np.array(Id))


runs = {  # label -> (logfile, Ncells for spatial axis)
    "Anderson 40x34": ("anderson_40x34.log", 40 * 34),
    "Anderson 60x50": ("anderson_60x50.log", 60 * 50),
    "Anderson 80x66": ("anderson_80x66.log", 80 * 66),
}
picard = parse("coupled_conv.log")           # damped-Picard 60x50 baseline (superseded)
data = {k: parse(f) for k, (f, _) in runs.items()}

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
colors = {"Anderson 40x34": "#1f77b4", "Anderson 60x50": "#2ca02c", "Anderson 80x66": "#d62728"}

# (a) residual convergence: Anderson (per grid) vs Picard
a = ax[0, 0]
for k, d in data.items():
    if d is not None:
        a.semilogy(d["it"], d["dphi"] * 1e3, "o-", ms=3, color=colors[k], label=k)
if picard is not None:
    a.semilogy(picard["it"], picard["dphi"] * 1e3 / 0.35, "s--", ms=2, color="gray",
               label="Picard 60x50 (raw)")  # picard logged damped step (x0.35); show raw
a.axhline(2.0, color="k", ls=":", lw=0.8)
a.set_xlabel("outer iteration"); a.set_ylabel("max$|G(\\phi)-\\phi|$ [mV]")
a.set_title("(a) fixed-point residual: Anderson converges, Picard stalls"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

# (b) Gii trajectory: Anderson plateaus vs Picard runaway
a = ax[0, 1]
for k, d in data.items():
    if d is not None:
        a.plot(d["it"], d["G"] / 1e26, "o-", ms=3, color=colors[k], label=k)
if picard is not None:
    a.plot(picard["it"], picard["G"] / 1e26, "s--", ms=2, color="gray", label="Picard 60x50 (runaway)")
a.set_xlabel("outer iteration"); a.set_ylabel("$G_{ii,\\max}$ [$10^{26}$cm$^{-3}$s$^{-1}$]")
a.set_title("(b) $G_{ii}$: Anderson plateaus (per grid) vs Picard free-fall"); a.legend(fontsize=8); a.grid(alpha=0.3)

# (c) spatial sequence: converged Gii vs mesh
a = ax[1, 0]
Ns, Gs, labs = [], [], []
for k, (f, Nc) in runs.items():
    d = data[k]
    if d is not None:
        Ns.append(Nc); Gs.append(float(np.mean(d["G"][-4:])) / 1e26); labs.append(k.split()[-1])
if Ns:
    Ns, Gs = np.array(Ns), np.array(Gs)
    a.plot(Ns, Gs, "o-", ms=7, color="purple")
    for xN, yG, lb in zip(Ns, Gs, labs):
        a.annotate("%s\n%.2f" % (lb, yG), (xN, yG), textcoords="offset points", xytext=(6, 6), fontsize=8)
a.set_xlabel("spatial cells $N_x\\times N_y$"); a.set_ylabel("converged $G_{ii,\\max}$ [$10^{26}$]")
a.set_title("(c) SPATIAL sequence: $G_{ii}$ still grid-dependent (mesh not converged)"); a.grid(alpha=0.3)

# (d) Te / Id plateau (Anderson, finest available)
a = ax[1, 1]
d = data.get("Anderson 80x66") or data.get("Anderson 60x50") or data.get("Anderson 40x34")
if d is not None:
    a.plot(d["it"], d["Te"], "o-", ms=3, color="darkorange", label="$T_{e,\\max}$ [K]")
    a2 = a.twinx()
    a2.plot(d["it"], d["Id"] * 1e3, "s-", ms=3, color="navy", label="$I_d$ [mA/$\\mu$m]")
    a2.set_ylabel("$I_d$ [mA/$\\mu$m]", color="navy")
    a.set_ylabel("$T_{e,\\max}$ [K]", color="darkorange")
a.set_xlabel("outer iteration"); a.set_title("(d) $T_e$, $I_d$ plateau (Anderson, finest grid)"); a.grid(alpha=0.3)

fig.suptitle("Anderson acceleration of the coupled loop (frozen H-grid): loop convergence achieved; "
             "$G_{ii}$ spatial convergence still open", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "she2d_anderson_converge.png"), dpi=135)
print("wrote figures/she2d_anderson_converge.png")
print("\n=== converged fixed-point G_ii (mean of last 4 iters) vs grid ===")
for k, (f, Nc) in runs.items():
    d = data[k]
    if d is not None:
        g = np.mean(d["G"][-4:]); sp = (d["G"][-4:].max() - d["G"][-4:].min()) / g
        print("  %-16s Gii=%.3e  (last-4 spread %.1f%%, final res %.2e V, Te=%.0f, Id=%.3e, n=%.3e)"
              % (k, g, 100 * sp, d["dphi"][-1], d["Te"][-1], d["Id"][-1], d["n"][-1]))
