"""Plot the coupled SHE<->Poisson<->hole convergence from data/coupled.log."""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = os.path.join(os.path.dirname(__file__), "..", "data", "coupled.log")
pat = re.compile(r"outer\s+(\d+)\s+dphi=([\d.eE+-]+)\s+V\s+nmax=([\d.eE+-]+)\s+pmax=([\d.eE+-]+)"
                 r"\s+Temax=([\d.eE+-]+)\s+K\s+Gmax=([\d.eE+-]+)\s+Id=([\d.eE+-]+)")
it, dphi, nmax, pmax, Te, Gii, Id = ([] for _ in range(7))
for line in open(log):
    m = pat.search(line)
    if m:
        it.append(int(m.group(1)))
        for arr, g in zip((dphi, nmax, pmax, Te, Gii, Id), range(2, 8)):
            arr.append(float(m.group(g)))
it = np.array(it)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
a = ax[0]
a.semilogy(it, dphi, "o-", color="#4a56c8")
a.axhline(2e-3, color="k", ls="--", lw=1, label="tol 2 mV")
a.set_xlabel("outer iteration"); a.set_ylabel("max |$\\Delta\\phi$| [V]")
a.set_title("potential-update convergence"); a.grid(alpha=0.3, which="both"); a.legend()

a = ax[1]
a.plot(it, np.array(Gii) / 1e27, "o-", color="#b9791b")
a.set_xlabel("outer iteration"); a.set_ylabel("G$_{ii,max}$  [10$^{27}$ cm$^{-3}$s$^{-1}$]")
a.set_title("impact-ionization suppression by feedback"); a.grid(alpha=0.3)
a.annotate("one-way\n(outer 0)", (0, Gii[0]/1e27), textcoords="offset points", xytext=(20, -4), fontsize=8)

a = ax[2]
a.plot(it, np.array(Id) * 1e3, "o-", color="#1f8f52", label="drain current")
a.set_xlabel("outer iteration"); a.set_ylabel("I$_d$  [mA/$\\mu$m]")
a.set_title("drain current vs self-consistency"); a.grid(alpha=0.3)
a2 = a.twinx(); a2.plot(it, Te, "s--", color="#c0392b", ms=4, label="Te")
a2.set_ylabel("T$_{e,max}$ [K]", color="#c0392b"); a2.tick_params(axis="y", labelcolor="#c0392b")
a2.set_ylim(3000, 3400)

fig.suptitle("Self-consistent SHE$\\leftrightarrow$Poisson$\\leftrightarrow$hole loop (40$\\times$34): "
             "feedback lowers G$_{ii}$ %.0f%% and I$_d$ %.0f%%, T$_e$ stable"
             % (100*(1-Gii[-1]/Gii[0]), 100*(1-Id[-1]/Id[0])), fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figures", "she2d_coupled_convergence.png")
fig.savefig(out, dpi=135); print("wrote", out)
print("outer0 -> outer%d:  Gii %.3e -> %.3e (%.0f%%),  Id %.3e -> %.3e (%.0f%%),  dphi %.1f -> %.1f mV"
      % (it[-1], Gii[0], Gii[-1], 100*(1-Gii[-1]/Gii[0]), Id[0], Id[-1], 100*(1-Id[-1]/Id[0]),
         dphi[0]*1e3, dphi[-1]*1e3))
