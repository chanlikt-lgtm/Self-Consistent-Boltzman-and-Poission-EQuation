"""
anderson_converge_fig.py -- convergence comparison of baseline vs robustified Anderson (40x34).

Parses the two coupled-run logs and shows why the stagnation-restart + rcond regularization
converges G_ii where the naive Anderson only stalled. Panels:
  (a) fixed-point residual max|G(phi)-phi|   (b) G_ii,max  -- the money plot (drift vs plateau)
  (c) T_e,max                                (d) drain current I_d
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOGS = [
    ("baseline AA 40x34 (rcond 1e-12, no restart)", "anderson_40x34.log",       "#c0392b"),
    ("robustified AA 40x34 (restart + rcond)",       "anderson_40x34_robust.log", "#1e7a1e"),
    ("robustified AA 60x50 (restart + rcond)",       "anderson_60x50_robust.log", "#1f4e9e"),
]

# NH= column is present only in the robustified log; make it optional so both parse.
PAT = re.compile(
    r"outer\s+(?P<it>\d+)\s+dphi=(?P<dphi>[\d.eE+-]+)\s+V\s+"
    r"(?:NH=(?P<NH>\d+)\s+)?"
    r"nmax=(?P<n>[\d.eE+-]+)\s+pmax=(?P<p>[\d.eE+-]+)\s+"
    r"Temax=(?P<Te>[\d.]+)\s+K\s+Gmax=(?P<G>[\d.eE+-]+)\s+Id=(?P<Id>[\d.eE+-]+)"
)


def parse(fn):
    it, dphi, Te, G, Id = [], [], [], [], []
    with open(fn) as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            it.append(int(m.group("it")))
            dphi.append(float(m.group("dphi")))
            Te.append(float(m.group("Te")))
            G.append(float(m.group("G")))
            Id.append(float(m.group("Id")))
    return {k: np.array(v) for k, v in
            dict(it=it, dphi=dphi, Te=Te, G=G, Id=Id).items()}


runs = [(lbl, parse(os.path.join(ROOT, fn)), c) for lbl, fn, c in LOGS]

fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
for lbl, d, c in runs:
    ax[0, 0].semilogy(d["it"], d["dphi"] * 1e3, "-o", ms=3, color=c, label=lbl)
    ax[0, 1].plot(d["it"], d["G"] / 1e27, "-o", ms=3, color=c, label=lbl)
    ax[1, 0].plot(d["it"], d["Te"], "-o", ms=3, color=c, label=lbl)
    ax[1, 1].plot(d["it"], d["Id"] * 1e3, "-o", ms=3, color=c, label=lbl)

ax[0, 0].axhline(0.1, ls="--", color="gray", lw=1, label="1e-4 V tolerance")
ax[0, 0].set_title("(a) fixed-point residual"); ax[0, 0].set_ylabel(r"$\max|G(\phi)-\phi|$  [mV]")
ax[0, 0].legend(fontsize=8, loc="upper right")

ax[0, 1].set_title("(b) impact ionization -- both plateau, but at grid-dependent levels")
ax[0, 1].set_ylabel(r"$G_{ii,\max}$  [$10^{27}$ cm$^{-3}$s$^{-1}$]"); ax[0, 1].legend(fontsize=7.5)

ax[1, 0].set_title("(c) electron temperature"); ax[1, 0].set_ylabel(r"$T_{e,\max}$  [K]")
ax[1, 1].set_title("(d) drain current"); ax[1, 1].set_ylabel(r"$I_d$  [mA/$\mu$m]")

for a in ax.ravel():
    a.set_xlabel("outer iteration"); a.grid(alpha=0.3)

fig.suptitle(r"Restart-Anderson converges the coupled SHE$\leftrightarrow$Poisson$\leftrightarrow$hole "
             r"loop $G_{ii}$ at both grids -- but the converged plateau is strongly mesh-dependent",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(ROOT, "figures", "anderson_converge.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("saved", os.path.relpath(out, ROOT))

for lbl, d, _ in runs:
    l4 = d["G"][-4:]
    print("%-42s final_res=%.3f mV  Gii=%.3e  last4_spread=%.2e" %
          (lbl, d["dphi"][-1] * 1e3, d["G"][-1], (l4.max() - l4.min()) / l4.mean()))
