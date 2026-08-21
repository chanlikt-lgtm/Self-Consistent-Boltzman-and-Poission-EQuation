"""
gen_run_report.py -- auto-generate a self-contained RESULTS report from THIS run's fresh trio data.

Reads the three regenerated data/she2d_richardson_{71x58,100x70,141x85}.npz and (if present) the
data/richardson_trio_metrics.csv, then writes and compiles report/report_run_latest.pdf: run metadata
(timestamp, git commit), the comparability gates (signatures + H-hash + refinement ratios), the fresh
trio table (dx, G_ii/G_tot with plateau spreads, R_phi), the S_rev monotonicity verdict, a
timing/convergence summary, and a line certifying whether this run reproduced the archived Rev.6
result. This complements (does not replace) the canonical methodological report report_v6_full.tex.
Usage: python gen_run_report.py
"""
import os, sys, csv, math, subprocess, datetime, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); REP = os.path.join(ROOT, "report")
NODES = [(71, 58), (100, 70), (141, 85)]
ARCH_G = [2.990e25, 2.135e25, 2.413e25]; ARCH_SREV = 874.0        # archived Rev.6 reference
SPREAD_GATE = 3e-4

def npz(nx, ny): return os.path.join(DATA, "she2d_richardson_%dx%d.npz" % (nx, ny))

def load():
    zs = []
    for nx, ny in NODES:
        f = npz(nx, ny)
        if not os.path.exists(f):
            sys.exit("missing %s -- run run_richardson_trio.py first." % f)
        zs.append(np.load(f))
    return zs

def node_time_s():
    """Per-node cumulative wall seconds from the metrics CSV (last row per node), if available."""
    p = os.path.join(DATA, "richardson_trio_metrics.csv")
    if not os.path.exists(p): return {}
    last = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            last[row["node"]] = row
    out = {}
    prev = 0.0
    for nx, ny in NODES:
        k = "%dx%d" % (nx, ny)
        if k in last:
            cw = float(last[k]["cum_wall_s"]); out[k] = (cw - prev, int(last[k]["cum_iters"]),
                                                         float(last[k]["s_per_iter"])); prev = cw
    return out

def esc(s): return str(s).replace("_", r"\_").replace("%", r"\%")

def main():
    z = load()
    dx = [float(x["drain_dx"]) for x in z]
    G = [float(x["Gii_max"]) for x in z]; Gt = [float(x["G_tot"]) for x in z]
    sp = [float(x["spread"]) for x in z]; spt = [float(x["spread_Gtot"]) for x in z]
    Rp = [float(x["R_phi"]) for x in z]; Te = [float(x["Te_max"]) for x in z]
    u = [float((np.asarray(x["last4_Gii_max"]).max() - np.asarray(x["last4_Gii_max"]).min()) / 2) for x in z]
    hashes = {str(x["H_hash"]) for x in z if "H_hash" in x.files}
    laws = {(round(float(x["ratio"]), 4), round(float(x["W"]), 4), round(float(x["X_C"]), 4),
             round(float(x["dH_eV"]), 6), int(x["NH"]), float(x["she_tol"])) for x in z}
    rx = [dx[i] / dx[i + 1] for i in range(2)]
    d12, d23 = G[0] - G[1], G[1] - G[2]
    monotone = (d12 > 0 and d23 > 0) or (d12 < 0 and d23 < 0)
    Srev = abs(d23) / math.hypot(u[1], u[2]) if not monotone else float("nan")
    gate_ok = all(sp[i] <= SPREAD_GATE and spt[i] <= SPREAD_GATE for i in range(3))
    reproduced = (not monotone) and all(abs(G[i] - ARCH_G[i]) / ARCH_G[i] < 0.02 for i in range(3))
    times = node_time_s()
    commit = str(z[0]["git_commit"]) if "git_commit" in z[0].files else "?"
    dirty = str(z[0]["git_dirty"]) if "git_dirty" in z[0].files else "?"

    verdict = ("NON-MONOTONE, $S_{\\mathrm{rev}}=%.0f\\gg1$ --- Richardson suppressed, spatial "
               "convergence not demonstrated" % Srev) if not monotone else \
              ("MONOTONE --- differs from the archived result; inspect nodes")
    trow = lambda i: ("$G_%d$ (%d$\\times$%d) & %.4f & $%.3f\\times10^{25}$ & $%.1e$ & $%.3f\\times10^{13}$"
                      " & $%.1e$ & $%.1e$ \\\\" % (i + 1, int(z[i]["Nx"]), int(z[i]["Ny"]), dx[i],
                      G[i] / 1e25, sp[i], Gt[i] / 1e13, spt[i], Rp[i]))
    ttime = lambda k: (("%.1f h / %d it / %.0f s\\,it$^{-1}$" %
                        (times[k][0] / 3600, times[k][1], times[k][2])) if k in times else "n/a")

    tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}\usepackage{amsmath,booktabs,siunitx,xcolor}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\title{\textbf{Controlled Spatial-Convergence Trio --- Run Results}\\[3pt]
\large\textit{Auto-generated from run data; companion to report\_v6\_full}}
\date{""" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + r"""}\author{}
\begin{document}\maketitle\vspace{-2.2em}
\begin{center}\fcolorbox{black}{%s}{\parbox{0.92\linewidth}{\centering\textbf{%s}}}\end{center}
\section*{Run metadata}
Generated \texttt{%s}; code commit \texttt{%s} (%s). Nodes:
\texttt{data/she2d\_richardson\_\{71x58,100x70,141x85\}.npz}.
\section*{Comparability gates}
\begin{tabular}{ll}\toprule
mesh-law/solver signatures identical & \textbf{%s} \\
frozen $H$-array SHA1 identical & \textbf{%s} (%s) \\
drain-$dx$ refinement ratios & $%.5f,\ %.5f$ (target $\sqrt2=1.41421$) \\
observable-plateau gate ($\le%.0e$, both $G_{ii}$ \& $G_{\mathrm{tot}}$) & \textbf{%s} \\
\bottomrule\end{tabular}
\section*{Frozen trio (this run)}
\begin{tabular}{lcccccc}\toprule
 & drain $dx$ & $G_{ii,\max}$ & spread & $G_{\mathrm{tot}}$ & spread & $R_\phi$ \\
node & (\si{\micro m}) & (\si{cm^{-3}s^{-1}}) & ($G_{ii}$) & (\si{cm^{-1}s^{-1}}) & ($G_{\mathrm{tot}}$) & (V) \\ \midrule
%s
%s
%s
\bottomrule\end{tabular}\\[4pt]
{\footnotesize spread $=(\max_4-\min_4)/\bar G_4$; plateau band $u=(\max_4-\min_4)/2$.}
\section*{Verdict}
$G_{ii,\max}$: $%.3f \to %.3f \to %.3f \times10^{25}$; $\Delta_{12}=%+.3e$, $\Delta_{23}=%+.3e$.
\textbf{%s.}\\[2pt]
Reproduces archived Rev.\,6 ($2.990/2.135/2.413\times10^{25}$, $S_{\mathrm{rev}}\approx%.0f$):
\textbf{%s}.
\section*{Timing / convergence}
\begin{tabular}{lccc}\toprule
node & wall / iters / s per iter & final $R_\phi$ (V) & $T_{e,\max}$ (K) \\ \midrule
71$\times$58 & %s & $%.1e$ & %.0f \\
100$\times$70 & %s & $%.1e$ & %.0f \\
141$\times$85 & %s & $%.1e$ & %.0f \\
\bottomrule\end{tabular}\\[4pt]
See \texttt{data/richardson\_trio\_run.log} and \texttt{richardson\_trio\_metrics.csv} for the full
per-iteration history and any warnings. Scientific interpretation: report\_v6\_full, \S\,Controlled
spatial-convergence.
\end{document}
""" % ("green!14" if reproduced else ("red!12" if monotone else "yellow!18"),
       verdict, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), esc(commit), esc(dirty),
       "YES" if len(laws) == 1 else "NO", "YES" if len(hashes) == 1 else "NO",
       esc(list(hashes)[0]) if hashes else "?", rx[0], rx[1], SPREAD_GATE, "PASS" if gate_ok else "FAIL",
       trow(0), trow(1), trow(2),
       G[0] / 1e25, G[1] / 1e25, G[2] / 1e25, d12, d23, verdict, ARCH_SREV,
       "YES" if reproduced else "NO -- differs, investigate",
       ttime("71x58"), Rp[0], Te[0], ttime("100x70"), Rp[1], Te[1], ttime("141x85"), Rp[2], Te[2])

    out_tex = os.path.join(REP, "report_run_latest.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print("wrote", out_tex)
    pdflatex = shutil.which("pdflatex") or \
        r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if os.path.exists(pdflatex) or shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run([pdflatex, "--enable-installer", "--interaction=nonstopmode",
                            "report_run_latest.tex"], cwd=REP,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("compiled report/report_run_latest.pdf")
    else:
        print("pdflatex not found -- wrote .tex only.")

if __name__ == "__main__":
    main()
