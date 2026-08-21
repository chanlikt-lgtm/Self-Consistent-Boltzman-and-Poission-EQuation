"""
run_richardson_trio.py -- reproducible harness for the Rev. 6 controlled spatial-convergence trio,
with a structured run log + metrics CSV for unattended (overnight) runs.

Regenerates the three self-similar nodes 71x58 -> 100x70 -> 141x85 with the EXACT archival procedure:
each node is run, then CONTINUED (warm-started from its own saved npz) under byte-identical settings
until it passes the observable-plateau gate -- R_phi at its floor AND last-4 relative spreads
<= SPREAD_GATE in BOTH G_ii,max and G_tot. Nodes are warm-start chained (100x70 from 71x58, 141x85
from 100x70) exactly as archived. Runs strictly SEQUENTIALLY (each fine node needs ~5-6 GB).

Collects, per driver chunk: wall time, cumulative iterations, G_ii,max / G_tot / T_e,max, the
fixed-point residual R_phi, last-4 relative spreads (the plateau/convergence indicator), the
chunk-to-chunk relative drift of G_ii,max (a coarse convergence rate), and gate status. Emits
warnings for low free RAM, a node that does not reach the gate, and unusually slow iterations.
Two output files (under data/):
  richardson_trio_run.log      -- timestamped human-readable log (mirror of console)
  richardson_trio_metrics.csv  -- one row per chunk, machine-readable
Usage: python run_richardson_trio.py
"""
import os, sys, csv, time, subprocess, datetime
import numpy as np
try:
    import psutil
except Exception:
    psutil = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)
LOG_PATH = os.path.join(DATA, "richardson_trio_run.log")
CSV_PATH = os.path.join(DATA, "richardson_trio_metrics.csv")

SPREAD_GATE = 3e-4          # observable-plateau gate: O(1e-4), <= 3e-4, in BOTH G_ii and G_tot
RPHI_FLOOR_V = 1.5e-3       # R_phi "at its floor": final residual settled into the she_tol noise band
CHUNK = 60                  # outer iterations per driver invocation
MAX_ROUNDS = 4              # HARD cap on continuation chunks/node (up to 240 iters); exceed => FAIL
RAM_WARN_GB = 7.0           # warn if available RAM below this before a fine node
SLOW_ITER_S = {71: 160, 100: 260, 141: 460}   # per-iter wall-time above which we warn (by Nx)

# (Nx, Ny, warm-start source npz for the FIRST chunk, or None). Warm-start chaining as archived.
TRIO = [
    (71, 58, None),
    (100, 70, "she2d_richardson_71x58.npz"),
    (141, 85, "she2d_richardson_100x70.npz"),
]

_logf = None
def log(msg, level="INFO"):
    global _logf
    if _logf is None:                       # open lazily, so importing the module has no side-effect
        _logf = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    line = "%s [%s] %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, msg)
    print(line, flush=True)
    _logf.write(line + "\n")

def node_npz(Nx, Ny):
    return os.path.join(DATA, "she2d_richardson_%dx%d.npz" % (Nx, Ny))

def read_metrics(Nx, Ny):
    z = np.load(node_npz(Nx, Ny))
    return dict(Gii=float(z["Gii_max"]), Gtot=float(z["G_tot"]), Te=float(z["Te_max"]),
                Rphi=float(z["R_phi"]), sp=float(z["spread"]), spt=float(z["spread_Gtot"]),
                niter=int(z["n_iter"]),
                last4_R=np.asarray(z["last4_R_phi"], float) if "last4_R_phi" in z.files
                        else np.array([float(z["R_phi"])]))

def rphi_floored(last4_R):
    """R_phi 'at its floor': the final fixed-point residual has settled into the she_tol-set noise
    band (<= RPHI_FLOOR_V), i.e. no longer descending by orders of magnitude. The observable-plateau
    spreads (<= SPREAD_GATE in both G_ii and G_tot) are the primary convergence indicator; this is the
    corroborating residual condition (report, sec:spatial). A gentle monotone drift WITHIN the band
    (as on the tightest-converged 141x85 node) still counts as floored."""
    a = np.asarray(last4_R, float)
    return float(a[-1]) <= RPHI_FLOOR_V

def full_gate(m):
    """The FULL observable-plateau gate: spread_Gii<=gate AND spread_Gtot<=gate AND R_phi floored."""
    return (m["sp"] <= SPREAD_GATE) and (m["spt"] <= SPREAD_GATE) and rphi_floored(m["last4_R"])

def free_gb():
    if psutil is None:
        return None
    return psutil.virtual_memory().available / 1e9

def run_driver(Nx, Ny, warm_npz):
    cmd = [sys.executable, "-u", os.path.join(HERE, "coupled_drain_richardson.py"),
           str(Nx), str(Ny), str(CHUNK)]
    if warm_npz:
        cmd.append(warm_npz)
    log("RUN %dx%d  chunk=%d  warm=%s" % (Nx, Ny, CHUNK, os.path.basename(warm_npz) if warm_npz else "cold"))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        log("driver FAILED for %dx%d (exit %d) after %.0fs" % (Nx, Ny, r.returncode, dt), "ERROR")
        sys.exit(2)
    return dt

def main():
    log("=" * 70)
    log("Rev.6 controlled trio harness START  (gate<=%.0e both G_ii & G_tot, chunk=%d, max_rounds=%d)"
        % (SPREAD_GATE, CHUNK, MAX_ROUNDS))
    rg = free_gb()
    log("available RAM: %s" % ("%.1f GB" % rg if rg is not None else "unknown (psutil absent)"))
    t_start = time.time()

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["timestamp", "node", "chunk", "chunk_iters", "cum_iters", "wall_s", "cum_wall_s",
                    "s_per_iter", "Gii_max", "G_tot", "Te_max", "R_phi", "spread_Gii", "spread_Gtot",
                    "dGii_rel_vs_prev_chunk", "gate_pass", "warnings"])

        summary = []
        for Nx, Ny, warm_src in TRIO:
            rg = free_gb()
            if rg is not None and rg < RAM_WARN_GB:
                log("LOW RAM before %dx%d: %.1f GB free (< %.1f GB; fine node needs ~5-6 GB). "
                    "Close other apps to avoid an OOM kill." % (Nx, Ny, rg, RAM_WARN_GB), "WARN")

            warm = os.path.join(DATA, warm_src) if warm_src else None
            cum_iters = 0
            node_t0 = time.time()
            prev_Gii = None
            passed = False
            for chunk in range(MAX_ROUNDS):
                src = warm if chunk == 0 else node_npz(Nx, Ny)   # continue from own state after chunk 0
                dt = run_driver(Nx, Ny, src)
                m = read_metrics(Nx, Ny)
                cum_iters += m["niter"]
                spi = dt / max(m["niter"], 1)
                dGii = (abs(m["Gii"] - prev_Gii) / m["Gii"]) if prev_Gii else float("nan")
                prev_Gii = m["Gii"]
                passed = full_gate(m)

                warns = []
                if spi > SLOW_ITER_S.get(Nx, 1e9):
                    warns.append("slow: %.0fs/iter (>%ds)" % (spi, SLOW_ITER_S[Nx]))
                rg = free_gb()
                if rg is not None and rg < RAM_WARN_GB:
                    warns.append("low_RAM %.1fGB" % rg)
                wtxt = ";".join(warns)

                log("  %dx%d chunk %d: +%d iters (cum %d) in %.0fs (%.0fs/iter)  Gii=%.4e Gtot=%.4e "
                    "Te=%.0f  R_phi=%.2e  spread Gii/Gtot=%.2e/%.2e  dGii=%.2e  gate=%s%s"
                    % (Nx, Ny, chunk, m["niter"], cum_iters, dt, spi, m["Gii"], m["Gtot"], m["Te"],
                       m["Rphi"], m["sp"], m["spt"], dGii, "PASS" if passed else "cont",
                       ("  WARN:" + wtxt) if wtxt else ""),
                    "WARN" if wtxt else "INFO")
                w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                            "%dx%d" % (Nx, Ny), chunk, m["niter"], cum_iters, "%.0f" % dt,
                            "%.0f" % (time.time() - t_start), "%.1f" % spi, "%.6e" % m["Gii"],
                            "%.6e" % m["Gtot"], "%.0f" % m["Te"], "%.3e" % m["Rphi"],
                            "%.3e" % m["sp"], "%.3e" % m["spt"], "%.3e" % dGii, passed, wtxt])
                cf.flush()
                if passed:
                    break

            node_dt = time.time() - node_t0
            if not passed:
                # HARD FAIL: the node did not reach the full plateau gate within the chunk cap.
                # Fail the run rather than proceed on a non-plateaued (invalid) node.
                log("%dx%d FAILED to reach the full plateau gate in %d chunks (%d iters): "
                    "spread_Gii=%.2e spread_Gtot=%.2e R_phi_floored=%s. Aborting run. "
                    "(Increase MAX_ROUNDS and continue warm-started from its npz to extend.)"
                    % (Nx, Ny, MAX_ROUNDS, cum_iters, m["sp"], m["spt"],
                       rphi_floored(m["last4_R"])), "ERROR")
                sys.exit(3)
            log("%dx%d DONE: Gii=%.4e Gtot=%.4e R_phi=%.2e  (%d iters, %.1f h, gate=%s)"
                % (Nx, Ny, m["Gii"], m["Gtot"], m["Rphi"], cum_iters, node_dt / 3600.0,
                   "PASS" if passed else "FAIL"))
            summary.append((Nx, Ny, m["Gii"], m["Gtot"], m["sp"], m["spt"], cum_iters, node_dt, passed))

    # ---- final verdict summary (mirror of richardson_fit's monotonicity test) ----
    log("-" * 70)
    total_h = (time.time() - t_start) / 3600.0
    G = [s[2] for s in summary]
    if len(G) == 3:
        u = [(np.load(node_npz(s[0], s[1]))["last4_Gii_max"]) for s in summary]
        u = [float((a.max() - a.min()) / 2) for a in u]
        d12, d23 = G[0] - G[1], G[1] - G[2]
        monotone = (d12 > 0 and d23 > 0) or (d12 < 0 and d23 < 0)
        Srev = abs(d23) / (u[1] ** 2 + u[2] ** 2) ** 0.5 if not monotone else float("nan")
        log("TRIO Gii_max: %.4e -> %.4e -> %.4e  (D12=%+.3e, D23=%+.3e)" % (G[0], G[1], G[2], d12, d23))
        if monotone:
            log("sequence is MONOTONE -- differs from the archival non-monotone result; inspect nodes.", "WARN")
        else:
            log("sequence is NON-MONOTONE (as archived); reversal S_rev=%.0f %s"
                % (Srev, "(>>1: significant, Richardson suppressed)" if Srev > 3 else
                   "(~O(1): within plateau band -- grid-stable)"))
    if not all(s[8] for s in summary):
        log("NOTE: one or more nodes did not pass the plateau gate -- verdict is provisional.", "WARN")
    log("ALL DONE in %.1f h. Run: python src/richardson_fit.py %s"
        % (total_h, " ".join("data/she2d_richardson_%dx%d.npz" % (s[0], s[1]) for s in summary)))
    log("logs: %s  metrics: %s" % (LOG_PATH, CSV_PATH))
    log("=" * 70)

if __name__ == "__main__":
    main()
