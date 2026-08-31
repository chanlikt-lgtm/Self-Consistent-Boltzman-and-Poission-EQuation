"""
run_richardson_trio.py -- reproducible harness for the Rev. 6 controlled spatial-convergence trio,
with a structured run log + metrics CSV AND idempotent, provenance-checked resume.

Nodes 71x58 -> 100x70 -> 141x85 are run sequentially with the EXACT archival procedure: each node is
run, then CONTINUED (warm-started) under byte-identical settings, in CHUNK-iteration steps, until it
passes the observable-plateau gate -- last-4 relative spreads <= SPREAD_GATE in BOTH G_ii,max and
G_tot AND R_phi settled into the she_tol band (<= RPHI_FLOOR_V). The SCIENTIFIC GATES ARE NEVER
RELAXED; only the safety cap on continuation chunks (MAX_ROUNDS) was raised 4 -> 8 after a cold-start
first node needed more than 4 chunks (a harness-limit failure, not a scientific one).

Idempotent resume (per-node, provenance-checked, fail-closed):
  * warm-start priority: same-node checkpoint -> sanctioned previous-node -> cold start.
  * an existing node NPZ is validated against the expected run definition (mesh, grading law, dH,
    eps_max, she_tol, common phi-span). PROVENANCE MISMATCH => FAIL CLOSED (abort), never silently
    overwrite. A valid checkpoint that already passed the gate is SKIPPED; a valid incomplete
    checkpoint is RESUMED from (not cold), preserving the cumulative iteration count.
  * per-node progress (cumulative iters, chunks done, passed, resume history) is tracked in a sidecar
    data/she2d_richardson_NxxNy.progress.json so relaunch is idempotent.

Outputs under data/: richardson_trio_run.log, richardson_trio_metrics.csv, and the per-node sidecars.
Usage: python run_richardson_trio.py
"""
import os, sys, csv, json, time, subprocess, datetime
import numpy as np
import plateau_gate as pg    # corrected cumulative-window observable-plateau gate (shared thresholds)
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
CSV_HEADER = ["timestamp", "node", "cum_chunk", "cum_iters", "wall_s", "s_per_iter", "Gii_max",
              "G_tot", "Te_max", "R_phi", "spread_Gii", "spread_Gtot", "dGii_rel", "gate_pass",
              "resume_source", "warnings"]

# ---- SCIENTIFIC GATES (UNCHANGED -- sourced from plateau_gate so there is a SINGLE definition) ----
SPREAD_GATE = pg.SPREAD_GATE      # 3e-4: last-4 relative spread gate, in BOTH G_ii and G_tot
RPHI_FLOOR_V = pg.RPHI_FLOOR_V    # 1.5e-3: R_phi settled into the she_tol noise band
assert SPREAD_GATE == 3e-4 and RPHI_FLOOR_V == 1.5e-3, "plateau_gate thresholds drifted"
CHUNK = 60                  # outer iterations per driver invocation
MAX_ROUNDS = 16             # HARD safety cap on continuation chunks/node (4 -> 8 -> 16). NOT a scientific
                            # gate; raised only to give the 71x58 audit-verification recheck continuation
                            # room from chunks_done=7. Gate thresholds are unchanged.

# ---- Expected run-definition signature (MUST mirror coupled_drain_richardson.py frozen constants).
#      Used ONLY to validate/resume checkpoints; the driver remains the single source for RUNNING. ----
EXPECTED = dict(ratio=6.0, W=0.09, X_C=0.60, dH_eV=0.0125, eps_max_eV=3.02, she_tol=1e-8)
EXPECTED_SPAN = (-0.60, 3.65)

RAM_WARN_GB = 7.0
SLOW_ITER_S = {71: 160, 100: 260, 141: 460}
TRIO = [(71, 58, None), (100, 70, "she2d_richardson_71x58.npz"),
        (141, 85, "she2d_richardson_100x70.npz")]

_logf = None
def log(msg, level="INFO"):
    global _logf
    if _logf is None:
        _logf = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    line = "%s [%s] %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, msg)
    print(line, flush=True); _logf.write(line + "\n")

def node_npz(Nx, Ny): return os.path.join(DATA, "she2d_richardson_%dx%d.npz" % (Nx, Ny))
def prog_path(Nx, Ny): return os.path.join(DATA, "she2d_richardson_%dx%d.progress.json" % (Nx, Ny))

def load_progress(Nx, Ny):
    p = prog_path(Nx, Ny)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None

def save_progress(Nx, Ny, d):
    json.dump(d, open(prog_path(Nx, Ny), "w", encoding="utf-8"), indent=2)

def read_metrics(Nx, Ny):
    z = np.load(node_npz(Nx, Ny))
    def _l4(key, scalar):
        return np.asarray(z[key], float) if key in z.files else np.array([float(scalar)])
    return dict(Gii=float(z["Gii_max"]), Gtot=float(z["G_tot"]), Te=float(z["Te_max"]),
                Rphi=float(z["R_phi"]), sp=float(z["spread"]), spt=float(z["spread_Gtot"]),
                niter=int(z["n_iter"]), Hhash=str(z["H_hash"]) if "H_hash" in z.files else "?",
                last4_Gii=_l4("last4_Gii_max", z["Gii_max"]),
                last4_Gtot=_l4("last4_G_tot", z["G_tot"]),
                last4_R=_l4("last4_R_phi", z["R_phi"]))

def rphi_floored(last4_R):
    """R_phi 'at its floor': final residual settled into the she_tol noise band (<= RPHI_FLOOR_V).
    Observable spreads are the primary indicator; a gentle drift within the band still counts."""
    return float(np.asarray(last4_R, float)[-1]) <= RPHI_FLOOR_V

def chunk_samples(m, cum_iters):
    """The REAL trailing samples this chunk contributed, with absolute iteration indices, for the
    cumulative rolling buffer. A chunk that stopped early (tol_phi) supplies fewer than 4 samples."""
    return pg.chunk_samples_from_last4(m["last4_Gii"], m["last4_Gtot"], m["last4_R"], cum_iters)

def node_plateau_from_npz(m):
    """Reconstruct the observable-plateau decision for an EXISTING checkpoint using only the terminal
    chunk's stored last-4 real samples (per-iteration history across chunks is not persisted). If that
    terminal chunk ran a full window its own last-4 ARE the cumulative last-4, so the decision is exact;
    if it ran <4 real iterations (the degenerate case) evaluate() returns plateau_evaluable=False and
    the node is NOT certified -- it will be resumed and rebuild a genuine cumulative window."""
    return pg.evaluate(pg.roll_window([], chunk_samples(m, m["niter"])))

def validate_checkpoint(Nx, Ny):
    """Return (state, detail): state in {'absent','valid','mismatch'}. Fail-closed provenance check
    of an existing node NPZ against the expected run definition."""
    f = node_npz(Nx, Ny)
    if not os.path.exists(f):
        return "absent", ""
    z = np.load(f)
    mism = []
    for k, exp in (("Nx", Nx), ("Ny", Ny)):
        if k not in z.files or int(z[k]) != exp:
            mism.append("%s=%s!=%s" % (k, z[k] if k in z.files else "MISSING", exp))
    for k, exp in EXPECTED.items():
        got = float(z[k]) if k in z.files else None
        if got is None or abs(got - exp) > 1e-9 * max(1.0, abs(exp)):
            mism.append("%s=%s!=%s" % (k, got, exp))
    sp = np.asarray(z["common_phi_span"], float) if "common_phi_span" in z.files else None
    if sp is None or not np.allclose(sp, EXPECTED_SPAN):
        mism.append("common_phi_span=%s!=%s" % (sp, EXPECTED_SPAN))
    if mism:
        return "mismatch", "; ".join(mism)
    return "valid", "H=%s git=%s" % (str(z["H_hash"]), str(z["git_commit"]) if "git_commit" in z.files else "?")

def free_gb():
    return psutil.virtual_memory().available / 1e9 if psutil else None

def run_driver(Nx, Ny, warm_npz):
    cmd = [sys.executable, "-u", os.path.join(HERE, "coupled_drain_richardson.py"),
           str(Nx), str(Ny), str(CHUNK)]
    if warm_npz:
        cmd.append(warm_npz)
    log("RUN %dx%d  chunk=%d  warm=%s" % (Nx, Ny, CHUNK, os.path.basename(warm_npz) if warm_npz else "cold"))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        log("driver FAILED for %dx%d (exit %d)" % (Nx, Ny, r.returncode), "ERROR")
        sys.exit(2)
    return time.time() - t0

def main():
    log("=" * 70)
    log("Rev.6 controlled trio harness START (gate<=%.0e both G_ii & G_tot; R_phi<=%.1e; chunk=%d; "
        "max_rounds=%d) -- idempotent provenance-checked resume" % (SPREAD_GATE, RPHI_FLOOR_V, CHUNK, MAX_ROUNDS))
    rg = free_gb(); log("available RAM: %s" % ("%.1f GB" % rg if rg is not None else "unknown"))
    t_start = time.time()
    restart_ts = datetime.datetime.now().isoformat(timespec="seconds")

    # Guard against schema drift: if an existing CSV's header does not match the CURRENT writer schema,
    # rotate it aside (preserved, timestamped) and start fresh -- appending mismatched rows under a stale
    # header silently misaligns every column and corrupts any by-name reader (the timing-table bug).
    if os.path.exists(CSV_PATH):
        try:
            existing_hdr = next(csv.reader(open(CSV_PATH, newline="", encoding="utf-8")), [])
        except Exception:
            existing_hdr = []
        if existing_hdr != CSV_HEADER:
            rot = CSV_PATH.replace(".csv", ".MALFORMED_%s.csv" % datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
            os.rename(CSV_PATH, rot)
            log("metrics CSV header mismatch -> rotated stale file to %s; writing a fresh aligned CSV."
                % os.path.basename(rot), "WARN")
    new_csv = not os.path.exists(CSV_PATH)
    cf = open(CSV_PATH, "a", newline="", encoding="utf-8"); w = csv.writer(cf)
    if new_csv:
        w.writerow(CSV_HEADER)

    summary = []
    for Nx, Ny, warm_src in TRIO:
        npz = node_npz(Nx, Ny)
        prov, detail = validate_checkpoint(Nx, Ny)
        prog = load_progress(Nx, Ny)

        if prov == "mismatch":
            log("FAIL CLOSED: %dx%d existing NPZ provenance mismatch [%s]. Refusing to resume or "
                "overwrite. Remove the file to force a fresh node, or fix the config." % (Nx, Ny, detail), "ERROR")
            sys.exit(4)

        # ---- decide start mode (warm-start priority: same-node -> prev-node -> cold) ----
        if prov == "valid":
            m0 = read_metrics(Nx, Ny)
            # certify only via an EVALUABLE cumulative window: prefer a persisted buffer (new harness);
            # otherwise reconstruct from the NPZ terminal chunk. A degenerate <4-sample terminal chunk
            # (e.g. the 71x58 tol_phi early-stop) is NOT evaluable -> not certified -> resumed.
            buf0 = list(prog["plateau_buffer"]) if (prog and prog.get("plateau_buffer")) else None
            buf0_origin = "persisted_sidecar" if buf0 is not None else "npz_terminal_reconstruction"
            if buf0 is not None:
                ok, det = pg.check_monotonic(buf0)
                if not ok:
                    log("FAIL CLOSED: %dx%d persisted plateau_buffer failed integrity check [%s]. Refusing "
                        "to certify a possibly-fabricated plateau." % (Nx, Ny, det), "ERROR")
                    sys.exit(4)
            ev0 = pg.evaluate(buf0) if buf0 is not None else node_plateau_from_npz(m0)
            already = bool(ev0["plateau_evaluable"] and ev0["passed"])
            if already:
                cum = int(prog["cum_iters"]) if prog and prog.get("cum_iters") else m0["niter"]
                log("%dx%d ALREADY PASSED (Gii=%.4e cumulative-window spread=%.2e/%.2e over iters %s, "
                    "R_phi=%.2e, cum %d iters, buffer_origin=%s, %s) -- SKIP" % (Nx, Ny, m0["Gii"],
                    ev0["spread_Gii"], ev0["spread_Gtot"], ev0["iterations"], m0["Rphi"], cum,
                    buf0_origin, detail))
                certified_buf = buf0 if buf0 is not None else pg.roll_window([], chunk_samples(m0, m0["niter"]))
                save_progress(Nx, Ny, dict(cum_iters=cum, chunks_done=(prog or {}).get("chunks_done"),
                              passed=True, plateau_evaluable=True, plateau_buffer=certified_buf,
                              buffer_origin=buf0_origin, H_hash=m0["Hhash"],
                              resume_source="skip(already-passed)", last_update=restart_ts))
                summary.append((Nx, Ny, m0["Gii"], m0["Gtot"], ev0["spread_Gii"], ev0["spread_Gtot"], cum, 0.0, True))
                continue
            resume_source = "same-node-checkpoint(%s,%s)" % (os.path.basename(npz), detail)
            warm = npz
            cum_iters = int(prog["cum_iters"]) if prog and prog.get("cum_iters") else None
            chunks_done = int(prog["chunks_done"]) if prog and prog.get("chunks_done") is not None else None
            if cum_iters is None or chunks_done is None:
                log("%dx%d valid checkpoint but no/partial sidecar -- resuming from NPZ; prior cumulative "
                    "iteration count UNKNOWN, treated conservatively as chunks_done=0 for the cap." % (Nx, Ny), "WARN")
                cum_iters = cum_iters or 0
                chunks_done = chunks_done or 0
        else:  # absent
            prev = os.path.join(DATA, warm_src) if warm_src and os.path.exists(os.path.join(DATA, warm_src)) else None
            if prev:
                resume_source = "prev-node-warmstart(%s)" % warm_src; warm = prev
            else:
                resume_source = "cold"; warm = None
            cum_iters, chunks_done = 0, 0

        if chunks_done >= MAX_ROUNDS:
            log("%dx%d already at chunk cap (%d/%d) but gate not passed -- run cannot extend it further; "
                "raise MAX_ROUNDS." % (Nx, Ny, chunks_done, MAX_ROUNDS), "ERROR")
            sys.exit(3)
        rg = free_gb()
        if rg is not None and rg < RAM_WARN_GB:
            log("LOW RAM before %dx%d: %.1f GB (< %.1f). Close other apps to avoid OOM." % (Nx, Ny, rg, RAM_WARN_GB), "WARN")
        log("%dx%d START mode=%s resume_source=%s cum_iters=%d chunks_done=%d/%d restart=%s"
            % (Nx, Ny, prov, resume_source, cum_iters, chunks_done, MAX_ROUNDS, restart_ts))

        # rolling buffer of the last <=4 REAL cumulative iterations, spanning chunk boundaries and
        # persisted across solver invocations. On same-node resume we reload it; a cold / prev-node
        # warm start begins with an empty buffer (the previous node's trajectory does not carry over).
        if prog and prog.get("plateau_buffer"):
            buffer = list(prog["plateau_buffer"]); buffer_origin = "persisted_sidecar"
            ok, det = pg.check_monotonic(buffer)
            if not ok:
                log("FAIL CLOSED: %dx%d resumed plateau_buffer failed integrity check [%s]. A duplicated or "
                    "out-of-order iteration could fabricate a plateau -- refusing to certify." % (Nx, Ny, det), "ERROR")
                sys.exit(4)
        else:
            buffer = []; buffer_origin = "empty_fresh"
        node_t0 = time.time(); prev_Gii = None; passed = False
        for chunk in range(chunks_done, MAX_ROUNDS):
            src = warm if chunk == chunks_done else npz     # first (resumed) chunk uses `warm`, then own npz
            dt = run_driver(Nx, Ny, src)
            m = read_metrics(Nx, Ny)
            cum_iters += m["niter"]; spi = dt / max(m["niter"], 1)
            dGii = (abs(m["Gii"] - prev_Gii) / m["Gii"]) if prev_Gii else float("nan"); prev_Gii = m["Gii"]
            # --- cumulative observable-plateau gate over the last 4 REAL iterations (NOT the chunk's
            #     own last-4). tol_phi is only the driver's stopping rule; it never certifies here. ---
            buffer = pg.roll_window(buffer, chunk_samples(m, cum_iters))
            ev = pg.evaluate(buffer)
            passed = bool(ev["plateau_evaluable"] and ev["passed"])
            cs_gii = ev["spread_Gii"] if ev["plateau_evaluable"] else m["sp"]
            cs_gtot = ev["spread_Gtot"] if ev["plateau_evaluable"] else m["spt"]
            warns = []
            if not ev["plateau_evaluable"]:
                warns.append("plateau_not_evaluable(n=%d)" % ev["n_samples"])
            if spi > SLOW_ITER_S.get(Nx, 1e9): warns.append("slow %.0fs/iter" % spi)
            rg = free_gb()
            if rg is not None and rg < RAM_WARN_GB: warns.append("low_RAM %.1fGB" % rg)
            wtxt = ";".join(warns)
            gate_txt = "PASS" if passed else ("cont" if ev["plateau_evaluable"] else "not-evaluable")
            log("  %dx%d chunk %d (iters %d-%d): %.0fs (%.0fs/iter)  Gii=%.4e Gtot=%.4e Te=%.0f "
                "R_phi=%.2e cum-window(%s) spread Gii/Gtot=%.2e/%.2e dGii=%.2e gate=%s%s"
                % (Nx, Ny, chunk, cum_iters - m["niter"] + 1, cum_iters, dt, spi, m["Gii"], m["Gtot"],
                   m["Te"], m["Rphi"], ev["iterations"], cs_gii, cs_gtot, dGii, gate_txt,
                   ("  WARN:" + wtxt) if wtxt else ""), "WARN" if wtxt else "INFO")
            w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), "%dx%d" % (Nx, Ny),
                        chunk, cum_iters, "%.0f" % dt, "%.1f" % spi, "%.6e" % m["Gii"], "%.6e" % m["Gtot"],
                        "%.0f" % m["Te"], "%.3e" % m["Rphi"], "%.3e" % cs_gii, "%.3e" % cs_gtot,
                        "%.3e" % dGii, passed, resume_source if chunk == chunks_done else "", wtxt]); cf.flush()
            save_progress(Nx, Ny, dict(cum_iters=cum_iters, chunks_done=chunk + 1, passed=passed,
                          plateau_evaluable=bool(ev["plateau_evaluable"]), plateau_buffer=buffer,
                          buffer_origin=buffer_origin, H_hash=m["Hhash"], resume_source=resume_source,
                          restart_ts=restart_ts, last_update=datetime.datetime.now().isoformat(timespec="seconds")))
            if passed:
                break

        node_dt = time.time() - node_t0
        if not passed:
            log("%dx%d FAILED to reach the plateau gate within the chunk cap (%d chunks, %d iters): "
                "plateau_evaluable=%s cum-window spread_Gii=%s spread_Gtot=%s R_phi_floored=%s. Aborting "
                "run (scientific gates unchanged; raise MAX_ROUNDS and relaunch to resume from this "
                "checkpoint)." % (Nx, Ny, MAX_ROUNDS, cum_iters, ev["plateau_evaluable"],
                ("%.2e" % ev["spread_Gii"]) if ev["plateau_evaluable"] else "n/a",
                ("%.2e" % ev["spread_Gtot"]) if ev["plateau_evaluable"] else "n/a",
                rphi_floored(m["last4_R"])), "ERROR")
            sys.exit(3)
        log("%dx%d DONE: Gii=%.4e Gtot=%.4e R_phi=%.2e (cum %d iters, %.1f h this session, gate=PASS)"
            % (Nx, Ny, m["Gii"], m["Gtot"], m["Rphi"], cum_iters, node_dt / 3600.0))
        summary.append((Nx, Ny, m["Gii"], m["Gtot"], ev["spread_Gii"], ev["spread_Gtot"], cum_iters, node_dt, True))

    cf.close()
    # ---- final verdict summary (mirror of richardson_fit's monotonicity test) ----
    log("-" * 70)
    G = [s[2] for s in summary]
    if len(G) == 3:
        u = [float((np.load(node_npz(s[0], s[1]))["last4_Gii_max"]).max()
                   - (np.load(node_npz(s[0], s[1]))["last4_Gii_max"]).min()) / 2 for s in summary]
        d12, d23 = G[0] - G[1], G[1] - G[2]
        monotone = (d12 > 0 and d23 > 0) or (d12 < 0 and d23 < 0)
        Srev = abs(d23) / (u[1] ** 2 + u[2] ** 2) ** 0.5 if not monotone else float("nan")
        log("TRIO Gii_max: %.4e -> %.4e -> %.4e (D12=%+.3e, D23=%+.3e)" % (G[0], G[1], G[2], d12, d23))
        if monotone:
            log("sequence is MONOTONE -- differs from archival non-monotone result; inspect nodes.", "WARN")
        else:
            log("sequence is NON-MONOTONE (as archived); reversal S_rev=%.0f %s" % (Srev,
                "(>>1: significant, Richardson suppressed)" if Srev > 3 else "(~O(1): grid-stable)"))
    log("ALL DONE in %.1f h (this session). Run: python src/richardson_fit.py %s"
        % ((time.time() - t_start) / 3600.0,
           " ".join("data/she2d_richardson_%dx%d.npz" % (s[0], s[1]) for s in summary)))
    log("=" * 70)

if __name__ == "__main__":
    main()
