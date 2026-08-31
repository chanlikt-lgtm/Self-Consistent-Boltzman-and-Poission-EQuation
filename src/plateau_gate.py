"""
plateau_gate.py -- corrected Rev-6 observable-plateau gate over the last four REAL *cumulative*
iterations of a node's trajectory, independent of chunk boundaries.

WHY: the previous harness evaluated the last-4 spread on a single driver chunk's own history. When the
driver's solver-stopping criterion `tol_phi` fired early, a chunk could contribute only 1 real
iteration, and the "last-4" spread over one value was a trivial 0.0 -> a DEGENERATE FALSE PASS
(observed for 71x58, 2026-08-22). The fix maintains a rolling buffer of genuine per-iteration samples
that spans chunk boundaries and evaluates the gate over the last four REAL cumulative samples.

INVARIANTS (verified by test_plateau_gate.py):
  1. The rolling window spans chunk boundaries and uses the last four REAL iterations.
  2. plateau_evaluable is False until four genuine samples exist; the gate cannot pass before then.
  3. `tol_phi` is NEVER referenced here -- it remains solely the driver's solver-stopping condition and
     never short-circuits observable acceptance.
  4. The PRODUCTION spread formula and thresholds are UNCHANGED:
       spread(x) = (max(x) - min(x)) / mean(x)          # exactly coupled_drain_richardson.py
       gate: s(Gii) <= SPREAD_GATE AND s(Gtot) <= SPREAD_GATE AND R_phi_last <= RPHI_FLOOR_V
"""
import numpy as np

# UNCHANGED production thresholds (mirror coupled_drain_richardson.py / run_richardson_trio.py)
SPREAD_GATE = 3e-4
RPHI_FLOOR_V = 1.5e-3
WINDOW = 4


def spread(xs):
    """Production relative-spread formula (max-min)/mean over the given RAW samples. Unchanged."""
    a = np.asarray(xs, dtype=float)
    return float((a.max() - a.min()) / a.mean())


def chunk_samples_from_last4(last4_Gii, last4_Gtot, last4_Rphi, cum_iters):
    """Reconstruct a chunk's genuine trailing samples (min(n,4) of them) with ABSOLUTE iteration
    indices. The driver saves last4_* = the last min(n,4) real iterations of the chunk; their absolute
    indices are the final min(n,4) cumulative iterations, i.e. [cum_iters-k+1 .. cum_iters]."""
    g = np.atleast_1d(np.asarray(last4_Gii, float))
    t = np.atleast_1d(np.asarray(last4_Gtot, float))
    r = np.atleast_1d(np.asarray(last4_Rphi, float))
    k = len(g)
    if not (len(t) == k and len(r) == k):
        raise ValueError("misaligned last4 arrays: %d/%d/%d" % (k, len(t), len(r)))
    base = cum_iters - k + 1
    return [dict(abs_iter=int(base + i), Gii=float(g[i]), Gtot=float(t[i]), Rphi=float(r[i]))
            for i in range(k)]


def roll_window(prev_buffer, chunk_samples):
    """Append a chunk's new REAL samples to the rolling buffer and keep the last WINDOW, ordered
    oldest->newest. Spans chunk boundaries: for a full chunk (>=4 new) the window becomes that chunk's
    last 4; for a short chunk (1-3 new) it is {last (4-n) of prev} ++ {all n new}."""
    return (list(prev_buffer) + list(chunk_samples))[-WINDOW:]


def check_monotonic(buffer):
    """Fail-closed integrity check for a resumed buffer: absolute iteration indices must be STRICTLY
    increasing (hence unique). A duplicate or out-of-order index introduced by resume bookkeeping could
    silently manufacture an artificial plateau, so callers must abort rather than certify. Returns
    (ok: bool, detail: str)."""
    its = [int(b["abs_iter"]) for b in buffer]
    strictly_increasing = all(its[i] < its[i + 1] for i in range(len(its) - 1))
    unique = len(set(its)) == len(its)
    if strictly_increasing and unique:
        return True, "iters=%s" % its
    return False, "non-monotone/duplicate abs_iter: %s" % its


def evaluate(buffer):
    """Evaluate the cumulative gate over the rolling buffer of real samples.
    Returns a dict with plateau_evaluable, n_samples, spread_Gii, spread_Gtot, Rphi_last, passed,
    and the iteration indices actually used -- so the decision is independently checkable."""
    n = len(buffer)
    if n < WINDOW:
        return dict(plateau_evaluable=False, n_samples=n, spread_Gii=None, spread_Gtot=None,
                    Rphi_last=(buffer[-1]["Rphi"] if buffer else None),
                    iterations=[b["abs_iter"] for b in buffer], passed=False)
    win = buffer[-WINDOW:]
    s_gii = spread([b["Gii"] for b in win])
    s_gtot = spread([b["Gtot"] for b in win])
    rphi_last = float(win[-1]["Rphi"])
    passed = (s_gii <= SPREAD_GATE) and (s_gtot <= SPREAD_GATE) and (rphi_last <= RPHI_FLOOR_V)
    return dict(plateau_evaluable=True, n_samples=WINDOW, spread_Gii=s_gii, spread_Gtot=s_gtot,
                Rphi_last=rphi_last, iterations=[b["abs_iter"] for b in win], passed=passed)
