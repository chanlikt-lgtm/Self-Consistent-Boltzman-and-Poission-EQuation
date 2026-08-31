"""
test_plateau_gate.py -- unit tests for the corrected cumulative-window plateau gate.

Covers the failure modes actually observed plus the four invariants required for review:
  1. rolling window spans chunk boundaries and uses the last four REAL iterations;
  2. plateau_evaluable=False until four genuine samples exist (gate cannot pass before then);
  3. tol_phi never referenced (not a substitute for observable acceptance);
  4. production spread formula (max-min)/mean and thresholds (3e-4, 1.5e-3) unchanged.

Run: python test_plateau_gate.py   (exit 0 = all pass)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plateau_gate as pg

_fails = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + (("  -- " + detail) if detail else ""))
    if not cond:
        _fails.append(name)

def s(gii, gtot, rphi, it):        # convenience sample
    return dict(abs_iter=it, Gii=gii, Gtot=gtot, Rphi=rphi)

def old_chunk_spread(vals):        # the DEFECTIVE previous logic: spread over the chunk's own last-4
    return pg.spread(vals)         # for a 1-value chunk this is (v-v)/v = 0.0 -> trivial pass

print("== Invariant 4: production formula + thresholds unchanged ==")
check("spread formula = (max-min)/mean", abs(pg.spread([1.0, 1.2, 0.9, 1.1]) - (1.2-0.9)/((1.0+1.2+0.9+1.1)/4)) < 1e-15)
check("SPREAD_GATE == 3e-4", pg.SPREAD_GATE == 3e-4)
check("RPHI_FLOOR_V == 1.5e-3", pg.RPHI_FLOOR_V == 1.5e-3)
check("WINDOW == 4", pg.WINDOW == 4)

print("== Invariant 3: tol_phi never used in the plateau_gate LOGIC (docstring mention is OK) ==")
import ast
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plateau_gate.py")).read()
_tree = ast.parse(src)
_names = {n.id for n in ast.walk(_tree) if isinstance(n, ast.Name)}
_attrs = {n.attr for n in ast.walk(_tree) if isinstance(n, ast.Attribute)}
check("tol_phi not a code identifier (name/attr) in plateau_gate.py",
      "tol_phi" not in _names and "tol_phi" not in _attrs,
      "docstring may mention it; the LOGIC must not reference it")

print("== Test A: normal 60-iteration full-chunk PASS ==")
# 4 tightly-clustered genuine samples (spread ~1.4e-4), R_phi in band
c = pg.chunk_samples_from_last4([2.1360e25, 2.1362e25, 2.1361e25, 2.1363e25],
                                [6.024e13, 6.025e13, 6.0245e13, 6.0248e13],
                                [7e-4, 6.5e-4, 6.2e-4, 6.96e-4], cum_iters=60)
buf = pg.roll_window([], c); ev = pg.evaluate(buf)
check("A evaluable (4 real samples)", ev["plateau_evaluable"])
check("A both spreads <= gate", ev["spread_Gii"] <= pg.SPREAD_GATE and ev["spread_Gtot"] <= pg.SPREAD_GATE,
      "sGii=%.2e sGtot=%.2e" % (ev["spread_Gii"], ev["spread_Gtot"]))
check("A PASSED", ev["passed"])
check("A window uses last 4 of this chunk", ev["iterations"] == [57, 58, 59, 60])

print("== Test B: 1-iteration terminal chunk -> valid cumulative reconstruction ==")
prev = [s(2.1360e25, 6.024e13, 7e-4, 55), s(2.1362e25, 6.025e13, 6.5e-4, 56), s(2.1361e25, 6.0245e13, 6.2e-4, 57)]
chunk = pg.chunk_samples_from_last4([2.1363e25], [6.0248e13], [6.96e-4], cum_iters=58)  # n=1
buf = pg.roll_window(prev, chunk); ev = pg.evaluate(buf)
check("B rolling window spans chunk boundary (iters 55-58)", ev["iterations"] == [55, 56, 57, 58])
check("B evaluable (4 cumulative real samples)", ev["plateau_evaluable"])
check("B cumulative spread is GENUINE (not the degenerate 0)", ev["spread_Gii"] > 0.0,
      "cumulative sGii=%.2e vs old-degenerate=%.2e" % (ev["spread_Gii"], old_chunk_spread([2.1363e25])))
check("B degenerate old logic WOULD have passed (spread 0)", old_chunk_spread([2.1363e25]) == 0.0)
check("B new logic decides on the real 4-sample window", ev["passed"] == (ev["spread_Gii"] <= pg.SPREAD_GATE
      and ev["spread_Gtot"] <= pg.SPREAD_GATE and ev["Rphi_last"] <= pg.RPHI_FLOOR_V))

print("== Test C: degenerate FALSE-PASS prevented (still-moving value) ==")
# prior 3 samples show the value still moving; cumulative 4-window spread EXCEEDS the gate
prev = [s(2.05e25, 5.8e13, 8e-4, 55), s(2.08e25, 5.9e13, 7.5e-4, 56), s(2.11e25, 6.0e13, 7e-4, 57)]
chunk = pg.chunk_samples_from_last4([2.13e25], [6.05e13], [6.9e-4], cum_iters=58)  # n=1, tol_phi early-stop
buf = pg.roll_window(prev, chunk); ev = pg.evaluate(buf)
check("C cumulative spread exceeds gate", ev["spread_Gii"] > pg.SPREAD_GATE, "sGii=%.2e" % ev["spread_Gii"])
check("C new logic correctly FAILS", ev["passed"] is False)
check("C old degenerate logic would have FALSELY PASSED", old_chunk_spread([2.13e25]) == 0.0)

print("== Test D: 2- and 3-iteration terminal chunks reconstruct correctly ==")
prev2 = [s(2.1361e25, 6.024e13, 7e-4, 56), s(2.1362e25, 6.025e13, 6.6e-4, 57)]
d2 = pg.roll_window(prev2, pg.chunk_samples_from_last4([2.1362e25, 2.1363e25], [6.0245e13, 6.0248e13],
                                                       [6.4e-4, 6.96e-4], cum_iters=59))
ev2 = pg.evaluate(d2)
check("D2 spans boundary (iters 56-59)", ev2["iterations"] == [56, 57, 58, 59] and ev2["plateau_evaluable"])
prev1 = [s(2.1362e25, 6.025e13, 6.6e-4, 57)]
d3 = pg.roll_window(prev1, pg.chunk_samples_from_last4([2.1361e25, 2.1362e25, 2.1363e25],
                                                       [6.0242e13, 6.0245e13, 6.0248e13],
                                                       [6.5e-4, 6.4e-4, 6.96e-4], cum_iters=60))
ev3 = pg.evaluate(d3)
check("D3 spans boundary (iters 57-60)", ev3["iterations"] == [57, 58, 59, 60] and ev3["plateau_evaluable"])

print("== Test E: Invariant 2 -- plateau_evaluable=False until 4 genuine samples ==")
# three IDENTICAL samples: spread would be 0, but must NOT pass because <4 samples exist
buf3 = [s(2.1362e25, 6.025e13, 6.5e-4, 58), s(2.1362e25, 6.025e13, 6.5e-4, 59), s(2.1362e25, 6.025e13, 6.5e-4, 60)]
ev = pg.evaluate(buf3)
check("E 3 identical samples: not evaluable", ev["plateau_evaluable"] is False and ev["n_samples"] == 3)
check("E 3 identical samples: does NOT pass despite spread would be 0", ev["passed"] is False)

print("== Test F: R_phi band uses the most-recent sample ==")
# tight spreads but final R_phi out of band -> must fail on the residual condition
buf = pg.roll_window([], pg.chunk_samples_from_last4([2.1362e25]*4, [6.025e13]*4,
                                                     [1.0e-3, 1.1e-3, 1.2e-3, 2.0e-3], cum_iters=60))
ev = pg.evaluate(buf)
check("F spreads pass but R_phi_last out of band -> FAIL", ev["spread_Gii"] == 0.0 and ev["passed"] is False,
      "R_phi_last=%.2e" % ev["Rphi_last"])

print("== Test G: resume-buffer integrity (check_monotonic) ==")
good = [s(2.1e25, 6e13, 6e-4, 55), s(2.1e25, 6e13, 6e-4, 56), s(2.1e25, 6e13, 6e-4, 57), s(2.1e25, 6e13, 6e-4, 58)]
ok, _ = pg.check_monotonic(good); check("G strictly-increasing buffer OK", ok)
dup = good[:3] + [s(2.1e25, 6e13, 6e-4, 57)]  # duplicate abs_iter 57
ok, det = pg.check_monotonic(dup); check("G duplicate abs_iter rejected", ok is False, det)
oo = [good[0], good[2], good[1], good[3]]      # out of order
ok, det = pg.check_monotonic(oo); check("G out-of-order abs_iter rejected", ok is False, det)

print("\n" + ("ALL TESTS PASSED" if not _fails else "FAILURES: " + ", ".join(_fails)))
sys.exit(1 if _fails else 0)
