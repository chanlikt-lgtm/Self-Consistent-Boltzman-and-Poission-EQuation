"""
run_richardson_trio.py -- reproducible harness for the Rev. 6 controlled spatial-convergence trio.

Regenerates the three self-similar nodes 71x58 -> 100x70 -> 141x85 with the EXACT archival procedure:
each node is run, then CONTINUED (warm-started from its own saved npz) under byte-identical settings
until it passes the observable-plateau gate -- R_phi at its floor AND last-4 relative spreads
<= SPREAD_GATE in BOTH G_ii,max and G_tot. Nodes are warm-start chained (100x70 from 71x58, 141x85
from 100x70) exactly as archived. This is the "continue unchanged until the plateau gate passes"
procedure referenced in the report's reproducibility appendix; it removes the fixed-iteration-budget
bias documented there (60-iteration values were transients).

Runs strictly SEQUENTIALLY (each fine node needs ~5-6 GB). Total ~23 h on a single workstation.
Usage: python run_richardson_trio.py
Then:  python richardson_fit.py data/she2d_richardson_71x58.npz \
                                data/she2d_richardson_100x70.npz \
                                data/she2d_richardson_141x85.npz
"""
import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SPREAD_GATE = 3e-4          # observable-plateau gate: O(1e-4), <= 3e-4, both G_ii and G_tot
CHUNK = 60                  # outer iterations per driver invocation
MAX_ROUNDS = 4              # continuations before giving up on a node (60,120,180,240 iters)

# (Nx, Ny, warm-start source npz or None for the first node). Warm-start chaining as archived.
TRIO = [
    (71, 58, None),
    (100, 70, "she2d_richardson_71x58.npz"),
    (141, 85, "she2d_richardson_100x70.npz"),
]

def node_npz(Nx, Ny):
    return os.path.join(DATA, "she2d_richardson_%dx%d.npz" % (Nx, Ny))

def gate_passed(Nx, Ny):
    """True iff the saved node meets the observable-plateau gate in BOTH G_ii and G_tot."""
    f = node_npz(Nx, Ny)
    if not os.path.exists(f):
        return False, None, None
    z = np.load(f)
    sp_g = float(z["spread"]); sp_gt = float(z["spread_Gtot"])
    return (sp_g <= SPREAD_GATE and sp_gt <= SPREAD_GATE), sp_g, sp_gt

def run_driver(Nx, Ny, warm_npz):
    cmd = [sys.executable, "-u", os.path.join(HERE, "coupled_drain_richardson.py"),
           str(Nx), str(Ny), str(CHUNK)]
    if warm_npz:
        cmd.append(warm_npz)
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit("driver failed for %dx%d (exit %d)" % (Nx, Ny, r.returncode))

def main():
    for Nx, Ny, warm_src in TRIO:
        warm = os.path.join(DATA, warm_src) if warm_src else None
        # First invocation: cold (or chain warm-start from the previous node's converged npz).
        run_driver(Nx, Ny, warm)
        # Continue (warm-started from THIS node's own saved npz) until the plateau gate passes.
        for _ in range(MAX_ROUNDS - 1):
            ok, sp_g, sp_gt = gate_passed(Nx, Ny)
            print("  %dx%d gate: spread_Gii=%.2e spread_Gtot=%.2e (<= %.0e ? %s)"
                  % (Nx, Ny, sp_g, sp_gt, SPREAD_GATE, ok), flush=True)
            if ok:
                break
            run_driver(Nx, Ny, node_npz(Nx, Ny))       # continue warm-started from own state
        ok, sp_g, sp_gt = gate_passed(Nx, Ny)
        if not ok:
            print("  WARNING: %dx%d did not reach the plateau gate in %d rounds "
                  "(spread_Gii=%.2e, spread_Gtot=%.2e); increase MAX_ROUNDS."
                  % (Nx, Ny, MAX_ROUNDS, sp_g, sp_gt), flush=True)
        else:
            z = np.load(node_npz(Nx, Ny))
            print("  %dx%d PLATEAU: Gii_max=%.4e  G_tot=%.4e  R_phi=%.2e  (spreads %.2e/%.2e)"
                  % (Nx, Ny, float(z["Gii_max"]), float(z["G_tot"]), float(z["R_phi"]), sp_g, sp_gt),
                  flush=True)
    print("\nAll three nodes archived under data/she2d_richardson_*.npz. Now run richardson_fit.py.",
          flush=True)

if __name__ == "__main__":
    main()
