"""
recalibrate_vth.py -- Recalibrate the reconstructed device toward a paper-consistent
Vth ~ 0.6 V by adjusting the gate flat-band offset Phi_gate (rigid electrostatic shift,
dVth/dPhi_gate ~ +1 V/V), then extract Vth by max-gm linear extrapolation.

Usage: python recalibrate_vth.py [Phi_gate]   (default -0.74 V; N_A channel implant fixed 4e17)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver

Phi_gate = float(sys.argv[1]) if len(sys.argv) > 1 else -0.74

p = DeviceParams()
x, y, X, Y = make_mesh(p, Nx=56, Ny=48)
Nd, Na, Nnet = build_doping(p, X, Y)
ps = Poisson2D(x, y, Nnet, p, Phi_gate=Phi_gate)
dd = DDSolver(ps)

Vds = 0.1
Vgs = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
Id = np.zeros_like(Vgs)
for k, vg in enumerate(Vgs):
    dd.solve(Vg=vg, Vd=Vds, max_gummel=30, tol=1e-4)
    Id[k] = abs(dd.drain_current())
    print("  Vgs=%.2f  Id=%.3e A/um" % (vg, Id[k]))

gm = np.gradient(Id, Vgs)
k = int(np.argmax(gm))
Vth = Vgs[k] - Id[k] / gm[k] - Vds / 2.0
print("Phi_gate=%.3f V  ->  extracted Vth=%.3f V  (target ~0.6; was 1.64 at Phi_gate=0.30)"
      % (Phi_gate, Vth))
print("overdrive at Vg=3: Vg-Vth = %.2f V (was 1.36)" % (3.0 - Vth))
