"""
vth_extract.py -- Explicit threshold-voltage extraction (audit re-review F3): replace the
inferred "Vth ~ 0.5 V" with a measured value, and document the degeneracy-breaking rule.

Degeneracy rule: the two electrostatic knobs are the channel-implant peak N_A and the gate
flat-band offset Phi_gate. We FIX N_A = 4e17 cm^-3 (a physical shallow-channel-implant value)
and treat Phi_gate as the SINGLE free offset. Vth is then extracted, not assumed.

Method: linear-region (Vds=0.1 V) I_d-V_gs sweep; Vth by max-transconductance linear
extrapolation, Vth = Vgs* - Id*/gm_max - Vds/2.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver

if __name__ == "__main__":
    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=56, Ny=48)     # coarse mesh for a fast sweep
    Nd, Na, Nnet = build_doping(p, X, Y)
    ps = Poisson2D(x, y, Nnet, p, Phi_gate=0.30)
    dd = DDSolver(ps)

    Vds = 0.1
    Vgs = np.array([0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    Id = np.zeros_like(Vgs)
    for k, vg in enumerate(Vgs):
        dd.solve(Vg=vg, Vd=Vds, max_gummel=30, tol=1e-4)
        Id[k] = abs(dd.drain_current())     # A/um
        print("  Vgs=%.2f  Id=%.3e A/um" % (vg, Id[k]))

    gm = np.gradient(Id, Vgs)
    kmax = int(np.argmax(gm))
    Vth = Vgs[kmax] - Id[kmax] / gm[kmax] - Vds / 2.0
    print("\nmax-gm at Vgs=%.2f (gm=%.3e S/um)" % (Vgs[kmax], gm[kmax]))
    print("Extracted linear Vth = %.3f V  (Vds=%.2f, N_A=4e17 fixed, Phi_gate=0.30 the single knob)" % (Vth, Vds))

    # tiny sensitivity: dVth/dPhi_gate is ~ -1 by construction (rigid band shift); report it
    print("Sensitivity: Vth shifts ~ -1 V per +1 V of Phi_gate (rigid electrostatic offset);")
    print("             N_A held fixed, so the calibration is unique given Phi_gate.")
