#!/usr/bin/env python
"""dump_reference.py -- Export the Python reproduction results (npz) to plain-text
CSV / raw-binary files that Mathematica can Import, so the Mathematica port can be
verified side-by-side against the reference Python implementation.

Inputs (from the parent project data/ dir):
  data/she2d_result.npz                 : 40x34 one-way SHE on DD phi (Vg=Vd=3)
  data/dd_40x34_Vg3.0_Vd3.0.npz         : DD phi / contact map / equilibrium n

Outputs (mathematica/ref/):
  x.csv, y.csv, H_ref.csv               : 1-D grids (um) and H grid (J)
  phi_ref.csv, n_ref.csv, Te_ref.csv, Gii_ref.csv,
  vx_ref.csv, vy_ref.csv                : 2-D fields (rows = y/depth, cols = x/lateral)
  ctype_ref.csv, neq_ref.csv            : contact map (0 none,1 src,2 drn,3 body), n_eq cm^-3
  F3d_ref.bin                           : raw float32 (Ny,Nx,NH) full distribution
"""
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
REF = os.path.join(os.path.dirname(__file__), "ref")
os.makedirs(REF, exist_ok=True)

she = np.load(os.path.join(ROOT, "data", "she2d_result.npz"))
dd = np.load(os.path.join(ROOT, "data", "dd_40x34_Vg3.0_Vd3.0.npz"))

def csv(name, a):
    np.savetxt(os.path.join(REF, name), a, fmt="%.12g")
    print("wrote", name, a.shape)

csv("x.csv", she["x"])
csv("y.csv", she["y"])
csv("H_ref.csv", she["H"])
csv("phi_ref.csv", she["phi"])
csv("n_ref.csv", she["n"])
csv("Te_ref.csv", she["Te"])
csv("Gii_ref.csv", she["Gii"])
csv("vx_ref.csv", she["vx"])
csv("vy_ref.csv", she["vy"])
csv("ctype_ref.csv", dd["ctype"])
csv("neq_ref.csv", dd["neq"])

F = she["F3d"].astype(np.float32)
F.tofile(os.path.join(REF, "F3d_ref.bin"))
with open(os.path.join(REF, "F3d_dims.txt"), "w") as fh:
    fh.write("%d %d %d\n" % F.shape)
print("wrote F3d_ref.bin", F.shape, F.dtype)

print("--- reference key numbers ---")
print("n_max  = %.6e cm^-3" % she["n"].max())
print("Te_max = %.4f K" % she["Te"].max())
print("Gii_max= %.6e cm^-3/s" % she["Gii"].max())
print("|vx|max= %.4e cm/s  |vy|max= %.4e cm/s" % (np.abs(she["vx"]).max(), np.abs(she["vy"]).max()))
print("phi range = %.6f .. %.6f V" % (she["phi"].min(), she["phi"].max()))
print("H: NH=%d  dH=%.6e J  (%g eV)" % (len(she["H"]), she["H"][1]-she["H"][0], (she["H"][1]-she["H"][0])/1.602176634e-19))
print("ctype counts:", np.bincount(dd["ctype"].ravel()))
print("neq range = %.3e .. %.3e cm^-3" % (dd["neq"].min(), dd["neq"].max()))
