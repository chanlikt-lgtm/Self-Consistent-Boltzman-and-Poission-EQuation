"""Physical constants and material parameters (CGS-eV-cm mixed units)."""
import numpy as np

# fundamental
q = 1.602176634e-19        # C
eps0 = 8.8541878128e-14    # F/cm
kB = 1.380649e-23          # J/K
hbar = 1.054571817e-34     # J*s
h = 6.62607015e-34         # J*s
m0 = 9.1093837015e-31      # kg
eV = 1.602176634e-19       # J

# operating temperature
T = 300.0                  # K
Vt = kB * T / q            # thermal voltage [V] ~0.02585

# silicon
eps_si = 11.7
eps_ox = 3.9
ni = 1.45e10               # intrinsic carrier density [cm^-3]  (Si, 300 K)
mstar = 0.32 * m0          # DOS/conductivity effective mass proxy for SHE spherical band

# ---- Table I (paper) ----
D_ac = 4.0                 # eV      acoustic deformation potential
D_n = 5.0e8                # eV/cm   optical/intervalley deformation potential
E_op = 0.05                # eV      optical phonon energy
v_s = 9.0e6                # cm/s
Gamma_sr = 1.96            # cm^2/V^2/s   surface-roughness lumped param
n_scr = 1.45e15            # cm^-3   screening density (ionized-impurity)
D_sa = 17.8                # eV      surface acoustic deformation potential
delta_q = 13.1             # quantum term param (surface roughness)
eps_max = 3.02             # eV      top of energy grid
rho_si = 2.329             # g/cm^3  silicon mass density
u_sound = 9.04e5           # cm/s    longitudinal sound velocity in Si

def asinh(x):
    return np.arcsinh(x)
