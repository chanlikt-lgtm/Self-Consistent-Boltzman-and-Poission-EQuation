"""
diagnostics.py -- Quantitative benchmarks (audit Finding 7): replace hand-wave
"agrees with analytic values" statements with expected-vs-computed numbers.

Provides:
  * built-in potential V_bi and one-sided depletion width (analytic) for the
    n+/substrate junction, compared to the equilibrium-Poisson-extracted width;
  * the documented drift-diffusion saturation-current estimate with all inputs;
  * (optional) a coarse threshold-voltage extraction by constant-current criterion.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from constants import q, eps0, eps_si, eps_ox, Vt, ni
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D, UM


def builtin_and_depletion(Na_p=1e16, Nd_n=1e20):
    """Analytic V_bi and one-sided (into p) depletion width for an abrupt n+/p junction."""
    Vbi = Vt * np.log(Nd_n * Na_p / ni ** 2)
    # one-sided step junction, depletion extends mainly into the lighter p side:
    Xp = np.sqrt(2 * eps_si * eps0 * Vbi / q * Nd_n / (Na_p * (Na_p + Nd_n)))  # cm
    W = np.sqrt(2 * eps_si * eps0 * Vbi / q * (Na_p + Nd_n) / (Na_p * Nd_n))    # cm (total)
    return Vbi, Xp / UM, W / UM   # widths in um


def extract_depletion_from_poisson(ps, x_probe_um=0.75):
    """Vertical cut under the drain n+; return metallurgical-junction depth and the
    p-side depletion edge (where hole density recovers to Na/2), both in um."""
    i = np.argmin(np.abs(ps.x / UM - x_probe_um))
    y_um = ps.y / UM
    Nnet_col = ps.Nnet[:, i]
    # metallurgical junction: sign change of net doping
    sgn = np.sign(Nnet_col)
    jy = np.where(np.diff(sgn) < 0)[0]
    y_junc = y_um[jy[0]] if len(jy) else np.nan
    # depletion edge: below the junction, hole density recovers toward the substrate Na
    Na_sub = 1e16
    p_col = ps.p_h[:, i]
    below = y_um > y_junc
    rec = np.where(below & (p_col > 0.5 * Na_sub))[0]
    y_edge = y_um[rec[0]] if len(rec) else np.nan
    return y_junc, y_edge, (y_edge - y_junc)


def saturation_current_estimate(Vg=3.0, Vth=1.64, L_um=0.35, mu_inv=300.0):
    """Documented long-channel saturation estimate Id/W = (1/L) mu Cox (Vg-Vth)^2 / 2."""
    Cox = eps_ox * eps0 / (DeviceParams.tox * UM)     # F/cm^2
    L = L_um * UM                                     # cm
    Idw = (1.0 / L) * mu_inv * Cox * (Vg - Vth) ** 2 / 2.0   # A/cm width
    return Idw * UM, Cox   # A/um width, Cox


if __name__ == "__main__":
    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=90, Ny=80)
    Nd, Na, Nnet = build_doping(p, X, Y)

    # confirm retuned metallurgical junction depth under the drain
    i = np.argmin(np.abs(x - 0.75))
    col = Nnet[:, i]
    jy = np.where(np.diff(np.sign(col)) < 0)[0]
    print("Retuned metallurgical junction depth under drain: %.3f um (target ~0.17)"
          % (y[jy[0]] if len(jy) else np.nan))

    ps = Poisson2D(x, y, Nnet, p, Phi_gate=0.30)
    ps.solve(Vg=0.0, Vd=0.0)

    Vbi, Xp, W = builtin_and_depletion()
    yj, ye, Wp = extract_depletion_from_poisson(ps)
    print("\n=== n+/substrate junction benchmark (equilibrium) ===")
    print("  V_bi analytic          : %.3f V" % Vbi)
    print("  depletion into p (anal): Xp = %.3f um  (total W = %.3f um)" % (Xp, W))
    print("  Poisson junction depth : %.3f um" % yj)
    print("  Poisson depletion edge : %.3f um  ->  W_Poisson = %.3f um" % (ye, Wp))

    Idw, Cox = saturation_current_estimate()
    print("\n=== drift-diffusion saturation-current estimate ===")
    print("  Cox = %.3e F/cm^2,  mu_inv=300 cm^2/Vs,  L=0.35 um,  Vg-Vth=1.36 V" % Cox)
    print("  Id/W (sat estimate)    : %.3e A/um   (DD extracted ~0.27 mA/um)" % Idw)
