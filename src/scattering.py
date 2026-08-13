"""
scattering.py -- Scattering rates and the l=0 inelastic (optical/intervalley phonon)
operator for the first-order SHE, on a uniform kinetic-energy grid.

Design goals:
  * The optical-phonon operator is number-conserving and satisfies DETAILED BALANCE:
    a Maxwellian f0 ~ exp(-eps/kT) is an exact null of Q (verified numerically).
  * Momentum relaxation time tau_1(eps) collects the elastic (acoustic-like) processes;
    it feeds the generalized diffusion coefficient D(eps) = v^2 tau_1 / 3.

Units: SI internally (eps in Joules). Grid supplied in eV for convenience.
"""

import numpy as np
import scipy.sparse as sp

from constants import hbar, eV, kB, T, Vt
import bands

# ---- tunable model constants (calibrated; see report honesty notes) ----
TAU1_C = 1.0e-14          # sets tau_1(eps) = TAU1_C / (sqrt(gamma) gamma')  [s], acoustic-like
E_OP = 0.05               # optical phonon energy [eV] (Table I)
TAU_OP_REF = 1.0e-13      # target optical scattering time [s] at E_REF
E_REF = 0.15              # reference energy for calibrating the optical coupling [eV]

# DOS-weighted optical coupling c [J m^3 / s] so that 1/tau_op(E_REF) ~ c*Z(E_REF).
# Derived (not guessed) to keep units correct: c*Z has units 1/s.
COP = 1.0 / (TAU_OP_REF * bands.dos(E_REF))


def tau1(eps_eV):
    """Momentum relaxation time [s]; acoustic-like, ~ 1/DOS (decreases with energy)."""
    g = bands.gamma(eps_eV)
    gp = bands.gamma_prime(eps_eV)
    denom = np.sqrt(np.maximum(g, 1e-6)) * gp
    return TAU1_C / denom


def Dcoef(eps_eV):
    """Generalized diffusion coefficient D(eps) = v^2 tau_1 / 3 [m^2/s]."""
    v = bands.velocity(eps_eV)      # m/s
    return v ** 2 * tau1(eps_eV) / 3.0


def optical_operator(eps_eV):
    """
    Build sparse matrix Qmat such that (Z*Q)[i] = (Qmat @ f0)[i], i.e. the DOS-weighted
    l=0 optical/intervalley phonon collision term on the uniform eps grid.

    (Z Q)_i = COP * Z_i * [ (N+1) Z_{i+m} f_{i+m} + N Z_{i-m} f_{i-m}
                            - ((N+1) Z_{i-m} + N Z_{i+m}) f_i ]
    with m = round(E_OP/deps).  Satisfies detailed balance (Maxwellian -> 0).
    """
    ne = len(eps_eV)
    deps = eps_eV[1] - eps_eV[0]
    m = int(round(E_OP / deps))
    x = E_OP * eV / (kB * T)
    N = 1.0 / (np.expm1(x))          # Bose occupation
    Z = bands.dos(eps_eV)            # generalized DOS [1/(J m^3)]

    rows, cols, vals = [], [], []
    for i in range(ne):
        ip = i + m
        im = i - m
        # in from i+m via emission (rate ~ N+1), lands at i
        if ip < ne:
            rows.append(i); cols.append(ip); vals.append(COP * Z[i] * (N + 1) * Z[ip])
            # out of i via absorption to i+m (rate ~ N)
            rows.append(i); cols.append(i); vals.append(-COP * Z[i] * N * Z[ip])
        # in from i-m via absorption (rate ~ N), lands at i
        if im >= 0:
            rows.append(i); cols.append(im); vals.append(COP * Z[i] * N * Z[im])
            # out of i via emission to i-m (rate ~ N+1)
            rows.append(i); cols.append(i); vals.append(-COP * Z[i] * (N + 1) * Z[im])

    Qmat = sp.csr_matrix((vals, (rows, cols)), shape=(ne, ne))
    return Qmat


V_S = 9.04e3   # longitudinal sound velocity [m/s] (u_sound = 9.04e5 cm/s, Table I)


def acoustic_energy_coeff(eps_eV):
    """Acoustic-phonon energy-diffusion coefficient A(eps) = 4 m* v_s^2 eps / tau_ac  [J^2/s].
    Physically the mean-square energy transfer rate for quasi-elastic acoustic scattering
    (energy transfer ~ 2 v_s p per collision). eps in eV -> A in J^2/s."""
    return 4.0 * bands.MSTAR * V_S ** 2 * (eps_eV * eV) / tau1(eps_eV)


def _bern(x):
    if abs(x) < 1e-10:
        return 1.0 - x / 2.0
    return x / np.expm1(x)


def acoustic_energy_operator(eps_eV):
    """
    Conservative Scharfetter-Gummel discretization of the acoustic-phonon energy
    Fokker-Planck operator  -d/deps[ Z A ( df0/deps + f0/kT ) ], returned as a sparse
    matrix L_ac with L_ac @ f0 = LHS (same sign convention as the field diffusion Ldiff:
    positive diagonal, negative off-diagonals -> M-matrix).

    It has the Maxwellian f0 ~ exp(-eps/kT) as its EXACT discrete null (detailed balance),
    and -- crucially -- it couples ADJACENT energy nodes, making the combined collision
    operator irreducible so the E=0 equilibrium is the unique Maxwellian.
    """
    ne = len(eps_eV)
    deps_J = (eps_eV[1] - eps_eV[0]) * eV
    Z = bands.dos(eps_eV)
    ZA = Z * acoustic_energy_coeff(eps_eV)
    face = 0.5 * (ZA[:-1] + ZA[1:])       # face k between i=k and i=k+1
    delta = deps_J / (kB * T)             # >0
    Bp, Bm = _bern(delta), _bern(-delta)  # B(delta), B(-delta); Bp/Bm = exp(-delta)

    rows, cols, vals = [], [], []
    for i in range(ne):
        if i + 1 < ne:
            c = face[i] / deps_J
            rows += [i, i]; cols += [i, i + 1]; vals += [c * Bp, -c * Bm]
        if i - 1 >= 0:
            c = face[i - 1] / deps_J
            rows += [i, i]; cols += [i, i - 1]; vals += [c * Bm, -c * Bp]
    return sp.csr_matrix((vals, (rows, cols)), shape=(ne, ne))


def check_detailed_balance(eps_eV):
    """Return max |Qmat @ f_MB| / norm, which should be ~0 for a Maxwellian."""
    Qmat = optical_operator(eps_eV)
    fMB = np.exp(-eps_eV * eV / (kB * T))
    r = Qmat.dot(fMB)
    Z = bands.dos(eps_eV)
    scale = np.max(np.abs(COP * Z * Z * fMB))
    return np.max(np.abs(r)) / (scale + 1e-300)
