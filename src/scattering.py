"""
scattering.py -- Scattering surrogates and l=0 collision helpers for first-order SHE.

Design goals:
  * The optical-phonon operator is number-conserving and satisfies detailed balance.
  * The acoustic energy operator is written in the DOS-weighted conservative form
        -d/deps [ Z A (df0/deps + f0/kT) ]
    so conservation is with respect to n = int Z f0 deps.
  * tau_1(eps) is a CALIBRATED COMBINED MOMENTUM-RELAXATION SURROGATE.  The current
    implementation does not separately evaluate the Table-I acoustic, impurity, and
    surface-roughness rates; those parameters are retained elsewhere for provenance.
  * The impact-ionization rate is the disclosed Keldysh-like surrogate used by the
    device solver.  Redistribution is assembled in she2d.py because it depends on the
    local total-energy H grid.

Units: SI internally (kinetic energy passed to public helpers in eV for convenience).
"""

import numpy as np
import scipy.sparse as sp

from constants import eV, kB, T
import bands

# ---- calibrated surrogate constants (see report honesty notes) ----
TAU1_C = 1.0e-14          # tau_1 = TAU1_C / (sqrt(gamma) gamma') [s]
E_OP = 0.05               # optical phonon energy [eV]
TAU_OP_REF = 1.0e-13      # target optical scattering time [s] at E_REF
E_REF = 0.15              # optical calibration energy [eV]

# Impact ionization: disclosed Keldysh-like diagnostic/model surrogate.
II_ETH = 1.10              # threshold kinetic energy [eV]
II_PREF = 2.0e13           # prefactor [1/s]

# DOS-weighted optical coupling c [J m^3 / s], calibrated so c*Z(E_REF) ~ 1/tau_ref.
COP = 1.0 / (TAU_OP_REF * bands.dos(E_REF))


def tau1(eps_eV):
    """Combined momentum-relaxation surrogate tau_1(eps) [s]."""
    g = bands.gamma(eps_eV)
    gp = bands.gamma_prime(eps_eV)
    denom = np.sqrt(np.maximum(g, 1e-6)) * gp
    return TAU1_C / denom


def Dcoef(eps_eV):
    """Generalized diffusion coefficient D(eps) = v^2 tau_1 / 3 [m^2/s]."""
    v = bands.velocity(eps_eV)
    return v ** 2 * tau1(eps_eV) / 3.0


def optical_operator(eps_eV):
    """
    Build Qmat such that (Z*Q)_i = (Qmat @ f0)_i for the DOS-weighted optical
    phonon collision term on a uniform kinetic-energy grid.

    (Z Q)_i = COP Z_i [ (N+1) Z_{i+m} f_{i+m} + N Z_{i-m} f_{i-m}
                        - ((N+1) Z_{i-m} + N Z_{i+m}) f_i ] .
    """
    ne = len(eps_eV)
    deps = eps_eV[1] - eps_eV[0]
    m = int(round(E_OP / deps))
    x = E_OP * eV / (kB * T)
    N = 1.0 / np.expm1(x)
    Z = bands.dos(eps_eV)

    rows, cols, vals = [], [], []
    for i in range(ne):
        ip = i + m
        im = i - m
        if ip < ne:
            # in from i+m by emission; out from i by absorption
            rows.append(i); cols.append(ip); vals.append(COP * Z[i] * (N + 1) * Z[ip])
            rows.append(i); cols.append(i); vals.append(-COP * Z[i] * N * Z[ip])
        if im >= 0:
            # in from i-m by absorption; out from i by emission
            rows.append(i); cols.append(im); vals.append(COP * Z[i] * N * Z[im])
            rows.append(i); cols.append(i); vals.append(-COP * Z[i] * (N + 1) * Z[im])

    return sp.csr_matrix((vals, (rows, cols)), shape=(ne, ne))


V_S = 9.04e3   # longitudinal sound velocity [m/s]


def acoustic_energy_coeff(eps_eV):
    """A(eps)=4 m* v_s^2 eps/tau_ac [J^2/s], with tau_ac=tau_1 surrogate."""
    return 4.0 * bands.MSTAR * V_S ** 2 * (eps_eV * eV) / tau1(eps_eV)


def impact_ionization_rate(eps_eV):
    """Keldysh-like impact-ionization event rate 1/tau_ii(eps) [1/s]."""
    eps = np.asarray(eps_eV, dtype=float)
    x = np.maximum((eps - II_ETH) / II_ETH, 0.0)
    return II_PREF * x ** 2


def _bern(x):
    """Bernoulli function, scalar or ndarray safe."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    out[small] = 1.0 - x[small] / 2.0
    out[~small] = x[~small] / np.expm1(x[~small])
    return float(out) if out.ndim == 0 else out


def acoustic_energy_operator(eps_eV):
    """
    Conservative Scharfetter-Gummel discretization of

        -d/deps [ Z A (df0/deps + f0/kT) ].

    L_ac @ f0 uses the same positive-diagonal / nonpositive-offdiagonal sign
    convention as the field-diffusion operator.  The Maxwellian is an exact
    discrete null on a closed (zero-flux) energy interval.
    """
    ne = len(eps_eV)
    deps_J = (eps_eV[1] - eps_eV[0]) * eV
    Z = bands.dos(eps_eV)
    ZA = Z * acoustic_energy_coeff(eps_eV)
    face = 0.5 * (ZA[:-1] + ZA[1:])
    delta = deps_J / (kB * T)
    Bp, Bm = _bern(delta), _bern(-delta)

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
    """Relative optical-operator residual for a 300-K Maxwellian."""
    Qmat = optical_operator(eps_eV)
    fMB = np.exp(-eps_eV * eV / (kB * T))
    r = Qmat.dot(fMB)
    Z = bands.dos(eps_eV)
    scale = np.max(np.abs(COP * Z * Z * fMB))
    return np.max(np.abs(r)) / (scale + 1e-300)


def check_number_conservation(eps_eV):
    """Return relative column-sum defects of closed optical and acoustic operators."""
    Q = optical_operator(eps_eV)
    L = acoustic_energy_operator(eps_eV)
    qscale = np.max(np.abs(Q.data)) if Q.nnz else 1.0
    lscale = np.max(np.abs(L.data)) if L.nnz else 1.0
    qdef = np.max(np.abs(np.asarray(Q.sum(axis=0)).ravel())) / (qscale + 1e-300)
    ldef = np.max(np.abs(np.asarray(L.sum(axis=0)).ravel())) / (lscale + 1e-300)
    return qdef, ldef
