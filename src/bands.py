"""
bands.py -- Spherical, nonparabolic band structure for the first-order SHE.

The paper (Sec. II) uses the spherical-band approximation with a band-structure
function gamma(eps) taken from Fiegna [12] / Brunetti [13], where the kinetic energy
eps maps to wavevector via  hbar^2 k^2 / (2 m*) = gamma(eps).  From gamma we build:

    k(eps)    = sqrt(2 m* gamma) / hbar
    v(eps)    = sqrt(2 gamma / m*) / gamma'          (group-velocity magnitude, paper Eq.5)
    Z(eps)    = g_sv/(2 pi^2) * (2 m*)^{3/2}/(2 hbar^3) * sqrt(gamma) * gamma'
                (generalized DOS ~ m*^{3/2} sqrt(gamma) gamma', paper's h(eps))

APPROXIMATION (flagged): the exact Fiegna/Brunetti multiband gamma(eps) is a tabulated
fit to full-band Si.  Here we use the analytic Kane nonparabolic form
    gamma(eps) = eps (1 + alpha eps),   alpha ~ 0.5 eV^-1,
which reproduces the low-energy Si band and the qualitative high-energy DOS/velocity
roll-off.  This is documented in the report as a band-model approximation.
"""

import numpy as np
from constants import hbar, m0, eV

# spherical-band effective mass and Kane nonparabolicity
MSTAR = 0.32 * m0        # kg  (Si conductivity/DOS spherical proxy)
ALPHA = 0.5             # eV^-1  Kane nonparabolicity
G_SV = 6 * 2            # 6 equivalent valleys * 2 spins (degeneracy)


def gamma(eps):
    """Band function gamma(eps) [eV]. eps in eV."""
    return eps * (1.0 + ALPHA * eps)


def gamma_prime(eps):
    """d gamma / d eps [dimensionless]."""
    return 1.0 + 2.0 * ALPHA * eps


def kmag(eps):
    """|k| [1/m] from eps [eV]."""
    g_J = gamma(eps) * eV
    return np.sqrt(2.0 * MSTAR * np.maximum(g_J, 0.0)) / hbar


def velocity(eps):
    """group-velocity magnitude v(eps) [m/s]; eps in eV. v = sqrt(2 gamma/m*)/gamma'."""
    g_J = gamma(eps) * eV
    v = np.sqrt(2.0 * np.maximum(g_J, 0.0) / MSTAR) / gamma_prime(eps)
    return v


def dos(eps):
    """Generalized density of states Z(eps) [states/(J*m^3)]; eps in eV.
    Z = G_SV/(2 pi^2) * (2 m*)^{3/2}/(2 hbar^3) * sqrt(gamma_J) * gamma'(dimensionless)
    Note gamma' is d(gamma[J])/d(eps[J]) = dimensionless, same as d(gamma[eV])/d(eps[eV])."""
    g_J = np.maximum(gamma(eps) * eV, 0.0)
    pref = G_SV / (2.0 * np.pi ** 2) * (2.0 * MSTAR) ** 1.5 / (2.0 * hbar ** 3)
    return pref * np.sqrt(g_J) * gamma_prime(eps)


def dos_per_eV(eps):
    """DOS per eV per m^3 (multiply J-based DOS by eV)."""
    return dos(eps) * eV


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.linspace(1e-4, 3.02, 400)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(eps, gamma(eps), label="gamma(eps)=eps(1+alpha eps)")
    ax.plot(eps, eps, "k--", alpha=0.5, label="parabolic eps")
    ax.set_xlabel("eps [eV]"); ax.set_ylabel("gamma [eV]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("band function")

    ax = axes[1]
    ax.plot(eps, velocity(eps) / 1e5)   # cm/s -> 1e5 m/s? -> plot in 1e7 cm/s
    ax.set_xlabel("eps [eV]"); ax.set_ylabel("v [1e5 m/s]")
    ax.set_title("group velocity (nonparabolic roll-vs-parabolic)"); ax.grid(alpha=0.3)

    ax = axes[2]
    # DOS in cm^-3 eV^-1
    Z = dos_per_eV(eps) * 1e-6   # per m^3 -> per cm^3
    ax.semilogy(eps, Z)
    ax.set_xlabel("eps [eV]"); ax.set_ylabel("Z(eps) [cm^-3 eV^-1]")
    ax.set_title("generalized DOS"); ax.grid(alpha=0.3, which="both")

    fig.tight_layout(); fig.savefig("figures/bands.png", dpi=130)

    # ---- N_c benchmark (audit Finding 5): state T, use a CONVERGED energy grid,
    #      and validate the DOS prefactor against the closed-form parabolic N_c. ----
    from constants import kB, eV, T, m0, h
    epsf = np.linspace(1e-5, 3.02, 4000)     # converged grid (400 pts under-integrated by ~2.5%)
    Nc_np = np.trapezoid(dos_per_eV(epsf) * 1e-6 * np.exp(-epsf * eV / (kB * T)), epsf)
    a0 = ALPHA
    globals()['ALPHA'] = 0.0
    Nc_par = np.trapezoid(dos_per_eV(epsf) * 1e-6 * np.exp(-epsf * eV / (kB * T)), epsf)
    globals()['ALPHA'] = a0
    Nc_par_analytic = 2 * (G_SV // 2) * (2 * np.pi * MSTAR * kB * T / h ** 2) ** 1.5 * 1e-6
    print("=== N_c benchmark at T = %.1f K (converged 4000-pt grid) ===" % T)
    print("  nonparabolic (alpha=%.2f): Nc = %.4e cm^-3" % (a0, Nc_np))
    print("  parabolic limit (alpha=0): Nc = %.4e cm^-3   (code)" % Nc_par)
    print("  parabolic closed-form    : Nc = %.4e cm^-3   (2 g_v (2 pi m* kT/h^2)^{3/2})" % Nc_par_analytic)
    print("  prefactor error (code vs analytic, parabolic): %.2f %%"
          % (100 * abs(Nc_par - Nc_par_analytic) / Nc_par_analytic))
    print("v at 0.1 eV = %.3e cm/s, at 1 eV = %.3e cm/s" %
          (velocity(0.1) * 100, velocity(1.0) * 100))
    print("wrote figures/bands.png")
