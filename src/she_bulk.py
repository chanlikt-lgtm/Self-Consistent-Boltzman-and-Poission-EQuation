"""
she_bulk.py -- Homogeneous-bulk validation of the SHE energy-transport + scattering engine.

In a spatially uniform bulk with uniform field E, the first-order SHE equation reduces to
the energy-space two-point boundary-value problem
    -d/deps [ Z(eps) D_eps(eps) df0/deps ] = Z(eps) Q[f0],   D_eps = (qE)^2 D(eps),
solved here by box integration in energy.  This isolates and validates:
   (1) detailed balance of the optical operator  (Qmat @ Maxwellian ~ 0),
   (2) E=0  ->  f0 is a Maxwellian at the lattice temperature,
   (3) E>0  ->  the high-energy tail heats monotonically with field.
These are the go/no-go checks before trusting the operators in the 2-D device solve.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from constants import q, eV, kB, T, Vt
import bands
import scattering as sc


def solve_bulk(E_Vcm, eps_max=3.02, ne=1209, absorbing_top=True):
    """Solve the bulk energy BVP at field E_Vcm [V/cm]. Returns eps[eV], f0 (normalized).
    ne=1209 -> deps=0.0025 eV (E_op=20 steps); moment Te converges to 309.7 K analytic.

    Boundary conditions (audit-3 Major #3): the two energy boundaries are physically
    distinct and treated separately --
      * eps=0   : physical band-edge turning point -> reflecting (natural zero-flux),
      * eps_max : NUMERICAL cutoff -> absorbing (f0(eps_max)=0) when absorbing_top=True,
        so hot carriers are not artificially retained. Validate by eps_max-convergence."""
    eps = np.linspace(1e-4, eps_max, ne)
    deps_eV = eps[1] - eps[0]
    deps_J = deps_eV * eV

    Z = bands.dos(eps)                 # [1/(J m^3)]
    D = sc.Dcoef(eps)                  # [m^2/s]
    E_SI = E_Vcm * 100.0               # V/m
    qE = q * E_SI                      # J/m
    Deps = (qE) ** 2 * D               # [J^2/s]

    ZDeps = Z * Deps                   # face coefficient [J/(m^3 s)]
    # face-centered coefficient (arithmetic mean)
    face = 0.5 * (ZDeps[:-1] + ZDeps[1:])   # length ne-1, face k between i=k and i=k+1

    # box-integrated diffusion operator L_diff (tridiagonal):
    #  -[flux_{i+1/2} - flux_{i-1/2}] with flux_{i+1/2} = face_i (f_{i+1}-f_i)/deps
    rows, cols, vals = [], [], []
    for i in range(ne):
        if i + 1 < ne:
            c = face[i] / deps_J
            rows += [i, i]; cols += [i, i + 1]; vals += [c, -c]     # -( face(f_{i+1}-f_i) )
        if i - 1 >= 0:
            c = face[i - 1] / deps_J
            rows += [i, i]; cols += [i, i - 1]; vals += [c, -c]     # +( face(f_i-f_{i-1}) )
    Ldiff = sp.csr_matrix((vals, (rows, cols)), shape=(ne, ne))

    Qmat = sc.optical_operator(eps)    # (Z Q)_i = Qmat @ f0   [1/(J m^3 s)]
    Lac = sc.acoustic_energy_operator(eps)  # acoustic energy Fokker-Planck (couples adjacent nodes)
    # LHS = field diffusion (Ldiff) + acoustic FP (Lac); RHS = optical in/out.  M f0 = 0.
    M = Ldiff + Lac - deps_J * Qmat

    # normalization / regularization: pin low-energy node to break the null space
    M = M.tolil()
    ipin = 2
    M.rows[ipin] = [ipin]; M.data[ipin] = [1.0]
    b = np.zeros(ne); b[ipin] = 1.0
    if absorbing_top:
        # absorbing NUMERICAL cutoff at eps_max: f0(eps_max)=0 (not a reflecting turning point)
        M.rows[ne - 1] = [ne - 1]; M.data[ne - 1] = [1.0]
        b[ne - 1] = 0.0
    M = M.tocsr()

    f0 = spla.spsolve(M, b)
    f0 = np.maximum(f0, 1e-300)
    # normalize to unit density n = int Z f0 deps
    n = np.trapezoid(Z * f0, eps * eV)
    f0 = f0 / n
    return eps, f0


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.linspace(1e-4, 3.02, 1209)

    # (1) detailed balance
    db = sc.check_detailed_balance(eps)
    print("Detailed-balance residual (Qmat @ Maxwellian), relative: %.2e" % db)
    print("  COP = %.3e J m^3/s  (derived from tau_op_ref=%.1e s)" % (sc.COP, sc.TAU_OP_REF))

    # (2,3) field sweep
    fields = [0.0, 5e4, 1e5, 2e5, 3e5]   # V/cm
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    Te_list = []
    for E in fields:
        ep, f0 = solve_bulk(E)
        # electron temperature from <eps>
        Z = bands.dos(ep)
        n = np.trapezoid(Z * f0, ep * eV)
        emean = np.trapezoid(ep * eV * Z * f0, ep * eV) / n
        Te = (2.0 / 3.0) * emean / kB
        Te_list.append(Te)
        ax.semilogy(ep, f0 / f0[2], label=f"E={E/1e3:.0f} kV/cm,  Te={Te:.0f} K")
    # reference Maxwellian slope
    fMB = np.exp(-(eps - eps[2]) * eV / (kB * T))
    ax.semilogy(eps, fMB, "k--", lw=1, label="Maxwellian (300 K)")
    ax.set_xlabel("kinetic energy eps [eV]"); ax.set_ylabel("f0(eps) (normalized at eps~0)")
    ax.set_ylim(1e-20, 5); ax.set_xlim(0, 2.0)
    ax.set_title("Bulk distribution vs field (heating of the tail)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(np.array(fields) / 1e3, Te_list, "o-")
    ax.set_xlabel("field E [kV/cm]"); ax.set_ylabel("electron temperature T_e [K]")
    ax.axhline(300, color="k", ls="--", lw=1, label="lattice 300 K")
    ax.set_title("Electron temperature vs field"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout(); fig.savefig("figures/she_bulk_validation.png", dpi=130)
    print("Te vs field:", [f"{E/1e3:.0f}kV/cm->{Te:.0f}K" for E, Te in zip(fields, Te_list)])
    print("wrote figures/she_bulk_validation.png")
