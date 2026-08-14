"""
coupled_she.py -- Damped SHE <-> Poisson <-> hole-continuity outer iteration.

This closes the coupling that was one-way in the interim reproduction:
  1) solve SHE on current phi, including impact-ionization primary loss/secondary gain;
  2) solve hole continuity with G_ii as pair-generation source and n fixed to n_SHE;
  3) solve Poisson with the fixed n_SHE,p charge;
  4) damp phi and repeat.

Because the underlying device/scattering models remain reconstructed/surrogate, this is
still a reproduction model rather than a claim of exact Liang-parameter equivalence.
"""

import os
import time
import numpy as np

from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver
from she2d import SHE2D
from constants import Vt, ni


def solve_coupled(Nx=40, Ny=34, Vg=3.0, Vd=3.0, Vs=0.0, Vb=0.0,
                  dH_eV=0.0125, max_outer=12, phi_damp=0.25,
                  tol_phi=2e-3, she_tol=1e-8, verbose=True):
    pdev = DeviceParams()
    x, y, X, Y = make_mesh(pdev, Nx=Nx, Ny=Ny)
    Nd, Na, Nnet = build_doping(pdev, X, Y)
    ps = Poisson2D(x, y, Nnet, pdev, Phi_gate=0.30)
    dd = DDSolver(ps)

    # Drift-diffusion gives a robust initial electrostatic/hole state only.
    phi, n_dd, p_h = dd.solve(Vs=Vs, Vd=Vd, Vg=Vg, Vb=Vb,
                              max_gummel=90, tol=1e-5, verbose=False)

    history = []
    she = None
    for it in range(max_outer):
        t0 = time.time()
        she = SHE2D(x, y, phi, ps.contact_type, dd.n_eq, dHi_eV=dH_eV,
                    include_impact_ionization=True, absorbing_top=True)
        she.assemble()
        she.solve(tol=she_tol)
        n_she, Te, Gii = she.moments()

        # Pair generation drives holes.  n_SHE is fixed during this hole solve.
        p_h = dd.solve_holes(phi, n_she, p_init=p_h, generation=Gii,
                             max_iter=30, tol=1e-5)

        # Robust Gummel-Poisson update.  Effective quasi-Fermi potentials are chosen
        # so that the Poisson carrier model exactly matches n_SHE and p at the CURRENT
        # phi; their Boltzmann derivative is used only as an approximate Newton Jacobian.
        # The SHE is rerun after phi changes, so n is not constitutively replaced by DD.
        phi_n_eff = phi - Vt * np.log(np.maximum(n_she, 1e-30) / ni)
        phi_p_eff = phi + Vt * np.log(np.maximum(p_h, 1e-30) / ni)
        phi_target = ps.solve(phi_init=phi, phi_n=phi_n_eff, phi_p=phi_p_eff,
                              Vs=Vs, Vd=Vd, Vg=Vg, Vb=Vb,
                              max_newton=60, tol=1e-8, damp_clip=0.25)
        dphi = phi_target - phi
        phi = phi + phi_damp * dphi
        # Contacts are exact Dirichlet values; keep them exact after damping.
        phi[ps.dirichlet] = phi_target[ps.dirichlet]
        err = float(np.max(np.abs(phi_damp * dphi)))

        cons = she.conservation_diagnostics()
        rec = {
            "iteration": it,
            "max_dphi_V": err,
            "n_max_cm3": float(n_she.max()),
            "p_max_cm3": float(p_h.max()),
            "Te_max_K": float(Te.max()),
            "Gii_max_cm3s": float(Gii.max()),
            "relative_number_balance": float(cons["relative_number_balance"]),
            "drain_current_A_per_um": float(cons["drain_current_A_per_um"]),
            "seconds": float(time.time() - t0),
        }
        history.append(rec)
        if verbose:
            print("outer %2d  dphi=%.3e V  nmax=%.3e  pmax=%.3e  "
                  "Temax=%.0f K  Gmax=%.3e  Id=%.3e A/um  bal=%.2e  (%.1fs)" %
                  (it, err, rec["n_max_cm3"], rec["p_max_cm3"], rec["Te_max_K"],
                   rec["Gii_max_cm3s"], rec["drain_current_A_per_um"],
                   rec["relative_number_balance"], rec["seconds"]))
        if err < tol_phi:
            break

    # Final SHE solve on the final damped phi so all returned moments correspond to phi.
    she = SHE2D(x, y, phi, ps.contact_type, dd.n_eq, dHi_eV=dH_eV,
                include_impact_ionization=True, absorbing_top=True)
    she.assemble(); she.solve(tol=she_tol)
    n_she, Te, Gii = she.moments()
    p_h = dd.solve_holes(phi, n_she, p_init=p_h, generation=Gii,
                         max_iter=30, tol=1e-5)

    return {
        "x": x, "y": y, "phi": phi, "n": n_she, "p": p_h,
        "Te": Te, "Gii": Gii, "vx": she.vx, "vy": she.vy,
        "F3d": she.F3d, "H": she.H,
        "Gamma_x_face": she.Gamma_x_face,
        "Gamma_y_face": she.Gamma_y_face,
        "history": history,
        "conservation": she.conservation_diagnostics(),
        "she": she,
    }


if __name__ == "__main__":
    result = solve_coupled()
    outdir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "she2d_coupled_result.npz"),
             x=result["x"], y=result["y"], phi=result["phi"],
             n=result["n"], p=result["p"], Te=result["Te"], Gii=result["Gii"],
             vx=result["vx"], vy=result["vy"],
             F3d=result["F3d"].astype(np.float32), H=result["H"],
             Gamma_x_face=result["Gamma_x_face"],
             Gamma_y_face=result["Gamma_y_face"])
    print("final conservation:", result["conservation"])
    print("saved data/she2d_coupled_result.npz")
