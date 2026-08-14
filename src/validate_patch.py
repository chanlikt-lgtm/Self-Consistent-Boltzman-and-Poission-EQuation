"""Regression/sanity checks for the audited SHE patch."""

import numpy as np

import scattering as sc
from device import DeviceParams, make_mesh, build_doping
from poisson import Poisson2D
from dd import DDSolver
from she2d import SHE2D
from diagnostics import saturation_current_estimate


def main():
    # Closed collision operators conserve number; optical Maxwellian detailed balance remains.
    eps = np.linspace(1e-4, 3.02, 1209)
    db = sc.check_detailed_balance(eps)
    qdef, adef = sc.check_number_conservation(eps)
    assert qdef < 1e-12, qdef
    assert adef < 1e-12, adef
    assert db < 1e-3, db

    # Equilibrium fixed-charge Poisson diagnostic should reproduce the nonlinear solution.
    pdev = DeviceParams()
    x, y, X, Y = make_mesh(pdev, Nx=12, Ny=10)
    Nd, Na, Nnet = build_doping(pdev, X, Y)
    ps0 = Poisson2D(x, y, Nnet, pdev, Phi_gate=0.30)
    phi0 = ps0.solve(Vg=0.0, Vd=0.0)
    phi0_picard = ps0.solve_fixed_charge(ps0.n, ps0.p_h, Vg=0.0, Vd=0.0)
    assert np.max(np.abs(phi0_picard - phi0)) < 1e-8

    # Biased DD starting point.
    ps = Poisson2D(x, y, Nnet, pdev, Phi_gate=0.30)
    dd = DDSolver(ps)
    phi, ndd, pdd = dd.solve(Vg=3.0, Vd=3.0, max_gummel=60, tol=1e-5)

    # Patched SHE: absorbing cutoff + II redistribution.
    she = SHE2D(x, y, phi, ps.contact_type, dd.n_eq, dHi_eV=0.025,
                include_impact_ionization=True, absorbing_top=True)
    she.assemble(); she.solve(); n, Te, Gii = she.moments()
    cons = she.conservation_diagnostics()

    assert she.n_orphans == 0
    assert she.scaled_residual_inf < 1e-8
    assert cons["relative_number_balance"] < 1e-5
    assert cons["cutoff_loss_per_m_s"] > 0.0
    assert cons["ii_event_rate_per_m_s"] > 0.0
    assert np.isfinite(n).all() and np.isfinite(Te).all() and np.isfinite(Gii).all()
    assert np.isfinite(she.vx).all() and np.isfinite(she.vy).all()

    # The physical carrier-tail fraction is DOS weighted and should be well-defined.
    j = int(np.argmin(np.abs(y - 0.001)))
    i = int(np.argmin(np.abs(x - 0.6)))
    fw = she.tail_fraction(j, i, 1.0, dos_weighted=True)
    fu = she.tail_fraction(j, i, 1.0, dos_weighted=False)
    assert 0.0 <= fw <= 1.0 and 0.0 <= fu <= 1.0

    # Stale diagnostics default was corrected to the extracted Vth=1.64 V.
    Id_est, Cox = saturation_current_estimate()
    assert 2.5e-4 < Id_est < 3.2e-4, Id_est

    print("PASS")
    print("optical detailed balance       %.3e" % db)
    print("optical conservation defect    %.3e" % qdef)
    print("acoustic conservation defect   %.3e" % adef)
    print("SHE scaled residual            %.3e" % she.scaled_residual_inf)
    print("SHE relative number balance    %.3e" % cons["relative_number_balance"])
    print("drain current                  %.6e A/um" % cons["drain_current_A_per_um"])
    print("peak Te                        %.1f K" % Te.max())
    print("peak Gii                       %.6e cm^-3 s^-1" % Gii.max())
    print("tail fraction DOS/raw          %.6e / %.6e" % (fw, fu))
    print("DD analytic Id estimate        %.6e A/um" % Id_est)


if __name__ == "__main__":
    main()
