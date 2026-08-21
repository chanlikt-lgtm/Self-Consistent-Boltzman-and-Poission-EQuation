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
                  tol_phi=2e-3, she_tol=1e-8, Phi_gate=0.30, verbose=True,
                  accel="picard", aa_depth=6, aa_beta=0.5, freeze_H=False,
                  aa_restart=6, aa_rcond=1e-8, aa_step_cap=5.0,
                  x_mesh=None, y_mesh=None,
                  phi_init=None, p_init=None, phi_span_H_override=None):
    """Coupled SHE<->Poisson<->hole outer iteration.

    accel="picard"   : damped Picard, phi <- phi + phi_damp*(G(phi)-phi)  (default; unchanged).
    accel="anderson" : Anderson acceleration AA(aa_depth) with mixing aa_beta on the free (non-
                       contact) potential DOFs. The convergence measure is the TRUE fixed-point
                       residual max|G(phi)-phi| (not the damped step), so `max_dphi_V`/`tol_phi`
                       are on the raw residual in this mode.
    """
    pdev = DeviceParams()
    if x_mesh is not None:
        # Custom (e.g. drain-graded) tensor mesh: overrides make_mesh. The FV solver is
        # non-uniform-aware, so a graded x concentrates resolution where G_ii lives.
        x = np.asarray(x_mesh, dtype=float); y = np.asarray(y_mesh, dtype=float)
        X, Y = np.meshgrid(x, y); Nx, Ny = len(x), len(y)
    else:
        x, y, X, Y = make_mesh(pdev, Nx=Nx, Ny=Ny)
    Nd, Na, Nnet = build_doping(pdev, X, Y)
    ps = Poisson2D(x, y, Nnet, pdev, Phi_gate=Phi_gate)
    dd = DDSolver(ps)

    # Drift-diffusion gives a robust initial electrostatic/hole state only.
    phi, n_dd, p_h = dd.solve(Vs=Vs, Vd=Vd, Vg=Vg, Vb=Vb,
                              max_gummel=90, tol=1e-5, verbose=False)
    # Warm-start: override the DD initial guess with a previously converged (or near-converged)
    # coupled state on the SAME mesh, so a continuation run finishes the slow plateau drift instead
    # of retracing the whole trajectory. Contacts stay Dirichlet (set inside ps.solve each step).
    if phi_init is not None:
        phi = np.asarray(phi_init, dtype=float).copy()
    if p_init is not None:
        p_h = np.asarray(p_init, dtype=float).copy()

    # Finite-volume cell-area weights (um^2 -> cm^2) for the integrated generation G_tot = int Gii dA,
    # so G_tot is tracked every outer iteration (not just post-hoc) and its plateau can be audited.
    _dxc = np.zeros(len(x)); _dxc[1:-1] = 0.5 * (x[2:] - x[:-2]); _dxc[0] = 0.5 * (x[1] - x[0]); _dxc[-1] = 0.5 * (x[-1] - x[-2])
    _dyc = np.zeros(len(y)); _dyc[1:-1] = 0.5 * (y[2:] - y[:-2]); _dyc[0] = 0.5 * (y[1] - y[0]); _dyc[-1] = 0.5 * (y[-1] - y[-2])
    _area_cm2 = np.outer(_dyc, _dxc) * 1e-8

    history = []
    she = None
    free = ~ps.dirichlet            # free (non-contact) DOFs for Anderson acceleration
    X_hist, F_hist = [], []         # Anderson history of iterate / residual on the free DOFs
    best_raw, stall = np.inf, 0     # stagnation tracking for Anderson restart
    # Freeze the H-grid across the outer loop: a fixed span bounding the loop's phi range keeps
    # NH constant, so the tail-sensitive Gii moment is not re-discretized each iteration.
    # phi_span_H_override lets a caller impose ONE COMMON span across several runs (e.g. a spatial
    # Richardson sequence), so the frozen H-grid is byte-identical across meshes and no energy-
    # discretization change is smuggled into a spatial-only study. The caller must ensure the span
    # bounds every run's phi range (checked below).
    if phi_span_H_override is not None:
        phi_span_H = (float(phi_span_H_override[0]), float(phi_span_H_override[1]))
        # HARD FAIL: if the imposed common span does not bound this mesh's phi, the "common" energy
        # domain no longer safely covers -q*phi..eps_max-q*phi for this solution. Abort and require a
        # wider common span for ALL grids -- do NOT silently proceed on an under-covering H-grid.
        if phi.min() < phi_span_H[0] or phi.max() > phi_span_H[1]:
            raise ValueError(
                "phi range [%.4f, %.4f] V leaves imposed common H-span %s V -- widen COMMON_PHI_SPAN "
                "for ALL grids and rerun the whole sequence (energy domain must bound every mesh)."
                % (phi.min(), phi.max(), phi_span_H))
    elif freeze_H:
        phi_span_H = (float(phi.min()) - 0.25, float(phi.max()) + 0.05)
    else:
        phi_span_H = None
    conv = None                     # last in-loop SHE state (for internally-consistent return)
    for it in range(max_outer):
        t0 = time.time()
        she = SHE2D(x, y, phi, ps.contact_type, dd.n_eq, dHi_eV=dH_eV,
                    include_impact_ionization=True, absorbing_top=True, phi_span_H=phi_span_H)
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
        # ---- fixed-point update: damped Picard or Anderson acceleration ----
        f_full = phi_target - phi                       # residual G(phi) - phi (raw)
        # Snapshot this iteration's SHE solution on the CURRENT (pre-update) potential. The
        # returned/saved fields are this last snapshot, so its Gii, its residual (err below), and
        # the last-iterate plateau all refer to ONE potential -- no re-solve on a further-updated
        # phi whose residual was never measured.
        conv = dict(phi=phi.copy(), she=she, n=n_she, Te=Te, Gii=Gii, p=p_h)
        if accel == "anderson":
            raw = float(np.max(np.abs(f_full[free])))   # true fixed-point residual
            xk = phi[free].copy(); f = f_full[free].copy()   # xk: free-DOF iterate (NOT the mesh x)
            # Stagnation restart: once the residual stops improving, the AA difference history has
            # become near-dependent and the least-squares step just shuffles the iterate inside a
            # noise band. Flush the history so this iterate restarts a fresh (damped-Picard) cycle.
            if raw < 0.999 * best_raw:
                best_raw, stall = raw, 0
            else:
                stall += 1
                if aa_restart and stall >= aa_restart:
                    X_hist.clear(); F_hist.clear(); stall = 0
            X_hist.append(xk); F_hist.append(f)
            if len(F_hist) > aa_depth + 1:
                X_hist.pop(0); F_hist.pop(0)
            m = len(F_hist) - 1
            if m == 0:
                x_new = xk + aa_beta * f                # first step: damped Picard
            else:
                dF = np.column_stack([F_hist[-i] - F_hist[-i - 1] for i in range(1, m + 1)])
                dX = np.column_stack([X_hist[-i] - X_hist[-i - 1] for i in range(1, m + 1)])
                gamma, *_ = np.linalg.lstsq(dF, f, rcond=aa_rcond)
                x_new = xk + aa_beta * f - (dX + aa_beta * dF) @ gamma
                # Safeguard: cap a runaway AA extrapolation to aa_step_cap x the damped-Picard
                # step. The least-squares extrapolation occasionally produces a bad iterate (seen
                # as a late residual/n spike); clipping its length keeps AA acceleration while
                # rejecting the overshoot. Does not trigger on well-behaved (clean-plateau) runs.
                step = x_new - xk
                pn = float(np.linalg.norm(aa_beta * f))
                sn = float(np.linalg.norm(step))
                if pn > 0.0 and sn > aa_step_cap * pn:
                    x_new = xk + (aa_step_cap * pn / sn) * step
            phi = phi.copy()
            phi[free] = x_new
            phi[ps.dirichlet] = phi_target[ps.dirichlet]
            err = raw
        else:
            phi = phi + phi_damp * f_full
            # Contacts are exact Dirichlet values; keep them exact after damping.
            phi[ps.dirichlet] = phi_target[ps.dirichlet]
            err = float(np.max(np.abs(phi_damp * f_full)))

        cons = she.conservation_diagnostics()
        rec = {
            "iteration": it,
            "NH": int(she.NH),
            "max_dphi_V": err,
            "n_max_cm3": float(n_she.max()),
            "p_max_cm3": float(p_h.max()),
            "Te_max_K": float(Te.max()),
            "Gii_max_cm3s": float(Gii.max()),
            "G_tot_cm1s": float(np.sum(Gii * _area_cm2)),
            "relative_number_balance": float(cons["relative_number_balance"]),
            "drain_current_A_per_um": float(cons["drain_current_A_per_um"]),
            "seconds": float(time.time() - t0),
        }
        history.append(rec)
        if verbose:
            print("outer %2d  dphi=%.3e V  NH=%d  nmax=%.3e  pmax=%.3e  "
                  "Temax=%.0f K  Gmax=%.3e  Id=%.3e A/um  bal=%.2e  (%.1fs)" %
                  (it, err, rec["NH"], rec["n_max_cm3"], rec["p_max_cm3"], rec["Te_max_K"],
                   rec["Gii_max_cm3s"], rec["drain_current_A_per_um"],
                   rec["relative_number_balance"], rec["seconds"]))
        if err < tol_phi:
            break

    # Internal consistency: return the LAST in-loop SHE solve rather than re-solving on a further-
    # updated phi. Its moments, its residual (history[-1]["max_dphi_V"]), and its potential all
    # correspond to one iterate, so the quoted converged Gii is exactly the last plateau point.
    if conv is not None:
        she = conv["she"]
        phi, n_she, Te, Gii, p_h = conv["phi"], conv["n"], conv["Te"], conv["Gii"], conv["p"]
    else:                                   # max_outer == 0: single solve on the initial phi
        she = SHE2D(x, y, phi, ps.contact_type, dd.n_eq, dHi_eV=dH_eV,
                    include_impact_ionization=True, absorbing_top=True, phi_span_H=phi_span_H)
        she.assemble(); she.solve(tol=she_tol)
        n_she, Te, Gii = she.moments()
        p_h = dd.solve_holes(phi, n_she, p_init=p_h, generation=Gii, max_iter=30, tol=1e-5)

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
    import sys
    Nx = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    Ny = int(sys.argv[2]) if len(sys.argv) > 2 else 34
    Phi_gate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.30
    result = solve_coupled(Nx=Nx, Ny=Ny, Phi_gate=Phi_gate)
    outdir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(outdir, exist_ok=True)
    tag = "" if (Nx, Ny) == (40, 34) and Phi_gate == 0.30 else "_%dx%d_Phg%.2f" % (Nx, Ny, Phi_gate)
    fn = "she2d_coupled_result%s.npz" % tag
    np.savez(os.path.join(outdir, fn),
             x=result["x"], y=result["y"], phi=result["phi"],
             n=result["n"], p=result["p"], Te=result["Te"], Gii=result["Gii"],
             vx=result["vx"], vy=result["vy"],
             F3d=result["F3d"].astype(np.float32), H=result["H"],
             Gamma_x_face=result["Gamma_x_face"],
             Gamma_y_face=result["Gamma_y_face"])
    Gii = result["Gii"]; jx, ix = np.unravel_index(int(np.argmax(Gii)), Gii.shape)
    print("Gii_max=%.4e at (x=%.3f,y=%.3f) um  Te_max=%.1f  n_max=%.4e" %
          (Gii[jx, ix], result["x"][ix], result["y"][jx], result["Te"].max(), result["n"].max()))
    print("final conservation:", result["conservation"])
    print("saved data/%s" % fn)
