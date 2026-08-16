"""
she2d.py -- First-order SHE Boltzmann solver in 2-D real space + total energy H.

For a supplied electrostatic potential phi(x,y), solve

    div_r[ Z(eps) D(eps) grad_r f0 |_H ] + Z(eps) Q[f0] = 0,
    eps = H + q phi(r),

with optical phonons, a DOS-weighted conservative acoustic Fokker-Planck term,
and (optionally, default ON) the disclosed Keldysh-like impact-ionization model.
The impact-ionization event removes one primary electron and deposits two electrons
at equal post-threshold kinetic energy, so each event has net electron gain +1.

Energy boundaries:
  * eps = 0       : reflecting physical turning point;
  * eps = eps_max : absorbing numerical cutoff for spatial, acoustic, and optical
                    fluxes.  The upper boundary itself is not an active unknown.

Velocity/current moments are reconstructed from the SAME finite-volume face fluxes
used by the spatial SHE operator, rather than from central differences through the
masked 3-D array.

Units: SI internally; device inputs use um / cm^-3 at the public boundary.
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(__file__))

from constants import q, eV, kB, T
import bands
import scattering as sc

KT = kB * T
EPS_MAX = 3.02 * eV
E_OP_J = sc.E_OP * eV
II_ETH_J = sc.II_ETH * eV
UM = 1e-4                         # cm / um (device helper)


class SHE2D:
    def __init__(self, x_um, y_um, phi, contact_type, n_eq, dHi_eV=0.0125,
                 include_impact_ionization=True, absorbing_top=True):
        """
        x_um,y_um : tensor mesh [um]
        phi       : electrostatic potential [V], shape (Ny,Nx)
        contact_type : 0 none, 1 source, 2 drain, 3 body
        n_eq      : contact majority density [cm^-3]
        dHi_eV    : uniform H spacing [eV], chosen so E_op/dH is integer
        include_impact_ionization : assemble primary loss + equal-split secondary gain
        absorbing_top : impose f0=0 at eps_max through finite-volume boundary fluxes
        """
        self.x = np.asarray(x_um, dtype=float) * 1e-6
        self.y = np.asarray(y_um, dtype=float) * 1e-6
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.phi = np.asarray(phi, dtype=float)
        self.ctype = np.asarray(contact_type, dtype=int)
        self.n_eq = np.asarray(n_eq, dtype=float) * 1e6
        self.include_impact_ionization = bool(include_impact_ionization)
        self.absorbing_top = bool(absorbing_top)

        self.dH = dHi_eV * eV
        self.m_op = int(round(E_OP_J / self.dH))
        if not np.isclose(self.m_op * self.dH, E_OP_J, rtol=0.0, atol=1e-12 * eV):
            raise ValueError("E_OP/dH must be an integer for the optical jump stencil")

        phi_min, phi_max = self.phi.min(), self.phi.max()
        # Padding is not relied upon for absorption, but leaves room for diagnostics/interpolation.
        H_lo = (-phi_max) * eV - max(2, self.m_op) * self.dH
        H_hi = EPS_MAX + (-phi_min) * eV + max(2, self.m_op) * self.dH
        self.H = np.arange(H_lo, H_hi + 0.5 * self.dH, self.dH)
        self.NH = len(self.H)

        self._geom()
        self._grid()

    # ---------------------------------------------------------------- geometry
    def _geom(self):
        x, y = self.x, self.y
        dxc = np.zeros(self.Nx)
        dyc = np.zeros(self.Ny)
        dxc[1:-1] = 0.5 * (x[2:] - x[:-2])
        dxc[0] = 0.5 * (x[1] - x[0])
        dxc[-1] = 0.5 * (x[-1] - x[-2])
        dyc[1:-1] = 0.5 * (y[2:] - y[:-2])
        dyc[0] = 0.5 * (y[1] - y[0])
        dyc[-1] = 0.5 * (y[-1] - y[-2])
        self.dxc, self.dyc = dxc, dyc
        self.dxe, self.dyn = np.diff(x), np.diff(y)
        self.area = np.outer(dyc, dxc)       # m^2 per unit z-width

    def _grid(self):
        self.eps = self.H[None, None, :] + self.phi[:, :, None] * eV
        # eps=0 may be an active reflecting node. eps=eps_max is a Dirichlet boundary,
        # not an unknown, hence the strict upper inequality.
        self.active = (self.eps >= 0.0) & (self.eps < EPS_MAX)
        epsE = np.clip(self.eps / eV, 1e-8, EPS_MAX / eV)
        self.Z = np.where(self.active, bands.dos(epsE), 0.0)
        self.D = np.where(self.active, sc.Dcoef(epsE), 0.0)
        self.Aac = np.where(self.active, sc.acoustic_energy_coeff(epsE), 0.0)
        self.ii_rate = np.where(self.active, sc.impact_ionization_rate(epsE), 0.0)

        self.gid = -np.ones((self.Ny, self.Nx, self.NH), dtype=np.int64)
        self.gid[self.active] = np.arange(np.count_nonzero(self.active))
        self.Ndof = int(np.count_nonzero(self.active))

        # Boundary values of kinetic-energy coefficients at eps_max.
        emax_eV = EPS_MAX / eV
        self._ZD_top = float(bands.dos(emax_eV) * sc.Dcoef(emax_eV))
        self._ZA_top = float(bands.dos(emax_eV) * sc.acoustic_energy_coeff(emax_eV))

    # ---------------------------------------------------------- helper mappings
    def _ii_target_weights(self, j, i, k):
        """Return [(target_k, weight), ...] for equal-split II secondaries."""
        eps_src = self.eps[j, i, k]
        eps_tgt = 0.5 * (eps_src - II_ETH_J)
        if eps_tgt < 0.0:
            return []
        H_tgt = eps_tgt - self.phi[j, i] * eV
        u = (H_tgt - self.H[0]) / self.dH
        k0 = int(np.floor(u))
        a = float(u - k0)
        cand = [(k0, 1.0 - a), (k0 + 1, a)]
        valid = [(kk, w) for kk, w in cand
                 if w > 0.0 and 0 <= kk < self.NH and self.active[j, i, kk]]
        sw = sum(w for _, w in valid)
        if sw <= 0.0:
            return []
        return [(kk, w / sw) for kk, w in valid]

    # ---------------------------------------------------------------- assembly
    def assemble(self):
        Ny, Nx, NH, m = self.Ny, self.Nx, self.NH, self.m_op
        dH = self.dH
        Z, D, Aac, eps = self.Z, self.D, self.Aac, self.eps
        gid = self.gid
        N_op = 1.0 / np.expm1(E_OP_J / KT)
        c_op = sc.COP

        rows, cols, vals = [], [], []
        b = np.zeros(self.Ndof)

        # Per-DOF integrated coefficients [1/(m s)] for global balance diagnostics.
        cutoff_spatial = np.zeros(self.Ndof)
        cutoff_acoustic = np.zeros(self.Ndof)
        cutoff_optical = np.zeros(self.Ndof)
        ii_event_coeff = np.zeros(self.Ndof)
        contact_links = []   # (contact_type, interior_dof, contact_dof, Tf), exact FV boundary links

        def zint(j, i):
            ee = eps[j, i, :]
            act = self.active[j, i, :]
            if not act.any():
                return 1.0
            integrand = np.where(act, Z[j, i, :] * np.exp(-np.clip(ee / KT, 0, 200)), 0.0)
            return np.trapezoid(integrand, self.H)

        delta = dH / KT
        Bp, Bm = sc._bern(delta), sc._bern(-delta)

        for j in range(Ny):
            for i in range(Nx):
                ct = self.ctype[j, i]
                Zc = None
                for k in range(NH):
                    p = gid[j, i, k]
                    if p < 0:
                        continue

                    # Ohmic contacts: prescribed Maxwellian distribution.
                    if ct in (1, 2, 3):
                        if Zc is None:
                            Zc = zint(j, i)
                        rows.append(p); cols.append(p); vals.append(1.0)
                        b[p] = (self.n_eq[j, i]
                                * np.exp(-min(eps[j, i, k] / KT, 200.0))
                                / (Zc + 1e-300))
                        continue

                    diag = 0.0
                    eps_k = eps[j, i, k]
                    Zk = Z[j, i, k]

                    # ---------------- spatial diffusion at fixed H (box-integrated * dH)
                    neighbors = (
                        (j, i + 1, self.dxe[i] if i + 1 < Nx else 0.0, self.dyc[j]),
                        (j, i - 1, self.dxe[i - 1] if i - 1 >= 0 else 0.0, self.dyc[j]),
                        (j + 1, i, self.dyn[j] if j + 1 < Ny else 0.0, self.dxc[i]),
                        (j - 1, i, self.dyn[j - 1] if j - 1 >= 0 else 0.0, self.dxc[i]),
                    )
                    for nj, ni, hcell, wface in neighbors:
                        if hcell == 0.0 or nj < 0 or nj >= Ny or ni < 0 or ni >= Nx:
                            continue                      # physical exterior: reflecting/natural
                        q2 = gid[nj, ni, k]
                        if q2 >= 0:
                            ZDf = 0.5 * (Zk * D[j, i, k] + Z[nj, ni, k] * D[nj, ni, k])
                            Tf = dH * ZDf * wface / hcell
                            diag += Tf
                            rows.append(p); cols.append(q2); vals.append(-Tf)
                            ct_nb = int(self.ctype[nj, ni])
                            if ct_nb in (1, 2, 3):
                                contact_links.append((ct_nb, p, q2, Tf))
                            continue

                        # Inactive neighbor. Below the band edge is a reflecting turning point.
                        # Above eps_max is the absorbing numerical boundary f(eps_max)=0.
                        eps_nb = eps[nj, ni, k]
                        if self.absorbing_top and eps_nb >= EPS_MAX and eps_k < EPS_MAX:
                            de = eps_nb - eps_k
                            if de > 0.0:
                                frac = np.clip((EPS_MAX - eps_k) / de, 1e-8, 1.0)
                                dist = frac * hcell
                                Kcur = Zk * D[j, i, k]
                                Kface = 0.5 * (Kcur + self._ZD_top)
                                Tf = dH * Kface * wface / max(dist, 1e-12 * hcell)
                                diag += Tf
                                cutoff_spatial[p] += Tf

                    # ---------------- local collision terms, integrated over spatial area
                    Aij = self.area[j, i]

                    # Optical phonons. Emission below eps=0 is forbidden; absorption beyond
                    # eps_max is retained as a loss to the absorbing numerical reservoir.
                    kd, ku = k - m, k + m
                    ed_ok = kd >= 0 and self.active[j, i, kd]
                    if ed_ok:
                        Zd = Z[j, i, kd]
                        diag += Aij * dH * c_op * Zk * (N_op + 1.0) * Zd
                        rows.append(p); cols.append(gid[j, i, kd])
                        vals.append(-Aij * dH * c_op * Zk * N_op * Zd)

                    eps_up = eps_k + E_OP_J
                    if eps_up < EPS_MAX and ku < NH and self.active[j, i, ku]:
                        Zu = Z[j, i, ku]
                        diag += Aij * dH * c_op * Zk * N_op * Zu
                        rows.append(p); cols.append(gid[j, i, ku])
                        vals.append(-Aij * dH * c_op * Zk * (N_op + 1.0) * Zu)
                    elif self.absorbing_top and eps_up >= EPS_MAX:
                        Zu = float(bands.dos(eps_up / eV))
                        ccut = Aij * dH * c_op * Zk * N_op * Zu
                        diag += ccut
                        cutoff_optical[p] += ccut

                    # Conservative acoustic FP. Lower inactive face is reflecting; upper
                    # inactive face gets a Dirichlet f=0 SG flux at the exact eps_max crossing.
                    for kk, sgn in ((k + 1, +1), (k - 1, -1)):
                        if 0 <= kk < NH and self.active[j, i, kk]:
                            ZAf = 0.5 * (Zk * Aac[j, i, k]
                                         + Z[j, i, kk] * Aac[j, i, kk]) / dH
                            if sgn > 0:
                                diag += Aij * ZAf * Bp
                                rows.append(p); cols.append(gid[j, i, kk])
                                vals.append(-Aij * ZAf * Bm)
                            else:
                                diag += Aij * ZAf * Bm
                                rows.append(p); cols.append(gid[j, i, kk])
                                vals.append(-Aij * ZAf * Bp)
                        elif sgn > 0 and self.absorbing_top:
                            # Only the upper-energy inactive neighbor is absorbing.
                            eps_nb = eps[j, i, kk] if 0 <= kk < NH else eps_k + dH
                            if eps_nb >= EPS_MAX:
                                de_b = EPS_MAX - eps_k
                                if de_b > 0.0:
                                    ZAcur = Zk * Aac[j, i, k]
                                    ZAface = 0.5 * (ZAcur + self._ZA_top)
                                    db = max(de_b, 1e-12 * dH)
                                    cb = ZAface / db
                                    Bpb = sc._bern(db / KT)
                                    ccut = Aij * cb * Bpb
                                    diag += ccut
                                    cutoff_acoustic[p] += ccut

                    # Impact ionization: one primary is removed; two electrons are deposited
                    # at eps'=(eps-Eth)/2.  Integrated source weights sum to exactly 2.
                    if self.include_impact_ionization and self.ii_rate[j, i, k] > 0.0:
                        targets = self._ii_target_weights(j, i, k)
                        if targets:
                            cii = Aij * dH * Zk * self.ii_rate[j, i, k]
                            diag += cii
                            ii_event_coeff[p] += cii
                            for kt, wt in targets:
                                pt = gid[j, i, kt]
                                rows.append(pt); cols.append(p); vals.append(-2.0 * wt * cii)

                    rows.append(p); cols.append(p); vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(self.Ndof, self.Ndof))

        # Safety only: present validated grids should have zero orphan rows.
        dvec = A.diagonal()
        orphans = np.where(np.abs(dvec) < 1e-300)[0]
        if len(orphans):
            A = A.tolil()
            for r in orphans:
                A.rows[r] = [r]
                A.data[r] = [1.0]
                b[r] = 0.0
            A = A.tocsr()

        self.n_orphans = int(len(orphans))
        self.cutoff_spatial_coeff = cutoff_spatial
        self.cutoff_acoustic_coeff = cutoff_acoustic
        self.cutoff_optical_coeff = cutoff_optical
        self.ii_event_coeff = ii_event_coeff
        self.contact_links = contact_links
        self.A, self.b = A, b
        return A, b

    # -------------------------------------------------------------------- solve
    def solve(self, tol=1e-10, method="auto"):
        """Solve the assembled SHE system.

        Large systems use a physics-split multiplicative preconditioner:

          E: same-spatial-cell energy/collision block (acoustic + optical + II),
          S: same-H real-space diffusion block (diagonal + cross-cell couplings),
          preconditioner application: E^{-1} -> S^{-1} -> E^{-1}.

        This avoids the very expensive fill of a global ILU while retaining both stiff
        directions of the operator.  Any diagonal shift is applied ONLY to the
        preconditioner, never to the solved matrix.

        method = "auto"      : direct below 130k DOF, split_ilu otherwise
                 "direct"    : sparse direct solve
                 "split_ilu" : E-S-E ILU + LGMRES
                 "legacy_ilu": old shifted global-ILU path (comparison/fallback)
        """
        import time

        A, b = self.A, self.b
        D = A.diagonal().copy()
        D[np.abs(D) < 1e-300] = 1.0
        As = (sp.diags(1.0 / D) @ A).tocsr()
        bs = b / D

        if method == "auto":
            method = "direct" if self.Ndof <= 130000 else "split_ilu"

        stats = {"method": method}

        if method == "direct":
            t0 = time.time()
            f_raw = spla.spsolve(As.tocsc(), bs)
            info = 0
            stats["solve_s"] = time.time() - t0

        elif method == "split_ilu":
            # DOFs were numbered with H fastest inside each (j,i), so same-cell
            # entries form contiguous local energy blocks.  Build a cell id for each DOF.
            cell3 = np.broadcast_to(
                np.arange(self.Ny * self.Nx, dtype=np.int32).reshape(self.Ny, self.Nx, 1),
                self.active.shape,
            )
            dof_cell = cell3[self.active]

            coo = As.tocoo(copy=False)
            same_cell = dof_cell[coo.row] == dof_cell[coo.col]
            is_diag = coo.row == coo.col

            def factor_with_shifts(M, *, drop_tol, fill_factor, permc_spec):
                last = None
                for shift in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
                    try:
                        Mt = M if shift == 0.0 else (M + shift * sp.eye(self.Ndof, format="csc"))
                        fac = spla.spilu(
                            Mt, drop_tol=drop_tol, fill_factor=fill_factor,
                            permc_spec=permc_spec, diag_pivot_thresh=0.0,
                        )
                        return fac, shift
                    except Exception as exc:
                        last = exc
                raise RuntimeError("split-ILU factorization failed") from last

            # E preconditioner: all same-spatial-cell couplings.  This contains the
            # complete acoustic/optical/II energy block plus the spatial diagonal.
            t0 = time.time()
            PE = sp.csc_matrix(
                (coo.data[same_cell], (coo.row[same_cell], coo.col[same_cell])),
                shape=As.shape,
            )
            Efac, Eshift = factor_with_shifts(
                PE, drop_tol=1e-8, fill_factor=4, permc_spec="NATURAL"
            )
            stats["energy_setup_s"] = time.time() - t0
            stats["energy_shift"] = Eshift
            stats["energy_precond_nnz"] = int(Efac.L.nnz + Efac.U.nnz)
            del PE

            # S preconditioner: diagonal + cross-cell couplings only.  With local
            # energy couplings removed this is the fixed-H real-space transport part.
            t0 = time.time()
            space_keep = is_diag | (~same_cell)
            PS = sp.csc_matrix(
                (coo.data[space_keep], (coo.row[space_keep], coo.col[space_keep])),
                shape=As.shape,
            )
            Sfac, Sshift = factor_with_shifts(
                PS, drop_tol=1e-4, fill_factor=5, permc_spec="COLAMD"
            )
            stats["space_setup_s"] = time.time() - t0
            stats["space_shift"] = Sshift
            stats["space_precond_nnz"] = int(Sfac.L.nnz + Sfac.U.nnz)
            del PS, coo, same_cell, is_diag, dof_cell, cell3

            # Multiplicative E-S-E Schwarz/line relaxation.  Recomputing the residual
            # between stages is important; a plain additive E+S preconditioner converges
            # much more slowly on refined spatial grids.
            def apply_prec(r):
                z = Efac.solve(r)
                rr = r - As.dot(z)
                z += Sfac.solve(rr)
                rr = r - As.dot(z)
                z += Efac.solve(rr)
                return z

            M = spla.LinearOperator(As.shape, matvec=apply_prec, dtype=As.dtype)
            outer_iterations = [0]

            def count_outer(_x):
                outer_iterations[0] += 1

            t0 = time.time()
            f_raw, info = spla.lgmres(
                As, bs, M=M, rtol=tol, atol=0.0, maxiter=150,
                inner_m=40, outer_k=5, callback=count_outer,
            )
            stats["solve_s"] = time.time() - t0
            stats["outer_iterations"] = int(outer_iterations[0])
            if info != 0:
                raise RuntimeError("SHE split-ILU solve failed to converge (info=%s)" % info)

        elif method == "legacy_ilu":
            # Retained only for regression comparisons.  It can be orders of magnitude
            # slower to set up on refined grids because SuperLU sees the full 3-D graph.
            f_raw = None
            info = -1
            t0 = time.time()
            for shift in (1e-3, 1e-2, 5e-2, 1e-1):
                try:
                    Ashift = (As.tocsc() + shift * sp.eye(self.Ndof, format="csc")).tocsc()
                    ilu = spla.spilu(Ashift, drop_tol=1e-3, fill_factor=10)
                except Exception:
                    continue
                M = spla.LinearOperator(As.shape, ilu.solve)
                f_raw, info = spla.lgmres(As, bs, M=M, rtol=tol, atol=0.0, maxiter=800)
                if info == 0:
                    stats["legacy_shift"] = shift
                    break
            stats["solve_s"] = time.time() - t0
            if f_raw is None or info != 0:
                raise RuntimeError("SHE legacy large-grid solve failed to converge (info=%s)" % info)

        else:
            raise ValueError("unknown SHE solve method %r" % (method,))

        self.f_raw = np.nan_to_num(f_raw)
        self.negative_min = float(np.min(self.f_raw))
        self.negative_count = int(np.count_nonzero(
            self.f_raw < -1e-12 * max(np.max(np.abs(self.f_raw)), 1.0)
        ))
        self.f = np.maximum(self.f_raw, 0.0)
        self.info = info
        scale = max(np.max(np.abs(bs)), 1.0)
        self.scaled_raw_residual_inf = float(np.max(np.abs(As.dot(self.f_raw) - bs)) / scale)
        self.scaled_residual_inf = float(np.max(np.abs(As.dot(self.f) - bs)) / scale)
        stats["raw_residual_inf"] = self.scaled_raw_residual_inf
        stats["clipped_residual_inf"] = self.scaled_residual_inf
        self.preconditioner_stats = stats
        return self.f

    # ------------------------------------------------------------ distributions
    def _f3d(self):
        F = np.zeros((self.Ny, self.Nx, self.NH))
        F[self.active] = self.f
        return F

    def particle_flux_faces(self, F=None):
        """
        Return finite-volume particle-flux density on real-space faces [1/(m^2 s)].
        gx[j,i] is positive from (j,i) -> (j,i+1); gy[j,i] positive from
        (j,i) -> (j+1,i).  Only active-active energy channels carry real-space flux;
        upper-cutoff crossings are counted separately as numerical absorption.
        """
        if F is None:
            F = self._f3d()
        ZD = self.Z * self.D

        common_x = self.active[:, :-1, :] & self.active[:, 1:, :]
        Kx = 0.5 * (ZD[:, :-1, :] + ZD[:, 1:, :])
        dfdx = (F[:, 1:, :] - F[:, :-1, :]) / self.dxe[None, :, None]
        gx_H = np.where(common_x, -Kx * dfdx, 0.0)
        gx = np.sum(gx_H, axis=2) * self.dH

        common_y = self.active[:-1, :, :] & self.active[1:, :, :]
        Ky = 0.5 * (ZD[:-1, :, :] + ZD[1:, :, :])
        dfdy = (F[1:, :, :] - F[:-1, :, :]) / self.dyn[:, None, None]
        gy_H = np.where(common_y, -Ky * dfdy, 0.0)
        gy = np.sum(gy_H, axis=2) * self.dH
        return gx, gy

    def _cell_center_flux(self, gx, gy):
        Gx = np.zeros((self.Ny, self.Nx))
        Gy = np.zeros((self.Ny, self.Nx))
        if self.Nx > 1:
            Gx[:, 0] = gx[:, 0]
            Gx[:, -1] = gx[:, -1]
        if self.Nx > 2:
            Gx[:, 1:-1] = 0.5 * (gx[:, :-1] + gx[:, 1:])
        if self.Ny > 1:
            Gy[0, :] = gy[0, :]
            Gy[-1, :] = gy[-1, :]
        if self.Ny > 2:
            Gy[1:-1, :] = 0.5 * (gy[:-1, :] + gy[1:, :])
        return Gx, Gy

    def terminal_particle_rates(self, gx=None, gy=None):
        """
        Particle rate injected FROM each contact INTO the silicon domain, per unit
        out-of-plane width [1/(m s)].  Negative means net extraction into that contact.

        When the matrix has been assembled, this is evaluated from the exact FV
        contact links used in A; this makes the global number-balance diagnostic an
        algebraic check of the solved discrete equation.
        """
        rates = {1: 0.0, 2: 0.0, 3: 0.0}
        if hasattr(self, "contact_links") and hasattr(self, "f"):
            for ct, p_int, p_ct, Tf in self.contact_links:
                # Interior row contribution is Tf*(f_int-f_contact); the opposite
                # quantity is particle injection from the contact into the domain.
                rates[ct] += Tf * (self.f[p_ct] - self.f[p_int])
            return rates

        # Fallback for pre-assembly use: integrate the reconstructed face fluxes.
        if gx is None or gy is None:
            gx, gy = self.particle_flux_faces()
        for j in range(self.Ny):
            for i in range(self.Nx - 1):
                fl = gx[j, i] * self.dyc[j]
                cl, cr = int(self.ctype[j, i]), int(self.ctype[j, i + 1])
                if cl in rates and cl != cr:
                    rates[cl] += fl
                if cr in rates and cr != cl:
                    rates[cr] -= fl
        for j in range(self.Ny - 1):
            for i in range(self.Nx):
                fl = gy[j, i] * self.dxc[i]
                ct, cb = int(self.ctype[j, i]), int(self.ctype[j + 1, i])
                if ct in rates and ct != cb:
                    rates[ct] += fl
                if cb in rates and cb != ct:
                    rates[cb] -= fl
        return rates

    def terminal_currents(self, gx=None, gy=None):
        """Conventional electron currents [A/um] from contact into device."""
        rates = self.terminal_particle_rates(gx, gy)
        return {ct: -q * rate * 1e-6 for ct, rate in rates.items()}

    # ------------------------------------------------------------------ moments
    def moments(self):
        F = self._f3d()
        Z, eps, H = self.Z, self.eps, self.H

        n = np.trapezoid(Z * F, H, axis=2)
        num = np.trapezoid(eps * Z * F, H, axis=2)
        good = n > 1e19                         # 1e13 cm^-3
        emean = np.zeros_like(n)
        emean[good] = num[good] / n[good]
        Te = np.full_like(n, T)
        Te[good] = (2.0 / 3.0) * emean[good] / kB

        Gii = np.trapezoid(Z * F * self.ii_rate, H, axis=2)

        gx, gy = self.particle_flux_faces(F)
        Gx, Gy = self._cell_center_flux(gx, gy)
        vx = np.zeros_like(n)
        vy = np.zeros_like(n)
        vx[good] = Gx[good] / n[good] * 100.0   # cm/s
        vy[good] = Gy[good] / n[good] * 100.0

        self.n = n * 1e-6
        self.Te = Te
        self.Gii = Gii * 1e-6
        self.vx, self.vy = vx, vy
        self.Gamma_x_face, self.Gamma_y_face = gx, gy
        self.F3d = F
        return self.n, self.Te, self.Gii

    def tail_fraction(self, j, i, threshold_eV=1.0, dos_weighted=True):
        """High-energy fraction at one spatial node; DOS-weighted by default."""
        if not hasattr(self, "F3d"):
            F = self._f3d()
        else:
            F = self.F3d
        mask = self.active[j, i, :]
        high = mask & (self.eps[j, i, :] >= threshold_eV * eV)
        if dos_weighted:
            w = self.Z[j, i, :] * F[j, i, :]
        else:
            w = F[j, i, :]
        den = np.sum(w[mask]) * self.dH
        num = np.sum(w[high]) * self.dH
        return float(num / (den + 1e-300))

    # -------------------------------------------------------------- diagnostics
    def conservation_diagnostics(self):
        """
        Global number balance per unit device width.  For the assembled equation,

            contact injection + II net generation - eps_max loss = 0.

        Values are particle rates [1/(m s)] except currents [A/um].
        """
        if not hasattr(self, "Gamma_x_face"):
            self.moments()
        rates = self.terminal_particle_rates(self.Gamma_x_face, self.Gamma_y_face)
        contact_in = float(sum(rates.values()))
        ii_events = float(np.dot(self.ii_event_coeff, self.f)) if self.include_impact_ionization else 0.0
        cut_sp = float(np.dot(self.cutoff_spatial_coeff, self.f))
        cut_ac = float(np.dot(self.cutoff_acoustic_coeff, self.f))
        cut_op = float(np.dot(self.cutoff_optical_coeff, self.f))
        cutoff = cut_sp + cut_ac + cut_op
        balance = contact_in + ii_events - cutoff
        gross_contact = float(sum(abs(v) for v in rates.values()))
        scale = max(gross_contact, abs(ii_events), abs(cutoff), 1e-300)
        currents = self.terminal_currents(self.Gamma_x_face, self.Gamma_y_face)
        return {
            "contact_particle_injection_per_m_s": contact_in,
            "ii_event_rate_per_m_s": ii_events,
            "cutoff_loss_per_m_s": cutoff,
            "cutoff_spatial_per_m_s": cut_sp,
            "cutoff_acoustic_per_m_s": cut_ac,
            "cutoff_optical_per_m_s": cut_op,
            "number_balance_per_m_s": balance,
            "relative_number_balance": balance / scale,
            "source_current_A_per_um": currents[1],
            "drain_current_A_per_um": currents[2],
            "body_current_A_per_um": currents[3],
            "scaled_linear_residual_inf": getattr(self, "scaled_residual_inf", np.nan),
        }


def get_phi(Nx, Ny, Vg, Vd, Phi_gate=0.30):
    """Run (and cache) a DD solve to get phi, contact map, and contact n_eq."""
    from device import DeviceParams, make_mesh, build_doping
    from poisson import Poisson2D
    from dd import DDSolver

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    # cache key MUST include Phi_gate: the same grid/bias at a different gate offset yields a
    # different phi, so omitting it would silently reuse a wrong-gate potential (audit-flagged).
    cache = os.path.join(data_dir, f"dd_{Nx}x{Ny}_Vg{Vg}_Vd{Vd}_Phg{Phi_gate}.npz")
    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=Nx, Ny=Ny)
    if os.path.exists(cache):
        d = np.load(cache)
        return x, y, d["phi"], d["ctype"], d["neq"]
    Nd, Na, Nnet = build_doping(p, X, Y)
    ps = Poisson2D(x, y, Nnet, p, Phi_gate=Phi_gate)
    dd = DDSolver(ps)
    print(f"  DD solve {Nx}x{Ny} at Vg={Vg},Vd={Vd} (cached after)...")
    phi, n, pp = dd.solve(Vg=Vg, Vd=Vd, max_gummel=90, tol=1e-5)
    np.savez(cache, phi=phi, ctype=ps.contact_type, neq=dd.n_eq)
    return x, y, phi, ps.contact_type, dd.n_eq


if __name__ == "__main__":
    import time

    Nx, Ny = 40, 34
    x, y, phi, ctype, neq = get_phi(Nx, Ny, 3.0, 3.0)
    print("phi range %.3f..%.3f V" % (phi.min(), phi.max()))
    she = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.0125,
                include_impact_ionization=True, absorbing_top=True)
    print("H-grid: %d points, dH=%.4f eV, E_op=%d nodes; active DOF=%d" %
          (she.NH, she.dH / eV, she.m_op, she.Ndof))
    t0 = time.time(); she.assemble()
    print("assembled in %.1fs, nnz=%d, orphans pinned=%d" %
          (time.time() - t0, she.A.nnz, she.n_orphans))
    t0 = time.time(); she.solve()
    print("solved in %.1fs (info=%s, scaled residual %.2e)" %
          (time.time() - t0, she.info, she.scaled_residual_inf))
    n, Te, Gii = she.moments()
    vmag = np.sqrt(she.vx ** 2 + she.vy ** 2)
    print("n range %.2e..%.2e cm^-3" % (n.min(), n.max()))
    print("Te range %.0f..%.0f K" % (Te.min(), Te.max()))
    print("Gii max %.2e /cm^3/s" % Gii.max())
    print("velocity |v| max %.2e cm/s" % vmag.max())
    print("conservation:", she.conservation_diagnostics())

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, "she2d_result.npz"),
             x=x, y=y, phi=phi, n=n, Te=Te, Gii=Gii, vx=she.vx, vy=she.vy,
             F3d=she.F3d.astype(np.float32), H=she.H,
             Gamma_x_face=she.Gamma_x_face, Gamma_y_face=she.Gamma_y_face)
    print("saved data/she2d_result.npz")
