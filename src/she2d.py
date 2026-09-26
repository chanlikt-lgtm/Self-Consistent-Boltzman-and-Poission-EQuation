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

from constants import q, eV, kB, T, hbar
import bands
import scattering as sc

KT = kB * T
EPS_MAX = 3.02 * eV
E_OP_J = sc.E_OP * eV
II_ETH_J = sc.II_ETH * eV
UM = 1e-4                         # cm / um (device helper)

# Optional JIT accelerator for the sparse finite-volume assembly.  The pure-Python
# implementation is retained as a bit-for-bit-reference fallback (set
# SHE2D_DISABLE_NUMBA=1 to force it).  Numba changes only how the same discrete
# coefficients/triplets are generated; the equations and stencils are unchanged.
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on installations without numba
    njit = None
    _NUMBA_AVAILABLE = False

_DOS_PREF = (
    bands.G_SV / (2.0 * np.pi ** 2)
    * (2.0 * bands.MSTAR) ** 1.5
    / (2.0 * hbar ** 3)
)
_DOS_ALPHA = float(bands.ALPHA)


if _NUMBA_AVAILABLE:
    @njit(cache=True, inline="always")
    def _bern_scalar_numba(x):
        if abs(x) < 1e-10:
            return 1.0 - 0.5 * x
        return x / np.expm1(x)


    @njit(cache=True, inline="always")
    def _dos_scalar_numba(eps_eV, dos_pref, dos_alpha, eV_const):
        gamma_eV = eps_eV * (1.0 + dos_alpha * eps_eV)
        gamma_J = gamma_eV * eV_const
        if gamma_J < 0.0:
            gamma_J = 0.0
        return dos_pref * np.sqrt(gamma_J) * (1.0 + 2.0 * dos_alpha * eps_eV)


    @njit(cache=True)
    def _assemble_numba_kernel(
            Ny, Nx, NH, m, dH, H0, KT_const, eV_const, eps_max, eop_J, ii_eth_J,
            N_op, c_op, dos_pref, dos_alpha, include_ii, absorbing_top, ZD_top, ZA_top,
            Z, D, Aac, eps, active, gid, ctype, n_eq, phi, ii_rate,
            dxe, dyn, dxc, dyc, area,
            rows, cols, vals, b, cutoff_spatial, cutoff_acoustic, cutoff_optical,
            ii_event_coeff, link_ct, link_pint, link_pct, link_tf):
        """Fill preallocated COO/diagnostic buffers with the original FV stencil."""
        nnz = 0
        nlinks = 0
        delta = dH / KT_const
        Bp = _bern_scalar_numba(delta)
        Bm = _bern_scalar_numba(-delta)

        for j in range(Ny):
            for i in range(Nx):
                ct = ctype[j, i]

                # Original zint() evaluated once per Ohmic spatial cell.
                Zc = 1.0
                if ct == 1 or ct == 2 or ct == 3:
                    zsum = 0.0
                    if NH > 1:
                        prev = 0.0
                        if active[j, i, 0]:
                            arg = eps[j, i, 0] / KT_const
                            if arg < 0.0:
                                arg = 0.0
                            elif arg > 200.0:
                                arg = 200.0
                            prev = Z[j, i, 0] * np.exp(-arg)
                        for kz in range(1, NH):
                            cur = 0.0
                            if active[j, i, kz]:
                                arg = eps[j, i, kz] / KT_const
                                if arg < 0.0:
                                    arg = 0.0
                                elif arg > 200.0:
                                    arg = 200.0
                                cur = Z[j, i, kz] * np.exp(-arg)
                            zsum += 0.5 * (prev + cur) * dH
                            prev = cur
                    if zsum > 0.0:
                        Zc = zsum

                for k in range(NH):
                    p = gid[j, i, k]
                    if p < 0:
                        continue

                    # Ohmic contacts: prescribed Maxwellian distribution.
                    if ct == 1 or ct == 2 or ct == 3:
                        rows[nnz] = p
                        cols[nnz] = p
                        vals[nnz] = 1.0
                        nnz += 1
                        arg = eps[j, i, k] / KT_const
                        if arg > 200.0:
                            arg = 200.0
                        b[p] = n_eq[j, i] * np.exp(-arg) / (Zc + 1e-300)
                        continue

                    diag = 0.0
                    eps_k = eps[j, i, k]
                    Zk = Z[j, i, k]
                    Dik = D[j, i, k]

                    # Spatial diffusion at fixed H.  Keep the original neighbor order:
                    # +x, -x, +y, -y, preserving COO accumulation order as well.
                    for idir in range(4):
                        nj = j
                        ni = i
                        hcell = 0.0
                        wface = 0.0
                        if idir == 0:
                            ni = i + 1
                            if ni < Nx:
                                hcell = dxe[i]
                                wface = dyc[j]
                        elif idir == 1:
                            ni = i - 1
                            if ni >= 0:
                                hcell = dxe[i - 1]
                                wface = dyc[j]
                        elif idir == 2:
                            nj = j + 1
                            if nj < Ny:
                                hcell = dyn[j]
                                wface = dxc[i]
                        else:
                            nj = j - 1
                            if nj >= 0:
                                hcell = dyn[j - 1]
                                wface = dxc[i]

                        if hcell == 0.0 or nj < 0 or nj >= Ny or ni < 0 or ni >= Nx:
                            continue
                        q2 = gid[nj, ni, k]
                        if q2 >= 0:
                            ZDf = 0.5 * (Zk * Dik + Z[nj, ni, k] * D[nj, ni, k])
                            Tf = dH * ZDf * wface / hcell
                            diag += Tf
                            rows[nnz] = p
                            cols[nnz] = q2
                            vals[nnz] = -Tf
                            nnz += 1
                            ct_nb = ctype[nj, ni]
                            if ct_nb == 1 or ct_nb == 2 or ct_nb == 3:
                                link_ct[nlinks] = ct_nb
                                link_pint[nlinks] = p
                                link_pct[nlinks] = q2
                                link_tf[nlinks] = Tf
                                nlinks += 1
                            continue

                        # Inactive neighbor: reflect below band edge, absorb above eps_max.
                        eps_nb = eps[nj, ni, k]
                        if absorbing_top and eps_nb >= eps_max and eps_k < eps_max:
                            de = eps_nb - eps_k
                            if de > 0.0:
                                frac = (eps_max - eps_k) / de
                                if frac < 1e-8:
                                    frac = 1e-8
                                elif frac > 1.0:
                                    frac = 1.0
                                dist = frac * hcell
                                Kcur = Zk * Dik
                                Kface = 0.5 * (Kcur + ZD_top)
                                dmin = 1e-12 * hcell
                                if dist < dmin:
                                    dist = dmin
                                Tf = dH * Kface * wface / dist
                                diag += Tf
                                cutoff_spatial[p] += Tf

                    Aij = area[j, i]

                    # Optical phonons.
                    kd = k - m
                    ku = k + m
                    if kd >= 0 and active[j, i, kd]:
                        Zd = Z[j, i, kd]
                        diag += Aij * dH * c_op * Zk * (N_op + 1.0) * Zd
                        rows[nnz] = p
                        cols[nnz] = gid[j, i, kd]
                        vals[nnz] = -Aij * dH * c_op * Zk * N_op * Zd
                        nnz += 1

                    eps_up = eps_k + eop_J
                    if eps_up < eps_max and ku < NH and active[j, i, ku]:
                        Zu = Z[j, i, ku]
                        diag += Aij * dH * c_op * Zk * N_op * Zu
                        rows[nnz] = p
                        cols[nnz] = gid[j, i, ku]
                        vals[nnz] = -Aij * dH * c_op * Zk * (N_op + 1.0) * Zu
                        nnz += 1
                    elif absorbing_top and eps_up >= eps_max:
                        Zu = _dos_scalar_numba(eps_up / eV_const, dos_pref, dos_alpha, eV_const)
                        ccut = Aij * dH * c_op * Zk * N_op * Zu
                        diag += ccut
                        cutoff_optical[p] += ccut

                    # Conservative acoustic FP: upper then lower, matching original order.
                    kk = k + 1
                    if kk < NH and active[j, i, kk]:
                        ZAf = 0.5 * (Zk * Aac[j, i, k] + Z[j, i, kk] * Aac[j, i, kk]) / dH
                        diag += Aij * ZAf * Bp
                        rows[nnz] = p
                        cols[nnz] = gid[j, i, kk]
                        vals[nnz] = -Aij * ZAf * Bm
                        nnz += 1
                    elif absorbing_top:
                        if kk < NH:
                            eps_nb = eps[j, i, kk]
                        else:
                            eps_nb = eps_k + dH
                        if eps_nb >= eps_max:
                            de_b = eps_max - eps_k
                            if de_b > 0.0:
                                ZAcur = Zk * Aac[j, i, k]
                                ZAface = 0.5 * (ZAcur + ZA_top)
                                db = de_b
                                min_db = 1e-12 * dH
                                if db < min_db:
                                    db = min_db
                                cb = ZAface / db
                                Bpb = _bern_scalar_numba(db / KT_const)
                                ccut = Aij * cb * Bpb
                                diag += ccut
                                cutoff_acoustic[p] += ccut

                    kk = k - 1
                    if kk >= 0 and active[j, i, kk]:
                        ZAf = 0.5 * (Zk * Aac[j, i, k] + Z[j, i, kk] * Aac[j, i, kk]) / dH
                        diag += Aij * ZAf * Bm
                        rows[nnz] = p
                        cols[nnz] = gid[j, i, kk]
                        vals[nnz] = -Aij * ZAf * Bp
                        nnz += 1

                    # Impact ionization: same equal-split interpolation and renormalization.
                    if include_ii and ii_rate[j, i, k] > 0.0:
                        eps_tgt = 0.5 * (eps_k - ii_eth_J)
                        if eps_tgt >= 0.0:
                            H_tgt = eps_tgt - phi[j, i] * eV_const
                            u = (H_tgt - H0) / dH
                            k0 = int(np.floor(u))
                            a = u - k0

                            w0 = 1.0 - a
                            w1 = a
                            ok0 = (w0 > 0.0 and k0 >= 0 and k0 < NH and active[j, i, k0])
                            k1 = k0 + 1
                            ok1 = (w1 > 0.0 and k1 >= 0 and k1 < NH and active[j, i, k1])
                            sw = 0.0
                            if ok0:
                                sw += w0
                            if ok1:
                                sw += w1

                            if sw > 0.0:
                                cii = Aij * dH * Zk * ii_rate[j, i, k]
                                diag += cii
                                ii_event_coeff[p] += cii
                                if ok0:
                                    rows[nnz] = gid[j, i, k0]
                                    cols[nnz] = p
                                    vals[nnz] = -2.0 * (w0 / sw) * cii
                                    nnz += 1
                                if ok1:
                                    rows[nnz] = gid[j, i, k1]
                                    cols[nnz] = p
                                    vals[nnz] = -2.0 * (w1 / sw) * cii
                                    nnz += 1

                    rows[nnz] = p
                    cols[nnz] = p
                    vals[nnz] = diag
                    nnz += 1

        return nnz, nlinks


class SHE2D:
    def __init__(self, x_um, y_um, phi, contact_type, n_eq, dHi_eV=0.0125,
                 include_impact_ionization=True, absorbing_top=True, phi_span_H=None):
        """
        x_um,y_um : tensor mesh [um]
        phi       : electrostatic potential [V], shape (Ny,Nx)
        contact_type : 0 none, 1 source, 2 drain, 3 body
        n_eq      : contact majority density [cm^-3]
        dHi_eV    : uniform H spacing [eV], chosen so E_op/dH is integer
        include_impact_ionization : assemble primary loss + equal-split secondary gain
        absorbing_top : impose f0=0 at eps_max through finite-volume boundary fluxes
        phi_span_H : optional (phi_lo,phi_hi) to size the H grid from a FIXED potential span
                     instead of this iteration's phi.min()/max(); freezes NH across a self-
                     consistent outer loop (span must bound the loop's whole phi range).
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

        # H-grid energy bounds. By default from THIS iteration's phi range; but in a self-
        # consistent outer loop that lets NH drift as phi evolves, re-discretizing the tail-
        # sensitive Gii moment each iteration (a per-iteration noise source). A fixed phi_span_H
        # that bounds the whole loop freezes NH so the fixed-point map is not re-discretized.
        if phi_span_H is not None:
            phi_min, phi_max = float(phi_span_H[0]), float(phi_span_H[1])
        else:
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
        # Compact spatial-cell id per active DOF.  This is the block structure used by
        # the split preconditioner; computing it from per-cell active counts avoids the
        # large broadcasted 3-D temporary previously rebuilt on every solve.
        active_counts = np.count_nonzero(self.active, axis=2).ravel()
        self.dof_cell = np.repeat(
            np.arange(self.Ny * self.Nx, dtype=np.int32), active_counts
        )

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
    def assemble(self, backend="auto"):
        """Assemble the unchanged finite-volume SHE matrix.

        ``backend="auto"`` uses a Numba-compiled triplet fill when available and
        falls back to the original Python assembler otherwise.  Set the environment
        variable ``SHE2D_DISABLE_NUMBA=1`` or pass ``backend="python"`` to force
        the reference implementation.
        """
        use_numba = (
            backend in ("auto", "numba")
            and _NUMBA_AVAILABLE
            and os.environ.get("SHE2D_DISABLE_NUMBA", "0") not in ("1", "true", "TRUE")
        )
        if backend == "numba" and not _NUMBA_AVAILABLE:
            raise RuntimeError("Numba assembly requested but numba is not installed")
        if not use_numba:
            return self._assemble_python()

        # Each active source state contributes at most 4 spatial + 2 optical +
        # 2 acoustic + 2 II + 1 diagonal triplets.  Preallocation avoids millions
        # of Python list objects while retaining the exact COO stencil.
        max_nnz = max(1, 11 * self.Ndof)
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        vals = np.empty(max_nnz, dtype=np.float64)
        b = np.zeros(self.Ndof, dtype=np.float64)
        cutoff_spatial = np.zeros(self.Ndof, dtype=np.float64)
        cutoff_acoustic = np.zeros(self.Ndof, dtype=np.float64)
        cutoff_optical = np.zeros(self.Ndof, dtype=np.float64)
        ii_event_coeff = np.zeros(self.Ndof, dtype=np.float64)

        is_ct = (self.ctype == 1) | (self.ctype == 2) | (self.ctype == 3)
        nfaces = 0
        if self.Nx > 1:
            nfaces += int(np.count_nonzero(is_ct[:, 1:] ^ is_ct[:, :-1]))
        if self.Ny > 1:
            nfaces += int(np.count_nonzero(is_ct[1:, :] ^ is_ct[:-1, :]))
        max_links = max(1, nfaces * self.NH)
        link_ct = np.empty(max_links, dtype=np.int8)
        link_pint = np.empty(max_links, dtype=np.int64)
        link_pct = np.empty(max_links, dtype=np.int64)
        link_tf = np.empty(max_links, dtype=np.float64)

        N_op = 1.0 / np.expm1(E_OP_J / KT)
        nnz, nlinks = _assemble_numba_kernel(
            self.Ny, self.Nx, self.NH, self.m_op, self.dH, float(self.H[0]),
            KT, eV, EPS_MAX, E_OP_J, II_ETH_J, N_op, sc.COP, _DOS_PREF, _DOS_ALPHA,
            self.include_impact_ionization, self.absorbing_top, self._ZD_top, self._ZA_top,
            self.Z, self.D, self.Aac, self.eps, self.active, self.gid, self.ctype,
            self.n_eq, self.phi, self.ii_rate, self.dxe, self.dyn, self.dxc, self.dyc,
            self.area, rows, cols, vals, b, cutoff_spatial, cutoff_acoustic,
            cutoff_optical, ii_event_coeff, link_ct, link_pint, link_pct, link_tf,
        )
        if nnz > max_nnz or nlinks > max_links:
            raise RuntimeError("internal SHE assembly preallocation bound exceeded")

        A = sp.csr_matrix(
            (vals[:nnz], (rows[:nnz], cols[:nnz])),
            shape=(self.Ndof, self.Ndof),
        )

        # Same orphan-row safety as the reference assembler.
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
        self.contact_links = [
            (int(link_ct[n]), int(link_pint[n]), int(link_pct[n]), float(link_tf[n]))
            for n in range(nlinks)
        ]
        self.A, self.b = A, b
        self.assembly_backend = "numba"
        self.assembly_triplets = int(nnz)
        return A, b

    # ------------------------------------------------ reference Python assembly
    def _assemble_python(self):
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
        self.assembly_backend = "python"
        self.assembly_triplets = len(vals)
        return A, b

    # -------------------------------------------------------------------- solve
    def solve(self, tol=1e-10, method="auto", x0=None,
              preconditioner_cache=None, reuse_preconditioner=True,
              preconditioner_max_age=3):
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

        ``x0`` may be either the active-DOF vector or a full (Ny,Nx,NH) distribution.
        ``preconditioner_cache`` may be the cache returned by a previous solve on the
        same frozen H-grid/active pattern. Reuse affects only the Krylov preconditioner,
        never A or b; a failed reused solve is automatically retried with fresh factors.
        """
        import time

        A, b = self.A, self.b
        D = A.diagonal().copy()
        D[np.abs(D) < 1e-300] = 1.0
        As = (sp.diags(1.0 / D) @ A).tocsr()
        bs = b / D

        if method == "auto":
            # Restored to the original certified cutoff (was temporarily 1500 in the
            # optimization drop): keeps "same algorithm, faster execution" for every
            # production mesh, all of which are >130k DOF -> split_ilu either way.
            method = "direct" if self.Ndof <= 130000 else "split_ilu"

        x0_vec = None
        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=float)
            if x0_arr.shape == self.active.shape:
                x0_vec = np.ascontiguousarray(x0_arr[self.active])
            elif x0_arr.ndim == 1 and x0_arr.size == self.Ndof:
                x0_vec = np.ascontiguousarray(x0_arr)
            else:
                raise ValueError(
                    "x0 must have shape %s or (%d,), got %s"
                    % (self.active.shape, self.Ndof, x0_arr.shape)
                )
            x0_vec = np.nan_to_num(x0_vec, copy=False)

        stats = {"method": method, "used_x0": bool(x0_vec is not None)}
        self.preconditioner_cache = None

        if method == "direct":
            t0 = time.time()
            f_raw = spla.spsolve(As.tocsc(), bs)
            info = 0
            stats["solve_s"] = time.time() - t0

        elif method == "split_ilu":
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

            def cache_is_compatible(cache):
                if not reuse_preconditioner or not isinstance(cache, dict):
                    return False
                if cache.get("shape") != As.shape:
                    return False
                if int(cache.get("age", preconditioner_max_age)) >= int(preconditioner_max_age):
                    return False
                old_H = cache.get("H")
                old_dof_cell = cache.get("dof_cell")
                if old_H is None or old_dof_cell is None:
                    return False
                return (
                    old_H.shape == self.H.shape
                    and np.array_equal(old_H, self.H)
                    and old_dof_cell.shape == self.dof_cell.shape
                    and np.array_equal(old_dof_cell, self.dof_cell)
                    and cache.get("Efac") is not None
                    and cache.get("Sfac") is not None
                )

            def build_factors():
                # DOFs are numbered with H fastest inside each (j,i).  Splitting the
                # CURRENT scaled matrix is exactly the original preconditioner setup.
                dof_cell = self.dof_cell
                coo = As.tocoo(copy=False)
                same_cell = dof_cell[coo.row] == dof_cell[coo.col]
                is_diag = coo.row == coo.col

                t0 = time.time()
                PE = sp.csc_matrix(
                    (coo.data[same_cell], (coo.row[same_cell], coo.col[same_cell])),
                    shape=As.shape,
                )
                Efac, Eshift = factor_with_shifts(
                    PE, drop_tol=1e-8, fill_factor=4, permc_spec="NATURAL"
                )
                energy_setup_s = time.time() - t0
                energy_nnz = int(Efac.L.nnz + Efac.U.nnz)
                del PE

                t0 = time.time()
                space_keep = is_diag | (~same_cell)
                PS = sp.csc_matrix(
                    (coo.data[space_keep], (coo.row[space_keep], coo.col[space_keep])),
                    shape=As.shape,
                )
                Sfac, Sshift = factor_with_shifts(
                    PS, drop_tol=1e-4, fill_factor=5, permc_spec="COLAMD"
                )
                space_setup_s = time.time() - t0
                space_nnz = int(Sfac.L.nnz + Sfac.U.nnz)
                del PS, coo, same_cell, is_diag
                meta = dict(
                    energy_setup_s=energy_setup_s, energy_shift=Eshift,
                    energy_precond_nnz=energy_nnz, space_setup_s=space_setup_s,
                    space_shift=Sshift, space_precond_nnz=space_nnz,
                )
                return Efac, Sfac, meta

            reused = cache_is_compatible(preconditioner_cache)
            if reused:
                Efac = preconditioner_cache["Efac"]
                Sfac = preconditioner_cache["Sfac"]
                meta = dict(preconditioner_cache.get("meta", {}))
                stats.update(meta)
                stats["energy_setup_s"] = 0.0
                stats["space_setup_s"] = 0.0
                cache_age = int(preconditioner_cache.get("age", 0)) + 1
            else:
                Efac, Sfac, meta = build_factors()
                stats.update(meta)
                cache_age = 0
            stats["preconditioner_reused"] = bool(reused)
            stats["preconditioner_age"] = int(cache_age)

            def run_krylov(Efac_use, Sfac_use):
                # E-S-E multiplicative relaxation, but residuals are always formed with
                # the CURRENT As. Thus cached factors are merely an approximate inverse.
                def apply_prec(r):
                    z = Efac_use.solve(r)
                    rr = r - As.dot(z)
                    z += Sfac_use.solve(rr)
                    rr = r - As.dot(z)
                    z += Efac_use.solve(rr)
                    return z

                M = spla.LinearOperator(As.shape, matvec=apply_prec, dtype=As.dtype)
                outer_iterations = [0]

                def count_outer(_x):
                    outer_iterations[0] += 1

                t0 = time.time()
                sol, inf = spla.lgmres(
                    As, bs, x0=x0_vec, M=M, rtol=tol, atol=0.0, maxiter=150,
                    inner_m=40, outer_k=5, callback=count_outer,
                )
                return sol, inf, time.time() - t0, int(outer_iterations[0])

            f_raw, info, solve_s, outer_it = run_krylov(Efac, Sfac)
            stats["solve_s"] = solve_s
            stats["outer_iterations"] = outer_it

            # Stale factors are allowed only as an acceleration.  If they fail, rebuild
            # from this exact matrix and retry, preserving the original robustness.
            if info != 0 and reused:
                Efac, Sfac, meta = build_factors()
                stats.update(meta)
                stats["preconditioner_reused"] = False
                stats["preconditioner_rebuild_after_failure"] = True
                cache_age = 0
                f_raw, info, solve_s, outer_it = run_krylov(Efac, Sfac)
                stats["solve_s"] += solve_s
                stats["outer_iterations"] += outer_it

            if info != 0:
                raise RuntimeError("SHE split-ILU solve failed to converge (info=%s)" % info)

            self.preconditioner_cache = {
                "shape": As.shape,
                "H": self.H.copy(),
                "dof_cell": self.dof_cell.copy(),
                "Efac": Efac,
                "Sfac": Sfac,
                "age": int(cache_age),
                "meta": {
                    "energy_shift": stats.get("energy_shift", 0.0),
                    "energy_precond_nnz": stats.get("energy_precond_nnz", 0),
                    "space_shift": stats.get("space_shift", 0.0),
                    "space_precond_nnz": stats.get("space_precond_nnz", 0),
                },
            }

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
                f_raw, info = spla.lgmres(As, bs, x0=x0_vec, M=M, rtol=tol, atol=0.0, maxiter=800)
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
