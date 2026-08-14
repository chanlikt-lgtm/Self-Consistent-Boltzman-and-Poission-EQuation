"""
she2d.py -- First-order SHE Boltzmann solver in 2-D real space + total energy H (P5).

Solves, for a GIVEN self-consistent potential phi(x,y) from drift-diffusion:

    grad_r . [ Z(eps) D(eps) grad_r f0|_H ]  +  Z(eps) Q[f0]  =  0,     eps = H + q*phi(r)

in the total-energy (r,H) form: spatial diffusion at fixed H, the collision operator Q
(optical in/out + acoustic Fokker-Planck) coupling H-planes at the SAME (x,y).  Driven by
Maxwellian injection at the ohmic contacts.  Energy boundaries: eps=0 reflecting (physical
band edge), eps=eps_max absorbing (numerical cutoff).

From f0 we take moments: n, average velocity (current), electron temperature Te, and the
impact-ionization generation rate G_ii.  Units: SI internally (eps in J, lengths in m).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from constants import q, eV, kB, T, ni
import bands
import scattering as sc

KT = kB * T                      # J
EPS_MAX = 3.02 * eV              # J
E_OP_J = sc.E_OP * eV            # J
UM = 1e-4                        # cm/um (device geometry helper)


class SHE2D:
    def __init__(self, x_um, y_um, phi, contact_type, n_eq, dHi_eV=0.0125):
        """
        x_um,y_um : mesh [um];  phi[Ny,Nx] [V];  contact_type[Ny,Nx] (1 src,2 drn,3 body,0 none)
        n_eq[Ny,Nx] : equilibrium majority density [cm^-3] (contact injection amplitude)
        dHi_eV : H-grid spacing [eV] (chosen so E_op/dH is integer)
        """
        self.x = np.asarray(x_um) * 1e-6      # m
        self.y = np.asarray(y_um) * 1e-6      # m
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.phi = phi                        # V
        self.ctype = contact_type
        self.n_eq = n_eq * 1e6                # cm^-3 -> m^-3

        # ---- H grid covering the a-priori bias envelope ----
        self.dH = dHi_eV * eV
        self.m_op = int(round(E_OP_J / self.dH))     # optical jump in nodes (integer)
        phi_min, phi_max = phi.min(), phi.max()
        H_lo = (-phi_max) * eV - 2 * self.dH
        H_hi = (EPS_MAX + (-phi_min) * eV) + 2 * self.dH
        self.H = np.arange(H_lo, H_hi + self.dH, self.dH)   # J
        self.NH = len(self.H)

        self._geom()
        self._grid()

    def _geom(self):
        x, y = self.x, self.y
        dxc = np.zeros(self.Nx); dyc = np.zeros(self.Ny)
        dxc[1:-1] = 0.5 * (x[2:] - x[:-2]); dxc[0] = 0.5*(x[1]-x[0]); dxc[-1] = 0.5*(x[-1]-x[-2])
        dyc[1:-1] = 0.5 * (y[2:] - y[:-2]); dyc[0] = 0.5*(y[1]-y[0]); dyc[-1] = 0.5*(y[-1]-y[-2])
        self.dxc, self.dyc = dxc, dyc
        self.dxe = np.diff(x); self.dyn = np.diff(y)
        self.area = np.outer(dyc, dxc)        # cell area A_ij [m^2]

    def _grid(self):
        """Precompute eps, active mask, DOS Z, diffusion D, acoustic A on the 3-D grid."""
        # eps[j,i,k] = H[k] + q*phi[j,i]  (J);  q*phi in J = phi[V]*eV
        self.eps = self.H[None, None, :] + (self.phi[:, :, None] * eV)   # (Ny,Nx,NH) J
        self.active = (self.eps >= 0.0) & (self.eps <= EPS_MAX)
        epsE = np.clip(self.eps / eV, 1e-6, 3.02)      # eV for band funcs
        self.Z = np.where(self.active, bands.dos(epsE), 0.0)            # 1/(J m^3)
        self.D = np.where(self.active, sc.Dcoef(epsE), 0.0)            # m^2/s
        self.Aac = np.where(self.active, sc.acoustic_energy_coeff(epsE), 0.0)  # J^2/s
        # global index map for active nodes
        self.gid = -np.ones((self.Ny, self.Nx, self.NH), dtype=np.int64)
        self.gid[self.active] = np.arange(np.count_nonzero(self.active))
        self.Ndof = int(np.count_nonzero(self.active))
        # contact injection normalization Zint = int Z e^{-eps/kT} deps (per contact phi)
        self.Zint_grid = None

    # ------------------------------------------------------------------ assembly
    def assemble(self):
        Ny, Nx, NH, m = self.Ny, self.Nx, self.NH, self.m_op
        H, dH = self.H, self.dH
        Z, D, Aac = self.Z, self.D, self.Aac
        eps = self.eps
        N_op = 1.0 / np.expm1(E_OP_J / KT)          # Bose
        c_op = sc.COP
        gid = self.gid
        rows, cols, vals = [], [], []
        b = np.zeros(self.Ndof)

        # contact Maxwellian: f0 = n_c * exp(-eps/kT) / Zint(phi_contact)
        # Zint depends only on phi at the contact node; precompute per spatial node lazily
        def zint(j, i):
            ee = eps[j, i, :]
            act = self.active[j, i, :]
            if not act.any():
                return 1.0
            return np.trapezoid(np.where(act, Z[j, i, :] * np.exp(-np.clip(ee/KT, 0, 200)), 0.0), H)

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
                    # ---- contact Dirichlet (Maxwellian injection) ----
                    if ct in (1, 2, 3):
                        if Zc is None:
                            Zc = zint(j, i)
                        rows.append(p); cols.append(p); vals.append(1.0)
                        b[p] = self.n_eq[j, i] * np.exp(-min(eps[j, i, k]/KT, 200.0)) / (Zc + 1e-300)
                        continue

                    diag = 0.0
                    # ---- spatial diffusion at fixed H_k (box-integrated * dH) ----
                    for (nj, ni, hcell, wface) in (
                        (j, i+1, self.dxe[i] if i+1 < Nx else 0, self.dyc[j]),
                        (j, i-1, self.dxe[i-1] if i-1 >= 0 else 0, self.dyc[j]),
                        (j+1, i, self.dyn[j] if j+1 < Ny else 0, self.dxc[i]),
                        (j-1, i, self.dyn[j-1] if j-1 >= 0 else 0, self.dxc[i]),
                    ):
                        if hcell == 0 or nj < 0 or nj >= Ny or ni < 0 or ni >= Nx:
                            continue
                        q2 = gid[nj, ni, k]
                        if q2 < 0:
                            continue                     # inactive neighbor -> reflecting no-flux
                        ZDf = 0.5 * (Z[j, i, k]*D[j, i, k] + Z[nj, ni, k]*D[nj, ni, k])
                        Tf = dH * ZDf * wface / hcell     # * dH (H-box)
                        diag += Tf
                        rows.append(p); cols.append(q2); vals.append(-Tf)

                    # ---- collision (fixed i,j), box-integrated over area A_ij ----
                    Aij = self.area[j, i]
                    Zk = Z[j, i, k]
                    # optical: emission k->k-m (down), absorption k->k+m (up)
                    kd, ku = k - m, k + m
                    ed_ok = kd >= 0 and self.active[j, i, kd]
                    eu_ok = ku < NH and self.active[j, i, ku]
                    # emission out (down): drop if target below band edge (reflecting)
                    if ed_ok:
                        Zd = Z[j, i, kd]
                        diag += Aij * dH * c_op * Zk * (N_op + 1) * Zd     # out (emission)
                        rows.append(p); cols.append(gid[j,i,kd]); vals.append(-Aij*dH*c_op*Zk*N_op*Zd)  # in via absorption from kd? -> handled at kd
                    # absorption out (up): if target above eps_max -> absorbing loss (keep, no in)
                    if ku < NH:
                        Zu = Z[j, i, ku] if eu_ok else bands.dos(min(EPS_MAX/eV,3.02))
                        diag += Aij * dH * c_op * Zk * N_op * Zu            # out (absorption)
                        if eu_ok:
                            rows.append(p); cols.append(gid[j,i,ku]); vals.append(-Aij*dH*c_op*Zk*(N_op+1)*Zu)  # in via emission from ku
                    # ---- acoustic Fokker-Planck in H (adjacent nodes) ----
                    for (kk, sgn) in ((k+1, +1), (k-1, -1)):
                        if kk < 0 or kk >= NH or not self.active[j, i, kk]:
                            continue
                        ZAf = 0.5 * (Zk*Aac[j, i, k] + Z[j, i, kk]*Aac[j, i, kk]) / dH
                        # SG flux coefficients (drift toward low energy)
                        if sgn > 0:   # face to k+1
                            diag += Aij * ZAf * Bp
                            rows.append(p); cols.append(gid[j,i,kk]); vals.append(-Aij*ZAf*Bm)
                        else:         # face to k-1
                            diag += Aij * ZAf * Bm
                            rows.append(p); cols.append(gid[j,i,kk]); vals.append(-Aij*ZAf*Bp)

                    rows.append(p); cols.append(p); vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(self.Ndof, self.Ndof))
        # Pin "orphan" active nodes (no active spatial neighbor AND no active collision
        # partner -> zero-diagonal, all-zero row) to f0=0, else the matrix is singular.
        dvec = A.diagonal()
        orphans = np.where(np.abs(dvec) < 1e-300)[0]
        if len(orphans):
            A = A.tolil()
            for r in orphans:
                A.rows[r] = [r]; A.data[r] = [1.0]; b[r] = 0.0
            A = A.tocsr()
        self.n_orphans = int(len(orphans))
        self.A, self.b = A, b
        return A, b

    def solve(self, tol=1e-8):
        A, b = self.A, self.b
        # Jacobi row-scaling: contact rows have diag=1 while interior rows carry SI-scale
        # (~1e23) coefficients -> condition number ~1e23 wrecks the direct solve. Normalize
        # each row by its diagonal so all diagonals are 1 before solving.
        D = A.diagonal().copy()
        D[np.abs(D) < 1e-300] = 1.0
        As = (sp.diags(1.0 / D) @ A).tocsc()
        bs = b / D
        try:
            f = spla.spsolve(As, bs)
            info = 0
        except Exception as e:
            print("  spsolve failed (%s); trying lgmres+ILU" % e)
            ilu = spla.spilu(As, drop_tol=1e-4, fill_factor=15)
            M = spla.LinearOperator(As.shape, ilu.solve)
            f, info = spla.lgmres(As, bs, M=M, rtol=tol, maxiter=500)
        self.f = np.nan_to_num(np.maximum(f, 0.0))
        self.info = info
        return self.f

    # ------------------------------------------------------------------- moments
    def _f3d(self):
        F = np.zeros((self.Ny, self.Nx, self.NH))
        F[self.active] = self.f
        return F

    def moments(self):
        F = self._f3d()
        Z, eps, H = self.Z, self.eps, self.H
        # n = int Z f0 dH   (dH is the energy measure since eps = H + qphi at fixed r)
        n = np.trapezoid(Z * F, H, axis=2)                       # 1/m^3
        emean = np.zeros_like(n)
        num = np.trapezoid(eps * Z * F, H, axis=2)
        good = n > 1e6
        emean[good] = num[good] / n[good]
        Te = (2.0/3.0) * emean / kB                              # K
        Te[~good] = T
        # impact ionization generation G_ii = int Z f0 / tau_ii deps  (eps>eps_th)
        eth = 1.1 * eV
        P_ii = 2.0e13   # Keldysh prefactor [1/s] (surrogate; see report)
        tii_inv = np.where(eps > eth, P_ii * ((eps - eth)/eth)**2, 0.0)
        Gii = np.trapezoid(Z * F * tii_inv, H, axis=2)           # 1/(m^3 s)
        self.n = n * 1e-6                                        # cm^-3
        self.Te = Te
        self.Gii = Gii * 1e-6                                    # 1/(cm^3 s)
        self.F3d = F
        return self.n, self.Te, self.Gii


def get_phi(Nx, Ny, Vg, Vd, Phi_gate=0.30):
    """Run (and cache) a DD solve to get phi, contact map, n_eq on the mesh."""
    from device import DeviceParams, make_mesh, build_doping
    from poisson import Poisson2D
    from dd import DDSolver
    cache = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"dd_{Nx}x{Ny}_Vg{Vg}_Vd{Vd}.npz")
    p = DeviceParams(); x, y, X, Y = make_mesh(p, Nx=Nx, Ny=Ny)
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
    Nx, Ny = 32, 26
    x, y, phi, ctype, neq = get_phi(Nx, Ny, 3.0, 3.0)
    print("phi range %.3f..%.3f V" % (phi.min(), phi.max()))
    she = SHE2D(x, y, phi, ctype, neq, dHi_eV=0.025)
    print("H-grid: %d points, dH=%.4f eV, E_op=%d nodes; active DOF=%d" %
          (she.NH, she.dH/eV, she.m_op, she.Ndof))
    t0 = time.time(); she.assemble()
    print("assembled in %.1fs, nnz=%d, orphans pinned=%d" % (time.time()-t0, she.A.nnz, she.n_orphans))
    t0 = time.time(); she.solve(); print("solved in %.1fs (info=%s)" % (time.time()-t0, she.info))
    n, Te, Gii = she.moments()
    print("n range %.2e..%.2e cm^-3" % (n.min(), n.max()))
    print("Te range %.0f..%.0f K (max near drain?)" % (Te.min(), Te.max()))
    print("Gii max %.2e /cm^3/s" % Gii.max())
    np.savez(os.path.join(os.path.dirname(__file__), "..", "data", "she2d_result.npz"),
             x=x, y=y, phi=phi, n=n, Te=Te, Gii=Gii)
    print("saved data/she2d_result.npz")
