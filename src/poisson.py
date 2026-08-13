"""
poisson.py -- 2-D nonlinear Poisson solver (box-integration finite volume,
Newton iteration) with MOS boundary conditions.

Potential reference: intrinsic silicon has phi = 0, so at equilibrium
    n = ni * exp(+(phi - phi_n)/Vt),   p = ni * exp(-(phi - phi_p)/Vt),
with phi_n, phi_p the electron/hole quasi-Fermi *potentials* [V].

Boundary conditions:
  * Gate    : Robin through the oxide capacitance -> metal at (Vg - Phi_gate).
  * Source  : ohmic Dirichlet on the n+ top surface, phi = Vs + phi_eq(Nnet).
  * Drain   : ohmic Dirichlet on the n+ top surface, phi = Vd + phi_eq(Nnet).
  * Body    : ohmic Dirichlet on the bottom, phi = Vb + phi_eq(Nnet).
  * Elsewhere (edges, surface spacers): homogeneous Neumann (natural).

Geometry is a nonuniform tensor mesh, arrays indexed [j, i] = [depth, lateral],
flattened as idx = j*Nx + i.  All lengths converted to cm.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from constants import q, eps0, Vt, ni, eps_si, eps_ox, asinh


UM = 1e-4  # cm per micrometre


class Poisson2D:
    def __init__(self, x_um, y_um, Nnet, dev_params,
                 Phi_gate=0.30, eps_si=eps_si, eps_ox=eps_ox):
        """
        x_um, y_um : 1-D mesh coordinates [um]
        Nnet       : net doping [cm^-3], shape (Ny, Nx), >0 n-type
        dev_params : DeviceParams instance (for contact geometry)
        Phi_gate   : gate work-function/flatband offset [V]; metal node = Vg - Phi_gate.
                     Calibrated (device doping is reconstructed) to give a sensible V_th.
        """
        self.x = np.asarray(x_um) * UM        # cm
        self.y = np.asarray(y_um) * UM        # cm
        self.Nx = len(self.x)
        self.Ny = len(self.y)
        self.N = self.Nx * self.Ny
        self.Nnet = Nnet
        self.p = dev_params
        self.eps_si = eps_si
        self.eps_ox = eps_ox
        self.Phi_gate = Phi_gate
        self.tox = dev_params.tox * UM        # cm

        self._build_geometry()
        self._build_masks()
        self._build_laplacian()

    # ----------------------------------------------------------------- geometry
    def _build_geometry(self):
        x, y = self.x, self.y
        Nx, Ny = self.Nx, self.Ny

        # control-volume half-widths (nonuniform tensor mesh)
        dxc = np.zeros(Nx)
        dxc[1:-1] = 0.5 * (x[2:] - x[:-2])
        dxc[0] = 0.5 * (x[1] - x[0])
        dxc[-1] = 0.5 * (x[-1] - x[-2])
        dyc = np.zeros(Ny)
        dyc[1:-1] = 0.5 * (y[2:] - y[:-2])
        dyc[0] = 0.5 * (y[1] - y[0])
        dyc[-1] = 0.5 * (y[-1] - y[-2])
        self.dxc, self.dyc = dxc, dyc

        # face spacings
        self.dx_e = np.diff(x)   # length Nx-1, between i and i+1
        self.dy_n = np.diff(y)   # length Ny-1, between j and j+1

        # control volume per node (per unit z-width): dxc[i]*dyc[j]
        self.vol = np.outer(dyc, dxc)   # (Ny,Nx)

    def _idx(self, j, i):
        return j * self.Nx + i

    # -------------------------------------------------------------------- masks
    def _build_masks(self):
        x_um = self.x / UM
        p = self.p
        Nx, Ny = self.Nx, self.Ny

        self.dirichlet = np.zeros((Ny, Nx), dtype=bool)
        self.phi_bc = np.zeros((Ny, Nx))     # applied bias part (Vs/Vd/Vb) filled at solve
        self.contact_type = np.zeros((Ny, Nx), dtype=int)  # 1 src, 2 drn, 3 body

        # source contact: top surface over n+ (x < x_srcnplus)
        src = (x_um <= p.x_srcnplus)
        self.dirichlet[0, src] = True
        self.contact_type[0, src] = 1
        # drain contact: top surface over n+ (x > x_drnnplus)
        drn = (x_um >= p.x_drnnplus)
        self.dirichlet[0, drn] = True
        self.contact_type[0, drn] = 2
        # body contact: entire bottom row
        self.dirichlet[Ny - 1, :] = True
        self.contact_type[Ny - 1, :] = 3

        # gate surface nodes (Robin), only where NOT a dirichlet contact
        self.gate_mask = np.zeros((Ny, Nx), dtype=bool)
        gate = (x_um >= p.x_gate1) & (x_um <= p.x_gate2)
        self.gate_mask[0, gate & (~self.dirichlet[0])] = True

        # equilibrium contact potential from local net doping (charge neutrality)
        self.phi_eq = Vt * asinh(self.Nnet / (2.0 * ni))

    # --------------------------------------------------------------- laplacian
    def _build_laplacian(self):
        """Assemble the geometric transmissibility Laplacian L (sparse) plus the
        gate Robin diagonal and gate source vector b_gate (per unit Vg-Phi_gate)."""
        Nx, Ny = self.Nx, self.Ny
        rows, cols, vals = [], [], []
        diag = np.zeros(self.N)

        e0si = self.eps_si * eps0
        for j in range(Ny):
            for i in range(Nx):
                k = self._idx(j, i)
                # East (i+1)
                if i + 1 < Nx:
                    T = e0si * self.dyc[j] / self.dx_e[i]
                    rows.append(k); cols.append(self._idx(j, i + 1)); vals.append(-T)
                    diag[k] += T
                # West (i-1)
                if i - 1 >= 0:
                    T = e0si * self.dyc[j] / self.dx_e[i - 1]
                    rows.append(k); cols.append(self._idx(j, i - 1)); vals.append(-T)
                    diag[k] += T
                # North / South in depth (j+1, j-1)
                if j + 1 < Ny:
                    T = e0si * self.dxc[i] / self.dy_n[j]
                    rows.append(k); cols.append(self._idx(j + 1, i)); vals.append(-T)
                    diag[k] += T
                if j - 1 >= 0:
                    T = e0si * self.dxc[i] / self.dy_n[j - 1]
                    rows.append(k); cols.append(self._idx(j - 1, i)); vals.append(-T)
                    diag[k] += T

        # gate Robin: oxide transmissibility to metal node
        self.b_gate = np.zeros(self.N)    # multiply by (Vg - Phi_gate)
        e0ox = self.eps_ox * eps0
        for i in range(Nx):
            if self.gate_mask[0, i]:
                k = self._idx(0, i)
                Tox = e0ox * self.dxc[i] / self.tox
                diag[k] += Tox
                self.b_gate[k] += Tox    # * (Vg - Phi_gate)

        rows.extend(range(self.N)); cols.extend(range(self.N)); vals.extend(diag)
        self.L = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self.diag_geom = diag.copy()

    # ------------------------------------------------------------------- solve
    def _carriers(self, phi, phi_n, phi_p):
        n = ni * np.exp(np.clip((phi - phi_n) / Vt, -80, 80))
        pp = ni * np.exp(np.clip((phi_p - phi) / Vt, -80, 80))
        return n, pp

    def solve(self, phi_init=None, phi_n=None, phi_p=None,
              Vs=0.0, Vd=0.0, Vg=0.0, Vb=0.0,
              max_newton=60, tol=1e-8, damp_clip=0.5, verbose=False):
        """Newton solve of the nonlinear Poisson equation. Returns phi (Ny,Nx)."""
        Nx, Ny, N = self.Nx, self.Ny, self.N
        Nnet_flat = self.Nnet.ravel()
        vol_flat = self.vol.ravel()

        if phi_n is None:
            phi_n = np.zeros((Ny, Nx))
        if phi_p is None:
            phi_p = np.zeros((Ny, Nx))
        phi_n_f = phi_n.ravel(); phi_p_f = phi_p.ravel()

        # Dirichlet target potentials
        phi_bc = self.phi_eq.copy()
        phi_bc[self.contact_type == 1] += Vs
        phi_bc[self.contact_type == 2] += Vd
        phi_bc[self.contact_type == 3] += Vb
        dir_flat = self.dirichlet.ravel()
        phi_bc_flat = phi_bc.ravel()

        # initial guess
        if phi_init is None:
            phi = self.phi_eq.copy().ravel()
        else:
            phi = phi_init.copy().ravel()
        phi[dir_flat] = phi_bc_flat[dir_flat]

        b_gate = self.b_gate * (Vg - self.Phi_gate)
        L = self.L

        for it in range(max_newton):
            n, pp = self._carriers(phi.reshape(Ny, Nx), phi_n, phi_p)
            n = n.ravel(); pp = pp.ravel()

            # residual F = L phi - b_gate - q(p - n + Nnet)*vol
            charge = q * (pp - n + Nnet_flat) * vol_flat
            F = L.dot(phi) - b_gate - charge

            # Jacobian diagonal add: d/dphi[-q(p-n)vol] = q(p+n)/Vt * vol
            jac_diag = q * (pp + n) / Vt * vol_flat
            J = L + sp.diags(jac_diag)

            # apply Dirichlet: rows -> identity, residual -> phi - phi_bc
            F[dir_flat] = phi[dir_flat] - phi_bc_flat[dir_flat]
            J = J.tolil()
            for k in np.where(dir_flat)[0]:
                J.rows[k] = [k]
                J.data[k] = [1.0]
            J = J.tocsr()

            dphi = spla.spsolve(J, -F)
            # damp / clip the update for robustness
            dphi = np.clip(dphi, -damp_clip, damp_clip)
            phi += dphi

            res = np.max(np.abs(dphi))
            if verbose:
                print(f"    newton {it:2d}  max|dphi|={res:.3e}")
            if res < tol:
                break

        self.phi = phi.reshape(Ny, Nx)
        self.n = (ni * np.exp(np.clip((self.phi - phi_n) / Vt, -80, 80)))
        self.p_h = (ni * np.exp(np.clip((phi_p - self.phi) / Vt, -80, 80)))
        return self.phi


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from device import DeviceParams, make_mesh, build_doping

    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=90, Ny=80)
    Nd, Na, Nnet = build_doping(p, X, Y)

    ps = Poisson2D(x, y, Nnet, p, Phi_gate=0.30)
    print("Solving equilibrium Poisson (Vg=Vd=0)...")
    phi = ps.solve(Vg=0.0, Vd=0.0, verbose=True)
    print("phi range: %.3f .. %.3f V" % (phi.min(), phi.max()))
    print("n range: %.2e .. %.2e" % (ps.n.min(), ps.n.max()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]
    pcm = ax.pcolormesh(x, y, phi, cmap="turbo", shading="auto")
    ax.invert_yaxis(); ax.set_title("equilibrium potential phi [V]")
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]"); fig.colorbar(pcm, ax=ax)

    ax = axes[1]
    pcm = ax.pcolormesh(x, y, np.log10(ps.n), cmap="viridis", shading="auto", vmin=6, vmax=20)
    ax.invert_yaxis(); ax.set_title("log10 n  [cm^-3]")
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]"); fig.colorbar(pcm, ax=ax)

    ax = axes[2]
    pcm = ax.pcolormesh(x, y, np.log10(ps.p_h), cmap="magma", shading="auto", vmin=6, vmax=20)
    ax.invert_yaxis(); ax.set_title("log10 p  [cm^-3]")
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]"); fig.colorbar(pcm, ax=ax)

    fig.tight_layout()
    fig.savefig("figures/poisson_equilibrium.png", dpi=130)
    print("wrote figures/poisson_equilibrium.png")
