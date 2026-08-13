"""
dd.py -- Drift-diffusion device solver (decoupled Gummel map) built on Poisson2D.

Provides:
  * Scharfetter-Gummel electron & hole continuity assembly on the tensor mesh,
  * doping- and field-dependent mobility (Caughey-Thomas),
  * SRH recombination (semi-implicit),
  * a Gummel outer loop that returns self-consistent phi, n, p at a bias point,
  * terminal drain current extraction for I-V curves.

This serves two roles in the reproduction:
  (1) the drift-diffusion INITIAL GUESS the paper's Fig. 1 flowchart calls for, and
  (2) a self-consistent phi(x,y) operating point to drive the SHE-BTE solver.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from constants import q, Vt, ni
from poisson import Poisson2D, UM


# ----------------------------------------------------------------- Bernoulli
def bern(x):
    """Bernoulli function B(x) = x/(exp(x)-1) = x/expm1(x), numerically robust.
    Only large POSITIVE x risks overflow; large negative x is fine (expm1->-1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-10
    big = x > 40.0
    ok = ~(small | big)
    out[small] = 1.0 - x[small] / 2.0
    out[big] = x[big] * np.exp(-x[big])       # -> 0
    out[ok] = x[ok] / np.expm1(x[ok])
    return out


# --------------------------------------------------------------- mobility
def mobility_lowfield(Nabs, carrier="n"):
    """Caughey-Thomas low-field mobility vs total doping [cm^2/Vs]."""
    if carrier == "n":
        mu_min, mu_max, Nref, a = 68.5, 1414.0, 9.2e16, 0.711
    else:
        mu_min, mu_max, Nref, a = 44.9, 470.5, 2.23e17, 0.719
    return mu_min + (mu_max - mu_min) / (1.0 + (Nabs / Nref) ** a)


def mobility_field(mu_lf, Efield, carrier="n"):
    """Caughey-Thomas field-dependent mobility. Efield in V/cm."""
    if carrier == "n":
        vsat, beta = 1.07e7, 2.0
    else:
        vsat, beta = 8.37e6, 1.0
    x = (mu_lf * np.abs(Efield) / vsat)
    return mu_lf / (1.0 + x ** beta) ** (1.0 / beta)


class DDSolver:
    def __init__(self, poisson: Poisson2D, tau_n=1e-7, tau_p=1e-7):
        self.ps = poisson
        self.Nx, self.Ny, self.N = poisson.Nx, poisson.Ny, poisson.N
        self.tau_n = tau_n
        self.tau_p = tau_p

        Nabs = np.abs(poisson.Nnet)
        self.mun_lf = mobility_lowfield(Nabs, "n")
        self.mup_lf = mobility_lowfield(Nabs, "p")

        # equilibrium contact carrier densities (charge-neutral ohmic)
        Nnet = poisson.Nnet
        self.n_eq = 0.5 * (Nnet + np.sqrt(Nnet ** 2 + 4 * ni ** 2))
        self.p_eq = ni ** 2 / self.n_eq

        self.dirichlet = poisson.dirichlet
        self._geom()

    def _geom(self):
        ps = self.ps
        self.x, self.y = ps.x, ps.y
        self.dxc, self.dyc = ps.dxc, ps.dyc
        self.dx_e, self.dy_n = ps.dx_e, ps.dy_n
        self.vol = ps.vol

    def _idx(self, j, i):
        return j * self.Nx + i

    def _face_mobilities(self, phi, carrier):
        """Field-dependent mobility on E-faces (x) and N-faces (y)."""
        Nx, Ny = self.Nx, self.Ny
        mu_lf = self.mun_lf if carrier == "n" else self.mup_lf
        # x-faces between i,i+1
        Ex = np.zeros((Ny, Nx - 1))
        for i in range(Nx - 1):
            Ex[:, i] = np.abs(phi[:, i + 1] - phi[:, i]) / self.dx_e[i]
        mu_lf_x = 0.5 * (mu_lf[:, :-1] + mu_lf[:, 1:])
        mux = mobility_field(mu_lf_x, Ex, carrier)
        # y-faces between j,j+1
        Ey = np.zeros((Ny - 1, Nx))
        for j in range(Ny - 1):
            Ey[j, :] = np.abs(phi[j + 1, :] - phi[j, :]) / self.dy_n[j]
        mu_lf_y = 0.5 * (mu_lf[:-1, :] + mu_lf[1:, :])
        muy = mobility_field(mu_lf_y, Ey, carrier)
        return mux, muy

    def assemble_continuity(self, phi, n, p, carrier):
        """Return sparse A and rhs b for the SG continuity of `carrier` given phi."""
        Nx, Ny, N = self.Nx, self.Ny, self.N
        mux, muy = self._face_mobilities(phi, carrier)
        rows, cols, vals = [], [], []
        diag = np.zeros(N)
        b = np.zeros(N)

        # SRH recomb denominator (semi-implicit), using current n,p
        Dsrh = self.tau_p * (n + ni) + self.tau_n * (p + ni)
        vol = self.vol

        sign = 1.0  # electrons; holes handled by swapped Bernoulli args below
        for j in range(Ny):
            for i in range(Nx):
                k = self._idx(j, i)
                if self.dirichlet[j, i]:
                    continue
                # accumulate face contributions
                # neighbor list: (jj,ii, mu_face, h, Af, dphi = phi_nb - phi_here)
                faces = []
                if i + 1 < Nx:
                    faces.append((j, i + 1, mux[j, i], self.dx_e[i], self.dyc[j]))
                if i - 1 >= 0:
                    faces.append((j, i - 1, mux[j, i - 1], self.dx_e[i - 1], self.dyc[j]))
                if j + 1 < Ny:
                    faces.append((j + 1, i, muy[j, i], self.dy_n[j], self.dxc[i]))
                if j - 1 >= 0:
                    faces.append((j - 1, i, muy[j - 1, i], self.dy_n[j - 1], self.dxc[i]))

                for (jj, ii, mu, h, Af) in faces:
                    C = q * mu * Vt * Af / h
                    dphi = phi[jj, ii] - phi[j, i]
                    D = dphi / Vt
                    if carrier == "n":
                        # flux a->b = C[B(D) n_b - B(-D) n_a]
                        diag[k] += -C * bern(-D)
                        rows.append(k); cols.append(self._idx(jj, ii)); vals.append(C * bern(D))
                    else:
                        # holes: flux a->b uses swapped signs
                        diag[k] += -C * bern(D)
                        rows.append(k); cols.append(self._idx(jj, ii)); vals.append(C * bern(-D))

                # SRH: q R V ; R = (n p - ni^2)/Dsrh
                V = vol[j, i]
                if carrier == "n":
                    diag[k] += -q * V * p[j, i] / Dsrh[j, i]      # move n-term to LHS (note continuity sign)
                    b[k] += -q * V * ni ** 2 / Dsrh[j, i]
                else:
                    diag[k] += -q * V * n[j, i] / Dsrh[j, i]
                    b[k] += -q * V * ni ** 2 / Dsrh[j, i]

        # Dirichlet rows
        for j in range(Ny):
            for i in range(Nx):
                k = self._idx(j, i)
                if self.dirichlet[j, i]:
                    rows.append(k); cols.append(k); vals.append(1.0)
                    b[k] = (self.n_eq[j, i] if carrier == "n" else self.p_eq[j, i])

        rows.extend(range(N)); cols.extend(range(N)); vals.extend(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return A, b

    def solve(self, Vs=0.0, Vd=0.0, Vg=0.0, Vb=0.0,
              max_gummel=80, tol=1e-5, verbose=False):
        ps = self.ps
        Ny, Nx = self.Ny, self.Nx

        # start from equilibrium Poisson
        phi = ps.solve(Vs=Vs, Vd=Vd, Vg=Vg, Vb=Vb)
        n = np.maximum(ps.n, 1.0)
        p = np.maximum(ps.p_h, 1.0)

        phi_prev = phi.copy()
        for git in range(max_gummel):
            # electron continuity
            A, b = self.assemble_continuity(phi, n, p, "n")
            n = spla.spsolve(A, b).reshape(Ny, Nx)
            n = np.maximum(n, 1e-3)
            # hole continuity
            A, b = self.assemble_continuity(phi, n, p, "p")
            p = spla.spsolve(A, b).reshape(Ny, Nx)
            p = np.maximum(p, 1e-3)

            # quasi-Fermi potentials for Poisson update
            phi_n = phi - Vt * np.log(n / ni)
            phi_p = phi + Vt * np.log(p / ni)
            phi = ps.solve(phi_init=phi, phi_n=phi_n, phi_p=phi_p,
                           Vs=Vs, Vd=Vd, Vg=Vg, Vb=Vb)

            dphi = np.max(np.abs(phi - phi_prev))
            phi_prev = phi.copy()
            if verbose:
                print(f"  gummel {git:2d}  max|dphi|={dphi:.3e}")
            if dphi < tol:
                break

        self.phi, self.n, self.p = phi, n, p
        return phi, n, p

    def drain_current(self):
        """Terminal electron current at the drain contact [A/um width]. Sum SG
        electron fluxes across faces entering drain-contact nodes."""
        ps = self.ps
        phi, n = self.phi, self.n
        mux, muy = self._face_mobilities(phi, "n")
        Nx, Ny = self.Nx, self.Ny
        Id = 0.0
        # drain contact nodes are surface (j=0) with contact_type==2
        drain_nodes = np.argwhere(ps.contact_type == 2)
        for (j, i) in drain_nodes:
            # sum fluxes from interior neighbors INTO this contact node
            faces = []
            if i + 1 < Nx: faces.append((j, i + 1, mux[j, i], ps.dx_e[i], ps.dyc[j]))
            if i - 1 >= 0: faces.append((j, i - 1, mux[j, i - 1], ps.dx_e[i - 1], ps.dyc[j]))
            if j + 1 < Ny: faces.append((j + 1, i, muy[j, i], ps.dy_n[j], ps.dxc[i]))
            for (jj, ii, mu, h, Af) in faces:
                C = q * mu * Vt * Af / h
                D = (phi[jj, ii] - phi[j, i]) / Vt
                # electron particle flux a->b; current into contact = -(flux out)
                flux = C * (bern(D) * n[jj, ii] - bern(-D) * n[j, i])  # A * (from a to b)
                Id += flux
        # fluxes summed in A per cm of z-width. Convert to A per um width (*1e-4 = *UM).
        return Id * UM  # A / um width


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from device import DeviceParams, make_mesh, build_doping

    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=80, Ny=70)
    Nd, Na, Nnet = build_doping(p, X, Y)
    ps = Poisson2D(x, y, Nnet, p, Phi_gate=0.30)
    dd = DDSolver(ps)

    print("Solving DD at Vg=3, Vd=3 ...")
    phi, n, pp = dd.solve(Vg=3.0, Vd=3.0, verbose=True)
    print("phi range %.3f..%.3f  n max %.2e  Id=%.3e A/cm" %
          (phi.min(), phi.max(), n.max(), dd.drain_current()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]; pcm = ax.pcolormesh(x, y, phi, cmap="turbo", shading="auto")
    ax.invert_yaxis(); ax.set_title("phi [V]  Vg=Vd=3"); fig.colorbar(pcm, ax=ax)
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    ax = axes[1]; pcm = ax.pcolormesh(x, y, np.log10(np.maximum(n,1)), cmap="viridis", shading="auto", vmin=6, vmax=20)
    ax.invert_yaxis(); ax.set_title("log10 n"); fig.colorbar(pcm, ax=ax)
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    ax = axes[2]; pcm = ax.pcolormesh(x, y, np.log10(np.maximum(pp,1)), cmap="magma", shading="auto", vmin=6, vmax=20)
    ax.invert_yaxis(); ax.set_title("log10 p"); fig.colorbar(pcm, ax=ax)
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    fig.tight_layout(); fig.savefig("figures/dd_bias.png", dpi=130)
    print("wrote figures/dd_bias.png")
