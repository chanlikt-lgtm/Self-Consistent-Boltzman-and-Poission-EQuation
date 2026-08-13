"""
device.py -- Reconstructed 0.35-um LDD N-MOSFET cross-section for the
Liang/Goldsman/Mayergoyz/Oldiges (1997) IEEE TED 44(2) reproduction.

Coordinate convention (matches paper figures):
    x  = lateral / transverse distance along the channel  [source (x=0) -> drain (x=Lx)]
    y  = depth into the device                            [Si/SiO2 interface (y=0) -> bulk]

HONESTY NOTE
------------
The paper's exact doping profile comes from ref [25] (Khalil et al., 1994 VLSI Symp.),
which is not in hand.  This module RECONSTRUCTS a physically faithful analytic LDD
profile consistent with the values the paper *does* state:
    max n+  ~ 1e20 cm^-3,  n- (LDD) ~ 1e18 cm^-3,  p-substrate ~ 1e16-1e17 cm^-3,
    t_ox = 9.6 nm,  source/drain junction depth x_j ~ 0.17 um,  L_eff ~ 0.35 um.
Any result that depends on the exact doping (threshold voltage, subthreshold slope)
is therefore a qualitative reproduction, and is flagged as such in the report.

Units: SI internally is avoided for doping; concentrations in cm^-3, lengths in um for
the geometry description, converted to meters for the physics solvers.
"""

import numpy as np
from scipy.special import erfc


# ----------------------------------------------------------------------------
# Physical / geometric parameters of the reconstructed device (micrometres)
# ----------------------------------------------------------------------------
class DeviceParams:
    # domain
    Lx = 0.80          # lateral extent [um]
    Ly = 0.70          # depth extent   [um]
    tox = 0.0096       # gate-oxide thickness [um] (9.6 nm, from paper)

    # gate electrode footprint (poly length ~0.40 um -> L_eff ~0.35 um after encroachment)
    x_gate1 = 0.20
    x_gate2 = 0.60

    # source (left) / drain (right) n+ lateral edges [um]
    x_srcnplus = 0.150   # n+ present for x < this on source side
    x_drnnplus = 0.650   # n+ present for x > this on drain side

    # LDD (n-) inner edges -- the lightly doped spacer extends further under the gate
    x_srcldd = 0.220     # n- extends up to this x on source side
    x_drnldd = 0.580     # n- extends down to this x on drain side

    # junction depths [um].
    # NOTE ON DEFINITION (audit Finding 3): x_j here is calibrated so that the
    # *metallurgical* junction -- the depth where N_D(y)=N_A(y) under the S/D contact --
    # falls at ~0.17 um, matching the paper's quoted x_j. For an erfc profile crossing
    # a 1e16 substrate from a 1e20 peak, the metallurgical crossing sits at
    # xj_peak + 2.63*sig_dep, so xj_nplus=0.105 + 2.63*0.025 = 0.171 um.
    xj_nplus = 0.105     # -> metallurgical n+/p junction at ~0.171 um
    xj_ldd = 0.060       # shallow LDD junction

    # lateral / depth straggles for the erfc junctions [um]
    sig_lat = 0.028
    sig_dep = 0.025

    # channel implant (shallow, surface) to set a realistic V_th
    x_chimp1 = 0.18      # channel implant lateral span
    x_chimp2 = 0.62
    xj_chimp = 0.060
    sig_chimp_dep = 0.035

    # doping magnitudes [cm^-3]
    Nd_nplus = 1.0e20
    Nd_ldd = 1.0e18
    Na_sub = 1.0e16      # background p-substrate (paper: order 1e16)
    Na_chimp = 4.0e17    # peak channel-implant acceptor (reconstruction, sets V_th)


def _lateral_left(x, edge, sig):
    """~1 for x<edge, rolling smoothly to 0 for x>edge (source side)."""
    return 0.5 * erfc((x - edge) / sig)


def _lateral_right(x, edge, sig):
    """~1 for x>edge, rolling smoothly to 0 for x<edge (drain side)."""
    return 0.5 * erfc((edge - x) / sig)


def _depth(y, xj, sig):
    """Peak at surface y=0, rolling to 0 near depth xj (retrograde-ish erfc)."""
    return 0.5 * erfc((y - xj) / sig)


def build_doping(params: DeviceParams, X, Y):
    """
    Return donor, acceptor, and net doping arrays [cm^-3] on the (X,Y) mesh grids.
    X, Y are 2-D meshgrid arrays in micrometres, shape (Ny, Nx).
    Net > 0 => n-type, Net < 0 => p-type.
    """
    p = params

    # --- Donors: n+ source/drain + n- LDD extensions ---
    src_nplus = p.Nd_nplus * _lateral_left(X, p.x_srcnplus, p.sig_lat) * _depth(Y, p.xj_nplus, p.sig_dep)
    drn_nplus = p.Nd_nplus * _lateral_right(X, p.x_drnnplus, p.sig_lat) * _depth(Y, p.xj_nplus, p.sig_dep)

    src_ldd = p.Nd_ldd * _lateral_left(X, p.x_srcldd, p.sig_lat) * _depth(Y, p.xj_ldd, p.sig_dep)
    drn_ldd = p.Nd_ldd * _lateral_right(X, p.x_drnldd, p.sig_lat) * _depth(Y, p.xj_ldd, p.sig_dep)

    Nd = src_nplus + drn_nplus + src_ldd + drn_ldd

    # --- Acceptors: uniform p-substrate + shallow channel implant ---
    sub = p.Na_sub * np.ones_like(X)
    chimp = (p.Na_chimp
             * 0.5 * (erfc((X - p.x_chimp2) / p.sig_lat) - erfc((X - p.x_chimp1) / p.sig_lat))
             * _depth(Y, p.xj_chimp, p.sig_chimp_dep))
    Na = sub + chimp

    Nnet = Nd - Na
    return Nd, Na, Nnet


def make_mesh(params: DeviceParams, Nx=100, Ny=90, refine_surface=True):
    """
    Build a (possibly surface-refined) tensor-product mesh.
    Returns x, y (1-D, um), X, Y (2-D meshgrid, shape (Ny,Nx)).
    Surface refinement clusters y-nodes near the Si/SiO2 interface where the
    inversion layer and hot-carrier action live.
    """
    p = params
    x = np.linspace(0.0, p.Lx, Nx)

    if refine_surface:
        # geometric-ish clustering: dense near y=0, coarser toward the bulk
        s = np.linspace(0.0, 1.0, Ny)
        # power-law stretch
        y = p.Ly * (s ** 1.8)
    else:
        y = np.linspace(0.0, p.Ly, Ny)

    X, Y = np.meshgrid(x, y)   # shape (Ny, Nx)
    return x, y, X, Y


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    p = DeviceParams()
    x, y, X, Y = make_mesh(p, Nx=160, Ny=140)
    Nd, Na, Nnet = build_doping(p, X, Y)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (1) net doping, symmetric log
    ax = axes[0, 0]
    vmax = 1e20
    pcm = ax.pcolormesh(x, y, Nnet, cmap="RdBu_r",
                        norm=SymLogNorm(linthresh=1e15, vmin=-vmax, vmax=vmax, base=10),
                        shading="auto")
    ax.invert_yaxis()
    ax.set_title("Net doping  N_D - N_A  [cm$^{-3}$]  (red=n, blue=p)")
    ax.set_xlabel("x  transverse [um]"); ax.set_ylabel("y  depth [um]")
    ax.axhline(0, color="k", lw=0.5)
    ax.plot([p.x_gate1, p.x_gate2], [0, 0], "g-", lw=4, label="gate")
    ax.legend(loc="lower right")
    fig.colorbar(pcm, ax=ax)

    # (2) |net| doping log10
    ax = axes[0, 1]
    pcm = ax.pcolormesh(x, y, np.log10(np.abs(Nnet) + 1), cmap="viridis", shading="auto")
    ax.invert_yaxis()
    ax.set_title("log10 |N_D - N_A|")
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    fig.colorbar(pcm, ax=ax)

    # (3) surface doping line (y ~ 0)
    ax = axes[1, 0]
    j0 = 0
    ax.semilogy(x, np.abs(Nnet[j0, :]), "b-")
    ax.set_title("surface (y=0) |net doping| along channel")
    ax.set_xlabel("x [um]"); ax.set_ylabel("|N_net| [cm^-3]")
    ax.axvspan(p.x_gate1, p.x_gate2, color="g", alpha=0.12, label="gate")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    # (4) vertical cut through source and channel
    ax = axes[1, 1]
    ix_src = np.argmin(np.abs(x - 0.05))
    ix_chan = np.argmin(np.abs(x - 0.40))
    ix_drn = np.argmin(np.abs(x - 0.75))
    ax.semilogy(y, np.abs(Nnet[:, ix_src]), label="source x=0.05")
    ax.semilogy(y, np.abs(Nnet[:, ix_chan]), label="channel x=0.40")
    ax.semilogy(y, np.abs(Nnet[:, ix_drn]), label="drain x=0.75")
    ax.set_title("vertical doping cuts")
    ax.set_xlabel("y depth [um]"); ax.set_ylabel("|N_net| [cm^-3]")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    fig.tight_layout()
    out = "figures/device_doping.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print("Nnet range: %.2e .. %.2e" % (Nnet.min(), Nnet.max()))
    print("peak |Nnet| = %.2e" % np.abs(Nnet).max())
