"""
coupled_drain_richardson.py -- ONE parameterized driver for a self-similar drain-graded mesh
sequence, so a later Richardson study uses byte-identical grading + solver settings on every grid
(only Nx,Ny change). Run ONLY after the 5th fixed point is established, and run the grids
SEQUENTIALLY (memory: ~6 GB per fine grid).

CONTROLLED SEQUENCE (Option A -- isotropic sqrt(2) in BOTH measured spacings; verified):
    71x58  ->  100x70  ->  141x85
    drain dx refines 1.4144, 1.4144 ; surface dy refines 1.410, 1.425  (both ~ sqrt2)
The literal 71x50/141x98 was REJECTED: under the y-law Ly*s^1.8 its surface dy refines by ~1.85,
not sqrt2, so r_x != r_y and a single representative h would be contaminated. Ny adjusted to 58/85
restores r_y ~ sqrt2. The 100x70 middle grid == the converged 4th point's mesh/law (legitimate
reuse); rerunning it here guarantees an identical stopping rule / diagnostic pipeline across the trio.

FROZEN grading law: ratio=6, W=0.09, X_C=0.60.  FROZEN solver: Phi_gate=-0.74, accel=anderson,
aa_depth=6, aa_beta=0.5, aa_step_cap=5.0, freeze_H=True, she_tol=1e-8, dH=12.5 meV, tol_phi=1e-4,
Vg=Vd=3 V. Do NOT change these between grids -- solver accuracy would leak into the spatial order p.

Usage: python coupled_drain_richardson.py Nx Ny [maxouter] [warmstart_npz]
"""
import os, sys, hashlib, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from device import DeviceParams
from coupled_she import solve_coupled
from she2d import EPS_MAX
from constants import eV

# ONE COMMON frozen-H span imposed on every grid so the H-grid (H_lo,H_hi,NH,H-array) is byte-
# identical across the trio -- prevents a hidden energy-discretization change in a spatial-only study.
# Bounds the loop's phi range (measured ~[-0.348, 3.586] V at Vg=Vd=3 V) with margin; verified per run.
COMMON_PHI_SPAN = (-0.60, 3.65)

def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "?"
_git_commit = _git("rev-parse", "--short", "HEAD")
_git_dirty = "dirty" if _git("status", "--porcelain") else "clean"

# ---- FROZEN grading law + solver settings (identical across the whole sequence) ----
RATIO, W, X_C = 6.0, 0.09, 0.60
PHI_GATE, AA_DEPTH, AA_BETA, AA_STEP_CAP, AA_RESTART = -0.74, 6, 0.5, 5.0, 6
SHE_TOL, DH_EV, TOL_PHI = 1e-8, 0.0125, 1e-4
EPS_MAX_EV = EPS_MAX / eV                                   # energy ceiling (absorbing top), fixed
SPREAD_DEF = "(max-min)/mean of the last 4 outer-iteration Gii_max values"
STOP_DEF = "max|G(phi)-phi| over free DOFs < tol_phi (Anderson true fixed-point residual)"
# ---- SANCTIONED self-similar trio (Option A: r_x ~ r_y ~ sqrt2 on MEASURED spacings) ----
SEQUENCE = [(71, 58), (100, 70), (141, 85)]
TARGET_R = 2.0 ** 0.5
RATIO_TOL = 0.05                                            # allowed |r-sqrt2| before fail-fast
REF_DEPTH = 0.02                                            # a-priori avalanche-depth probe [um]

p = DeviceParams(); Lx, Ly = p.Lx, p.Ly

def dy_at(y, depth):
    """Local mesh interval size at a given depth [um] -- because y=s^1.8, dy grows with depth, so a
    surface-isotropic sequence is not automatically isotropic at the avalanche depth (user's caveat).
    Verifying dy(REF_DEPTH) also refines ~sqrt2 guards the G_tot (vertically-integrated) observable."""
    k = int(np.clip(np.searchsorted(y, depth) - 1, 0, len(y) - 2))
    return float(y[k + 1] - y[k])

def build_mesh(Nx, Ny):
    xf = np.linspace(0.0, Lx, 8000)
    dens = 1.0 + (RATIO - 1.0) * np.exp(-((xf - X_C) / W) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xf))]); cum /= cum[-1]
    x = np.interp(np.linspace(0.0, 1.0, Nx), cum, xf); x[0], x[-1] = 0.0, Lx
    y = Ly * (np.linspace(0.0, 1.0, Ny) ** 1.8)
    dxd = float(np.diff(x)[np.argmin(np.abs(0.5 * (x[:-1] + x[1:]) - X_C))])
    dys = float(np.diff(y).min())
    dyd = dy_at(y, REF_DEPTH)
    return x, y, dxd, dys, dyd

Nx = int(sys.argv[1]); Ny = int(sys.argv[2])
maxo = int(sys.argv[3]) if len(sys.argv) > 3 else 60
warm = sys.argv[4] if len(sys.argv) > 4 else None

# ---- FAIL-FAST gate 1: this grid must be a sanctioned Richardson node ----
if (Nx, Ny) not in SEQUENCE:
    sys.exit("REFUSED: (Nx=%d,Ny=%d) is not a sanctioned self-similar node %s. "
             "Richardson comparability requires the fixed trio." % (Nx, Ny, SEQUENCE))

# ---- FAIL-FAST gate 2: verify the WHOLE trio refines by ~sqrt2 in drain dx, surface dy, AND the
#      avalanche-depth dy (surface-isotropic is not automatically isotropic at depth under y=s^1.8) --
seq = [build_mesh(a, b) for a, b in SEQUENCE]
rx = [seq[i][2] / seq[i + 1][2] for i in range(len(SEQUENCE) - 1)]
ry = [seq[i][3] / seq[i + 1][3] for i in range(len(SEQUENCE) - 1)]
ryd = [seq[i][4] / seq[i + 1][4] for i in range(len(SEQUENCE) - 1)]
print("pre-solve ratio check  r_x=%s  r_y(surf)=%s  r_y(@%.3fum)=%s  (target sqrt2=%.4f, tol %.2f)"
      % (["%.4f" % r for r in rx], ["%.4f" % r for r in ry], REF_DEPTH,
         ["%.4f" % r for r in ryd], TARGET_R, RATIO_TOL))
bad = [r for r in rx + ry if abs(r - TARGET_R) > RATIO_TOL]
if bad:
    sys.exit("REFUSED: surface refinement ratios %s deviate from sqrt2 beyond tol -- not self-similar."
             % ["%.4f" % r for r in bad])
if [r for r in ryd if abs(r - TARGET_R) > RATIO_TOL]:
    print("WARNING: avalanche-depth dy ratios %s deviate from sqrt2 -- single-h G_tot extrapolation "
          "is surface-isotropic only; treat depth-sensitive G_tot fit with caution."
          % ["%.4f" % r for r in ryd], flush=True)

x, y, dx_drain, dy_surf, dy_depth = build_mesh(Nx, Ny)
h_rep = float(np.sqrt(dx_drain * dy_surf))

# ---------- RUN MANIFEST: printed before solving so every node proves it is comparable ----------
print("=" * 78)
print("RICHARDSON NODE MANIFEST (must be identical across the sequence except Nx,Ny,dx,dy)")
print("  mesh law     : ratio=%.3f  W=%.3f um  X_C=%.3f um   (x graded, y = Ly*s^1.8)" % (RATIO, W, X_C))
print("  grid         : Nx=%d Ny=%d  (x-intervals=%d, y-intervals=%d, cells=%d)" % (Nx, Ny, Nx - 1, Ny - 1, Nx * Ny))
print("  measured     : drain dx=%.6f um  surface dy=%.6f um  dy@%.3fum=%.6f um  h_rep=%.6f um"
      % (dx_drain, dy_surf, REF_DEPTH, dy_depth, h_rep))
print("  energy grid  : dH=%.4f eV  eps_max=%.3f eV  COMMON frozen H-span=%s V (identical all grids)" % (DH_EV, EPS_MAX_EV, COMMON_PHI_SPAN))
print("  SHE solve    : she_tol=%.1e" % SHE_TOL)
print("  Anderson     : depth=%d  beta=%.2f  step_cap=%.1f  restart=%d" % (AA_DEPTH, AA_BETA, AA_STEP_CAP, AA_RESTART))
print("  outer loop   : max_outer=%d   tol_phi=%.1e" % (maxo, TOL_PHI))
print("  stopping     : %s" % STOP_DEF)
print("  last-4 spread: %s" % SPREAD_DEF)
print("  init policy  : DD Gummel warm-up, then override with %s" % ("warm-start npz" if warm else "no override (cold DD start)"))
print("  warm source  : %s" % (os.path.basename(warm) if warm else "(none)"))
print("  provenance   : git %s (%s)" % (_git_commit, _git_dirty))
print("  bias         : Vg=Vd=3 V  Vs=Vb=0   Phi_gate=%.3f V" % PHI_GATE)
print("=" * 78, flush=True)

phi0 = p0 = None
if warm and os.path.exists(warm):
    prev = np.load(warm); xo, yo = prev["x"], prev["y"]
    def regrid(F):
        tmp = np.empty((len(yo), Nx))
        for j in range(len(yo)):
            tmp[j] = np.interp(x, xo, F[j])
        out = np.empty((Ny, Nx))
        for i in range(Nx):
            out[:, i] = np.interp(y, yo, tmp[:, i])
        return out
    phi0, p0 = regrid(prev["phi"]), regrid(prev["p"])   # keep 2-D (Ny, Nx)
    print("warm-started from %s (interpolated phi,p)" % os.path.basename(warm), flush=True)

r = solve_coupled(x_mesh=x, y_mesh=y, Phi_gate=PHI_GATE, accel="anderson", aa_depth=AA_DEPTH,
                  aa_beta=AA_BETA, aa_step_cap=AA_STEP_CAP, freeze_H=True, dH_eV=DH_EV,
                  she_tol=SHE_TOL, max_outer=maxo, tol_phi=TOL_PHI, verbose=True,
                  phi_init=phi0, p_init=p0, phi_span_H_override=COMMON_PHI_SPAN)

# HARD FAIL: the CONVERGED phi must also stay inside the imposed common span, else the frozen H-grid
# (fixed at loop start) no longer bounds the final solution's energy domain -- invalidate the run.
_pmin, _pmax = float(r["phi"].min()), float(r["phi"].max())
if _pmin < COMMON_PHI_SPAN[0] or _pmax > COMMON_PHI_SPAN[1]:
    sys.exit("ABORT: converged phi [%.4f, %.4f] V leaves COMMON_PHI_SPAN %s V. Widen the common span "
             "for ALL three grids and rerun the whole sequence." % (_pmin, _pmax, COMMON_PHI_SPAN))

h = r["history"]
G = np.array([rec["Gii_max_cm3s"] for rec in h])
Gt_hist = np.array([rec.get("G_tot_cm1s", np.nan) for rec in h])   # now tracked per outer iteration
n_iter = len(h)
Gii, xg, yg = r["Gii"], r["x"], r["y"]
jg, ig = np.unravel_index(int(np.argmax(Gii)), Gii.shape)
dxc = np.zeros(len(xg)); dxc[1:-1] = 0.5 * (xg[2:] - xg[:-2]); dxc[0] = 0.5 * (xg[1] - xg[0]); dxc[-1] = 0.5 * (xg[-1] - xg[-2])
dyc = np.zeros(len(yg)); dyc[1:-1] = 0.5 * (yg[2:] - yg[:-2]); dyc[0] = 0.5 * (yg[1] - yg[0]); dyc[-1] = 0.5 * (yg[-1] - yg[-2])
G_tot = float(np.sum(Gii * np.outer(dyc, dxc) * 1e-8))
sp = (G[-4:].max() - G[-4:].min()) / G[-4:].mean()
last4_G = G[-4:]; last4_Gt = Gt_hist[-4:]
mean4_G = float(last4_G.mean()); mean4_Gt = float(np.nanmean(last4_Gt))
sp_Gt = float((np.nanmax(last4_Gt) - np.nanmin(last4_Gt)) / np.nanmean(last4_Gt))
R_phi = float(h[-1]["max_dphi_V"])
last4_R = np.array([rec["max_dphi_V"] for rec in h[-4:]])
she = r["she"]; NH = int(she.NH)
Hmin, Hmax = float(she.H.min()) / eV, float(she.H.max()) / eV   # report in eV (H stored in Joules)
# Hash the ACTUAL frozen H-array: the strongest guarantee the three grids share one energy grid.
H_hash = hashlib.sha1(np.ascontiguousarray(she.H, dtype=np.float64).tobytes()).hexdigest()[:12]

# Avalanche vertical extent: y-FWHM of Gii along the peak x-column, and the local dy there.
col = Gii[:, ig]
half = 0.5 * col.max()
below = np.where(col[jg:] < half)[0]
j_fwhm = (jg + int(below[0])) if len(below) else len(yg) - 1
y_fwhm = float(yg[min(j_fwhm, len(yg) - 1)])
dy_avalanche = dy_at(yg, max(y_fwhm, REF_DEPTH * 0.0))

# ---------- REALIZED H-grid manifest (known only after the first SHE solve) ----------
print("\nrealized H-grid: NH=%d  H-domain=[%.4f, %.4f] eV  H-array sha1=%s  (COMMON frozen grid; must match across trio)" % (NH, Hmin, Hmax, H_hash))
print("=== self-similar node Nx=%d Ny=%d  (converged in %d outer iters) ===" % (Nx, Ny, n_iter))
print("drain dx=%.6f um  surface dy=%.6f um  dy@%.3fum=%.6f um  h_rep=%.6f um" % (dx_drain, dy_surf, REF_DEPTH, dy_depth, h_rep))
print("Gii_max = %.4e at (x=%.4f, y=%.4f) um   avalanche y-FWHM=%.4f um (local dy=%.6f)" % (Gii.max(), xg[ig], yg[jg], y_fwhm, dy_avalanche))
print("G_tot   = %.4e cm^-1 s^-1" % G_tot)
print("Te_max  = %.0f K   n_max = %.3e" % (r["Te"].max(), r["n"].max()))
print("R_phi (final fixed-point residual) = %.3e V   last-4 spread: Gii=%.2e  Gtot=%.2e" % (R_phi, sp, sp_Gt))
print("last 4 Gii_max: " + " ".join("%.4e" % v for v in last4_G) + "   (mean %.4e)" % mean4_G)
print("last 4 G_tot  : " + " ".join("%.4e" % v for v in last4_Gt) + "   (mean %.4e)" % mean4_Gt)
np.savez(os.path.join(os.path.dirname(__file__), "..", "data", "she2d_richardson_%dx%d.npz" % (Nx, Ny)),
         x=xg, y=yg, phi=r["phi"], n=r["n"], p=r["p"], Te=r["Te"], Gii=Gii, vx=r["vx"], vy=r["vy"],
         F3d=r["F3d"].astype(np.float32), H=r["H"], Gamma_x_face=r["Gamma_x_face"], Gamma_y_face=r["Gamma_y_face"],
         # --- Richardson diagnostic vector ---
         Nx=Nx, Ny=Ny, ratio=RATIO, W=W, X_C=X_C, dH_eV=DH_EV, eps_max_eV=EPS_MAX_EV,
         she_tol=SHE_TOL, NH=NH, Hmin=Hmin, Hmax=Hmax, H_hash=H_hash,
         common_phi_span=np.array(COMMON_PHI_SPAN), n_iter=n_iter,
         git_commit=_git_commit, git_dirty=_git_dirty, warm_source=(os.path.basename(warm) if warm else ""),
         drain_dx=dx_drain, surface_dy=dy_surf, dy_depth=dy_depth, ref_depth=REF_DEPTH,
         avalanche_y_fwhm=y_fwhm, dy_avalanche=dy_avalanche, h_rep=h_rep,
         Gii_max=float(Gii.max()), G_tot=G_tot, Te_max=float(r["Te"].max()),
         xy_Gii_max=np.array([xg[ig], yg[jg]]), R_phi=R_phi,
         last4_Gii_max=last4_G, last4_G_tot=last4_Gt, last4_R_phi=last4_R,
         mean4_Gii_max=mean4_G, mean4_G_tot=mean4_Gt, spread=sp, spread_Gtot=sp_Gt)
print("saved data/she2d_richardson_%dx%d.npz" % (Nx, Ny))
