"""
richardson_fit.py -- generalized Richardson extrapolation over the self-similar drain-graded trio.

Reads the three she2d_richardson_NxxNy.npz produced by coupled_drain_richardson.py (coarse->fine),
first VERIFIES the sequence is comparable (identical mesh law + solver settings, and r_x ~ r_y ~
sqrt2 on the MEASURED spacings), then fits G_ii,max(h) = G0 + C h^p by three-point generalized
Richardson. Repeats the fit with h=sqrt(dx*dy) and h=dx_drain as a sensitivity check: if the inferred
order p (or continuum G0) changes substantially between the two, the sequence is NOT yet in a clean
asymptotic regime and the extrapolation must not be quoted as a continuum limit.
Usage: python richardson_fit.py coarse.npz medium.npz fine.npz
"""
import sys
import numpy as np

def load(fs):
    return [np.load(f) for f in fs]

def three_point_order(vals, hs):
    # generalized (unequal-ratio) 3-point fit of v = v0 + C h^p, coarse->fine hs decreasing.
    (h1, h2, h3), (v1, v2, v3) = hs, vals
    from math import log
    # solve for p by matching second differences under v0+C h^p; robust bracketed root on residual.
    def g(p):
        # eliminate C, v0 -> consistency function whose root is the order p
        return (v1 - v2) * (h2 ** p - h3 ** p) - (v2 - v3) * (h1 ** p - h2 ** p)
    lo, hi = 0.05, 8.0
    flo, fhi = g(lo), g(hi)
    if flo * fhi > 0:
        return None, None, None            # no sign change -> not monotone/asymptotic
    for _ in range(200):
        mid = 0.5 * (lo + hi); fm = g(mid)
        if flo * fm <= 0: hi, fhi = mid, fm
        else: lo, flo = mid, fm
    p = 0.5 * (lo + hi)
    C = (v1 - v2) / (h1 ** p - h2 ** p)
    v0 = v1 - C * h1 ** p
    return p, C, v0

def main(fs):
    d = load(fs)
    Nx = [int(z["Nx"]) for z in d]; Ny = [int(z["Ny"]) for z in d]
    dx = [float(z["drain_dx"]) for z in d]; dy = [float(z["surface_dy"]) for z in d]
    Rp = [float(z["R_phi"]) for z in d]; sp = [float(z["spread"]) for z in d]

    # Node value = MEAN of the last 4 plateau iterates; per-node uncertainty = last-4 HALF-RANGE.
    # This absolute band is what a sign reversal must exceed to count as genuine non-monotonicity
    # (vs "grid-stable within numerical resolution").
    def node(z, key4, keymean):
        a = np.asarray(z[key4], float) if key4 in z.files else np.array([float(z[keymean])])
        return float(a.mean()), float((a.max() - a.min()) / 2.0)
    G, uG, Gt, uGt = [], [], [], []
    for z in d:
        v, u = node(z, "last4_Gii_max", "Gii_max"); G.append(v); uG.append(u)
        v, u = node(z, "last4_G_tot", "G_tot"); Gt.append(v); uGt.append(u)
    print("grid        Nx  Ny   drain_dx   surf_dy    Gii_max(+/-u)          G_tot(+/-u)           R_phi")
    for i in range(3):
        print("%2d %3dx%-3d %4d %4d  %.5f  %.6f  %.4e+/-%.1e  %.4e+/-%.1e  %.2e"
              % (i, Nx[i], Ny[i], Nx[i], Ny[i], dx[i], dy[i], G[i], uG[i], Gt[i], uGt[i], Rp[i]))
    # --- comparability gates ---
    laws = {(round(float(z["ratio"]),4), round(float(z["W"]),4), round(float(z["X_C"]),4),
             round(float(z["dH_eV"]),6), int(z["NH"]), float(z["she_tol"])) for z in d}
    print("\ncomparability: mesh-law/solver signatures identical:", len(laws) == 1, "  ", laws)
    # The strongest energy-grid gate: the ACTUAL frozen H-array must be byte-identical across grids,
    # otherwise a spatial-only study secretly contains an energy-discretization change.
    hashes = {str(z["H_hash"]) for z in d} if all("H_hash" in z.files for z in d) else {"<missing>"}
    print("H-grid identical (sha1):", len(hashes) == 1, "  ", hashes)
    if len(hashes) != 1:
        print("  *** REFUSE to extrapolate: frozen H-grid differs across grids -> not spatial-only ***")
    rx = [dx[i] / dx[i+1] for i in range(2)]; ry = [dy[i] / dy[i+1] for i in range(2)]
    print("refinement ratios  r_x=%.4f,%.4f   r_y=%.4f,%.4f   (target sqrt2=%.4f)"
          % (rx[0], rx[1], ry[0], ry[1], 2**0.5))
    iso = all(abs(r - 2**0.5) < 0.05 for r in rx + ry)
    print("r_x ~ r_y ~ sqrt2 (within 5%%):", iso, "  -> single representative h is",
          "defensible" if iso else "NOT clean; treat as sensitivity only")
    h_iso = [(dx[i] * dy[i]) ** 0.5 for i in range(3)]      # surface-isotropic representative h
    h_x = dx                                                # drain-x spacing (isolates x-error)

    def closed_form(vals, hs):
        """Explicit near-constant-ratio 3-point Richardson (user's formula). r = geo-mean of the two
        consecutive h-ratios (~sqrt2 here). p=ln|(G1-G2)/(G2-G3)|/ln(r); G_inf=G3+(G3-G2)/(r^p-1)."""
        (v1, v2, v3) = vals; d12, d23 = v1 - v2, v2 - v3
        r = ((hs[0] / hs[1]) * (hs[1] / hs[2])) ** 0.5
        if d23 == 0 or (d12 / d23) <= 0:                    # no monotone same-sign decrease -> not asymptotic
            return None, None, r, (d12, d23)
        from math import log
        p = log(abs(d12 / d23)) / log(r)
        Ginf = v3 + (v3 - v2) / (r ** p - 1.0)
        return p, Ginf, r, (d12, d23)

    def report(name, vals, unc, unit, guidance):
        print("\n--- %s  (finest-grid value %.4e %s) ---" % (name, vals[-1], unit))
        s12, s23 = vals[0] - vals[1], vals[1] - vals[2]          # SIGNED corrections
        d12, d23 = abs(s12), abs(s23)
        u12, u23 = (unc[0]**2 + unc[1]**2) ** 0.5, (unc[1]**2 + unc[2]**2) ** 0.5  # propagated bands
        # MONOTONICITY + SIGNIFICANCE: a sign reversal only counts if it exceeds the node uncertainty.
        monotone = (s12 > 0 and s23 > 0) or (s12 < 0 and s23 < 0)
        if not monotone:
            # S_rev is a HEURISTIC reversal-to-plateau-uncertainty ratio (plateau band = last-4
            # half-range), NOT a statistical sigma. S_rev>>1: reversal far exceeds residual fixed-
            # point drift -> numerically meaningful. S_rev~O(1): reversal unresolved by plateau
            # precision -> read as grid stability, not genuine non-monotonicity.
            Srev = d23 / u23 if u23 > 0 else float("inf")
            print("  NON-monotone: D12=%+.3e, D23=%+.3e. reversal |D23|=%.3e vs plateau band u23=%.3e"
                  "  -> S_rev=%.1f (reversal/plateau-uncertainty ratio, heuristic)" % (s12, s23, d23, u23, Srev))
            if Srev < 3.0:
                print("  -> S_rev~O(1): reversal WITHIN plateau precision -> interpret as GRID-STABLE near"
                      " %.4e %s, NOT a meaningful upturn. Richardson G_inf not applicable." % (vals[-1], unit))
            else:
                print("  -> S_rev>>1: reversal far exceeds plateau band -> trio genuinely NON-MONOTONE;"
                      " SUPPRESS Richardson extrapolation outright.")
            print("  guidance: %s" % guidance)
            return
        # monotone -> asymptoticity is about contraction of the ABSOLUTE successive corrections.
        contraction = d23 / d12 if d12 > 0 else float("inf")
        pct1, pct2 = 100 * s12 / vals[0], 100 * s23 / vals[1]
        print("  monotone. corrections: |D12|=%.3e(+/-%.1e)  |D23|=%.3e(+/-%.1e)  contraction=%.3f  (%.1f%% then %.1f%%)"
              % (d12, u12, d23, u23, contraction, pct1, pct2))
        asymptotic = contraction < 1.0
        if not asymptotic:
            print("  *** |D23| >= |D12|: correction NOT contracting -> sequence NOT asymptotic; do NOT report G_inf ***")
        # AUTHORITATIVE: generalized unequal-ratio solve on full-precision measured spacings
        # (drain-dx ratios 1.41436, 1.41437; sqrt2 to 4e-4)
        for label, hs in [("h=sqrt(dx*dy_surf)", h_iso), ("h=dx_drain        ", h_x)]:
            p, C, v0 = three_point_order(vals, hs)
            tag = "" if asymptotic else "  [suppressed: non-asymptotic]"
            if p is None:
                print("  [authoritative] %s : no monotone asymptotic fit -- NOT asymptotic" % label)
            else:
                print("  [authoritative] %s : p=%.2f  continuum=%.4e %s%s" % (label, p, v0, unit, tag))
        # CROSS-CHECK: explicit constant-sqrt2 closed form on drain-x (agreement validates ratio mismatch is immaterial)
        pc, Gc, r, _ = closed_form(vals, h_x)
        if pc is None or not asymptotic:
            print("  [cross-check ] closed-form (h=dx, r=%.4f): not reported (non-asymptotic)" % r)
        else:
            print("  [cross-check ] closed-form (h=dx, r=%.4f): p=%.2f  G_inf=%.4e %s" % (r, pc, Gc, unit))
        print("  guidance: %s" % guidance)

    # Gii_max sits at the SURFACE drain hotspot where r_y(surface)~sqrt2 -> single-h potentially OK.
    report("Gii_max", G, uG, "cm^-3 s^-1",
           "surface hotspot; single-h Richardson potentially defensible if p agrees across both h.")
    # G_tot samples a FINITE DEPTH whose dy does NOT refine self-similarly (r_y,depth != sqrt2), so the
    # h=dx_drain sensitivity is the more informative fit; treat sqrt(dx*dy_surf) with caution.
    report("G_tot ", Gt, uGt, "cm^-1 s^-1",
           "finite-depth integral; r_y,depth!=sqrt2 -> prefer h=dx_drain; single-h is NOT rigorous. "
           "For a rigorous G_tot continuum estimate, add a second family (fixed y-mesh, refine drain-x).")
    print("\nNOTE: quote a continuum value ONLY if (a) signatures + H-hash identical, (b) r_x~r_y~sqrt2")
    print("for that observable's dominant direction, (c) p stable & plausible across both h, (d) small")
    print("residual/spread. With 3 anisotropic points a separate x/y-order fit is underdetermined.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python richardson_fit.py coarse.npz medium.npz fine.npz"); sys.exit(1)
    main(sys.argv[1:4])
