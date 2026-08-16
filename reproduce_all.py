#!/usr/bin/env python
"""
reproduce_all.py -- Standalone driver for the Liang, Goldsman, Mayergoyz & Oldiges (1997)
deterministic SHE-Boltzmann 2-D MOSFET reproduction. Runs the ENTIRE pipeline from scratch
(reconstructed device -> nonlinear Poisson + drift-diffusion -> first-order SHE-BTE with impact
ionization -> self-consistent coupling -> moments) and regenerates EVERY figure used in the report,
by orchestrating the physics modules in src/ (which together are the from-scratch implementation).

The heavy numerics live in src/ (device.py, poisson.py, dd.py, bands.py, scattering.py, she2d.py,
coupled_she.py, ...). This file is the single entry point that reproduces all plots in the right
dependency order; each stage is one or more of those scripts run as a subprocess.

--------------------------------------------------------------------------------------------------
USAGE
    python reproduce_all.py                 # core physics figures (fast-ish, ~15-20 min)
    python reproduce_all.py --full          # EVERYTHING incl. 50-iter coupled + SHE-moment I-V
                                            #   (~1.5-2 h; this is "from scratch till all plots")
    python reproduce_all.py --list          # list stages, their outputs, deps and runtimes
    python reproduce_all.py --stage overlays        # run a single stage (deps must already exist)
    python reproduce_all.py --from coupled_recal    # run from this stage to the end
    python reproduce_all.py --full --keep-going     # don't stop on a failing stage

REQUIREMENTS
    Python 3.11+, numpy, scipy, matplotlib. The C .eff parser is NOT needed (the device is analytic).
    The side-by-side paper crops additionally need `pdftoppm` (ships with MiKTeX/poppler) and the
    source paper PDF in the repo root; that stage skips gracefully if either is missing.
--------------------------------------------------------------------------------------------------
"""
import os
import sys
import time
import shutil
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
FIGS = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")

# recalibrated coupled result: coupled_converge.py writes *_conv.npz; the overlays read the
# canonical name, so the driver copies conv -> canonical after the convergence stage.
CONV_NPZ = os.path.join(DATA, "she2d_coupled_result_60x50_Phg-0.74_conv.npz")
CANON_NPZ = os.path.join(DATA, "she2d_coupled_result_60x50_Phg-0.74.npz")


def _run_script(script, *args):
    # Run from ROOT as `python src/<script>`: relative "figures/..." outputs resolve to
    # ROOT/figures, __file__-relative outputs also resolve correctly, and Python adds src/ to
    # sys.path so `from device import ...` works. This satisfies both path conventions in src/.
    cmd = [sys.executable, "-u", os.path.join("src", script), *map(str, args)]
    print("    $ python %s %s" % (os.path.join("src", script), " ".join(map(str, args))), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print("      -> exit %d  (%.0f s)" % (r.returncode, dt), flush=True)
    if r.returncode != 0:
        raise RuntimeError("script failed: %s (exit %d)" % (script, r.returncode))


def _copy_conv_to_canonical():
    if os.path.exists(CONV_NPZ):
        if os.path.exists(CANON_NPZ):
            shutil.copy(CANON_NPZ, CANON_NPZ.replace(".npz", "_snap12.npz"))
        shutil.copy(CONV_NPZ, CANON_NPZ)
        print("    (copied 50-iter conv result -> canonical recalibrated npz)", flush=True)
    else:
        print("    ! conv npz missing; overlays will use whatever canonical npz exists", flush=True)


def _paper_crops():
    """Render paper Figs 2-8 from the source PDF (needs pdftoppm) and crop them. Skips if absent."""
    import glob
    import numpy as np
    import matplotlib.image as mpimg

    pdfs = [f for f in glob.glob(os.path.join(ROOT, "*.pdf")) if "MOSFET_modeling" in f]
    pdftoppm = shutil.which("pdftoppm") or \
        r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdftoppm.exe"
    if not pdfs or not (os.path.exists(pdftoppm) or shutil.which("pdftoppm")):
        print("    ! paper PDF or pdftoppm not found -- skipping paper crops "
              "(side-by-side needs figures/paper_fig*.png)", flush=True)
        return
    pdf = pdfs[0]
    tmp = os.path.join(DATA, "_paper_pages")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([pdftoppm, "-png", "-r", "220", "-f", "6", "-l", "8", pdf,
                    os.path.join(tmp, "pg")], check=True)
    pages = {p: mpimg.imread(os.path.join(tmp, "pg-%02d.png" % p)) for p in (6, 7, 8)}
    BOX = {  # (page, y0,y1, x0,x1) fractions -- see report Appendix / crop provenance
        "fig2": (6, 0.045, 0.828, 0.035, 0.485), "fig3": (6, 0.058, 0.315, 0.505, 0.985),
        "fig4": (6, 0.328, 0.615, 0.505, 0.985), "fig5": (7, 0.065, 0.335, 0.035, 0.485),
        "fig6": (7, 0.345, 0.600, 0.035, 0.485), "fig7": (7, 0.045, 0.372, 0.505, 0.985),
        "fig8": (8, 0.045, 0.600, 0.035, 0.485),
    }

    def autotrim(sub, thr=0.6, pad=8):
        ink = sub[..., :3].mean(axis=2) < thr
        rr, cc = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
        if rr.size == 0 or cc.size == 0:
            return sub
        return sub[max(rr[0] - pad, 0):rr[-1] + pad + 1, max(cc[0] - pad, 0):cc[-1] + pad + 1]

    for name, (pg, y0, y1, x0, x1) in BOX.items():
        im = pages[pg]; H, W = im.shape[:2]
        sub = autotrim(im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)])
        mpimg.imsave(os.path.join(FIGS, "paper_%s.png" % name), sub)
    print("    wrote figures/paper_fig2..8.png", flush=True)


# stage -> (description, runtime hint, callable-or-list-of-(script,*args))
STAGES = [
    ("foundation", "device doping, bands, equilibrium Poisson, DD bias point, bulk hot-electron",
     "~2-4 min", [("device.py",), ("bands.py",), ("poisson.py",), ("dd.py",), ("she_bulk.py",)]),
    ("she_oneway", "one-way SHE core -> she2d_result.npz, audit, moments (Fig 3/4/7), Fig 2 dist.",
     "~2-4 min", [("run_she_clean.py",), ("audit_result.py",), ("she2d_figures.py",),
                  ("she2d_fig2.py",)]),
    ("coupled40", "self-consistent 40x34 coupled loop + convergence figure",
     "~5-10 min", [("coupled_she.py", 40, 34, 0.30), ("she2d_coupled_fig.py",)]),
    ("coupled_recal", "recalibrated 60x50 coupled loop, 50 iters toward 2 mV + convergence study",
     "~45-55 min", [("coupled_converge.py", 0.35, 50), _copy_conv_to_canonical,
                    ("she2d_converge_fig.py",)]),
    ("overlays", "Liang Fig 3/4/7 overlays, Fig 5/6, quantitative Fig 7, our-vs-paper standalones",
     "~1 min", [("she2d_liang_overlay.py",), ("she2d_liang_compare.py",),
                ("she2d_liang_fig5_6.py",), ("she2d_paper_compare_figs.py",)]),
    ("paper_crops", "crop Liang Figs 2-8 from the source PDF for the side-by-side (needs pdftoppm)",
     "~1 min", [_paper_crops]),
    ("iv", "terminal I-V: DD proxy (Fig 8) and SHE velocity-moment (face flux) vs DD",
     "~5 + ~14 min", [("she2d_iv.py",), ("she2d_iv_moment.py",)]),
]
STAGE_NAMES = [s[0] for s in STAGES]
# default 'core' run: physics figures that do not need the ~50-min recal loop
CORE = ["foundation", "she_oneway", "coupled40"]


def run_stage(name):
    desc, rt, actions = next((d, r, a) for n, d, r, a in STAGES if n == name)
    print("\n=== STAGE %s (%s) -- %s ===" % (name, rt, desc), flush=True)
    t0 = time.time()
    for act in actions:
        if callable(act):
            act()
        else:
            _run_script(*act)
    print("=== stage %s done (%.0f s) ===" % (name, time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Reproduce all figures for the Liang 1997 SHE-BTE study.")
    ap.add_argument("--full", action="store_true", help="run every stage (from scratch, ~1.5-2 h)")
    ap.add_argument("--stage", help="run a single stage by name")
    ap.add_argument("--from", dest="from_stage", help="run from this stage to the end")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--keep-going", action="store_true", help="continue if a stage fails")
    args = ap.parse_args()

    if args.list:
        print("Stages (dependency order):")
        for n, d, rt, _ in STAGES:
            tag = "  [core]" if n in CORE else ""
            print("  %-14s %-11s %s%s" % (n, rt, d, tag))
        print("\ndefault (no flag) runs: %s" % ", ".join(CORE))
        print("--full runs all; overlays/paper_crops/iv need coupled_recal first.")
        return

    if args.stage:
        todo = [args.stage]
    elif args.from_stage:
        i = STAGE_NAMES.index(args.from_stage)
        todo = STAGE_NAMES[i:]
    elif args.full:
        todo = STAGE_NAMES
    else:
        todo = CORE

    os.makedirs(FIGS, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    print("Reproducing: %s" % ", ".join(todo), flush=True)
    t0 = time.time()
    failed = []
    for name in todo:
        try:
            run_stage(name)
        except Exception as e:
            print("!! STAGE %s FAILED: %s" % (name, e), flush=True)
            failed.append(name)
            if not args.keep_going:
                raise
    print("\nAll requested stages finished in %.0f s. Failed: %s"
          % (time.time() - t0, failed or "none"), flush=True)
    print("Figures in %s ; data in %s" % (FIGS, DATA), flush=True)


if __name__ == "__main__":
    main()
