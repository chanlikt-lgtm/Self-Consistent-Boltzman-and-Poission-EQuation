"""
archive_run.py -- snapshot every plot (and the run's key products) into a timestamped folder.

At the end of a run this copies all generated figures into runs/<YYYY-MM-DD_HHMMSS>/plots/ so each
run's plots are preserved instead of overwritten, alongside the run-results report, the trio data,
and the convergence logs. Writes a MANIFEST.txt (timestamp, git commit, file list).
Usage: python archive_run.py [label]      (optional short label appended to the folder name)
"""
import os, sys, glob, shutil, subprocess, datetime

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); REP = os.path.join(ROOT, "report"); DATA = os.path.join(ROOT, "data")

def git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "?"

def copy_into(dst, patterns):
    n = 0
    for pat in patterns:
        for src in glob.glob(pat):
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, os.path.basename(src))); n += 1
    return n

def main():
    label = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run = os.path.join(ROOT, "runs", ts + label)
    plots = os.path.join(run, "plots")
    os.makedirs(plots, exist_ok=True)

    n_png = copy_into(plots, [os.path.join(FIG, "*.png"), os.path.join(FIG, "*.pdf")])
    n_rep = copy_into(run, [os.path.join(REP, "report_run_latest.pdf"),
                            os.path.join(REP, "report_v6_full.pdf")])
    n_dat = copy_into(run, [os.path.join(DATA, "she2d_richardson_*.npz"),
                            os.path.join(DATA, "richardson_trio_run.log"),
                            os.path.join(DATA, "richardson_trio_metrics.csv"),
                            os.path.join(DATA, "richardson_*.log"),
                            os.path.join(DATA, "run_from_scratch.log")])

    with open(os.path.join(run, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("run snapshot: %s%s\n" % (ts, label))
        f.write("generated:    %s\n" % datetime.datetime.now().isoformat(timespec="seconds"))
        f.write("git commit:   %s (%s)\n" % (git("rev-parse", "--short", "HEAD"),
                                             "dirty" if git("status", "--porcelain") else "clean"))
        f.write("plots:        %d  (plots/)\n" % n_png)
        f.write("reports:      %d\n" % n_rep)
        f.write("data/logs:    %d\n\n" % n_dat)
        f.write("plots/:\n")
        for p in sorted(os.listdir(plots)):
            f.write("  %s\n" % p)

    print("archived %d plots + %d reports + %d data/logs -> %s"
          % (n_png, n_rep, n_dat, os.path.relpath(run, ROOT)))

if __name__ == "__main__":
    main()
