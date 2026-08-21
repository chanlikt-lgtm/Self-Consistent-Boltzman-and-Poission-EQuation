@echo off
REM ===================================================================
REM  Liang 1997 SHE-BTE reproduction -- full pipeline from scratch.
REM  Phase 1 (foundation) -> coupled -> Rev 5 Anderson -> Rev 6 trio.
REM
REM  Total ~26-27 h on a single workstation (Rev 6 trio ~23 h, run
REM  strictly sequentially; the 141x85 node alone is ~12 h). See the
REM  report reproducibility appendix.
REM
REM  Run from the repository root:  run_from_scratch.bat
REM ===================================================================
setlocal
cd /d "%~dp0"

REM --- Python: the WindowsApps "python" is a broken stub on this box,
REM     so use the real interpreter by full path; fall back to PATH. ---
set "PY=C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"
echo Using interpreter: %PY%
"%PY%" -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)" || goto :error

echo.
echo === STAGE 1/4: foundation -> one-way SHE -> coupled -> Figs 2-8 overlays (~1.5-2 h) ===
"%PY%" reproduce_all.py --full || goto :error

echo.
echo === STAGE 2/4: Rev 5 Anderson-accelerated coupled fixed point, 3 grids (~1.5 h) ===
"%PY%" src\coupled_anderson.py 40 34 6 0.5 40 1 || goto :error
"%PY%" src\coupled_anderson.py 60 50 6 0.5 40 1 || goto :error
"%PY%" src\coupled_anderson.py 80 66 6 0.5 40 1 || goto :error
"%PY%" src\she2d_anderson_fig.py || goto :error

echo.
echo === STAGE 3/4: Rev 6 controlled spatial-convergence trio (~23 h, SEQUENTIAL) ===
echo     71x58 -^> 100x70 -^> 141x85, each continued until the observable-plateau gate passes.
"%PY%" src\run_richardson_trio.py || goto :error

echo.
echo === STAGE 4/4: gated Richardson fit (S_rev verdict) ===
"%PY%" src\richardson_fit.py data\she2d_richardson_71x58.npz data\she2d_richardson_100x70.npz data\she2d_richardson_141x85.npz || goto :error

echo.
echo === (optional) compile the full report if MiKTeX is present ===
set "PDFLATEX=C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
if exist "%PDFLATEX%" (
  pushd report
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  popd
  echo Report compiled: report\report_v6_full.pdf
) else (
  echo MiKTeX not found; skipping report compile.
)

echo.
echo === DONE. Trio data in data\she2d_richardson_*.npz ; verdict printed above. ===
goto :eof

:error
echo.
echo *** FAILED at the stage above (exit code %errorlevel%). Fix and re-run. ***
exit /b 1
