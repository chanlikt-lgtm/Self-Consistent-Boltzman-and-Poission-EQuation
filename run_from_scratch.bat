@echo off
REM ===================================================================
REM  Liang 1997 SHE-BTE reproduction -- full pipeline from scratch.
REM  Phase 1 (foundation) -> coupled -> Rev 5 Anderson -> Rev 6 trio.
REM
REM  Total ~26-27 h on a single workstation (Rev 6 trio ~23 h, run
REM  strictly sequentially; the 141x85 node alone is ~12 h).
REM
REM  Logs written under data\:
REM    run_from_scratch.log        -- stage start/end timestamps (this script)
REM    richardson_trio_run.log     -- per-chunk convergence/timing/warnings (Stage 3)
REM    richardson_trio_metrics.csv -- machine-readable key metrics per chunk
REM  Tip: also capture full console output:
REM    run_from_scratch.bat > data\run_console.log 2>&1
REM
REM  Run from the repository root:  run_from_scratch.bat
REM ===================================================================
setlocal
cd /d "%~dp0"
set "RUNLOG=data\run_from_scratch.log"
if not exist data mkdir data

REM --- Python: the WindowsApps "python" is a broken stub on this box,
REM     so use the real interpreter by full path; fall back to PATH. ---
set "PY=C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"

call :stamp "=== RUN START.  Interpreter: %PY%"
"%PY%" -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)" || goto :error

call :stamp "STAGE 1/6: foundation -> one-way SHE -> coupled -> Figs 2-8 overlays (~1.5-2 h)"
"%PY%" reproduce_all.py --full || goto :error

call :stamp "STAGE 2/6: Rev 5 Anderson-accelerated coupled fixed point, 3 grids (~1.5 h)"
"%PY%" src\coupled_anderson.py 40 34 6 0.5 40 1 || goto :error
"%PY%" src\coupled_anderson.py 60 50 6 0.5 40 1 || goto :error
"%PY%" src\coupled_anderson.py 80 66 6 0.5 40 1 || goto :error
"%PY%" src\she2d_anderson_fig.py || goto :error

call :stamp "STAGE 3/6: Rev 6 controlled spatial-convergence trio (~23 h, SEQUENTIAL, continue-until-plateau)"
"%PY%" src\run_richardson_trio.py || goto :error

call :stamp "STAGE 4/6: gated Richardson fit (S_rev verdict)"
"%PY%" src\richardson_fit.py data\she2d_richardson_71x58.npz data\she2d_richardson_100x70.npz data\she2d_richardson_141x85.npz || goto :error

call :stamp "STAGE 5/6: auto-generate run-results report from THIS run's data"
"%PY%" src\gen_run_report.py || goto :error

set "PDFLATEX=C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
if exist "%PDFLATEX%" (
  call :stamp "optional: recompiling canonical report_v6_full.pdf"
  pushd report
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  "%PDFLATEX%" --enable-installer --interaction=nonstopmode report_v6_full.tex >nul 2>&1
  popd
)

call :stamp "STAGE 6/6: archive all plots + reports + data/logs into runs\<timestamp>\"
"%PY%" src\archive_run.py || goto :error

call :stamp "=== DONE. Snapshot: runs\<timestamp>\ (plots\, reports, MANIFEST.txt) ; run report: report\report_run_latest.pdf ==="
goto :eof

:stamp
echo %date% %time%  %~1
echo %date% %time%  %~1>> "%RUNLOG%"
goto :eof

:error
call :stamp "*** FAILED at the stage above (exit code %errorlevel%). See logs; fix and re-run. ***"
exit /b 1
