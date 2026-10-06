@echo off
rem usage: run_detached.cmd <case> <mesh> <outdir> [extra args]  (runs from project root, survives the caller)
cd /d "%~dp0..\..\..\.."
python -m pipeline.cfd.solve.run_case %1 --mesh %2 --out %3 --np 4 %4 %5 %6 %7 > "%~3\run.log" 2>&1
echo exit %errorlevel% >> "%~3\run.log"
