"""Core placement for this project's solvers (Core Ultra 7 255HX: 8 P + 12 E, no SMT).

Owner rule (2026-10-04): a job that can use all 20 cores runs on all 20; a job using fewer
gets P-cores first (best first), then E-cores.
Ranking from GetSystemCpuSetInformation (EfficiencyClass 1 = P, SchedulingClass 1 = favoured
Turbo Boost Max 3.0), measured 2026-10-04: favoured P 9, 19; other P 0, 1, 6, 7, 8, 18.
- Fluent compute ranks (fl_mpi2610.exe; also the GPU solver's single rank): one core each in
  CORE_RANKED order (P first). With 20 or more ranks nothing is pinned.
- Fluent host/cortex/launcher: the cores not used by ranks (all 20 if every core is used).
- flow5.exe (multithreaded, max_threads 20): all 20 cores.
Only these processes are touched. Usage: python pin_pcores.py [--interval 5] [--until-file PATH]
"""
from __future__ import annotations

import argparse
import os
import time

import psutil

# Fluent ranks avoid the favoured cores 9,19 and interrupt core 0,1 (laptop stutter, owner 2026-10-04)
P_RANKED = [6, 7, 8, 18, 9, 19, 0, 1]
E_CORES = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17]
CORE_RANKED = P_RANKED + E_CORES
ALL = sorted(CORE_RANKED)
RANK_NAMES = {"fl_mpi2610.exe"}
HOST_NAMES = {"fl2610.exe", "cx2610.exe", "fluent.exe"}
ALL_CORE_NAMES = {"flow5.exe"}


def cores_for(n: int) -> list[int]:
    """Best n cores: P-cores first in ranked order, then E-cores."""
    return ALL if n >= len(ALL) else CORE_RANKED[:n]


def pin_once(log=print) -> int:
    procs = list(psutil.process_iter(["name", "create_time"]))
    ranks = sorted((p for p in procs if (p.info["name"] or "").lower() in RANK_NAMES),
                   key=lambda p: (p.info["create_time"], p.pid))
    changed = 0
    used = cores_for(len(ranks))
    if len(ranks) < len(ALL):
        for proc, core in zip(ranks, used):
            changed += _set(proc, [core], log)
    else:
        for proc in ranks:
            changed += _set(proc, ALL, log)
    rest = [c for c in ALL if c not in used] if len(ranks) < len(ALL) else ALL
    for proc in procs:
        name = (proc.info["name"] or "").lower()
        if name in HOST_NAMES:
            changed += _set(proc, rest or ALL, log)
        elif name in ALL_CORE_NAMES:
            changed += _set(proc, ALL, log)
    return changed


def _set(proc: psutil.Process, cores: list[int], log) -> int:
    try:
        if sorted(proc.cpu_affinity()) != sorted(cores):
            proc.cpu_affinity(cores)
            log(f"{time.strftime('%H:%M:%S')} pinned {proc.name()} {proc.pid} -> {cores}")
            return 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--until-file")
    a = ap.parse_args()
    while not (a.until_file and os.path.exists(a.until_file)):
        pin_once(lambda s: print(s, flush=True))
        time.sleep(a.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
