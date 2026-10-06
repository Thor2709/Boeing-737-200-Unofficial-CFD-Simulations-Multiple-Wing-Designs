"""Launch a CFD chain in a detached process and write its process id."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import chain


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="JSON chain plan")
    parser.add_argument("--pid-file", help="path for chain.pid")
    return parser


def launch(plan_file, pid_file=None):
    root = chain.PROJECT_ROOT
    plan_path = Path(plan_file).expanduser().resolve()
    plan, _ = chain._plan_data(plan_path, root)
    log_path = chain._path(plan["log"], root)
    pid_path = Path(pid_file).expanduser().resolve() if pid_file else log_path.parent / "chain.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "b737wing.cfd.solve.chain", str(plan_path)]
    creationflags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                     | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    options = {"cwd": str(root), "stdin": subprocess.DEVNULL,
               "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        options["creationflags"] = creationflags
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(command, **options)
    pid_path.write_text(f"{process.pid}\n", encoding="ascii")
    return process.pid


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        launch(args.plan, args.pid_file)
    except (OSError, ValueError, KeyError) as exc:
        print(f"chain_detached: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
