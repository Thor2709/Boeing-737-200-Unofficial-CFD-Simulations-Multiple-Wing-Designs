"""Run a fixed-alpha polar sweep with one configured Fluent solve session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from . import run_case
from .conditions import condition_for_case
from .fluent_io import (COURANT_RAMP, FluentBackend, launch_fluent,
                        write_interpolation_data)

POLAR_COLUMNS = ("alpha", "CL", "CD", "CM", "CD_ex_duct", "L_over_D",
                 "iterations", "status")


def parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("--alphas must be comma-separated numbers") from None
    if not alphas or any(not math.isfinite(alpha) for alpha in alphas):
        raise argparse.ArgumentTypeError("--alphas must contain finite values")
    return alphas


def run_sweep(backend, case, mesh, out, alphas, processor_count=4,
              init_data=None, interp_from=None, warm_start_file=None,
              warm_start_error=None, solver="cpu-db", precision="double"):
    alphas = [float(alpha) for alpha in alphas]
    if not alphas or any(not math.isfinite(alpha) for alpha in alphas):
        raise ValueError("alphas must contain finite values")
    mesh = Path(mesh).expanduser().resolve()
    output_dir = Path(out).expanduser().resolve()
    init_data = Path(init_data).expanduser().resolve() if init_data else None
    interp_from = Path(interp_from).expanduser().resolve() if interp_from else None
    output_dir.mkdir(parents=True, exist_ok=True)
    condition = condition_for_case(case)
    courant_ramp = tuple(COURANT_RAMP)
    courant_ramp_used = run_case._used_courant_ramp(solver, (), courant_ramp)
    backend.solver_name = solver
    backend.steering_active = False
    if interp_from and init_data:
        warm_start_error = "--interp-from cannot be combined with --init-data"
    elif interp_from and warm_start_file is None and warm_start_error is None:
        try:
            warm_start_file = write_interpolation_data(
                interp_from, output_dir, processor_count)
        except Exception as exc:
            warm_start_error = f"{type(exc).__name__}: {exc}"

    started = time.time()
    points = []
    records = []
    total_iterations = 0
    history_path = output_dir / "history.csv"
    polar_path = output_dir / "polar.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_stream, \
            polar_path.open("w", newline="", encoding="utf-8") as polar_stream:
        history_writer = csv.DictWriter(
            history_stream, fieldnames=run_case.history_columns())
        polar_writer = csv.DictWriter(polar_stream, fieldnames=POLAR_COLUMNS)
        history_writer.writeheader()
        polar_writer.writeheader()
        history_stream.flush()
        polar_stream.flush()

        if interp_from:
            backend.configure(
                str(mesh), condition, str(init_data) if init_data else None,
                warm_start_file=str(warm_start_file) if warm_start_file else None,
                warm_start_source=str(interp_from),
                warm_start_error=warm_start_error,
                initial_courant=courant_ramp[0][1])
        else:
            backend.configure(
                str(mesh), condition, str(init_data) if init_data else None,
                initial_courant=courant_ramp[0][1])
        audit = (backend.solver_audit(output_dir)
                 if solver == "gpu-pb" and hasattr(backend, "solver_audit") else [])

        def write_history(row):
            history_writer.writerow(row)
            history_stream.flush()

        for alpha in alphas:
            state, status, convergence, total_iterations = run_case._advance_alpha(
                backend, alpha, condition, run_case.MIN_SEARCH_ITERATIONS,
                total_iterations, records, history_callback=write_history,
                courant_ramp=courant_ramp_used or courant_ramp)
            sample = state["samples"][-1]
            point = {"alpha": alpha, "CL": sample["CL"], "CD": sample["CD"],
                     "CM": sample["CM"], "CD_ex_duct": sample["CD_ex_duct"],
                     "L_over_D": sample["CL"] / sample["CD"] if sample["CD"] else None,
                     "iterations": state["iterations"], "status": status}
            polar_writer.writerow(point)
            polar_stream.flush()
            points.append(point)

    exports = {}
    try:
        backend.export_walls(str(output_dir / "walls.csv"))
        exports["walls"] = str(output_dir / "walls.csv")
    except Exception as exc:
        if solver != "gpu-pb":
            raise
        exports["walls"] = f"skipped: {type(exc).__name__}: {exc}"
    backend.save_final(str(output_dir))
    result = {"case": condition["case"], "mesh": str(mesh), "np": processor_count,
              "solver": solver, "precision": precision, "solver_audit": audit,
              "alphas_deg": alphas, "points": points, "iterations": total_iterations,
              "wall_time_s": round(time.time() - started, 3), "exports": exports,
              "status": "converged" if all(point["status"] == "converged"
                                            for point in points) else "not_converged"}
    (output_dir / "sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alphas", required=True, type=parse_alphas)
    parser.add_argument("--solver", choices=["cpu-db", "gpu-pb"], default="cpu-db")
    parser.add_argument("--precision", choices=["single", "double"])
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--init-data")
    parser.add_argument("--interp-from", metavar="RUN_DIR")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.np <= 0:
        raise SystemExit("--np must be positive")
    args.precision = args.precision or ("single" if args.solver == "gpu-pb" else "double")
    if args.solver == "gpu-pb":
        args.np = 1
    args.mesh = str(Path(args.mesh).expanduser().resolve())
    args.out = str(Path(args.out).expanduser().resolve())
    if args.init_data:
        args.init_data = str(Path(args.init_data).expanduser().resolve())
    if args.interp_from:
        args.interp_from = str(Path(args.interp_from).expanduser().resolve())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    warm_start_file = None
    warm_start_error = None
    if args.interp_from:
        if args.init_data:
            warm_start_error = "--interp-from cannot be combined with --init-data"
        else:
            try:
                warm_start_file = write_interpolation_data(
                    args.interp_from, args.out, args.np)
            except Exception as exc:
                warm_start_error = f"{type(exc).__name__}: {exc}"
    session = launch_fluent(args.np, args.out, args.solver, args.precision)
    backend = FluentBackend(session)
    try:
        result = run_sweep(backend, args.case, args.mesh, args.out, args.alphas,
                           args.np, args.init_data, args.interp_from,
                           warm_start_file, warm_start_error,
                           solver=args.solver, precision=args.precision)
        print(json.dumps({"status": result["status"], "points": len(result["points"]),
                          "iterations": result["iterations"]}))
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
