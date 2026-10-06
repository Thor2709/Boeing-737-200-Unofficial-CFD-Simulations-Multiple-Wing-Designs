"""CLI and orchestration for one Fluent case on one production mesh."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from .conditions import (S_REF_M2, condition_for_case, reference_values)
from .fluent_io import (COURANT_RAMP, PB_DISCRETIZATION, REPORT_COMPONENTS,
                        WALL_PREFIXES, FluentBackend, launch_fluent,
                        record_fluent_pids, reset_fluent_pids,
                        write_interpolation_data)
from .search import (ASSUMED_CL_ALPHA_PER_DEG, CL_TOLERANCE,
                     MAX_ALPHA_DEG, MAX_SINGLE_POINT_STEP_DEG,
                     MIN_ALPHA_DEG, MIN_REACHABLE_CL_ALPHA_PER_DEG,
                     ITERATION_CHUNK,
                     MAX_ITERATIONS_PER_ALPHA, MAX_SECANT_STEPS,
                     MIN_FINAL_ITERATIONS, MIN_SEARCH_ITERATIONS,
                     SETTLE_WINDOW_ITERATIONS, CL_SETTLE_TOLERANCE,
                     CD_SETTLE_TOLERANCE, CM_SETTLE_TOLERANCE,
                     settle_check, secant_next)

try:
    from b737wing.config import REPO_ROOT
    PROJECT_ROOT = REPO_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]

LBF_PER_N = 4.4482216152605
DEFAULT_IH_DEG = 0.0
ACCELERATORS = frozenset(("fmg", "steer", "casm"))
FLUENT_CLOSE_TIMEOUT_SECONDS = 90
EXPORT_RETRY_ATTEMPTS = 3
EXPORT_RETRY_PAUSE_SECONDS = 30


def parse_accel(value):
    values = [part.strip().lower() for part in value.split(",")]
    if not values or any(not part for part in values):
        raise argparse.ArgumentTypeError("--accel needs a comma-separated list")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--accel entries must not repeat")
    if "none" in values:
        if values == ["none"]:
            return []
        raise argparse.ArgumentTypeError("none cannot be combined with accelerators")
    invalid = [part for part in values if part not in ACCELERATORS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "--accel entries must be from none,fmg,steer,casm")
    return values


def parse_courant_ramp(value):
    try:
        ramp = []
        for item in value.split(","):
            iteration_text, value_text = item.split(":")
            iteration = int(iteration_text.strip())
            courant = float(value_text.strip())
            if iteration <= 0 or not math.isfinite(courant) or courant <= 0:
                raise ValueError
            ramp.append((iteration, courant))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            "--courant-ramp must use positive ITER:VAL entries") from None
    if not ramp:
        raise argparse.ArgumentTypeError("--courant-ramp cannot be empty")
    if any(current[0] <= previous[0]
           for previous, current in zip(ramp, ramp[1:])):
        raise argparse.ArgumentTypeError(
            "--courant-ramp ITER values must be strictly increasing")
    if ramp[-1][0] < 6000:
        raise argparse.ArgumentTypeError(
            "--courant-ramp last ITER must be at least 6000")
    return tuple(ramp)


def _ramp_records(ramp):
    return [{"through_iteration": end, "value": value} for end, value in ramp]


def _used_courant_ramp(solver, accel, ramp):
    if solver == "gpu-pb" or "steer" in accel:
        return ()
    return tuple(ramp)


def dry_setup(case, mesh, out, processor_count=4, alpha0=None,
              fixed_alpha=None, init_data=None, ih=DEFAULT_IH_DEG,
              interp_from=None, solver="cpu-db", precision="double",
              min_final_iters=MIN_FINAL_ITERATIONS, seed_info=None,
              accel=(), courant_ramp=None, autosave_every=500,
              first_slope=None):
    mesh = Path(mesh).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    init_data = Path(init_data).expanduser().resolve() if init_data else None
    interp_from = Path(interp_from).expanduser().resolve() if interp_from else None
    condition = condition_for_case(case)
    start_alpha = alpha0 if alpha0 is not None else (8.0 if condition["condition"] == "takeoff" else 2.0)
    start_alpha = min(MAX_ALPHA_DEG, max(MIN_ALPHA_DEG, start_alpha))
    pressure_based = solver == "gpu-pb"
    accel = tuple(accel or ())
    ramp = tuple(COURANT_RAMP if courant_ramp is None else courant_ramp)
    ramp_used = _used_courant_ramp(solver, accel, ramp)
    first_slope = (first_slope if first_slope is not None else
                   seed_info.get("slope") if seed_info else None)
    first_slope = (ASSUMED_CL_ALPHA_PER_DEG if first_slope is None
                   else first_slope)
    return {"case": condition["case"], "mesh": str(mesh), "out": str(out),
            "conditions": condition, "np": processor_count,
             "alpha0_deg": start_alpha, "fixed_alpha_deg": fixed_alpha,
             "autosave_every": autosave_every,
            "init_data": str(init_data) if init_data else None,
            "interp_from": str(interp_from) if interp_from else None,
            "ih_deg": ih,
            "accel": list(accel),
            "courant_ramp": _ramp_records(ramp_used),
            "courant_ramp_control": ("solution-steering" if "steer" in accel
                                     else "driver" if not pressure_based else "solver-default"),
            "references": reference_values(condition),
            "zone_groups": {group: [prefix + "*" for prefix in prefixes]
                            for group, prefixes in WALL_PREFIXES.items()},
            "combined_zone_groups": {"total": "all wall groups",
                                     "total_ex_duct": "all wall groups except duct"},
            "solver": {"precision": precision,
                       "type": "pressure-based" if pressure_based else "density-based-implicit",
                       "energy": True, "viscous_model": "k-omega SST",
                       "density": "ideal-gas", "viscosity": "Sutherland",
                       "operating_pressure_Pa": 0,
                       "discretization": (PB_DISCRETIZATION if pressure_based
                                           else "second-order-upwind"),
                       "turbulent_intensity": 0.001,
                       "turbulent_viscosity_ratio": 10.0,
                       "courant_ramp": _ramp_records(ramp_used),
                       "residual_stop": False},
             "iteration_plan": {"search_minimum": MIN_SEARCH_ITERATIONS,
                                "alpha_bounds_deg": [MIN_ALPHA_DEG, MAX_ALPHA_DEG],
                                "unreachable_lift_slope_per_deg": MIN_REACHABLE_CL_ALPHA_PER_DEG,
                                 "final_minimum": max(min_final_iters,
                                                       SETTLE_WINDOW_ITERATIONS),
                                 "requested_final_minimum": min_final_iters,
                               "check_every": ITERATION_CHUNK,
                               "maximum_per_alpha": MAX_ITERATIONS_PER_ALPHA,
                                "max_secant_steps": MAX_SECANT_STEPS,
                                "cl_tolerance": CL_TOLERANCE,
                                 "settle_window": SETTLE_WINDOW_ITERATIONS,
                                 "assumed_cl_alpha_per_deg": ASSUMED_CL_ALPHA_PER_DEG,
                                 "first_slope_cl_per_deg": first_slope,
                                 "max_single_point_step_deg": MAX_SINGLE_POINT_STEP_DEG},
            **({"warm_start_seed": seed_info} if seed_info is not None else {})}


def read_seed(run_dir):
    """Read the last alpha result and slope evidence from a completed run."""
    source = Path(run_dir).expanduser().resolve()
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    last_by_alpha = {}
    alpha_order = []
    try:
        with (source / "history.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                alpha = float(row["alpha"])
                if alpha in last_by_alpha:
                    alpha_order.remove(alpha)
                alpha_order.append(alpha)
                last_by_alpha[alpha] = float(row["CL"])
    except OSError:
        pass
    converged = alpha_order
    if result.get("status") not in {"converged", "cl_not_matched"} and converged:
        converged = converged[:-1]
    slope = None
    if len(converged) >= 2:
        a0, a1 = converged[-2:]
        if a1 != a0:
            slope = (last_by_alpha[a1] - last_by_alpha[a0]) / (a1 - a0)
    return {"source": str(source), "alpha0_deg": float(result["alpha_deg"]),
            "slope": slope, "mesh": result.get("mesh")}


def _seed_warm_start(seed_info, mesh, init_data, interp_from):
    if not seed_info or init_data or interp_from:
        return init_data, interp_from
    source_mesh = seed_info["mesh"]
    if not source_mesh:
        raise ValueError("--seed-from result.json has no mesh; specify --init-data or --interp-from")
    same_mesh = Path(source_mesh).expanduser().resolve() == Path(mesh).expanduser().resolve()
    if same_mesh:
        return str(Path(seed_info["source"]) / "final.dat.h5"), None
    return None, seed_info["source"]


def history_columns():
    columns = ["iter", "alpha"]
    for group in (*REPORT_COMPONENTS, "total", "total_ex_duct"):
        suffix = "" if group == "total" else "_" + group
        columns.extend(key + suffix for key in ("CL", "CD", "CM"))
    return columns + ["mass_imbalance", "settle"]


def _history_row(total_iterations, alpha, sample):
    row = {"iter": total_iterations, "alpha": alpha,
           "mass_imbalance": sample["mass_imbalance"]}
    for group, reports in sample["all_reports"].items():
        suffix = "" if group == "total" else "_" + group
        for key, value in reports.items():
            row[key + suffix] = value
    row["settle"] = ""
    return row


def _advance_alpha(backend, alpha, condition, minimum, total_iterations,
                   records, state=None, history_callback=None,
                   courant_ramp=None, stop_file=None, *, settle_on="all"):
    if settle_on not in ("cl", "all"):
        raise ValueError("settle_on must be 'cl' or 'all'")
    state = state or {"iterations": 0, "samples": []}
    local_iterations = state["iterations"]
    samples = state["samples"]
    courant_ramp = COURANT_RAMP if courant_ramp is None else courant_ramp
    convergence = {"ok": False, "dcl_100": None, "dcd_100": None,
                   "mass_imbalance": None, "iterations": local_iterations,
                   "settle": None, "settle_on": settle_on}
    if (local_iterations and stop_file is not None
            and Path(stop_file).is_file()):
        settled = settle_check(samples)
        convergence.update({"ok": (settled["CL"]["settled"]
                                    if settle_on == "cl"
                                    else settled["settled"]),
                            "iterations": local_iterations,
                            "settle": settled})
        return state, "stopped_by_owner", convergence, total_iterations
    backend.set_alpha(alpha)
    while local_iterations < MAX_ITERATIONS_PER_ALPHA:
        count = min(ITERATION_CHUNK, MAX_ITERATIONS_PER_ALPHA - local_iterations)
        courant = next(value for boundary, value in courant_ramp
                       if local_iterations + count <= boundary)
        if (getattr(backend, "solver_name", "cpu-db") != "gpu-pb"
                and not getattr(backend, "steering_active", False)):
            backend.set_courant(courant)
        backend.iterate(count)
        local_iterations += count
        total_iterations += count
        sample = dict(backend.sample())
        sample["iteration"] = local_iterations
        samples.append(sample)
        row = _history_row(total_iterations, alpha, sample)
        settled = None
        if local_iterations >= max(minimum, SETTLE_WINDOW_ITERATIONS):
            settled = settle_check(samples)
            convergence = {
                "ok": (settled["CL"]["settled"] if settle_on == "cl"
                       else settled["settled"]), "dcl_100": None,
                "dcd_100": None,
                "mass_imbalance": abs(float(sample["mass_imbalance"])),
                "iterations": local_iterations, "settle": settled,
                "settle_on": settle_on}
            row["settle"] = json.dumps(settled, sort_keys=True)
        stop_requested = stop_file is not None and Path(stop_file).is_file()
        if stop_requested:
            settled = settled or settle_check(samples)
            convergence = {
                "ok": (settled["CL"]["settled"] if settle_on == "cl"
                       else settled["settled"]), "dcl_100": None,
                "dcd_100": None,
                "mass_imbalance": abs(float(sample["mass_imbalance"])),
                "iterations": local_iterations, "settle": settled,
                "settle_on": settle_on}
            row["settle"] = json.dumps(settled, sort_keys=True)
        records.append(row)
        if history_callback is not None:
            history_callback(row)
        state = {"iterations": local_iterations, "samples": samples}
        if stop_requested:
            return state, "stopped_by_owner", convergence, total_iterations
        if convergence["ok"]:
            return state, "converged", convergence, total_iterations
    return state, "not_converged", convergence, total_iterations


def _solve_alpha(backend, target, alpha0, fixed_alpha, records,
                 history_callback=None, min_final_iters=MIN_FINAL_ITERATIONS,
                 initial_slope=None, courant_ramp=None, stop_file=None):
    total_iterations = 0
    if fixed_alpha is not None:
        if not MIN_ALPHA_DEG <= fixed_alpha <= MAX_ALPHA_DEG:
            raise ValueError(
                f"fixed alpha must be within [{MIN_ALPHA_DEG}, {MAX_ALPHA_DEG}] deg")
        state, status, convergence, total_iterations = _advance_alpha(
            backend, fixed_alpha, target, min_final_iters,
            total_iterations, records, history_callback=history_callback,
            courant_ramp=courant_ramp, stop_file=stop_file,
            settle_on="all")
        return fixed_alpha, state, status, convergence, total_iterations

    points = []
    slope_replacements = []
    alpha = min(MAX_ALPHA_DEG, max(MIN_ALPHA_DEG, float(alpha0)))
    state = None
    best_point = None
    status = "cl_not_matched"
    convergence = {}
    steps = 0
    while True:
        state, point_status, convergence, total_iterations = _advance_alpha(
            backend, alpha, target, MIN_SEARCH_ITERATIONS,
            total_iterations, records, history_callback=history_callback,
            courant_ramp=courant_ramp, stop_file=stop_file,
            settle_on="cl")
        if point_status == "stopped_by_owner":
            convergence["slope_replacements"] = slope_replacements
            return alpha, state, point_status, convergence, total_iterations
        if point_status == "not_converged":
            convergence["slope_replacements"] = slope_replacements
            return alpha, state, point_status, convergence, total_iterations
        cl = state["samples"][-1]["CL"]
        points.append((alpha, cl))
        if best_point is None or cl > best_point[1]:
            best_point = (alpha, cl, state, convergence)
        if abs(cl - target) <= CL_TOLERANCE:
            state, point_status, convergence, total_iterations = _advance_alpha(
                backend, alpha, target, min_final_iters,
                total_iterations, records, state, history_callback,
                courant_ramp, stop_file, settle_on="all")
            if point_status == "stopped_by_owner":
                convergence["slope_replacements"] = slope_replacements
                return alpha, state, point_status, convergence, total_iterations
            if point_status == "not_converged":
                convergence["slope_replacements"] = slope_replacements
                return alpha, state, point_status, convergence, total_iterations
            cl = state["samples"][-1]["CL"]
            points[-1] = (alpha, cl)
            if best_point is None or cl > best_point[1]:
                best_point = (alpha, cl, state, convergence)
            if abs(cl - target) <= CL_TOLERANCE:
                convergence["slope_replacements"] = slope_replacements
                if stop_file is not None and Path(stop_file).is_file():
                    return (alpha, state, "stopped_by_owner", convergence,
                            total_iterations)
                return alpha, state, "converged", convergence, total_iterations
        if stop_file is not None and Path(stop_file).is_file():
            convergence["slope_replacements"] = slope_replacements
            return alpha, state, "stopped_by_owner", convergence, total_iterations
        if len(points) >= 2:
            (previous_alpha, previous_cl), (current_alpha, current_cl) = points[-2:]
            if current_alpha != previous_alpha:
                slope = (current_cl - previous_cl) / (current_alpha - previous_alpha)
                if (current_cl < target
                        and slope <= MIN_REACHABLE_CL_ALPHA_PER_DEG):
                    best_alpha, best_cl, best_state, best_convergence = best_point
                    best_sample = best_state["samples"][-1]
                    convergence["search"] = {
                        "status": "cl_unreachable",
                        "selected_point": "current_stalled_field",
                        "alpha_deg": alpha,
                        "CL": state["samples"][-1]["CL"],
                    }
                    convergence["diagnostic_best_settled_point"] = {
                        "alpha_deg": best_alpha, "CL": best_sample["CL"],
                        "CD": best_sample["CD"], "CM": best_sample["CM"],
                        "iterations": best_state["iterations"],
                        "search_status": best_convergence.get("status"),
                    }
                    convergence["slope_replacements"] = slope_replacements
                    return alpha, state, "cl_unreachable", convergence, total_iterations
        if steps >= MAX_SECANT_STEPS:
            status = "cl_not_matched"
            break
        candidate = secant_next(
            points, target, initial_slope, settled_points=points,
            slope_replacements=slope_replacements)
        if candidate is None:
            status = "cl_not_matched"
            break
        alpha, state = candidate, None
        steps += 1
    final_minimum = max(min_final_iters, SETTLE_WINDOW_ITERATIONS)
    if (state["iterations"] < final_minimum
            or convergence.get("settle_on") != "all"):
        state, point_status, convergence, total_iterations = _advance_alpha(
            backend, alpha, target, min_final_iters,
            total_iterations, records, state, history_callback,
            courant_ramp, stop_file, settle_on="all")
        if point_status == "stopped_by_owner":
            convergence["slope_replacements"] = slope_replacements
            return alpha, state, point_status, convergence, total_iterations
        if point_status == "not_converged":
            status = point_status
    convergence["slope_replacements"] = slope_replacements
    return alpha, state, status, convergence, total_iterations


def build_result(case, mesh, processor_count, condition, alpha, sample,
                 iterations, wall_time_s, status, ih_deg, cells=None,
                 convergence=None, warm_start=None, solver="cpu-db",
                 precision="double", min_final_iters=MIN_FINAL_ITERATIONS,
                 solver_audit=None, exports=None, warm_start_seed=None,
                 accel=(), courant_ramp=(), courant_ramp_control="driver",
                 solver_settings="solver_settings.json",
                  solver_transcript_available=None,
                  gpu_native_solver_active=None,
                  first_slope=ASSUMED_CL_ALPHA_PER_DEG):
    q = 0.5 * condition["density_kg_m3"] * condition["velocity_m_s"] ** 2
    half_area = S_REF_M2 / 2.0
    lift_half = sample["CL"] * q * half_area
    drag_half = sample["CD"] * q * half_area
    lift = 2.0 * lift_half
    drag = 2.0 * drag_half
    cd_ex = sample["CD_ex_duct"]
    cl_ex = sample["all_reports"]["total_ex_duct"]["CL"]
    result = {"case": case, "mesh": str(mesh), "cells": cells,
            "np": processor_count, "alpha_deg": alpha,
            "CL": sample["CL"], "CD": sample["CD"], "CM": sample["CM"],
            "CD_ex_duct": cd_ex, "components": sample["components"],
            "lift_half_N": lift_half, "drag_half_N": drag_half,
            "lift_N": lift, "drag_N": drag,
            "lift_lbf": lift / LBF_PER_N, "drag_lbf": drag / LBF_PER_N,
            "L_over_D": sample["CL"] / sample["CD"] if sample["CD"] else None,
            "L_over_D_ex_duct": cl_ex / cd_ex if cd_ex else None,
            "iterations": iterations, "wall_time_s": wall_time_s,
            "s_per_iter": wall_time_s / iterations if iterations else None,
             "status": status,
             "converged": status == "converged",
             "convergence_numbers": {"cl_error": abs(sample["CL"] - condition["target_cl"]),
                                    "dcl_100": (convergence or {}).get("dcl_100"),
                                    "dcd_100": (convergence or {}).get("dcd_100"),
                                    "settle": (convergence or {}).get("settle"),
                                    "settle_on": (convergence or {}).get("settle_on"),
                                    "slope_replacements": (convergence or {}).get(
                                        "slope_replacements", []),
                                    "mass_imbalance": sample["mass_imbalance"],
                                    "final_alpha_iterations": (convergence or {}).get("iterations")},
             "ih_deg": ih_deg, "warm_start": warm_start,
             "solver": solver, "precision": precision,
             "solver_settings": solver_settings,
             "min_final_iterations": min_final_iters,
             "final_iteration_rule": {
                 "window_iterations": SETTLE_WINDOW_ITERATIONS,
                 "minimum_iterations": max(min_final_iters,
                                            SETTLE_WINDOW_ITERATIONS),
                 "tolerances": {"CL": CL_SETTLE_TOLERANCE,
                                "CD": CD_SETTLE_TOLERANCE,
                                "CM": CM_SETTLE_TOLERANCE}},
             "first_slope_cl_per_deg": first_slope,
            "accel": list(accel), "courant_ramp": _ramp_records(courant_ramp),
            "courant_ramp_control": courant_ramp_control}
    if solver_audit is not None:
        result["solver_audit"] = solver_audit
    if solver == "gpu-pb":
        result["solver_transcript_available"] = solver_transcript_available
        result["gpu_native_solver_active"] = gpu_native_solver_active
        if solver_audit is None:
            result["solver_audit"] = "unavailable"
    if (convergence or {}).get("search") is not None:
        result["cl_search"] = convergence["search"]
    if (convergence or {}).get("diagnostic_best_settled_point") is not None:
        result["diagnostic_best_settled_point"] = convergence[
            "diagnostic_best_settled_point"]
    if (convergence or {}).get("gpu_qualification_reason") is not None:
        result["gpu_qualification_reason"] = convergence[
            "gpu_qualification_reason"]
    if exports is not None:
        result["exports"] = exports
    if warm_start_seed is not None:
        result["warm_start"] = {"status": warm_start,
                                 "seed_source": warm_start_seed["source"],
                                 "slope": warm_start_seed["slope"]}
    return result


def run_solve(backend, case, mesh, out, processor_count=4, alpha0=None,
              fixed_alpha=None, init_data=None, ih=DEFAULT_IH_DEG,
              interp_from=None, warm_start_file=None, warm_start_error=None,
              solver="cpu-db", precision="double",
              min_final_iters=MIN_FINAL_ITERATIONS, initial_slope=None,
              warm_start_seed=None, accel=(), courant_ramp=None,
              autosave_every=500, first_slope=None, stop_file=None):
    accel = tuple(accel or ())
    if accel and solver != "cpu-db":
        raise ValueError("--accel is only supported with --solver cpu-db")
    if not isinstance(autosave_every, int) or autosave_every < 0:
        raise ValueError("--autosave-every must be a non-negative integer")
    courant_ramp = tuple(COURANT_RAMP if courant_ramp is None else courant_ramp)
    courant_ramp_used = _used_courant_ramp(solver, accel, courant_ramp)
    mesh = Path(mesh).expanduser().resolve()
    output_dir = Path(out).expanduser().resolve()
    init_data = Path(init_data).expanduser().resolve() if init_data else None
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_slope = first_slope if first_slope is not None else initial_slope
    first_slope = (ASSUMED_CL_ALPHA_PER_DEG if initial_slope is None
                   else initial_slope)
    stop_file = Path(stop_file) if stop_file is not None else output_dir / "STOP"
    condition = condition_for_case(case)
    start_alpha = alpha0 if alpha0 is not None else (8.0 if condition["condition"] == "takeoff" else 2.0)
    started = time.time()
    interp_from = Path(interp_from).expanduser().resolve() if interp_from else None
    warm_start_file = (Path(warm_start_file).expanduser().resolve()
                       if warm_start_file else None)
    if interp_from and init_data:
        warm_start_error = "--interp-from cannot be combined with --init-data"
    elif interp_from and warm_start_file is None and warm_start_error is None:
        warm_start_error = "interpolation file was not prepared"
    records = []
    backend.solver_name = solver
    backend.steering_active = "steer" in accel
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=history_columns())
        writer.writeheader()
        target.flush()
        if interp_from:
            backend.configure(
                str(mesh), condition, str(init_data) if init_data else None,
                warm_start_file=str(warm_start_file) if warm_start_file else None,
                warm_start_source=str(interp_from),
                warm_start_error=warm_start_error,
                initial_courant=courant_ramp[0][1])
        else:
            backend.configure(str(mesh), condition, str(init_data) if init_data else None,
                              initial_courant=courant_ramp[0][1])
        backend.configure_autosave(output_dir, autosave_every)
        settings_path = output_dir / "solver_settings.json"
        backend.write_solver_settings(settings_path, "after_setup")

        cold_hybrid_start = not init_data and not interp_from and not warm_start_file
        if "fmg" in accel and cold_hybrid_start:
            backend.initialize_fmg()
        if "steer" in accel:
            backend.enable_solution_steering()
        if "casm" in accel:
            backend.enable_casm()

        def write_history(row):
            writer.writerow(row)
            target.flush()

        alpha, state, status, convergence, iterations = _solve_alpha(
            backend, condition["target_cl"], start_alpha, fixed_alpha,
            records, write_history, min_final_iters, initial_slope,
            courant_ramp_used or courant_ramp, stop_file)
        backend.write_solver_settings(settings_path, "after_last_iteration")
        audit = None
        transcript_available = None
        native_gpu_active = None
        if solver == "gpu-pb":
            audit = (backend.solver_audit(output_dir)
                     if hasattr(backend, "solver_audit") else "unavailable")
            transcript_available = audit != "unavailable"
            if hasattr(backend, "solver_transcript_available"):
                transcript_available = backend.solver_transcript_available
            native_gpu_active = getattr(backend, "native_gpu_solver_active", None)
            if audit != [] or native_gpu_active is not True:
                status = "audit_failed"
                audit_reasons = []
                if audit != []:
                    audit_reasons.append("GPU solver audit is missing or non-empty")
                if native_gpu_active is not True:
                    audit_reasons.append(
                        "native GPU solver activation is missing or unconfirmed")
                convergence["gpu_qualification_reason"] = "; ".join(audit_reasons)
    wall_time = time.time() - started
    final_sample = state["samples"][-1]
    backend.save_final(str(output_dir))
    exports = {}
    result = build_result(condition["case"], mesh, processor_count, condition,
                          alpha, final_sample, iterations, round(wall_time, 3),
                          status, ih, getattr(backend, "cells", None), convergence,
                          getattr(backend, "warm_start_status", None), solver,
                          precision, min_final_iters,
                          audit, exports,
                          warm_start_seed, accel, courant_ramp_used,
                          "solution-steering" if "steer" in accel else
                          "driver" if solver == "cpu-db" else "solver-default",
                           settings_path.name, transcript_available,
                           native_gpu_active, first_slope)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    export_failures = False
    walls_path = str(output_dir / "walls.csv")
    for attempt in range(EXPORT_RETRY_ATTEMPTS):
        try:
            backend.export_walls(walls_path)
            exports["walls"] = walls_path
            break
        except Exception as exc:
            if attempt + 1 == EXPORT_RETRY_ATTEMPTS:
                export_failures = True
                outcome = "skipped" if solver == "gpu-pb" else "failed"
                exports["walls"] = f"{outcome}: {type(exc).__name__}: {exc}"
            else:
                time.sleep(EXPORT_RETRY_PAUSE_SECONDS)

    ensight_path = str(output_dir / "ensight")
    for attempt in range(EXPORT_RETRY_ATTEMPTS):
        try:
            backend.export_ensight(ensight_path)
            exports["ensight"] = ensight_path
            break
        except Exception as exc:
            if attempt + 1 == EXPORT_RETRY_ATTEMPTS:
                export_failures = True
                outcome = "skipped" if solver == "gpu-pb" else "failed"
                exports["ensight"] = f"{outcome}: {type(exc).__name__}: {exc}"
            else:
                time.sleep(EXPORT_RETRY_PAUSE_SECONDS)

    result["export_status"] = "incomplete" if export_failures else "complete"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _wait_and_kill_fluent_pids(output_dir):
    pid_path = Path(output_dir) / "fluent_pids.json"
    try:
        recorded = json.loads(pid_path.read_text(encoding="utf-8"))["pids"]
        if not isinstance(recorded, list):
            return False, []
        pids = sorted({pid for pid in recorded
                       if isinstance(pid, int) and not isinstance(pid, bool)
                       and pid > 0})
    except (OSError, ValueError, KeyError, TypeError):
        return False, []

    deadline = time.monotonic() + FLUENT_CLOSE_TIMEOUT_SECONDS
    while True:
        active = [pid for pid in pids if psutil.pid_exists(pid)]
        if not active:
            return True, []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.25, remaining))

    killed = set()
    kill_failed = False
    targets = {}
    for pid in active:
        try:
            root = psutil.Process(pid)
            for process in (*root.children(recursive=True), root):
                targets[process.pid] = process
        except psutil.NoSuchProcess:
            continue
        except psutil.Error:
            kill_failed = True
    for pid, process in targets.items():
        try:
            process.kill()
            killed.add(pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.Error:
            kill_failed = True
    if targets:
        try:
            _, alive = psutil.wait_procs(list(targets.values()), timeout=5)
        except psutil.Error:
            kill_failed = True
        else:
            if alive:
                kill_failed = True
    return not killed and not kill_failed, sorted(killed)


def _write_close_result(output_dir, result, close_clean, killed_pids):
    if result is None:
        return
    result["close_clean"] = close_clean
    result["killed_pids"] = killed_pids
    (Path(output_dir) / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


def _start_metrics(output_dir):
    stop = output_dir / ("metrics-" + uuid.uuid4().hex + ".stop")
    process = subprocess.Popen([sys.executable, "-m", "b737wing.tools.metrics_logger",
                                "--out", str(output_dir / "metrics.csv"),
                                "--interval", "5", "--until-file", str(stop)],
                               cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    return process, stop


def _stop_metrics(process, stop):
    stop.write_text("stop", encoding="utf-8")
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        stop.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--solver", choices=["cpu-db", "gpu-pb"], default="cpu-db")
    parser.add_argument("--accel", type=parse_accel, default=(), metavar="LIST")
    parser.add_argument("--courant-ramp", type=parse_courant_ramp, metavar="RAMP")
    parser.add_argument("--precision", choices=["single", "double"])
    parser.add_argument("--alpha0", type=float)
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--init-data")
    parser.add_argument("--interp-from", metavar="RUN_DIR")
    parser.add_argument("--seed-from", metavar="RUN_DIR")
    parser.add_argument("--min-final-iters", type=int, default=MIN_FINAL_ITERATIONS)
    parser.add_argument("--first-slope", type=float, metavar="CL_PER_DEG")
    parser.add_argument("--autosave-every", type=int, default=500)
    parser.add_argument("--ih", type=float, default=DEFAULT_IH_DEG)
    parser.add_argument("--dry", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.accel and args.solver != "cpu-db":
        raise SystemExit("--accel is only supported with --solver cpu-db")
    if args.np <= 0:
        raise SystemExit("--np must be positive")
    if args.min_final_iters <= 0:
        raise SystemExit("--min-final-iters must be positive")
    if (args.first_slope is not None
            and (not math.isfinite(args.first_slope) or args.first_slope <= 0)):
        raise SystemExit("--first-slope must be finite and positive")
    if args.autosave_every < 0:
        raise SystemExit("--autosave-every must be non-negative")
    args.precision = args.precision or ("single" if args.solver == "gpu-pb" else "double")
    if args.solver == "gpu-pb":
        args.np = 1
    args.mesh = str(Path(args.mesh).expanduser().resolve())
    args.out = str(Path(args.out).expanduser().resolve())
    if args.init_data:
        args.init_data = str(Path(args.init_data).expanduser().resolve())
    if args.interp_from:
        args.interp_from = str(Path(args.interp_from).expanduser().resolve())
    seed_info = read_seed(args.seed_from) if args.seed_from else None
    initial_slope = args.first_slope
    if initial_slope is None:
        initial_slope = seed_info.get("slope") if seed_info else None
    first_slope = (ASSUMED_CL_ALPHA_PER_DEG if initial_slope is None
                   else initial_slope)
    if seed_info:
        if args.alpha0 is None:
            args.alpha0 = seed_info["alpha0_deg"]
        args.init_data, args.interp_from = _seed_warm_start(
            seed_info, args.mesh, args.init_data, args.interp_from)
    setup = dry_setup(args.case, args.mesh, args.out, args.np, args.alpha0,
                      args.fixed_alpha, args.init_data, args.ih, args.interp_from,
                             args.solver, args.precision, args.min_final_iters, seed_info,
                             args.accel, args.courant_ramp, args.autosave_every,
                             first_slope)
    if args.dry:
        print(json.dumps(setup, indent=2))
        return 0
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_fluent_pids(output_dir)
    metrics, stop = _start_metrics(output_dir)
    backend = None
    session = None
    result = None
    try:
        warm_start_file = None
        warm_start_error = None
        if args.interp_from:
            if args.init_data:
                warm_start_error = "--interp-from cannot be combined with --init-data"
            else:
                try:
                    warm_start_file = write_interpolation_data(
                        args.interp_from, output_dir, args.np)
                except Exception as exc:
                    warm_start_error = f"{type(exc).__name__}: {exc}"
        transcript_baseline = {path.name for path in output_dir.glob("fluent-*.trn")}
        session = launch_fluent(args.np, str(output_dir), args.solver, args.precision)
        record_fluent_pids(output_dir, session)
        backend = FluentBackend(session)
        backend.set_transcript_baseline(transcript_baseline)
        result = run_solve(backend, args.case, args.mesh, output_dir, args.np,
                           args.alpha0, args.fixed_alpha, args.init_data, args.ih,
                           args.interp_from, warm_start_file, warm_start_error,
                            solver=args.solver, precision=args.precision,
                            min_final_iters=args.min_final_iters,
                             initial_slope=initial_slope,
                             warm_start_seed=seed_info, accel=args.accel,
                             courant_ramp=args.courant_ramp,
                             autosave_every=args.autosave_every,
                             first_slope=args.first_slope)
        print(json.dumps({"status": result["status"], "alpha_deg": result["alpha_deg"],
                          "CL": result["CL"], "CD": result["CD"]}))
        return 0 if result["status"] == "converged" else 2
    finally:
        close_clean = True
        try:
            try:
                if backend is not None:
                    backend.close()
                elif session is not None:
                    session.exit()
            except BaseException:
                close_clean = False
                raise
        finally:
            try:
                try:
                    pids_clean, killed_pids = _wait_and_kill_fluent_pids(output_dir)
                except Exception:
                    close_clean = False
                    killed_pids = []
                else:
                    close_clean = close_clean and pids_clean
                _write_close_result(output_dir, result, close_clean, killed_pids)
            finally:
                _stop_metrics(metrics, stop)


if __name__ == "__main__":
    raise SystemExit(main())
