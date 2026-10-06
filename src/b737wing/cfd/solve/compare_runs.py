"""Compare two Fluent runs and evaluate warm-start and GPU-only gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from b737wing.post import sections

WARM_START_MAX_ABS_DCD_COUNTS = 0.5
WARM_START_MAX_ABS_DCL = 0.001
WARM_START_MAX_ABS_DCM = 0.002
GPU_ONLY_MAX_ABS_DCD_COUNTS = 2.0
GPU_ONLY_MAX_ABS_DCL = 0.005
GPU_ONLY_MAX_ABS_SHOCK_X_C = 0.02
SHOCK_ETAS = (0.30, 0.50, 0.70)
SHOCK_X_C_MIN = 0.05
SHOCK_X_C_MAX = 0.95


def locate_shock_x_c(x_c, cp):
    """Return the start x/c of the strongest positive Cp recovery after suction minimum."""
    import numpy as np

    x = np.asarray(x_c, dtype=float)
    pressure = np.asarray(cp, dtype=float)
    valid = np.isfinite(x) & np.isfinite(pressure)
    x, pressure = x[valid], pressure[valid]
    order = np.argsort(x)
    x, pressure = x[order], pressure[order]
    if len(x) < 3 or np.any(np.diff(x) <= 0):
        return "absent"

    suction_minimum = int(np.argmin(pressure))
    gradients = np.diff(pressure) / np.diff(x)
    candidates = [index for index, gradient in enumerate(gradients)
                  if index >= suction_minimum
                  and x[index] >= SHOCK_X_C_MIN
                  and x[index] <= SHOCK_X_C_MAX
                  and gradient > 0.0]
    if not candidates:
        return "absent"
    steepest = max(candidates, key=lambda index: gradients[index])
    return float(x[steepest])


def _read_result(run_dir):
    directory = Path(run_dir).expanduser().resolve()
    with (directory / "result.json").open("r", encoding="utf-8") as stream:
        return directory, json.load(stream)


def _shock_positions_for_run(run_dir, result, mesh_path):
    from b737wing.cfd.solve.fluent_io import resolve_zone_groups
    from b737wing.post.case_post import (_component_meshes, _map_wall_data,
                                         _wall_rows)
    from b737wing.post.mesh_view import read_walls

    grouped = _wall_rows(Path(run_dir) / "walls.csv")
    wall_meshes, zone_groups = read_walls(mesh_path, mirror=False, return_groups=True)
    zone_groups = resolve_zone_groups(zone_groups["total"] + zone_groups["farfield"])
    mapped = _map_wall_data(wall_meshes, grouped, zone_groups)
    wing = _component_meshes(mapped, zone_groups)["wing"]
    alpha = float(result.get("alpha_deg", 0.0))
    return {eta: locate_shock_x_c(
                (section := sections.section_at(
                    wing, eta * sections.SEMISPAN_M, alpha, eta))["upper"]["x_c"],
                section["upper"]["cp"])
            for eta in SHOCK_ETAS}


def _mesh_for(directory, result, mesh):
    if mesh is not None:
        path = Path(mesh).expanduser().resolve()
    elif result.get("mesh"):
        path = Path(result["mesh"]).expanduser().resolve()
    else:
        return None
    return path if path.is_file() else None


def _shock_checks(ref_dir, ref, test_dir, test, mesh):
    missing = []
    if not (ref_dir / "walls.csv").is_file():
        missing.append("RUN_REF walls.csv is missing")
    if not (test_dir / "walls.csv").is_file():
        missing.append("RUN_TEST walls.csv is missing")
    ref_mesh, test_mesh = _mesh_for(ref_dir, ref, mesh), _mesh_for(test_dir, test, mesh)
    if ref_mesh is None:
        missing.append("RUN_REF mesh is unavailable")
    if test_mesh is None:
        missing.append("RUN_TEST mesh is unavailable")
    if missing:
        reason = "; ".join(missing)
        return {f"{eta:.2f}": {"status": f"skipped: {reason}", "ref_x_c": None,
                               "test_x_c": None, "d_x_c": None}
                for eta in SHOCK_ETAS}, False

    try:
        ref_positions = _shock_positions_for_run(ref_dir, ref, ref_mesh)
        test_positions = _shock_positions_for_run(test_dir, test, test_mesh)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reason = f"shock sections unavailable: {type(exc).__name__}: {exc}"
        return {f"{eta:.2f}": {"status": f"skipped: {reason}", "ref_x_c": None,
                               "test_x_c": None, "d_x_c": None}
                for eta in SHOCK_ETAS}, False
    checks = {}
    complete = True
    for eta in SHOCK_ETAS:
        ref_x, test_x = ref_positions[eta], test_positions[eta]
        if not isinstance(ref_x, (int, float)) or not isinstance(test_x, (int, float)):
            checks[f"{eta:.2f}"] = {"status": "absent", "ref_x_c": ref_x,
                                    "test_x_c": test_x, "d_x_c": None}
            complete = False
            continue
        checks[f"{eta:.2f}"] = {"status": "compared", "ref_x_c": ref_x,
                                "test_x_c": test_x, "d_x_c": test_x - ref_x}
    return checks, complete


def _iterations_to_settle(result):
    numbers = result.get("convergence_numbers")
    if not isinstance(numbers, dict):
        return None
    settle = numbers.get("settle")
    iterations = numbers.get("final_alpha_iterations")
    if (not isinstance(settle, dict) or settle.get("settled") is not True
            or isinstance(iterations, bool) or not isinstance(iterations, int)
            or iterations <= 0):
        return None
    return iterations


def _early_gpu_report(run_ref, run_test):
    reasons = []
    inputs = {}
    results = {}
    for name, run_dir in (("cpu", run_ref), ("gpu", run_test)):
        try:
            directory, result = _read_result(run_dir)
            if not isinstance(result, dict):
                raise ValueError("result.json must contain an object")
            inputs[name] = str(directory)
            results[name] = result
        except (OSError, ValueError, TypeError) as exc:
            inputs[name] = str(Path(run_dir).expanduser())
            results[name] = None
            reasons.append(f"{name.upper()} result could not be read: {exc}")

    checks = {
        "cpu_status": None,
        "gpu_status": None,
        "cpu_converged": False,
        "gpu_converged": False,
        "gpu_solver_audit_passed": False,
        "alpha_cpu_deg": None,
        "alpha_gpu_deg": None,
        "abs_d_alpha_deg": None,
        "max_abs_d_alpha_deg": 0.01,
        "CL_cpu": None,
        "CL_gpu": None,
        "abs_dCL": None,
        "abs_dCL_percent": None,
        "max_abs_dCL_percent": 1.5,
        "CD_cpu": None,
        "CD_gpu": None,
        "abs_dCD": None,
        "abs_dCD_percent": None,
        "max_abs_dCD_percent": 1.8,
    }

    def finite_value(result, key, label):
        if result is None:
            return None
        value = result.get(key)
        try:
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
        except (TypeError, ValueError):
            reasons.append(f"{label} {key} is missing or not numeric")
            return None
        if not math.isfinite(value):
            reasons.append(f"{label} {key} is not finite")
            return None
        return value

    cpu, gpu = results["cpu"], results["gpu"]
    for name, result in (("cpu", cpu), ("gpu", gpu)):
        label = name.upper()
        if result is None:
            continue
        status = result.get("status")
        checks[f"{name}_status"] = status
        checks[f"{name}_converged"] = status == "converged"
        if status != "converged":
            reasons.append(f"{label} run is not converged (status={status!r})")
    checks["gpu_solver_audit_passed"] = gpu is not None and gpu.get("solver_audit") == []
    if not checks["gpu_solver_audit_passed"]:
        reasons.append("GPU solver audit is missing or non-empty")
    checks["gpu_native_solver_active_confirmed"] = (
        gpu is not None and gpu.get("gpu_native_solver_active") is True)
    if not checks["gpu_native_solver_active_confirmed"]:
        reasons.append("native GPU solver activation is missing or unconfirmed")

    values = {}
    for key in ("alpha_deg", "CL", "CD"):
        values[("cpu", key)] = finite_value(cpu, key, "CPU")
        values[("gpu", key)] = finite_value(gpu, key, "GPU")

    checks["alpha_cpu_deg"] = values[("cpu", "alpha_deg")]
    checks["alpha_gpu_deg"] = values[("gpu", "alpha_deg")]
    cpu_alpha, gpu_alpha = checks["alpha_cpu_deg"], checks["alpha_gpu_deg"]
    if cpu_alpha is not None and gpu_alpha is not None:
        checks["abs_d_alpha_deg"] = abs(gpu_alpha - cpu_alpha)
        if checks["abs_d_alpha_deg"] > checks["max_abs_d_alpha_deg"]:
            reasons.append("absolute alpha difference exceeds 0.01 degrees")

    for coefficient in ("CL", "CD"):
        cpu_value = values[("cpu", coefficient)]
        gpu_value = values[("gpu", coefficient)]
        checks[f"{coefficient}_cpu"] = cpu_value
        checks[f"{coefficient}_gpu"] = gpu_value
        if cpu_value is None or gpu_value is None:
            continue
        difference = abs(gpu_value - cpu_value)
        checks[f"abs_d{coefficient}"] = difference
        if cpu_value == 0.0:
            reasons.append(f"CPU {coefficient} is zero; relative difference is undefined")
            continue
        relative_percent = difference / abs(cpu_value) * 100.0
        checks[f"abs_d{coefficient}_percent"] = relative_percent
        limit = checks[f"max_abs_d{coefficient}_percent"]
        if relative_percent > limit:
            reasons.append(f"relative {coefficient} difference exceeds {limit}%")

    verdict = "fail" if reasons else "pass"
    return {
        "run_ref": inputs["cpu"],
        "run_test": inputs["gpu"],
        "gates": {"early_gpu": {
            "verdict": verdict,
            "checks": checks,
            "reason": "; ".join(reasons) if reasons else "all early GPU checks passed",
        }},
    }


def compare_runs(run_ref, run_test, mesh=None, out=None, mode="standard"):
    if mode == "early_gpu":
        report = _early_gpu_report(run_ref, run_test)
        output = Path(out).expanduser().resolve() if out else Path("compare.json").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    ref_dir, ref = _read_result(run_ref)
    test_dir, test = _read_result(run_test)
    dcl = float(test["CL"]) - float(ref["CL"])
    dcd_counts = (float(test["CD"]) - float(ref["CD"])) / 1e-4
    dcm = float(test["CM"]) - float(ref["CM"])
    dcd_ex_duct = float(test["CD_ex_duct"]) - float(ref["CD_ex_duct"])
    try:
        alpha_delta = abs(float(test["alpha_deg"]) - float(ref["alpha_deg"]))
    except (KeyError, TypeError, ValueError):
        alpha_delta = None
    alpha_matches = (alpha_delta is not None and math.isfinite(alpha_delta)
                     and alpha_delta <= 1e-6)
    shocks, shocks_complete = _shock_checks(ref_dir, ref, test_dir, test, mesh)
    warm_checks = {"abs_dCD_counts": abs(dcd_counts), "abs_dCL": abs(dcl),
                   "abs_dCM": abs(dcm), "abs_d_alpha_deg": alpha_delta}
    warm_pass = (warm_checks["abs_dCD_counts"] <= WARM_START_MAX_ABS_DCD_COUNTS
                 and warm_checks["abs_dCL"] <= WARM_START_MAX_ABS_DCL
                 and warm_checks["abs_dCM"] <= WARM_START_MAX_ABS_DCM
                 and alpha_matches)
    warm_reasons = []
    if not alpha_matches:
        warm_reasons.append("run alpha_deg values differ by more than 1e-6 degrees")
    if warm_checks["abs_dCD_counts"] > WARM_START_MAX_ABS_DCD_COUNTS:
        warm_reasons.append("CD difference exceeds warm_start tolerance")
    if warm_checks["abs_dCL"] > WARM_START_MAX_ABS_DCL:
        warm_reasons.append("CL difference exceeds warm_start tolerance")
    if warm_checks["abs_dCM"] > WARM_START_MAX_ABS_DCM:
        warm_reasons.append("CM difference exceeds warm_start tolerance")
    audit_empty = test.get("solver_audit") == []
    gpu_checks = {"abs_dCD_counts": abs(dcd_counts), "abs_dCL": abs(dcl),
                  "abs_d_shock_x_c": {eta: abs(row["d_x_c"])
                                       for eta, row in shocks.items()
                                       if row["d_x_c"] is not None},
                  "solver_audit_empty": audit_empty}
    gpu_numeric_pass = (gpu_checks["abs_dCD_counts"] <= GPU_ONLY_MAX_ABS_DCD_COUNTS
                        and gpu_checks["abs_dCL"] <= GPU_ONLY_MAX_ABS_DCL
                        and shocks_complete
                        and all(value <= GPU_ONLY_MAX_ABS_SHOCK_X_C
                                for value in gpu_checks["abs_d_shock_x_c"].values())
                        and audit_empty)
    gpu_reasons = []
    if not shocks_complete:
        gpu_reasons.append("shock comparison is incomplete or absent")
    if gpu_checks["abs_dCD_counts"] > GPU_ONLY_MAX_ABS_DCD_COUNTS:
        gpu_reasons.append("CD difference exceeds gpu_only tolerance")
    if gpu_checks["abs_dCL"] > GPU_ONLY_MAX_ABS_DCL:
        gpu_reasons.append("CL difference exceeds gpu_only tolerance")
    if not audit_empty:
        gpu_reasons.append("GPU solver audit is missing or non-empty")
    if any(value > GPU_ONLY_MAX_ABS_SHOCK_X_C
           for value in gpu_checks["abs_d_shock_x_c"].values()):
        gpu_reasons.append("shock difference exceeds gpu_only tolerance")
    report = {"run_ref": str(ref_dir), "run_test": str(test_dir),
              "alpha_ref_deg": ref.get("alpha_deg"),
              "alpha_test_deg": test.get("alpha_deg"),
              "dCL": dcl, "dCD_counts": dcd_counts, "dCM": dcm,
              "dCD_ex_duct": dcd_ex_duct, "shock_checks": shocks,
              "gates": {"warm_start": {"verdict": "pass" if warm_pass else "fail",
                                          "checks": warm_checks,
                                          "reason": "; ".join(warm_reasons)},
                         "gpu_only": {"verdict": "pass" if gpu_numeric_pass else "fail",
                                      "checks": gpu_checks,
                                      "reason": "; ".join(gpu_reasons)}}}
    if mode == "courant_ab":
        ref_iterations = _iterations_to_settle(ref)
        test_iterations = _iterations_to_settle(test)
        courant_checks = {"abs_dCL": abs(dcl), "abs_dCD_counts": abs(dcd_counts),
                          "abs_dCM": abs(dcm),
                          "iterations_to_settle": {"CFL_10": ref_iterations,
                                                    "CFL_20": test_iterations},
                          "abs_d_alpha_deg": alpha_delta}
        reasons = []
        if courant_checks["abs_dCL"] > 0.001:
            reasons.append("CL difference exceeds 0.001")
        if courant_checks["abs_dCD_counts"] > 0.1:
            reasons.append("CD difference exceeds 0.1 count")
        if courant_checks["abs_dCM"] > 0.002:
            reasons.append("CM difference exceeds 0.002")
        if not alpha_matches:
            reasons.append("run alpha_deg values differ by more than 1e-6 degrees")
        if ref_iterations is None:
            reasons.append("CFL-10 run did not reach settle criterion")
        if test_iterations is None:
            reasons.append("CFL-20 run did not reach settle criterion")
        elif ref_iterations is not None and test_iterations >= ref_iterations:
            reasons.append("CFL-20 run did not reach settle criterion in fewer iterations")
        report["iterations_to_settle"] = {"CFL_10": ref_iterations,
                                          "CFL_20": test_iterations}
        report["gates"]["courant_ab"] = {
            "verdict": "pass" if not reasons else "fail",
            "checks": courant_checks, "reason": "; ".join(reasons)}
    elif mode != "standard":
        raise ValueError(f"unsupported comparison mode: {mode}")
    output = Path(out).expanduser().resolve() if out else Path("compare.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ref")
    parser.add_argument("run_test")
    parser.add_argument("--mesh")
    parser.add_argument("--mode", choices=["standard", "courant_ab", "early_gpu"],
                        default="standard")
    parser.add_argument("--out", default="compare.json")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = compare_runs(args.run_ref, args.run_test, args.mesh, args.out, args.mode)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
