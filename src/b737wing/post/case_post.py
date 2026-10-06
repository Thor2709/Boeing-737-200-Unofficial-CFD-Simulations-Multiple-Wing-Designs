"""Post-process one completed Fluent run or summarise all completed runs.

Examples:
  python -m b737wing.post.case_post C5 --run cfd_v2/runs/C5/cruise
  python -m b737wing.post.case_post --summary
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from b737wing.cfd.solve.conditions import condition_for_case
from b737wing.cfd.solve.fluent_io import resolve_zone_groups
from .breguet import breguet_range_nmi
from .mesh_view import read_walls
from .render_cfd import (render_flow_images, render_mach_slice,
                         write_convergence, write_yplus_products)
from .sections import (FACE_SIZE_RELATIVE_TOLERANCE, nearest_centroid_map,
                       write_cp_sections, write_spanwise)


try:
    from b737wing.config import REPO_ROOT
    PROJECT_ROOT = REPO_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
S_REF_M2 = 91.04
SOLVER_MAC_M = 3.403
MAC_M = 3.4716
CM_SCALE = SOLVER_MAC_M / MAC_M
CM_CONVENTION = ("geometric MAC 3.4716 m; CM_geom = CM_solver * "
                 "3.403/3.4716")
COMMON_CRUISE_CL = 0.4895
FALLBACK_DCD_DCL = 0.13
X_REF_M = 12.760
SEMISPAN_M = 14.17
LBF_PER_N = 4.4482216
WALL_REQUIRED = ("x", "y", "z", "zone", "pressure-coefficient", "wall-yplus")
SUMMARY_COLUMNS = (
    "case", "tag", "role", "authoritative", "solver", "precision",
    "alpha", "CL", "CD", "CD_raw",
    "CD_corrected_CL_0.4895", "correction_dCD_dCL", "correction_source",
    "correction_flag", "CD_ex_duct", "CM", "CM_solver_raw", "CM_convention",
    "L/D", "L/D_raw", "L/D_corrected", "L/D_ex_duct", "lift_kN", "lift_lbf",
    "drag_kN", "drag_lbf", "Breguet_nmi", "Breguet_raw_nmi",
    "Breguet_corrected_nmi", "Breguet_ex_duct_nmi", "status", "iterations")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _wall_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "zone" not in reader.fieldnames:
            raise ValueError(f"{path}: walls.csv requires a zone column (NEEDS_OPUS_DECISION)")
        missing = sorted(set(WALL_REQUIRED) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: walls.csv lacks required columns: {', '.join(missing)}")
        grouped = {}
        for row in reader:
            zone = row["zone"].strip()
            grouped.setdefault(zone, []).append(row)
    if not grouped:
        raise ValueError(f"{path}: walls.csv contains no wall faces")
    return grouped


def _map_wall_data(wall_meshes, grouped_rows, zone_groups):
    required_zones = list(zone_groups["total"])
    missing_meshes = sorted(set(required_zones) - set(wall_meshes))
    missing_rows = sorted(set(required_zones) - set(grouped_rows))
    extra_rows = sorted(set(grouped_rows) - set(wall_meshes))
    if missing_meshes or missing_rows or extra_rows:
        found = ", ".join(sorted(wall_meshes))
        unmatched = sorted(set(missing_meshes + missing_rows + extra_rows))
        raise ValueError("wall zones do not match between mesh and walls.csv: "
                         + ", ".join(unmatched) + "; zones found: " + found)
    mapped = {}
    for zone in required_zones:
        mesh = wall_meshes[zone]
        rows = grouped_rows[zone]
        source = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows])
        target = np.asarray(mesh.cell_centers().points, dtype=float)
        if len(rows) != mesh.n_cells:
            raise ValueError(f"{zone}: walls.csv has {len(rows)} faces, mesh has {mesh.n_cells}")
        sizes = np.sqrt(np.asarray(mesh.compute_cell_sizes(length=False, volume=False)["Area"]))
        fields = {}
        for field, key in (("Cp", "pressure-coefficient"), ("yplus", "wall-yplus"),
                           ("x-shear", "x-wall-shear"), ("y-shear", "y-wall-shear"),
                           ("z-shear", "z-wall-shear")):
            if key in rows[0]:
                values = np.asarray([float(row[key]) for row in rows], dtype=float)
                fields[field], distances = nearest_centroid_map(source, target, values, face_sizes=sizes)
                if field == "Cp":
                    mapped[zone] = {
                        "median_distance_m": float(np.median(distances)),
                        "max_distance_m": float(np.max(distances)),
                        "max_face_size_ratio": float(np.max(distances / sizes)),
                        "per_face_tolerance_fraction": FACE_SIZE_RELATIVE_TOLERANCE}
        for name, values in fields.items():
            mesh.cell_data[name] = values
        mapped.setdefault(zone, {})["mesh"] = mesh
    return mapped


def _component_meshes(mapped, zone_groups):
    components = {}
    for component in ("wing", "fuselage", "htail", "vtail", "nacelle", "duct", "fairing"):
        zones = zone_groups[component]
        meshes = [mapped[zone]["mesh"] for zone in zones]
        combined = meshes[0].copy(deep=True)
        for mesh in meshes[1:]:
            combined = combined.merge(mesh, merge_points=False)
        components[component] = {"mesh": combined, "zones": list(zones)}
    return components


def _combined_wall_surface(wall_meshes):
    meshes = list(wall_meshes.values())
    if not meshes:
        raise ValueError("cannot frame a Mach slice without aircraft wall surfaces")
    combined = meshes[0].copy(deep=True)
    for mesh in meshes[1:]:
        combined = combined.merge(mesh, merge_points=False)
    return combined


def _write_forces(result, output):
    components = result.get("components", {})
    cd_sum = sum(float(row.get("CD", 0.0)) for row in components.values())
    q_s = None
    if result.get("CL") and result.get("lift_N") is not None:
        q_s = float(result["lift_N"]) / float(result["CL"])
    elif result.get("CD") and result.get("drag_N") is not None:
        q_s = float(result["drag_N"]) / float(result["CD"])
    elif result.get("case"):
        from b737wing.cfd.solve.conditions import condition_for_case
        condition = condition_for_case(result["case"])
        q_s = (0.5 * condition["density_kg_m3"] * condition["velocity_m_s"] ** 2
               * S_REF_M2)
    columns = ("component", "CL", "CD", "CM", "CM_solver_raw", "CM_convention",
               "CD_share", "lift_N", "lift_lbf",
               "drag_N", "drag_lbf", "model")
    with Path(output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, coeffs in components.items():
            cl, cd = (float(coeffs.get(key, 0.0)) for key in ("CL", "CD"))
            cm_raw = float(coeffs.get("CM", 0.0))
            cm = _rescale_cm(cm_raw)
            lift_n = cl * q_s if q_s is not None else None
            drag_n = cd * q_s if q_s is not None else None
            writer.writerow({"component": name, "CL": cl, "CD": cd, "CM": cm,
                             "CM_solver_raw": cm_raw, "CM_convention": CM_CONVENTION,
                             "CD_share": cd / cd_sum if cd_sum else None,
                             "lift_N": lift_n, "lift_lbf": lift_n / LBF_PER_N if lift_n is not None else None,
                             "drag_N": drag_n, "drag_lbf": drag_n / LBF_PER_N if drag_n is not None else None,
                             "model": "full-aircraft"})


def _rescale_cm(value):
    return None if value is None else float(value) * CM_SCALE


def _settled_alpha_points(result, history_path=None):
    points_by_alpha = {}
    history_error = None
    if history_path is not None and Path(history_path).is_file():
        try:
            with Path(history_path).open("r", newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    state = row.get("settle", "")
                    if not state:
                        continue
                    try:
                        settled = json.loads(state).get("settled") is True
                    except (AttributeError, TypeError, json.JSONDecodeError):
                        continue
                    if not settled:
                        continue
                    alpha, cl, cd = (float(row[key]) for key in ("alpha", "CL", "CD"))
                    if np.isfinite([alpha, cl, cd]).all():
                        points_by_alpha[alpha] = (cl, cd)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            history_error = f"history.csv unavailable: {type(exc).__name__}"
    if len(points_by_alpha) < 2:
        for key in ("settled_alpha_points", "settled_points"):
            for point in result.get(key) or ():
                if not isinstance(point, dict) or point.get("settled", True) is not True:
                    continue
                try:
                    alpha = float(point.get("alpha_deg", point.get("alpha")))
                    cl = float(point.get("CL", point.get("cl")))
                    cd = float(point.get("CD", point.get("cd")))
                except (TypeError, ValueError):
                    continue
                if np.isfinite([alpha, cl, cd]).all():
                    points_by_alpha[alpha] = (cl, cd)
    points = list(points_by_alpha.items())
    if len(points) < 2:
        reason = history_error or "fewer than two settled alpha points"
        return None, reason
    (alpha0, (cl0, cd0)), (alpha1, (cl1, cd1)) = points[-2:]
    if alpha1 == alpha0 or cl1 == cl0:
        return None, "last two settled alpha points have no finite CL slope"
    slope = (cd1 - cd0) / (cl1 - cl0)
    if not np.isfinite(slope):
        return None, "last two settled alpha points have a non-finite dCD/dCL"
    return float(slope), None


def common_cl_correction(result, case, history_path=None):
    """Correct cruise drag and range to the workbook's common lift coefficient."""
    case = str(case).upper()
    if condition_for_case(case)["condition"] != "cruise":
        return {"CD_corrected": None, "L_over_D_corrected": None,
                "Breguet_corrected_nmi": None, "dCD_dCL": None,
                "slope_source": "not-cruise", "flag": ""}
    try:
        cl, cd = float(result.get("CL", float("nan"))), float(result.get("CD", float("nan")))
    except (TypeError, ValueError):
        cl = cd = float("nan")
    if not np.isfinite([cl, cd]).all():
        return {"CD_corrected": None, "L_over_D_corrected": None,
                "Breguet_corrected_nmi": None, "dCD_dCL": None,
                "slope_source": "unavailable", "flag": "CL or CD is unavailable"}
    slope, reason = _settled_alpha_points(result, history_path)
    if slope is None:
        slope, source = FALLBACK_DCD_DCL, "fallback_0.13"
        flag = reason or "local dCD/dCL unavailable; used 0.13"
    else:
        source, flag = "last_two_settled_alpha_points", ""
    corrected_cd = cd + slope * (COMMON_CRUISE_CL - cl)
    ld = COMMON_CRUISE_CL / corrected_cd if corrected_cd > 0 else None
    range_result = breguet_range_nmi(COMMON_CRUISE_CL, corrected_cd, case)
    return {"CD_corrected": corrected_cd, "L_over_D_corrected": ld,
            "Breguet_corrected_nmi": range_result["range_nmi"],
            "dCD_dCL": slope, "slope_source": source, "flag": flag}


def _breguet(result, case=None, history_path=None):
    case = str(case or result.get("case", "")).upper()
    base = breguet_range_nmi(float(result.get("CL", 0.0)), float(result.get("CD", 0.0)), case)
    cd_ex = result.get("CD_ex_duct")
    cl_ex = None
    if cd_ex is not None and result.get("L_over_D_ex_duct") is not None:
        cl_ex = float(result["L_over_D_ex_duct"]) * float(cd_ex)
    elif cd_ex is not None:
        cl_ex = sum(float(values.get("CL", 0.0)) for name, values in result.get("components", {}).items()
                    if name != "duct")
    ex = (breguet_range_nmi(cl_ex, float(cd_ex), case) if cl_ex is not None and cd_ex is not None
          else {"range_nmi": None, "reason": "CL_ex_duct or CD_ex_duct is unavailable."})
    common = common_cl_correction(result, case, history_path)
    return {"range_nmi": base["range_nmi"], "reason": base["reason"],
            "CD_raw": result.get("CD"), "L_over_D_raw": result.get("L_over_D"),
            "Breguet_raw_nmi": base["range_nmi"], **common,
            "ex_duct_range_nmi": ex["range_nmi"], "ex_duct_reason": ex["reason"]}


def _summary_row(result, case, tag, role="calibration", authoritative=False,
                 history_path=None):
    br = breguet_range_nmi(float(result.get("CL", 0)), float(result.get("CD", 0)), case)
    cd_ex = result.get("CD_ex_duct")
    ld_ex = result.get("L_over_D_ex_duct")
    if ld_ex is None and cd_ex:
        cl_ex = sum(float(values.get("CL", 0)) for name, values in result.get("components", {}).items()
                    if name != "duct")
        ld_ex = cl_ex / float(cd_ex) if cd_ex else None
    cl_ex = float(ld_ex) * float(cd_ex) if ld_ex is not None and cd_ex is not None else None
    br_ex = (breguet_range_nmi(cl_ex, float(cd_ex), case)["range_nmi"]
             if cl_ex is not None and cd_ex is not None else None)
    correction = common_cl_correction(result, case, history_path)
    cm_raw = result.get("CM")
    return {"case": case, "tag": tag, "role": role,
            "authoritative": str(bool(authoritative)).lower(),
            "solver": result.get("solver"), "precision": result.get("precision"),
            "alpha": result.get("alpha_deg"), "CL": result.get("CL"),
            "CD": result.get("CD"), "CD_raw": result.get("CD"),
            "CD_corrected_CL_0.4895": correction["CD_corrected"],
            "correction_dCD_dCL": correction["dCD_dCL"],
            "correction_source": correction["slope_source"],
            "correction_flag": correction["flag"], "CD_ex_duct": cd_ex,
            "CM": _rescale_cm(cm_raw), "CM_solver_raw": cm_raw,
            "CM_convention": CM_CONVENTION,
            "L/D": result.get("L_over_D"), "L/D_raw": result.get("L_over_D"),
            "L/D_corrected": correction["L_over_D_corrected"],
            "L/D_ex_duct": ld_ex,
            "lift_kN": float(result["lift_N"]) / 1000.0 if result.get("lift_N") is not None else None,
            "lift_lbf": result.get("lift_lbf"),
            "drag_kN": float(result["drag_N"]) / 1000.0 if result.get("drag_N") is not None else None,
            "drag_lbf": result.get("drag_lbf"), "Breguet_nmi": br["range_nmi"],
            "Breguet_raw_nmi": br["range_nmi"],
            "Breguet_corrected_nmi": correction["Breguet_corrected_nmi"],
            "Breguet_ex_duct_nmi": br_ex, "status": result.get("status"),
            "iterations": result.get("iterations")}


def _plan_summary_metadata(plan_path, runs_dir):
    if plan_path is None:
        plan_path = PROJECT_ROOT / "research" / "chain_plan_v2.json"
    plan_path = Path(plan_path)
    if not plan_path.is_file():
        return {}, None
    plan = _read_json(plan_path)
    base = (PROJECT_ROOT if plan_path.parent.name == "research" else plan_path.parent)
    entries = {}
    gate_path = None
    for step in plan.get("steps", []):
        args = step.get("args", [])
        if "--out" not in args:
            continue
        raw_path = Path(args[args.index("--out") + 1])
        resolved = raw_path if raw_path.is_absolute() else base / raw_path
        if step.get("id") == "gci_gate":
            gate_path = resolved
        if step.get("cmd") != "run_case":
            continue
        case = str(args[0]).upper() if args else ""
        key = (case, raw_path.parts[-1].lower())
        entries.setdefault(key, []).append(step)
    if gate_path is None:
        gate_path = Path(runs_dir) / "C5" / "gci_gate.json"
    return entries, gate_path


def _gci_verdict(path):
    if path is None or not Path(path).is_file():
        return None
    try:
        report = _read_json(path)
        verdict = report.get("gates", {}).get("gpu_within_gci", {}).get("verdict")
    except (OSError, ValueError, AttributeError):
        return None
    return verdict if verdict in ("pass", "fail") else None


def _run_role_and_authority(result_path, case, entries, gci_verdict, runs_dir=None):
    run = Path(result_path).parent
    leaf = run.name.lower()
    try:
        relative = run.relative_to(runs_dir) if runs_dir is not None else run
    except ValueError:
        relative = run
    path_parts = [part.lower() for part in relative.parts]
    planned = entries.get((case, leaf), [])
    step_ids = [str(step.get("id", "")).lower() for step in planned]
    if any("smoke" in part for part in path_parts):
        return "smoke", False
    if any(token in leaf for token in ("gate", "cfl", "t2", "t5")):
        return "gate", False
    if any(token in leaf for token in ("g400", "g650")) or any(
            token in planned_id for planned_id in step_ids for token in ("g400", "g650")):
        return "gci", False
    if any(token in leaf for token in ("calib", "abort", "fail")):
        return "calibration", False
    if case == "C5" and leaf == "ih0":
        return "production", True
    if case in {"C1", "C2", "C3", "C4"} and leaf in {"ih0", "gpu"}:
        if leaf == "gpu":
            try:
                result = _read_json(result_path)
            except (OSError, ValueError, TypeError):
                return "calibration", False
            if (result.get("solver_audit") != []
                    or result.get("gpu_native_solver_active") is not True
                    or result.get("status") != "converged"):
                return "calibration", False
            return ("production", True) if gci_verdict == "pass" else ("calibration", False)
        return "production", gci_verdict != "pass"
    if planned:
        if any(step.get("when") for step in planned):
            return "production", False
        return "production", True
    return "calibration", False


def write_summary(runs_dir=None, output=None, diagnostics_output=None, plan_path=None):
    runs_dir = Path(runs_dir or PROJECT_ROOT / "cfd_v2" / "runs")
    output = Path(output or PROJECT_ROOT / "post_v2" / "summary.csv")
    diagnostics_output = Path(diagnostics_output or output.with_name(
        output.stem + "_diagnostics" + output.suffix))
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(runs_dir.glob("**/result.json"))
    entries, gate_path = _plan_summary_metadata(plan_path, runs_dir)
    gci_verdict = _gci_verdict(gate_path)
    tables = {"production": [], "diagnostics": []}
    for result_path in paths:
        result = _read_json(result_path)
        case = str(result.get("case", result_path.parent.parent.name)).upper()
        role, authoritative = _run_role_and_authority(result_path, case, entries, gci_verdict,
                                                       runs_dir)
        row = _summary_row(result, case, result.get("tag", result_path.parent.name),
                           role, authoritative, result_path.parent / "history.csv")
        if role == "production" and authoritative:
            tables["production"].append(row)
        else:
            tables["diagnostics"].append(row)
    for path, rows in ((output, tables["production"]),
                       (diagnostics_output, tables["diagnostics"])):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return output


def process_run(case, run_dir, mesh_path=None, out_dir=None):
    case = str(case).upper()
    run = Path(run_dir).expanduser().resolve()
    result_path = run / "result.json"
    walls_path = run / "walls.csv"
    for required in (result_path, walls_path):
        if not required.is_file():
            raise FileNotFoundError(str(required))
    result = _read_json(result_path)
    mesh = Path(mesh_path or result.get("mesh") or
                PROJECT_ROOT / "cfd_v2" / "mesh" / case / "prod" / "mesh.msh.h5").expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(str(mesh))
    output = Path(out_dir or PROJECT_ROOT / "post_v2" / case).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"case": case, "run": str(run), "products": {}}
    _write_forces(result, output / "forces.csv")
    history = run / "history.csv"
    breguet = _breguet(result, case, history)
    (output / "breguet.json").write_text(json.dumps(breguet, indent=2), encoding="utf-8")
    grouped = _wall_rows(walls_path)
    wall_meshes, zone_groups = read_walls(mesh, mirror=False, return_groups=True)
    zone_groups = resolve_zone_groups(zone_groups["total"] + zone_groups["farfield"])
    mapped = _map_wall_data(wall_meshes, grouped, zone_groups)
    components = _component_meshes(mapped, zone_groups)
    report["wall_centroid_median_distance_m"] = {
        zone: values["median_distance_m"] for zone, values in mapped.items()
        if "median_distance_m" in values}
    report["wall_centroid_mapping"] = {
        zone: {key: value for key, value in values.items() if key != "mesh"}
        for zone, values in mapped.items() if "median_distance_m" in values}
    write_cp_sections(components["wing"], components["fuselage"],
                      float(result.get("alpha_deg", 0.0)), output, SEMISPAN_M)
    report["products"]["cp_sections"] = "written"
    write_spanwise(components["wing"], float(result.get("alpha_deg", 0.0)), output, SEMISPAN_M, MAC_M)
    report["products"]["spanwise_loading"] = "written"
    flow_walls = {zone: value["mesh"] for zone, value in mapped.items()}
    write_yplus_products(components["wing"], output, flow_walls,
                         run / "walls_area.csv", zone_groups,
                         walls_path=walls_path)
    report["products"]["yplus"] = "written"
    if history.is_file():
        write_convergence(history, output)
        report["products"]["convergence"] = "written"
    else:
        report["products"]["convergence"] = "skipped: missing history.csv"

    ensight_dir = run / "ensight"
    cases = sorted(ensight_dir.glob("*.case")) if ensight_dir.is_dir() else []
    if not ensight_dir.is_dir():
        reason = "missing ensight/*.case final volume export"
        for product in ("mach_slice", "mach_slice_eta40", "mach_slice_symmetry",
                        "flow_image", "flow_image_top"):
            report["products"][product] = f"skipped: {reason}"
    else:
        if not cases:
            raise FileNotFoundError(f"{ensight_dir}: no EnSight .case final volume export")
        volume = pv.read(str(cases[0]))
        wall_surface = _combined_wall_surface(flow_walls)
        render_mach_slice(volume, 0.4 * SEMISPAN_M, output / "mach_slice.png", wall_surface)
        render_mach_slice(volume, 0.0, output / "mach_slice_symmetry.png", wall_surface)
        render_flow_images(flow_walls, volume, output, SEMISPAN_M)
        report["products"]["mach_slice"] = "written"
        report["products"]["mach_slice_eta40"] = "written"
        report["products"]["mach_slice_symmetry"] = "written"
        report["products"]["flow_image"] = "written"
        report["products"]["flow_image_top"] = "written"
    report_path = output / "post_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", help="case identifier, for example C5")
    parser.add_argument("--run", help="completed Fluent run folder")
    parser.add_argument("--mesh", help="half-model Fluent mesh.msh.h5")
    parser.add_argument("--out", help="output folder (default post_v2/<case>)")
    parser.add_argument("--summary", action="store_true", help="write post_v2/summary.csv")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.summary:
        print(write_summary())
        return 0
    if not args.case or not args.run:
        _parser().error("case and --run are required unless --summary is selected")
    print(process_run(args.case, args.run, args.mesh, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
