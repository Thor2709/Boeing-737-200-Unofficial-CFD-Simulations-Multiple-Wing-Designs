"""Case result interpolation, coverage checks, tables, and plots."""
from __future__ import annotations

import csv
import json
import math
import re
from bisect import bisect_left
from pathlib import Path
from xml.etree import ElementTree

from .cases import case_config
from .parse import (FoilPolar, NumericTable, OperatingPoint, alpha_gap_diagnostics,
                    parse_foil_polar, parse_op_points, parse_plane_polar)
from .xmlgen import compute_mac, plane_sections

REFERENCE_AREA_M2 = 91.04
REFERENCE_SPAN_M = 28.346
ASPECT_RATIO = REFERENCE_SPAN_M**2 / REFERENCE_AREA_M2
N_TO_LBF = 0.22480894387096
ETA_STATIONS = (0.15, 0.30, 0.50, 0.70, 0.90)


def breguet_range_nmi(cl: float, cd: float) -> float:
    if cl <= 0.0 or cd <= 0.0:
        raise ValueError("Breguet range requires positive CL and CD")
    tsfc = 1.662e-4
    rho_sl = 0.000889
    area_ft2 = 979.9
    weight_start_lbf = 115500.0
    weight_end_lbf = 83474.0
    miles_per_nmi = 6076.11549
    return ((1.0 / tsfc) * (math.sqrt(cl) / cd)
            * (2.0 * math.sqrt(2.0) / math.sqrt(rho_sl * area_ft2))
            * (math.sqrt(weight_start_lbf) - math.sqrt(weight_end_lbf)) / miles_per_nmi)


def interpolate_target(table: NumericTable, target_cl: float) -> dict[str, float]:
    """Interpolate T1 coefficients on an increasing, monotonic CL-alpha segment."""
    needed = {name: table.index(name) for name in ("alpha", "cl", "cd", "cdi", "cdv", "cm")}
    rows = [row for row in table.rows if all(math.isfinite(row[index]) for index in needed.values())]
    rows.sort(key=lambda row: row[needed["alpha"]])
    if len(rows) < 2:
        raise ValueError("plane polar needs at least two finite points for interpolation")
    candidates: list[tuple[int, list[float], list[float]]] = []
    for index, (left, right) in enumerate(zip(rows, rows[1:])):
        alpha0, alpha1 = left[needed["alpha"]], right[needed["alpha"]]
        cl0, cl1 = left[needed["cl"]], right[needed["cl"]]
        if alpha1 <= alpha0 or cl1 <= cl0 or not (cl0 <= target_cl <= cl1):
            continue
        # Prefer a bracket embedded in the longest locally increasing CL run.
        lo = index
        while lo > 0 and rows[lo][needed["cl"]] > rows[lo - 1][needed["cl"]]:
            lo -= 1
        hi = index + 1
        while hi + 1 < len(rows) and rows[hi + 1][needed["cl"]] > rows[hi][needed["cl"]]:
            hi += 1
        candidates.append((hi - lo, left, right))
    if not candidates:
        segments = [(left[needed["cl"]], right[needed["cl"]])
                    for left, right in zip(rows, rows[1:])
                    if right[needed["alpha"]] > left[needed["alpha"]]
                    and right[needed["cl"]] > left[needed["cl"]]]
        if not segments:
            raise ValueError("plane polar has no increasing CL-alpha segment")
        ranges = ", ".join(f"[{low:.6g}, {high:.6g}]" for low, high in segments)
        raise ValueError(f"target CL {target_cl:g} is outside increasing polar segment(s) {ranges}")
    _, left, right = max(candidates, key=lambda candidate: candidate[0])
    cl0, cl1 = left[needed["cl"]], right[needed["cl"]]
    fraction = (target_cl - cl0) / (cl1 - cl0)
    labels = ("alpha", "cl", "cd", "cdi", "cdv", "cm")
    result = {
        label: left[needed[label]] + fraction * (right[needed[label]] - left[needed[label]])
        for label in labels
    }
    result["cl"] = float(target_cl)
    return result


def _normalised_foil(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _foils_at_y(y: float, sections: list[dict]) -> list[str]:
    target = abs(y)
    for left, right in zip(sections, sections[1:]):
        if left["y"] <= target <= right["y"]:
            return list(dict.fromkeys((str(left["foil"]), str(right["foil"]))))
    endpoint = sections[0] if target < sections[0]["y"] else sections[-1]
    return [str(endpoint["foil"])]


def _polar_limits(foil_polars: list[FoilPolar], reynolds: float, cl: float) -> tuple[bool, str | None]:
    reynolds_values = sorted(polar.reynolds for polar in foil_polars)
    if not reynolds_values or reynolds < reynolds_values[0] or reynolds > reynolds_values[-1]:
        return False, "Re outside available 2D polar range"
    ordered = sorted(foil_polars, key=lambda polar: polar.reynolds)
    lower = max((polar for polar in ordered if polar.reynolds <= reynolds), key=lambda polar: polar.reynolds)
    upper = min((polar for polar in ordered if polar.reynolds >= reynolds), key=lambda polar: polar.reynolds)
    lo0, hi0 = min(lower.cl), max(lower.cl)
    lo1, hi1 = min(upper.cl), max(upper.cl)
    fraction = 0.0 if lower.reynolds == upper.reynolds else (reynolds - lower.reynolds) / (upper.reynolds - lower.reynolds)
    cl_min = lo0 + fraction * (lo1 - lo0)
    cl_max = hi0 + fraction * (hi1 - hi0)
    if cl < cl_min or cl > cl_max:
        return False, f"Cl {cl:.5g} outside [{cl_min:.5g}, {cl_max:.5g}] at Re {reynolds:.6g}"
    return True, None


def check_coverage(strips: list[dict], case_name: str, foil_polars: list[FoilPolar], *,
                   alpha_step: float = 0.25,
                   alpha_gaps: list[dict] | None = None) -> list[dict]:
    sections = plane_sections(case_name, "coarse")
    groups: dict[str, list[FoilPolar]] = {}
    for polar in foil_polars:
        groups.setdefault(_normalised_foil(polar.foil), []).append(polar)
    violations = []
    for strip in strips:
        for foil_name in _foils_at_y(float(strip["y_m"]), sections):
            polar_group = groups.get(_normalised_foil(foil_name), [])
            if not polar_group:
                valid, reason = False, "no matching foil polar"
            else:
                valid, reason = _polar_limits(polar_group, float(strip["re"]), float(strip["cl"]))
            if not valid:
                violations.append({"y_m": strip["y_m"], "foil": foil_name, "re": strip["re"],
                                   "cl": strip["cl"], "reason": reason})
    if alpha_gaps is None:
        alpha_gaps = [diagnostic for polar in foil_polars
                      for diagnostic in alpha_gap_diagnostics(polar, alpha_step)]
    violations.extend(alpha_gaps)
    return violations


def _strip_field(table, name: str) -> int:
    aliases = {"y": ("y(m)",), "re": ("Re",), "cl": ("Cl", "CL"),
               "cdi": ("Cd_i", "CDi", "CD_induced"), "cdv": ("Cd_v", "CDv", "CD_viscous"),
               "cm": ("Cm_i", "Cm", "Cm_v")}
    for index, label in enumerate(table.columns):
        if label.casefold() in {value.casefold() for value in aliases[name]}:
            return index
    raise ValueError(f"strip table is missing {name} column; found {table.columns}")


def _interpolate_strip_rows(points: list[OperatingPoint], target_alpha: float) -> list[dict]:
    usable = [point for point in points if point.alpha is not None and point.strips]
    if not usable:
        raise ValueError("no owning-polar operating points have an alpha and strip table")
    usable.sort(key=lambda point: float(point.alpha))
    lower = max((point for point in usable if point.alpha <= target_alpha), key=lambda point: float(point.alpha), default=None)
    upper = min((point for point in usable if point.alpha >= target_alpha), key=lambda point: float(point.alpha), default=None)
    if lower is None or upper is None:
        lower_alpha = min(float(point.alpha) for point in usable)
        upper_alpha = max(float(point.alpha) for point in usable)
        raise ValueError(f"alpha {target_alpha:g} is outside operating-point strip range [{lower_alpha:g}, {upper_alpha:g}]")
    alpha0, alpha1 = float(lower.alpha), float(upper.alpha)
    fraction = 0.0 if alpha1 == alpha0 else (target_alpha - alpha0) / (alpha1 - alpha0)
    lower_table = next(iter(lower.strips.values()))
    upper_table = next(iter(upper.strips.values()))
    if len(lower_table.rows) != len(upper_table.rows):
        raise ValueError("neighboring operating-point strip tables have different row counts")
    fields = {name: _strip_field(lower_table, name) for name in ("y", "re", "cl", "cdi", "cdv", "cm")}
    upper_fields = {name: _strip_field(upper_table, name) for name in fields}
    result = []
    for row0, row1 in zip(lower_table.rows, upper_table.rows):
        values = {}
        for name, index in fields.items():
            value0, value1 = row0[index], row1[upper_fields[name]]
            values[name] = value0 + fraction * (value1 - value0)
        result.append({"y_m": values["y"], "eta": abs(values["y"]) / (REFERENCE_SPAN_M / 2.0),
                       "re": values["re"], "cl": values["cl"], "cdi": values["cdi"],
                       "cdv": values["cdv"], "cm": values["cm"]})
    return result


def _section_geometry(sections: list[dict], y_m: float, wing_x_m: float) -> tuple[float, float]:
    target = abs(y_m)
    ordered = sorted(sections, key=lambda section: float(section["y"]))
    if target <= float(ordered[0]["y"]):
        section = ordered[0]
        return wing_x_m + float(section["x"]), float(section["chord"])
    if target >= float(ordered[-1]["y"]):
        section = ordered[-1]
        return wing_x_m + float(section["x"]), float(section["chord"])
    for left, right in zip(ordered, ordered[1:]):
        y0, y1 = float(left["y"]), float(right["y"])
        if y0 <= target <= y1:
            fraction = (target - y0) / (y1 - y0)
            x_le = float(left["x"]) + fraction * (float(right["x"]) - float(left["x"]))
            chord = float(left["chord"]) + fraction * (float(right["chord"]) - float(left["chord"]))
            return wing_x_m + x_le, chord
    raise ValueError(f"cannot interpolate wing geometry at y={y_m:g} m")


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _unit(vector: tuple[float, float, float], description: str) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"cannot determine {description} from Cp panel coordinates")
    return tuple(value / length for value in vector)


def _section_axes(sections: list[dict], y_m: float) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    ordered = sorted(sections, key=lambda section: float(section["y"]))
    target = abs(y_m)
    if target <= float(ordered[0]["y"]):
        left = right = ordered[0]
        fraction = 0.0
    elif target >= float(ordered[-1]["y"]):
        left = right = ordered[-1]
        fraction = 0.0
    else:
        left, right = next((pair for pair in zip(ordered, ordered[1:])
                            if pair[0]["y"] <= target <= pair[1]["y"]))
        fraction = (target - float(left["y"])) / (float(right["y"]) - float(left["y"]))
    if left is right:
        sweep = 0.0
        dihedral = float(left.get("dihedral", 0.0))
    else:
        sweep = ((float(right["x"]) - float(left["x"])) /
                 (float(right["y"]) - float(left["y"])))
        dihedral = (float(left.get("dihedral", 0.0)) +
                    fraction * (float(right.get("dihedral", 0.0)) - float(left.get("dihedral", 0.0))))
    span = _unit((sweep, 1.0, math.tan(math.radians(dihedral))), "local span direction")
    global_x = (1.0, 0.0, 0.0)
    projection = _dot(global_x, span)
    chord_reference = _unit(tuple(global_x[i] - projection * span[i] for i in range(3)),
                            "projected chord reference")
    chord = float(left["chord"]) + fraction * (float(right["chord"]) - float(left["chord"]))
    return chord, span, chord_reference


def _wing_position_x(project_dir: str | Path, case_name: str) -> float:
    plane_path = Path(project_dir).parents[1] / "planes" / f"{case_name}.xml"
    if not plane_path.is_file():
        return 0.0  # plane_xml currently places the main wing at (0, 0, 0).
    root = ElementTree.parse(plane_path).getroot()
    position = root.findtext(".//wing/Position")
    if position is None:
        raise ValueError(f"{plane_path}: main wing Position is missing")
    values = [value.strip() for value in position.split(",")]
    if len(values) != 3:
        raise ValueError(f"{plane_path}: invalid main wing Position {position!r}")
    return float(values[0])


def _panel_surface_curves(table, case: dict, wing_x_m: float) -> dict[str, tuple[list[float], list[float]]]:
    columns = {column.casefold(): index for index, column in enumerate(table.columns)}
    needed = ("ctrlpt.x", "ctrlpt.y", "ctrlpt.z", "n.x", "n.y", "n.z", "cp")
    if any(name not in columns for name in needed):
        raise ValueError(f"Cp panel block is missing one of {needed}; found {table.columns}")
    x_index, y_index, z_index = (columns[name] for name in ("ctrlpt.x", "ctrlpt.y", "ctrlpt.z"))
    nx_index, ny_index, nz_index, cp_index = (columns[name] for name in ("n.x", "n.y", "n.z", "cp"))
    finite_rows = [row for row in table.rows
                   if all(math.isfinite(row[index]) for index in
                          (x_index, y_index, z_index, nx_index, ny_index, nz_index, cp_index))]
    if len(finite_rows) < 6:
        raise ValueError("Cp panel block needs at least six finite control points to define a local chord")
    y_m = float(table.strip_y_m) if table.strip_y_m is not None else (
        sum(row[y_index] for row in finite_rows) / len(finite_rows))
    chord, span, chord_reference = _section_axes(case["sections"], y_m)
    if chord <= 0.0:
        raise ValueError(f"non-positive wing chord at y={y_m:g} m")
    points = [(row[x_index], row[y_index], row[z_index]) for row in finite_rows]
    ordered_points = sorted(points, key=lambda point: _dot(point, chord_reference))
    forward = tuple(sum(point[axis] for point in ordered_points[:2]) / 2.0 for axis in range(3))
    aft = tuple(sum(point[axis] for point in ordered_points[-2:]) / 2.0 for axis in range(3))
    chord_vector = tuple(aft[axis] - forward[axis] for axis in range(3))
    chord_vector = tuple(chord_vector[axis] - _dot(chord_vector, span) * span[axis] for axis in range(3))
    direction = _unit(chord_vector, "local chord direction")
    previous_aft = tuple(sum(point[axis] for point in ordered_points[-4:-2]) / 2.0 for axis in range(3))
    aft_panel_length = _dot(tuple(aft[axis] - previous_aft[axis] for axis in range(3)), direction)
    if aft_panel_length <= 0.0:
        raise ValueError("Cp panel coordinates do not define a positive local aft panel length")
    trailing_edge = tuple(aft[axis] + 0.5 * aft_panel_length * direction[axis] for axis in range(3))
    leading_edge = tuple(trailing_edge[axis] - chord * direction[axis] for axis in range(3))
    upper_normal = _unit((direction[1] * span[2] - direction[2] * span[1],
                          direction[2] * span[0] - direction[0] * span[2],
                          direction[0] * span[1] - direction[1] * span[0]), "local upper normal")
    grouped: dict[str, list[tuple[float, float]]] = {"upper": [], "lower": []}
    for row in finite_rows:
        normal = (row[nx_index], row[ny_index], row[nz_index])
        normal_side = _dot(normal, upper_normal)
        if normal_side == 0.0:
            continue
        point = (row[x_index], row[y_index], row[z_index])
        x_c = _dot(tuple(point[axis] - leading_edge[axis] for axis in range(3)), direction) / chord
        grouped["upper" if normal_side > 0.0 else "lower"].append((min(1.0, max(0.0, x_c)), row[cp_index]))
    result = {}
    for surface, values in grouped.items():
        if not values:
            continue
        values.sort()
        by_x: dict[float, list[float]] = {}
        for x_value, cp_value in values:
            by_x.setdefault(x_value, []).append(cp_value)
        xs = sorted(by_x)
        result[surface] = (xs, [sum(by_x[x_value]) / len(by_x[x_value]) for x_value in xs])
    return result


def _curve_at(xs: list[float], values: list[float], x_value: float) -> float:
    index = bisect_left(xs, x_value)
    if index < len(xs) and math.isclose(xs[index], x_value, abs_tol=1e-12):
        return values[index]
    if index == 0 or index == len(xs):
        raise ValueError(f"x/c={x_value:g} is outside common Cp range [{xs[0]:g}, {xs[-1]:g}]")
    fraction = (x_value - xs[index - 1]) / (xs[index] - xs[index - 1])
    return values[index - 1] + fraction * (values[index] - values[index - 1])


def _cp_curves_for_eta(point: OperatingPoint, case: dict, eta: float,
                       wing_x_m: float) -> dict[str, tuple[list[float], list[float]]]:
    tables = [table for table in point.cp_tables if table.strip_y_m is not None and table.strip_y_m > 0.0]
    if not tables:
        raise ValueError("flow5 did not export mapped Cp blocks on the positive wing side")
    tables.sort(key=lambda table: float(table.strip_y_m))
    target_y = eta * REFERENCE_SPAN_M / 2.0
    ys = [float(table.strip_y_m) for table in tables]
    upper = next((index for index, y_value in enumerate(ys) if y_value >= target_y), len(ys) - 1)
    lower = max(0, upper - 1)
    if target_y <= ys[0]:
        lower = upper = 0
    elif target_y >= ys[-1]:
        lower = upper = len(ys) - 1
    fraction = 0.0 if lower == upper else (target_y - ys[lower]) / (ys[upper] - ys[lower])
    lower_curves = _panel_surface_curves(tables[lower], case, wing_x_m)
    upper_curves = _panel_surface_curves(tables[upper], case, wing_x_m)
    result = {}
    for surface in ("upper", "lower"):
        if surface not in lower_curves or surface not in upper_curves:
            raise ValueError(f"Cp surface {surface} is missing at eta={eta:.2f} on a neighboring strip")
        xs0, cps0 = lower_curves[surface]
        xs1, cps1 = upper_curves[surface]
        common_min, common_max = max(xs0[0], xs1[0]), min(xs0[-1], xs1[-1])
        grid = sorted({x_value for x_value in (*xs0, *xs1)
                       if common_min <= x_value <= common_max})
        if not grid:
            raise ValueError(f"neighboring Cp strips have no common x/c range at eta={eta:.2f}")
        cps = []
        for x_value in grid:
            cp0 = _curve_at(xs0, cps0, x_value)
            cp1 = _curve_at(xs1, cps1, x_value)
            cps.append(cp0 + fraction * (cp1 - cp0))
        result[surface] = (grid, cps)
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def postprocess(case_name: str, polar_path: str | Path, project_dir: str | Path,
                output_dir: str | Path, foil_polar_dir: str | Path, *,
                alpha_gaps: list[dict] | None = None, alpha_step: float = 0.25) -> dict:
    case = case_config(case_name)
    polar = parse_plane_polar(polar_path)
    flight = case["flight"]
    target_cl = float(flight["target_cl"])
    interpolation_error = None
    clmax_limited = False
    try:
        interpolated = interpolate_target(polar, target_cl)
        output_alpha = interpolated["alpha"]
    except ValueError as error:
        if "outside increasing polar segment" not in str(error):
            raise
        interpolation_error = str(error)
        alpha_index, cl_index = polar.index("alpha"), polar.index("cl")
        finite_rows = [row for row in polar.rows
                       if math.isfinite(row[alpha_index]) and math.isfinite(row[cl_index])]
        if not finite_rows:
            raise ValueError("plane polar has no finite alpha/CL rows for diagnostic output") from error
        nearest = min(finite_rows, key=lambda row: abs(row[cl_index] - target_cl))
        output_alpha = nearest[alpha_index]
        interpolated = None
        # Opus F3 decision (2026-10-03): target above the converged CLmax -> report the
        # CLmax point itself, flagged "clmax_limited", instead of empty metrics.
        labels = ("alpha", "cl", "cd", "cdi", "cdv", "cm")
        full_rows = [row for row in finite_rows
                     if all(math.isfinite(row[polar.index(label)]) for label in labels)]
        if full_rows and target_cl > max(row[cl_index] for row in full_rows):
            peak = max(full_rows, key=lambda row: row[cl_index])
            interpolated = {label: float(peak[polar.index(label)]) for label in labels}
            output_alpha = interpolated["alpha"]
            clmax_limited = True

    plane_directory = Path(project_dir) / case["plane_name"]
    polar_name = case["polar_name"]
    op_files = [path for path in plane_directory.rglob("*.csv") if path.parent != plane_directory]
    points = parse_op_points(op_files, expected_polar=polar_name)
    strips = _interpolate_strip_rows(points, output_alpha)
    foil_polars = [parse_foil_polar(path) for path in Path(foil_polar_dir).glob("*.txt")]
    violations = check_coverage(strips, case_name, foil_polars,
                                alpha_step=alpha_step, alpha_gaps=alpha_gaps)
    detected_alpha_gaps = [violation for violation in violations if violation.get("kind") == "alpha_gap"]
    cp_point = min((point for point in points if point.alpha is not None),
                   key=lambda point: abs(float(point.alpha) - output_alpha), default=None)
    if cp_point is None:
        raise ValueError("no owning-polar operating point has an alpha for Cp output")
    wing_x_m = _wing_position_x(project_dir, case_name)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "strips.csv", ["y_m", "eta", "re", "cl", "cdi", "cdv", "cm"], strips)
    cp_records = []
    for eta in ETA_STATIONS:
        curves = _cp_curves_for_eta(cp_point, case, eta, wing_x_m)
        rows = [{"x_c": x_value, "cp": cp_value, "surface": surface}
                for surface, (x_values, cp_values) in curves.items()
                for x_value, cp_value in zip(x_values, cp_values)]
        _write_csv(destination / f"cp_eta{eta:.2f}.csv", ["x_c", "cp", "surface"], rows)
        cp_records.append((eta, curves))

    if interpolated is None:
        metrics = {"alpha_deg": None, "CL": None, "CDi": None, "CDv": None, "CD": None,
                   "Cm": None, "lift_N": None, "lift_lbf": None, "drag_N": None,
                   "drag_lbf": None, "L_over_D": None, "span_efficiency_e": None,
                   "breguet_range_nmi": None}
    else:
        q = 0.5 * float(flight["rho"]) * float(flight["velocity"]) ** 2
        lift_n = q * REFERENCE_AREA_M2 * interpolated["cl"]
        drag_n = q * REFERENCE_AREA_M2 * interpolated["cd"]
        e = interpolated["cl"] ** 2 / (math.pi * ASPECT_RATIO * interpolated["cdi"]) if interpolated["cdi"] > 0 else None
        breguet = breguet_range_nmi(interpolated["cl"], interpolated["cd"]) if case["condition"] == "cruise" else None
        metrics = {
            "alpha_deg": interpolated["alpha"], "CL": interpolated["cl"], "CDi": interpolated["cdi"],
            "CDv": interpolated["cdv"], "CD": interpolated["cd"], "Cm": interpolated["cm"],
            "lift_N": lift_n, "lift_lbf": lift_n * N_TO_LBF, "drag_N": drag_n,
            "drag_lbf": drag_n * N_TO_LBF, "L_over_D": interpolated["cl"] / interpolated["cd"],
            "span_efficiency_e": e, "breguet_range_nmi": breguet,
        }

    result = {
        "case": case["name"], "condition": case["condition"], "target_cl": target_cl,
        "target_cl_status": ("clmax_limited" if clmax_limited else "out_of_range") if interpolation_error else "interpolated",
        "target_cl_message": interpolation_error,
        "diagnostic_alpha_deg": output_alpha if interpolation_error else None,
        "cp_alpha_deg": cp_point.alpha, **metrics, "coverage_violations": violations,
        "alpha_gap_diagnostics": detected_alpha_gaps,
        "reference": {"S_m2": REFERENCE_AREA_M2, "b_m": REFERENCE_SPAN_M,
                      "MAC_m": compute_mac(case["sections"]), "AR": ASPECT_RATIO},
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    axis.plot([row["y_m"] for row in strips], [row["cl"] for row in strips], marker="o")
    axis.set(xlabel="Spanwise y (m)", ylabel="Strip Cl", title=f"{case['name']} span loading")
    axis.grid(True)
    fig.savefig(destination / "cl_span.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots()
    for eta, curves in cp_records:
        for surface, (x_values, cp_values) in curves.items():
            axis.plot(x_values, cp_values, label=f"eta {eta:.2f} {surface}")
    axis.set(xlabel="x/c", ylabel="Cp", title=f"{case['name']} Cp by span station")
    axis.invert_yaxis()
    axis.grid(True)
    axis.legend()
    fig.savefig(destination / "cp_eta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return result
