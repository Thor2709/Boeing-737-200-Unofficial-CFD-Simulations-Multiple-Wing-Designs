"""Estimate CFD starting angles from C5 and viscous section polars."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from .cases import case_config
from .xmlgen import plane_sections, project_root

GAMMA = 1.4
SWEEP_DEG = 25.0
REFERENCE_AREA_M2 = 91.04
CRUISE_MACH = 0.7400
TAKEOFF_MACH = 0.2180
CASE_TO_FLOW5 = {"C1": "F1", "C2": "F2", "C3": "F3", "C4": "F4", "C5": "F5"}


@dataclass(frozen=True)
class Polar:
    path: Path
    foil: str
    reynolds: float
    mach: float
    alpha: tuple[float, ...]
    cl: tuple[float, ...]
    cp_min: tuple[float, ...]


def _number(value: str) -> float | None:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_polar(path: str | Path) -> Polar:
    """Read the alpha, CL and Cpmin columns of a flow5/XFoil text polar."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    foil = path.name.split("__", 1)[0]
    mach = None
    reynolds = None
    for line in lines[:24]:
        match = re.search(r"Calculated polar for:\s*(.+?)\s*$", line, re.I)
        if match:
            foil = match.group(1).strip()
        match = re.search(r"\bMach\s*=\s*([-+0-9.eE]+)", line, re.I)
        if match:
            mach = float(match.group(1))
        match = re.search(r"\bRe\s*=\s*([0-9.]+)\s*e\s*([+-]?\d+)", line, re.I)
        if match:
            reynolds = float(match.group(1)) * 10.0 ** int(match.group(2))
        elif (match := re.search(r"\bRe\s*=\s*([0-9.eE+-]+)", line, re.I)):
            reynolds = float(match.group(1))

    header = None
    delimiter = None
    for index, line in enumerate(lines):
        folded = line.casefold()
        if "alpha" in folded and re.search(r"\bcl\b", folded) and "cpmin" in folded.replace(" ", ""):
            delimiter = "," if "," in line else None
            if delimiter:
                labels = [part.strip().casefold().replace("_", "") for part in line.split(",")]
                alpha_index = labels.index("alpha") if "alpha" in labels else None
                cl_index = labels.index("cl") if "cl" in labels else None
                cp_index = next((i for i, label in enumerate(labels) if label == "cpmin"), None)
            else:
                # The plain-text flow5 table uses combined labels "Top Xtr" and
                # "Bot Xtr" but one value for each, so Cpmin is numeric column 7.
                alpha_index, cl_index, cp_index = 0, 1, 7
            if alpha_index is not None and cl_index is not None and cp_index is not None:
                header = (index, alpha_index, cl_index, cp_index)
                break
    if header is None:
        raise ValueError(f"{path}: no alpha/CL/Cpmin table found")
    header_index, alpha_index, cl_index, cp_index = header
    alpha_values, cl_values, cp_values = [], [], []
    for line in lines[header_index + 1:]:
        parts = [part.strip() for part in (line.split(delimiter) if delimiter else line.split())]
        if max(alpha_index, cl_index, cp_index) >= len(parts):
            continue
        alpha_value, cl_value, cp_value = (_number(parts[i]) for i in (alpha_index, cl_index, cp_index))
        if alpha_value is None or cl_value is None or cp_value is None:
            continue
        alpha_values.append(alpha_value)
        cl_values.append(cl_value)
        cp_values.append(cp_value)
    if len(alpha_values) < 3:
        raise ValueError(f"{path}: fewer than three finite alpha/CL/Cpmin samples")
    if reynolds is None or reynolds <= 0.0:
        match = re.search(r"Re[-_]?([0-9]+(?:\.[0-9]+)?)", path.stem, re.I)
        if match:
            reynolds = float(match.group(1)) * 1e6
    if reynolds is None or reynolds <= 0.0 or mach is None or mach < 0.0:
        raise ValueError(f"{path}: polar is missing a valid Reynolds or Mach number")
    return Polar(path, foil, reynolds, mach, tuple(alpha_values), tuple(cl_values), tuple(cp_values))


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear fit needs at least two paired samples")
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ValueError("linear fit samples have no span")
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    return slope, ybar - slope * xbar


def polar_zero_lift_and_slope(polar: Polar) -> tuple[float, float, float]:
    near_zero = [(alpha, cl) for alpha, cl in zip(polar.alpha, polar.cl) if abs(cl) <= 0.5]
    extrapolated = False
    if len(near_zero) < 3:
        near_zero = sorted(zip(polar.alpha, polar.cl), key=lambda pair: abs(pair[1]))[:7]
        extrapolated = True
    slope, intercept = _linear_fit([row[0] for row in near_zero], [row[1] for row in near_zero])
    if slope <= 0.0:
        raise ValueError(f"{polar.path}: non-positive section lift slope {slope:g} CL/deg")
    zero_lift = -intercept / slope
    extrapolation_span = max((abs(zero_lift - row[0]) for row in near_zero), default=0.0) if extrapolated else 0.0
    return zero_lift, slope, extrapolation_span


def prandtl_glauert_slope(slope_mach0: float, mach: float) -> float:
    if not 0.0 <= mach < 1.0:
        raise ValueError("Prandtl-Glauert Mach must be in [0, 1)")
    return slope_mach0 / math.sqrt(1.0 - mach * mach)


def karman_tsien_cp(cp_mach0: float, mach: float) -> float:
    """Convert an incompressible Cp using the Karman-Tsien relation."""
    if not math.isfinite(cp_mach0) or not 0.0 <= mach < 1.0:
        raise ValueError("Karman-Tsien conversion needs finite Cp and Mach in [0, 1)")
    beta = math.sqrt(1.0 - mach * mach)
    denominator = beta + (mach * mach / (2.0 * (1.0 + beta))) * cp_mach0
    if denominator <= 0.0:
        raise ValueError("Karman-Tsien denominator is non-positive")
    return cp_mach0 / denominator


def critical_cp(mach: float) -> float:
    if not 0.0 < mach < 1.0:
        raise ValueError("critical Cp requires Mach in (0, 1)")
    pressure_ratio = (1.0 + (GAMMA - 1.0) * mach * mach / 2.0) / (1.0 + (GAMMA - 1.0) / 2.0)
    return 2.0 / (GAMMA * mach * mach) * (pressure_ratio ** (GAMMA / (GAMMA - 1.0)) - 1.0)


def _polar_cp_at_cl(polar: Polar, cl_needed: float, *, cl_scale: float = 1.0) -> tuple[float, float, bool]:
    samples = sorted((alpha, cl * cl_scale, cp_min)
                     for alpha, cl, cp_min in zip(polar.alpha, polar.cl, polar.cp_min))
    peak = max(range(len(samples)), key=lambda index: samples[index][1])
    attached = samples[:peak + 1]
    if cl_needed <= attached[0][1]:
        return attached[0][2], attached[0][1], cl_needed < attached[0][1]
    for (_, cl0, cp0), (_, cl1, cp1) in zip(attached, attached[1:]):
        if cl0 <= cl_needed <= cl1 and cl1 > cl0:
            fraction = (cl_needed - cl0) / (cl1 - cl0)
            return cp0 + fraction * (cp1 - cp0), cl_needed, False
    peak_row = samples[peak]
    return peak_row[2], peak_row[1], cl_needed > peak_row[1]


def _load_candidates(directory: Path, foil: str) -> list[Polar]:
    if not directory.is_dir():
        return []
    candidates = []
    for path in sorted(directory.rglob("*.txt")):
        if not path.name.split("__", 1)[0].casefold() == foil.casefold():
            continue
        try:
            polar = read_polar(path)
        except (OSError, ValueError):
            continue
        if polar.foil.casefold() == foil.casefold():
            candidates.append(polar)
    return candidates


def _choose_polar(directory: Path, foil: str, reynolds: float,
                  *, mach: float | None = None) -> Polar | None:
    candidates = _load_candidates(directory, foil)
    if mach is not None:
        candidates = [polar for polar in candidates if abs(polar.mach - mach) <= 0.002]
    if not candidates:
        return None
    return min(candidates, key=lambda polar: abs(math.log(polar.reynolds / reynolds)))


def _section_geometry(case_name: str) -> list[dict]:
    config = case_config(case_name)
    if not config.get("flap"):
        return config["sections"]
    detailed = plane_sections(case_name, "fine")
    result = []
    for section in config["sections"]:
        closest = min(detailed, key=lambda candidate: abs(candidate["y"] - section["y"]))
        result.append({**section, "foil": closest["foil"]})
    return result


def _station_weights(sections: list[dict]) -> list[float]:
    semispan = max(float(section["y"]) for section in sections)
    weights = []
    for section in sections:
        eta = float(section["y"]) / semispan
        weights.append(float(section["chord"]) * math.sqrt(max(0.0, 1.0 - eta * eta)))
    if sum(weights) <= 0.0:
        raise ValueError("section geometry has no positive elliptic-chord weight")
    return weights


def _local_section_cls(wing_cl: float, sections: list[dict]) -> list[float]:
    """Normalize an elliptic span-load distribution to the reference wing CL."""
    semispan = max(float(section["y"]) for section in sections)
    integrand = [float(section["chord"]) * math.sqrt(max(0.0, 1.0 - (float(section["y"]) / semispan) ** 2))
                 for section in sections]
    integral = sum((float(right["y"]) - float(left["y"])) * (q0 + q1) / 2.0
                   for left, right, q0, q1 in zip(sections, sections[1:], integrand, integrand[1:]))
    if integral <= 0.0:
        raise ValueError("section geometry has no elliptic load integral")
    scale = wing_cl * REFERENCE_AREA_M2 / (2.0 * integral)
    return [scale * math.sqrt(max(0.0, 1.0 - (float(section["y"]) / semispan) ** 2))
            for section in sections]


def _mach_for_case(case: str) -> float:
    if case == "C3":
        return TAKEOFF_MACH
    return CRUISE_MACH * math.cos(math.radians(SWEEP_DEG))


def _normal_plane_cl(cl: float, sweep_cos: float) -> float:
    return cl / (sweep_cos * sweep_cos)


def _streamwise_from_normal(value: float, sweep_cos: float) -> float:
    return value * sweep_cos


def _get_case_polar(root: Path, mach_dir: Path | None, case: str, section: dict,
                    reynolds: float, mach: float, *, sweep_normal: bool) -> tuple[Polar, str]:
    foil = str(section["foil"])
    if mach_dir is not None:
        polar_foil = f"{foil}_n25" if sweep_normal else foil
        compressible = _choose_polar(mach_dir, polar_foil, reynolds, mach=mach)
        if compressible is not None:
            return compressible, "compressible_polar"
    default_dir = root / "flow5_v2" / case / "fine" / "xfoil_polars"
    incompressible = _choose_polar(default_dir, foil, reynolds, mach=0.0)
    if incompressible is None:
        raise FileNotFoundError(f"no Mach 0 polar for {foil} near Re={reynolds:.0f} under {default_dir}")
    return incompressible, "Mach0_PG_KT"


def _cfd_data(c5_dir: Path) -> dict:
    history_path = c5_dir / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{history_path}: no history rows")
    required = {"alpha", "CL_wing", "CL"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{history_path}: requires columns {sorted(required)}")
    final = rows[-1]
    try:
        final_alpha = float(final["alpha"])
        wing_cl = float(final["CL_wing"])
        total_cl = float(final["CL"])
    except (ValueError, TypeError) as error:
        raise ValueError(f"{history_path}: final row has invalid alpha/CL values") from error
    result_path = c5_dir / "result.json"
    provisional = not result_path.is_file()
    if not provisional:
        try:
            final_alpha = float(json.loads(result_path.read_text(encoding="utf-8"))["alpha_deg"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{result_path}: cannot read alpha_deg") from error
    groups: dict[float, list[float]] = {}
    for row in rows:
        try:
            alpha, cl_wing = float(row["alpha"]), float(row["CL_wing"])
        except (ValueError, TypeError):
            continue
        if math.isfinite(alpha) and math.isfinite(cl_wing):
            groups.setdefault(alpha, []).append(cl_wing)
    points = []
    for alpha, values in groups.items():
        if len(values) >= 3:
            points.append((alpha, statistics.median(values[-5:])))
    if len(points) < 3:
        raise ValueError(f"{history_path}: need at least three C5 search angles for a local CFD slope")
    points.sort()
    slope, _ = _linear_fit([point[0] for point in points], [point[1] for point in points])
    if slope <= 0.0:
        raise ValueError(f"{history_path}: fitted C5 local lift slope is not positive")
    return {
        "alpha_deg": final_alpha,
        "alpha_provisional": provisional,
        "CL_wing": wing_cl,
        "CL_total": total_cl,
        "CL_nonwing": total_cl - wing_cl,
        "local_lift_slope_per_deg": slope,
        "slope_method": "linear fit of the median of the last five CL_wing samples at each C5 search angle",
        "history_path": history_path,
        "angle_samples": len(points),
    }


def _flow5_alphas(root: Path) -> dict[str, float]:
    summary = root / "flow5_v2" / "summary.csv"
    selected = {}
    with summary.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                if row["mesh"] == "fine" and float(row["wake_spans"]) == 20.0:
                    selected[row["case"]] = float(row["alpha_deg"])
            except (KeyError, ValueError, TypeError):
                continue
    if "F5" not in selected:
        raise ValueError(f"{summary}: missing fine/20-span F5 row")
    return selected


def _seed_terms(alpha_cfd: float, twist_c5: float, twist_i: float,
                alpha0_c5: float, alpha0_i: float, cl_needed: float,
                a_i: float, a_c5: float) -> tuple[float, float, float]:
    if a_i <= 0.0 or a_c5 <= 0.0:
        raise ValueError("wing lift slopes must be positive")
    d_geom = twist_c5 - twist_i
    d_aero_alpha0 = alpha0_i - alpha0_c5
    d_aero_slope = cl_needed * (1.0 / a_i - 1.0 / a_c5)
    return d_geom, d_aero_alpha0 + d_aero_slope, d_aero_slope


def estimate(c5_dir: str | Path, *, mach_polars: str | Path | None = None) -> dict:
    root = project_root()
    c5_dir = Path(c5_dir)
    cfd = _cfd_data(c5_dir)
    flow5 = _flow5_alphas(root)
    mach_dir = None if mach_polars is None else Path(mach_polars)
    if mach_dir is not None and not mach_dir.is_dir():
        raise FileNotFoundError(f"compressible polar directory does not exist: {mach_dir}")

    baseline_case = "F5"
    baseline_sections = _section_geometry(baseline_case)
    baseline_weights = _station_weights(baseline_sections)
    baseline_mach = _mach_for_case("C1")
    baseline_reynolds = [float(case_config(baseline_case)["flight"]["velocity"]) * float(section["chord"])
                         / float(case_config(baseline_case)["flight"]["nu"])
                         for section in baseline_sections]
    baseline_polars = []
    baseline_sources = []
    for section, reynolds in zip(baseline_sections, baseline_reynolds):
        polar, source = _get_case_polar(root, mach_dir, "F5", section, reynolds, baseline_mach,
                                        sweep_normal=True)
        baseline_polars.append(polar)
        baseline_sources.append(source)
    baseline_alpha0_values, baseline_slopes, baseline_alpha0_spans = _polar_properties(
        baseline_polars, baseline_sources, baseline_mach,
        sweep_cos=math.cos(math.radians(SWEEP_DEG)))
    baseline_alpha0 = sum(w * value for w, value in zip(baseline_weights, baseline_alpha0_values)) / sum(baseline_weights)
    baseline_alpha0_span = sum(w * value for w, value in zip(baseline_weights, baseline_alpha0_spans)) / sum(baseline_weights)
    baseline_a2d = sum(w * slope for w, slope in zip(baseline_weights, baseline_slopes)) / sum(baseline_weights)
    cfd_slope = cfd["local_lift_slope_per_deg"]
    planform_factor = cfd_slope / baseline_a2d
    if planform_factor <= 0.0:
        raise ValueError("C5 CFD to 2D polar planform factor is not positive")

    rows = {}
    escalation = False
    gamma_cp_star_by_mach = {}
    for cfd_case, flow5_case in (("C1", "F1"), ("C2", "F2"), ("C3", "F3"), ("C4", "F4")):
        if flow5_case not in flow5:
            raise ValueError(f"flow5 summary is missing {flow5_case} fine/20-span alpha")
        target = case_config(flow5_case)
        sections = _section_geometry(flow5_case)
        weights = _station_weights(sections)
        mach = _mach_for_case(cfd_case)
        sweep_normal = cfd_case != "C3"
        sweep_cos = math.cos(math.radians(SWEEP_DEG)) if sweep_normal else 1.0
        condition = target["flight"]
        reynolds = [float(condition["velocity"]) * float(section["chord"]) / float(condition["nu"])
                    for section in sections]
        target_polars, target_sources = [], []
        for section, local_re in zip(sections, reynolds):
            polar, source = _get_case_polar(root, mach_dir, flow5_case, section, local_re, mach,
                                            sweep_normal=sweep_normal)
            target_polars.append(polar)
            target_sources.append(source)
        alpha0_values, section_slopes, alpha0_spans = _polar_properties(
            target_polars, target_sources, mach, sweep_cos=sweep_cos)
        alpha0_i = sum(w * value for w, value in zip(weights, alpha0_values)) / sum(weights)
        alpha0_span = sum(w * value for w, value in zip(weights, alpha0_spans)) / sum(weights)
        a2d_i = sum(w * value for w, value in zip(weights, section_slopes)) / sum(weights)
        a_i = planform_factor * a2d_i
        target_cl = float(condition["target_cl"])
        cl_needed = target_cl - cfd["CL_nonwing"]
        section_cls = _local_section_cls(cl_needed, sections)
        normal_section_cls = [_normal_plane_cl(cl, sweep_cos) for cl in section_cls]
        cp_stats = []
        cp_star = gamma_cp_star_by_mach.setdefault(round(mach, 8), critical_cp(mach))
        for polar, source, local_cl in zip(target_polars, target_sources, normal_section_cls):
            polar_cl_scale = 1.0 / (sweep_cos * sweep_cos) if source == "Mach0_PG_KT" else 1.0
            cp_value, at_cl, extrapolated = _polar_cp_at_cl(polar, local_cl, cl_scale=polar_cl_scale)
            if source == "Mach0_PG_KT":
                cp_value = karman_tsien_cp(cp_value, mach)
            cp_stats.append({"cp_min": cp_value, "at_cl": at_cl,
                             "margin": cp_value - cp_star,
                             "extrapolated": extrapolated})
        d_geom = sum(w * (float(base["twist"]) - float(section["twist"]))
                     for w, base, section in zip(weights, baseline_sections, sections)) / sum(weights)
        d_aero_alpha0 = alpha0_i - baseline_alpha0
        d_aero_slope = cl_needed * (1.0 / a_i - 1.0 / cfd_slope)
        d_aero = d_aero_alpha0 + d_aero_slope
        corrected = cfd["alpha_deg"] + d_geom + d_aero
        current_rule = cfd["alpha_deg"] + flow5[flow5_case] - flow5["F5"]
        spatial_terms = [float(base["twist"]) - float(section["twist"]) + alpha0 - alpha0_base
                         for base, section, alpha0, alpha0_base in
                         zip(baseline_sections, sections, alpha0_values, baseline_alpha0_values)]
        lifted_terms = [term for term, weight in zip(spatial_terms, weights) if weight > 0.0]
        band = [cfd["alpha_deg"] + d_aero_slope + min(lifted_terms) - alpha0_span - baseline_alpha0_span,
                cfd["alpha_deg"] + d_aero_slope + max(lifted_terms) + alpha0_span + baseline_alpha0_span]
        section_rows = []
        flags = ["shock_correction_not_applied"]
        if cfd["alpha_provisional"]:
            flags.append("C5_alpha_from_history_provisional")
        if any(stat["margin"] < 0.0 for stat in cp_stats):
            flags.append("supercritical_sections")
        if any(span > 0.0 for span in alpha0_spans) or baseline_alpha0_span > 0.0:
            flags.append("zero_lift_angle_linearly_extrapolated")
        for index, (section, polar, source, cp_stat, local_cl, local_cl_normal, alpha0, slope) in enumerate(
                zip(sections, target_polars, target_sources, cp_stats, section_cls, normal_section_cls,
                    alpha0_values, section_slopes)):
            polar_cl_scale = 1.0 / (sweep_cos * sweep_cos) if source == "Mach0_PG_KT" else 1.0
            cl_max = max(polar.cl) * polar_cl_scale
            supercritical = cp_stat["margin"] < 0.0
            row = {
                "y_m": float(section["y"]),
                "foil": section["foil"],
                "Re": reynolds[index],
                "polar_Re": polar.reynolds,
                "polar_Mach": polar.mach,
                "polar_source": source,
                "alpha_0L_deg": alpha0,
                "alpha_0L_extrapolation_span_deg": alpha0_spans[index],
                "a_2D_per_deg": slope,
                "CL_needed": local_cl,
                "CL_needed_normal_plane": local_cl_normal,
                "cl_max_2D": cl_max,
                "viscous_clmax_below_needed": local_cl > cl_max,
                "Cp_min": cp_stat["cp_min"],
                "Cp_min_compressible": cp_stat["cp_min"],
                "Cp_min_at_CL": cp_stat["at_cl"],
                "Cp_star": cp_star,
                "Cp_margin": cp_stat["margin"],
                "supercritical": supercritical,
                "Cp_CL_extrapolated": cp_stat["extrapolated"],
            }
            section_rows.append(row)
        if cfd_case == "C3" and any(section["viscous_clmax_below_needed"] for section in section_rows):
            flags.append("viscous_CLmax_may_prevent_target_CL_1.675")
        difference = corrected - current_rule
        if cfd_case != "C3" and abs(difference) > 1.0:
            escalation = True
            flags.append("corrected_seed_differs_from_current_rule_by_more_than_1_deg")
        rows[cfd_case] = {
            "condition": target["condition"],
            "Mach_normal": mach,
            "target_CL": target_cl,
            "CL_nonwing_C5": cfd["CL_nonwing"],
            "CL_wing_needed": cl_needed,
            "alpha_CFD_C5_deg": cfd["alpha_deg"],
            "current_rule_alpha0_deg": current_rule,
            "alpha0_CFD_deg": corrected,
            "d_geom_deg": d_geom,
            "d_aero_deg": d_aero,
            "d_aero_alpha0L_deg": d_aero_alpha0,
            "d_aero_lift_slope_deg": d_aero_slope,
            "a_2D_weighted_per_deg": a2d_i,
            "a_3D_per_deg": a_i,
            "first_secant_step_slope_per_deg": cfd_slope * (a_i / cfd_slope),
            "alpha0_band_deg": {"lower": min(band), "upper": max(band),
                                "basis": "four-station spread expanded by zero-lift linear extrapolation span; shock uncertainty is unquantified"},
            "difference_from_current_rule_deg": difference,
            "sections": section_rows,
            "flags": flags,
        }
    return {
        "task_id": "P10SEED",
        "provisional": cfd["alpha_provisional"],
        "C5_reference": {key: value for key, value in cfd.items()
                         if key not in {"history_path"}},
        "method": {
            "spanwise_weights": "chord times elliptic factor sqrt(1-(y/b/2)^2) at the four basis stations",
            "planform_factor": planform_factor,
            "planform_factor_definition": "single C5 CFD/weighted F5 2D slope ratio applied to all case wings",
            "Mach_rule": "cruise uses 0.7400 cos(25 deg); take-off uses M=0.2180",
            "sweep_rule": "cruise uses the sweep-normal plane: M_n=M cos(25 deg), cl_n=cl/cos^2(25 deg), alpha_n=alpha/cos(25 deg), and (t/c)_n=(t/c)/cos(25 deg); alpha_0L and section slope map back by multiplying by cos(25 deg) before the planform factor; take-off remains streamwise",
            "fallback": "Mach 0 streamwise section polars are transformed algebraically to the cruise normal plane; slope uses Prandtl-Glauert at M_n; Cp_min uses Karman-Tsien at M_n",
            "shock_note": "no shock correction applied; Cp_star margin and supercritical flags are reported, with model uncertainty unquantified",
            "alpha0_band": "min/max lifted-station correction spread plus weighted zero-lift extrapolation span; not a shock-correction estimate",
        },
        "needs_opus_decision": escalation,
        "cases": rows,
    }


def _polar_properties(polars: list[Polar], sources: list[str], mach: float, *, sweep_cos: float = 1.0):
    alpha0, slopes, extrapolation_spans = [], [], []
    for polar, source in zip(polars, sources):
        zero_lift, slope, extrapolation_span = polar_zero_lift_and_slope(polar)
        zero_lift /= sweep_cos
        slope /= sweep_cos
        extrapolation_span /= sweep_cos
        if source == "Mach0_PG_KT":
            slope = prandtl_glauert_slope(slope, mach)
        alpha0.append(_streamwise_from_normal(zero_lift, sweep_cos))
        slopes.append(_streamwise_from_normal(slope, sweep_cos))
        extrapolation_spans.append(_streamwise_from_normal(extrapolation_span, sweep_cos))
    return alpha0, slopes, extrapolation_spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c5", required=True, help="C5 CFD run directory")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--mach-polars", help="optional directory of compressible flow5 2D polars")
    args = parser.parse_args(argv)
    try:
        report = estimate(args.c5, mach_polars=args.mach_polars)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"alpha_seed: {error}", file=sys.stderr)
        return 1
    print(f"wrote {output}; provisional={report['provisional']} needs_opus_decision={report['needs_opus_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
