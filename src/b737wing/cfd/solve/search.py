"""Pure secant-search and coefficient-drift checks."""

from __future__ import annotations

import math

CL_TOLERANCE = 0.003
MAX_SECANT_STEPS = 6
ASSUMED_CL_ALPHA_PER_DEG = 0.1
MAX_SINGLE_POINT_STEP_DEG = 2.0
MAX_SUBSEQUENT_STEP_DEG = 1.0
MIN_ALPHA_DEG = -2.0
MAX_ALPHA_DEG = 16.0
MIN_REACHABLE_CL_ALPHA_PER_DEG = 0.005
MIN_CL_ALPHA_PER_DEG = 0.02
MAX_CL_ALPHA_PER_DEG = 0.20
SECANT_SLOPE_EPSILON = 1e-10
MIN_SEARCH_ITERATIONS = 400
MIN_FINAL_ITERATIONS = 900
ITERATION_CHUNK = 50
MAX_ITERATIONS_PER_ALPHA = 6000
SETTLE_WINDOW_ITERATIONS = 600
CL_SETTLE_TOLERANCE = 0.0005
CD_SETTLE_TOLERANCE = 0.5e-4
CM_SETTLE_TOLERANCE = 0.001
DRIFT_WINDOW_ITERATIONS = 100
MAX_CD_DRIFT = 0.5e-4
MAX_CL_DRIFT = 1e-3
MAX_MASS_IMBALANCE = 1e-4


def _sample_iteration(sample: dict, index: int) -> int:
    return int(sample.get("iteration", sample.get(
        "iter", (index + 1) * ITERATION_CHUNK)))


def _settle_result(y_inf=None, remaining=None, tau_iterations=None,
                   settled=False) -> dict:
    return {"y_inf": y_inf, "remaining": remaining,
            "tau_iterations": tau_iterations, "settled": settled}


def _fit_settle_window(values: list[float], tolerance: float) -> dict:
    start, middle, end = values[0], values[len(values) // 2], values[-1]
    scale = max(1.0, *(abs(value) for value in values))
    if max(values) - min(values) <= math.ulp(scale):
        return _settle_result(end, 0.0, None, True)

    first_change = middle - start
    second_change = end - middle
    if first_change == 0.0:
        return _settle_result()
    q = second_change / first_change
    if not 0.0 < q < 1.0:
        return _settle_result()

    y_inf = (middle - q * start) / (1.0 - q)
    remaining = abs(end - y_inf)
    tau_iterations = -(SETTLE_WINDOW_ITERATIONS / 2.0) / math.log(q)
    return _settle_result(y_inf, remaining, tau_iterations,
                          remaining <= tolerance)


def settle_check(samples: list[dict]) -> dict:
    """Extrapolate CL, CD and CM from the latest 600-iteration window."""
    result = {"CL": _settle_result(), "CD": _settle_result(),
              "CM": _settle_result(), "settled": False}
    if not samples:
        return result

    indexed = {_sample_iteration(sample, index): sample
               for index, sample in enumerate(samples)}
    end_iteration = max(indexed)
    start_iteration = end_iteration - SETTLE_WINDOW_ITERATIONS
    middle_iteration = start_iteration + SETTLE_WINDOW_ITERATIONS // 2
    required_iterations = range(start_iteration, end_iteration + 1,
                                ITERATION_CHUNK)
    if any(iteration not in indexed for iteration in required_iterations):
        return result

    window = [indexed[iteration] for iteration in required_iterations]
    start = indexed[start_iteration]
    middle = indexed[middle_iteration]
    end = indexed[end_iteration]
    tolerances = {"CL": CL_SETTLE_TOLERANCE,
                  "CD": CD_SETTLE_TOLERANCE,
                  "CM": CM_SETTLE_TOLERANCE}
    for quantity, tolerance in tolerances.items():
        values = [float(sample[quantity]) for sample in window]
        selected = [float(start[quantity]), float(middle[quantity]),
                    float(end[quantity])]
        increasing = all(right >= left for left, right in zip(values, values[1:]))
        decreasing = all(right <= left for left, right in zip(values, values[1:]))
        if increasing or decreasing:
            result[quantity] = _fit_settle_window(selected, tolerance)
        else:
            y_inf = sum(selected) / len(selected)
            remaining = abs(selected[-1] - y_inf)
            result[quantity] = _settle_result(
                y_inf, remaining, None,
                max(values) - min(values) <= 2.0 * tolerance)
    result["settled"] = all(result[quantity]["settled"]
                            for quantity in tolerances)
    result["iterations"] = end_iteration
    return result


def _slope_from(points: list[tuple[float, float]]) -> float | None:
    for (a0, cl0), (a1, cl1) in reversed(list(zip(points, points[1:]))):
        if a1 != a0:
            return (cl1 - cl0) / (a1 - a0)
    return None


def _slope_in_band(slope: float) -> bool:
    return (MIN_CL_ALPHA_PER_DEG <= slope <= MAX_CL_ALPHA_PER_DEG
            or math.isclose(slope, MIN_CL_ALPHA_PER_DEG, rel_tol=1e-12)
            or math.isclose(slope, MAX_CL_ALPHA_PER_DEG, rel_tol=1e-12))


def _bound_alpha(alpha: float) -> float:
    return min(MAX_ALPHA_DEG, max(MIN_ALPHA_DEG, float(alpha)))


def secant_next(points: list[tuple[float, float]], target: float,
                initial_slope: float | None = None, *,
                settled_points: list[tuple[float, float]] | None = None,
                slope_replacements: list[dict] | None = None) -> float | None:
    """Return the next angle, or None when a usable secant is unavailable."""
    if len(points) == 1:
        slope = (initial_slope if initial_slope is not None
                 else ASSUMED_CL_ALPHA_PER_DEG)
        if not math.isfinite(slope) or abs(slope) < SECANT_SLOPE_EPSILON:
            return None
        delta = (target - points[0][1]) / slope
        if initial_slope is None:
            delta = min(MAX_SINGLE_POINT_STEP_DEG,
                        max(-MAX_SINGLE_POINT_STEP_DEG, delta))
        candidate = _bound_alpha(points[0][0] + delta)
        return candidate if candidate != points[0][0] else None

    a0, cl0 = points[-2]
    a1, cl1 = points[-1]
    slope = (cl1 - cl0) / (a1 - a0) if a1 != a0 else 0.0
    if not math.isfinite(slope) or not _slope_in_band(slope):
        fallback_points = list(settled_points or [])[:-1]
        fallback_slope = _slope_from(fallback_points)
        if (fallback_slope is not None and math.isfinite(fallback_slope)
                and _slope_in_band(fallback_slope)):
            slope = fallback_slope
            source = "settled_points"
        elif initial_slope is not None:
            slope = initial_slope
            source = "seed"
        else:
            slope = ASSUMED_CL_ALPHA_PER_DEG
            source = "assumed"
        if slope_replacements is not None:
            slope_replacements.append({"computed_slope": (cl1 - cl0) / (a1 - a0)
                                       if a1 != a0 else None,
                                       "used_slope": slope,
                                       "source": source})
    if not math.isfinite(slope) or abs(slope) < SECANT_SLOPE_EPSILON:
        return None
    delta = (target - cl1) / slope
    delta = min(MAX_SUBSEQUENT_STEP_DEG,
                max(-MAX_SUBSEQUENT_STEP_DEG, delta))
    next_alpha = _bound_alpha(a1 + delta)
    return next_alpha if next_alpha != a1 else None


def secant_search(evaluate, target: float, alpha0: float,
                  tolerance: float = CL_TOLERANCE,
                  max_steps: int = MAX_SECANT_STEPS,
                  initial_slope: float | None = None) -> dict:
    """Evaluate a CL(alpha) callback until matched or secant steps are exhausted."""
    points: list[tuple[float, float]] = []
    alpha = _bound_alpha(alpha0)
    for step in range(max_steps + 1):
        cl = float(evaluate(alpha))
        points.append((alpha, cl))
        if abs(cl - target) <= tolerance:
            return {"status": "matched", "alpha": alpha, "cl": cl,
                    "steps": step, "points": points}
        if step == max_steps:
            break
        candidate = secant_next(points, target, initial_slope)
        if candidate is None:
            break
        alpha = candidate
    alpha, cl = points[-1]
    return {"status": "cl_not_matched", "alpha": alpha, "cl": cl,
            "steps": len(points) - 1, "points": points}


def drift_check(history: list[dict], minimum_iterations: int) -> dict:
    """Check the latest sample against the sample 100 iterations earlier."""
    if not history:
        return {"ok": False, "dcl_100": None, "dcd_100": None,
                "mass_imbalance": None, "mass_imbalance_ok": None,
                "iterations": 0}
    current = history[-1]
    iteration = int(current["iteration"])
    previous = [row for row in history[:-1]
                if int(row["iteration"]) <= iteration - DRIFT_WINDOW_ITERATIONS]
    if iteration < minimum_iterations or not previous:
        imbalance = abs(float(current["mass_imbalance"]))
        return {"ok": False, "dcl_100": None, "dcd_100": None,
                "mass_imbalance": imbalance,
                "mass_imbalance_ok": imbalance < MAX_MASS_IMBALANCE,
                "iterations": iteration}
    prior = max(previous, key=lambda row: int(row["iteration"]))
    dcl = abs(float(current["CL"]) - float(prior["CL"]))
    dcd = abs(float(current["CD"]) - float(prior["CD"]))
    imbalance = abs(float(current["mass_imbalance"]))
    mass_imbalance_ok = imbalance < MAX_MASS_IMBALANCE
    ok = dcd < MAX_CD_DRIFT and dcl < MAX_CL_DRIFT
    return {"ok": ok, "dcl_100": dcl, "dcd_100": dcd,
            "mass_imbalance": imbalance,
            "mass_imbalance_ok": mass_imbalance_ok,
            "iterations": iteration}
