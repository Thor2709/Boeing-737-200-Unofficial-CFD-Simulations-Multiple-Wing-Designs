"""Small, deterministic airfoil geometry operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


_BUNCHING_PARAMETER = 1.0
_TE_LE_PANEL_DENSITY_RATIO = 0.15
_REFINED_AREA_RATIO = 0.9
# CTRRAT only sets fictitious curvature inside a configured refinement zone.
# No refinement zone is enabled here, so its 0.9 setting is intentionally inert.
_PANGEN_REFINED_AREA_RATIO = _REFINED_AREA_RATIO
_PANGEN_IPFAC = 5
_PANGEN_TE_STEP_RATIO = 0.334


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _xy_array(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError("airfoil data must contain at least five x/y points")
    if not np.isfinite(points).all():
        raise ValueError("airfoil data contains a non-finite coordinate")
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12
    points = points[keep]
    if len(points) < 5:
        raise ValueError("airfoil data has fewer than five distinct consecutive points")
    return points.copy()


def _lednicer_to_selig(points: np.ndarray, upper_count: int, lower_count: int) -> np.ndarray:
    if upper_count < 3 or lower_count < 3 or upper_count + lower_count > len(points):
        raise ValueError("invalid Lednicer upper/lower point counts")
    upper = points[:upper_count]
    lower = points[upper_count:upper_count + lower_count]
    if len(upper) < 3 or len(lower) < 3:
        raise ValueError("Lednicer file does not contain two complete surfaces")
    if float(np.mean(upper[1:-1, 1])) < float(np.mean(lower[1:-1, 1])):
        upper, lower = lower, upper
    return np.vstack((upper[::-1], lower[1:]))


def read_dat(path: str | Path) -> np.ndarray:
    """Read two-column coordinates from Selig or Lednicer .dat files."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read airfoil file {source}: {exc}") from exc

    rows: list[tuple[float, float]] = []
    row_lines: list[int] = []
    for line_number, line in enumerate(lines):
        fields = line.replace(",", " ").split()
        if len(fields) != 2:
            continue
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            rows.append((x, y))
            row_lines.append(line_number)

    if not rows:
        raise ValueError(f"no numeric x/y coordinates found in {source}")

    points = np.asarray(rows, dtype=float)
    first_x, first_y = points[0]
    counts = (int(round(first_x)), int(round(first_y)))
    first_is_count = (
        first_x >= 3
        and first_y >= 3
        and abs(first_x - counts[0]) < 1e-9
        and abs(first_y - counts[1]) < 1e-9
        and len(points) - 1 >= counts[0] + counts[1]
    )
    if first_is_count:
        points = _lednicer_to_selig(points[1:], *counts)
    elif len(points) >= 3:
        points = _detect_unlabelled_lednicer(points)

    return _xy_array(points)


def _detect_unlabelled_lednicer(points: np.ndarray) -> np.ndarray:
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diameter > 1e-12 and float(np.linalg.norm(points[0] - points[-1])) <= 0.1 * diameter:
        return points
    x = points[:, 0]
    scale = float(np.ptp(x))
    if scale <= 1e-12:
        return points
    split = int(np.argmax(x))
    looks_lednicer = (
        split >= 3
        and len(points) - split - 1 >= 3
        and abs(x[0] - np.min(x)) <= 0.025 * scale
        and abs(x[split + 1] - np.min(x)) <= 0.05 * scale
        and x[-1] >= np.max(x) - 0.05 * scale
    )
    if not looks_lednicer:
        return points
    return _lednicer_to_selig(points, split + 1, len(points) - split - 1)


def normalise(xy: np.ndarray) -> np.ndarray:
    """Return a unit-chord Selig loop using the continuous spline leading edge."""
    points = _xy_array(xy)
    points = _detect_unlabelled_lednicer(points)
    current_te = 0.5 * (points[0] + points[-1])
    current_le_index = int(np.argmin(np.linalg.norm(points, axis=1)))

    if float(np.linalg.norm(points[0] - points[-1])) > 0.25 * float(np.linalg.norm(np.ptp(points, axis=0))):
        raise ValueError("airfoil point order is neither Selig nor Lednicer")

    te_midpoint = 0.5 * (points[0] + points[-1])
    split_index = int(np.argmax(np.linalg.norm(points - te_midpoint, axis=1)))
    if split_index < 3 or split_index > len(points) - 4:
        raise ValueError("cannot locate an interior leading edge in airfoil data")
    upper_curve, upper_length = _arc_spline(points[:split_index + 1][::-1])
    lower_curve, lower_length = _arc_spline(points[split_index:])
    upper_parameter, upper_le, upper_distance = _farthest_spline_point(upper_curve, upper_length, te_midpoint)
    lower_parameter, lower_le, lower_distance = _farthest_spline_point(lower_curve, lower_length, te_midpoint)
    if upper_distance >= lower_distance:
        leading_edge = upper_le
        parameter = upper_parameter
        knots = upper_curve.x
        knot_index = int(np.searchsorted(knots, parameter, side="left"))
        if knot_index < len(knots) and abs(float(knots[knot_index]) - parameter) <= 1e-12:
            insertion = split_index - knot_index
        else:
            insertion = split_index - knot_index + 1
    else:
        leading_edge = lower_le
        parameter = lower_parameter
        knots = lower_curve.x
        knot_index = int(np.searchsorted(knots, parameter, side="left"))
        insertion = split_index + knot_index

    if (
        abs(float(current_te[0]) - 1.0) <= 1e-10
        and abs(float(current_te[1])) <= 1e-10
        and np.linalg.norm(points[current_le_index]) <= 1e-10
        and np.linalg.norm(leading_edge) <= 1e-5
    ):
        return points.copy()
    chord_vector = te_midpoint - leading_edge
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 1e-12:
        raise ValueError("airfoil has zero chord")
    chord_direction = chord_vector / chord

    insertion = min(max(int(insertion), 1), len(points) - 2)
    contour = np.vstack((points[:insertion], leading_edge, points[insertion + 1:]))
    relative = contour - leading_edge
    along = relative @ chord_direction / chord
    normal = (chord_direction[0] * relative[:, 1] - chord_direction[1] * relative[:, 0]) / chord
    result = np.column_stack((along, normal))
    le_index = insertion
    result[le_index] = (0.0, 0.0)

    upper_mean = float(np.mean(result[:le_index, 1]))
    lower_mean = float(np.mean(result[le_index + 1:, 1]))
    if upper_mean < lower_mean:
        result = result[::-1].copy()
        le_index = len(result) - 1 - le_index
    if float(np.mean(result[:le_index, 1])) < float(np.mean(result[le_index + 1:, 1])):
        result[:, 1] *= -1.0
    result[le_index] = (0.0, 0.0)
    return result


def naca4(m: float, p: float, t: float, closed_te: bool = True) -> np.ndarray:
    """Generate a four-digit NACA section in Selig order using cosine x spacing."""
    m, p, t = float(m), float(p), float(t)
    if not (0.0 <= m < 1.0 and 0.0 <= p < 1.0 and 0.0 < t < 1.0):
        raise ValueError("NACA 4-digit parameters require 0<=m,p<1 and 0<t<1")
    x_le_te = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))
    coefficient = -0.1036 if closed_te else -0.1015
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x_le_te)
        - 0.1260 * x_le_te
        - 0.3516 * x_le_te**2
        + 0.2843 * x_le_te**3
        + coefficient * x_le_te**4
    )
    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x_le_te)
        dyc = np.zeros_like(x_le_te)
    else:
        front = x_le_te < p
        yc = np.empty_like(x_le_te)
        dyc = np.empty_like(x_le_te)
        yc[front] = m / p**2 * (2.0 * p * x_le_te[front] - x_le_te[front] ** 2)
        dyc[front] = 2.0 * m / p**2 * (p - x_le_te[front])
        yc[~front] = m / (1.0 - p) ** 2 * (
            (1.0 - 2.0 * p) + 2.0 * p * x_le_te[~front] - x_le_te[~front] ** 2
        )
        dyc[~front] = 2.0 * m / (1.0 - p) ** 2 * (p - x_le_te[~front])
    theta = np.arctan(dyc)
    upper = np.column_stack((x_le_te - yt * np.sin(theta), yc + yt * np.cos(theta)))
    lower = np.column_stack((x_le_te + yt * np.sin(theta), yc - yt * np.cos(theta)))
    return np.vstack((upper[::-1], lower[1:]))


def _surface_splines(xy: np.ndarray) -> tuple[CubicSpline, float, CubicSpline, float, int]:
    points = _xy_array(xy)
    le_index = _leading_edge_index(points)
    if le_index < 2 or le_index > len(points) - 3:
        raise ValueError("airfoil Selig loop must contain upper and lower points around the LE")
    upper = points[:le_index + 1][::-1]
    lower = points[le_index:]
    upper_curve, upper_length = _arc_spline(upper)
    lower_curve, lower_length = _arc_spline(lower)
    return upper_curve, upper_length, lower_curve, lower_length, le_index


def _leading_edge_index(points: np.ndarray) -> int:
    origin_distance = np.linalg.norm(points, axis=1)
    index = int(np.argmin(origin_distance))
    if float(origin_distance[index]) <= 1e-10:
        return index
    return int(np.argmin(points[:, 0]))


def _arc_spline(points: np.ndarray) -> tuple[CubicSpline, float]:
    points = _xy_array(points)
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-12]
    points, lengths = points[keep], lengths[keep]
    if len(points) < 4 or lengths[-1] <= 1e-12:
        raise ValueError("surface needs four distinct points to build an arc-length spline")
    start_tangent = (points[1] - points[0]) / (lengths[1] - lengths[0])
    end_tangent = (points[-1] - points[-2]) / (lengths[-1] - lengths[-2])
    boundary = ((1, start_tangent), (1, end_tangent))
    return CubicSpline(lengths, points, axis=0, bc_type=boundary), float(lengths[-1])


def _farthest_spline_point(
    curve: CubicSpline, length: float, reference: np.ndarray
) -> tuple[float, np.ndarray, float]:
    knots = np.asarray(curve.x, dtype=float)
    samples = np.concatenate(
        [np.linspace(left, right, 5, endpoint=False) for left, right in zip(knots[:-1], knots[1:])] + [knots[-1:]]
    )
    derivative = np.einsum(
        "ij,ij->i", np.asarray(curve(samples)) - reference, np.asarray(curve(samples, 1))
    )
    candidates = [0.0, length]
    for left, right, f_left, f_right in zip(samples[:-1], samples[1:], derivative[:-1], derivative[1:]):
        if f_left == 0.0:
            candidates.append(float(left))
        elif f_left * f_right < 0.0:
            candidates.append(float(brentq(
                lambda value: float(np.dot(curve(value) - reference, curve(value, 1))),
                float(left),
                float(right),
                xtol=1e-14,
            )))
    locations = np.asarray(curve(np.asarray(candidates)), dtype=float)
    distances = np.linalg.norm(locations - reference, axis=1)
    index = int(np.argmax(distances))
    return candidates[index], locations[index], float(distances[index])


def _curvature(curve: CubicSpline, parameter: np.ndarray | float) -> np.ndarray | float:
    first = np.asarray(curve(parameter, 1), dtype=float)
    second = np.asarray(curve(parameter, 2), dtype=float)
    speed = np.maximum(np.linalg.norm(first, axis=-1), 1e-12)
    value = np.abs(first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]) / speed**3
    return float(value) if np.ndim(value) == 0 else value


def _smooth_curvature(
    curve: CubicSpline,
    length: float,
    cv_le: float,
    cv_average: float,
    sb_reference: float,
    panel_count: int,
) -> CubicSpline:
    parameter = np.asarray(curve.x, dtype=float)
    values = np.asarray(_curvature(curve, parameter), dtype=float) * sb_reference
    values[0] = cv_le
    values[-1] = cv_average * _TE_LE_PANEL_DENSITY_RATIO

    smooth_length = min(
        0.05,
        max(1.0 / max(cv_average, 20.0), 0.25 / max(panel_count / 2.0, 1.0)),
    ) * sb_reference
    smooth_squared = smooth_length**2
    interior_count = len(parameter) - 2
    if interior_count > 0 and smooth_squared > 0.0:
        lower = np.zeros(interior_count)
        diagonal = np.ones(interior_count)
        upper = np.zeros(interior_count)
        rhs = values[1:-1].copy()
        for row, index in enumerate(range(1, len(parameter) - 1)):
            previous_step = parameter[index] - parameter[index - 1]
            next_step = parameter[index + 1] - parameter[index]
            centered_step = 0.5 * (parameter[index + 1] - parameter[index - 1])
            lower[row] = -smooth_squared / (previous_step * centered_step)
            diagonal[row] += smooth_squared * (1.0 / next_step + 1.0 / previous_step) / centered_step
            upper[row] = -smooth_squared / (next_step * centered_step)
        rhs[0] -= lower[0] * values[0]
        rhs[-1] -= upper[-1] * values[-1]
        banded = np.zeros((3, interior_count))
        banded[1] = diagonal
        banded[0, 1:] = upper[:-1]
        banded[2, :-1] = lower[1:]
        values[1:-1] = solve_banded((1, 1), banded, rhs)

    maximum = float(np.max(np.abs(values)))
    if maximum <= 1e-12:
        values.fill(0.0)
    else:
        values /= maximum
    return CubicSpline(parameter, values)


def _newton_panel_positions(
    length: float,
    panel_count: int,
    curvature: CubicSpline,
) -> np.ndarray:
    fine_panel_count = _PANGEN_IPFAC * panel_count
    te_ratio = _PANGEN_TE_STEP_RATIO
    average_step = length / (fine_panel_count - 1 + te_ratio)
    positions = np.r_[np.arange(fine_panel_count, dtype=float) * average_step, length]
    curvature_weight = 6.0 * _BUNCHING_PARAMETER

    for _ in range(20):
        values = np.asarray(curvature(positions), dtype=float)
        derivatives = np.asarray(curvature(positions, 1), dtype=float)
        interior = len(positions) - 2
        lower = np.zeros(interior)
        diagonal = np.zeros(interior)
        upper = np.zeros(interior)
        rhs = np.zeros(interior)

        for row, index in enumerate(range(1, len(positions) - 1)):
            previous = index - 1
            following = index + 1
            previous_step = positions[index] - positions[previous]
            next_step = positions[following] - positions[index]
            average_previous = np.hypot(values[previous], values[index])
            average_next = np.hypot(values[index], values[following])
            if average_previous > 1e-14:
                previous_derivative = values[previous] * derivatives[previous] / average_previous
                center_previous_derivative = values[index] * derivatives[index] / average_previous
            else:
                previous_derivative = center_previous_derivative = 0.0
            if average_next > 1e-14:
                center_next_derivative = values[index] * derivatives[index] / average_next
                following_derivative = values[following] * derivatives[following] / average_next
            else:
                center_next_derivative = following_derivative = 0.0

            factor_previous = 1.0 + curvature_weight * average_previous
            factor_next = 1.0 + curvature_weight * average_next
            residual = previous_step * factor_previous - next_step * factor_next
            lower[row] = -factor_previous + curvature_weight * previous_step * previous_derivative
            diagonal[row] = (
                factor_previous
                + factor_next
                + curvature_weight * (
                    previous_step * center_previous_derivative - next_step * center_next_derivative
                )
            )
            upper[row] = -factor_next - curvature_weight * next_step * following_derivative
            rhs[row] = -residual

        # XFOIL's terminal-panel constraint keeps the TE panel at 0.334 of
        # its neighbour while CTERAT controls the broader TE curvature field.
        previous_step = positions[-2] - positions[-3]
        te_step = positions[-1] - positions[-2]
        lower[-1] = -te_ratio
        diagonal[-1] = 1.0 + te_ratio
        upper[-1] = 0.0
        rhs[-1] = te_step - te_ratio * previous_step

        banded = np.zeros((3, interior))
        banded[1] = diagonal
        banded[0, 1:] = upper[:-1]
        banded[2, :-1] = lower[1:]
        changes = np.zeros(len(positions))
        changes[1:-1] = solve_banded((1, 1), banded, rhs)

        relaxation = 1.0
        maximum_change = 0.0
        for index in range(len(positions) - 1):
            step = positions[index + 1] - positions[index]
            delta_step = changes[index + 1] - changes[index]
            step_ratio = 1.0 + relaxation * delta_step / step
            if step_ratio > 4.0:
                relaxation = min(relaxation, 3.0 * step / delta_step)
            elif step_ratio < 0.2:
                relaxation = min(relaxation, -0.8 * step / delta_step)
            maximum_change = max(maximum_change, abs(changes[index]))
        positions[1:-1] += relaxation * changes[1:-1]
        if maximum_change * relaxation < 1e-11:
            break

    return positions[::_PANGEN_IPFAC]


def _distribute_surface(
    curve: CubicSpline,
    length: float,
    panel_count: int,
    cv_le: float,
    cv_average: float,
    sb_reference: float,
    total_panel_count: int,
) -> np.ndarray:
    if panel_count < 1:
        raise ValueError("each airfoil surface needs at least one panel")
    curvature = _smooth_curvature(curve, length, cv_le, cv_average, sb_reference, total_panel_count)
    parameter = _newton_panel_positions(length, panel_count, curvature)
    result = np.asarray(curve(parameter), dtype=float)
    result[0] = np.asarray(curve(0.0), dtype=float)
    result[-1] = np.asarray(curve(length), dtype=float)
    return result


def _repanel_selig(xy: np.ndarray, n_panels: int = 200) -> np.ndarray:
    if int(n_panels) != n_panels or n_panels < 4:
        raise ValueError("n_panels must be an integer of at least four")
    upper_curve, upper_length, lower_curve, lower_length, _ = _surface_splines(xy)
    n_upper = int(n_panels) // 2
    n_lower = int(n_panels) - n_upper
    sb_reference = 0.5 * (upper_length + lower_length)
    upper_cv_le = float(_curvature(upper_curve, 0.0)) * sb_reference
    lower_cv_le = float(_curvature(lower_curve, 0.0)) * sb_reference
    cv_le = 0.5 * (upper_cv_le + lower_cv_le)
    average_radius = sb_reference / max(cv_le, 20.0)
    curvature_samples: list[float] = []
    for offset in np.linspace(-average_radius, average_radius, 7):
        if offset < 0.0:
            parameter = min(-float(offset), upper_length)
            curve = upper_curve
        else:
            parameter = min(float(offset), lower_length)
            curve = lower_curve
        curvature_samples.append(float(_curvature(curve, parameter)) * sb_reference)
    cv_average = float(np.mean(curvature_samples))

    upper_le_te = _distribute_surface(
        upper_curve, upper_length, n_upper, cv_le, cv_average, sb_reference, int(n_panels)
    )
    lower_le_te = _distribute_surface(
        lower_curve, lower_length, n_lower, cv_le, cv_average, sb_reference, int(n_panels)
    )
    result = np.vstack((upper_le_te[::-1], lower_le_te[1:]))
    if not np.isfinite(result).all():
        raise ValueError("repanel produced non-finite coordinates")
    return result


def repanel(xy: np.ndarray, n_panels: int = 200) -> np.ndarray:
    """Repanel both surfaces by curvature-weighted arc length; return n+1 Selig points."""
    return _repanel_selig(normalise(xy), n_panels)


def _interpolate_surface(surface_le_te: np.ndarray, x: np.ndarray) -> np.ndarray:
    order = np.argsort(surface_le_te[:, 0], kind="mergesort")
    xs, ys = surface_le_te[order, 0], surface_le_te[order, 1]
    unique_x, inverse = np.unique(xs, return_inverse=True)
    if len(unique_x) != len(xs):
        sums = np.bincount(inverse, weights=ys)
        counts = np.bincount(inverse)
        ys = sums / counts
        xs = unique_x
    return np.interp(x, xs, ys)


def _curvature_changes(curve: CubicSpline, length: float) -> int:
    parameter = np.linspace(0.0, length, 1001)
    first, second = curve(parameter, 1), curve(parameter, 2)
    speed = np.maximum(np.linalg.norm(first, axis=1), 1e-12)
    curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / speed**3
    threshold = max(float(np.max(np.abs(curvature))) * 1e-4, 1e-9)
    signs = np.sign(curvature[np.abs(curvature) > threshold])
    if len(signs) < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _leading_edge_radius(surface: np.ndarray) -> float:
    sample = np.asarray(surface[:min(4, len(surface))], dtype=float)
    if len(sample) < 4:
        return float("nan")
    matrix = np.column_stack((2.0 * sample[:, 0], 2.0 * sample[:, 1], np.ones(len(sample))))
    rhs = sample[:, 0] ** 2 + sample[:, 1] ** 2
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 3:
        return float("nan")
    center_x, center_y, offset = coefficients
    radius_squared = center_x**2 + center_y**2 + offset
    if radius_squared <= 0.0:
        return float("nan")
    return float(np.sqrt(radius_squared))


def _signed_max_camber_on_chord(points: np.ndarray, le_index: int) -> float:
    leading_edge = points[le_index]
    trailing_edge = 0.5 * (points[0] + points[-1])
    chord_vector = trailing_edge - leading_edge
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 1e-12:
        return float("nan")
    direction = chord_vector / chord
    relative = points - leading_edge
    chord_x = relative @ direction / chord
    chord_y = (direction[0] * relative[:, 1] - direction[1] * relative[:, 0]) / chord
    upper = np.column_stack((chord_x[:le_index + 1][::-1], chord_y[:le_index + 1][::-1]))
    lower = np.column_stack((chord_x[le_index:], chord_y[le_index:]))
    x_grid = np.linspace(0.0, 1.0, 5001)
    camber = 0.5 * (_interpolate_surface(upper, x_grid) + _interpolate_surface(lower, x_grid))
    return float(camber[int(np.argmax(np.abs(camber)))])


def self_intersects(xy: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Return whether nonadjacent contour segments cross or touch."""
    points = _xy_array(xy)
    starts, ends = points[:-1], points[1:]
    segment_count = len(starts)
    for i in range(segment_count):
        a, b = starts[i], ends[i]
        for j in range(i + 2, segment_count):
            if i == 0 and j == segment_count - 1:
                continue
            c, d = starts[j], ends[j]
            if (
                max(a[0], b[0]) < min(c[0], d[0]) - tolerance
                or max(c[0], d[0]) < min(a[0], b[0]) - tolerance
                or max(a[1], b[1]) < min(c[1], d[1]) - tolerance
                or max(c[1], d[1]) < min(a[1], b[1]) - tolerance
            ):
                continue
            o1 = _cross2(b - a, c - a)
            o2 = _cross2(b - a, d - a)
            o3 = _cross2(d - c, a - c)
            o4 = _cross2(d - c, b - c)
            if ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)) and (
                (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
            ):
                return True
            if abs(o1) <= tolerance or abs(o2) <= tolerance or abs(o3) <= tolerance or abs(o4) <= tolerance:
                if _collinear_touch(a, b, c, d, tolerance):
                    return True
    return False


def _collinear_touch(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tol: float) -> bool:
    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol
            and min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol
        )

    for p, q, r in ((a, c, b), (a, d, b), (c, a, d), (c, b, d)):
        cross = _cross2(r - p, q - p)
        if abs(cross) <= tol and on_segment(p, q, r):
            return True
    return False


def geometry_metrics(xy: np.ndarray, *, normalize_input: bool = True) -> dict[str, float | int | bool]:
    """Compute thickness, camber, TE gap, LE radius, and curvature metrics."""
    points = normalise(xy) if normalize_input else _xy_array(xy)
    upper_curve, upper_length, lower_curve, lower_length, le_index = _surface_splines(points)
    upper = points[:le_index + 1][::-1]
    lower = points[le_index:]
    x_grid = np.linspace(0.0, 1.0, 5001)
    y_upper = _interpolate_surface(upper, x_grid)
    y_lower = _interpolate_surface(lower, x_grid)
    thickness = y_upper - y_lower
    camber_line = 0.5 * (y_upper + y_lower)
    thickness_index = int(np.argmax(thickness))
    camber_index = int(np.argmax(camber_line))
    trailing_edge = 0.5 * (points[0] + points[-1])

    upper_radius = _leading_edge_radius(upper)
    lower_radius = _leading_edge_radius(lower)
    valid_radii = [radius for radius in (upper_radius, lower_radius) if np.isfinite(radius) and radius > 0.0]
    le_radius = float(np.mean(valid_radii)) if valid_radii else float("nan")

    return {
        "t_c": float(thickness[thickness_index]),
        "x_t": float(x_grid[thickness_index]),
        "camber": float(camber_line[camber_index]),
        "x_camber": float(x_grid[camber_index]),
        "te_gap": float(points[0, 1] - points[-1, 1]),
        "te_x": float(trailing_edge[0]),
        "te_y": float(trailing_edge[1]),
        "camber_signed_max_abs": _signed_max_camber_on_chord(points, le_index),
        "le_radius": le_radius,
        "curvature_sign_changes_upper": _curvature_changes(upper_curve, upper_length),
        "curvature_sign_changes_lower": _curvature_changes(lower_curve, lower_length),
        "self_intersect": self_intersects(points),
    }


def _close_te(xy: np.ndarray) -> np.ndarray:
    points = _xy_array(xy)
    result = points.copy()
    result[0, 1] = 0.0
    result[-1, 1] = 0.0
    result[0, 0] = result[-1, 0] = 1.0
    return result


def blunt_te(xy: np.ndarray, gap_c: float) -> np.ndarray:
    """Set total TE thickness with an x-squared offset in the chord frame."""
    gap_c = float(gap_c)
    if not np.isfinite(gap_c) or gap_c < 0.0:
        raise ValueError("gap_c must be a finite nonnegative chord fraction")
    points = _xy_array(xy)
    le_index = _leading_edge_index(points)
    upper_gap, lower_gap = points[0, 1], points[-1, 1]
    current_gap = float(upper_gap - lower_gap)
    if current_gap < -1e-9:
        raise ValueError("airfoil TE order is inverted")
    added_gap = gap_c - current_gap
    result = points.copy()
    result[:le_index, 1] += 0.5 * added_gap * result[:le_index, 0] ** 2
    result[le_index + 1:, 1] -= 0.5 * added_gap * result[le_index + 1:, 0] ** 2
    return result if len(result) == 201 else _repanel_selig(result, 200)


def _rotate(points: np.ndarray, hinge: np.ndarray, angle: float) -> np.ndarray:
    relative = np.asarray(points, dtype=float) - hinge
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, sine), (-sine, cosine)))
    return relative @ rotation.T + hinge


def _parameter_at_x(curve: CubicSpline, length: float, x_value: float) -> float:
    parameter = np.linspace(0.0, length, 4001)
    x = curve(parameter)[:, 0]
    order = np.argsort(x, kind="mergesort")
    xs, ts = x[order], parameter[order]
    unique_x, indices = np.unique(xs, return_index=True)
    ts = ts[indices]
    if x_value < unique_x[0] - 1e-8 or x_value > unique_x[-1] + 1e-8:
        raise ValueError("hinge x/c lies outside the airfoil chord")
    return float(np.interp(x_value, unique_x, ts))


def _circle_tangent_parameter(curve: CubicSpline, length: float, hinge: np.ndarray, target_x: float) -> float:
    parameter = np.linspace(0.0, length, 8001)
    points = curve(parameter)
    tangents = curve(parameter, 1)
    radial_derivative = np.einsum("ij,ij->i", points - hinge, tangents)
    in_window = np.abs(points[:, 0] - target_x) <= 0.18
    roots: list[float] = []
    for index in range(len(parameter) - 1):
        if not (in_window[index] or in_window[index + 1]):
            continue
        if radial_derivative[index] == 0.0:
            roots.append(float(parameter[index]))
        elif radial_derivative[index] * radial_derivative[index + 1] < 0.0:
            root = brentq(
                lambda value: float(np.dot(curve(value) - hinge, curve(value, 1))),
                float(parameter[index]),
                float(parameter[index + 1]),
            )
            roots.append(float(root))
    if not roots:
        raise ValueError("upper surface has no hinge-centred circular tangent near the requested hinge")
    return min(roots, key=lambda value: abs(float(curve(value)[0]) - target_x))


def _segment_intersections(first: np.ndarray, second: np.ndarray) -> list[tuple[int, int, float, float, np.ndarray]]:
    result: list[tuple[int, int, float, float, np.ndarray]] = []
    first_starts, first_ends = first[:-1], first[1:]
    second_starts, second_ends = second[:-1], second[1:]
    second_min = np.minimum(second_starts, second_ends)
    second_max = np.maximum(second_starts, second_ends)
    for i, (a, b) in enumerate(zip(first_starts, first_ends)):
        low, high = np.minimum(a, b), np.maximum(a, b)
        candidates = np.flatnonzero(
            (second_max[:, 0] >= low[0] - 1e-10)
            & (second_min[:, 0] <= high[0] + 1e-10)
            & (second_max[:, 1] >= low[1] - 1e-10)
            & (second_min[:, 1] <= high[1] + 1e-10)
        )
        vector_a = b - a
        for j in candidates:
            c, d = second_starts[j], second_ends[j]
            vector_b = d - c
            denominator = _cross2(vector_a, vector_b)
            if abs(denominator) <= 1e-14:
                continue
            delta = c - a
            t = _cross2(delta, vector_b) / denominator
            u = _cross2(delta, vector_a) / denominator
            if -1e-10 <= t <= 1.0 + 1e-10 and -1e-10 <= u <= 1.0 + 1e-10:
                location = a + np.clip(t, 0.0, 1.0) * vector_a
                result.append((i, int(j), float(np.clip(t, 0.0, 1.0)), float(np.clip(u, 0.0, 1.0)), location))
    return result


def flap(xy: np.ndarray, hinge_xc: float, defl_deg: float) -> np.ndarray:
    """Create a single-element plain flap with a tangent upper cove and trimmed lower join."""
    hinge_xc, defl_deg = float(hinge_xc), float(defl_deg)
    if not (0.0 < hinge_xc < 1.0) or not np.isfinite(defl_deg):
        raise ValueError("flap requires 0 < hinge_xc < 1 and a finite deflection")
    points = normalise(xy)
    upper_curve, upper_length, lower_curve, lower_length, _ = _surface_splines(points)
    hinge_parameter = _parameter_at_x(lower_curve, lower_length, hinge_xc)
    hinge_upper_parameter = _parameter_at_x(upper_curve, upper_length, hinge_xc)
    upper_at_hinge = np.asarray(upper_curve(hinge_upper_parameter), dtype=float)
    lower_at_hinge = np.asarray(lower_curve(hinge_parameter), dtype=float)
    hinge = np.array((hinge_xc, 0.5 * (upper_at_hinge[1] + lower_at_hinge[1])))
    angle = np.deg2rad(defl_deg)

    tangent_parameter = _circle_tangent_parameter(upper_curve, upper_length, hinge, hinge_xc)
    tangent_point = np.asarray(upper_curve(tangent_parameter), dtype=float)
    tangent_rotated = _rotate(tangent_point, hinge, angle)
    start_angle = float(np.arctan2(tangent_rotated[1] - hinge[1], tangent_rotated[0] - hinge[0]))
    arc_count = max(10, int(np.ceil(abs(defl_deg) / 2.0)) + 1)
    arc_angles = np.linspace(start_angle, start_angle + angle, arc_count)
    radius = float(np.linalg.norm(tangent_point - hinge))
    upper_arc = hinge + radius * np.column_stack((np.cos(arc_angles), np.sin(arc_angles)))
    upper_arc[0], upper_arc[-1] = tangent_rotated, tangent_point

    upper_parameter = np.linspace(0.0, upper_length, 2401)
    upper_le_te = np.asarray(upper_curve(upper_parameter), dtype=float)
    upper_tangent_index = int(np.argmin(np.abs(upper_parameter - tangent_parameter)))
    upper_aft = _rotate(upper_le_te[upper_tangent_index:], hinge, angle)[::-1]
    upper_fore = upper_le_te[:upper_tangent_index + 1][::-1]
    upper_loop = np.vstack((upper_aft, upper_arc[1:], upper_fore[1:]))

    lower_fixed_parameter = np.linspace(0.0, hinge_parameter, 1201)
    lower_aft_parameter = np.linspace(hinge_parameter, lower_length, 2401)
    lower_fixed = np.asarray(lower_curve(lower_fixed_parameter), dtype=float)
    lower_aft = _rotate(np.asarray(lower_curve(lower_aft_parameter), dtype=float), hinge, angle)
    intersections = _segment_intersections(lower_fixed, lower_aft)
    if not intersections:
        raise ValueError("deflected lower flap and fixed lower surface do not intersect for trimming")
    fixed_i, aft_j, _, _, intersection = min(
        intersections,
        key=lambda item: float(np.linalg.norm(item[4] - hinge)),
    )
    lower_loop = np.vstack((lower_fixed[:fixed_i + 1], intersection, lower_aft[aft_j + 1:]))

    raw_flap = np.vstack((upper_loop, lower_loop[1:]))
    if self_intersects(raw_flap):
        raise ValueError("flapped airfoil contains a self-intersection")
    repanelled = _repanel_selig(raw_flap, 200)
    if self_intersects(repanelled):
        raise ValueError("repanelled flapped airfoil contains a self-intersection")
    return repanelled


def smooth_digitised(xy: np.ndarray, factor: float = 3.0e-9, max_dev_c: float = 1.0e-3) -> np.ndarray:
    """Remove digitising noise with a quintic smoothing spline on y(sqrt(x)) per surface.

    Opus decision 2026-10-03: B737B's raw wiggles stop XFoil converging above
    alpha 3.5 deg at Re >= 2e7. factor 3e-9 moves the surface <= 0.053 % c, keeps
    t/c and camber, and matches raw cl within 0.5 % where both converge.
    """
    from scipy.interpolate import UnivariateSpline

    points = normalise(xy)
    le = int(np.argmin(points[:, 0]))
    upper, lower = points[: le + 1][::-1], points[le:]
    s = np.linspace(0.0, 1.0, 401)
    x = s**2
    smoothed = []
    for surface in (upper, lower):
        y = np.interp(x, surface[:, 0], surface[:, 1])
        ys = UnivariateSpline(s, y, s=factor * len(s), k=5)(s)
        ys[0] = ys[-1] = 0.0
        if np.abs(ys - y).max() > max_dev_c:
            raise ValueError("smoothing moved the surface more than max_dev_c")
        smoothed.append(ys)
    out = np.vstack([np.c_[x[::-1], smoothed[0][::-1]], np.c_[x[1:], smoothed[1][1:]]])
    return repanel(out, 200)
