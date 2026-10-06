from pathlib import Path

import numpy as np

from b737wing.aero import build_foils
from b737wing.aero.foils import (
    _interpolate_surface,
    _rotate,
    _surface_splines,
    _circle_tangent_parameter,
    blunt_te,
    flap,
    geometry_metrics,
    naca4,
    normalise,
    read_dat,
    repanel,
    self_intersects,
)


from b737wing.config import AEROFOIL_DIR, REPO_ROOT

ROOT = REPO_ROOT


def _closed_te(points):
    result = normalise(points)
    result[0, 1] = result[-1, 1] = 0.0
    return result


def _surface_values(points, x_values, *, normalize_points=True):
    normalized = normalise(points) if normalize_points else np.asarray(points, dtype=float)
    le = int(np.argmin(normalized[:, 0]))
    upper = normalized[:le + 1][::-1]
    lower = normalized[le:]
    return _interpolate_surface(upper, x_values), _interpolate_surface(lower, x_values)


def test_naca0012_repaneled_metrics_and_symmetry():
    points = repanel(naca4(0.0, 0.0, 0.12), 200)
    metrics = geometry_metrics(points)
    assert len(points) == 201
    assert abs(metrics["t_c"] - 0.12) <= 0.0005
    assert abs(metrics["camber"]) < 1e-6
    assert np.isfinite(metrics["le_radius"]) and metrics["le_radius"] > 0.0
    assert 0.01 < metrics["le_radius"] < 0.025
    x = np.linspace(0.0, 1.0, 1001)
    upper, lower = _surface_values(points, x)
    assert np.max(np.abs(upper + lower)) < 1e-6


def test_repanel_bunches_at_le_and_te_without_duplicate_points():
    points = repanel(naca4(0.0, 0.0, 0.12), 200)
    assert len(points) == 201
    assert np.min(np.linalg.norm(np.diff(points, axis=0), axis=1)) > 0.0
    le = int(np.argmin(points[:, 0]))
    le_panel = max(np.linalg.norm(points[le] - points[le - 1]), np.linalg.norm(points[le + 1] - points[le]))
    mid_index = int(np.argmin(np.abs(0.5 * (points[:-1, 0] + points[1:, 0]) - 0.5)))
    mid_panel = np.linalg.norm(points[mid_index + 1] - points[mid_index])
    te_panel = np.linalg.norm(points[1] - points[0])
    assert le_panel < mid_panel / 3.0
    panel_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert le_panel < te_panel < np.max(panel_lengths)
    assert 3.0 < te_panel / le_panel < 12.0


def test_normalise_is_invariant_to_rigid_transform_and_preserves_te_points():
    source = naca4(0.04, 0.4, 0.12, closed_te=False)
    normalized = normalise(source)
    assert abs(float(normalized[0, 0] - normalized[-1, 0])) > 1e-6

    angle = np.deg2rad(30.0)
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    transformed = 3.25 * (source @ rotation.T) + np.array((4.7, -2.3))
    normalized_transformed = normalise(transformed)

    assert np.max(np.abs(normalized_transformed - normalized)) < 1e-6
    assert np.allclose(normalized_transformed[[0, -1]], normalized[[0, -1]], atol=1e-6)


def test_repanel_includes_continuous_spline_le_for_naca4412():
    source = normalise(naca4(0.04, 0.4, 0.12))
    upper_curve, _, _, _, _ = _surface_splines(source)
    expected_le = np.asarray(upper_curve(0.0), dtype=float)
    panelled = repanel(source, 200)
    le_index = int(np.argmin(np.linalg.norm(panelled, axis=1)))

    assert len(panelled) == 201
    assert np.allclose(panelled[le_index], expected_le, atol=1e-12)


def test_aopt_geometry_matches_header_claim():
    points = normalise(read_dat(AEROFOIL_DIR / "LatestOptimised.dat"))
    metrics = geometry_metrics(points)
    assert abs(metrics["camber"] - 0.0262) <= 0.001
    assert abs(metrics["x_camber"] - 0.356) <= 0.02
    assert abs(metrics["t_c"] - 0.12) <= 0.002


def test_flap_hinge_geometry_trailing_edge_and_loop():
    source = normalise(read_dat(AEROFOIL_DIR / "LatestOptimised.dat"))
    result = flap(source, 0.6483810, 21.35900)
    assert len(result) == 201
    assert not self_intersects(result)

    xs = np.linspace(0.05, 0.64, 100)
    source_upper, source_lower = _surface_values(source, xs)
    flap_upper, flap_lower = _surface_values(result, xs, normalize_points=False)
    assert np.max(np.abs(source_upper - flap_upper)) < 2e-4
    fore_xs = np.linspace(0.05, 0.5, 100)
    _, source_lower = _surface_values(source, fore_xs)
    _, flap_lower = _surface_values(result, fore_xs, normalize_points=False)
    assert np.max(np.abs(source_lower - flap_lower)) < 2e-4

    expected_down = (1.0 - 0.6483810) * np.sin(np.deg2rad(21.35900))
    te_y = 0.5 * (result[0, 1] + result[-1, 1])
    assert abs(te_y + expected_down) < 0.01
    source_upper, source_lower = _surface_values(source, np.array((0.6483810,)))
    hinge = np.array((0.6483810, 0.5 * (source_upper[0] + source_lower[0])))
    assert np.allclose(_rotate(hinge, hinge, np.deg2rad(21.35900)), hinge, atol=1e-12)
    source_upper_curve, source_upper_length, _, _, _ = _surface_splines(_closed_te(source))
    tangent_parameter = _circle_tangent_parameter(source_upper_curve, source_upper_length, hinge, 0.6483810)
    tangent_point = source_upper_curve(tangent_parameter)
    result_le = int(np.argmin(result[:, 0]))
    upper_result = result[:result_le + 1]
    local = upper_result[(upper_result[:, 0] > 0.63) & (upper_result[:, 0] < 0.68)]
    expected_radius = np.linalg.norm(tangent_point - hinge)
    radius_errors = np.abs(np.linalg.norm(local - hinge, axis=1) - expected_radius)
    assert np.count_nonzero(radius_errors < 0.003) >= 2


def test_flapped_metrics_keep_source_chord_frame_and_report_actual_chord_camber():
    source = normalise(read_dat(AEROFOIL_DIR / "LatestOptimised.dat"))
    result = flap(source, 0.6483810, 21.35900)
    metrics = geometry_metrics(result, normalize_input=False)
    te_midpoint = 0.5 * (result[0] + result[-1])

    assert np.allclose(result[int(np.argmin(result[:, 0]))], (0.0, 0.0), atol=1e-12)
    assert te_midpoint[0] < 1.0
    assert te_midpoint[1] < 0.0
    assert np.allclose((metrics["te_x"], metrics["te_y"]), te_midpoint, atol=1e-12)

    le_index = int(np.argmin(result[:, 0]))
    chord_vector = te_midpoint - result[le_index]
    chord_length = np.linalg.norm(chord_vector)
    chord = chord_vector / chord_length
    relative = result - result[le_index]
    projected = np.column_stack((relative @ chord / chord_length, (chord[0] * relative[:, 1] - chord[1] * relative[:, 0]) / chord_length))
    upper = projected[:le_index + 1][::-1]
    lower = projected[le_index:]
    x_grid = np.linspace(0.0, 1.0, 5001)
    signed_camber = 0.5 * (_interpolate_surface(upper, x_grid) + _interpolate_surface(lower, x_grid))
    expected = signed_camber[int(np.argmax(np.abs(signed_camber)))]
    assert abs(float(metrics["camber_signed_max_abs"]) - expected) < 1e-12


def test_build_blunts_aopt_before_applying_the_flap():
    output = build_foils.OUTPUT
    build_foils.build()

    source = normalise(read_dat(AEROFOIL_DIR / "LatestOptimised.dat"))
    sharp_aopt = repanel(build_foils._sharp_te(source), 200)
    expected = flap(blunt_te(sharp_aopt, 0.0025), 0.6483810, 21.35900)
    actual = read_dat(output / "AOPT_F_bluntTE.dat")
    metrics = geometry_metrics(actual, normalize_input=False)

    assert np.allclose(actual, expected, atol=6e-9)
    assert abs(float(np.linalg.norm(actual[0] - actual[-1])) - 0.0025) < 1e-5
    assert float(metrics["te_x"]) < 1.0
    assert float(metrics["te_y"]) < 0.0


def test_blunt_te_gap_and_shape_offset():
    sharp = repanel(_closed_te(read_dat(AEROFOIL_DIR / "LatestOptimised.dat")), 200)
    blunt = blunt_te(sharp, 0.0025)
    metrics = geometry_metrics(blunt)
    assert abs(metrics["te_gap"] - 0.0025) <= 1e-5
    x = np.linspace(0.0, 1.0, 1001)
    sharp_upper, sharp_lower = _surface_values(sharp, x)
    blunt_upper, blunt_lower = _surface_values(blunt, x)
    max_difference = max(np.max(np.abs(sharp_upper - blunt_upper)), np.max(np.abs(sharp_lower - blunt_lower)))
    assert max_difference < 0.0013
    assert not np.array_equal(blunt[0], blunt[-1])
    added_gap = 0.0025 - float(sharp[0, 1] - sharp[-1, 1])
    assert abs((blunt[0, 1] - sharp[0, 1]) - 0.5 * added_gap * sharp[0, 0] ** 2) < 1e-8
    assert abs((sharp[-1, 1] - blunt[-1, 1]) - 0.5 * added_gap * sharp[-1, 0] ** 2) < 1e-8


def test_b737_sections_parse_normalize_to_selig_and_do_not_intersect():
    sources = (
        "737-200 Root Refined.dat",
        "737-200 33% Span Refined.dat",
        "737-200 66% Span Refined.dat",
        "737-200 Tip Refined.dat",
    )
    for source_name in sources:
        points = normalise(read_dat(AEROFOIL_DIR / source_name))
        te_midpoint = 0.5 * (points[0] + points[-1])
        assert np.allclose(te_midpoint, (1.0, 0.0), atol=1e-6)
        le = int(np.argmin(points[:, 0]))
        upper = points[:le + 1][::-1]
        lower = points[le:]
        x = np.linspace(0.02, 0.98, 201)
        upper_y = _interpolate_surface(upper, x)
        lower_y = _interpolate_surface(lower, x)
        assert np.mean(upper_y >= 0.5 * (upper_y + lower_y)) > 0.9
        assert not self_intersects(points)
