from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b737wing.flow5.cases import ROOT_SECTIONS, case_config
from b737wing.flow5.parse import CpTable, FoilPolar, NumericTable, parse_op_point, parse_plane_polar
from b737wing.flow5 import post
from b737wing.flow5.post import (_cp_curves_for_eta, _panel_surface_curves,
                                 breguet_range_nmi, check_coverage, interpolate_target)
from b737wing.flow5.xmlgen import compute_mac

FIXTURES = Path(__file__).parent / "fixtures" / "flow5"


def _polar(foil: str, cl: list[float]) -> FoilPolar:
    return FoilPolar(foil, 20_000_000.0, 9.0, [-0.25, 0.0, 0.25], cl,
                     [0.02] * len(cl), [0.0] * len(cl), Path(f"{foil}.txt"))


def test_synthetic_linear_polar_interpolates_target_exactly():
    columns = ["α (°)", "CL", "CD", "CD_induced", "CD_viscous", "Cm"]
    rows = [
        [0.0, 0.2, 0.02, 0.005, 0.015, -0.01],
        [1.0, 0.4, 0.04, 0.010, 0.030, -0.02],
        [2.0, 0.6, 0.06, 0.015, 0.045, -0.03],
    ]
    table = NumericTable({}, columns, rows)
    result = interpolate_target(table, 0.5)
    assert result == pytest.approx({"alpha": 1.5, "cl": 0.5, "cd": 0.05,
                                    "cdi": 0.0125, "cdv": 0.0375, "cm": -0.025})


def test_breguet_range_matches_xlsx_reference():
    assert breguet_range_nmi(0.4895, 0.04322) == pytest.approx(2474.0, abs=2.0)


def test_mac_matches_area_weighted_closed_form_segment_trapezoids():
    weighted_area = 0.0
    area_sum = 0.0
    for left, right in zip(ROOT_SECTIONS, ROOT_SECTIONS[1:]):
        dy = right["y"] - left["y"]
        c0, c1 = left["chord"], right["chord"]
        area = dy * (c0 + c1) / 2.0
        segment_mac = (2.0 / 3.0) * (c0 * c0 + c0 * c1 + c1 * c1) / (c0 + c1)
        area_sum += area
        weighted_area += area * segment_mac
    assert compute_mac(ROOT_SECTIONS) == pytest.approx(weighted_area / area_sum, rel=1e-12)


def test_real_cp_panels_use_local_geometry_and_both_surfaces_at_eta_half():
    point = parse_op_point(FIXTURES / "real_op_F1_alpha2_coarse.csv")
    case = case_config("F1")
    panel_x_c = []
    for table in point.cp_tables:
        curves = _panel_surface_curves(table, case, wing_x_m=0.0)
        panel_x_c.extend(x_values for x_values, _ in curves.values())
        assert set(curves) == {"upper", "lower"}
    assert panel_x_c
    assert min(min(values) for values in panel_x_c) >= -0.01
    assert max(max(values) for values in panel_x_c) <= 1.01

    curves = _cp_curves_for_eta(point, case, 0.5, wing_x_m=0.0)
    assert set(curves) == {"upper", "lower"}
    for x_values, cp_values in curves.values():
        assert len(x_values) == len(cp_values) > 1
        assert x_values == sorted(x_values)
        assert min(x_values) >= -0.01
        assert max(x_values) <= 1.01


def test_real_polar_interpolates_quarter_lift_and_rejects_out_of_range_target():
    polar = parse_plane_polar(FIXTURES / "real_polar_F1_coarse.csv")
    quarter_lift = interpolate_target(polar, 0.25)
    assert quarter_lift["cl"] == pytest.approx(0.25)
    cl0, cl1 = polar.values("cl")[1:3]
    alpha0, alpha1 = polar.values("alpha")[1:3]
    expected_alpha = alpha0 + (0.25 - cl0) * (alpha1 - alpha0) / (cl1 - cl0)
    assert quarter_lift["alpha"] == pytest.approx(expected_alpha)
    with pytest.raises(ValueError, match=r"target CL .* outside increasing polar segment"):
        interpolate_target(polar, max(polar.values("cl")) + 0.01)


@pytest.mark.parametrize(
    ("case_name", "y_m", "present_foil", "missing_foil"),
    [("F5", 2.34926, "B737A", "B737B"), ("F3", 1.89, "AOPT", "AOPT_F")],
)
def test_coverage_requires_both_endpoint_foils_including_flap_transition(
        case_name, y_m, present_foil, missing_foil):
    strips = [{"y_m": y_m, "re": 20_000_000.0, "cl": 0.5}]

    violations = check_coverage(strips, case_name, [_polar(present_foil, [-1.0, 0.0, 1.0])])

    assert [(item["foil"], item["reason"]) for item in violations] == [
        (missing_foil, "no matching foil polar")]


def test_coverage_reports_insufficient_cl_on_outboard_endpoint_foil():
    strips = [{"y_m": 2.34926, "re": 20_000_000.0, "cl": 0.5}]
    polars = [_polar("B737A", [-1.0, 0.0, 1.0]), _polar("B737B", [-1.0, -0.5, 0.0])]

    violations = check_coverage(strips, "F5", polars)

    assert len(violations) == 1
    assert violations[0]["foil"] == "B737B"
    assert "Cl 0.5 outside" in violations[0]["reason"]


def test_cp_x_over_c_uses_twisted_dihedral_local_chord_with_translation():
    chord = 2.0
    twist = math.radians(3.0)
    dihedral = math.radians(6.0)
    span = (0.0, math.cos(dihedral), math.sin(dihedral))
    base_chord = (1.0, 0.0, 0.0)
    span_cross_chord = (span[1] * base_chord[2] - span[2] * base_chord[1],
                        span[2] * base_chord[0] - span[0] * base_chord[2],
                        span[0] * base_chord[1] - span[1] * base_chord[0])
    direction = tuple(math.cos(twist) * base_chord[i] + math.sin(twist) * span_cross_chord[i]
                      for i in range(3))
    upper = (direction[1] * span[2] - direction[2] * span[1],
             direction[2] * span[0] - direction[0] * span[2],
             direction[0] * span[1] - direction[1] * span[0])
    translation = (12.5, 0.0, 4.0)
    leading_edge = (translation[0] + 3.0, 5.0,
                    translation[2] + math.tan(dihedral) * 5.0)
    columns = ["CtrlPt.x", "CtrlPt.y", "CtrlPt.z", "N.x", "N.y", "N.z", "Cp"]
    rows = []
    for x_c in (0.125, 0.375, 0.625, 0.875):
        for sign, cp in ((1.0, -0.5), (-1.0, 0.2)):
            point = tuple(leading_edge[i] + x_c * chord * direction[i] +
                          sign * 0.02 * chord * upper[i] for i in range(3))
            normal = tuple(sign * value for value in upper)
            rows.append([*point, *normal, cp])
    table = CpTable("Main Wing", columns, rows, strip_number=1, strip_y_m=5.0)
    case = {"sections": [
        {"y": 0.0, "chord": chord, "x": 3.0, "dihedral": 6.0, "twist": 3.0},
        {"y": 10.0, "chord": chord, "x": 3.0, "dihedral": 6.0, "twist": 3.0},
    ]}

    curves = _panel_surface_curves(table, case, wing_x_m=translation[0])

    assert set(curves) == {"upper", "lower"}
    for x_values, _ in curves.values():
        assert x_values == pytest.approx([0.125, 0.375, 0.625, 0.875], abs=0.002)


def test_postprocess_writes_alpha_gaps_to_coverage_diagnostics_and_result_json(monkeypatch, tmp_path):
    gap = {"kind": "alpha_gap", "foil": "A0012", "re": 20_000_000.0,
           "alpha_from_deg": 0.0, "alpha_to_deg": 16.0, "gap_deg": 16.0,
           "requested_step_deg": 4.0,
           "reason": "alpha gap 16 deg exceeds 2.5 requested steps (10 deg)"}
    table = NumericTable({}, ["alpha", "cl", "cd", "cdi", "cdv", "cm"], [
        [0.0, 0.4, 0.04, 0.01, 0.03, -0.02],
        [1.0, 0.6, 0.06, 0.015, 0.045, -0.03],
    ])
    point = SimpleNamespace(alpha=0.5)
    strips = [{"y_m": 2.34926, "eta": 0.16, "re": 20_000_000.0, "cl": 0.5,
               "cdi": 0.01, "cdv": 0.02, "cm": -0.01}]
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "A0012.txt").write_text("staged", encoding="utf-8")
    monkeypatch.setattr(post, "parse_plane_polar", lambda path: table)
    monkeypatch.setattr(post, "parse_op_points", lambda *args, **kwargs: [point])
    monkeypatch.setattr(post, "_interpolate_strip_rows", lambda points, alpha: strips)
    monkeypatch.setattr(post, "parse_foil_polar", lambda path: _polar("A0012", [-1.0, 0.0, 1.0]))
    monkeypatch.setattr(post, "_wing_position_x", lambda *args: 0.0)
    monkeypatch.setattr(post, "_cp_curves_for_eta", lambda *args: {
        "upper": ([0.0, 1.0], [-1.0, 0.0]), "lower": ([0.0, 1.0], [0.2, 0.1])})

    result = post.postprocess("F1", tmp_path / "plane.csv", tmp_path / "project",
                              tmp_path / "output", staged, alpha_gaps=[gap], alpha_step=4.0)
    written = json.loads((tmp_path / "output" / "result.json").read_text(encoding="utf-8"))

    assert result["alpha_gap_diagnostics"] == [gap]
    assert written["alpha_gap_diagnostics"] == [gap]
    assert gap in written["coverage_violations"]
