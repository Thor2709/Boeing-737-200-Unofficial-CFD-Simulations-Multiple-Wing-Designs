from __future__ import annotations

from pathlib import Path

import pytest

from b737wing.flow5.parse import parse_op_point, parse_op_point_texts, parse_plane_polar, parse_plane_polar_text

FIXTURES = Path(__file__).parent / "fixtures" / "flow5"


def test_welded_header_inf_row_and_duplicate_op_point_files():
    data0 = "    0.000          0.000          0.100          inf".ljust(60)
    data1 = "    0.000          1.000          0.200          0.010".ljust(60)
    header = "Ctrl          α (°)          CL          CD"
    plane_polar = (
        "Density = 0.4582 kg/m3\nNbr. of data points = 2\n"
        + header + data0 + "\n"
        + data1 + "\n"
    )
    parsed = parse_plane_polar_text(plane_polar)
    assert parsed.columns == ["Ctrl", "α (°)", "CL", "CD"]
    assert len(parsed.rows) == 2
    assert parsed.rows[0][3] == float("inf")
    assert parsed.nonfinite == [(0, "CD")]

    contents = """Operating point: alpha 2.00\nF1_T1\nMain Wing\ny(m) Re Cl\n 1.000 1000000 0.250\nx/c Cp\n 0.000 -0.500\n 1.000 0.100\n"""
    points = parse_op_point_texts([contents, contents],
                                  ["first 2_00°_224_33m_s.csv", "duplicate 2_00°_224_33m_s.csv"],
                                  expected_polar="F1_T1")
    assert len(points) == 1
    assert points[0].alpha == pytest.approx(2.0)
    assert points[0].strips["Main Wing"].rows[0][2] == pytest.approx(0.25)
    assert len(points[0].cp_tables) == 1


def test_real_flow5_op_point_coefficients_strips_and_cp_mapping():
    point = parse_op_point(FIXTURES / "real_op_F1_alpha2_coarse.csv")
    assert point.alpha == pytest.approx(2.0)
    assert point.coefficients["CL"] == pytest.approx(0.258514)
    table = point.strips["Main Wing"]
    assert len(table.rows) == 24
    y_index = table.columns.index("y(m)")
    cl_index = table.columns.index("Cl")
    fz_index = table.columns.index("F.z")
    for index in range(12):
        negative, positive = table.rows[index], table.rows[-index - 1]
        assert negative[y_index] == pytest.approx(-positive[y_index], abs=1e-9)
        assert negative[cl_index] == pytest.approx(positive[cl_index], abs=1e-9)

    assert len(point.cp_tables) == 24
    for strip_number, cp_table in enumerate(point.cp_tables, start=1):
        assert cp_table.strip_number == strip_number
        assert cp_table.strip_y_m == pytest.approx(table.rows[strip_number - 1][y_index])
        cp_y_index = cp_table.columns.index("CtrlPt.y")
        mean_y = sum(row[cp_y_index] for row in cp_table.rows) / len(cp_table.rows)
        distances = []
        if strip_number > 1:
            distances.append(abs(table.rows[strip_number - 1][y_index] - table.rows[strip_number - 2][y_index]))
        if strip_number < len(table.rows):
            distances.append(abs(table.rows[strip_number][y_index] - table.rows[strip_number - 1][y_index]))
        half_width = sum(distances) / len(distances) / 2.0
        assert abs(mean_y - cp_table.strip_y_m) <= half_width

    q = 0.5 * 0.4582 * 224.33**2
    strip_cl = sum(row[fz_index] for row in table.rows) / (q * 91.04)
    assert strip_cl == pytest.approx(point.coefficients["CL"], rel=0.005)


def test_real_flow5_plane_polar_has_seven_increasing_points():
    polar = parse_plane_polar(FIXTURES / "real_polar_F1_coarse.csv")
    alphas, cls = polar.values("alpha"), polar.values("cl")
    assert len(polar.rows) == 7
    assert all(right > left for left, right in zip(alphas, alphas[1:]))
    assert all(right > left for left, right in zip(cls, cls[1:]))
