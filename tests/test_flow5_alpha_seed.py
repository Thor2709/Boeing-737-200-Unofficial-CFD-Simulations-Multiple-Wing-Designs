from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from b737wing.flow5 import alpha_seed
from b737wing.flow5.alpha_seed import (karman_tsien_cp, polar_zero_lift_and_slope,
                                       read_polar)
from b737wing.flow5.xmlgen import script_xml, write_sweep_normal_foils


def test_mach_zero_keeps_default_batch_xml_unchanged():
    default = script_xml("F1", 1, run_root=Path("run"), foil_files=["A0012.dat"],
                         reynolds=[2e6, 10e6, 50e6])
    explicit_zero = script_xml("F1", 1, run_root=Path("run"), foil_files=["A0012.dat"],
                               reynolds=[2e6, 10e6, 50e6], mach=0.0)
    assert default == explicit_zero
    root = ET.fromstring(default)
    assert root.findtext(".//Batch_Range/Mach") == "0, 0, 0"


def test_compressible_batch_repeats_requested_mach_for_every_reynolds():
    reynolds = [2e6, 10e6, 50e6]
    xml = script_xml("F1", 1, run_root=Path("run"), foil_files=["A0012.dat"],
                     reynolds=reynolds, mach=0.6707)
    root = ET.fromstring(xml)
    re_values = root.findtext(".//Batch_Range/Reynolds").split(", ")
    mach_values = root.findtext(".//Batch_Range/Mach").split(", ")
    assert len(mach_values) == len(re_values) == len(reynolds)
    assert mach_values == ["0.6707"] * len(reynolds)


def test_section_cl_maps_to_sweep_normal_plane():
    cosine = math.cos(math.radians(25.0))
    assert alpha_seed._normal_plane_cl(0.5, cosine) == pytest.approx(0.6087, abs=0.0001)


def test_normal_plane_zero_lift_angle_maps_back_to_streamwise():
    cosine = math.cos(math.radians(25.0))
    assert alpha_seed._streamwise_from_normal(-2.0, cosine) == pytest.approx(-1.8126, abs=0.0001)


def test_normal_plane_foil_scales_thickness_and_preserves_x(tmp_path):
    source_directory = tmp_path / "source"
    output_directory = tmp_path / "foils"
    source_directory.mkdir()
    source = source_directory / "TEST.dat"
    source.write_text("TEST\n1.0 0.0\n0.5 0.1\n0.0 0.0\n0.5 -0.1\n1.0 0.0\n",
                      encoding="utf-8")

    filenames = write_sweep_normal_foils(source_directory, output_directory, ["TEST"])

    assert filenames == ["TEST_n25.dat"]
    original_rows = [[float(value) for value in line.split()]
                     for line in source.read_text(encoding="utf-8").splitlines()[1:]]
    normal_lines = (output_directory / filenames[0]).read_text(encoding="utf-8").splitlines()
    normal_rows = [[float(value) for value in line.split()] for line in normal_lines[1:]]
    assert [row[0] for row in normal_rows] == [row[0] for row in original_rows]
    original_thickness = max(row[1] for row in original_rows) - min(row[1] for row in original_rows)
    normal_thickness = max(row[1] for row in normal_rows) - min(row[1] for row in normal_rows)
    assert normal_thickness == pytest.approx(
        original_thickness / math.cos(math.radians(25.0)), abs=1e-8)


def test_mach_polars_select_normal_plane_foil_for_cruise(monkeypatch, tmp_path):
    requested = []
    selected = object()

    def choose(directory, foil, reynolds, *, mach=None):
        requested.append((foil, mach))
        return selected

    monkeypatch.setattr(alpha_seed, "_choose_polar", choose)
    polar, source = alpha_seed._get_case_polar(tmp_path, tmp_path, "F1", {"foil": "AOPT"},
                                               1.0e6, 0.6707, sweep_normal=True)

    assert polar is selected
    assert source == "compressible_polar"
    assert requested == [("AOPT_n25", 0.6707)]


def test_symmetric_foil_polar_has_zero_lift_angle_near_zero():
    path = Path("flow5_v2/F1/fine/xfoil_polars/A0012__T1-Re10.000-N9.0.txt")
    if not path.is_file():
        pytest.skip("project-only flow5 polar not in repo")
    polar = read_polar(path)
    alpha0, _, extrapolation_span = polar_zero_lift_and_slope(polar)
    assert alpha0 == pytest.approx(0.0, abs=0.02)
    assert extrapolation_span == 0.0
    assert polar.cp_min[0] == pytest.approx(-3.6667)


def test_karman_tsien_cp_conversion_matches_hand_value():
    assert karman_tsien_cp(-1.0, 0.5) == pytest.approx(-1.251506, abs=2e-6)


def test_identical_wings_preserve_c5_alpha():
    alpha_c5 = 3.624
    d_geom, d_aero, _ = alpha_seed._seed_terms(alpha_c5, 1.0, 1.0, -1.0, -1.0,
                                               0.42, 0.08, 0.08)
    assert alpha_c5 + d_geom + d_aero == pytest.approx(alpha_c5)


def test_one_degree_positive_twist_offset_reduces_seed_by_one_degree():
    alpha_c5 = 3.624
    d_geom, d_aero, _ = alpha_seed._seed_terms(alpha_c5, 1.0, 2.0, -1.0, -1.0,
                                               0.42, 0.08, 0.08)
    assert alpha_c5 + d_geom + d_aero == pytest.approx(alpha_c5 - 1.0)


def test_missing_polar_file_makes_cli_exit_nonzero(monkeypatch, capsys):
    root = alpha_seed.project_root()
    c5_dir = root / "cfd_v2/runs/C5/iH0"
    if not (c5_dir / "history.csv").is_file():
        pytest.skip("project-only CFD history not in repo")
    monkeypatch.setattr(alpha_seed, "project_root", lambda: root / "missing_alpha_seed_inputs")
    monkeypatch.setattr(alpha_seed, "_flow5_alphas", lambda _: {"F1": 1.0, "F2": 1.0,
                                                                  "F3": 1.0, "F4": 1.0, "F5": 1.0})
    result = alpha_seed.main(["--c5", str(c5_dir), "--out", "research/alpha_seed_missing_test.json"])
    assert result != 0
    assert "no Mach 0 polar" in capsys.readouterr().err
