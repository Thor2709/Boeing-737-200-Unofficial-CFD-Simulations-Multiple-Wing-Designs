from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from b737wing.flow5.xmlgen import plane_sections, plane_xml, polar_xml, script_xml


def test_f3_has_flap_edge_pairs_and_correct_foils():
    sections = plane_sections("F3", "coarse")
    by_y = {round(section["y"], 7): section for section in sections}
    assert by_y[1.88]["foil"] == "AOPT"
    assert by_y[1.90]["foil"] == "AOPT_F"
    outer = 0.74 * 14.09556
    assert by_y[round(outer - 0.02, 7)]["foil"] == "AOPT_F"
    assert by_y[round(outer, 7)]["foil"] == "AOPT"
    xml = ET.fromstring(plane_xml("F3", "coarse"))
    positions = [float(section.findtext("y_position")) for section in xml.findall(".//Section")]
    assert any(abs(y - 1.88) < 1e-8 for y in positions)
    assert any(abs(y - outer) < 1e-6 for y in positions)


def test_f4_twist_is_linear_and_reference_dimensions_are_custom_nonzero():
    sections = plane_sections("F4", "medium")
    assert sections[0]["twist"] == pytest.approx(-0.459177)
    assert sections[-1]["twist"] == pytest.approx(0.459175)
    for section in sections:
        expected = -0.459177 + (0.459175 + 0.459177) * section["y"] / 14.09556
        assert section["twist"] == pytest.approx(expected)
    polar = ET.fromstring(polar_xml("F4"))
    refs = polar.find(".//Polar/Reference_Dimensions")
    assert refs.findtext("Reference_Dimensions") == "CUSTOM"
    for field in ("Reference_Area", "Reference_Span_Length", "Reference_Chord_Length"):
        assert float(refs.findtext(field)) > 0.0
    assert float(polar.findtext(".//Fluid/Density")) > 0.0
    assert float(polar.findtext(".//Fluid/Viscosity")) > 0.0


def test_method_is_validated_before_xml_generation():
    with pytest.raises(ValueError, match="invalid flow5 method"):
        polar_xml("F1", method="NOT_A_FLOW5_METHOD")


def test_scripts_keep_passes_separate_and_put_alpha_in_documented_range():
    pass1 = script_xml("F1", 1, run_root=__import__("pathlib").Path("run"),
                       foil_files=["A0012.dat"], reynolds=[2e6, 1e7, 5e7])
    assert "<foil_analysis>" in pass1
    assert "<Plane_analysis>" not in pass1
    assert "<Batch_Range><Reynolds>" in pass1
    assert "<OpPoint_Range><Alpha>" in pass1
    assert "<Batch_Range><Alpha>" not in pass1
    assert "<Repanel_Foils>false</Repanel_Foils>" in pass1
    assert "<make_polars_text_file>true</make_polars_text_file>" in pass1

    pass2 = script_xml("F1", 2, run_root=__import__("pathlib").Path("run"),
                       foil_files=["A0012.dat"], alpha=(0.0, 6.0, 1.0))
    assert "<Plane_analysis>" in pass2
    assert "<foil_analysis>" not in pass2
    assert "<T12_Range>0, 6, 1</T12_Range>" in pass2
    assert "<export_oppoint_Cp>true</export_oppoint_Cp>" in pass2
    assert "<export_stl_mesh>true</export_stl_mesh>" in pass2
