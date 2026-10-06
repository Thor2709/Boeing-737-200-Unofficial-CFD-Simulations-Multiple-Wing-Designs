"""Generate the flow5 plane, polar, and separate two-pass scripts."""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

from .cases import case_config

ALLOWED_METHODS = {"LLT", "VLM1", "VLM2", "QUADS", "TRIUNIFORM", "TRILINEAR"}
MESHES = {
    # Opus 2026-10-03: refinement ratio 1.5 in both directions; production = fine.
    "coarse": {"x_panels": 16, "spacing": 0.60},
    "medium": {"x_panels": 24, "spacing": 0.40},
    "fine": {"x_panels": 36, "spacing": 0.27},
}
FLAP_INNER_Y = 1.88
FLAP_OUTER_Y = 0.74 * 14.09556
FLAP_EDGE_GAP = 0.02


def project_root() -> Path:
    try:
        from b737wing.config import REPO_ROOT
        return REPO_ROOT
    except ImportError:
        return Path(__file__).resolve().parents[3]


def reynolds_values() -> list[float]:
    values = {round(2.0e6 * (5.0e7 / 2.0e6) ** (i / 17.0)) for i in range(18)}
    values.update(round(6.0e6 + i * 0.8e6) for i in range(11))
    return sorted(float(value) for value in values)


def write_sweep_normal_foils(source_directory: Path, output_directory: Path,
                             foil_names: list[str], *, sweep_deg: float = 25.0) -> list[str]:
    """Write normal-plane foils by scaling ordinates and return their filenames."""
    cosine = math.cos(math.radians(sweep_deg))
    if not math.isfinite(sweep_deg) or cosine <= 0.0:
        raise ValueError("sweep angle must have a positive cosine")
    suffix = f"n{round(sweep_deg):g}"
    source_directory, output_directory = Path(source_directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    filenames = []
    for foil in foil_names:
        source = source_directory / f"{foil}.dat"
        rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines()[1:], start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"{source}:{line_number}: expected x/y foil coordinates")
            try:
                x, y = float(fields[0]), float(fields[1])
            except ValueError as error:
                raise ValueError(f"{source}:{line_number}: invalid x/y foil coordinates") from error
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{source}:{line_number}: non-finite x/y foil coordinates")
            rows.append((x, y / cosine))
        if len(rows) < 2:
            raise ValueError(f"{source}: fewer than two foil coordinate rows")
        filename = f"{foil}_{suffix}.dat"
        target = output_directory / filename
        content = [f"{foil}_{suffix}", *(f"{x:.8f} {y:.8f}" for x, y in rows)]
        target.write_text("\n".join(content) + "\n", encoding="utf-8")
        filenames.append(filename)
    return filenames


def compute_mac(sections: list[dict]) -> float:
    """Integrate the piecewise-linear chord distribution over the semi-span."""
    if len(sections) < 2:
        raise ValueError("at least two wing sections are required to compute MAC")
    area = 0.0
    chord_squared_integral = 0.0
    for left, right in zip(sections, sections[1:]):
        dy = float(right["y"]) - float(left["y"])
        c0, c1 = float(left["chord"]), float(right["chord"])
        if dy <= 0.0 or min(c0, c1) <= 0.0:
            raise ValueError("wing section y positions and chords must increase/be positive")
        area += dy * (c0 + c1) / 2.0
        chord_squared_integral += dy * (c0 * c0 + c0 * c1 + c1 * c1) / 3.0
    return chord_squared_integral / area


def _base_sections(case: dict) -> list[dict]:
    sections = []
    for source, foil in zip(case["sections"], case["foils"]):
        sections.append({**source, "foil": foil})
    if case.get("flap"):
        sections = _insert_flap_edges(sections)
    return sections


def _insert_flap_edges(sections: list[dict]) -> list[dict]:
    base = sorted(sections, key=lambda section: section["y"])

    def at(y: float, foil: str) -> dict:
        for left, right in zip(base, base[1:]):
            if left["y"] <= y <= right["y"]:
                fraction = (y - left["y"]) / (right["y"] - left["y"])
                return {
                    "y": y,
                    "chord": left["chord"] + fraction * (right["chord"] - left["chord"]),
                    "x": left["x"] + fraction * (right["x"] - left["x"]),
                    "dihedral": left["dihedral"],
                    "twist": left["twist"] + fraction * (right["twist"] - left["twist"]),
                    "foil": foil,
                }
        raise ValueError(f"flap edge y={y:g} falls outside the wing sections")

    start_outer = FLAP_INNER_Y
    start_inner = start_outer + FLAP_EDGE_GAP
    end_inner = FLAP_OUTER_Y - FLAP_EDGE_GAP
    end_outer = FLAP_OUTER_Y
    if not (0.0 < start_outer < start_inner < end_inner < end_outer < base[-1]["y"]):
        raise ValueError("flap edge sections do not fit within the wing")
    special = [
        at(start_outer, "AOPT"),
        at(start_inner, "AOPT_F"),
        at(end_inner, "AOPT_F"),
        at(end_outer, "AOPT"),
    ]
    merged = [section for section in base if not (start_outer < section["y"] < end_outer)] + special
    return sorted(merged, key=lambda section: section["y"])


def plane_sections(name: str, mesh: str = "coarse") -> list[dict]:
    if mesh not in MESHES:
        raise ValueError(f"invalid mesh {mesh!r}; expected coarse, medium, or fine")
    case = case_config(name)
    sections = _base_sections(case)
    mesh_settings = MESHES[mesh]
    for index, section in enumerate(sections):
        if index == len(sections) - 1:
            section["y_panels"] = 1
        else:
            length = sections[index + 1]["y"] - section["y"]
            section["y_panels"] = max(1, int(math.floor(length / mesh_settings["spacing"] + 0.5)))
        section["x_panels"] = mesh_settings["x_panels"]
        section["distribution"] = "COSINE"
        section["y_distribution"] = "UNIFORM"
    return sections


def _num(value: float) -> str:
    return f"{float(value):.8g}"


def plane_xml(name: str, mesh: str = "coarse", *, sections: list[dict] | None = None) -> str:
    case = case_config(name)
    rows = []
    source = plane_sections(name, mesh) if sections is None else sections
    for section in source:
        foil = escape(str(section["foil"]))
        # FLOW5 7.57 linearly interpolates section twist through the tip; preserve endpoint values.
        rows.append(f"""        <Section>
          <y_position>{_num(section['y'])}</y_position>
          <Chord>{_num(section['chord'])}</Chord>
          <xOffset>{_num(section['x'])}</xOffset>
          <Dihedral>{_num(section['dihedral'])}</Dihedral>
          <Twist>{_num(section['twist'])}</Twist>
          <x_number_of_panels>{int(section.get('x_panels', MESHES[mesh]['x_panels']))}</x_number_of_panels>
          <x_panel_distribution>{section.get('distribution', 'COSINE')}</x_panel_distribution>
          <y_number_of_panels>{int(section.get('y_panels', 1))}</y_number_of_panels>
          <y_panel_distribution>{section.get('y_distribution', 'UNIFORM')}</y_panel_distribution>
          <Left_Side_FoilName>{foil}</Left_Side_FoilName>
          <Right_Side_FoilName>{foil}</Right_Side_FoilName>
        </Section>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflplane version="1.0">
  <Units><meter_to_length_unit>1.0</meter_to_length_unit><kg_to_mass_unit>1.0</kg_to_mass_unit></Units>
  <Plane>
    <Name>{escape(case['plane_name'])}</Name>
    <Description>Wing-only B737-200 {case['name']} study</Description>
    <wing>
      <Name>Main Wing</Name><Type>MAINWING</Type><Position>0.0, 0.0, 0.0</Position>
      <Symmetric>true</Symmetric><Two_Sided>true</Two_Sided><Tip_Strips>0</Tip_Strips>
      <Sections>
{chr(10).join(rows)}
      </Sections>
    </wing>
  </Plane>
</xflplane>
"""


def polar_xml(name: str, *, method: str = "TRIUNIFORM", wake_spans: float = 20.0,
              viscosity: float | None = None) -> str:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"invalid flow5 method {method!r}; allowed: {', '.join(sorted(ALLOWED_METHODS))}")
    case = case_config(name)
    flight = case["flight"]
    mac = compute_mac(case["sections"])
    nu = float(flight["nu"] if viscosity is None else viscosity)
    if wake_spans <= 0.0 or nu <= 0.0:
        raise ValueError("wake_spans and viscosity must be positive")
    ar = 28.346**2 / 91.04
    wake_length = wake_spans * ar
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflPlanePolar version="1.0">
  <Units><meter_to_length_unit>1.0</meter_to_length_unit><kg_to_mass_unit>1.0</kg_to_mass_unit><ms_to_speed_unit>1.0</ms_to_speed_unit></Units>
  <Polar>
    <Polar_Name>{escape(case['polar_name'])}</Polar_Name>
    <Plane_Name>{escape(case['plane_name'])}</Plane_Name>
    <Type>FIXEDSPEEDPOLAR</Type><Method>{method}</Method><Thin_Surfaces>false</Thin_Surfaces>
    <Reference_Dimensions>
      <Reference_Dimensions>CUSTOM</Reference_Dimensions>
      <Reference_Area>91.04</Reference_Area><Reference_Span_Length>28.346</Reference_Span_Length>
      <Reference_Chord_Length>{_num(mac)}</Reference_Chord_Length><Include_Other_Wing_Area>false</Include_Other_Wing_Area>
    </Reference_Dimensions>
    <Viscous_Analysis><Is_Viscous_Analysis>true</Is_Viscous_Analysis><XFoil_OnTheFly>false</XFoil_OnTheFly>
      <From_CL>false</From_CL><NCrit>9.0</NCrit><XTrTop>1.0</XTrTop><XTrBot>1.0</XTrBot></Viscous_Analysis>
    <Fluid><Density>{_num(flight['rho'])}</Density><Viscosity>{_num(nu)}</Viscosity></Fluid>
    <Ground_Effect>false</Ground_Effect><Ground_Height>0.0</Ground_Height>
    <Include_Fuse_Moments>false</Include_Fuse_Moments>
    <Fuselage_Drag><Friction_Drag>false</Friction_Drag><Friction_Drag_Method>Karman-Schoenherr</Friction_Drag_Method></Fuselage_Drag>
    <Use_plane_inertia>true</Use_plane_inertia><Fixed_Velocity>{_num(flight['velocity'])}</Fixed_Velocity>
    <Inertia><Mass>0.0</Mass><CoG>0.0, 0.0, 0.0</CoG></Inertia>
    <Wake><FlatPanelWake>true</FlatPanelWake><NX>5</NX><ProgressionFactor>1.0</ProgressionFactor><LengthFactor>{_num(wake_length)}</LengthFactor></Wake>
  </Polar>
</xflPlanePolar>
"""


def script_xml(name: str, pass_number: int, *, run_root: Path, foil_files: list[str],
               mesh: str = "coarse", alpha: tuple[float, float, float] | None = None,
               reynolds: list[float] | None = None, project_name: str | None = None,
               mach: float = 0.0) -> str:
    if pass_number not in (1, 2):
        raise ValueError("pass_number must be 1 or 2")
    case = case_config(name)
    run_root = Path(run_root).resolve()
    stage = run_root / ("pass1" if pass_number == 1 else "pass2")
    output_dir = stage / "flow5-output"
    project = project_name or f"{case['name']}_{mesh}_pass{pass_number}"
    foil_names = "\n".join(f"      <Foil_File_Name>{escape(filename)}</Foil_File_Name>" for filename in foil_files)
    text_format = "" if pass_number == 1 else "    <polar_text_output_format>csv</polar_text_output_format>\n"
    if pass_number == 1:
        reynolds = reynolds or reynolds_values()
        if len(reynolds) < 1 or any(value <= 0 for value in reynolds):
            raise ValueError("pass 1 needs positive Reynolds numbers")
        if not math.isfinite(mach) or mach < 0.0 or mach >= 1.0:
            raise ValueError("pass 1 Mach number must be finite and in [0, 1)")
        alpha = alpha or ((-10.0, 24.0, 0.25) if case["condition"] == "takeoff" else (-8.0, 16.0, 0.25))
        alpha_text = ", ".join(_num(value) for value in alpha)
        re_text = ", ".join(f"{value:.0f}" for value in reynolds)
        mach_text = ", ".join(_num(mach) for _ in reynolds)
        ncrit_text = ", ".join("9" for _ in reynolds)
        directories = f"""      <output_dir>{escape(output_dir.as_posix())}</output_dir>
      <foil_files_dir>{escape((run_root / 'foils').as_posix())}</foil_files_dir>
      <foil_analysis_xml_dir>{escape((stage / 'foil-analyses').as_posix())}</foil_analysis_xml_dir>
      <foil_polars_dir>{escape((output_dir / 'Foil_polars').as_posix())}</foil_polars_dir>
      <xfoil_polars_dir>{escape((run_root / 'xfoil_polars').as_posix())}</xfoil_polars_dir>"""
        body = f"""  <foil_analysis>
    <Foil_Files>
{foil_names}
    </Foil_Files>
    <Batch_Analysis_Data><Polar_Type>FIXEDSPEEDPOLAR</Polar_Type>
      <Batch_Range><Reynolds>{re_text}</Reynolds><NCrit>{ncrit_text}</NCrit><Mach>{mach_text}</Mach></Batch_Range>
    </Batch_Analysis_Data>
    <OpPoint_Range><Alpha>{alpha_text}</Alpha><Spec_Alpha>true</Spec_Alpha><From_Zero>true</From_Zero></OpPoint_Range>
    <Options><Max_XFoil_Iterations>200</Max_XFoil_Iterations><Repanel_Foils>false</Repanel_Foils><Foil_Panels>200</Foil_Panels></Options>
    <Output><make_polars_bin_file>false</make_polars_bin_file><make_polars_text_file>true</make_polars_text_file><make_oppoints>false</make_oppoints></Output>
  </foil_analysis>"""
    else:
        alpha = alpha or (-8.0, 16.0, 0.5)
        alpha_text = ", ".join(_num(value) for value in alpha)
        directories = f"""      <output_dir>{escape(output_dir.as_posix())}</output_dir>
      <foil_files_dir>{escape((run_root / 'foils').as_posix())}</foil_files_dir>
      <xfoil_polars_dir>{escape((run_root / 'xfoil_polars').as_posix())}</xfoil_polars_dir>
      <plane_definition_xml_dir>{escape((stage / 'planes').as_posix())}</plane_definition_xml_dir>
      <plane_analysis_xml_dir>{escape((stage / 'analyses').as_posix())}</plane_analysis_xml_dir>"""
        body = f"""  <Plane_analysis>
    <Plane_Analysis_Output><make_polars_text_file>true</make_polars_text_file><make_oppoints>true</make_oppoints>
      <make_oppoints_text_file>true</make_oppoints_text_file><export_oppoint_Cp>true</export_oppoint_Cp><export_stl_mesh>true</export_stl_mesh></Plane_Analysis_Output>
    <Foil_Dat_Files>
{foil_names}
    </Foil_Dat_Files>
    <Plane_Definition_Files><Process_All_Files>true</Process_All_Files></Plane_Definition_Files>
    <Plane_Analysis_Files><Process_All_Files>true</Process_All_Files></Plane_Analysis_Files>
    <Plane_Analysis_Data><T12_Range>{alpha_text}</T12_Range></Plane_Analysis_Data>
  </Plane_analysis>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xflscript>
<xflscript version="1.0">
  <Metadata><make_project_file>true</make_project_file><project_file_name>{escape(project)}</project_file_name>
    <Double_Precision>true</Double_Precision>
    <MultiThreading><Allow_Multithreading>true</Allow_Multithreading><Thread_Priority>Normal</Thread_Priority><max_threads>20</max_threads></MultiThreading>
{text_format}    <Directories>
{directories}
    </Directories>
  </Metadata>
{body}
</xflscript>
"""


def write_case_files(name: str, run_root: Path, mesh: str, *, method: str = "TRIUNIFORM",
                     wake_spans: float = 20.0, viscosity: float | None = None) -> tuple[Path, Path, Path, Path]:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"invalid flow5 method {method!r}; allowed: {', '.join(sorted(ALLOWED_METHODS))}")
    case = case_config(name)
    run_root = Path(run_root).resolve()
    pass1 = run_root / "pass1"
    pass2 = run_root / "pass2"
    for path in (pass1, pass2):
        path.mkdir(parents=True, exist_ok=True)
    plane_path = pass2 / "planes" / f"{case['name']}.xml"
    polar_path = pass2 / "analyses" / f"{case['name']}.xml"
    plane_path.parent.mkdir(parents=True, exist_ok=True)
    polar_path.parent.mkdir(parents=True, exist_ok=True)
    plane_path.write_text(plane_xml(name, mesh), encoding="utf-8")
    polar_path.write_text(polar_xml(name, method=method, wake_spans=wake_spans, viscosity=viscosity), encoding="utf-8")
    return plane_path, polar_path, pass1 / "script.xml", pass2 / "script.xml"
