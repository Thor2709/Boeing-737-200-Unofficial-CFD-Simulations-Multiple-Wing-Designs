import csv
import json
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from b737wing.post import case_post, force_breakdown, render_cfd
from b737wing.post.render_cfd import component_yplus_bands
from b737wing.cfd.solve.fluent_io import polygon_area_vectors
from b737wing.post.sections import nearest_centroid_map, section_from_cut


def _section_segments(vertices, cp_by_edge):
    rows = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        middle = (np.asarray(start) + np.asarray(end)) * 0.5
        rows.append({"p1": np.asarray(start, dtype=float), "p2": np.asarray(end, dtype=float),
                     "x": float(middle[0]), "z": float(middle[1]),
                     "cp": float(cp_by_edge[index])})
    return rows


def test_section_pressure_integration_includes_blunt_te_face():
    vertices = [(0, 0), (0.5, 0.05), (1, 0.00125), (1, -0.00125), (0.5, -0.05)]
    cap_cp = -0.2
    section = section_from_cut(_section_segments(vertices, [0, 0, cap_cp, 0, 0]),
                               0.0, 0.0, 1.0)
    assert section["cdp"] == pytest.approx(-cap_cp * 0.0025, abs=1e-12)
    constant_cp = section_from_cut(_section_segments(vertices, [0.3] * len(vertices)),
                                  0.0, 0.0, 1.0)
    assert constant_cp["cdp"] == pytest.approx(0.0, abs=1e-12)
    assert constant_cp["cl"] == pytest.approx(0.0, abs=1e-12)


def test_wall_mapping_rejects_outlier_and_duplicate_source():
    source = np.asarray([[x, 0.0, 0.0] for x in range(5)], dtype=float)
    values = np.arange(5, dtype=float)
    outlier = source.copy()
    outlier[-1, 0] = 100.0
    with pytest.raises(ValueError, match=r"rejected 1 target face.*96"):
        nearest_centroid_map(source, outlier, values)
    duplicate = source.copy()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match=r"rejected 1 target face.*1 duplicate source"):
        nearest_centroid_map(source, duplicate, values)


def test_summary_keeps_smoke_run_out_of_authoritative_table(tmp_path):
    runs = tmp_path / "runs"
    production = runs / "C5" / "iH0"
    smoke = runs / "_smoke"
    production.mkdir(parents=True)
    smoke.mkdir(parents=True)
    record = {"case": "C5", "CL": 0.4895, "CD": 0.04322, "CM": -0.07,
              "L_over_D": 11.3, "status": "converged"}
    (production / "result.json").write_text(json.dumps(record), encoding="utf-8")
    (smoke / "result.json").write_text(json.dumps({**record, "status": "smoke"}),
                                        encoding="utf-8")
    output = tmp_path / "summary.csv"
    diagnostics = tmp_path / "summary_diagnostics.csv"
    case_post.write_summary(runs, output, diagnostics_output=diagnostics)
    with output.open(newline="", encoding="utf-8") as stream:
        main_rows = list(csv.DictReader(stream))
    with diagnostics.open(newline="", encoding="utf-8") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    assert len(main_rows) == 1
    assert main_rows[0]["role"] == "production"
    assert main_rows[0]["authoritative"] == "true"
    assert len(diagnostic_rows) == 1
    assert diagnostic_rows[0]["role"] == "smoke"


def test_cm_uses_geometric_mac_scale_and_preserves_raw_value():
    raw = -0.07
    assert case_post._rescale_cm(raw) == pytest.approx(raw * 3.403 / 3.4716)
    row = case_post._summary_row({"case": "C5", "CL": 0.4895, "CD": 0.04322,
                                  "CM": raw}, "C5", "fixture")
    assert row["CM"] == pytest.approx(raw * 3.403 / 3.4716)
    assert row["CM_solver_raw"] == raw
    assert "3.4716 m" in row["CM_convention"]


def test_common_cl_correction_uses_last_two_settled_points():
    corrected = case_post.common_cl_correction(
        {"case": "C5", "CL": 0.48, "CD": 0.04,
         "settled_alpha_points": [
             {"settled": True, "alpha_deg": 2.0, "CL": 0.48, "CD": 0.04},
             {"settled": True, "alpha_deg": 3.0, "CL": 0.50, "CD": 0.042}]}, "C5")
    assert corrected["dCD_dCL"] == pytest.approx(0.1)
    assert corrected["CD_corrected"] == pytest.approx(0.04095)
    assert corrected["L_over_D_corrected"] == pytest.approx(0.4895 / 0.04095)
    assert corrected["slope_source"] == "last_two_settled_alpha_points"
    assert corrected["flag"] == ""


class _FakeSurfaceRequest:
    def __init__(self, surfaces, data_types):
        self.surfaces = surfaces
        self.data_types = data_types


class _FakeScalarRequest:
    def __init__(self, field_name, surfaces, node_value):
        self.field_name = field_name
        self.surfaces = surfaces
        self.node_value = node_value


class _FakeSurfaceDataType:
    FacesNormal = "face-normal"
    FacesCentroid = "centroid"
    Vertices = "vertices"
    FacesConnectivity = "connectivity"


def _unit_squares(centroids):
    """One 1 m^2 square per centroid in the y-z plane, vertex order giving +x."""
    offsets = np.asarray([[0.0, -0.5, -0.5], [0.0, 0.5, -0.5],
                          [0.0, 0.5, 0.5], [0.0, -0.5, 0.5]])
    vertices = np.concatenate([centroid + offsets for centroid in centroids])
    connectivity = [np.arange(4 * index, 4 * index + 4) for index in range(len(centroids))]
    return vertices, connectivity


class _FakeFieldData:
    def __init__(self, face_sets):
        self.face_sets = face_sets

    def get_field_data(self, request):
        if isinstance(request, _FakeSurfaceRequest):
            return {zone: SimpleNamespace(
                # Fluent-like: direction right, magnitude not the face area
                face_normals=0.34 * self.face_sets[zone]["area_vectors"],
                face_centroids=self.face_sets[zone]["centroids"],
                vertices=_unit_squares(self.face_sets[zone]["centroids"])[0],
                connectivity=_unit_squares(self.face_sets[zone]["centroids"])[1])
                    for zone in request.surfaces}
        return {zone: self.face_sets[zone][request.field_name]
                for zone in request.surfaces}


class _FakeFileSettings:
    def read_case(self, file_name):
        self.case_path = file_name

    def read_data(self, file_name):
        self.data_path = file_name


class _FakeSession:
    def __init__(self, face_sets):
        names = list(dict.fromkeys([*face_sets, "farfield:1"]))
        self.settings = SimpleNamespace(
            file=_FakeFileSettings(),
            setup=SimpleNamespace(boundary_conditions=SimpleNamespace(
                wall={name: {} for name in names if not name.startswith("farfield")},
                pressure_far_field={name: {} for name in names if name.startswith("farfield")})))
        self.fields = SimpleNamespace(field_data=_FakeFieldData(face_sets))
        self.exited = False

    def exit(self):
        self.exited = True


def _synthetic_faces():
    names = ("mainwing:1", "fuselage:1", "horstab:1", "vertstab:1",
             "engine:1", "pylon:1", "nacelle_duct:1", "fairing:1", "farfield:1")
    output = {}
    for name in names:
        x_values = [0.0, 5.0, 10.0] if name.startswith("engine") else [5.0]
        n = len(x_values)
        output[name] = {
            "centroids": np.asarray([[x, 1.0, 0.0] for x in x_values]),
            "area_vectors": np.tile(np.asarray([[1.0, 0.0, 0.0]]), (n, 1)),
            "pressure-coefficient": np.full(n, -0.2), "y-plus": np.full(n, 12.0),
            "x-wall-shear": np.full(n, 2.0), "y-wall-shear": np.zeros(n),
            "z-wall-shear": np.zeros(n)}
    del output["farfield:1"]
    return output


def test_force_breakdown_fake_session_writes_zone_and_nacelle_region_forces(tmp_path,
                                                                           monkeypatch):
    run = tmp_path / "C5" / "iH0"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "case": "C5", "alpha_deg": 4.0, "CL": 0.49, "CD": 0.05}), encoding="utf-8")
    (run / "final.cas.h5").touch()
    (run / "final.dat.h5").touch()
    session = _FakeSession(_synthetic_faces())
    calls = []
    monkeypatch.setattr(force_breakdown, "launch_fluent",
                        lambda count, cwd: calls.append((count, cwd)) or session)
    monkeypatch.setattr(force_breakdown, "_field_request_types",
                        lambda: (_FakeScalarRequest, _FakeSurfaceRequest, _FakeSurfaceDataType))
    outputs = force_breakdown.process_run(run)
    assert calls[0][0] == 4
    assert session.exited
    assert outputs["face_count"] == 10
    with outputs["force_breakdown"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["scope"] for row in rows} == {"zone", "region", "total"}
    engine = next(row for row in rows if row["scope"] == "zone" and row["zone"] == "engine:1")
    assert float(engine["CD_pressure"]) != 0.0
    assert float(engine["CD_viscous"]) > 0.0
    regions = {row["region"]: row for row in rows if row["scope"] == "region"}
    assert {"inlet_face_lip", "external_barrel", "base_exit", "pylon"} <= set(regions)
    assert [int(regions[name]["face_count"]) for name in
            ("inlet_face_lip", "external_barrel", "base_exit", "pylon")] == [1, 1, 1, 1]
    with outputs["walls_area"].open(newline="", encoding="utf-8") as stream:
        area_rows = list(csv.DictReader(stream))
    assert area_rows[0]["area-vector-x"] == "1.0"
    assert area_rows[0]["wall-yplus"] == "12.0"


def test_yplus_component_bands_use_wall_area_vectors(tmp_path):
    zone_groups = {"wing": ["mainwing:1"], "fuselage": [], "htail": [], "vtail": [],
                   "nacelle": [], "duct": [], "fairing": [], "total": ["mainwing:1"]}
    wall_meshes = {"mainwing:1": SimpleNamespace(n_cells=3)}
    area_path = tmp_path / "walls_area.csv"
    with area_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("zone", "wall-yplus", "area-vector-x",
                                                    "area-vector-y", "area-vector-z"))
        writer.writeheader()
        for value, area in ((0.5, 1.0), (2.0, 3.0), (10.0, 6.0)):
            writer.writerow({"zone": "mainwing:1", "wall-yplus": value,
                             "area-vector-x": area, "area-vector-y": 0, "area-vector-z": 0})
    stats = component_yplus_bands(wall_meshes, zone_groups, area_path)
    wing = stats["components"]["wing"]
    assert stats["weighting"] == "area"
    assert wing["fraction_lt_1"] == pytest.approx(0.1)
    assert wing["fraction_1_to_5"] == pytest.approx(0.3)
    assert wing["fraction_5_to_30"] == pytest.approx(0.6)
    assert wing["fraction_gt_300"] == pytest.approx(0.0)


def test_yplus_component_bands_prefer_area_vectors_in_walls_csv(tmp_path):
    zone_groups = {"wing": ["mainwing:1"], "fuselage": [], "htail": [], "vtail": [],
                   "nacelle": [], "duct": [], "fairing": [], "total": ["mainwing:1"]}
    wall_meshes = {"mainwing:1": SimpleNamespace(n_cells=3)}
    headers = ("zone", "wall-yplus", "area-vector-x", "area-vector-y", "area-vector-z")
    walls_path = tmp_path / "walls.csv"
    area_path = tmp_path / "walls_area.csv"
    for path, areas in ((walls_path, (1.0, 3.0, 6.0)),
                        (area_path, (1.0, 1.0, 1.0))):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            for value, area in zip((0.5, 2.0, 10.0), areas):
                writer.writerow({"zone": "mainwing:1", "wall-yplus": value,
                                 "area-vector-x": area, "area-vector-y": 0,
                                 "area-vector-z": 0})

    stats = component_yplus_bands(
        wall_meshes, zone_groups, area_path, walls_path=walls_path)

    assert stats["weighting"] == "area"
    assert stats["components"]["wing"]["fraction_lt_1"] == pytest.approx(0.1)


def test_run_summary_tables_include_solver_and_precision(tmp_path):
    runs = tmp_path / "runs"
    production = runs / "C5" / "iH0"
    diagnostic = runs / "C2" / "gpu"
    production.mkdir(parents=True)
    diagnostic.mkdir(parents=True)
    (production / "result.json").write_text(json.dumps({
        "case": "C5", "status": "converged", "alpha_deg": 4.0,
        "CL": 0.49, "CD": 0.04, "solver": "cpu-db", "precision": "double"}),
        encoding="utf-8")
    (diagnostic / "result.json").write_text(json.dumps({
        "case": "C2", "status": "audit_failed", "alpha_deg": 4.0,
        "CL": 0.49, "CD": 0.04, "solver": "gpu-pb", "precision": "single",
        "solver_audit": ["unsupported setting"], "gpu_native_solver_active": False}),
        encoding="utf-8")
    summary = tmp_path / "summary.csv"
    diagnostics = tmp_path / "summary_diagnostics.csv"

    case_post.write_summary(runs, summary, diagnostics)

    with summary.open(newline="", encoding="utf-8") as stream:
        production_table = list(csv.DictReader(stream))
    with diagnostics.open(newline="", encoding="utf-8") as stream:
        diagnostic_table = list(csv.DictReader(stream))
    assert production_table[0]["solver"] == "cpu-db"
    assert production_table[0]["precision"] == "double"
    assert diagnostic_table[0]["solver"] == "gpu-pb"
    assert diagnostic_table[0]["precision"] == "single"


def test_unqualified_gpu_result_is_non_authoritative(tmp_path):
    result_path = tmp_path / "C2" / "gpu" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "status": "audit_failed", "solver": "gpu-pb",
        "solver_audit": [], "gpu_native_solver_active": False}),
        encoding="utf-8")

    role, authoritative = case_post._run_role_and_authority(
        result_path, "C2", {}, "pass")

    assert role == "calibration"
    assert authoritative is False


def test_yplus_band_fractions_are_area_weighted():
    fractions = render_cfd.yplus_band_fractions(
        np.asarray([0.5, 2.0, 10.0]), np.asarray([1.0, 3.0, 6.0]))
    assert fractions["fraction_lt_1"] == pytest.approx(0.1)
    assert fractions["fraction_1_to_5"] == pytest.approx(0.3)
    assert fractions["fraction_5_to_30"] == pytest.approx(0.6)


def test_mach_slice_camera_uses_surface_bounds_not_farfield_volume(monkeypatch):
    volume = pv.ImageData(dimensions=(8, 8, 8), spacing=(25.0, 25.0, 25.0),
                          origin=(-100.0, -100.0, -100.0))
    volume.point_data["velocity"] = np.tile([1.0, 0.0, 0.0], (volume.n_points, 1))
    volume.point_data["Mach-number"] = np.full(volume.n_points, 0.74)
    surface = pv.Cube(bounds=(11.0, 13.0, -0.05, 0.05, 3.0, 4.0))
    captured = []

    class _Camera:
        parallel_projection = False
        parallel_scale = None

    class _Plotter:
        def __init__(self, **kwargs):
            self.camera = _Camera()
            captured.append(self)

        def set_background(self, value):
            pass

        def add_mesh(self, *args, **kwargs):
            pass

        def screenshot(self, path):
            self.path = path

        def close(self):
            pass

    monkeypatch.setattr(render_cfd.pv, "Plotter", _Plotter)
    render_cfd.render_mach_slice(volume, 0.0, "mach.png", surface)
    focus = np.asarray(captured[0].camera_position[1])
    assert focus[0] == pytest.approx(12.0)
    assert focus[2] == pytest.approx(3.5)
    assert captured[0].camera.parallel_projection
    assert captured[0].camera.parallel_scale < 100.0


def test_polygon_area_vectors_use_exact_area_and_fluent_direction():
    centroids = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vertices, connectivity = _unit_squares(centroids)
    normals = np.asarray([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]])
    item = SimpleNamespace(vertices=vertices, connectivity=connectivity)
    vectors = polygon_area_vectors(item, normals)
    assert np.allclose(vectors, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    flat = np.concatenate([[4, *face] for face in connectivity])
    assert np.allclose(polygon_area_vectors(
        SimpleNamespace(vertices=vertices, connectivity=flat), normals), vectors)


def test_mapping_resolves_shared_nearest_source_to_unused_neighbour():
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    target = np.asarray([[0.05, 0.0, 0.0], [0.45, 0.0, 0.0], [2.0, 0.0, 0.0]])
    values = np.asarray([10.0, 11.0, 12.0])
    mapped, _ = nearest_centroid_map(source, target, values,
                                     face_sizes=np.ones(3))
    assert mapped.tolist() == [10.0, 11.0, 12.0]
