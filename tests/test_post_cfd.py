import csv
import json

import h5py
import numpy as np
import pyvista as pv
import pytest

from b737wing.cfd.solve.fluent_io import resolve_zone_groups
from b737wing.post import case_post
from b737wing.post.breguet import breguet_range_nmi
from b737wing.post.case_post import SUMMARY_COLUMNS, _map_wall_data, _write_forces, write_summary
from b737wing.post.mesh_view import read_walls
from b737wing.post.render_cfd import _streamlines, render_flow_images
from b737wing.post.sections import nearest_centroid_map, section_from_cut, wall_cut, write_cp_sections


def test_breguet_cruise_reference_and_takeoff_case():
    cruise = breguet_range_nmi(0.4895, 0.04322, "C5")
    assert cruise["range_nmi"] == pytest.approx(2474, abs=3)
    takeoff = breguet_range_nmi(0.4895, 0.04322, "C3")
    assert takeoff["range_nmi"] is None
    assert "take-off" in takeoff["reason"]


def test_wall_cut_closed_extruded_box():
    x0, x1, y0, y1, z0, z1 = 1.0, 3.0, -0.4, 0.4, -0.5, 0.5
    polygons = [
        np.asarray([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]),
        np.asarray([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),
        np.asarray([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
        np.asarray([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
    ]
    cut = wall_cut(polygons, 0.0)
    endpoints = [tuple(np.round(point, 9)) for row in cut for point in (row["p1"], row["p2"])]
    assert len(cut) == 4
    assert set(endpoints) == {(x0, z0), (x0, z1), (x1, z0), (x1, z1)}
    assert all(endpoints.count(point) == 2 for point in set(endpoints))
    assert min(point[0] for point in endpoints) == x0
    assert max(point[0] for point in endpoints) == x1


def test_section_integrates_known_upper_and_lower_cp():
    x_edges = np.linspace(0.0, 1.0, 21)
    polygons, cp = [], []
    for left, right in zip(x_edges[:-1], x_edges[1:]):
        upper_left = 0.12 * np.sqrt(max(left * (1.0 - left), 0.0))
        upper_right = 0.12 * np.sqrt(max(right * (1.0 - right), 0.0))
        lower_left, lower_right = -upper_left, -upper_right
        polygons.append(np.asarray([[left, -0.1, upper_left], [left, 0.1, upper_left],
                                    [right, 0.1, upper_right], [right, -0.1, upper_right]]))
        cp.append(-1.0)
        polygons.append(np.asarray([[left, -0.1, lower_left], [left, 0.1, lower_left],
                                    [right, 0.1, lower_right], [right, -0.1, lower_right]]))
        cp.append(0.5)
    cut = wall_cut(polygons, 0.0, cp)
    section = section_from_cut(cut, 0.0, 0.0, 1.0)
    assert section["cl"] == pytest.approx(1.5, rel=0.02)
    assert np.allclose(section["upper"]["cp"], -1.0)
    assert np.allclose(section["lower"]["cp"], 0.5)
    assert np.all(section["upper"]["x_c"] < 1.0)
    assert np.all(section["lower"]["x_c"] > 0.0)


def test_nearest_centroid_map_rejects_five_mm_geometry_offset():
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = source + np.asarray([0.005, 0.0, 0.0])
    with pytest.raises(ValueError, match="median wall-centroid match distance"):
        nearest_centroid_map(source, target, np.asarray([1.0, 2.0]))


def test_forces_csv_shares_and_full_aircraft_lbf(tmp_path):
    result = {"case": "C1", "CL": 0.5, "CD": 0.05, "lift_N": 10000.0, "drag_N": 1000.0,
              "components": {"wing": {"CL": 0.4, "CD": 0.03, "CM": -0.02},
                             "fuselage": {"CL": 0.1, "CD": 0.02, "CM": 0.01}}}
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    path = tmp_path / "forces.csv"
    _write_forces(json.loads(result_path.read_text(encoding="utf-8")), path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(float(row["CD_share"]) for row in rows) == pytest.approx(1.0, abs=1e-6)
    for row in rows:
        assert float(row["lift_lbf"]) == pytest.approx(float(row["lift_N"]) / 4.4482216)
        assert float(row["drag_lbf"]) == pytest.approx(float(row["drag_N"]) / 4.4482216)
        assert row["model"] == "full-aircraft"


def test_summary_writes_two_run_rows_and_required_columns(tmp_path):
    runs = tmp_path / "runs"
    for case, tag, alpha in (("C1", "cruise_a", 2.0), ("C5", "cruise_b", 3.0)):
        folder = runs / case / "iH0"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({
            "case": case, "tag": tag, "alpha_deg": alpha, "CL": 0.49, "CD": 0.043,
            "CD_ex_duct": 0.041, "CM": -0.02, "L_over_D": 11.4,
            "L_over_D_ex_duct": 11.8, "lift_N": 510000, "lift_lbf": 114600,
            "drag_N": 44000, "drag_lbf": 9891, "iterations": 900,
            "status": "converged"}), encoding="utf-8")
    output = tmp_path / "post_v2" / "summary.csv"
    assert write_summary(runs, output) == output
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames) == SUMMARY_COLUMNS
    assert len(rows) == 2
    assert {(row["case"], row["tag"]) for row in rows} == {
        ("C1", "cruise_a"), ("C5", "cruise_b")}


def test_render_flow_cfd_smoke_writes_1920_by_1080_png(tmp_path):
    surface = pv.Cube(bounds=(0, 1, -0.5, 0.5, 0, 0.5))
    surface.cell_data["Cp"] = np.linspace(-1.0, 0.5, surface.n_cells)
    volume = pv.ImageData(dimensions=(12, 10, 8), spacing=(0.25, 0.2, 0.2),
                          origin=(-1, -1, -0.5))
    velocity = np.zeros((volume.n_points, 3), dtype=float)
    velocity[:, 0] = 1.0
    volume.point_data["velocity"] = velocity
    volume.point_data["Mach-number"] = np.full(volume.n_points, 0.74)
    paths = render_flow_images({"mainwing": surface}, volume, tmp_path, 1.0)
    assert [path.name for path in paths] == ["flow_image.png", "flow_image_top.png"]
    for path in paths:
        assert path.is_file()
        from matplotlib.image import imread
        pixels = imread(path)
        assert pixels.shape[1] == 1920
        assert pixels.shape[0] == 1080


ZONE_NAMES = ("mainwing:1", "fuselage-1", "horstab", "vertstab:1", "engine",
              "pylon-1", "nacelle_duct", "fairing:1", "farfield")


def _write_zone_mesh(path, names=ZONE_NAMES):
    with h5py.File(path, "w") as output:
        mesh = output.create_group("meshes/1")
        coords = mesh.create_group("nodes/coords")
        coords.create_dataset("1", data=np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                                                    dtype=float))
        topology = mesh.create_group("faces/zoneTopology")
        topology.create_dataset("name", data=np.asarray([";".join(names).encode()]))
        topology.create_dataset("minId", data=np.arange(1, len(names) + 1, dtype=int))
        face_nodes = mesh.create_group("faces/nodes")
        for zone_index in range(1, len(names) + 1):
            section = face_nodes.create_group(str(zone_index))
            section.attrs["minId"] = np.asarray([zone_index])
            section.create_dataset("nnodes", data=np.asarray([4]))
            section.create_dataset("nodes", data=np.asarray([1, 2, 3, 4]))


def test_suffixed_wall_zones_map_and_missing_group_reports_zones(tmp_path):
    mesh_path = tmp_path / "mesh.msh.h5"
    _write_zone_mesh(mesh_path)
    walls, groups = read_walls(mesh_path, mirror=False, return_groups=True)
    assert set(walls) == set(ZONE_NAMES) - {"farfield"}
    assert groups["wing"] == ["mainwing:1"]
    rows = {}
    for zone, mesh in walls.items():
        center = mesh.cell_centers().points[0]
        rows[zone] = [{"x": str(center[0]), "y": str(center[1]), "z": str(center[2]),
                       "pressure-coefficient": "-0.2", "wall-yplus": "1.3"}]
    mapped = _map_wall_data(walls, rows, groups)
    assert set(mapped) == set(walls)
    assert mapped["mainwing:1"]["mesh"].cell_data["Cp"][0] == pytest.approx(-0.2)

    missing_mesh = tmp_path / "mesh_missing_group.msh.h5"
    _write_zone_mesh(missing_mesh, tuple(name for name in ZONE_NAMES if name != "fairing:1"))
    with pytest.raises(ValueError, match="zones found:.*mainwing:1"):
        read_walls(missing_mesh, mirror=False)


def test_inclined_thin_section_splits_surfaces_and_keeps_positive_cl():
    alpha = np.radians(8.0)
    x_edges = np.linspace(0.0, 1.0, 21)
    polygons, cp = [], []
    for left, right in zip(x_edges[:-1], x_edges[1:]):
        thickness_left = 0.08 * np.sqrt(max(left * (1.0 - left), 0.0))
        thickness_right = 0.08 * np.sqrt(max(right * (1.0 - right), 0.0))
        for sign, value in ((1.0, -1.0), (-1.0, 0.5)):
            local = ((left, sign * thickness_left), (right, sign * thickness_right))
            transformed = [(x * np.cos(alpha) - z * np.sin(alpha),
                            x * np.sin(alpha) + z * np.cos(alpha)) for x, z in local]
            (x0, z0), (x1, z1) = transformed
            polygons.append(np.asarray([[x0, -0.1, z0], [x0, 0.1, z0],
                                        [x1, 0.1, z1], [x1, -0.1, z1]]))
            cp.append(value)
    cut = wall_cut(polygons, 0.0, cp)
    points = [point for row in cut for point in (row["p1"], row["p2"])]
    x_le = min(point[0] for point in points)
    x_te = max(point[0] for point in points)
    section = section_from_cut(cut, 0.0, x_le, x_te - x_le, 8.0)
    assert np.allclose(section["upper"]["cp"], -1.0)
    assert np.allclose(section["lower"]["cp"], 0.5)
    assert section["cl"] > 0.0
    assert section["cl"] == pytest.approx(1.5, rel=0.03)


def test_streamline_geometry_is_reflected_to_negative_y():
    volume = pv.ImageData(dimensions=(15, 10, 8), spacing=(0.25, 0.1, 0.2),
                          origin=(-1, 0, -0.5))
    velocity = np.zeros((volume.n_points, 3), dtype=float)
    velocity[:, 0] = 1.0
    volume.point_data["p7_velocity"] = velocity
    volume.set_active_vectors("p7_velocity")
    surface = pv.Cube(bounds=(0, 1, 0.1, 0.5, 0, 0.5))
    lines = _streamlines(volume, surface, 1.0)
    assert lines.n_points > 0
    assert np.min(lines.points[:, 1]) < 0.0
    assert np.max(lines.points[:, 1]) > 0.0


def test_present_ensight_streamline_failure_propagates(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"case": "C1", "CL": 0.5, "CD": 0.05}),
                                      encoding="utf-8")
    (run / "walls.csv").touch()
    mesh_path = tmp_path / "mesh.msh.h5"
    mesh_path.touch()
    ensight = run / "ensight"
    ensight.mkdir()
    (ensight / "final.case").touch()

    groups = resolve_zone_groups(list(ZONE_NAMES))
    wall_meshes = {zone: pv.Cube(bounds=(0, 1, 0, 1, -0.1, 0.1))
                   for zone in groups["total"]}
    rows = {}
    for zone, mesh in wall_meshes.items():
        centers = mesh.cell_centers().points
        rows[zone] = [{"x": str(point[0]), "y": str(point[1]), "z": str(point[2]),
                       "pressure-coefficient": "-0.2", "wall-yplus": "1.3"}
                      for point in centers]
    monkeypatch.setattr(case_post, "read_walls", lambda *args, **kwargs: (wall_meshes, groups))
    monkeypatch.setattr(case_post, "_wall_rows", lambda path: rows)
    monkeypatch.setattr(case_post, "_write_forces", lambda result, output: None)
    monkeypatch.setattr(case_post, "write_cp_sections", lambda *args: None)
    monkeypatch.setattr(case_post, "write_spanwise", lambda *args: None)
    monkeypatch.setattr(case_post, "write_yplus_products", lambda *args, **kwargs: None)
    monkeypatch.setattr(case_post.pv, "read", lambda path: object())
    monkeypatch.setattr(case_post, "render_mach_slice", lambda *args: None)

    def fail_streamlines(*args):
        raise ValueError("streamline integration returned no lines")

    monkeypatch.setattr(case_post, "render_flow_images", fail_streamlines)
    with pytest.raises(ValueError, match="streamline integration returned no lines"):
        case_post.process_run("C1", run, mesh_path=mesh_path, out_dir=tmp_path / "post")


def test_empty_required_fuselage_cut_raises(tmp_path, monkeypatch):
    def section_at(wing, y_m, alpha, eta):
        return {"y_m": y_m, "eta": eta, "cl": 0.0, "cdp": 0.0,
                "upper": {"x_c": np.asarray([0.0, 1.0]), "cp": np.asarray([-1.0, -1.0])},
                "lower": {"x_c": np.asarray([0.0, 1.0]), "cp": np.asarray([0.5, 0.5])}}

    monkeypatch.setattr("b737wing.post.sections.section_at", section_at)
    empty_fuselage = pv.PolyData()
    empty_fuselage.cell_data["Cp"] = np.asarray([], dtype=float)
    with pytest.raises(ValueError, match="required fuselage centreline cut at y=0.02 m is empty"):
        write_cp_sections({}, {"mesh": empty_fuselage}, 0.0, tmp_path)
