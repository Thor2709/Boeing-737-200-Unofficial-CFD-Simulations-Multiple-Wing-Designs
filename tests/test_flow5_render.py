from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from b737wing.flow5.cases import ROOT_SECTIONS
from b737wing.flow5.render import (
    _cp_coordinates,
    _clip_streamlines,
    _induced_velocity,
    _map_cp_to_mesh,
    _velocity,
    _vortex_segments,
    render_case,
)


def _symmetric_pair():
    return _vortex_segments([(-7.0, 0.5), (7.0, 0.5)], velocity=224.33, span=28.19112)


def test_horseshoe_pair_has_downwash_and_outboard_upwash():
    segments = _symmetric_pair()
    downwash = _induced_velocity(np.array([[10.0, 0.0, 1.0]]), *segments)[0]
    upwash = _induced_velocity(np.array([[10.0, 15.0, 1.0]]), *segments)[0]
    assert downwash[2] < 0.0
    assert upwash[2] > 0.0


def test_field_far_upstream_matches_freestream_within_one_percent():
    segments = _symmetric_pair()
    freestream = np.array([224.33, 0.0, 0.0])
    point = np.array([[-10.0 * 28.19112, 0.0, 1.0]])
    total = _velocity(point, segments, freestream)[0]
    assert np.linalg.norm(total - freestream) / np.linalg.norm(freestream) < 0.01


def test_cp_mapping_uses_nearest_controls_and_mirrors_positive_side():
    from b737wing.flow5.parse import CpTable, OperatingPoint

    faces = np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7])
    points = np.array([
        [0.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.4, 0.0], [0.0, 0.4, 0.0],
        [1.0, -0.4, 0.0], [2.0, -0.4, 0.0], [2.0, -0.1, 0.0], [1.0, -0.1, 0.0],
    ])
    mesh = pv.PolyData(points, faces)
    table = CpTable("Main Wing", ["CtrlPt.x", "CtrlPt.y", "CtrlPt.z", "Cp"],
                    [[0.5, 0.25, 0.0, 0.7], [1.5, 0.25, 0.0, -0.2]])
    operating_point = OperatingPoint(Path("synthetic.csv"), "F2", 2.0, {}, {}, [table], "")
    cp_xyz, cp_values = _cp_coordinates(operating_point)
    mapped, distances = _map_cp_to_mesh(mesh, cp_xyz, cp_values)
    assert mapped == pytest.approx([0.7, -0.2])
    assert distances == pytest.approx([0.0, 0.0])


def test_render_f2_writes_full_size_front_and_top_images():
    source = Path("flow5_v2/F2")
    result_path = source / "result.json"
    if not result_path.is_file():
        pytest.skip("flow5_v2/F2/result.json is absent")
    paths = render_case("F2", output_root=Path("flow5_v2"))
    from PIL import Image

    for key in ("png", "png_top"):
        with Image.open(paths[key]) as image:
            assert image.size == (1920, 1080)


def test_streamline_clipping_limits_displayed_x_extent():
    points = np.array([
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    streamlines = pv.PolyData(points)
    streamlines.lines = np.array([4, 0, 1, 2, 3])
    streamlines.point_data["speed_ratio"] = np.array([0.9, 1.0, 1.1, 1.2])
    root_chord = float(ROOT_SECTIONS[0]["chord"])
    tip = ROOT_SECTIONS[-1]
    x_min = float(ROOT_SECTIONS[0]["x"]) - 0.5 * root_chord
    x_max = float(tip["x"]) + float(tip["chord"]) + 1.5 * root_chord

    clipped = _clip_streamlines(streamlines, x_min, x_max)

    assert np.min(clipped.points[:, 0]) == pytest.approx(x_min)
    assert np.max(clipped.points[:, 0]) == pytest.approx(x_max)
    assert np.all((clipped.points[:, 0] >= x_min) & (clipped.points[:, 0] <= x_max))
