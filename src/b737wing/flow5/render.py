"""Off-screen flow-field renders for the five B737-200 flow5 cases."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from .cases import ROOT_SECTIONS, case_config, case_names
from .parse import OperatingPoint, parse_op_points


pv.OFF_SCREEN = True


def _chord(y: float) -> float:
    stations = [float(section["y"]) for section in ROOT_SECTIONS]
    chords = [float(section["chord"]) for section in ROOT_SECTIONS]
    return float(np.interp(abs(y), stations, chords))


def _section_values(y: float) -> tuple[float, float, float]:
    """Return the section LE x, quarter-chord x, and dihedral z at |y|."""
    distance = abs(float(y))
    for left, right in zip(ROOT_SECTIONS, ROOT_SECTIONS[1:]):
        if distance <= float(right["y"]):
            dy = float(right["y"]) - float(left["y"])
            fraction = (distance - float(left["y"])) / dy
            x = float(left["x"]) + fraction * (float(right["x"]) - float(left["x"]))
            chord = _chord(distance)
            z = 0.0
            for before, after in zip(ROOT_SECTIONS, ROOT_SECTIONS[1:]):
                segment = min(distance, float(after["y"])) - float(before["y"])
                if segment > 0:
                    z += segment * math.tan(math.radians(float(before["dihedral"])))
                if distance <= float(after["y"]):
                    break
            return x, x + 0.25 * chord, z
    last = ROOT_SECTIONS[-1]
    x = float(last["x"])
    chord = float(last["chord"])
    z = sum(
        (float(after["y"]) - float(before["y"]))
        * math.tan(math.radians(float(before["dihedral"])))
        for before, after in zip(ROOT_SECTIONS, ROOT_SECTIONS[1:])
    )
    return x, x + 0.25 * chord, z


def _cp_coordinates(point: OperatingPoint) -> tuple[np.ndarray, np.ndarray]:
    xyz: list[list[float]] = []
    cp: list[float] = []
    for table in point.cp_tables:
        lookup = {name.strip().casefold(): i for i, name in enumerate(table.columns)}
        required = ("ctrlpt.x", "ctrlpt.y", "ctrlpt.z", "cp")
        if any(name not in lookup for name in required):
            raise ValueError(f"{point.path}: Cp panel table lacks CtrlPt xyz or Cp columns")
        for row in table.rows:
            xyz.append([float(row[lookup[name]]) for name in required[:3]])
            cp.append(float(row[lookup["cp"]]))
    if not xyz or not np.isfinite(xyz).all() or not np.isfinite(cp).all():
        raise ValueError(f"{point.path}: empty or non-finite panel Cp data")
    coordinates = np.asarray(xyz, dtype=float)
    values = np.asarray(cp, dtype=float)
    # flow5 exports the right half in some versions; reflect those controls to
    # cover the full-span STL while preserving the symmetric Cp distribution.
    if np.all(coordinates[:, 1] >= -1e-10) and np.any(coordinates[:, 1] > 1e-10):
        mirrored = coordinates.copy()
        mirrored[:, 1] *= -1.0
        coordinates = np.vstack((coordinates, mirrored))
        values = np.concatenate((values, values.copy()))
    return coordinates, values


def _map_cp_to_mesh(mesh: pv.PolyData, coordinates: np.ndarray,
                    values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map nearest panel Cp to cell centers and return values and distances."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) != len(values):
        raise ValueError("Cp coordinates and values have incompatible shapes")
    centers = mesh.cell_centers().points
    distances, indices = cKDTree(coordinates).query(centers, k=1)
    return np.asarray(values[indices], dtype=float), np.asarray(distances, dtype=float)


def _read_strips(path: Path) -> list[tuple[float, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"y_m", "cl"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected y_m and cl columns")
        rows = [(float(row["y_m"]), float(row["cl"])) for row in reader]
    if len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError(f"{path}: requires at least two finite strip rows")
    rows.sort(key=lambda row: row[0])
    if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError(f"{path}: strip y coordinates must be unique and increasing")
    return rows


def _vortex_segments(strips: list[tuple[float, float]], velocity: float,
                     span: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make finite horseshoes with a Rankine core and 20-span wake legs."""
    semi_span = float(ROOT_SECTIONS[-1]["y"])
    centers = np.asarray([row[0] for row in strips], dtype=float)
    if centers[0] < -semi_span - 0.05 or centers[-1] > semi_span + 0.05:
        raise ValueError("strip rows lie outside the configured wing span")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[0], edges[-1] = -semi_span, semi_span
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    strengths: list[float] = []
    cores: list[float] = []
    far_x = max(float(section["x"]) + float(section["chord"]) for section in ROOT_SECTIONS) + 20.0 * span

    def add(start: tuple[float, float, float], end: tuple[float, float, float],
            gamma: float, core: float) -> None:
        starts.append(list(start))
        ends.append(list(end))
        strengths.append(gamma)
        cores.append(core)

    for index, (y, cl) in enumerate(strips):
        y0, y1 = float(edges[index]), float(edges[index + 1])
        x0, xq0, z0 = _section_values(y0)
        x1, xq1, z1 = _section_values(y1)
        del x0, x1
        chord = _chord(y)
        gamma = 0.5 * velocity * chord * cl
        core = 0.05 * chord
        # All bound elements use increasing y; the opposite signed root/tip
        # wake legs make adjacent horseshoes cancel as a lifting-line sheet.
        add((xq0, y0, z0), (xq1, y1, z1), gamma, core)
        add((xq0, y0, z0), (far_x, y0, z0), -gamma, core)
        add((xq1, y1, z1), (far_x, y1, z1), gamma, core)
    return (np.asarray(starts), np.asarray(ends), np.asarray(strengths),
            np.asarray(cores))


def _induced_velocity(points: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                      strengths: np.ndarray, cores: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    r1 = points[:, None, :] - starts[None, :, :]
    r2 = points[:, None, :] - ends[None, :, :]
    filament = ends - starts
    cross = np.cross(r1, r2)
    cross_sq = np.einsum("nsi,nsi->ns", cross, cross)
    r1_norm = np.maximum(np.linalg.norm(r1, axis=2), 1e-14)
    r2_norm = np.maximum(np.linalg.norm(r2, axis=2), 1e-14)
    direction = r1 / r1_norm[:, :, None] - r2 / r2_norm[:, :, None]
    projection = np.einsum("si,nsi->ns", filament, direction)
    length_sq = np.maximum(np.einsum("si,si->s", filament, filament), 1e-24)
    perpendicular_sq = cross_sq / length_sq[None, :]
    # Rankine solid-body core: regularize each filament inside its 0.05c core.
    core_factor = np.minimum(1.0, perpendicular_sq / np.maximum(cores[None, :] ** 2, 1e-24))
    coefficient = (strengths[None, :] * projection * core_factor
                   / (4.0 * math.pi * np.maximum(cross_sq, 1e-28)))
    coefficient[cross_sq < 1e-28] = 0.0
    return np.sum(cross * coefficient[:, :, None], axis=1)


def _velocity(points: np.ndarray, segments: tuple[np.ndarray, ...],
              freestream: np.ndarray) -> np.ndarray:
    return _induced_velocity(points, *segments) + np.asarray(freestream, dtype=float)


def _clip_streamlines(streamlines: pv.PolyData, x_min: float,
                      x_max: float) -> pv.PolyData:
    """Clip polyline points to an x interval, interpolating clipped endpoints."""
    if x_min > x_max:
        raise ValueError("streamline x minimum must not exceed its maximum")
    source_points = np.asarray(streamlines.points, dtype=float)
    source_data = {name: np.asarray(values)
                   for name, values in streamlines.point_data.items()}
    output_points: list[np.ndarray] = []
    output_data: dict[str, list[np.ndarray]] = {name: [] for name in source_data}
    output_lines: list[int] = []

    def point_data_at(name: str, first: int, second: int, fraction: float) -> np.ndarray:
        values = source_data[name]
        if np.issubdtype(values.dtype, np.number):
            return values[first] + (values[second] - values[first]) * fraction
        return values[first if fraction <= 0.5 else second]

    def append_path(points: list[np.ndarray], data: dict[str, list[np.ndarray]]) -> None:
        if len(points) < 2:
            return
        start = len(output_points)
        output_points.extend(points)
        for name in output_data:
            output_data[name].extend(data[name])
        output_lines.extend([len(points), *range(start, start + len(points))])

    offset = 0
    connectivity = np.asarray(streamlines.lines, dtype=np.int64)
    while offset < len(connectivity):
        count = int(connectivity[offset])
        ids = connectivity[offset + 1:offset + 1 + count]
        offset += count + 1
        path_points: list[np.ndarray] = []
        path_data: dict[str, list[np.ndarray]] = {name: [] for name in source_data}
        for first, second in zip(ids[:-1], ids[1:]):
            p0, p1 = source_points[first], source_points[second]
            dx = p1[0] - p0[0]
            if dx == 0.0:
                if x_min <= p0[0] <= x_max:
                    lower_fraction, upper_fraction = 0.0, 1.0
                else:
                    lower_fraction, upper_fraction = 1.0, 0.0
            else:
                crossings = sorted(((x_min - p0[0]) / dx, (x_max - p0[0]) / dx))
                lower_fraction = max(0.0, crossings[0])
                upper_fraction = min(1.0, crossings[1])
            if lower_fraction > upper_fraction:
                append_path(path_points, path_data)
                path_points = []
                path_data = {name: [] for name in source_data}
                continue

            start_point = p0 + (p1 - p0) * lower_fraction
            end_point = p0 + (p1 - p0) * upper_fraction
            start_data = {name: point_data_at(name, int(first), int(second), lower_fraction)
                          for name in source_data}
            end_data = {name: point_data_at(name, int(first), int(second), upper_fraction)
                        for name in source_data}
            if path_points and np.allclose(path_points[-1], start_point):
                path_points.append(end_point)
                for name in source_data:
                    path_data[name].append(end_data[name])
            else:
                append_path(path_points, path_data)
                path_points = [start_point, end_point]
                path_data = {name: [start_data[name], end_data[name]]
                             for name in source_data}
        append_path(path_points, path_data)

    result = pv.PolyData(np.asarray(output_points, dtype=float).reshape((-1, 3)))
    result.lines = np.asarray(output_lines, dtype=np.int64)
    for name, values in output_data.items():
        result.point_data[name] = np.asarray(values)
    return result


def _streamlines(segments: tuple[np.ndarray, ...], freestream: np.ndarray,
                 span: float) -> pv.PolyData:
    semi_span = float(ROOT_SECTIONS[-1]["y"])
    y_values = np.linspace(-0.95 * semi_span, 0.95 * semi_span, 18)
    seeds = []
    root_chord = _chord(0.0)
    for offset in (-0.15, 0.15):
        for y in y_values:
            _, _, z = _section_values(float(y))
            chord = _chord(float(y))
            le_x = _section_values(float(y))[0]
            seeds.append((le_x - 0.5 * chord, y, z + offset * root_chord))
    current = np.asarray(seeds, dtype=float)
    paths = [[point.copy()] for point in current]
    step_length = max(0.25, span / 50.0)
    dt = step_length / float(np.linalg.norm(freestream))
    tip = ROOT_SECTIONS[-1]
    x_min = float(ROOT_SECTIONS[0]["x"]) - 0.5 * root_chord
    x_limit = float(tip["x"]) + float(tip["chord"]) + 1.5 * root_chord
    max_steps = int(math.ceil((x_limit - min(point[0] for point in seeds)) / step_length)) + 8
    active = np.ones(len(current), dtype=bool)
    for _ in range(max_steps):
        if not np.any(active):
            break
        live = np.flatnonzero(active)
        p = current[live]
        k1 = _velocity(p, segments, freestream)
        k2 = _velocity(p + 0.5 * dt * k1, segments, freestream)
        k3 = _velocity(p + 0.5 * dt * k2, segments, freestream)
        k4 = _velocity(p + dt * k3, segments, freestream)
        updated = p + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        current[live] = updated
        for seed_index, point in zip(live, updated):
            paths[int(seed_index)].append(point.copy())
        active[live[updated[:, 0] >= x_limit]] = False

    all_points: list[np.ndarray] = []
    lines: list[int] = []
    speed_ratios: list[float] = []
    for path in paths:
        array = np.asarray(path)
        start = len(all_points)
        all_points.extend(array)
        lines.extend([len(array), *range(start, start + len(array))])
        speed_ratios.extend(np.linalg.norm(_velocity(array, segments, freestream), axis=1)
                            / float(np.linalg.norm(freestream)))
    result = pv.PolyData(np.asarray(all_points))
    result.lines = np.asarray(lines, dtype=np.int64)
    result.point_data["speed_ratio"] = np.asarray(speed_ratios, dtype=float)
    return _clip_streamlines(result, x_min, x_limit)


def _load_case(case: str, output_root: Path):
    config = case_config(case)
    case_root = output_root / case
    result_path = case_root / "result.json"
    strips_path = case_root / "strips.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cp_alpha = float(result["cp_alpha_deg"])
    alpha = float(result["alpha_deg"])
    lift = float(result["CL"])
    mesh_dir = case_root / "fine" / "pass2" / "flow5-output" / f"{case}_fine_pass2"
    stl_path = mesh_dir / "STL" / f"B737-200 {case} Wing.stl"
    plane_dir = mesh_dir / f"B737-200 {case} Wing" / case
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)
    op_files = sorted(plane_dir.glob("*.csv"))
    if not op_files:
        raise FileNotFoundError(plane_dir / "*.csv")
    if not strips_path.is_file():
        raise FileNotFoundError(strips_path)
    mesh = pv.read(stl_path).clean()
    if mesh.n_cells == 0 or mesh.bounds[2] >= -1e-4 or mesh.bounds[3] <= 1e-4:
        raise ValueError(f"NEEDS_OPUS_DECISION: {stl_path} is not a full-span STL")
    points = parse_op_points(op_files, expected_polar=case)
    point = min((item for item in points if item.alpha is not None),
                key=lambda item: abs(float(item.alpha) - cp_alpha), default=None)
    if point is None:
        raise ValueError(f"{plane_dir}: no parsed operating point has alpha for Cp mapping")
    cp_xyz, cp_values = _cp_coordinates(point)
    # Compare control points with the actual triangle surface, not its vertices.
    _, closest = mesh.find_closest_cell(cp_xyz, return_closest_point=True)
    cp_distance = np.linalg.norm(cp_xyz - closest, axis=1)
    local_chord = np.asarray([_chord(float(y)) for y in cp_xyz[:, 1]])
    if np.any(cp_distance > 0.05 * local_chord):
        ratio = float(np.max(cp_distance / local_chord))
        raise ValueError(f"NEEDS_OPUS_DECISION: Cp controls miss {stl_path} by {ratio:.1%} chord")
    mapped_cp, _ = _map_cp_to_mesh(mesh, cp_xyz, cp_values)
    mesh.cell_data["Cp"] = mapped_cp
    mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
    strips = _read_strips(strips_path)
    velocity = float(config["flight"]["velocity"])
    span = 2.0 * float(ROOT_SECTIONS[-1]["y"])
    return mesh, strips, velocity, span, alpha, lift


def _add_scene(plotter: pv.Plotter, mesh: pv.PolyData, streamline_data: pv.PolyData,
               case: str, alpha: float, lift: float) -> None:
    limits = (-4.0, 1.0) if case == "F3" else (-1.5, 1.0)
    speed_ratio = np.asarray(streamline_data.point_data["speed_ratio"], dtype=float)
    speed_ratio = speed_ratio[np.isfinite(speed_ratio)]
    if not len(speed_ratio):
        raise ValueError("streamlines have no finite speed ratios to display")
    speed_limits = np.percentile(speed_ratio, (2.0, 98.0))
    if speed_limits[0] >= speed_limits[1]:
        delta = max(abs(float(speed_limits[0])) * 1e-6, 1e-6)
        speed_limits = (float(speed_limits[0]) - delta, float(speed_limits[1]) + delta)
    plotter.set_background("white")
    surface = plotter.add_mesh(mesh, scalars="Cp", cmap="turbo", clim=limits,
                               smooth_shading=True, show_scalar_bar=False)
    plotter.add_mesh(streamline_data, scalars="speed_ratio", cmap="coolwarm",
                     clim=tuple(float(value) for value in speed_limits),
                     line_width=2, render_lines_as_tubes=True,
                     show_scalar_bar=False)
    plotter.add_scalar_bar("Cp", mapper=surface.mapper, vertical=False,
                           position_x=0.37, position_y=0.035,
                           width=0.28, height=0.045, title_font_size=18,
                           label_font_size=14, n_labels=6, fmt="%.1f")
    plotter.add_text(f"flow5 {case}  alpha={alpha:.2f} deg  CL={lift:.3f}",
                     position="upper_left", font_size=16, color="black")
    plotter.enable_anti_aliasing("fxaa")


def render_case(case: str, output_root: Path = Path("flow5_v2")) -> dict[str, Path]:
    """Render perspective and planform images for one case."""
    case = case.upper()
    mesh, strips, velocity, span, alpha, lift = _load_case(case, Path(output_root))
    radians = math.radians(alpha)
    freestream = velocity * np.array([math.cos(radians), 0.0, -math.sin(radians)])
    segments = _vortex_segments(strips, velocity, span)
    wake = _streamlines(segments, freestream, span)
    destination = Path(output_root) / case
    destination.mkdir(parents=True, exist_ok=True)
    front_path = destination / "flow_image.png"
    top_path = destination / "flow_image_top.png"
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    try:
        _add_scene(plotter, mesh, wake, case, alpha, lift)
        bounds = mesh.bounds
        wing_focus = (0.5 * (bounds[0] + bounds[1]), 0.0,
                      0.5 * (bounds[4] + bounds[5]))
        elevation = math.radians(25.0)
        azimuth = math.radians(35.0)
        camera_distance = 36.0
        camera_offset = camera_distance * np.array((
            -math.cos(elevation) * math.cos(azimuth),
            -math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ))
        camera_forward = -camera_offset / camera_distance
        camera_right = np.cross(camera_forward, (0.0, 0.0, 1.0))
        camera_right /= np.linalg.norm(camera_right)
        camera_position = np.asarray(wing_focus) + camera_offset
        relative_points = np.asarray(mesh.points) - camera_position
        depths = relative_points @ camera_forward
        lateral = relative_points @ camera_right
        projected_x = lateral / depths
        left = int(np.argmin(projected_x))
        right = int(np.argmax(projected_x))
        focus_shift = ((lateral[left] / depths[left] + lateral[right] / depths[right])
                       / (1.0 / depths[left] + 1.0 / depths[right]))
        focus = tuple(np.asarray(wing_focus) + focus_shift * camera_right)
        plotter.camera_position = [
            tuple(np.asarray(focus) + camera_offset),
            focus,
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.view_angle = 32.0
        plotter.show(auto_close=False, interactive=False)
        plotter.render()
        plotter.screenshot(str(front_path), transparent_background=False)
        plotter.camera_position = [
            (wing_focus[0], wing_focus[1], wing_focus[2] + 250.0),
            wing_focus,
            (1.0, 0.0, 0.0),
        ]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = (0.5 * span) / (0.80 * plotter.window_size[0]
                                                           / plotter.window_size[1])
        plotter.render()
        plotter.screenshot(str(top_path), transparent_background=False)
    finally:
        plotter.close()
    if not front_path.is_file() or not top_path.is_file():
        raise RuntimeError(f"rendering {case} did not write both requested images")
    return {"png": front_path, "png_top": top_path}


def render_all(output_root: Path = Path("flow5_v2"),
               cases: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    """Render all cases (or selected CLI cases) and write the image index."""
    output_root = Path(output_root)
    selected = case_names() if cases is None else tuple(case.upper() for case in cases)
    rows: list[dict[str, object]] = []
    for case in selected:
        paths = render_case(case, output_root=output_root)
        result = json.loads((output_root / case / "result.json").read_text(encoding="utf-8"))
        rows.append({"case": case, "alpha_deg": result["alpha_deg"], "CL": result["CL"],
                     "png_path": paths["png"].relative_to(output_root).as_posix(),
                     "png_top_path": paths["png_top"].relative_to(output_root).as_posix()})
    index_path = output_root / "flow_images_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "alpha_deg", "CL", "png_path", "png_top_path"))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=case_names(), help="render one flow5 case")
    args = parser.parse_args(argv)
    render_all(cases=(args.case,) if args.case else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
