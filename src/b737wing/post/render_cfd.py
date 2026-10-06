"""Headless surface, streamline and Mach-slice renders for P7 CFD cases."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


pv.OFF_SCREEN = True
FLOW_IMAGE_SIZE = (1920, 1080)
MACH_IMAGE_SIZE = (1600, 900)
CP_COLOUR_RANGE = (-1.5, 1.0)
STREAMLINE_SEED_Y_COUNT = 31
STREAMLINE_SEED_HEIGHT_FRACTIONS = (0.25, 0.55, 0.85)
STREAMLINE_X_OFFSET_M = 0.5
STREAMLINE_COLOUR_MAP = "coolwarm"
MACH_FRAME_SCALE = 1.5


def _field(dataset, wanted, components=None):
    for location, association in (("point", dataset.point_data), ("cell", dataset.cell_data)):
        for name in association.keys():
            key = str(name).casefold().replace("_", "-").replace(" ", "-")
            if wanted in key:
                values = np.asarray(association[name])
                if components is None or (values.ndim == 2 and values.shape[1] == components):
                    return values, location
    return None, None


def prepare_volume(source):
    """Copy a VTK/EnSight volume and standardise its velocity and Mach arrays."""
    if isinstance(source, pv.MultiBlock):
        try:
            source = source.combine()
        except Exception:
            candidates = [block for block in source if block is not None and block.n_cells]
            if not candidates:
                raise ValueError("EnSight data contains no non-empty volume blocks")
            source = max(candidates, key=lambda block: block.n_cells)
    if not isinstance(source, pv.DataSet) or not source.n_cells:
        raise ValueError("volume data must be a non-empty PyVista dataset")
    velocity, velocity_location = _field(source, "velocity", components=3)
    mach, mach_location = _field(source, "mach-number")
    if velocity is None:
        raise ValueError("volume data has no three-component velocity field")
    if mach is None:
        raise ValueError("volume data has no Mach-number field")
    volume = source.copy(deep=True)
    if velocity_location == "point":
        volume.point_data["p7_velocity"] = velocity
    elif velocity_location == "cell":
        volume.cell_data["p7_velocity"] = velocity
    if mach_location == "point":
        volume.point_data["p7_mach"] = np.asarray(mach).reshape(-1)
    elif mach_location == "cell":
        volume.cell_data["p7_mach"] = np.asarray(mach).reshape(-1)
    if "p7_velocity" in volume.cell_data:
        volume = volume.cell_data_to_point_data(pass_cell_data=True)
    if "p7_velocity" not in volume.point_data:
        raise ValueError("velocity could not be converted to point data for streamlines")
    volume.set_active_vectors("p7_velocity")
    return volume


def _full_surface(walls):
    parts = []
    for mesh in walls.values():
        parts.append(mesh)
        parts.append(mesh.reflect((0, 1, 0), point=(0, 0, 0)))
    if not parts:
        raise ValueError("no wall surfaces to render")
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.merge(part, merge_points=False)
    return combined


def _streamlines(volume, surface, semispan_m):
    bounds = surface.bounds
    v_bounds = volume.bounds
    x_seed = max(v_bounds[0] + 0.01 * (v_bounds[1] - v_bounds[0]),
                 bounds[0] - STREAMLINE_X_OFFSET_M)
    y0, y1 = max(v_bounds[2], -float(semispan_m)), min(v_bounds[3], float(semispan_m))
    z0, z1 = max(v_bounds[4], bounds[4] - 0.25), min(v_bounds[5], bounds[5] + 0.25)
    if y1 <= y0 or z1 <= z0:
        raise ValueError("streamline seed rake does not intersect the EnSight volume")
    ys = np.linspace(y0, y1, STREAMLINE_SEED_Y_COUNT)
    heights = [z0 + fraction * (z1 - z0) for fraction in STREAMLINE_SEED_HEIGHT_FRACTIONS]
    points = np.asarray([[x_seed, y, z] for z in heights for y in ys], dtype=float)
    rake = pv.PolyData(points)
    bounds_array = np.asarray(surface.bounds, dtype=float)
    diagonal = float(np.linalg.norm(bounds_array[1::2] - bounds_array[::2]))
    lines = volume.streamlines_from_source(
        rake, vectors="p7_velocity", integration_direction="forward",
        max_length=max(20.0, 3.0 * diagonal),
        max_steps=2000, terminal_speed=1e-12)
    if not lines.n_points or not lines.n_lines:
        raise ValueError("streamline integration returned no lines")
    lines["speed"] = np.linalg.norm(np.asarray(lines.point_data["p7_velocity"]), axis=1)
    return lines.merge(lines.reflect((0, 1, 0), point=(0, 0, 0)), merge_points=False)


def render_flow_images(walls, volume_source, out_dir, semispan_m):
    """Write mirrored full-aircraft Cp/streamline iso and top views."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    volume = prepare_volume(volume_source)
    surface = _full_surface(walls)
    lines = _streamlines(volume, surface, semispan_m)
    iso_path, top_path = output / "flow_image.png", output / "flow_image_top.png"
    for path, top in ((iso_path, False), (top_path, True)):
        plotter = pv.Plotter(off_screen=True, window_size=FLOW_IMAGE_SIZE)
        plotter.set_background("white")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(surface, scalars="Cp", cmap="turbo", clim=CP_COLOUR_RANGE,
                         smooth_shading=True, specular=0.2,
                         scalar_bar_args={"title": "Cp", "color": "black"})
        plotter.add_mesh(lines, scalars="speed", cmap=STREAMLINE_COLOUR_MAP,
                         line_width=2.0, scalar_bar_args={"title": "Velocity (m/s)",
                                                          "color": "black"})
        bounds = surface.bounds
        center = ((bounds[0] + bounds[1]) * 0.5,
                  (bounds[2] + bounds[3]) * 0.5,
                  (bounds[4] + bounds[5]) * 0.5)
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1.0)
        if top:
            plotter.camera_position = [(center[0], center[1], bounds[5] + 3.0 * span),
                                       center, (1, 0, 0)]
        else:
            plotter.camera_position = [(bounds[0] - 1.3 * span,
                                        bounds[2] - 1.6 * span,
                                        bounds[5] + 1.2 * span), center, (0, 0, 1)]
        plotter.camera.zoom(1.15)
        plotter.screenshot(str(path))
        plotter.close()
    return iso_path, top_path


def render_mach_slice(volume_source, y_m, out_path, surface_source=None):
    """Save Mach contours on a constant-y plane."""
    volume = prepare_volume(volume_source)
    sliced = volume.slice(normal=(0, 1, 0), origin=(0, float(y_m), 0))
    if not sliced.n_cells:
        raise ValueError(f"Mach slice y={y_m:g} m does not intersect the EnSight volume")
    plotter = pv.Plotter(off_screen=True, window_size=MACH_IMAGE_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(sliced, scalars="p7_mach", cmap="turbo",
                     scalar_bar_args={"title": "Mach", "color": "black"})
    if surface_source is None:
        frame_source = sliced
    else:
        frame_source = surface_source.slice(normal=(0, 1, 0), origin=(0, float(y_m), 0))
        if not frame_source.n_points:
            frame_source = surface_source
    bounds = np.asarray(frame_source.bounds, dtype=float)
    center = ((bounds[0] + bounds[1]) * 0.5,
              float(y_m),
              (bounds[4] + bounds[5]) * 0.5)
    width = max(float(bounds[1] - bounds[0]), 0.1)
    height = max(float(bounds[5] - bounds[4]), 0.1)
    view_height = MACH_FRAME_SCALE * max(height, width / (MACH_IMAGE_SIZE[0] / MACH_IMAGE_SIZE[1]))
    distance = max(width, height, 1.0) * 2.0
    plotter.camera_position = [(center[0], center[1] + distance, center[2]), center, (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = view_height * 0.5
    plotter.screenshot(str(out_path))
    plotter.close()
    return Path(out_path)


YPLUS_BANDS = (("fraction_lt_1", 0.0, 1.0),
               ("fraction_1_to_5", 1.0, 5.0),
               ("fraction_5_to_30", 5.0, 30.0),
               ("fraction_30_to_300", 30.0, 300.0),
               ("fraction_gt_300", 300.0, np.inf))


def yplus_band_fractions(values, weights=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("y+ values must be a non-empty finite non-negative vector")
    weights = np.ones(len(values), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("y+ weights must be finite positive values matching the faces")
    total = float(np.sum(weights))
    fractions = {}
    for name, lower, upper in YPLUS_BANDS:
        mask = (values < upper) if lower == 0 else ((values >= lower) & (values < upper))
        if name == "fraction_30_to_300":
            mask = (values >= lower) & (values <= upper)
        fractions[name] = float(np.sum(weights[mask]) / total)
    return fractions


def component_yplus_bands(wall_meshes, zone_groups, walls_area_path=None,
                          walls_path=None):
    names_by_component = {
        "wing": list(zone_groups["wing"]),
        "fuselage": list(zone_groups["fuselage"]),
        "htail": list(zone_groups["htail"]),
        "vtail": list(zone_groups["vtail"]),
        "nacelle": [name for name in zone_groups["nacelle"] if name.startswith("engine")],
        "pylon": [name for name in zone_groups["nacelle"] if name.startswith("pylon")],
        "duct": list(zone_groups["duct"]),
        "fairing": list(zone_groups["fairing"]),
    }
    face_values = {name: [] for name in zone_groups["total"]}
    face_weights = {name: [] for name in zone_groups["total"]}
    area_columns = {"zone", "wall-yplus", "area-vector-x",
                    "area-vector-y", "area-vector-z"}
    area_path = None
    if walls_path is not None and Path(walls_path).is_file():
        with Path(walls_path).open("r", newline="", encoding="utf-8-sig") as stream:
            if area_columns <= set(csv.DictReader(stream).fieldnames or ()):
                area_path = Path(walls_path)
    if area_path is None and walls_area_path is not None and Path(walls_area_path).is_file():
        area_path = Path(walls_area_path)
    if area_path is not None:
        with area_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = area_columns - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{area_path}: missing columns for area-weighted y+: "
                                 + ", ".join(sorted(missing)))
            for row in reader:
                zone = row["zone"]
                if zone not in face_values:
                    raise ValueError(f"{area_path}: unexpected wall zone {zone!r}")
                vector = np.asarray([float(row[key]) for key in
                                     ("area-vector-x", "area-vector-y", "area-vector-z")])
                yplus = float(row["wall-yplus"])
                area = float(np.linalg.norm(vector))
                if not np.isfinite([*vector, yplus, area]).all() or yplus < 0 or area <= 0:
                    raise ValueError(f"{area_path}: invalid wall area or y+ in zone {zone}")
                face_values[zone].append(yplus)
                face_weights[zone].append(area)
        for zone in zone_groups["total"]:
            expected = wall_meshes[zone].n_cells
            if len(face_values[zone]) != expected:
                raise ValueError(f"{area_path}: zone {zone} has {len(face_values[zone])} "
                                 f"faces; mesh has {expected}")
        weighting = "area"
        flag = ""
    else:
        for zone in zone_groups["total"]:
            mesh = wall_meshes[zone]
            values = np.asarray(mesh.cell_data.get("yplus", []), dtype=float)
            if values.shape != (mesh.n_cells,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{zone}: missing or invalid y+ values for face-count bands")
            face_values[zone].extend(values.tolist())
        weighting = "face_count"
        flag = ("area-vector columns unavailable in walls.csv and walls_area.csv; "
                "component y+ bands are count-weighted")

    output = {}
    for component, zones in names_by_component.items():
        values = np.asarray([value for zone in zones for value in face_values[zone]], dtype=float)
        if not len(values):
            continue
        weights = (np.asarray([weight for zone in zones for weight in face_weights[zone]], dtype=float)
                   if weighting == "area" else None)
        output[component] = {"face_count": int(len(values)), "weighting": weighting,
                             **yplus_band_fractions(values, weights)}
    return {"components": output, "weighting": weighting, "flag": flag}


def write_yplus_products(wing, output, aircraft_walls=None, walls_area_path=None,
                         zone_groups=None, walls_path=None):
    output = Path(output)
    mesh = wing["mesh"]
    values = np.asarray(mesh.cell_data["yplus"], dtype=float)
    areas = np.asarray(mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"],
                       dtype=float)
    good = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
    values, areas = values[good], areas[good]
    if not len(values):
        raise ValueError("mainwing y+ map has no finite faces")
    order = np.argsort(values)
    cumulative = np.cumsum(areas[order]) / np.sum(areas)
    def quantile(fraction):
        index = min(np.searchsorted(cumulative, fraction), len(order) - 1)
        return float(values[order[index]])

    stats = {"median": quantile(0.5), "p95": quantile(0.95), "max": float(np.max(values)),
             "fraction_gt_1": float(np.sum(areas[values > 1]) / np.sum(areas)),
             "fraction_gt_5": float(np.sum(areas[values > 5]) / np.sum(areas))}
    wall_meshes = aircraft_walls or {"mainwing": mesh}
    if zone_groups is None:
        prefixes = {"wing": ("mainwing",), "fuselage": ("fuselage",),
                    "htail": ("horstab",), "vtail": ("vertstab",),
                    "nacelle": ("engine", "pylon"), "duct": ("nacelle_duct",),
                    "fairing": ("fairing",)}
        zone_groups = {group: [name for name in wall_meshes for start in starts
                               if name.startswith(start)]
                       for group, starts in prefixes.items()}
        zone_groups["total"] = list(wall_meshes)
    component_stats = component_yplus_bands(
        wall_meshes, zone_groups, walls_area_path, walls_path=walls_path)
    stats["component_yplus_bands"] = component_stats["components"]
    stats["component_band_weighting"] = component_stats["weighting"]
    stats["component_band_weighting_flag"] = component_stats["flag"]
    (output / "yplus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    wall_meshes = list(wall_meshes.values())
    full = wall_meshes[0].copy(deep=True)
    for wall in wall_meshes[1:]:
        full = full.merge(wall, merge_points=False)
    full = full.merge(full.reflect((0, 1, 0), point=(0, 0, 0)), merge_points=False)
    centers = full.cell_centers().points
    figure, axis = plt.subplots(figsize=(12, 7))
    plot_values = np.asarray(full.cell_data["yplus"], dtype=float)
    positive = plot_values[plot_values > 0]
    vmin = max(float(np.min(positive)) if len(positive) else 1e-3, 1e-3)
    vmax = max(float(np.max(plot_values)), vmin * 10.0)
    plot = axis.scatter(centers[:, 0], centers[:, 1], c=plot_values,
                        s=2, cmap="turbo", norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                        rasterized=True)
    figure.colorbar(plot, ax=axis, label="wall y+ (log scale)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Mirrored aircraft wall y+")
    figure.tight_layout()
    figure.savefig(output / "yplus.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    wing_values = np.asarray(values, dtype=float)
    wing_areas = np.asarray(areas, dtype=float)
    wing_positive = wing_values > 0
    axis.hist(wing_values[wing_positive] if np.any(wing_positive) else wing_values,
              bins=np.logspace(np.log10(vmin), np.log10(vmax), 41),
              weights=(wing_areas[wing_positive] if np.any(wing_positive) else wing_areas),
              color="#3b82c4")
    axis.set_xscale("log")
    axis.set_xlabel("wall y+ (log scale)")
    axis.set_ylabel("area (m²)")
    axis.set_title("Mainwing y+ area distribution")
    figure.tight_layout()
    figure.savefig(output / "yplus_hist.png", dpi=150)
    plt.close(figure)
    return stats


def write_convergence(history_path, output):
    with Path(history_path).open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{history_path}: history.csv contains no iterations")
    x = np.asarray([float(row["iter"]) for row in rows])
    alpha = np.asarray([float(row["alpha"]) for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for axis, key in zip(axes, ("CL", "CD", "CM")):
        values = np.asarray([float(row[key]) for row in rows])
        axis.plot(x, values, color="#2563eb", linewidth=1.2)
        axis.set_ylabel(key)
        axis.grid(True, alpha=0.25)
        for index in range(1, len(alpha)):
            if alpha[index] != alpha[index - 1]:
                axis.axvline(x[index], color="#dc2626", linestyle="--", alpha=0.55)
                axis.text(x[index], axis.get_ylim()[1], f" α={alpha[index]:g}°", fontsize=7,
                          va="top", color="#991b1b")
    axes[-1].set_xlabel("iteration")
    figure.tight_layout()
    figure.savefig(Path(output) / "convergence.png", dpi=150)
    plt.close(figure)
