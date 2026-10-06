"""Wall-plane cuts and sectional pressure integrations.

The polygon-plane intersection follows ``pipeline/legacy_v1/facedata.py``'s
``wall_cut`` method; the Cp split and chord-normalised integration follow
``pipeline/legacy_v1/analyse.py``'s ``section`` method, with data passed in.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CENTROID_TOLERANCE_M = 0.001
FACE_SIZE_RELATIVE_TOLERANCE = 0.75
SECTION_ETAS = (0.15, 0.30, 0.50, 0.70, 0.90)
SPANWISE_ETAS = tuple(np.linspace(0.15, 0.95, 17))
SEMISPAN_M = 14.17
MAC_M = 3.4716


def _resolve_duplicate_matches(source, target, distances, indices, limits, k=8):
    """Give each target a distinct source face: on a shared nearest source the closest
    target keeps it and the others take their nearest unused source within limits.
    Unresolvable conflicts stay duplicated so the caller fails closed."""
    distances, indices = distances.copy(), indices.copy()
    unique, counts = np.unique(indices, return_counts=True)
    shared = unique[counts > 1]
    if not len(shared):
        return distances, indices
    used = set(indices.tolist())
    tree = cKDTree(source)
    for source_index in shared:
        members = np.flatnonzero(indices == source_index)
        for target_index in members[np.argsort(distances[members])][1:]:
            near_d, near_i = tree.query(target[target_index], k=min(k, len(source)))
            for d, i in zip(np.atleast_1d(near_d), np.atleast_1d(near_i)):
                if int(i) not in used and d <= limits[target_index]:
                    used.add(int(i))
                    indices[target_index], distances[target_index] = int(i), float(d)
                    break
    return distances, indices


def nearest_centroid_map(source_xyz, target_xyz, values, tolerance_m=CENTROID_TOLERANCE_M,
                         face_sizes=None, relative_tolerance=FACE_SIZE_RELATIVE_TOLERANCE):
    """Map source face values to target mesh centers; reject geometry drift.

    With face_sizes (sqrt of face area per target face) the gate is relative: Fluent's
    area-weighted polygon centroid and VTK's vertex-average centre differ by ~3 % of the
    face size on the real meshes (live check 2026-10-04), so a fixed 1 mm gate is wrong there.
    """
    source = np.asarray(source_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    values = np.asarray(values)
    if source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("centroids must be Nx3 arrays")
    if len(source) != len(values) or not len(source) or not len(target):
        raise ValueError("centroid mapping requires non-empty, equally sized source values")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("centroids must be finite")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    distances, indices = cKDTree(source).query(target, k=1)
    if face_sizes is None:
        limits = np.full(len(target), float(tolerance_m))
        tolerance_description = f"{float(tolerance_m):g} m"
    else:
        if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be finite and positive")
        sizes = np.asarray(face_sizes, dtype=float)
        if sizes.shape != (len(target),) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("face_sizes must contain one finite positive value per target face")
        limits = sizes * float(relative_tolerance)
        tolerance_description = f"{float(relative_tolerance):g} times each target face size"
    distances, indices = _resolve_duplicate_matches(source, target, distances, indices, limits)
    distant = distances > limits
    _, counts = np.unique(indices, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    if np.any(distant) or duplicate_count:
        rejected = int(np.sum(distant)) + duplicate_count
        max_rejected_distance = float(np.max(distances[distant])) if np.any(distant) else 0.0
        raise ValueError(
            f"median wall-centroid match distance {float(np.median(distances)):.6g} m "
            f"(diagnostic); per-face validation rejected {rejected} target face(s): "
            f"{int(np.sum(distant))} exceed per-face tolerance {tolerance_description}; "
            f"{duplicate_count} duplicate source mapping(s); "
            f"max rejected distance {max_rejected_distance:.6g} m")
    if face_sizes is not None:
        return values[indices], distances
    return values[indices], distances


def polygon_normal(points):
    """Area-weighted normal for a planar polygon."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(3, dtype=float)
    return 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)


def wall_cut(polygons, y0, cp=None, yplus=None):
    """Cut polygon faces by ``y=y0`` and return one record per crossed face."""
    cp = np.zeros(len(polygons), dtype=float) if cp is None else np.asarray(cp, dtype=float)
    yplus = np.full(len(polygons), np.nan) if yplus is None else np.asarray(yplus, dtype=float)
    if len(cp) != len(polygons) or len(yplus) != len(polygons):
        raise ValueError("wall fields must contain one value per polygon")
    rows = []
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=float)
        delta = points[:, 1] - float(y0)
        if delta.min() > 1e-12 or delta.max() < -1e-12:
            continue
        intersections = []
        for edge in range(len(points)):
            a, b = points[edge], points[(edge + 1) % len(points)]
            da, db = delta[edge], delta[(edge + 1) % len(points)]
            if abs(da) <= 1e-12:
                intersections.append(a)
            if (da < -1e-12 and db > 1e-12) or (da > 1e-12 and db < -1e-12):
                intersections.append(a + (da / (da - db)) * (b - a))
        unique = []
        for point in intersections:
            if not any(np.linalg.norm(point - old) <= 1e-10 for old in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        for p1, p2 in zip(unique[::2], unique[1::2]):
            tangent = p2[[0, 2]] - p1[[0, 2]]
            length = float(np.linalg.norm(tangent))
            if length <= 1e-12:
                continue
            normal = polygon_normal(points)
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 0 else np.zeros(3)
            rows.append({"p1": p1[[0, 2]], "p2": p2[[0, 2]],
                         "x": float((p1[0] + p2[0]) * 0.5),
                         "z": float((p1[2] + p2[2]) * 0.5),
                         "length": length,
                         "nx": float(normal[0]), "nz": float(normal[2]),
                         "cp": float(cp[index]), "yplus": float(yplus[index])})
    return rows


def _collapse_x(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    unique, starts = np.unique(x, return_index=True)
    if len(unique) == len(x):
        return unique, y
    ends = np.r_[starts[1:], len(x)]
    return unique, np.asarray([y[a:b].mean() for a, b in zip(starts, ends)])


def _ordered_cut_loops(segments):
    endpoints = [np.asarray(point, dtype=float) for row in segments
                 for point in (row["p1"], row["p2"])]
    scale = max(1.0, float(np.ptp(np.asarray(endpoints), axis=0).max()))
    tolerance = 1e-9 * scale
    nodes = []
    endpoint_ids = []
    for point in endpoints:
        for index, existing in enumerate(nodes):
            if np.linalg.norm(point - existing) <= tolerance:
                endpoint_ids.append(index)
                break
        else:
            endpoint_ids.append(len(nodes))
            nodes.append(point)
    edges = [(endpoint_ids[2 * index], endpoint_ids[2 * index + 1], row)
             for index, row in enumerate(segments)]
    adjacent = {index: [] for index in range(len(nodes))}
    for index, (start, end, _) in enumerate(edges):
        adjacent[start].append(index)
        adjacent[end].append(index)

    loops = []
    unseen = set(range(len(edges)))
    while unseen:
        first_edge = next(iter(unseen))
        component_edges = set()
        component_nodes = set()
        pending = [edges[first_edge][0]]
        while pending:
            node = pending.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            for edge_index in adjacent[node]:
                component_edges.add(edge_index)
                start, end, _ = edges[edge_index]
                pending.append(end if node == start else start)
        if any(len(adjacent[node]) != 2 for node in component_nodes):
            raise ValueError("section cut segments do not form closed loops")
        start_node = min(component_nodes, key=lambda index: (nodes[index][0], nodes[index][1]))
        ordered = []
        used = set()
        current = start_node
        while True:
            choices = [edge_index for edge_index in adjacent[current] if edge_index not in used]
            if not choices:
                break
            edge_index = choices[0]
            edge_start, edge_end, row = edges[edge_index]
            following = edge_end if current == edge_start else edge_start
            ordered.append((row, current, following))
            used.add(edge_index)
            current = following
            if current == start_node:
                break
        if current != start_node or used != component_edges:
            raise ValueError("section cut segments do not form closed loops")
        loops.append((ordered, nodes))
        unseen -= component_edges
    return loops


def _branch_edges(start, end, step, count):
    indices = []
    current = start
    while current != end:
        edge = current if step > 0 else (current - 1) % count
        indices.append(edge)
        current = (current + step) % count
    return indices


def _section_branches(ordered, nodes):
    node_path = [ordered[0][1]] + [edge[2] for edge in ordered]
    node_path = node_path[:-1]
    node_points = np.asarray([nodes[index] for index in node_path])
    x_values = node_points[:, 0]
    scale = max(1.0, float(np.ptp(node_points, axis=0).max()))
    tolerance = 1e-9 * scale
    le_x, te_x = float(x_values.min()), float(x_values.max())
    le_indices = np.flatnonzero(np.abs(x_values - le_x) <= tolerance).tolist()
    te_indices = np.flatnonzero(np.abs(x_values - te_x) <= tolerance).tolist()
    if len(le_indices) != 1 or len(te_indices) not in (1, 2):
        raise ValueError("section cut has ambiguous min-x LE or max-x TE points")
    le_index = le_indices[0]
    count = len(ordered)
    if len(te_indices) == 1:
        te_index = te_indices[0]
        branch_indices = (_branch_edges(le_index, te_index, 1, count),
                          _branch_edges(le_index, te_index, -1, count))
        chord_end = node_points[te_index]
    else:
        candidates = []
        for first_te, second_te in (te_indices, te_indices[::-1]):
            first = _branch_edges(le_index, first_te, 1, count)
            second = _branch_edges(le_index, second_te, -1, count)
            if not set(first).intersection(second):
                candidates.append((first, second))
        branch_indices = max(candidates, key=lambda pair: len(pair[0]) + len(pair[1])) if candidates else None
        if branch_indices is None:
            raise ValueError("section cut cannot split at its max-x TE points")
        chord_end = (node_points[te_indices[0]] + node_points[te_indices[1]]) * 0.5
    le_point = node_points[le_index]
    chord_vector = chord_end - le_point
    chord_length = float(np.linalg.norm(chord_vector))
    if chord_length <= tolerance:
        raise ValueError("section cut has zero LE-TE chord")
    means = []
    branches = []
    for indices in branch_indices:
        rows = [ordered[index][0] for index in indices]
        relative_height = [chord_vector[0] * (row["z"] - le_point[1])
                           - chord_vector[1] * (row["x"] - le_point[0])
                           for row in rows]
        means.append(float(np.mean(relative_height)))
        branches.append(rows)
    upper_index = int(np.argmax(means))
    return branches[upper_index], branches[1 - upper_index]


def section_from_cut(segments, y, x_le, chord, alpha_deg=0.0):
    """Split a cut into upper/lower Cp and integrate sectional ``cl``/``cdp``."""
    if not segments:
        return None
    if not np.isfinite(chord) or chord <= 0:
        raise ValueError("section chord must be positive and finite")
    loops = _ordered_cut_loops(segments)
    if len(loops) != 1:
        raise ValueError("mainwing cut contains multiple closed loops (NEEDS_OPUS_DECISION)")
    upper_rows, lower_rows = _section_branches(*loops[0])
    x_upper, cp_upper = _collapse_x([row["x"] for row in upper_rows],
                                    [row["cp"] for row in upper_rows])
    x_lower, cp_lower = _collapse_x([row["x"] for row in lower_rows],
                                    [row["cp"] for row in lower_rows])
    if len(x_upper) < 2 or len(x_lower) < 2:
        raise ValueError("section cut does not contain at least two upper and lower Cp points")
    upper_edges = np.asarray([point[0] for row in upper_rows for point in (row["p1"], row["p2"])])
    lower_edges = np.asarray([point[0] for row in lower_rows for point in (row["p1"], row["p2"])])
    x0 = max(float(upper_edges.min()), float(lower_edges.min()))
    x1 = min(float(upper_edges.max()), float(lower_edges.max()))
    if x1 <= x0:
        raise ValueError("upper and lower section cuts have no common chordwise extent")
    # Integrate pressure over the ordered closed contour, including a blunt TE cap.
    ordered, nodes = loops[0]
    node_path = [ordered[0][1]] + [edge[2] for edge in ordered]
    contour = np.asarray([nodes[index] for index in node_path], dtype=float)
    signed_area = 0.5 * float(np.sum(
        contour[:-1, 0] * contour[1:, 1] - contour[1:, 0] * contour[:-1, 1]))
    if abs(signed_area) <= 1e-15:
        raise ValueError("section contour has zero signed area")
    orientation = 1.0 if signed_area > 0 else -1.0
    force_x = force_z = 0.0
    for row, start, end in ordered:
        p0, p1 = np.asarray(nodes[start]), np.asarray(nodes[end])
        dx, dz = (p1 - p0).tolist()
        # For a CCW contour the right-hand edge normal points out of the section.
        area_normal_x, area_normal_z = orientation * dz, -orientation * dx
        force_x -= float(row["cp"]) * area_normal_x
        force_z -= float(row["cp"]) * area_normal_z
    cx, cz = force_x / chord, force_z / chord
    alpha = np.radians(float(alpha_deg))
    cl = -np.sin(alpha) * cx + np.cos(alpha) * cz
    cdp = np.cos(alpha) * cx + np.sin(alpha) * cz
    xc_upper = (x_upper - float(x_le)) / chord
    xc_lower = (x_lower - float(x_le)) / chord
    return {"y_m": float(y), "eta": None, "chord_m": float(chord),
            "x_le_m": float(x_le), "cl": float(cl), "cdp": float(cdp),
            "upper": {"x_c": xc_upper, "cp": cp_upper},
            "lower": {"x_c": xc_lower, "cp": cp_lower}}


def polygons(mesh):
    """Return the wall polygons from a PyVista PolyData face array."""
    out = []
    faces = np.asarray(mesh.faces, dtype=np.int64)
    offset = 0
    while offset < len(faces):
        count = int(faces[offset])
        indices = faces[offset + 1:offset + count + 1]
        out.append(np.asarray(mesh.points[indices], dtype=float))
        offset += count + 1
    return out


def cut_zone(zone_data, y_m):
    mesh = zone_data["mesh"]
    return wall_cut(polygons(mesh), y_m, mesh.cell_data["Cp"], mesh.cell_data.get("yplus"))


def section_at(wing, y_m, alpha, eta=None):
    segments = cut_zone(wing, y_m)
    if not segments:
        raise ValueError(f"mainwing has no faces at section y={y_m:g} m")
    xs = [float(point[0]) for row in segments for point in (row["p1"], row["p2"])]
    x_le, x_te = float(min(xs)), float(max(xs))
    section = section_from_cut(segments, y_m, x_le, x_te - x_le, alpha)
    section["eta"] = float(eta if eta is not None else y_m / SEMISPAN_M)
    section["segments"] = segments
    return section


def write_cp_sections(wing, fuselage, alpha, output, semispan_m=SEMISPAN_M):
    section_rows, plotted = [], []
    for eta in SECTION_ETAS:
        y_m = eta * semispan_m
        section = section_at(wing, y_m, alpha, eta)
        plotted.append(section)
        for surface in ("upper", "lower"):
            for x_c, cp in zip(section[surface]["x_c"], section[surface]["cp"]):
                section_rows.append({"section": f"eta_{eta:.2f}", "eta": eta, "y_m": y_m,
                                     "surface": surface, "x_c": float(x_c), "Cp": float(cp),
                                     "cl": section["cl"], "cdp": section["cdp"]})
    if fuselage is None:
        raise ValueError("required fuselage wall group is unavailable for centreline cut at y=0.02 m")
    fuselage_cut = cut_zone(fuselage, 0.02)
    if not fuselage_cut:
        raise ValueError("required fuselage centreline cut at y=0.02 m is empty")
    low = min(row["x"] for row in fuselage_cut)
    high = max(row["x"] for row in fuselage_cut)
    length = max(high - low, 1e-12)
    for row in fuselage_cut:
        section_rows.append({"section": "fuselage_y_0.02m", "eta": "", "y_m": 0.02,
                             "surface": "fuselage", "x_c": (row["x"] - low) / length,
                             "Cp": row["cp"], "cl": "", "cdp": ""})
    plotted.append({"eta": None, "fuselage": [((row["x"] - low) / length, row["cp"])
                                                for row in fuselage_cut]})
    columns = ("section", "eta", "y_m", "surface", "x_c", "Cp", "cl", "cdp")
    with (Path(output) / "cp_sections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(section_rows)
    figure, axis = plt.subplots(figsize=(11, 7))
    colors = plt.cm.viridis(np.linspace(0.05, 0.92, len(SECTION_ETAS)))
    for color, section in zip(colors, plotted[:len(SECTION_ETAS)]):
        for surface in ("upper", "lower"):
            axis.plot(section[surface]["x_c"], section[surface]["cp"],
                      color=color, label=f"η={section['eta']:.2f} {surface}")
    if plotted and plotted[-1].get("fuselage"):
        values = np.asarray(plotted[-1]["fuselage"], dtype=float)
        order = np.argsort(values[:, 0])
        axis.plot(values[order, 0], values[order, 1], "k.", label="fuselage y=0.02 m")
    axis.set_xlabel("x/c")
    axis.set_ylabel("Cp")
    axis.set_title("Pressure coefficient sections (−Cp upward)")
    axis.invert_yaxis()
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(output) / "cp_sections.png", dpi=150)
    plt.close(figure)
    return plotted[:len(SECTION_ETAS)]


def write_spanwise(wing, alpha, output, semispan_m=SEMISPAN_M, reference_chord_m=MAC_M):
    rows = []
    for eta in SPANWISE_ETAS:
        try:
            section = section_at(wing, float(eta * semispan_m), alpha, eta)
        except ValueError as error:
            # The pylon/wing junction leaves a hole in the mainwing surface (eta ~0.35 on
            # the 737 model, live check 2026-10-04); such a station is recorded, not plotted.
            if "closed loops" not in str(error):
                raise
            rows.append({"eta": eta, "y_m": float(eta * semispan_m), "chord_m": float("nan"),
                         "cl": float("nan"), "cdp": float("nan"),
                         "c_cl_over_c_ref": float("nan"), "note": "open cut (pylon junction)"})
            continue
        rows.append({"eta": eta, "y_m": section["y_m"], "chord_m": section["chord_m"],
                     "cl": section["cl"], "cdp": section["cdp"],
                     "c_cl_over_c_ref": section["chord_m"] * section["cl"] / reference_chord_m})
    columns = ("eta", "y_m", "chord_m", "cl", "cdp", "c_cl_over_c_ref", "note")
    if sum(1 for row in rows if row.get("note")) > 2:
        raise ValueError("more than two spanwise stations have open section cuts")
    with (Path(output) / "spanwise_loading.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    figure, left = plt.subplots(figsize=(10, 6))
    right = left.twinx()
    left.plot([row["eta"] for row in rows], [row["c_cl_over_c_ref"] for row in rows],
              "b-o", label="c·cl/MAC")
    right.plot([row["eta"] for row in rows], [row["cl"] for row in rows], "r-s", label="section cl")
    left.set_xlabel("η = y / semispan")
    left.set_ylabel("c·cl / MAC", color="b")
    right.set_ylabel("section cl", color="r")
    left.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(Path(output) / "spanwise_loading.png", dpi=150)
    plt.close(figure)
