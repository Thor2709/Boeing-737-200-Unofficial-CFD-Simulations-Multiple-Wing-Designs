"""Integrate saved Fluent wall fields into per-zone and nacelle force reports.

This utility reads a completed run's final case and data. It does not set up a
solver or perform iterations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from b737wing.cfd.solve.conditions import S_REF_M2, condition_for_case
from b737wing.cfd.solve.fluent_io import (FluentBackend, launch_fluent, polygon_area_vectors,
                                          resolve_zone_groups)


FORCE_COLUMNS = (
    "scope", "component", "zone", "region", "face_count",
    "CL_pressure", "CD_pressure", "CL_viscous", "CD_viscous",
    "CL_total", "CD_total", "solver_CL", "solver_CD", "delta_CL", "delta_CD")
WALL_AREA_COLUMNS = (
    "zone", "component", "x", "y", "z", "area-vector-x", "area-vector-y",
    "area-vector-z", "pressure-coefficient", "wall-yplus", "x-wall-shear",
    "y-wall-shear", "z-wall-shear")


def _field_request_types():
    from ansys.fluent.core.fields.field_data_interfaces import (
        ScalarFieldDataRequest, SurfaceFieldDataRequest, SurfaceDataType)
    return ScalarFieldDataRequest, SurfaceFieldDataRequest, SurfaceDataType


def _component_by_zone(groups):
    result = {}
    for component in ("wing", "fuselage", "htail", "vtail", "duct", "fairing"):
        result.update({zone: component for zone in groups[component]})
    result.update({zone: ("pylon" if zone.startswith("pylon") else "nacelle")
                   for zone in groups["nacelle"]})
    return result


def _load_face_fields(session, zones, ScalarFieldDataRequest, SurfaceFieldDataRequest,
                      SurfaceDataType):
    geometry = session.fields.field_data.get_field_data(
        SurfaceFieldDataRequest(surfaces=zones,
                                data_types=[SurfaceDataType.FacesNormal,
                                            SurfaceDataType.FacesCentroid,
                                            SurfaceDataType.Vertices,
                                            SurfaceDataType.FacesConnectivity]))
    scalar_names = ("pressure-coefficient", "y-plus", "x-wall-shear",
                    "y-wall-shear", "z-wall-shear")
    scalar_data = {}
    for field in scalar_names:
        response = session.fields.field_data.get_field_data(
            ScalarFieldDataRequest(field_name=field, surfaces=zones, node_value=False))
        scalar_data[field] = {
            zone: np.asarray(response[zone], dtype=float).reshape(-1) for zone in zones}
    output = {}
    for zone in zones:
        centroids = np.asarray(FluentBackend._field_item(geometry[zone], "centroid"), dtype=float)
        fluent_normals = np.asarray(FluentBackend._field_item(geometry[zone], "face-normal"),
                                    dtype=float)
        area_vectors = polygon_area_vectors(geometry[zone], fluent_normals)
        if centroids.ndim != 2 or centroids.shape[1] != 3 or area_vectors.shape != centroids.shape:
            raise ValueError(f"{zone}: Fluent face centroid/area-vector arrays must be Nx3")
        fields = {name: scalar_data[name][zone] for name in scalar_names}
        if any(values.shape != (len(centroids),) for values in fields.values()):
            raise ValueError(f"{zone}: Fluent wall fields have different face counts")
        if (not np.isfinite(centroids).all() or not np.isfinite(area_vectors).all()
                or any(not np.isfinite(values).all() for values in fields.values())):
            raise ValueError(f"{zone}: Fluent wall fields contain non-finite values")
        if np.any(np.linalg.norm(area_vectors, axis=1) <= 0) or np.any(fields["y-plus"] < 0):
            raise ValueError(f"{zone}: Fluent returned a zero-area face or negative y+")
        output[zone] = {"centroids": centroids, "area_vectors": area_vectors, **fields}
    return output


def _force_coefficients(pressure_force, viscous_force, alpha_deg, q_half_area):
    alpha = np.radians(float(alpha_deg))
    drag_axis = np.asarray([np.cos(alpha), 0.0, np.sin(alpha)])
    lift_axis = np.asarray([-np.sin(alpha), 0.0, np.cos(alpha)])
    return {
        "CL_pressure": float(np.dot(pressure_force, lift_axis) / q_half_area),
        "CD_pressure": float(np.dot(pressure_force, drag_axis) / q_half_area),
        "CL_viscous": float(np.dot(viscous_force, lift_axis) / q_half_area),
        "CD_viscous": float(np.dot(viscous_force, drag_axis) / q_half_area),
    }


def _integrate_faces(face_fields, component_by_zone, condition, alpha_deg):
    q = 0.5 * condition["density_kg_m3"] * condition["velocity_m_s"] ** 2
    q_half_area = q * (S_REF_M2 / 2.0)
    faces = {}
    zone_sums = {}
    for zone, data in face_fields.items():
        area_vectors = data["area_vectors"]
        areas = np.linalg.norm(area_vectors, axis=1)
        # Fluent face normals point out of the fluid control volume, so Cp times
        # the area vector is the pressure force exerted by the fluid on the wall.
        pressure_forces = (data["pressure-coefficient"] * q)[:, None] * area_vectors
        shear = np.column_stack([data[name] for name in
                                 ("x-wall-shear", "y-wall-shear", "z-wall-shear")])
        viscous_forces = shear * areas[:, None]
        coefficients = _force_coefficients(np.sum(pressure_forces, axis=0),
                                           np.sum(viscous_forces, axis=0),
                                           alpha_deg, q_half_area)
        zone_sums[zone] = {**coefficients, "face_count": len(areas),
                           "component": component_by_zone[zone]}
        faces[zone] = {"areas": areas, "pressure_forces": pressure_forces,
                       "viscous_forces": viscous_forces}
    return faces, zone_sums


def _sum_coefficients(rows):
    keys = ("CL_pressure", "CD_pressure", "CL_viscous", "CD_viscous")
    result = {key: sum(float(row[key]) for row in rows) for key in keys}
    result["CL_total"] = result["CL_pressure"] + result["CL_viscous"]
    result["CD_total"] = result["CD_pressure"] + result["CD_viscous"]
    return result


def _nacelle_regions(face_fields, faces, component_by_zone):
    engine_points = [data["centroids"][:, 0] for zone, data in face_fields.items()
                     if component_by_zone[zone] == "nacelle"]
    all_x = np.concatenate(engine_points) if engine_points else np.asarray([])
    if not len(all_x) or not np.isfinite(all_x).all() or float(np.ptp(all_x)) <= 0:
        raise ValueError("cannot split nacelle drag: engine wall zones have no finite x length")
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    length = x_max - x_min
    regions = {name: [] for name in ("inlet_face_lip", "external_barrel", "base_exit", "pylon")}
    for zone, data in face_fields.items():
        component = component_by_zone[zone]
        if component not in ("nacelle", "pylon"):
            continue
        for index, x in enumerate(data["centroids"][:, 0]):
            if component == "pylon":
                region = "pylon"
            elif x <= x_min + 0.10 * length:
                region = "inlet_face_lip"
            elif x >= x_max - 0.10 * length:
                region = "base_exit"
            else:
                region = "external_barrel"
            regions[region].append((zone, index))
    return regions, faces


def _write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def process_run(run_dir):
    run = Path(run_dir).expanduser().resolve()
    result_path = run / "result.json"
    case_path = run / "final.cas.h5"
    data_path = run / "final.dat.h5"
    missing = [str(path) for path in (result_path, case_path, data_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("run is missing required final files: " + ", ".join(missing))
    with result_path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    case = str(result.get("case", "")).upper()
    if not case or result.get("alpha_deg") is None:
        raise ValueError(f"{result_path}: case and alpha_deg are required")
    condition = condition_for_case(case)
    session = launch_fluent(4, str(run))
    try:
        session.settings.file.read_case(file_name=str(case_path))
        session.settings.file.read_data(file_name=str(data_path))
        boundaries = session.settings.setup.boundary_conditions
        zones = list(boundaries.wall.keys())
        zones.extend(name for name in getattr(boundaries, "pressure_far_field", {}).keys()
                     if name not in zones)
        groups = resolve_zone_groups(zones)
        component_by_zone = _component_by_zone(groups)
        request_types = _field_request_types()
        face_fields = _load_face_fields(
            session, groups["total"], *request_types)
        faces, zone_sums = _integrate_faces(
            face_fields, component_by_zone, condition, float(result["alpha_deg"]))

        force_rows = []
        for zone in groups["total"]:
            coeffs = zone_sums[zone]
            force_rows.append({"scope": "zone", "component": coeffs["component"],
                               "zone": zone, "region": "", "face_count": coeffs["face_count"],
                               **coeffs,
                               "CL_total": coeffs["CL_pressure"] + coeffs["CL_viscous"],
                               "CD_total": coeffs["CD_pressure"] + coeffs["CD_viscous"]})
        totals = _sum_coefficients(list(zone_sums.values()))
        solver_cl, solver_cd = result.get("CL"), result.get("CD")
        force_rows.append({"scope": "total", "component": "aircraft", "zone": "",
                           "region": "", "face_count": sum(row["face_count"]
                           for row in zone_sums.values()), **totals,
                           "solver_CL": solver_cl, "solver_CD": solver_cd,
                           "delta_CL": totals["CL_total"] - float(solver_cl)
                           if solver_cl is not None else None,
                           "delta_CD": totals["CD_total"] - float(solver_cd)
                           if solver_cd is not None else None})
        regions, _ = _nacelle_regions(face_fields, faces, component_by_zone)
        for region, members in regions.items():
            pressure_force = np.zeros(3)
            viscous_force = np.zeros(3)
            for zone, index in members:
                pressure_force += faces[zone]["pressure_forces"][index]
                viscous_force += faces[zone]["viscous_forces"][index]
            coefficients = _force_coefficients(
                pressure_force, viscous_force, float(result["alpha_deg"]),
                0.5 * condition["density_kg_m3"] * condition["velocity_m_s"] ** 2
                * (S_REF_M2 / 2.0))
            force_rows.append({"scope": "region", "component": "nacelle_installation",
                               "zone": "", "region": region, "face_count": len(members),
                               **coefficients,
                               "CL_total": coefficients["CL_pressure"] + coefficients["CL_viscous"],
                               "CD_total": coefficients["CD_pressure"] + coefficients["CD_viscous"]})

        wall_area_rows = []
        for zone in groups["total"]:
            data = face_fields[zone]
            for index, centroid in enumerate(data["centroids"]):
                wall_area_rows.append({
                    "zone": zone, "component": component_by_zone[zone],
                    "x": centroid[0], "y": centroid[1], "z": centroid[2],
                    "area-vector-x": data["area_vectors"][index, 0],
                    "area-vector-y": data["area_vectors"][index, 1],
                    "area-vector-z": data["area_vectors"][index, 2],
                    "pressure-coefficient": data["pressure-coefficient"][index],
                    "wall-yplus": data["y-plus"][index],
                    "x-wall-shear": data["x-wall-shear"][index],
                    "y-wall-shear": data["y-wall-shear"][index],
                    "z-wall-shear": data["z-wall-shear"][index]})
    finally:
        session.exit()

    force_path = run / "force_breakdown.csv"
    area_path = run / "walls_area.csv"
    _write_csv(force_path, FORCE_COLUMNS, force_rows)
    _write_csv(area_path, WALL_AREA_COLUMNS, wall_area_rows)
    return {"force_breakdown": force_path, "walls_area": area_path,
            "zone_count": len(groups["total"]), "face_count": len(wall_area_rows)}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="completed Fluent run folder with final case and data")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    outputs = process_run(args.run_dir)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value
                      for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
