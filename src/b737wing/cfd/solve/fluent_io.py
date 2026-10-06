"""Thin PyFluent adapter; importing this module does not import Fluent."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

from .conditions import reference_values

WALL_PREFIXES = {
    "wing": ("mainwing",), "fuselage": ("fuselage",),
    "htail": ("horstab",), "vtail": ("vertstab",),
    "nacelle": ("engine", "pylon"), "duct": ("nacelle_duct",),
    "fairing": ("fairing",),
}
REPORT_COMPONENTS = tuple(WALL_PREFIXES)
TURBULENT_INTENSITY = 0.001
TURBULENT_VISCOSITY_RATIO = 10.0
COURANT_RAMP = ((100, 2.0), (300, 5.0), (6000, 10.0))
PB_DISCRETIZATION = {
    "pressure": "second-order",
    "momentum": "second-order-upwind",
    "density": "second-order-upwind",
    "energy": "second-order-upwind",
    "k": "second-order-upwind",
    "omega": "second-order-upwind",
}
UNVERIFIED_API = (
    "PB_DISCRETIZATION keys for Fluent 26.1 pressure-based cases",
    "Far-field initialization TUI fallback argument sequence",
    "FMG initialization: /solve/initialize/fmg-initialization (no arguments)",
    "Solution steering: /solve/set/solution-steering transonic (argument sequence unverified)",
    "CASM: /solve/set/convergence-acceleration-for-stretched-meshes (argument sequence unverified)",
)


def _available_discretization_keys(discretization) -> str:
    setting = getattr(discretization, "discretization_scheme", None)
    for name in ("available_keys", "allowed_keys", "child_names", "keys"):
        value = getattr(setting, name, None)
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            try:
                return ", ".join(sorted(str(key) for key in value))
            except TypeError:
                pass
    for name in ("get_state", "to_dict"):
        getter = getattr(setting, name, None)
        try:
            state = getter() if callable(getter) else None
        except Exception:
            continue
        if isinstance(state, dict):
            return ", ".join(sorted(str(key) for key in state))
    return "(not exposed by setting)"


def resolve_zone_groups(zone_names: list[str]) -> dict[str, list[str]]:
    """Map actual (possibly suffixed) Fluent names to component wall groups."""
    names = list(dict.fromkeys(str(name) for name in zone_names))
    groups = {group: [name for name in names
                      if name.startswith(prefixes)]
              for group, prefixes in WALL_PREFIXES.items()}
    missing = [prefix for prefixes in WALL_PREFIXES.values() for prefix in prefixes
               if not any(name.startswith(prefix) for name in names)]
    farfield = [name for name in names if name.startswith("farfield")]
    if not farfield:
        missing.append("farfield")
    if missing:
        raise ValueError("missing expected zone group(s): " + ", ".join(missing)
                         + "; zones found: " + ", ".join(names))
    groups["total"] = [name for group in REPORT_COMPONENTS for name in groups[group]]
    groups["total_ex_duct"] = [name for name in groups["total"]
                               if not name.startswith(WALL_PREFIXES["duct"])]
    groups["farfield"] = farfield
    return groups


def parse_zone_names(output) -> list[str]:
    """Read zone names from Fluent's ``/mesh/modify-zones/list-zones`` reply."""
    if isinstance(output, dict):
        found = []
        for key, value in output.items():
            if str(key).lower() in {"name", "zone_name", "zone-name"}:
                found.extend(parse_zone_names(value))
            else:
                found.extend(parse_zone_names(value))
        return list(dict.fromkeys(found))
    if isinstance(output, (tuple, list, set)):
        return list(dict.fromkeys(name for value in output
                                 for name in parse_zone_names(value)))
    text = str(output or "")
    names = []
    for line in text.splitlines():
        row = re.match(r"^\s*(?:zone\s+)?\d+\s*[:|]?\s+([A-Za-z_][\w:.-]*)", line, re.I)
        if row:
            names.append(row.group(1))
    if not names:
        pattern = r"\b((?:mainwing|horstab|vertstab|fuselage|engine|pylon|nacelle_duct|fairing|farfield)[\w:.-]*)"
        names = re.findall(pattern, text, re.I)
    return list(dict.fromkeys(names))


def launch_fluent(processor_count: int, cwd: str, solver: str = "cpu-db",
                  precision: str | None = None):
    """Launch the requested Fluent 26.1 solver."""
    os.environ["AWP_ROOT261"] = r"C:\Program Files\ANSYS Inc\ANSYS Student\v261"
    import ansys.fluent.core as pyfluent
    gpu = solver == "gpu-pb"
    options = {"mode": "solver", "precision": precision or ("single" if gpu else "double"),
               "processor_count": 1 if gpu else processor_count,
               "product_version": "26.1.0", "cleanup_on_exit": True,
               "start_watchdog": False,
               "cwd": str(Path(cwd).expanduser().resolve())}
    if gpu:
        options["gpu"] = True
    return pyfluent.launch_fluent(**options)


def reset_fluent_pids(output_dir: str | Path):
    path = Path(output_dir) / "fluent_pids.json"
    path.write_text(json.dumps({"pids": [], "launches": []}, indent=2),
                    encoding="utf-8")
    return path


def record_fluent_pids(output_dir: str | Path, session):
    path = Path(output_dir) / "fluent_pids.json"
    connection = getattr(session, "connection_properties", None)
    process = getattr(session, "_process", None)
    values = {
        "cortex_pid": getattr(connection, "cortex_pid", None),
        "fluent_host_pid": getattr(connection, "fluent_host_pid", None),
        "launcher_pid": getattr(process, "pid", None),
    }
    pids = sorted({value for value in values.values()
                   if isinstance(value, int) and not isinstance(value, bool)
                   and value > 0})
    if not pids:
        raise RuntimeError("Fluent launch returned no process ids")
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
    else:
        saved = {"pids": [], "launches": []}
    saved["pids"] = sorted(set(saved.get("pids", [])) | set(pids))
    saved.setdefault("launches", []).append({**values, "pids": pids})
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return path


def write_interpolation_data(source_run_dir: str | Path, output_dir: str | Path,
                             processor_count: int) -> Path:
    """Write a Fluent interpolation file from a completed source run."""
    source_dir = Path(source_run_dir).expanduser().resolve()
    destination_dir = Path(output_dir).expanduser().resolve()
    case_path = source_dir / "final.cas.h5"
    data_path = source_dir / "final.dat.h5"
    interpolation_path = destination_dir / "warmstart.ip"
    missing = [str(path) for path in (case_path, data_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("source run is missing: " + ", ".join(missing))
    destination_dir.mkdir(parents=True, exist_ok=True)
    session = launch_fluent(processor_count, str(source_dir))
    try:
        record_fluent_pids(destination_dir, session)
        session.settings.file.read_case(file_name=str(case_path))
        session.settings.file.read_data(file_name=str(data_path))
        session.tui.file.interpolate.write_data(
            str(interpolation_path), "all",
            ["pressure", "velocity", "temperature", "k", "omega"])
        if not interpolation_path.is_file():
            raise RuntimeError("Fluent did not write interpolation file: "
                               + str(interpolation_path))
    finally:
        session.exit()
    return interpolation_path



def write_portable_case(encas: Path) -> Path:
    """Fluent writes cwd-relative file paths into the .encas; write final.case with bare names."""
    import re as _re
    text = Path(encas).read_text(encoding="utf-8", errors="replace")
    text = _re.sub(r'"([^"]+)"', lambda m: '"' + m.group(1).replace("\\", "/").split("/")[-1] + '"', text)
    target = Path(encas).with_suffix(".case")
    target.write_text(text, encoding="utf-8")
    return target


def _face_lists(connectivity):
    """Faces as vertex-index arrays from structured or flat (count-prefixed) connectivity."""
    import numpy as np
    if isinstance(connectivity, np.ndarray) and connectivity.dtype != object             and connectivity.ndim == 1:
        faces, index = [], 0
        while index < len(connectivity):
            count = int(connectivity[index])
            faces.append(np.asarray(connectivity[index + 1:index + 1 + count], dtype=int))
            index += 1 + count
        return faces
    return [np.asarray(face, dtype=int) for face in connectivity]


def polygon_area_vectors(item, fluent_normals):
    """Exact polygon area vectors (Newell) oriented like Fluent's face normals.

    Fluent 26.1 face_normals on polygonal wall faces are not area-weighted
    (|n| is about 0.34 x the polygon area on the C5 mesh), so only their
    direction is used.
    """
    import numpy as np
    vertices = np.asarray(getattr(item, "vertices", None) if hasattr(item, "vertices")
                          else FluentBackend._field_item(item, "vertices"),
                          dtype=float).reshape(-1, 3)
    connectivity = (item.connectivity if hasattr(item, "connectivity")
                    else FluentBackend._field_item(item, "connectivity"))
    faces = _face_lists(connectivity)
    if len(faces) != len(fluent_normals):
        raise ValueError(f"face connectivity count {len(faces)} differs from "
                         f"normal count {len(fluent_normals)}")
    vectors = np.zeros((len(faces), 3))
    for index, face in enumerate(faces):
        if len(face) < 3:
            raise ValueError("Fluent returned a face with fewer than 3 vertices")
        points = vertices[face]
        vectors[index] = 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    flip = np.einsum("ij,ij->i", vectors, fluent_normals) < 0
    vectors[flip] *= -1.0
    return vectors



class FluentBackend:
    """Repository's verified Fluent settings and field-data calls in one adapter."""

    def __init__(self, session):
        self.session = session
        self.settings = session.settings
        self.setup = self.settings.setup
        self.report_definitions = self.settings.solution.report_definitions
        self.groups = {}
        self.names = {}
        self._lift_names = []
        self._drag_names = []
        self.cells = None
        self._transcript_chunks = []
        self._transcript_baseline = set()
        self.solver_transcript_available = None
        self.native_gpu_solver_active = None
        transcript = getattr(session, "transcript", None)
        register_callback = getattr(transcript, "register_callback", None)
        if callable(register_callback):
            try:
                register_callback(self._record_transcript)
            except Exception:
                pass

    def _record_transcript(self, value):
        if isinstance(value, str):
            self._transcript_chunks.append(value)

    def set_transcript_baseline(self, existing_names):
        self._transcript_baseline = set(existing_names)

    def configure(self, mesh: str, condition: dict, init_data: str | None = None,
                  warm_start_file: str | None = None,
                  warm_start_source: str | None = None,
                  warm_start_error: str | None = None,
                  initial_courant: float = COURANT_RAMP[0][1]):
        self.warm_start_status = None
        self._warm_start_file = warm_start_file
        self._warm_start_source = (
            str(Path(warm_start_source).expanduser().resolve())
            if warm_start_source else None)
        self._warm_start_error = warm_start_error
        mesh_path = Path(mesh).expanduser().resolve()
        self.settings.file.read_mesh(file_name=str(mesh_path))
        # The TUI list-zones prints to the transcript and returns None (live run 2026-10-04),
        # so read the boundary-condition zone names from the settings API instead.
        bcs = self.setup.boundary_conditions
        zones = [name for kind in ("wall", "pressure_far_field", "symmetry")
                 for name in getattr(bcs, kind).keys()]
        self.groups = resolve_zone_groups(zones)
        solver = self.setup
        solver_name = getattr(self, "solver_name", "cpu-db")
        is_pressure_based = solver_name == "gpu-pb"
        solver.general.solver.type = ("pressure-based" if is_pressure_based
                                      else "density-based-implicit")
        if is_pressure_based:
            self.settings.solution.methods.p_v_coupling.flow_scheme = "Coupled"
        solver.models.energy.enabled = True
        solver.models.viscous.model = "k-omega"
        solver.models.viscous.k_omega_model = "sst"
        air = solver.materials.fluid["air"]
        air.density.option = "ideal-gas"
        air.viscosity.option = "sutherland"
        solver.general.operating_conditions.operating_pressure = 0
        self.pffs = [solver.boundary_conditions.pressure_far_field[name]
                     for name in self.groups["farfield"]]
        for pff in self.pffs:
            pff.momentum.gauge_pressure = condition["pressure_Pa"]
            pff.momentum.mach_number = condition["mach"]
            pff.thermal.temperature = condition["temperature_K"]
            pff.turbulence.turbulent_intensity = TURBULENT_INTENSITY
            pff.turbulence.turbulent_viscosity_ratio = TURBULENT_VISCOSITY_RATIO
        refs = reference_values(condition)
        rf = solver.reference_values
        rf.area = refs["area_m2"]
        rf.length = refs["length_m"]
        rf.density = refs["density_kg_m3"]
        rf.pressure = refs["pressure_Pa"]
        rf.temperature = refs["temperature_K"]
        rf.velocity = refs["velocity_m_s"]
        self._create_reports(refs)
        residual = self.settings.solution.monitor.residual
        # criterion "none" disables the residual stop; per-equation check flags are then
        # inactive objects and must not be set (live run 2026-10-04).
        residual.options.criterion_type = "none"
        discretization = self.settings.solution.methods.spatial_discretization
        if is_pressure_based:
            try:
                discretization.discretization_scheme = PB_DISCRETIZATION.copy()
            except Exception as exc:
                available = _available_discretization_keys(discretization)
                raise ValueError(
                    f"Fluent rejected pressure-based discretization; available keys: "
                    f"{available}; {type(exc).__name__}: {exc}") from exc
        else:
            discretization.discretization_scheme = {
                "amg-c": "second-order-upwind", "k": "second-order-upwind",
                "omega": "second-order-upwind"}
            if not getattr(self, "steering_active", False):
                self.set_courant(initial_courant)
        if init_data and (is_pressure_based or not self._warm_start_error):
            init_path = Path(init_data).expanduser().resolve()
            self.settings.file.read_data(file_name=str(init_path))
            if self._warm_start_error:
                self.warm_start_status = "failed: " + self._warm_start_error
        elif self._warm_start_error:
            self._initialize_fallback(is_pressure_based)
            self.warm_start_status = "failed: " + self._warm_start_error
        elif self._warm_start_file:
            try:
                self.session.tui.file.interpolate.read_data(self._warm_start_file)
            except Exception as exc:
                self._initialize_fallback(is_pressure_based)
                self.warm_start_status = f"failed: {type(exc).__name__}: {exc}"
            else:
                self.warm_start_status = self._warm_start_source
        else:
            self._initialize_fallback(is_pressure_based)
        return self.groups

    def _initialize_fallback(self, is_pressure_based: bool):
        if is_pressure_based:
            self.initialize_from_farfield()
        else:
            self.settings.solution.initialization.hybrid_initialize()

    def initialize_from_farfield(self):
        initialization = self.settings.solution.initialization
        zone = self.groups["farfield"][0]
        try:
            initialization.compute_defaults(
                from_zone_type="pressure-far-field",
                from_zone_name=zone, phase="mixture")
        except Exception:
            self.session.tui.solve.initialize.compute_defaults(
                "pressure-far-field", zone)
        initialization.standard_initialize()

    def configure_autosave(self, output_dir: str | Path, every: int = 500):
        if not isinstance(every, int) or every < 0:
            raise ValueError("autosave interval must be a non-negative integer")
        auto_save = self.settings.solution.calculation_activity.auto_save
        auto_save.case_frequency = "if-case-is-modified"
        auto_save.data_frequency = every
        auto_save.root_name = str(Path(output_dir).expanduser().resolve() / "autosave")
        auto_save.retain_most_recent_files = True
        auto_save.max_files = 2

    def write_solver_settings(self, path: str | Path, stage: str):
        if stage not in ("after_setup", "after_last_iteration"):
            raise ValueError(f"unknown solver settings stage: {stage}")
        snapshots = getattr(self, "_solver_settings_snapshots", {})
        snapshots[stage] = {
            "setup": self.setup.get_state(),
            "solution": self.settings.solution.get_state(),
        }
        self._solver_settings_snapshots = snapshots
        Path(path).write_text(json.dumps(snapshots, indent=2, default=str),
                              encoding="utf-8")

    def initialize_fmg(self):
        self.session.tui.solve.initialize.fmg_initialization()

    def enable_solution_steering(self):
        self.session.tui.solve.set.solution_steering("transonic")

    def enable_casm(self):
        self.session.tui.solve.set.convergence_acceleration_for_stretched_meshes()

    def solver_audit(self, cwd: str | Path) -> list[str] | str:
        transcript = getattr(self.session, "transcript", None)
        text = transcript if isinstance(transcript, str) else ""
        if transcript is not None and not text:
            for name in ("get_contents", "contents", "get_text"):
                source = getattr(transcript, name, None)
                try:
                    value = source() if callable(source) else source
                except Exception:
                    continue
                if isinstance(value, str):
                    text = value
                    break
        if not text:
            text = "".join(self._transcript_chunks)
        if not text:
            try:
                logs = [path for path in Path(cwd).glob("fluent-*.trn")
                        if path.name not in self._transcript_baseline]
                logs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
                for path in logs:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    if text:
                        break
            except OSError:
                text = ""
        self.solver_transcript_available = bool(text.strip())
        self.native_gpu_solver_active = (
            bool(re.search(r"GPU Solver enabled", text, re.IGNORECASE))
            if self.solver_transcript_available else None)
        if not self.solver_transcript_available:
            return "unavailable"
        return [line for line in text.splitlines()
                if any(word in line.lower() for word in
                       ("ignored", "converted", "unsupported"))]

    def _create_reports(self, refs: dict):
        rd = self.report_definitions
        definitions = {**{group: self.groups[group] for group in REPORT_COMPONENTS},
                       "total": self.groups["total"],
                       "total_ex_duct": self.groups["total_ex_duct"]}
        for group, zones in definitions.items():
            suffix = "" if group == "total" else "_" + group
            cl_name = "cl" + suffix
            cd_name = "cd" + suffix
            cm_name = "cm" + suffix
            rd.lift.create(name=cl_name)
            cl = rd.lift[cl_name]
            rd.drag.create(name=cd_name)
            cd = rd.drag[cd_name]
            rd.moment.create(name=cm_name)
            cm = rd.moment[cm_name]
            cl.zones = list(zones)
            cd.zones = list(zones)
            cm.zones = list(zones)
            cl.average_over = 1
            cd.average_over = 1
            cm.mom_center = refs["moment_center_m"]
            cm.mom_axis = refs["moment_axis"]
            self.names[group] = {"CL": cl_name, "CD": cd_name, "CM": cm_name}
            self._lift_names.append(cl_name)
            self._drag_names.append(cd_name)

    def set_alpha(self, alpha: float):
        radians = math.radians(alpha)
        cosine, sine = math.cos(radians), math.sin(radians)
        for pff in self.pffs:
            pff.momentum.flow_direction = [
                {"option": "value", "value": cosine}, {"option": "value", "value": 0},
                {"option": "value", "value": sine}]
        for name in self._lift_names:
            self.report_definitions.lift[name].force_vector = [-sine, 0, cosine]
        for name in self._drag_names:
            self.report_definitions.drag[name].force_vector = [cosine, 0, sine]

    def set_courant(self, value: float):
        self.settings.solution.controls.courant_number = value

    def iterate(self, count: int):
        self.settings.solution.run_calculation.iterate(iter_count=count)

    def sample(self) -> dict:
        names = [name for group in self.names.values() for name in group.values()]
        raw = self.report_definitions.compute(report_defs=names)
        values = {}
        for row in raw:
            for key, value in row.items():
                values[key] = float(value[0] if isinstance(value, (tuple, list)) else value)
        components = {group: {key: values[name] for key, name in reports.items()}
                      for group, reports in self.names.items() if group in REPORT_COMPONENTS}
        return {"CL": values[self.names["total"]["CL"]],
                "CD": values[self.names["total"]["CD"]],
                "CM": values[self.names["total"]["CM"]],
                "CD_ex_duct": values[self.names["total_ex_duct"]["CD"]],
                "CM_ex_duct": values[self.names["total_ex_duct"]["CM"]],
                "components": components,
                "all_reports": {group: {key: values[name] for key, name in reports.items()}
                                for group, reports in self.names.items()},
                "mass_imbalance": self._mass_imbalance()}

    @staticmethod
    def _field_item(data, label: str):
        if hasattr(data, "face_normals") and label == "face-normal":
            return data.face_normals
        if hasattr(data, "face_centroids") and label == "centroid":
            return data.face_centroids
        if isinstance(data, dict):
            for key, value in data.items():
                normalized = re.sub(r"[^a-z]", "", str(getattr(key, "value", key)).lower())
                if label.replace("-", "") in normalized:
                    return value
        raise ValueError(f"Fluent field response did not contain {label}")

    def _scalar_fields(self, surfaces: list[str], field: str):
        from ansys.fluent.core.fields.field_data_interfaces import ScalarFieldDataRequest
        import numpy as np
        data = self.session.fields.field_data.get_field_data(
            ScalarFieldDataRequest(field_name=field, surfaces=surfaces, node_value=False))
        return {surface: np.asarray(data[surface], dtype=float).reshape(-1)
                for surface in surfaces}

    def _mass_imbalance(self) -> float:
        from ansys.fluent.core.fields.field_data_interfaces import (
            SurfaceFieldDataRequest, SurfaceDataType as SDT)
        import numpy as np
        surfaces = self.groups["farfield"]
        geometry = self.session.fields.field_data.get_field_data(
            SurfaceFieldDataRequest(surfaces=surfaces, data_types=[SDT.FacesNormal]))
        density = self._scalar_fields(surfaces, "density")
        velocity = [self._scalar_fields(surfaces, name) for name in
                    ("x-velocity", "y-velocity", "z-velocity")]
        net = inflow = 0.0
        for surface in surfaces:
            normals = np.asarray(self._field_item(geometry[surface], "face-normal"), dtype=float)
            mass_flux = density[surface] * np.sum(
                np.column_stack([field[surface] for field in velocity]) * normals, axis=1)
            net += float(np.sum(mass_flux))
            inflow += float(-np.sum(mass_flux[mass_flux < 0.0]))
        if inflow <= 0.0:
            raise ValueError("farfield mass-flow imbalance cannot be computed: no inflow")
        return abs(net) / inflow

    def export_walls(self, path: str):
        from ansys.fluent.core.fields.field_data_interfaces import (
            SurfaceFieldDataRequest, SurfaceDataType as SDT)
        import numpy as np
        surfaces = self.groups["total"]
        geometry = self.session.fields.field_data.get_field_data(
            SurfaceFieldDataRequest(
                surfaces=surfaces,
                data_types=[SDT.FacesCentroid, SDT.FacesNormal, SDT.Vertices,
                            SDT.FacesConnectivity]))
        fields = {name: self._scalar_fields(surfaces, name) for name in
                  ("pressure-coefficient", "y-plus", "x-wall-shear", "y-wall-shear", "z-wall-shear")}
        header = ["x", "y", "z", "zone", "pressure-coefficient", "wall-yplus",
                  "x-wall-shear", "y-wall-shear", "z-wall-shear",
                  "area-vector-x", "area-vector-y", "area-vector-z"]
        with open(path, "w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(header)
            for surface in surfaces:
                face_geometry = geometry[surface]
                points = np.asarray(
                    self._field_item(face_geometry, "centroid"), dtype=float)
                normals = np.asarray(self._field_item(face_geometry, "face-normal"), dtype=float)
                area_vectors = polygon_area_vectors(face_geometry, normals)
                arrays = [fields[name][surface] for name in
                          ("pressure-coefficient", "y-plus", "x-wall-shear", "y-wall-shear", "z-wall-shear")]
                if (any(len(values) != len(points) for values in arrays)
                        or area_vectors.shape != points.shape):
                    raise ValueError(f"wall field sizes differ on zone {surface}")
                for index, point in enumerate(points):
                    writer.writerow([*point, surface,
                                     *(float(values[index]) for values in arrays),
                                     *(float(value) for value in area_vectors[index])])

    def save_final(self, output_dir: str):
        self.settings.file.write_case_data(file_name=str(Path(output_dir) / "final.cas.h5"))

    def export_ensight(self, output_dir: str):
        """Export final cell-centered solution fields in EnSight Gold format."""
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.settings.file.export.ensight_gold(
            file_name=str(directory / "final"),
            # Fluent 26.1 accepts velocity only as components (live check 2026-10-04);
            # node-centred values give smooth slices and streamlines in pyvista.
            quantities=["x-velocity", "y-velocity", "z-velocity", "velocity-magnitude",
                        "pressure", "mach-number", "pressure-coefficient", "density"],
            binary_format=True,
            cellzones=list(self.setup.cell_zone_conditions.keys()),
            interior_zone_surfaces=[],
            cell_centered=False)
        write_portable_case(directory / "final.encas")


    def close(self):
        self.session.exit()
