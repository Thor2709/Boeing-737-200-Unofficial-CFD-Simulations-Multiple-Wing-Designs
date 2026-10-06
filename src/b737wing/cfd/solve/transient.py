"""Native-GPU URANS driver: continue a steady Fluent case in time-accurate mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DT = 1.0e-4
DEFAULT_STEPS = 200
DEFAULT_INNER = 15
DEFAULT_CHUNK = 10
DEFAULT_AUTOSAVE = 50
DEFAULT_ANIM = 10
DEFAULT_MAX_HOURS = 6.0
STATS_WINDOW = 50
VELOCITY_RANGE_MAX = 320.0
# Transient inner-iteration controls. The steady GPU case carries flow Courant 200
# with temperature/density relaxation 1.0; with 20 inner iterations per step the
# energy equation blew up in step 2 (5,000 K clip -> NaN, URANS T2 2026-10-06).
# The GPU solver replaces Coupled with SIMPLE in transient (Coupled is steady
# only), so there is no Courant control. Probe 3 (2026-10-06): SIMPLEC with
# these URFs plus a 400 K temperature cap ran 60 steps clean; without the cap a
# few cells at the pylon/nacelle junction overheat and the run dies (step ~22).
FLOW_SCHEME = "SIMPLEC"
# URF profiles: "default" keeps Fluent's SIMPLEC defaults (2026 R1 uses the
# pressure URF in the compressible transient linearisation, so do not lower it
# blindly); "low" is the probe-3 fallback.
URF_PROFILES = {
    "default": {},
    "low": {"pressure": 0.3, "mom": 0.5, "density": 0.5,
            "temperature": 0.7, "k": 0.6, "omega": 0.6},
}
# Ansys GPU limitation: a steady solution loaded from file does not initialise a
# transient energy solution correctly; iterate steady in the same session first.
DEFAULT_STEADY_ITERS = 100
# SBES on the GPU: k-omega SST + hybrid_rans_les, WALE subgrid, bounded central
# momentum. Low-diffusion central / optimized LES numerics are not used (they
# cannot capture shocks).
SBES_VALUE = "stress-blended-eddy-simulation"
MOMENTUM_SCHEMES = {"sou": ("second-order-upwind",),
                    "bcd": ("bounded-central-differencing", "second-order-upwind")}
DEFAULT_MIN_FREE_GB = 15.0
DEFAULT_MAX_TEMPERATURE = 400.0
CRITICAL_CONTROLS = ("flow_scheme", "max_temperature")
DIVERGED_CL_ABS, DIVERGED_CD = 3.0, 1.0
TIME_SCHEMES = ("unsteady-2nd-order-bounded", "unsteady-2nd-order",
                "unsteady-1st-order")
# GPU probe 2026-10-06: 2nd-order-bounded from the steady C5 field diverged in
# step 1-2 (energy); 1st order ran clean at dt 2.5e-5..2.5e-4. Default is 1st.
FIRST_ORDER_SCHEMES = ("unsteady-1st-order",)


def schemes_for(order: int):
    return FIRST_ORDER_SCHEMES if order == 1 else TIME_SCHEMES
EXPORT_QUANTITIES = ["x-velocity", "y-velocity", "z-velocity", "pressure",
                     "density", "temperature", "mach-number"]
EXPORT_MAX_FAILURES = 2
X_WAKE_M = 19.0
Y_MID_M = 8.0
PROGRESS_FIELDS = ["step", "flow_time", "CL", "CD", "CM", "s_per_step"]
FINISHED = ("completed", "stopped_by_owner", "budget_reached")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def resolve_settings(args) -> dict:
    return {
        "case": args.case, "init_case": args.init_case, "init_data": args.init_data,
        "out": str(args.out), "dt": args.dt, "steps": args.steps,
        "inner": args.inner, "chunk": args.chunk,
        "autosave_every": args.autosave_every, "export_every": args.export_every,
        "anim_every": args.anim_every, "max_hours": args.max_hours,
        "max_temperature": args.max_temperature,
        "urf_profile": args.urf_profile, "steady_iters": args.steady_iters,
        "turbulence": args.turbulence, "momentum": args.momentum,
        "export_start": args.export_start, "min_free_gb": args.min_free_gb,
        "gui": bool(args.gui), "solver": "gpu-pb", "precision": "single",
        "time_order": args.time_order,
        "time_schemes_tried": list(schemes_for(args.time_order)),
    }


# --------------------------------------------------------------------------
# Fluent launch and backend
# --------------------------------------------------------------------------

def launch_session(out_dir, gui: bool = False, launcher=None):
    """Launch Fluent 26.1 native GPU (single precision, 1 process)."""
    os.environ.setdefault("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")
    if launcher is None:
        import ansys.fluent.core as pyfluent
        launcher = pyfluent.launch_fluent
    # Pin the NVIDIA RTX 5070 Laptop GPU (CUDA device 0; the Intel iGPU has no CUDA).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    options = {"mode": "solver", "precision": "single", "processor_count": 1,
               "gpu": [0], "product_version": "26.1.0", "cleanup_on_exit": True,
               "start_watchdog": False,
               "cwd": str(Path(out_dir).expanduser().resolve())}
    if gui:
        options["ui_mode"] = "gui"
    return launcher(**options)



def share_ensight_geometry(frame: Path, owner) -> None:
    """Keep one static .geo (the mesh never moves): later frames point to the first."""
    geo, encas = frame.with_suffix(".geo"), frame.with_suffix(".encas")
    first = getattr(owner, "_ensight_geo", None)
    if first is None:
        owner._ensight_geo = geo.name
        return
    if geo.exists() and encas.exists():
        text = encas.read_text(encoding="utf-8")
        encas.write_text(text.replace(f'"{geo.name}"', f'"{first}"'), encoding="utf-8")
        geo.unlink()

class FluentTransientBackend:
    """Thin wrapper over a pyfluent solver session."""

    def __init__(self, session, out_dir):
        self.session = session
        self.out_dir = Path(out_dir).resolve()
        self.not_set: dict[str, str] = {}
        self.applied: dict = {}
        self.native_gpu_solver_active = None
        self._baseline = {p.name for p in self.out_dir.glob("fluent-*.trn")}

    @property
    def settings(self):
        return self.session.settings

    def _guard(self, key, func):
        try:
            func()
            self.applied[key] = "ok"
            return True
        except Exception as exc:
            self.not_set[key] = f"not_set: {_reason(exc)}"
            return False

    def read(self, case_file, data_file):
        self.session.settings.file.read_case(file_name=str(Path(case_file).resolve()))
        self.session.settings.file.read_data(file_name=str(Path(data_file).resolve()))

    def audit(self) -> bool:
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
            try:
                logs = list(self.out_dir.glob("fluent-*.trn"))
                logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for path in logs:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if text:
                        break
            except OSError:
                text = ""
        self.native_gpu_solver_active = (
            bool(re.search(r"GPU Solver enabled", text, re.IGNORECASE))
            if text.strip() else None)
        return bool(self.native_gpu_solver_active)

    def prepare_physics(self):
        def _viscous_heating():
            self.settings.setup.models.viscous.options.viscous_heating = True
        self._guard("viscous_heating", _viscous_heating)

    def steady_iterate(self, n):
        self.settings.solution.run_calculation.iterate(iter_count=int(n))

    def set_turbulence(self, model, momentum):
        viscous = self.settings.setup.models.viscous
        if model == "sbes":
            def _sbes():
                viscous.hybrid_rans_les = SBES_VALUE
                self.applied["hybrid_rans_les_value"] = viscous.hybrid_rans_les()
                if self.applied["hybrid_rans_les_value"] != SBES_VALUE:
                    raise RuntimeError("SBES readback " + str(self.applied["hybrid_rans_les_value"]))
            self._guard("sbes", _sbes)

            def _wale():
                viscous.sbes.les_subgrid_scale_model = "wale"
                self.applied["sbes_subgrid_value"] = viscous.sbes.les_subgrid_scale_model()
            self._guard("sbes_subgrid", _wale)
        scheme = self.settings.solution.methods.discretization_scheme["mom"]
        rejected = []
        for name in MOMENTUM_SCHEMES[momentum]:
            try:
                scheme.set_state(name)
                self.applied["momentum_scheme"] = scheme()
                break
            except Exception as exc:
                rejected.append(f"{name}: {_reason(exc)}")
        if rejected:
            self.applied["momentum_rejections"] = rejected

    def set_controls(self, max_temperature, urf_profile="default"):
        ctrl = self.settings.solution.controls

        def _scheme():
            coupling = self.settings.solution.methods.p_v_coupling
            coupling.flow_scheme = FLOW_SCHEME
            self.applied["flow_scheme_value"] = coupling.flow_scheme()
        self._guard("flow_scheme", _scheme)
        for name, value in URF_PROFILES[urf_profile].items():
            self._guard(f"relax_{name}", lambda n=name, v=value: ctrl.under_relaxation.__setitem__(n, v))
        self._guard("max_temperature", lambda: setattr(ctrl.limits, "max_temperature", max_temperature))

    def set_transient(self, dt, inner, chunk, schemes=TIME_SCHEMES) -> dict:
        solver = self.settings.setup.general.solver
        rejections = []
        scheme = None
        for name in schemes:
            try:
                solver.time = name
                scheme = name
                break
            except Exception as exc:
                rejections.append(f"{name}: {_reason(exc)}")
        accepted = None
        if scheme is not None:
            try:
                accepted = solver.time()
            except Exception:
                accepted = scheme
        formulation = None
        formulation_rejections = []
        if scheme is not None:
            for name in schemes:
                try:
                    methods = self.settings.solution.methods
                    methods.transient_formulation = name
                    formulation = methods.transient_formulation()
                    break
                except Exception as exc:
                    formulation_rejections.append(f"{name}: {_reason(exc)}")
        info = {"time_scheme": accepted, "transient_formulation": formulation,
                "time_scheme_rejections": rejections,
                "formulation_rejections": formulation_rejections}
        self.applied.update(info)
        if scheme is None:
            raise RuntimeError("no time scheme accepted: " + "; ".join(rejections))
        rc = self.settings.solution.run_calculation
        try:
            rc.transient_controls.type = "Fixed"
            self.applied["transient_type"] = "Fixed"
        except Exception as exc:
            self.not_set["transient_type"] = f"not_set: {_reason(exc)}"
        try:
            rc.parameters.time_step_size = dt
        except Exception as exc:
            raise RuntimeError(f"time_step_size not set: {_reason(exc)}") from exc
        self.applied["dt"] = dt
        self._guard("max_iter_per_time_step",
                    lambda: setattr(rc.parameters, "max_iter_per_time_step", inner))
        self._guard("time_step_count",
                    lambda: setattr(rc.parameters, "time_step_count", chunk))
        return info

    def setup_monitors(self, out_dir, autosave_every):
        out = Path(out_dir).resolve()
        sol = self.settings.solution

        def report_file():
            rf = sol.monitor.report_files.create(name="tr_forces")
            rf.file_name = str(out / "report_cl_cd_cm.out")
            rf.report_defs = ["cl", "cd", "cm"]
            rf.frequency_of = "time-step"
            rf.frequency = 1
            rf.write_instantaneous_values = True
            rf.print = True
        self._guard("report_file", report_file)

        def report_plot():
            rp = sol.monitor.report_plots.create(name="tr_plot")
            rp.report_defs = ["cl", "cd"]
            rp.frequency_of = "time-step"
            rp.frequency = 1
        self._guard("report_plot", report_plot)

        def autosave():
            a = sol.calculation_activity.auto_save
            a.case_frequency = "if-case-is-modified"
            a.data_frequency = autosave_every
            try:
                a.max_files = 2  # keep only the latest autosaves
            except Exception:
                pass
        self._guard("autosave_frequency", autosave)
        self._guard("autosave_root_name", lambda: setattr(
            sol.calculation_activity.auto_save, "root_name", str(out / "autosave")))

    def setup_live_views(self, out_dir, anim_every):
        out = Path(out_dir).resolve()
        res = self.settings.results

        def plane(name, p0, p1, p2):
            # bounded rectangle (three points) so the view fits the region of
            # interest, not the 1 km far-field plane
            ps = res.surfaces.plane_surface.create(name=name)
            ps.method = "three-points"
            ps.p0, ps.p1, ps.p2 = p0, p1, p2
            ps.bounded = True
        # cross-flow plane 1.2 m behind the C5 tip trailing edge (wing x 9.8-17.8 m,
        # tip y 14.1 m); chordwise plane at y 8 m from the nose region to past the tail
        self._guard("surface_x_wake", lambda: plane(
            "x_wake", [X_WAKE_M, 0.0, -3.0], [X_WAKE_M, 17.0, -3.0], [X_WAKE_M, 0.0, 4.0]))
        self._guard("surface_y_mid", lambda: plane(
            "y_mid", [6.0, Y_MID_M, -3.0], [32.0, Y_MID_M, -3.0], [6.0, Y_MID_M, 4.0]))

        def contour(name, field, surfaces):
            c = res.graphics.contour.create(name=name)
            c.field = field
            c.surfaces_list = surfaces
            if field == "velocity-magnitude":
                # fixed scale: auto-range is blown out by a single fast cell
                c.range_options.auto_range = False
                c.range_options.minimum = 0.0
                c.range_options.maximum = VELOCITY_RANGE_MAX
        # Fluent camera presets are fixed to Fluent's axes, not the aircraft's:
        # "front" looks along -z (x-y plane), "right" along x (y-z plane), "top"
        # along -y (x-z plane). Aircraft frame: x aft, y span, z up.
        views = [("c_vel_wake", "velocity-magnitude", ["x_wake"], "right"),
                 ("c_vel_mid", "velocity-magnitude", ["y_mid"], "top"),
                 ("c_p_wing", "pressure", ["mainwing"], "front")]
        for name, field, surfaces, camera in views:
            self._guard(f"contour_{name}",
                        lambda n=name, f=field, s=surfaces: contour(n, f, s))
            frames = out / f"frames_{name}"

            def anim(n=name, d=frames):
                a = self.settings.solution.calculation_activity.solution_animations \
                    .create(name=f"anim_{n}")
                a.animate_on = n
                a.frequency_of = "time-step"
                a.frequency = anim_every
                a.storage_type = "png"
                d.mkdir(parents=True, exist_ok=True)
                a.storage_dir = str(d)
            self._guard(f"animation_{name}", anim)
            self._guard(f"view_{name}", lambda n=name, v=camera: setattr(
                self.settings.solution.calculation_activity.solution_animations[f"anim_{n}"],
                "view", v))

    def advance(self, n, inner):
        self.settings.solution.run_calculation.dual_time_iterate(
            time_step_count=n, max_iter_per_step=inner)

    def sample(self) -> dict:
        names = ["cl", "cd", "cm"]
        rows = self.settings.solution.report_definitions.compute(report_defs=names)
        merged = {}
        for row in rows:
            for key, value in dict(row).items():
                merged[key] = value
        out = {}
        for name in names:
            value = merged[name]
            if isinstance(value, (list, tuple)):
                value = value[0]
            out[name.upper()] = float(value)
        return out

    def flow_time(self):
        try:
            return float(self.settings.solution.run_calculation
                         .transient_controls.flow_time())
        except Exception:
            return None

    def export_ensight(self, path):
        self.settings.file.export.ensight_gold(
            file_name=str(Path(path).resolve()), quantities=EXPORT_QUANTITIES,
            binary_format=True,
            # without explicit cell zones Fluent 26.1 fails ("invalid cell thread id")
            # and the GPU node then segfaults (smoke test 2026-10-06)
            cellzones=list(self.settings.setup.cell_zone_conditions.keys()),
            interior_zone_surfaces=[], cell_centered=False)
        share_ensight_geometry(Path(path).resolve(), self)



    def save_final(self, path):
        self.session.settings.file.write_case_data(file_name=str(Path(path).resolve()))

    def close(self):
        try:
            self.session.exit()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _stats(values):
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"mean": mean, "min": min(values), "max": max(values), "std": std,
            "n": n}


def _write_result(out, result):
    out.mkdir(parents=True, exist_ok=True)
    (out / "transient_result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")


def run_transient(backend, settings: dict, clock=time.monotonic, log=print) -> dict:
    out = Path(settings["out"])
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "error", "settings": settings, "applied": {},
              "not_set": {}, "steps_done": 0, "flow_time": None,
              "s_per_step": None, "stats": {}, "started": _now(), "finished": None,
              "message": None}
    samples = {"CL": [], "CD": [], "CM": []}
    state = {"steps_done": 0, "wall": 0.0}

    def finalize():
        result["steps_done"] = state["steps_done"]
        if state["steps_done"]:
            result["s_per_step"] = state["wall"] / state["steps_done"]
        result["stats"] = {k: _stats(v[-STATS_WINDOW:]) for k, v in samples.items()}
        result["applied"] = dict(getattr(backend, "applied", {}) or {})
        result["not_set"] = dict(getattr(backend, "not_set", {}) or {})
        result["finished"] = _now()
        _write_result(out, result)

    try:
        backend.read(settings["init_case"], settings["init_data"])
        if not backend.audit():
            result["status"] = "audit_failed"
            result["message"] = "native GPU solver not confirmed (no 'GPU Solver enabled')"
            return result
        prepare = getattr(backend, "prepare_physics", None)
        if prepare is not None:
            prepare()
        steady_iters = settings.get("steady_iters", 0)
        if steady_iters and hasattr(backend, "steady_iterate"):
            log(f"steady iterations in session before transient: {steady_iters}", flush=True)
            backend.steady_iterate(steady_iters)
        try:
            backend.set_transient(settings["dt"], settings["inner"], settings["chunk"],
                                  schemes=tuple(settings.get("time_schemes_tried", TIME_SCHEMES)))
            set_controls = getattr(backend, "set_controls", None)
            if set_controls is not None:
                set_controls(settings.get("max_temperature", DEFAULT_MAX_TEMPERATURE),
                             settings.get("urf_profile", "default"))
            set_turbulence = getattr(backend, "set_turbulence", None)
            if set_turbulence is not None:
                set_turbulence(settings.get("turbulence", "sst"),
                               settings.get("momentum", "sou"))
            critical = CRITICAL_CONTROLS + (
                ("sbes",) if settings.get("turbulence") == "sbes" else ())
            missing = [k for k in critical if k in getattr(backend, "not_set", {})]
            if missing:
                raise RuntimeError("stability controls not applied: " + ", ".join(missing))
        except Exception as exc:
            result["message"] = f"transient setup failed: {_reason(exc)}"
            return result
        backend.setup_monitors(out, settings["autosave_every"])
        if settings["anim_every"] > 0:
            backend.setup_live_views(out, settings["anim_every"])

        budget_s = settings["max_hours"] * 3600.0
        t_start = clock()
        export_failures = 0
        export_error = None
        status = "completed"
        progress = out / "progress.csv"
        with progress.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
            writer.writeheader()
            try:
                last_time = backend.flow_time()
            except Exception:
                last_time = None
            while state["steps_done"] < settings["steps"]:
                if (out / "STOP").exists():
                    status = "stopped_by_owner"
                    break
                if clock() - t_start >= budget_s:
                    status = "budget_reached"
                    break
                n = min(settings["chunk"], settings["steps"] - state["steps_done"])
                t0 = clock()
                backend.advance(n, settings["inner"])
                dt_wall = clock() - t0
                state["steps_done"] += n
                state["wall"] += dt_wall
                sample = backend.sample()
                flow_time = backend.flow_time()
                result["flow_time"] = flow_time
                for key in samples:
                    samples[key].append(sample[key])
                values = [sample["CL"], sample["CD"], sample["CM"]]
                # Fluent aborts a chunk on a floating point exception without
                # raising; a chunk that did not advance flow time has diverged.
                if (last_time is not None and flow_time is not None
                        and flow_time - last_time < 0.5 * n * settings["dt"]):
                    status = "diverged"
                    result["message"] = (f"Fluent stopped the chunk ending at step "
                                         f"{state['steps_done']}: flow time "
                                         f"{last_time} -> {flow_time}")
                    log(result["message"], flush=True)
                    break
                last_time = flow_time
                if (not all(math.isfinite(v) for v in values)
                        or abs(sample["CL"]) > DIVERGED_CL_ABS or sample["CD"] > DIVERGED_CD
                        or sample["CD"] < 0.0):
                    status = "diverged"
                    result["message"] = (f"diverged at step {state['steps_done']}: "
                                         f"CL {sample['CL']} CD {sample['CD']}")
                    log(result["message"], flush=True)
                    break
                per_step = dt_wall / n
                writer.writerow({"step": state["steps_done"], "flow_time": flow_time,
                                 "CL": sample["CL"], "CD": sample["CD"],
                                 "CM": sample["CM"], "s_per_step": per_step})
                handle.flush()
                log(f"step {state['steps_done']} flow_time {flow_time} "
                    f"CL {sample['CL']:.6f} CD {sample['CD']:.6f} "
                    f"CM {sample['CM']:.6f} {per_step:.3f} s/step", flush=True)
                every = settings["export_every"]
                due = (every > 0 and state["steps_done"] % every == 0
                       and state["steps_done"] >= settings.get("export_start", 0)
                       and export_failures < EXPORT_MAX_FAILURES)
                if due and not result.get("disk_guard"):
                    free_gb = shutil.disk_usage(out).free / 1e9
                    if free_gb < settings.get("min_free_gb", DEFAULT_MIN_FREE_GB):
                        result["disk_guard"] = (f"exports stopped at step {state['steps_done']}: "
                                                f"{free_gb:.1f} GB free")
                        log(result["disk_guard"], flush=True)
                if due and not result.get("disk_guard"):
                    try:
                        (out / "ensight").mkdir(exist_ok=True)
                        backend.export_ensight(
                            out / "ensight" / f"frame_{state['steps_done']:05d}")
                    except Exception as exc:
                        export_failures += 1
                        export_error = export_error or _reason(exc)
        if export_error:
            result["export_error"] = export_error
        try:
            backend.save_final(out / "final")
        except Exception as exc:
            result["save_final_error"] = _reason(exc)
        result["status"] = status
    except Exception as exc:
        result["status"] = "error"
        result["message"] = _reason(exc)
    finally:
        finalize()
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--init-case", required=True)
    p.add_argument("--init-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--inner", type=int, default=DEFAULT_INNER)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--autosave-every", type=int, default=DEFAULT_AUTOSAVE)
    p.add_argument("--export-every", type=int, default=0)
    p.add_argument("--anim-every", type=int, default=DEFAULT_ANIM)
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    p.add_argument("--max-temperature", type=float, default=DEFAULT_MAX_TEMPERATURE)
    p.add_argument("--urf-profile", choices=sorted(URF_PROFILES), default="default")
    p.add_argument("--steady-iters", type=int, default=DEFAULT_STEADY_ITERS)
    p.add_argument("--turbulence", choices=("sst", "sbes"), default="sst")
    p.add_argument("--momentum", choices=sorted(MOMENTUM_SCHEMES), default="sou")
    p.add_argument("--export-start", type=int, default=0,
                   help="first step eligible for EnSight export (skip spin-up)")
    p.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    p.add_argument("--time-order", type=int, choices=(1, 2), default=1)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--dry", action="store_true")
    return p


def main(argv=None, launcher=None, backend_factory=None) -> int:
    args = build_parser().parse_args(argv)
    settings = resolve_settings(args)
    if args.dry:
        print(json.dumps(settings, indent=2))
        return 0
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = None
    try:
        from .fluent_io import record_fluent_pids, reset_fluent_pids
        reset_fluent_pids(out)
        session = launch_session(out, args.gui, launcher)
        try:
            record_fluent_pids(out, session)
        except Exception as exc:
            print(f"fluent pid recording failed: {_reason(exc)}", file=sys.stderr)
        backend = (backend_factory or FluentTransientBackend)(session, out)
        result = run_transient(backend, settings)
    except Exception as exc:
        result = {"status": "error", "message": _reason(exc), "settings": settings,
                  "finished": _now()}
        _write_result(out, result)
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()
    print(json.dumps({"status": result["status"], "steps_done": result.get("steps_done")}))
    return 0 if result["status"] in FINISHED else 1


if __name__ == "__main__":
    raise SystemExit(main())
