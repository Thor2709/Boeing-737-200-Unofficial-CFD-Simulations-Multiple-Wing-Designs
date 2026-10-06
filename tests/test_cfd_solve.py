import builtins
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from b737wing.cfd.solve.conditions import condition_for_case
from b737wing.cfd.solve.fluent_io import (FluentBackend,
                                          launch_fluent,
                                          resolve_zone_groups,
                                          write_interpolation_data)
from b737wing.cfd.solve.search import settle_check, secant_next, secant_search
from b737wing.cfd.solve import run_case


ZONES = ["mainwing:1", "fuselage-1", "horstab", "vertstab:1", "engine",
         "pylon-1", "nacelle_duct", "fairing:1", "farfield"]


def _fake_settings_state(value):
    if isinstance(value, SimpleNamespace):
        return {key: _fake_settings_state(item)
                for key, item in vars(value).items()
                if key != "get_state" and not callable(item)}
    if isinstance(value, dict):
        return {key: _fake_settings_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fake_settings_state(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def test_conditions_cruise_and_takeoff():
    cruise = condition_for_case("C5")
    takeoff = condition_for_case("C3")
    assert cruise["mach"] == pytest.approx(0.74, abs=0.005)
    assert cruise["pressure_Pa"] == pytest.approx(30080, rel=0.005)
    assert takeoff["mach"] == pytest.approx(0.218, abs=0.005)


def test_secant_linear_nonlinear_and_flat_curves():
    target = 0.4895
    linear = secant_search(lambda alpha: 0.11 * alpha + 0.27,
                           target, alpha0=0.0)
    nonlinear = secant_search(lambda alpha: 0.001 * alpha**2 + 0.1 * alpha + 0.2,
                              target, alpha0=1.0)
    flat = secant_search(lambda alpha: 0.2, target, alpha0=2.0)
    assert linear["status"] == nonlinear["status"] == "matched"
    assert abs(linear["cl"] - target) <= 0.003
    assert abs(nonlinear["cl"] - target) <= 0.003
    assert linear["steps"] <= 6 and nonlinear["steps"] <= 6
    assert flat["status"] == "cl_not_matched"


def test_secant_single_point_uses_assumed_slope_and_seed_slope():
    point = [(2.5, 0.396)]
    assert secant_next(point, 0.4895) == pytest.approx(3.435, abs=0.001)
    assert secant_next([(2.5, 0.0)], 0.4895) == 4.5
    assert secant_next(point, 0.4895, initial_slope=0.2) == pytest.approx(2.9675)


def test_settle_check_c5_exponential_remaining_change():
    amplitude = 0.004 * math.exp(1200 / 540)
    samples = [{"iteration": iteration,
                "CL": 0.4831 + amplitude * math.exp(-iteration / 540),
                "CD": 0.04, "CM": -0.02}
               for iteration in range(50, 2601, 50)]

    unsettled = settle_check([sample for sample in samples
                              if sample["iteration"] <= 1200])
    settled = settle_check(samples)

    assert unsettled["CL"]["remaining"] == pytest.approx(0.004, abs=1e-6)
    assert unsettled["CL"]["tau_iterations"] == pytest.approx(540, rel=1e-6)
    assert unsettled["settled"] is False
    assert settled["CL"]["remaining"] == pytest.approx(0.0003, abs=3e-6)
    assert settled["settled"] is True


def test_slow_exponential_does_not_settle_on_small_window_change_alone():
    tau = 10000.0
    final_remainder = 0.003
    amplitude = final_remainder * math.exp(900 / tau)
    samples = [{"iteration": iteration,
                "CL": 0.48 + amplitude * math.exp(-iteration / tau),
                "CD": 0.04, "CM": -0.02}
               for iteration in range(50, 901, 50)]

    result = settle_check(samples)
    window_change = samples[-1]["CL"] - samples[-13]["CL"]

    assert window_change < run_case.CL_SETTLE_TOLERANCE
    assert result["CL"]["remaining"] > run_case.CL_SETTLE_TOLERANCE
    assert result["settled"] is False


def test_settle_check_flat_series():
    samples = [{"iteration": iteration, "CL": 0.4831,
                "CD": 0.04, "CM": -0.02}
               for iteration in range(50, 651, 50)]

    result = settle_check(samples)

    assert result["settled"] is True
    assert all(result[quantity]["settled"]
               for quantity in ("CL", "CD", "CM"))


def test_settle_check_oscillation_uses_peak_to_peak_tolerance():
    within_tolerance = [{"iteration": iteration,
                         "CL": 0.4831 + (0.0002 if index % 2 else -0.0002),
                         "CD": 0.04, "CM": -0.02}
                        for index, iteration in enumerate(range(50, 651, 50))]
    outside_tolerance = [{"iteration": sample["iteration"],
                          "CL": 0.4831 + (0.0006 if index % 2 else -0.0006),
                          "CD": sample["CD"], "CM": sample["CM"]}
                         for index, sample in enumerate(within_tolerance)]

    assert settle_check(within_tolerance)["settled"] is True
    assert settle_check(outside_tolerance)["settled"] is False


def test_secant_replaces_negative_slope_with_assumed_slope():
    replacements = []
    candidate = secant_next([(3.5009, 0.48626), (3.5536, 0.4835)],
                            0.4895, slope_replacements=replacements)
    sample = {"CL": 0.4895, "CD": 0.04, "CM": -0.02,
              "CD_ex_duct": 0.03, "components": {}, "mass_imbalance": 0.0,
              "all_reports": {"total_ex_duct": {"CL": 0.48}}}
    result = run_case.build_result(
        "C5", "mesh.msh.h5", 4, condition_for_case("C5"), candidate,
        sample, 0, 0.0, "converged", 0,
        convergence={"slope_replacements": replacements})

    assert candidate == pytest.approx(3.5536 + 0.06, abs=0.001)
    assert result["convergence_numbers"]["slope_replacements"][0][
        "used_slope"] == pytest.approx(0.1)


def test_secant_bad_slope_uses_prior_settled_pair_or_seed():
    points = [(3.3, 0.4), (3.4, 0.405), (3.5, 0.404)]
    replacements = []
    from_settled = secant_next(
        points, 0.4045, settled_points=points,
        slope_replacements=replacements)
    from_seed = secant_next(points[-2:], 0.4045, initial_slope=0.05,
                            slope_replacements=[])

    assert from_settled == pytest.approx(3.51)
    assert replacements[0]["source"] == "settled_points"
    assert from_seed == pytest.approx(3.51)


def test_secant_accepts_c5_slope_bounds_unchanged():
    low = secant_next([(3.5, 0.4), (4.0, 0.4175)], 0.435, 0.1)
    high = secant_next([(3.5, 0.4), (4.0, 0.445)], 0.49, 0.1)

    assert low == pytest.approx(4.5)
    assert high == pytest.approx(4.5)


def test_secant_clamps_steps_after_the_first_step():
    first = secant_next([(3.5, 0.2)], 0.4895)
    subsequent = secant_next([(3.5, 0.4), (4.0, 0.41)], 0.5)

    assert first == pytest.approx(5.5)
    assert subsequent == pytest.approx(5.0)


def test_secant_bounds_alpha_and_low_slope_step():
    candidate = secant_next([(10.0, 1.60), (11.0, 1.61)], 1.675)
    outside = secant_search(lambda alpha: alpha, 0.0, alpha0=20.0)

    assert candidate <= 12.0
    assert candidate > 11.0
    assert outside["points"][0][0] == 16.0
    assert secant_next([(16.0, 0.5)], 1.0) is None


def test_falling_lift_branch_returns_best_settled_point(monkeypatch):
    settled = iter((0.80, 0.79))

    def advance(_backend, _target, alpha, _minimum, total, _records,
                **_kwargs):
        cl = next(settled)
        sample = {"CL": cl, "CD": 0.04 + cl / 100,
                  "CM": -0.02 - cl / 100}
        state = {"iterations": 600, "samples": [sample]}
        return state, "converged", {"iterations": 600}, total + 600

    monkeypatch.setattr(run_case, "_advance_alpha", advance)
    alpha, state, status, convergence, _ = run_case._solve_alpha(
        object(), 1.0, 10.0, None, [])

    assert status == "cl_unreachable"
    assert alpha == pytest.approx(10.0 + (1.0 - 0.80) / 0.1)
    assert state["samples"][-1]["CL"] == 0.79
    assert convergence["search"] == {
        "status": "cl_unreachable",
        "selected_point": "current_stalled_field",
        "alpha_deg": alpha,
        "CL": 0.79,
    }
    assert convergence["diagnostic_best_settled_point"]["alpha_deg"] == 10.0
    assert convergence["diagnostic_best_settled_point"]["CL"] == 0.80


def test_advance_alpha_waits_for_settle_window_and_reports_mass_imbalance(
        monkeypatch):
    check_iterations = []
    original_check = run_case.settle_check

    def record_check(samples):
        check_iterations.append(samples[-1]["iteration"])
        return original_check(samples)

    monkeypatch.setattr(run_case, "settle_check", record_check)

    class Backend:
        def set_alpha(self, _alpha):
            pass

        def set_courant(self, _value):
            pass

        def iterate(self, _count):
            pass

        def sample(self):
            return {"CL": 0.4831, "CD": 0.04, "CM": -0.02,
                    "mass_imbalance": 2e-4,
                    "all_reports": {"total": {"CL": 0.4831,
                                                 "CD": 0.04, "CM": -0.02}}}

    records = []
    _, status, convergence, _ = run_case._advance_alpha(
        Backend(), 3.5, {}, 400, 0, records)

    assert check_iterations[0] == 600
    assert status == "converged"
    assert convergence["mass_imbalance"] == pytest.approx(2e-4)
    assert convergence["settle"]["settled"] is True
    assert json.loads(records[-1]["settle"])["settled"] is True


def test_search_advances_on_settled_cl_while_cm_drifts():
    target = condition_for_case("C5")["target_cl"]

    class SearchBackend:
        def __init__(self):
            self.alpha = None
            self.iterations = 0
            self.alpha_steps = []

        def set_alpha(self, alpha):
            if alpha != self.alpha:
                self.iterations = 0
            self.alpha = alpha
            self.alpha_steps.append(alpha)

        def set_courant(self, _value):
            pass

        def iterate(self, count):
            self.iterations += count

        def sample(self):
            if self.alpha == 3.5:
                cl = 0.4
                cm = -0.02 + self.iterations * 0.000002
            else:
                cl = target
                cm = -0.02
            return {"CL": cl, "CD": 0.04, "CM": cm,
                    "mass_imbalance": 0.0,
                    "all_reports": {"total": {"CL": cl, "CD": 0.04,
                                                 "CM": cm}}}

    backend = SearchBackend()
    alpha, _, status, convergence, _ = run_case._solve_alpha(
        backend, target, 3.5, None, [], min_final_iters=900)

    assert backend.alpha_steps[0] == 3.5
    assert backend.alpha_steps[1] != backend.alpha_steps[0]
    assert alpha == backend.alpha_steps[-1]
    assert status == "converged"
    assert convergence["settle_on"] == "all"


def test_final_alpha_waits_for_cm_to_settle(monkeypatch):
    settle_checks = []
    original_check = run_case.settle_check

    def record_check(samples):
        result = original_check(samples)
        settle_checks.append((result.get("iterations"), result["settled"],
                              result["CM"]["settled"]))
        return result

    target = condition_for_case("C5")["target_cl"]

    class DelayedCMBackend:
        def __init__(self):
            self.alpha = None
            self.iterations = 0

        def set_alpha(self, alpha):
            if alpha != self.alpha:
                self.iterations = 0
            self.alpha = alpha

        def set_courant(self, _value):
            pass

        def iterate(self, count):
            self.iterations += count

        def sample(self):
            cm = -0.02 + min(self.iterations, 900) * 0.000002
            return {"CL": target, "CD": 0.04, "CM": cm,
                    "mass_imbalance": 0.0,
                    "all_reports": {"total": {"CL": target, "CD": 0.04,
                                                 "CM": cm}}}

    monkeypatch.setattr(run_case, "settle_check", record_check)
    _, state, status, convergence, _ = run_case._solve_alpha(
        DelayedCMBackend(), target, 2.0, None, [], min_final_iters=900)

    assert (900, False, False) in settle_checks
    assert status == "converged"
    assert state["iterations"] > 900
    assert convergence["settle"]["CM"]["settled"] is True
    assert convergence["settle_on"] == "all"


def test_fixed_alpha_does_not_converge_while_cm_drifts():
    class DriftingCMBackend:
        def __init__(self):
            self.iterations = 0

        def set_alpha(self, _alpha):
            pass

        def set_courant(self, _value):
            pass

        def iterate(self, count):
            self.iterations += count

        def sample(self):
            cm = -0.02 + self.iterations * 0.000002
            return {"CL": 0.4895, "CD": 0.04, "CM": cm,
                    "mass_imbalance": 0.0,
                    "all_reports": {"total": {"CL": 0.4895, "CD": 0.04,
                                                 "CM": cm}}}

    _, _, status, convergence, _ = run_case._solve_alpha(
        DriftingCMBackend(), 0.4895, 2.0, 2.0, [])

    assert status == "not_converged"
    assert convergence["settle_on"] == "all"
    assert convergence["settle"]["CM"]["settled"] is False


def test_search_and_final_steps_record_their_settle_rule(monkeypatch):
    target = condition_for_case("C5")["target_cl"]

    class Backend:
        def __init__(self):
            self.iterations = 0

        def set_alpha(self, _alpha):
            pass

        def set_courant(self, _value):
            pass

        def iterate(self, count):
            self.iterations += count

        def sample(self):
            return {"CL": target, "CD": 0.04, "CM": -0.02,
                    "mass_imbalance": 0.0,
                    "all_reports": {"total": {"CL": target, "CD": 0.04,
                                                 "CM": -0.02}}}

    calls = []
    original_advance = run_case._advance_alpha

    def record_advance(*args, **kwargs):
        result = original_advance(*args, **kwargs)
        calls.append((kwargs.get("settle_on", "all"),
                      result[2]["settle_on"]))
        return result

    monkeypatch.setattr(run_case, "_advance_alpha", record_advance)
    _, _, status, convergence, _ = run_case._solve_alpha(
        Backend(), target, 2.0, None, [], min_final_iters=900)

    assert status == "converged"
    assert calls == [("cl", "cl"), ("all", "all")]
    assert convergence["settle_on"] == "all"


def test_zone_prefix_groups_and_duct_exclusion():
    groups = resolve_zone_groups(ZONES)
    assert groups["wing"] == ["mainwing:1"]
    assert groups["nacelle"] == ["engine", "pylon-1"]
    assert "nacelle_duct" in groups["total"]
    assert "nacelle_duct" not in groups["total_ex_duct"]
    with pytest.raises(ValueError, match="fairing.*zones found"):
        resolve_zone_groups([zone for zone in ZONES if not zone.startswith("fairing")])


class FakeBackend:
    cells = 123

    def configure(self, mesh, condition, init_data, **warm_start):
        self.alpha = None
        self.iterations = 0
        self.calls = []
        self.sample_count = 0
        self.warm_start_status = None
        session = _ConfigureSession(getattr(self, "configure_events", None))
        session.fail_interpolation_read = getattr(
            self, "fail_interpolation_read", False)
        self.configure_session = session
        adapter = FluentBackend(session)
        adapter.solver_name = self.solver_name
        adapter.steering_active = getattr(self, "steering_active", False)
        adapter.set_courant = self.set_courant
        self.adapter = adapter
        adapter.configure(mesh, condition, init_data, **warm_start)
        self.warm_start_status = adapter.warm_start_status
        return resolve_zone_groups(ZONES)

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.configure_session.events.append(("set_alpha", alpha))

    def set_courant(self, value):
        self.calls.append(("courant", value))
        if hasattr(self, "configure_session"):
            self.configure_session.events.append(("courant", value))
            self.adapter.settings.solution.controls.courant_number = value

    def iterate(self, count):
        self.configure_session.events.append(("iterate", count))
        if getattr(self, "history_path", None):
            with self.history_path.open(encoding="utf-8", newline="") as source:
                assert next(csv.reader(source)) == run_case.history_columns()
        self.iterations += count

    def initialize_fmg(self):
        self.adapter.initialize_fmg()

    def enable_solution_steering(self):
        self.adapter.enable_solution_steering()

    def enable_casm(self):
        self.adapter.enable_casm()

    def sample(self):
        if getattr(self, "history_path", None):
            with self.history_path.open(encoding="utf-8", newline="") as source:
                assert len(list(csv.DictReader(source))) == self.sample_count
        self.sample_count += 1
        components = {group: {"CL": 0.1, "CD": 0.01, "CM": 0.001}
                      for group in ("wing", "fuselage", "htail", "vtail",
                                    "nacelle", "duct", "fairing")}
        reports = {group: dict(values) for group, values in components.items()}
        reports["total"] = {"CL": 0.4895, "CD": 0.04, "CM": -0.02}
        reports["total_ex_duct"] = {"CL": 0.48, "CD": 0.03, "CM": -0.02}
        return {"CL": 0.4895, "CD": 0.04, "CM": -0.02,
                "CD_ex_duct": 0.03, "CM_ex_duct": -0.02,
                "components": components, "all_reports": reports,
                "mass_imbalance": 0.0}

    def export_walls(self, path):
        Path(path).write_text("x,y,z,zone\n", encoding="utf-8")

    def save_final(self, output_dir):
        self.calls.append("save_final")
        for name in ("final.cas.h5", "final.dat.h5"):
            (Path(output_dir) / name).write_bytes(b"fake")

    def export_ensight(self, output_dir):
        self.calls.append("export_ensight")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "final.case").write_text("fake", encoding="utf-8")

    def configure_autosave(self, output_dir, every):
        self.adapter.configure_autosave(output_dir, every)

    def write_solver_settings(self, path, stage):
        self.adapter.write_solver_settings(path, stage)

    def solver_audit(self, cwd):
        result = self.adapter.solver_audit(cwd)
        self.solver_transcript_available = self.adapter.solver_transcript_available
        self.native_gpu_solver_active = self.adapter.native_gpu_solver_active
        return result


def test_c3_stall_reports_and_saves_the_current_field(tmp_path):
    class FallingLiftBackend(FakeBackend):
        def sample(self):
            sample = super().sample()
            cl = 0.40 if self.alpha == 10.0 else 0.39
            sample["CL"] = cl
            sample["CD"] = 0.04 + self.alpha / 1000
            sample["CM"] = -0.02 - self.alpha / 1000
            sample["all_reports"]["total"]["CL"] = cl
            return sample

        def save_final(self, output_dir):
            self.saved_field_alpha = self.alpha
            super().save_final(output_dir)

        def export_walls(self, path):
            self.exported_walls_alpha = self.alpha
            super().export_walls(path)

        def export_ensight(self, output_dir):
            self.exported_ensight_alpha = self.alpha
            super().export_ensight(output_dir)

    backend = FallingLiftBackend()
    result = run_case.run_solve(backend, "C5", "mesh.msh.h5", tmp_path,
                                alpha0=10.0, initial_slope=0.1)
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result["status"] == saved["status"] == "cl_unreachable"
    assert result["alpha_deg"] == pytest.approx(backend.saved_field_alpha)
    assert result["alpha_deg"] == pytest.approx(backend.exported_walls_alpha)
    assert result["alpha_deg"] == pytest.approx(backend.exported_ensight_alpha)
    assert (result["CL"], result["CD"], result["CM"]) == pytest.approx(
        (0.39, 0.04 + result["alpha_deg"] / 1000,
         -0.02 - result["alpha_deg"] / 1000))
    assert saved["diagnostic_best_settled_point"]["alpha_deg"] == 10.0
    assert saved["diagnostic_best_settled_point"]["CL"] == pytest.approx(0.40)
    assert saved["cl_search"]["selected_point"] == "current_stalled_field"


def test_fake_run_writes_result_schema_and_full_aircraft_forces(tmp_path):
    output_dir = tmp_path
    generated = ("result.json", "history.csv", "walls.csv", "final.cas.h5",
                 "final.dat.h5", "ensight/final.case")
    backend = FakeBackend()
    backend.history_path = output_dir / "history.csv"
    result = run_case.run_solve(backend, "C5", "mesh.msh.h5", output_dir,
                                fixed_alpha=2.0)
    saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "converged"
    assert result["warm_start"] is None
    assert result["iterations"] == 900
    assert result["lift_N"] == pytest.approx(2 * result["lift_half_N"])
    assert result["drag_N"] == pytest.approx(2 * result["drag_half_N"])
    assert result["lift_lbf"] == pytest.approx(result["lift_N"] / 4.4482216152605)
    assert result["drag_lbf"] == pytest.approx(result["drag_N"] / 4.4482216152605)
    assert result["L_over_D"] == pytest.approx(0.4895 / 0.04)
    assert result["L_over_D_ex_duct"] == pytest.approx(0.48 / 0.03)
    assert result["components"]["wing"]["CL"] == 0.1
    assert all((output_dir / name).is_file() for name in generated)
    with (output_dir / "history.csv").open(newline="", encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == backend.sample_count
    assert [event for event in backend.calls if isinstance(event, str)] == [
        "save_final", "export_ensight"]
    required = {"case", "mesh", "cells", "np", "alpha_deg", "CL", "CD", "CM",
                "CD_ex_duct", "components", "lift_N", "drag_N", "lift_lbf",
                "drag_lbf", "L_over_D", "L_over_D_ex_duct", "iterations",
                "wall_time_s", "s_per_iter", "status", "convergence_numbers", "ih_deg"}
    assert required <= saved.keys()


def test_walls_export_retries_then_records_complete_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(run_case, "EXPORT_RETRY_PAUSE_SECONDS", 0)

    class RetryingBackend(FakeBackend):
        wall_export_attempts = 0

        def export_walls(self, path):
            self.wall_export_attempts += 1
            if self.wall_export_attempts < 3:
                raise RuntimeError("temporary export failure")
            super().export_walls(path)

    backend = RetryingBackend()
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0)

    assert backend.wall_export_attempts == 3
    assert (tmp_path / "walls.csv").is_file()
    assert result["export_status"] == "complete"
    assert result["exports"]["walls"] == str((tmp_path / "walls.csv").resolve())
    assert result["exports"]["ensight"] == str((tmp_path / "ensight").resolve())


def test_failed_cpu_walls_export_keeps_converged_result(tmp_path, monkeypatch):
    monkeypatch.setattr(run_case, "EXPORT_RETRY_PAUSE_SECONDS", 0)

    class FailingBackend(FakeBackend):
        def export_walls(self, _path):
            raise RuntimeError("walls unavailable")

    backend = FailingBackend()
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0)
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result["status"] == saved["status"] == "converged"
    assert "save_final" in backend.calls
    assert (tmp_path / "final.cas.h5").is_file()
    assert saved["export_status"] == "incomplete"
    assert saved["exports"]["walls"].startswith("failed:")


def test_result_and_saved_field_precede_first_export(tmp_path):
    events = []

    class OrderedBackend(FakeBackend):
        def save_final(self, output_dir):
            super().save_final(output_dir)
            events.append("save_final")

        def export_walls(self, path):
            events.append("export_walls")
            self.result_at_first_export = json.loads(
                (tmp_path / "result.json").read_text(encoding="utf-8"))
            super().export_walls(path)

    backend = OrderedBackend()
    run_case.run_solve(backend, "C5", "mesh.msh.h5", tmp_path,
                       fixed_alpha=2.0)

    assert events[:2] == ["save_final", "export_walls"]
    assert backend.result_at_first_export["status"] == "converged"
    assert (tmp_path / "final.cas.h5").is_file()


def test_gpu_export_failures_keep_skipped_records_and_order(tmp_path, monkeypatch):
    monkeypatch.setattr(run_case, "EXPORT_RETRY_PAUSE_SECONDS", 0)
    events = []

    class FailingGPUBackend(FakeBackend):
        native_gpu_solver_active = True
        solver_transcript_available = True

        def save_final(self, output_dir):
            super().save_final(output_dir)
            events.append("save_final")

        def solver_audit(self, _cwd):
            return []

        def export_walls(self, _path):
            events.append("walls")
            raise RuntimeError("walls unavailable")

        def export_ensight(self, _output_dir):
            events.append("ensight")
            raise RuntimeError("ensight unavailable")

    result = run_case.run_solve(
        FailingGPUBackend(), "C5", "mesh.msh.h5", tmp_path,
        fixed_alpha=2.0, solver="gpu-pb", min_final_iters=600)

    assert events == ["save_final", *(["walls"] * 3), *(["ensight"] * 3)]
    assert result["exports"]["walls"].startswith("skipped:")
    assert result["exports"]["ensight"].startswith("skipped:")
    assert result["export_status"] == "incomplete"


def test_owner_stop_after_three_chunks_saves_without_next_alpha_step(tmp_path):
    backend = FakeBackend()
    original_iterate = backend.iterate

    def stop_after_third_chunk(count):
        original_iterate(count)
        if backend.iterations == 150:
            (tmp_path / "STOP").write_text("stop", encoding="utf-8")

    backend.iterate = stop_after_third_chunk
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, alpha0=2.0)

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stopped_by_owner"
    assert result["converged"] is False
    assert result["iterations"] == 150
    assert [event for event in backend.configure_session.events
            if event[0] == "set_alpha"] == [("set_alpha", 2.0)]
    assert [call for call in backend.calls if isinstance(call, str)] == [
        "save_final", "export_ensight"]
    assert all((tmp_path / name).is_file()
               for name in ("final.cas.h5", "final.dat.h5", "walls.csv",
                            "ensight/final.case", "result.json"))
    assert saved["status"] == "stopped_by_owner"
    assert saved["convergence_numbers"]["settle"]["settled"] is False


def test_owner_stop_inside_first_settle_window_is_not_converged(tmp_path):
    backend = FakeBackend()
    original_iterate = backend.iterate

    def stop_inside_window(count):
        original_iterate(count)
        if backend.iterations == 600:
            (tmp_path / "STOP").write_text("stop", encoding="utf-8")

    backend.iterate = stop_inside_window
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0)
    settle = result["convergence_numbers"]["settle"]

    assert result["status"] == "stopped_by_owner"
    assert result["converged"] is False
    assert result["iterations"] == 600
    assert settle["settled"] is False


def test_final_point_uses_900_iteration_default_minimum(tmp_path):
    result = run_case.run_solve(
        FakeBackend(), "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0)

    assert result["status"] == "converged"
    assert result["iterations"] == 900


def test_min_final_iters_1500_overrides_settle_window_minimum(tmp_path):
    result = run_case.run_solve(
        FakeBackend(), "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        min_final_iters=1500)

    assert result["status"] == "converged"
    assert result["iterations"] == 1500


def test_result_records_final_iteration_rule(tmp_path):
    run_case.run_solve(
        FakeBackend(), "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        min_final_iters=1500)

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["final_iteration_rule"] == {
        "window_iterations": 600,
        "minimum_iterations": 1500,
        "tolerances": {"CL": 0.0005, "CD": 0.00005, "CM": 0.001},
    }


def test_first_slope_cli_controls_first_secant_step_and_result(
        tmp_path, monkeypatch):
    target = condition_for_case("C5")["target_cl"]

    class SlopeBackend(FakeBackend):
        def sample(self):
            sample = super().sample()
            cl = 0.3 if self.alpha == 2.0 else target
            sample["CL"] = cl
            sample["all_reports"]["total"]["CL"] = cl
            return sample

    backend = SlopeBackend()
    backend.set_transcript_baseline = lambda _names: None
    backend.close = lambda: None
    session = SimpleNamespace(
        connection_properties=SimpleNamespace(cortex_pid=None,
                                               fluent_host_pid=None),
        _process=SimpleNamespace(pid=987654321))
    monkeypatch.setattr(run_case, "_start_metrics",
                        lambda _out: (object(), "metrics.stop"))
    monkeypatch.setattr(run_case, "_stop_metrics", lambda _process, _stop: None)
    monkeypatch.setattr(run_case, "launch_fluent", lambda *args: session)
    monkeypatch.setattr(run_case, "FluentBackend", lambda _session: backend)
    monkeypatch.setattr(run_case.psutil, "pid_exists", lambda _pid: False)

    out = tmp_path / "slope-run"
    result_code = run_case.main([
        "C5", "--mesh", "mesh.msh.h5", "--out", str(out),
        "--alpha0", "2", "--first-slope", "0.079"])
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    alpha_steps = [event[1] for event in backend.configure_session.events
                   if event[0] == "set_alpha"]
    expected_alpha = 2.0 + (target - 0.3) / 0.079

    assert result_code == 0
    assert alpha_steps[:2] == pytest.approx([2.0, expected_alpha])
    assert saved["first_slope_cl_per_deg"] == pytest.approx(0.079)
    assert saved["close_clean"] is True


def test_c3_dry_resolves_takeoff_without_fluent_import(monkeypatch, capsys):
    original_import = builtins.__import__
    attempts = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith("ansys.fluent"):
            attempts.append(name)
            raise AssertionError("dry mode imported Fluent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert run_case.main(["C3", "--mesh", "mesh.msh.h5", "--out", "out",
                          "--init-data", "restart.dat.h5", "--dry"]) == 0
    setup = json.loads(capsys.readouterr().out)
    assert setup["conditions"]["condition"] == "takeoff"
    assert setup["conditions"]["target_cl"] == 1.675
    assert setup["conditions"]["mach"] == pytest.approx(0.218, abs=0.005)
    assert Path(setup["mesh"]).is_absolute()
    assert Path(setup["out"]).is_absolute()
    assert Path(setup["init_data"]).is_absolute()
    assert not attempts


def test_seed_from_supplies_alpha_when_alpha0_is_omitted(tmp_path, capsys):
    seed = tmp_path / "gpu-seed"
    seed.mkdir()
    mesh = str(Path("mesh.msh.h5").resolve())
    (seed / "result.json").write_text(json.dumps({
        "alpha_deg": 6.25, "mesh": mesh, "status": "converged"}),
        encoding="utf-8")

    assert run_case.main([
        "C5", "--mesh", mesh, "--seed-from", str(seed),
        "--out", str(tmp_path / "cpu-warm"), "--dry"]) == 0
    setup = json.loads(capsys.readouterr().out)
    assert setup["alpha0_deg"] == pytest.approx(6.25)


class _FakeReports(dict):
    def create(self, name):
        self[name] = SimpleNamespace()


class _ConfigureSession:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.read_mesh_path = None
        self.read_data_path = None
        self.read_interpolation_path = None
        self.hybrid_calls = 0
        self.standard_calls = 0
        self.fail_interpolation_read = False
        self.transcript = ""
        pff = SimpleNamespace(momentum=SimpleNamespace(),
                              thermal=SimpleNamespace(),
                              turbulence=SimpleNamespace())
        self.setup = SimpleNamespace(
            general=SimpleNamespace(solver=SimpleNamespace(),
                                    operating_conditions=SimpleNamespace()),
            models=SimpleNamespace(
                energy=SimpleNamespace(),
                viscous=SimpleNamespace(
                    k_omega=SimpleNamespace(k_omega_low_re_correction=False,
                                             k_omega_shear_correction=False,
                                             coefficients=SimpleNamespace()),
                    near_wall_treatment=SimpleNamespace(wall_treatment=None),
                    turbulence_expert=SimpleNamespace(k_omega_vorticity_based_production=False))),
            materials=SimpleNamespace(fluid={
                "air": SimpleNamespace(density=SimpleNamespace(),
                                       viscosity=SimpleNamespace(
                                           sutherland=SimpleNamespace(
                                               c1=None, c2=None,
                                               reference_viscosity=None,
                                               reference_temperature=None,
                                               effective_temperature=None)))}),
            boundary_conditions=SimpleNamespace(
                pressure_far_field={"farfield": pff}, symmetry={"symmetry": None},
                wall={name: SimpleNamespace(
                    thermal=SimpleNamespace(temperature=None))
                    for name in ("mainwing:1", "fuselage-1", "horstab", "vertstab",
                                 "engine", "pylon", "nacelle_duct", "fairing")}),
            reference_values=SimpleNamespace(),
            cell_zone_conditions={"fluid": object()})
        reports = SimpleNamespace(lift=_FakeReports(), drag=_FakeReports(),
                                  moment=_FakeReports())
        equations = {name: SimpleNamespace() for name in ("continuity", "energy")}
        def hybrid_initialize():
            self.hybrid_calls += 1
            self.events.append(("hybrid_initialize", None))

        def standard_initialize():
            self.standard_calls += 1
            self.events.append(("standard_initialize", None))

        def compute_defaults(**kwargs):
            self.events.append(("compute_defaults", kwargs))

        auto_save = SimpleNamespace()
        solution = SimpleNamespace(
            report_definitions=reports,
            monitor=SimpleNamespace(residual=SimpleNamespace(
                options=SimpleNamespace(), equations=equations)),
            methods=SimpleNamespace(
                spatial_discretization=SimpleNamespace(
                    gradient_scheme="least-square-cell-based"),
                p_v_coupling=SimpleNamespace(flux_type="Roe-FDS")),
            controls=SimpleNamespace(),
            calculation_activity=SimpleNamespace(
                auto_save=auto_save),
            initialization=SimpleNamespace(
                hybrid_initialize=hybrid_initialize,
                compute_defaults=compute_defaults,
                standard_initialize=standard_initialize))

        self.settings = SimpleNamespace(
            setup=self.setup,
            file=SimpleNamespace(read_mesh=self._read_mesh,
                                 read_case=self._read_case,
                                 read_data=self._read_settings_data),
            solution=solution)
        self.setup.get_state = lambda: _fake_settings_state(self.setup)
        solution.get_state = lambda: _fake_settings_state(solution)
        self.tui = SimpleNamespace(
            solve=SimpleNamespace(
                initialize=SimpleNamespace(
                    fmg_initialization=lambda: self.events.append(("fmg", None))),
                set=SimpleNamespace(
                    solution_steering=lambda flow_type: self.events.append(
                        ("steer", flow_type)),
                    convergence_acceleration_for_stretched_meshes=lambda: (
                        self.events.append(("casm", None))))),
            mesh=SimpleNamespace(modify_zones=SimpleNamespace(
                list_zones=lambda: "1 mainwing:1\n2 fuselage-1\n3 horstab\n"
                "4 vertstab:1\n5 engine\n6 pylon-1\n7 nacelle_duct\n"
                "8 fairing:1\n9 farfield\n")),
            file=SimpleNamespace(
                read_data=self._read_data,
                interpolate=SimpleNamespace(read_data=self._read_interpolation,
                                            write_data=self._write_interpolation)))

    def _read_mesh(self, file_name):
        self.read_mesh_path = file_name
        self.events.append(("read_mesh", file_name))

    def _read_data(self, file_name):
        self.read_data_path = file_name

    def _read_case(self, file_name):
        self.events.append(("read_case", file_name))

    def _read_settings_data(self, file_name):
        self.events.append(("read_settings_data", file_name))

    def _read_interpolation(self, file_name):
        self.read_interpolation_path = file_name
        self.events.append(("read_interpolation", file_name))
        if self.fail_interpolation_read:
            raise RuntimeError("interpolation read failed")

    def _write_interpolation(self, *args):
        self.events.append(("write_interpolation", args))
        Path(args[0]).write_text("fake interpolation", encoding="utf-8")


def test_configure_resolves_paths_and_sets_static_pressure_reference(tmp_path,
                                                                     monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = _ConfigureSession()
    backend = FluentBackend(session)
    courant = []
    backend.set_courant = courant.append
    condition = condition_for_case("C5")
    backend.configure("mesh.msh.h5", condition, "restart.dat.h5")
    assert session.read_mesh_path == str((tmp_path / "mesh.msh.h5").resolve())
    assert ("read_settings_data", str((tmp_path / "restart.dat.h5").resolve())) in (
        session.events)
    assert session.setup.general.operating_conditions.operating_pressure == 0
    assert session.setup.reference_values.pressure == condition["pressure_Pa"]
    assert session.setup.general.solver.type == "density-based-implicit"
    assert session.settings.solution.methods.spatial_discretization.discretization_scheme == {"amg-c": "second-order-upwind", "k": "second-order-upwind", "omega": "second-order-upwind"}
    assert courant == [2.0]
    pb_session = _ConfigureSession()
    pb = FluentBackend(pb_session)
    pb.solver_name = "gpu-pb"
    pb.set_courant = courant.append
    pb.configure("mesh.msh.h5", condition)
    assert (pb.settings.setup.general.solver.type,
            pb.settings.solution.methods.p_v_coupling.flow_scheme,
            pb.settings.solution.methods.spatial_discretization.discretization_scheme,
            courant) == ("pressure-based", "Coupled", run_case.PB_DISCRETIZATION, [2.0])


def test_configure_without_interp_from_keeps_hybrid_initialization(tmp_path):
    session = _ConfigureSession()
    backend = FluentBackend(session)
    backend.configure(str(tmp_path / "mesh.msh.h5"), condition_for_case("C5"))
    assert session.hybrid_calls == 1
    assert backend.warm_start_status is None


def test_missing_transcript_returns_unavailable():
    session = _ConfigureSession()
    session.transcript = None
    backend = FluentBackend(session)

    assert backend.solver_audit("P10H1-no-transcript") == "unavailable"
    assert backend.solver_transcript_available is False
    assert backend.native_gpu_solver_active is None


def test_final_transcript_audit_includes_post_configure_warning(tmp_path):
    backend = FakeBackend()
    iterate = backend.iterate

    def emit_final_transcript(count):
        iterate(count)
        backend.configure_session.transcript = (
            "normal\nignored after configure\nGPU Solver enabled\n")

    backend.iterate = emit_final_transcript
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        solver="gpu-pb", precision="single", min_final_iters=600)
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))

    assert result["solver_audit"] == ["ignored after configure"]
    assert saved["solver_audit"] == ["ignored after configure"]
    assert saved["solver_transcript_available"] is True
    assert saved["gpu_native_solver_active"] is True


@pytest.mark.parametrize(("audit", "native"), [
    (None, True), ([], False), (["unsupported setting"], True)])
def test_gpu_audit_failure_returns_nonzero_and_is_not_converged(
        tmp_path, monkeypatch, audit, native):
    backend = FakeBackend()
    backend.solver_audit = lambda _cwd: audit
    backend.native_gpu_solver_active = native
    backend.set_transcript_baseline = lambda _names: None
    backend.close = lambda: None
    session = SimpleNamespace(
        connection_properties=SimpleNamespace(cortex_pid=None, fluent_host_pid=None),
        _process=SimpleNamespace(pid=987654321))
    monkeypatch.setattr(run_case, "_start_metrics",
                        lambda _out: (object(), "metrics.stop"))
    monkeypatch.setattr(run_case, "_stop_metrics", lambda _process, _stop: None)
    monkeypatch.setattr(run_case, "launch_fluent", lambda *args: session)
    monkeypatch.setattr(run_case, "FluentBackend", lambda _session: backend)
    monkeypatch.setattr(run_case.psutil, "pid_exists", lambda _pid: False)
    out = tmp_path / "gpu-run"

    result_code = run_case.main([
        "C5", "--mesh", "mesh.msh.h5", "--out", str(out), "--solver", "gpu-pb",
        "--fixed-alpha", "2", "--min-final-iters", "600"])
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))

    assert result_code != 0
    assert saved["status"] == "audit_failed"
    assert saved["converged"] is False


def test_gpu_initialization_and_cpu_hybrid_behavior():
    condition = condition_for_case("C5")
    gpu_session = _ConfigureSession()
    gpu = FluentBackend(gpu_session)
    gpu.solver_name = "gpu-pb"
    gpu.configure("mesh.msh.h5", condition)

    assert gpu_session.standard_calls == 1
    assert gpu_session.hybrid_calls == 0
    assert ("compute_defaults", {
        "from_zone_type": "pressure-far-field",
        "from_zone_name": "farfield",
        "phase": "mixture",
    }) in gpu_session.events

    restart = Path("restart.dat.h5")
    restart_session = _ConfigureSession()
    restart_backend = FluentBackend(restart_session)
    restart_backend.solver_name = "gpu-pb"
    restart_backend.configure("mesh.msh.h5", condition, str(restart))
    assert ("read_settings_data", str(restart.resolve())) in restart_session.events
    assert restart_session.standard_calls == 0
    assert restart_session.hybrid_calls == 0

    cpu_session = _ConfigureSession()
    FluentBackend(cpu_session).configure("mesh.msh.h5", condition)
    assert cpu_session.hybrid_calls == 1
    assert cpu_session.standard_calls == 0


def test_autosave_settings_use_requested_interval_and_retention():
    session = _ConfigureSession()
    backend = FluentBackend(session)

    output_dir = Path("autosave-test-output")
    backend.configure_autosave(output_dir, every=500)

    settings = session.settings.solution.calculation_activity.auto_save
    assert settings.data_frequency == 500
    assert settings.root_name == str((output_dir / "autosave").resolve())
    assert settings.retain_most_recent_files is True
    assert settings.max_files == 2


def test_result_json_records_solver_and_precision(tmp_path):
    backend = FakeBackend()
    run_case.run_solve(backend, "C5", "mesh.msh.h5", tmp_path,
                       fixed_alpha=2.0, min_final_iters=600)

    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["solver"] == "cpu-db"
    assert saved["precision"] == "double"


def test_solver_settings_snapshot_uses_fake_settings_api(tmp_path):
    backend = FakeBackend()
    run_case.run_solve(backend, "C5", "mesh.msh.h5", tmp_path,
                       fixed_alpha=2.0, min_final_iters=600)

    saved = json.loads((tmp_path / "solver_settings.json").read_text(
        encoding="utf-8"))
    setup = saved["after_last_iteration"]["setup"]
    solution = saved["after_last_iteration"]["solution"]
    viscous = setup["models"]["viscous"]
    air = setup["materials"]["fluid"]["air"]
    assert solution["methods"]["spatial_discretization"][
        "gradient_scheme"] == "least-square-cell-based"
    assert solution["methods"]["p_v_coupling"]["flux_type"] == "Roe-FDS"
    assert viscous["k_omega"]["k_omega_shear_correction"] is False
    assert air["viscosity"]["sutherland"]["c1"] is None
    assert setup["reference_values"]["pressure"] == condition_for_case("C5")[
        "pressure_Pa"]
    assert "thermal" in setup["boundary_conditions"]["pressure_far_field"][
        "farfield"]
    assert setup["boundary_conditions"]["wall"]["mainwing:1"][
        "thermal"]["temperature"] is None
    assert saved["after_setup"]
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["solver_settings"] == "solver_settings.json"


def test_accel_none_preserves_configure_and_iterate_sequence(tmp_path):
    backend = FakeBackend()
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        min_final_iters=600, accel=run_case.parse_accel("none"))
    events = backend.configure_session.events
    assert events[:4] == [
        ("read_mesh", str(Path("mesh.msh.h5").resolve())),
        ("courant", 2.0), ("hybrid_initialize", None), ("set_alpha", 2.0)]
    # the settle fit needs its 600-iteration window after the first sample: 650 iterations on flat data
    default_courants = (2.0, 2.0, 5.0, 5.0, 5.0, 5.0,
                        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    assert events[4:] == [event for courant in default_courants
                          for event in (("courant", courant), ("iterate", 50))]
    assert result["accel"] == []


def test_fmg_runs_after_cold_hybrid_init_and_skips_init_data(tmp_path):
    cold = FakeBackend()
    run_case.run_solve(cold, "C5", "mesh.msh.h5", tmp_path / "cold",
                       fixed_alpha=2.0, min_final_iters=600, accel=("fmg",))
    cold_events = cold.configure_session.events
    assert cold_events.index(("hybrid_initialize", None)) < cold_events.index(("fmg", None))
    assert cold_events.index(("fmg", None)) < next(
        i for i, event in enumerate(cold_events) if event[0] == "iterate")

    warm = FakeBackend()
    run_case.run_solve(warm, "C5", "mesh.msh.h5", tmp_path / "warm",
                       fixed_alpha=2.0, init_data="restart.dat.h5",
                       min_final_iters=600, accel=("fmg",))
    assert ("read_settings_data", str(Path("restart.dat.h5").resolve())) in (
        warm.configure_session.events)
    assert not any(event[0] == "fmg" for event in warm.configure_session.events)


def test_steer_suppresses_driver_courant_and_casm_precedes_iterate(tmp_path):
    backend = FakeBackend()
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        min_final_iters=600, accel=("steer", "casm"))
    events = backend.configure_session.events
    first_iterate = next(i for i, event in enumerate(events)
                         if event[0] == "iterate")
    assert events.index(("steer", "transonic")) < first_iterate
    assert events.index(("casm", None)) < first_iterate
    assert not any(event[0] == "courant" for event in events)
    assert backend.calls == ["save_final", "export_ensight"]
    assert result["accel"] == ["steer", "casm"]
    assert result["courant_ramp"] == []
    assert result["courant_ramp_control"] == "solution-steering"


def test_accel_rejects_non_cpu_db_before_fluent_launch(monkeypatch):
    launches = []
    monkeypatch.setattr(run_case, "launch_fluent",
                        lambda *args, **kwargs: launches.append((args, kwargs)))
    with pytest.raises(SystemExit, match="only supported with --solver cpu-db"):
        run_case.main(["C5", "--mesh", "mesh.msh.h5", "--out", "out",
                       "--solver", "gpu-pb", "--accel", "fmg"])
    assert not launches


def test_courant_ramp_parse_validation_and_use(tmp_path):
    ramp = run_case.parse_courant_ramp("100:3,300:6,6000:12")
    backend = FakeBackend()
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        min_final_iters=600, courant_ramp=ramp)
    assert backend.calls[:4] == [
        ("courant", 3.0), ("courant", 3.0),
        ("courant", 3.0), ("courant", 6.0)]
    assert result["courant_ramp"] == [
        {"through_iteration": 100, "value": 3.0},
        {"through_iteration": 300, "value": 6.0},
        {"through_iteration": 6000, "value": 12.0}]
    with pytest.raises(Exception, match="strictly increasing"):
        run_case.parse_courant_ramp("100:2,100:3,6000:4")
    with pytest.raises(Exception, match="at least 6000"):
        run_case.parse_courant_ramp("100:2,300:3,5999:4")


def test_interp_from_writes_then_reads_interpolation_and_records_source(tmp_path,
                                                                       monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("final.cas.h5", "final.dat.h5"):
        (source_dir / name).write_bytes(b"fake source")
    output_dir = tmp_path / "target"
    events = []
    source_session = SimpleNamespace(
        connection_properties=SimpleNamespace(cortex_pid=201,
                                               fluent_host_pid=202),
        _process=SimpleNamespace(pid=203),
        settings=SimpleNamespace(file=SimpleNamespace(
            read_case=lambda **kwargs: events.append(("read_case", kwargs["file_name"])),
            read_data=lambda **kwargs: events.append(("read_settings_data", kwargs["file_name"]))
        )),
        tui=SimpleNamespace(file=SimpleNamespace(interpolate=SimpleNamespace(
            write_data=lambda *args: (
                events.append(("write_interpolation", args)),
                Path(args[0]).write_text("fake interpolation", encoding="utf-8"))))),
        exit=lambda: events.append(("exit", None)))
    monkeypatch.setattr("b737wing.cfd.solve.fluent_io.launch_fluent",
                        lambda processor_count, cwd: source_session)
    interpolation_file = write_interpolation_data(source_dir, output_dir, 4)
    backend = FakeBackend()
    backend.configure_events = events
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", output_dir, fixed_alpha=2.0,
        interp_from=source_dir, warm_start_file=interpolation_file)
    saved = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    event_names = [event[0] for event in events]
    assert event_names.index("write_interpolation") < event_names.index("exit")
    assert event_names.index("exit") < event_names.index("read_interpolation")
    write_event = next(event for event in events
                       if event[0] == "write_interpolation")
    assert write_event[1] == (
        str(interpolation_file), "all",
        ["pressure", "velocity", "temperature", "k", "omega"])
    assert backend.configure_session.read_interpolation_path == str(interpolation_file)
    assert result["warm_start"] == str(source_dir.resolve())
    assert saved["warm_start"] == str(source_dir.resolve())
    assert json.loads((output_dir / "fluent_pids.json").read_text(
        encoding="utf-8"))["pids"] == [201, 202, 203]


def test_interp_read_failure_falls_back_to_hybrid_and_records_reason(tmp_path):
    backend = FakeBackend()
    backend.fail_interpolation_read = True
    result = run_case.run_solve(
        backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0,
        interp_from=tmp_path / "source", warm_start_file=tmp_path / "warmstart.ip")
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert backend.configure_session.hybrid_calls == 1
    assert result["warm_start"] == "failed: RuntimeError: interpolation read failed"
    assert saved["warm_start"] == result["warm_start"]


def test_walls_export_includes_zone_name_for_each_face(tmp_path):
    pytest.importorskip("ansys.fluent.core")
    session = SimpleNamespace(
        settings=SimpleNamespace(setup=SimpleNamespace(),
                                 solution=SimpleNamespace(
                                     report_definitions=SimpleNamespace())))

    class FakeFieldData:
        def __init__(self):
            self.call_count = 0
            self.requests = []

        def get_field_data(self, request):
            self.call_count += 1
            self.requests.append(request)
            surfaces = ["mainwing:1", "fuselage-1"]
            if self.call_count == 1:
                return {surface: SimpleNamespace(
                    face_centroids=np.array([[1.0, 2.0, 3.0]]),
                    # Fluent-like: right direction, magnitude not the face area
                    face_normals=np.array([[0.34, 0.0, 0.0]]),
                    vertices=np.array([[1.0, 1.5, 2.5], [1.0, 2.5, 2.5],
                                       [1.0, 2.5, 3.5], [1.0, 1.5, 3.5]]),
                    connectivity=[np.array([0, 1, 2, 3])])
                        for surface in surfaces}
            return {surface: np.array([float(self.call_count)])
                    for surface in surfaces}

    session.fields = SimpleNamespace(field_data=FakeFieldData())
    backend = FluentBackend(session)
    backend.groups = {"total": ["mainwing:1", "fuselage-1"]}
    output = tmp_path / "walls.csv"
    backend.export_walls(str(output))
    with output.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames[:4] == ["x", "y", "z", "zone"]
        rows = list(reader)
        assert [row["zone"] for row in rows] == ["mainwing:1", "fuselage-1"]
        assert reader.fieldnames[-3:] == [
            "area-vector-x", "area-vector-y", "area-vector-z"]
        assert [rows[0][column] for column in reader.fieldnames[-3:]] == [
            "1.0", "0.0", "0.0"]
    requested_types = backend.session.fields.field_data.requests[0].data_types
    requested = {str(value).lower().replace("_", "") for value in requested_types}
    assert any("facesnormal" in value for value in requested)
    assert any("vertices" in value for value in requested)
    assert any("facesconnectivity" in value for value in requested)


def test_ensight_export_requests_final_volume_fields(tmp_path):
    calls = []

    def fake_export(**kwargs):
        calls.append(kwargs)
        Path(kwargs["file_name"] + ".encas").write_text(
            'GEOMETRY\nmodel: "_run/ensight/final.geo"\n'
            'scalar per node: Mach_Number "_run/ensight/final.scl6"\n', encoding="utf-8")

    setup = SimpleNamespace(cell_zone_conditions={"fluid": object()})
    file_settings = SimpleNamespace(export=SimpleNamespace(ensight_gold=fake_export))
    session = SimpleNamespace(settings=SimpleNamespace(
        setup=setup, solution=SimpleNamespace(report_definitions=SimpleNamespace()),
        file=file_settings))
    backend = FluentBackend(session)
    destination = tmp_path / "ensight"
    backend.export_ensight(destination)
    assert destination.is_dir()
    assert calls == [{
        "file_name": str(destination / "final"),
        "quantities": ["x-velocity", "y-velocity", "z-velocity", "velocity-magnitude",
                       "pressure", "mach-number", "pressure-coefficient", "density"],
        "binary_format": True,
        "cellzones": ["fluid"],
        "interior_zone_surfaces": [],
        "cell_centered": False,
    }]
    portable = (destination / "final.case").read_text(encoding="utf-8")
    assert '"final.geo"' in portable and '"final.scl6"' in portable


class TestPressureSolverAndSeed:
    def test_gpu_launch_precision(self, tmp_path, monkeypatch):
        original = builtins.__import__
        fake = SimpleNamespace(fluent=SimpleNamespace(core=SimpleNamespace(
            launch_fluent=lambda **kwargs: kwargs)))
        monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: fake if name == "ansys.fluent.core" else original(name, *a, **k))
        options = launch_fluent(8, str(tmp_path), "gpu-pb", "single")
        assert (options["gpu"], options["processor_count"], options["precision"]) == (True, 1, "single")
        class Reject:
            discretization_scheme = SimpleNamespace(available_keys=("pressure", "momentum"))
            def __setattr__(self, _, value):
                raise RuntimeError("invalid key")
        session = _ConfigureSession()
        session.settings.solution.methods.p_v_coupling = SimpleNamespace()
        session.settings.solution.methods.spatial_discretization = Reject()
        backend = FluentBackend(session)
        backend.solver_name = "gpu-pb"
        with pytest.raises(ValueError, match="momentum, pressure"):
            backend.configure("mesh.msh.h5", condition_for_case("C5"))
    def test_seed_and_minimum_iteration_paths(self, tmp_path):
        (tmp_path / "result.json").write_text(json.dumps({"alpha_deg": 0, "mesh": "mesh.msh.h5", "status": "converged"}))
        history = tmp_path / "history.csv"
        history.write_text("alpha,CL\n-1,0.16\n0,0.27\n")
        seed = run_case.read_seed(tmp_path)
        plain = secant_search(lambda a: 0.11 * a + 0.27, 0.4895, 0.0)
        seeded = secant_search(lambda a: 0.11 * a + 0.27, 0.4895, seed["alpha0_deg"], initial_slope=seed["slope"])
        assert seed["alpha0_deg"] == 0 and seed["slope"] == pytest.approx(0.11) and seeded["steps"] <= plain["steps"] and abs(seeded["cl"] - 0.4895) <= 0.003
        history.write_text("alpha,CL\n0,0.27\n")
        assert run_case.read_seed(tmp_path)["slope"] is None
        backend = FakeBackend()
        session = SimpleNamespace(
            settings=SimpleNamespace(
                setup=SimpleNamespace(),
                solution=SimpleNamespace(report_definitions=SimpleNamespace())),
            transcript="ignored option\nconverted value\nunsupported feature\nnormal")
        backend.export_walls = backend.export_ensight = lambda _: (_ for _ in ()).throw(RuntimeError("export blocked"))
        backend.solver_audit = lambda path: FluentBackend(session).solver_audit(path)
        result = run_case.run_solve(backend, "C5", "mesh.msh.h5", tmp_path, fixed_alpha=2.0, solver="gpu-pb", min_final_iters=600)
        saved = json.loads((tmp_path / "result.json").read_text())
        # GPU audit warnings fail closed (P10AUDIT3 B03)
        assert (result["status"] == "audit_failed" and result["converged"] is False
                and result["iterations"] == 650
                and saved["min_final_iterations"] == 600
                and backend.calls == ["save_final"])
        assert (all(value.startswith("skipped:") for value in saved["exports"].values()), saved["solver_audit"]) == (True, ["ignored option", "converted value", "unsupported feature"])

    def test_seed_uses_final_alpha_occurrences_and_tolerates_missing_history(self, tmp_path):
        (tmp_path / "result.json").write_text(
            json.dumps({"alpha_deg": 0, "mesh": "mesh.msh.h5", "status": "converged"}),
            encoding="utf-8")
        history = tmp_path / "history.csv"
        history.write_text("alpha,CL\n-1,0.1\n0,0.2\n1,0.3\n0,0.4\n",
                           encoding="utf-8")

        seed = run_case.read_seed(tmp_path)

        assert seed["slope"] == pytest.approx(-0.1)
        history.unlink()
        assert run_case.read_seed(tmp_path)["slope"] is None


def test_main_closes_session_on_keyboard_interrupt_and_records_pids(
        tmp_path, monkeypatch):
    events = []
    session = SimpleNamespace(
        connection_properties=SimpleNamespace(cortex_pid=101,
                                               fluent_host_pid=102),
        _process=SimpleNamespace(pid=103),
        exit=lambda: events.append("session_exit"))

    class Backend:
        def __init__(self, _session):
            pass

        def set_transcript_baseline(self, _names):
            pass

        def close(self):
            events.append("backend_close")

    monkeypatch.setattr(run_case, "_start_metrics",
                        lambda _out: (object(), "metrics.stop"))
    monkeypatch.setattr(run_case, "_stop_metrics",
                        lambda _process, _stop: events.append("metrics_stop"))
    monkeypatch.setattr(run_case, "launch_fluent", lambda *args: session)
    monkeypatch.setattr(run_case, "FluentBackend", Backend)
    monkeypatch.setattr(run_case.psutil, "pid_exists", lambda _pid: False)

    def interrupt(*_args, **_kwargs):
        pids = json.loads((tmp_path / "fluent_pids.json").read_text(
            encoding="utf-8"))
        assert pids["pids"] == [101, 102, 103]
        raise KeyboardInterrupt

    monkeypatch.setattr(run_case, "run_solve", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_case.main(["C5", "--mesh", "mesh.msh.h5", "--out", str(tmp_path)])

    assert events == ["backend_close", "metrics_stop"]


def test_main_kills_recorded_pid_tree_after_close(tmp_path, monkeypatch):
    root_pid = 987650001
    child_pid = 987650002
    killed = []

    class Backend:
        def __init__(self, _session):
            pass

        def set_transcript_baseline(self, _names):
            pass

        def close(self):
            pass

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            assert recursive is True
            return [FakeProcess(child_pid)] if self.pid == root_pid else []

        def kill(self):
            killed.append(self.pid)

    session = SimpleNamespace(
        connection_properties=SimpleNamespace(cortex_pid=root_pid,
                                               fluent_host_pid=None),
        _process=SimpleNamespace(pid=None))

    def fake_solve(_backend, _case, _mesh, out, *_args, **_kwargs):
        result = {"status": "converged", "alpha_deg": 2.0,
                  "CL": 0.4895, "CD": 0.04}
        (Path(out) / "result.json").write_text(
            json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(run_case, "FLUENT_CLOSE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(run_case, "_start_metrics",
                        lambda _out: (object(), "metrics.stop"))
    monkeypatch.setattr(run_case, "_stop_metrics", lambda _process, _stop: None)
    monkeypatch.setattr(run_case, "launch_fluent", lambda *args: session)
    monkeypatch.setattr(run_case, "FluentBackend", Backend)
    monkeypatch.setattr(run_case, "run_solve", fake_solve)
    monkeypatch.setattr(run_case.psutil, "pid_exists", lambda _pid: True)
    monkeypatch.setattr(run_case.psutil, "Process", FakeProcess)
    monkeypatch.setattr(run_case.psutil, "wait_procs",
                        lambda _processes, timeout: ([], []))

    out = tmp_path / "pid-run"
    run_case.main(["C5", "--mesh", "mesh.msh.h5", "--out", str(out)])

    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert set(killed) == {root_pid, child_pid}
    assert saved["close_clean"] is False
    assert saved["killed_pids"] == [root_pid, child_pid]


def test_exhausted_search_final_step_requires_all_coefficients(monkeypatch):
    target = condition_for_case("C5")["target_cl"]

    class OffTargetDriftingCMBackend:
        def __init__(self):
            self.iterations = 0

        def set_alpha(self, _alpha):
            pass

        def set_courant(self, _value):
            pass

        def iterate(self, count):
            self.iterations += count

        def sample(self):
            cm = -0.02 + min(self.iterations, 1200) * 0.000002
            cl = target - 0.1
            return {"CL": cl, "CD": 0.04, "CM": cm, "mass_imbalance": 0.0,
                    "all_reports": {"total": {"CL": cl, "CD": 0.04, "CM": cm}}}

    monkeypatch.setattr(run_case, "secant_next", lambda *a, **k: None)
    _, state, status, convergence, _ = run_case._solve_alpha(
        OffTargetDriftingCMBackend(), target, 2.0, None, [],
        min_final_iters=600)

    assert status == "cl_not_matched"
    assert convergence["settle_on"] == "all"
    assert convergence["settle"]["CM"]["settled"] is True
    assert state["iterations"] > 1200
