import csv
import json
from pathlib import Path

import pytest

from b737wing.cfd.solve import run_case
from b737wing.cfd.solve import compare_runs as compare_module
from b737wing.cfd.solve import sweep


class SweepBackend:
    def __init__(self, output):
        self.output = Path(output)
        self.configure_count = 0
        self.alphas = []
        self.iterations = 0
        self.sample_index = 0

    def configure(self, mesh, condition, init_data=None, **kwargs):
        self.configure_count += 1
        self.mesh = mesh
        self.condition = condition

    def set_alpha(self, alpha):
        self.alphas.append(alpha)
        if self.output.joinpath("polar.csv").is_file():
            with self.output.joinpath("polar.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == len(self.alphas) - 1
        self.iterations = 0
        self.sample_index = 0

    def set_courant(self, value):
        pass

    def iterate(self, count):
        self.iterations += count

    def sample(self):
        self.sample_index += 1
        drifting = self.alphas[-1] == 1.0
        cd = 0.04 + (self.sample_index * 1e-4 if drifting else 0.0)
        component = {"CL": 0.1, "CD": 0.01, "CM": -0.001}
        all_reports = {name: dict(component) for name in
                       (*run_case.REPORT_COMPONENTS, "total", "total_ex_duct")}
        return {"CL": 0.49, "CD": cd, "CM": -0.02, "CD_ex_duct": 0.03,
                "components": {name: dict(component) for name in run_case.REPORT_COMPONENTS},
                "all_reports": all_reports, "mass_imbalance": 0.0}

    def export_walls(self, path):
        Path(path).write_text("x,y,z,zone\n", encoding="utf-8")

    def save_final(self, output_dir):
        for name in ("final.cas.h5", "final.dat.h5"):
            (Path(output_dir) / name).write_bytes(b"fake")


def test_sweep_appends_ordered_rows_and_continues_after_not_converged(tmp_path):
    backend = SweepBackend(tmp_path)
    result = sweep.run_sweep(backend, "C5", "mesh.msh.h5", tmp_path,
                             [0.0, 1.0, 2.0])

    with (tmp_path / "polar.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["alpha"]) for row in rows] == [0.0, 1.0, 2.0]
    assert [row["status"] for row in rows] == ["converged", "not_converged", "converged"]
    assert backend.alphas == [0.0, 1.0, 2.0]
    assert backend.configure_count == 1
    assert len(list(csv.DictReader((tmp_path / "history.csv").open(
        newline="", encoding="utf-8")))) == result["iterations"] // 50
    assert (tmp_path / "sweep.json").is_file()
    assert all((tmp_path / name).is_file()
               for name in ("final.cas.h5", "final.dat.h5"))


def _run_dir(root, name, result, walls=True):
    run = root / name
    run.mkdir()
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if walls:
        (run / "walls.csv").write_text("x,y,z,zone\n", encoding="utf-8")
    return run


def _mock_shocks(monkeypatch, position=0.55):
    monkeypatch.setattr(
        compare_module, "_shock_positions_for_run",
        lambda run_dir, result, mesh: {eta: position for eta in compare_module.SHOCK_ETAS})


def _result(cd=0.04):
    return {"alpha_deg": 2.0, "CL": 0.49, "CD": cd, "CM": -0.02,
            "CD_ex_duct": 0.03, "solver_audit": []}


def test_identical_runs_pass_both_gates(tmp_path, monkeypatch):
    _mock_shocks(monkeypatch)
    mesh = tmp_path / "mesh.msh.h5"
    mesh.write_bytes(b"fake mesh")
    reference = _run_dir(tmp_path, "ref", dict(_result(), mesh=str(mesh)))
    test = _run_dir(tmp_path, "test", dict(_result(), mesh=str(mesh)))

    report = compare_module.compare_runs(reference, test, out=tmp_path / "compare.json")

    assert report["gates"]["warm_start"]["verdict"] == "pass"
    assert report["gates"]["gpu_only"]["verdict"] == "pass"
    assert json.loads((tmp_path / "compare.json").read_text(encoding="utf-8")) == report


def test_dcd_08_count_fails_warm_start_but_passes_gpu_numerics(tmp_path, monkeypatch):
    _mock_shocks(monkeypatch)
    mesh = tmp_path / "mesh.msh.h5"
    mesh.write_bytes(b"fake mesh")
    reference = _run_dir(tmp_path, "ref", dict(_result(), mesh=str(mesh)))
    test = _run_dir(tmp_path, "test", dict(_result(0.04008), mesh=str(mesh)))

    report = compare_module.compare_runs(reference, test, mesh=mesh,
                                         out=tmp_path / "compare.json")

    assert report["dCD_counts"] == pytest.approx(0.8)
    assert report["gates"]["warm_start"]["verdict"] == "fail"
    assert report["gates"]["gpu_only"]["verdict"] == "pass"


def test_shock_locator_finds_synthetic_negative_cp_step():
    x_c = [0.05, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.80, 0.95]
    cp = [-0.4, -1.2, -2.0, -1.7, -1.4, -3.0, -2.8, -2.5, -2.2]

    assert compare_module.locate_shock_x_c(x_c, cp) == pytest.approx(0.55, abs=0.01)


def test_shock_locator_finds_positive_recovery_after_suction_minimum():
    x_c = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    cp = [-0.5, -1.2, -2.5, -2.0, -1.0, -0.8, -0.7]

    assert compare_module.locate_shock_x_c(x_c, cp) == pytest.approx(0.50)


def test_absent_shock_keeps_coefficient_gates_and_fails_gpu_reason(tmp_path, monkeypatch):
    assert compare_module.locate_shock_x_c(
        [0.10, 0.20, 0.30, 0.40], [-1.0, -2.0, -2.5, -2.5]) == "absent"
    _mock_shocks(monkeypatch)
    monkeypatch.setattr(compare_module, "_shock_positions_for_run",
                        lambda run_dir, result, mesh: {
                            eta: ("absent" if eta == 0.5 else 0.55)
                            for eta in compare_module.SHOCK_ETAS})
    mesh = tmp_path / "mesh.msh.h5"
    mesh.write_bytes(b"mesh")
    reference = _run_dir(tmp_path, "ref", dict(_result(), mesh=str(mesh)))
    test = _run_dir(tmp_path, "test", dict(_result(), mesh=str(mesh)))

    report = compare_module.compare_runs(reference, test, out=tmp_path / "compare.json")

    assert report["shock_checks"]["0.50"]["status"] == "absent"
    assert report["gates"]["warm_start"]["verdict"] == "pass"
    assert report["gates"]["gpu_only"]["verdict"] == "fail"
    assert "shock comparison is incomplete" in report["gates"]["gpu_only"]["reason"]
    assert "abs_dCD_counts" in report["gates"]["gpu_only"]["checks"]


def test_shock_locator_keeps_suction_peak_before_candidate_interval():
    x_c = [0.0, 0.03, 0.08, 0.20, 0.40, 0.50, 0.55, 0.60, 0.80, 0.95, 1.0]
    cp = [-1.0, -2.5, -2.3, -1.9, -1.4, -1.2, -3.0, -2.8, -2.5, -2.2, -2.0]

    assert compare_module.locate_shock_x_c(x_c, cp) == pytest.approx(0.55, abs=0.01)


def test_compare_without_walls_skips_shocks_and_marks_gpu_incomplete(tmp_path):
    reference = _run_dir(tmp_path, "ref", _result(), walls=False)
    test = _run_dir(tmp_path, "test", _result(), walls=False)

    report = compare_module.compare_runs(reference, test, out=tmp_path / "compare.json")

    assert all(row["status"].startswith("skipped: ")
               for row in report["shock_checks"].values())
    assert report["gates"]["warm_start"]["verdict"] == "pass"
    assert report["gates"]["gpu_only"]["verdict"] == "fail"
    assert "shock comparison is incomplete" in report["gates"]["gpu_only"]["reason"]


def test_warm_start_fails_when_alpha_differs_by_more_than_one_microdegree(
        tmp_path, monkeypatch):
    _mock_shocks(monkeypatch)
    reference = _run_dir(tmp_path, "ref", _result())
    test = _run_dir(tmp_path, "test", dict(_result(), alpha_deg=2.000002))

    report = compare_module.compare_runs(reference, test, out=tmp_path / "compare.json")

    assert report["gates"]["warm_start"]["verdict"] == "fail"
    assert "alpha_deg values differ" in report["gates"]["warm_start"]["reason"]


def test_courant_ab_fails_on_point_two_count_cd_difference_and_reports_settle_iterations(
        tmp_path, monkeypatch):
    _mock_shocks(monkeypatch)
    reference_result = dict(_result(), convergence_numbers={
        "settle": {"settled": True}, "final_alpha_iterations": 1500})
    test_result = dict(_result(0.04002), convergence_numbers={
        "settle": {"settled": True}, "final_alpha_iterations": 1000})
    reference = _run_dir(tmp_path, "ref", reference_result)
    test = _run_dir(tmp_path, "test", test_result)

    report = compare_module.compare_runs(reference, test, out=tmp_path / "compare.json",
                                         mode="courant_ab")

    assert report["gates"]["courant_ab"]["verdict"] == "fail"
    assert report["gates"]["courant_ab"]["checks"]["abs_dCD_counts"] == pytest.approx(0.2)
    assert report["iterations_to_settle"] == {"CFL_10": 1500, "CFL_20": 1000}


def _early_result(cl, cd, alpha=2.0):
    return {"status": "converged", "alpha_deg": alpha, "CL": cl, "CD": cd,
            "solver_audit": [], "gpu_native_solver_active": True}


def test_early_gpu_passes_at_cl_1_point_4_percent_and_cd_1_point_7_percent(tmp_path):
    cpu = _run_dir(tmp_path, "cpu", _early_result(0.5, 0.04))
    gpu = _run_dir(tmp_path, "gpu", _early_result(0.507, 0.04068))

    report = compare_module.compare_runs(cpu, gpu, mode="early_gpu",
                                         out=tmp_path / "early.json")

    gate = report["gates"]["early_gpu"]
    assert gate["verdict"] == "pass"
    assert gate["checks"]["abs_dCL_percent"] == pytest.approx(1.4)
    assert gate["checks"]["abs_dCD_percent"] == pytest.approx(1.7)
    assert gate["checks"]["max_abs_dCL_percent"] == 1.5
    assert gate["checks"]["max_abs_dCD_percent"] == 1.8
    assert gate["reason"] == "all early GPU checks passed"
    assert json.loads((tmp_path / "early.json").read_text(encoding="utf-8")) == report


def test_early_gpu_fails_at_cd_1_point_9_percent(tmp_path):
    cpu = _run_dir(tmp_path, "cpu", _early_result(0.5, 0.04))
    gpu = _run_dir(tmp_path, "gpu", _early_result(0.5, 0.04076))

    report = compare_module.compare_runs(cpu, gpu, mode="early_gpu",
                                         out=tmp_path / "early.json")

    gate = report["gates"]["early_gpu"]
    assert gate["verdict"] == "fail"
    assert gate["checks"]["abs_dCD_percent"] == pytest.approx(1.9)
    assert "relative CD difference exceeds 1.8%" in gate["reason"]


def test_early_gpu_fails_on_alpha_mismatch_and_missing_input(tmp_path):
    cpu = _run_dir(tmp_path, "cpu", _early_result(0.5, 0.04))
    mismatch = _run_dir(tmp_path, "mismatch", _early_result(0.5, 0.04, alpha=2.02))

    mismatch_report = compare_module.compare_runs(cpu, mismatch, mode="early_gpu",
                                                  out=tmp_path / "mismatch.json")
    mismatch_gate = mismatch_report["gates"]["early_gpu"]
    assert mismatch_gate["verdict"] == "fail"
    assert mismatch_gate["checks"]["abs_d_alpha_deg"] == pytest.approx(0.02)
    assert "alpha difference exceeds 0.01 degrees" in mismatch_gate["reason"]

    missing_report = compare_module.compare_runs(cpu, tmp_path / "missing", mode="early_gpu",
                                                 out=tmp_path / "missing.json")
    missing_gate = missing_report["gates"]["early_gpu"]
    assert missing_gate["verdict"] == "fail"
    assert "GPU result could not be read" in missing_gate["reason"]
    assert json.loads((tmp_path / "missing.json").read_text(encoding="utf-8")) == missing_report


def test_early_gpu_fails_without_native_solver_confirmation(tmp_path):
    cpu = _run_dir(tmp_path, "cpu", _early_result(0.5, 0.04))
    gpu_result = _early_result(0.5, 0.04)
    gpu_result["gpu_native_solver_active"] = False
    gpu = _run_dir(tmp_path, "gpu", gpu_result)

    report = compare_module.compare_runs(cpu, gpu, mode="early_gpu",
                                         out=tmp_path / "early.json")
    gate = report["gates"]["early_gpu"]

    assert gate["verdict"] == "fail"
    assert gate["checks"]["gpu_native_solver_active_confirmed"] is False
    assert "native GPU solver activation is missing or unconfirmed" in gate["reason"]
