import json

import pytest

from b737wing.cfd.solve import gci


COUNTS = (997858, 656292, 394198)


def _study(tmp_path, values=None, *, warm_pass=True, audit=None, native=True):
    values = values or {}
    names = ("fine", "medium", "coarse", "gpu")
    runs = {}
    h_squared = [count ** (-2.0 / 3.0) for count in COUNTS]
    base = {"CL": (0.49, 25.0), "CD": (0.04, 8.0), "CM": (-0.02, -5.0)}
    for name in names:
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        mesh_dir = tmp_path / "meshes" / name
        mesh_dir.mkdir(parents=True)
        mesh = mesh_dir / "mesh.msh.h5"
        mesh.write_bytes(b"mesh")
        if name == "gpu":
            mesh_count = COUNTS[0]
            index = 0
        else:
            index = names.index(name)
            mesh_count = COUNTS[index]
        (mesh_dir / "mesh_report.json").write_text(
            json.dumps({"cells": mesh_count}), encoding="utf-8")
        result = {"status": "converged", "alpha_deg": 2.0, "mesh": str(mesh),
                  "solver_audit": [] if audit is None else audit,
                  "gpu_native_solver_active": native}
        for quantity, (extrapolated, coefficient) in base.items():
            if name == "gpu":
                result[quantity] = values.get("gpu", {}).get(
                    quantity, extrapolated + coefficient * h_squared[0])
            elif quantity in values and name in values[quantity]:
                result[quantity] = values[quantity][name]
            else:
                result[quantity] = extrapolated + coefficient * h_squared[index]
        (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
        runs[name] = run
    warm = tmp_path / "warm.json"
    warm.write_text(json.dumps({"gates": {
        "warm_start": {"verdict": "pass" if warm_pass else "fail"}}}), encoding="utf-8")
    return runs, warm


def _evaluate(runs, warm):
    return gci.evaluate_gci(runs["fine"], runs["medium"], runs["coarse"],
                            runs["gpu"], warm)


def test_celik_unequal_grid_worked_check_recovers_order_and_extrapolation(tmp_path):
    runs, warm = _study(tmp_path)
    report = _evaluate(runs, warm)

    assert report["quantities"]["CL"]["p"] == pytest.approx(2.0, abs=0.01)
    assert report["quantities"]["CL"]["extrapolated"] == pytest.approx(0.49, abs=1e-6)
    assert report["quantities"]["CL"]["r21"] != pytest.approx(
        report["quantities"]["CL"]["r32"])


def test_oscillatory_grid_data_invalidates_p_and_gpu_gate(tmp_path):
    values = {"CL": {"fine": 1.0, "medium": 0.95, "coarse": 1.1}}
    runs, warm = _study(tmp_path, values)
    report = _evaluate(runs, warm)

    assert report["quantities"]["CL"]["convergence"] == "oscillatory"
    assert report["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "CL observed order is invalid" in report["gates"]["gpu_within_gci"]["reason"]


def test_gpu_gaps_below_gci925_pass_and_cd_gap_above_fails(tmp_path):
    runs, warm = _study(tmp_path)
    baseline = _evaluate(runs, warm)
    gpu_result = json.loads((runs["gpu"] / "result.json").read_text(encoding="utf-8"))
    for quantity in ("CL", "CD"):
        fine_value = baseline["quantities"][quantity]["cpu_fine"]
        gpu_result[quantity] = fine_value * (
            1.0 + baseline["quantities"][quantity]["gci925"] * 0.5)
    (runs["gpu"] / "result.json").write_text(json.dumps(gpu_result), encoding="utf-8")
    below = _evaluate(runs, warm)
    assert below["gates"]["gpu_within_gci"]["verdict"] == "pass"

    gpu_result["CD"] = below["quantities"]["CD"]["cpu_fine"] * (
        1.0 + below["quantities"]["CD"]["gci925"] * 1.5)
    (runs["gpu"] / "result.json").write_text(json.dumps(gpu_result), encoding="utf-8")
    above = _evaluate(runs, warm)
    assert above["quantities"]["CD"]["gpu_gap"] > above["quantities"]["CD"]["gci925"]
    assert above["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "CD GPU gap exceeds GCI92.5" in above["gates"]["gpu_within_gci"]["reason"]


def test_nonempty_solver_audit_fails_gpu_gate(tmp_path):
    runs, warm = _study(tmp_path, audit=["solver warning"])
    report = _evaluate(runs, warm)

    assert report["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "GPU solver audit is missing or non-empty" in report["gates"]["gpu_within_gci"]["reason"]


def test_unavailable_solver_audit_fails_gpu_gate(tmp_path):
    runs, warm = _study(tmp_path, audit="unavailable")
    report = _evaluate(runs, warm)

    assert report["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "GPU solver audit is missing or non-empty" in report["gates"]["gpu_within_gci"]["reason"]


def test_missing_native_gpu_confirmation_fails_gpu_gate(tmp_path):
    runs, warm = _study(tmp_path, native=False)
    result_path = runs["gpu"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("gpu_native_solver_active")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    report = _evaluate(runs, warm)

    assert report["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "native GPU solver activation is missing or unconfirmed" in report["gates"]["gpu_within_gci"]["reason"]


@pytest.mark.parametrize("gci_pass,warm_pass", [(True, True), (True, False),
                                                  (False, True), (False, False)])
def test_batch_warm_and_cold_gate_combinations(gci_pass, warm_pass):
    gates = gci._gates(gci_pass, warm_pass, "test")
    assert gates["gpu_batch"]["verdict"] == ("pass" if gci_pass or warm_pass else "fail")
    assert gates["cpu_warm"]["verdict"] == (
        "pass" if not gci_pass and warm_pass else "fail")
    assert gates["cpu_cold"]["verdict"] == (
        "pass" if not gci_pass and not warm_pass else "fail")


def test_missing_input_fails_gpu_gate_closed(tmp_path):
    runs, warm = _study(tmp_path)
    (runs["gpu"] / "result.json").unlink()
    report = _evaluate(runs, warm)

    assert report["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert "gpu run:" in report["gates"]["gpu_within_gci"]["reason"]


def test_cli_parser_accepts_cells_override():
    args = gci._parser().parse_args([
        "--fine", "fine", "--medium", "medium", "--coarse", "coarse", "--gpu", "gpu",
        "--warm-compare", "warm.json", "--cells", "997858,656292,394198", "--out", "gci.json"])

    assert args.cells == COUNTS


def test_cli_without_warm_compare_writes_only_gpu_gci_gate(tmp_path, capsys):
    runs, _ = _study(tmp_path)
    output = tmp_path / "gci.json"

    assert gci.main([
        "--fine", str(runs["fine"]), "--medium", str(runs["medium"]),
        "--coarse", str(runs["coarse"]), "--gpu", str(runs["gpu"]),
        "--cells", "997858,656292,394198", "--out", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["gates"]) == {"gpu_within_gci"}
    assert "warm_start" not in report
