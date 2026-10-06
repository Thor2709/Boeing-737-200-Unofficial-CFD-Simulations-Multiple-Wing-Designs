import csv
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from b737wing.cfd.solve import chain, compare_runs, run_case, sweep
from b737wing.cfd.solve import chain_detached
from b737wing.cfd.solve import gci as gci_module
from b737wing.post import case_post


def _write_plan(root, steps, log="chain_log.csv"):
    path = root / "plan.json"
    path.write_text(json.dumps({"log": log, "steps": steps}), encoding="utf-8")
    return path


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_run_artifacts(output, args, *, walls=True, ensight=True, alpha=3.25):
    output.mkdir(parents=True, exist_ok=True)
    mesh = chain._option_value(args, "--mesh", "mesh.msh.h5")
    solver = chain._option_value(args, "--solver", "cpu-db")
    precision = chain._option_value(args, "--precision", "single" if solver == "gpu-pb" else "double")
    fixed = chain._option_value(args, "--fixed-alpha")
    result = {"status": "converged", "case": args[0],
              "mesh": str(chain._path(mesh, chain.PROJECT_ROOT).resolve()),
              "alpha_deg": float(fixed) if fixed is not None else alpha,
              "solver": solver, "precision": precision,
              "converged": True,
              "np": 1 if solver == "gpu-pb" else int(chain._option_value(args, "--np", 4)),
              "min_final_iterations": int(chain._option_value(
                  args, "--min-final-iters", 900)),
              "final_iteration_rule": {"minimum_iterations": max(
                  int(chain._option_value(args, "--min-final-iters", 900)), 600)},
              "convergence_numbers": {"final_alpha_iterations": max(
                  int(chain._option_value(args, "--min-final-iters", 900)), 600)}}
    if solver == "gpu-pb":
        result["solver_audit"] = []
        result["gpu_native_solver_active"] = True
    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if walls:
        (output / "walls.csv").write_text("x,y,z,zone\n", encoding="utf-8")
    if ensight:
        (output / "ensight").mkdir(exist_ok=True)
        (output / "ensight" / "final.case").write_text("fake", encoding="utf-8")


def _fake_runner(calls, *, reports=None):
    reports = reports or {}

    def run(command, **kwargs):
        calls.append(command[2])
        module = command[2]
        args = command[3:]
        if module.endswith("compare_runs"):
            out = args[args.index("--out") + 1]
            output = chain._path(out, chain.PROJECT_ROOT)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(reports.get(out, {"gates": {}})), encoding="utf-8")
        elif module.endswith("run_case"):
            out = chain._path(args[args.index("--out") + 1], chain.PROJECT_ROOT)
            _write_run_artifacts(out, args)
        elif module.endswith("sweep"):
            out = chain._path(args[args.index("--out") + 1], chain.PROJECT_ROOT)
            out.mkdir(parents=True, exist_ok=True)
            (out / "sweep.json").write_text(
                json.dumps({"status": "converged"}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    return run


def test_sequential_order_and_log_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "solve", "cmd": "run_case", "args": ["C5", "--out", "runs/solve"]},
        {"id": "sweep", "cmd": "sweep", "args": ["C5", "--out", "runs/sweep"]},
        {"id": "compare", "cmd": "compare", "args": ["runs/a", "runs/b", "--out", "runs/cmp.json"]},
        {"id": "post", "cmd": "post", "args": ["C5", "--run", "runs/solve",
         "--out", "runs/solve/post"]},
    ]
    calls = []
    def popen(command, **kwargs):
        calls.append(command[2])
        output = chain._path(command[command.index("--out") + 1], tmp_path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "post_report.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0,
                               wait=lambda timeout=None: 0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=_fake_runner(calls),
                           popen=popen) == 0
    assert calls == ["b737wing.cfd.solve.run_case", "b737wing.cfd.solve.sweep",
                     "b737wing.cfd.solve.compare_runs", "b737wing.post.case_post"]
    rows = _rows(tmp_path / "chain_log.csv")
    assert [(row["step_id"], row["status"]) for row in rows] == [
        ("solve", "started"), ("solve", "done"),
        ("sweep", "started"), ("sweep", "done"),
        ("compare", "started"), ("compare", "done"),
        ("post", "started"), ("post", "done"),
    ]
    assert all(row["iso_time"] and row["resolved_args"] for row in rows)


def test_alpha_and_flow5_placeholders_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    reference = tmp_path / "runs" / "ref"
    reference.mkdir(parents=True)
    (reference / "result.json").write_text(json.dumps({"alpha_deg": 8.0}), encoding="utf-8")
    flow5 = tmp_path / "flow5_v2"
    flow5.mkdir()
    (flow5 / "summary.csv").write_text(
        "case,mesh,wake_spans,alpha_deg\nF2,fine,20.0,1.8747\nF5,fine,20.0,3.8042\n",
        encoding="utf-8")
    calls = []
    steps = [{"id": "resolve", "cmd": "run_case", "args": [
        "C2", "--mesh", "mesh.msh.h5", "--alpha0", "{flow5_alpha:C2:runs/ref}",
        "--fixed-alpha", "{alpha:runs/ref}", "--out", "runs/result"]}]

    def runner(command, **kwargs):
        calls.append(command[3:])
        out = tmp_path / "runs" / "result"
        _write_run_artifacts(out, command[3:])
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    assert calls[0][calls[0].index("--alpha0") + 1] == f"{8.0 + (1.8747 - 3.8042):.3f}"
    assert calls[0][calls[0].index("--fixed-alpha") + 1] == "8.0"


def test_seed_alpha_resolves_c2_and_fails_when_case_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    source = Path(__file__).resolve().parents[1] / "research" / "alpha_seeds.json"
    if not source.is_file():
        pytest.skip("project-only plan file not in repo")
    seed_path = tmp_path / "research" / "alpha_seeds.json"
    seed_path.parent.mkdir()
    shutil.copyfile(source, seed_path)

    assert chain._resolve_arg("{seed_alpha:C2:research/alpha_seeds.json}", tmp_path) == "1.6508"
    with pytest.raises(ValueError, match="cannot read C5 alpha0_CFD_deg"):
        chain._resolve_arg("{seed_alpha:C5:research/alpha_seeds.json}", tmp_path)


def test_when_pass_and_fail_branching(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    calls = []
    steps = [
        {"id": "pass_gate", "cmd": "compare", "args": ["a", "b", "--out", "pass.json"]},
        {"id": "fail_gate", "cmd": "compare", "args": ["a", "b", "--out", "fail.json"]},
        {"id": "warm", "cmd": "run_case", "args": ["C2", "--out", "runs/warm"],
         "when": {"step": "pass_gate", "gate": "warm_start", "verdict": "pass"}},
        {"id": "cold", "cmd": "run_case", "args": ["C2", "--out", "runs/cold"],
         "when": {"step": "fail_gate", "gate": "warm_start", "verdict": "fail"}},
        {"id": "skip", "cmd": "run_case", "args": ["C2", "--out", "runs/skip"],
         "when": {"step": "pass_gate", "gate": "warm_start", "verdict": "fail"}},
        {"id": "all", "cmd": "run_case", "args": ["C2", "--out", "runs/all"],
         "when": [
             {"step": "pass_gate", "gate": "warm_start", "verdict": "pass"},
             {"step": "fail_gate", "gate": "warm_start", "verdict": "fail"},
         ]},
        {"id": "not_all", "cmd": "run_case", "args": ["C2", "--out", "runs/not_all"],
         "when": [
             {"step": "pass_gate", "gate": "warm_start", "verdict": "pass"},
             {"step": "fail_gate", "gate": "warm_start", "verdict": "pass"},
         ]},
    ]

    def runner(command, **kwargs):
        calls.append(command[2])
        if command[2].endswith("compare_runs"):
            args = command[3:]
            out = tmp_path / args[args.index("--out") + 1]
            verdict = "pass" if out.name == "pass.json" else "fail"
            out.write_text(json.dumps({"gates": {"warm_start": {"verdict": verdict}}}),
                           encoding="utf-8")
        else:
            args = command[3:]
            out = tmp_path / args[args.index("--out") + 1]
            _write_run_artifacts(out, args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    assert calls == ["b737wing.cfd.solve.compare_runs", "b737wing.cfd.solve.compare_runs",
                     "b737wing.cfd.solve.run_case", "b737wing.cfd.solve.run_case",
                     "b737wing.cfd.solve.run_case"]
    rows = _rows(tmp_path / "chain_log.csv")
    assert [(row["step_id"], row["status"]) for row in rows if row["status"] != "started"] == [
        ("pass_gate", "done"), ("fail_gate", "done"), ("warm", "done"),
        ("cold", "done"), ("skip", "skipped"), ("all", "done"),
        ("not_all", "skipped")]


def test_failed_step_blocks_only_dependents(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    calls = []
    steps = [
        {"id": "bad", "cmd": "run_case", "args": ["C5", "--out", "runs/fail"]},
        {"id": "dependent", "cmd": "run_case", "args": ["C5", "--init-data",
         "runs/fail/final.dat.h5", "--out", "runs/dependent"]},
        {"id": "independent", "cmd": "sweep", "args": ["C5", "--out", "runs/independent"]},
    ]

    def runner(command, **kwargs):
        calls.append(command[2])
        if "runs/fail" in " ".join(command[4:]):
            return SimpleNamespace(returncode=2)
        if command[2].endswith("sweep"):
            args = command[4:]
            out = chain._path(args[args.index("--out") + 1], tmp_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "sweep.json").write_text(json.dumps({"status": "converged"}),
                                            encoding="utf-8")
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    assert calls == ["b737wing.cfd.solve.run_case", "b737wing.cfd.solve.sweep"]
    rows = _rows(tmp_path / "chain_log.csv")
    terminals = {row["step_id"]: row["status"] for row in rows if row["status"] != "started"}
    assert terminals == {"bad": "failed", "dependent": "blocked", "independent": "done"}


def test_skip_if_done_converged_run(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "runs" / "existing"
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps({"status": "converged"}), encoding="utf-8")
    calls = []
    steps = [{"id": "existing", "cmd": "run_case", "args": ["C5", "--out", "runs/existing"],
              "skip_if_done": True}]
    def runner(command, **kwargs):
        calls.append(command)
        _write_run_artifacts(output, command[3:])
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    assert calls
    assert list((tmp_path / "runs").glob("existing_stale_*"))
    assert _rows(tmp_path / "chain_log.csv")[-1]["status"] == "done"


def test_matching_skip_if_done_reuses_result_with_complete_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [{"id": "same", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
             "--fixed-alpha", "3.25", "--out", "runs/same"], "skip_if_done": True}]
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_run_artifacts(tmp_path / "runs" / "same", command[3:])
        return SimpleNamespace(returncode=0)

    plan = _write_plan(tmp_path, steps)
    assert chain.run_chain(plan, runner=runner) == 0
    assert chain.run_chain(plan, runner=runner) == 0
    assert len(calls) == 1
    assert _rows(tmp_path / "chain_log.csv")[-1]["status"] == "skipped"


@pytest.mark.parametrize("field,value", [
    ("mesh", "other.msh.h5"), ("alpha_deg", 4.0), ("solver", "gpu-pb")])
def test_skip_if_done_mismatched_result_moves_stale_directory(tmp_path, monkeypatch,
                                                               field, value):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "runs" / "same"
    steps = [{"id": "same", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
             "--fixed-alpha", "3.25", "--out", "runs/same"], "skip_if_done": True}]
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_run_artifacts(output, command[3:])
        return SimpleNamespace(returncode=0)

    plan = _write_plan(tmp_path, steps)
    assert chain.run_chain(plan, runner=runner) == 0
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result[field] = value
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert chain.run_chain(plan, runner=runner) == 0
    assert len(calls) == 2
    assert list((tmp_path / "runs").glob("same_stale_*"))


@pytest.mark.parametrize(("option", "value"), [
    ("--courant-ramp", "100:2,300:5,6000:10"),
    ("--precision", "single"), ("--min-final-iters", "1200"),
    ("--first-slope", "0.12"), ("--init-data", "restart.dat.h5"),
    ("--interp-from", "runs/interpolation_source"),
    ("--seed-from", "runs/seed_source"), ("--np", "2"),
    ("--solver", "gpu-pb")])
def test_skip_if_done_rejects_numerical_provenance_changes(
        tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    args = ["C5", "--mesh", "mesh.msh.h5", "--fixed-alpha", "3.25",
            "--out", "runs/same"]
    steps = [{"id": "same", "cmd": "run_case", "args": args.copy(),
              "skip_if_done": True}]
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        _write_run_artifacts(tmp_path / "runs" / "same", command[3:])
        return SimpleNamespace(returncode=0)

    plan = _write_plan(tmp_path, steps)
    assert chain.run_chain(plan, runner=runner) == 0
    changed_args = ["C5", "--mesh", "mesh.msh.h5", "--fixed-alpha", "3.25",
                    option, value, "--out", "runs/same"]
    steps[0]["args"] = changed_args
    plan.write_text(json.dumps({"log": "chain_log.csv", "steps": steps}),
                    encoding="utf-8")
    assert chain.run_chain(plan, runner=runner) == 0
    assert len(calls) == 2
    assert list((tmp_path / "runs").glob("same_stale_*"))


def test_unqualified_gpu_result_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [{"id": "gpu", "cmd": "run_case", "args": ["C2", "--mesh",
              "mesh.msh.h5", "--solver", "gpu-pb", "--fixed-alpha", "3.25",
              "--out", "runs/gpu"], "skip_if_done": True}]
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        _write_run_artifacts(tmp_path / "runs" / "gpu", command[3:])
        return SimpleNamespace(returncode=0)

    plan = _write_plan(tmp_path, steps)
    assert chain.run_chain(plan, runner=runner) == 0
    result_path = tmp_path / "runs" / "gpu" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["gpu_native_solver_active"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert chain.run_chain(plan, runner=runner) == 0
    assert len(calls) == 2
    assert list((tmp_path / "runs").glob("gpu_stale_*"))


def test_cold_run_is_not_blocked_by_skipped_warm_step_sharing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "selector", "cmd": "compare", "args": ["a", "b", "--out", "gate.json"]},
        {"id": "cpu_warm_Ci", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/shared"], "when": {"step": "selector", "gate": "warm_start",
                                               "verdict": "pass"}},
        {"id": "cpu_cold_Ci", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/shared"], "when": {"step": "selector", "gate": "warm_start",
                                               "verdict": "fail"}},
    ]
    calls = []

    def runner(command, **kwargs):
        args = command[3:]
        calls.append(command[2])
        if command[2].endswith("compare_runs"):
            (tmp_path / "gate.json").write_text(json.dumps({"gates": {
                "warm_start": {"verdict": "fail"}}}), encoding="utf-8")
        else:
            _write_run_artifacts(tmp_path / "runs" / "shared", args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    assert terminal["cpu_warm_Ci"] == "skipped"
    assert terminal["cpu_cold_Ci"] == "done"
    assert calls.count("b737wing.cfd.solve.run_case") == 1


def test_timeout_kills_run_pids_blocks_dependents_and_runs_independent_step(
        tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "slow", "cmd": "run_case", "timeout_s": 0.01,
         "args": ["C5", "--mesh", "slow.msh.h5", "--out", "runs/slow"]},
        {"id": "dependent", "cmd": "run_case", "args": ["C5", "--init-data",
         "runs/slow/final.dat.h5", "--out", "runs/dependent"]},
        {"id": "independent", "cmd": "run_case", "args": ["C5", "--mesh",
         "independent.msh.h5", "--out", "runs/independent"]},
    ]
    killed, waited, calls = [], [], []
    monkeypatch.setattr(chain, "_kill_pid", killed.append)
    monkeypatch.setattr(chain, "_wait_for_pids", lambda pids: waited.extend(pids))

    def runner(command, **kwargs):
        args = command[3:]
        calls.append(args[0])
        if args[args.index("--out") + 1] == "runs/slow":
            output = tmp_path / "runs" / "slow"
            output.mkdir(parents=True, exist_ok=True)
            (output / "fluent_pids.json").write_text(
                json.dumps({"pids": [8123], "launches": []}), encoding="utf-8")
            assert kwargs["timeout"] == 0.01
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        _write_run_artifacts(chain._path(args[args.index("--out") + 1], tmp_path), args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    terminal = {row["step_id"]: row for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    assert terminal["slow"]["status"] == "failed"
    assert terminal["slow"]["result_status"] == "timeout"
    assert terminal["dependent"]["status"] == "blocked"
    assert terminal["independent"]["status"] == "done"
    assert calls == ["C5", "C5"]
    assert killed == [8123] and waited == [8123]


@pytest.mark.parametrize("exit_code,write_report", [(1, True), (0, False)])
def test_post_failure_is_recorded(tmp_path, monkeypatch, exit_code, write_report):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [{"id": "post", "cmd": "post", "args": ["C5", "--run", "runs/input",
              "--out", "runs/input/post"]}]

    def popen(command, **kwargs):
        output = tmp_path / "runs" / "input" / "post"
        output.mkdir(parents=True, exist_ok=True)
        if write_report:
            (output / "post_report.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=123, returncode=exit_code,
                               poll=lambda: exit_code,
                               wait=lambda timeout=None: exit_code)

    assert chain.run_chain(_write_plan(tmp_path, steps), popen=popen) == 1
    row = _rows(tmp_path / "chain_log.csv")[-1]
    assert row["status"] == "failed"


def test_incomplete_exports_skip_post_with_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "run", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/incomplete"]},
        {"id": "post", "cmd": "post", "args": ["C5", "--run", "runs/incomplete",
         "--out", "runs/incomplete/post"]},
    ]
    post_calls = []

    def runner(command, **kwargs):
        args = command[3:]
        _write_run_artifacts(tmp_path / "runs" / "incomplete", args, walls=False)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner,
                           popen=lambda *args, **kwargs: post_calls.append(args)) == 1
    terminal = {row["step_id"]: row for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    assert terminal["run"]["status"] == "done_incomplete_exports"
    assert terminal["post"]["status"] == "skipped"
    assert "done_incomplete_exports" in terminal["post"]["result_status"]
    assert post_calls == []


def test_detached_chain_launch_writes_pid_and_uses_detached_windows_flags(
        tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    plan = _write_plan(tmp_path, [])
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=4567)

    monkeypatch.setattr(chain_detached.subprocess, "Popen", popen)

    pid = chain_detached.launch(plan)

    assert pid == 4567
    assert (tmp_path / "chain.pid").read_text(encoding="ascii").strip() == "4567"
    assert calls[0][0][1:3] == ["-m", "b737wing.cfd.solve.chain"]
    if chain_detached.os.name == "nt":
        assert calls[0][1]["creationflags"] == (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def test_stale_gate_from_skipped_compare_does_not_authorize_consumer(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    stale = tmp_path / "conditional.json"
    stale.write_text(json.dumps({"gates": {"warm_start": {"verdict": "pass"}}}),
                     encoding="utf-8")
    steps = [
        {"id": "selector", "cmd": "compare", "args": ["a", "b", "--out", "selector.json"]},
        {"id": "conditional", "cmd": "compare", "args": ["a", "b", "--out", "conditional.json"],
         "when": {"step": "selector", "gate": "warm_start", "verdict": "pass"}},
        {"id": "dependent", "cmd": "run_case", "args": ["C5", "--out", "runs/dependent"],
         "when": {"step": "conditional", "gate": "warm_start", "verdict": "pass"}},
    ]
    calls = []

    def runner(command, **kwargs):
        calls.append(command[2])
        args = command[3:]
        output = chain._path(args[args.index("--out") + 1], tmp_path)
        output.write_text(json.dumps({"gates": {"warm_start": {"verdict": "fail"}}}),
                          encoding="utf-8")
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    assert calls == ["b737wing.cfd.solve.compare_runs"]
    assert [(row["step_id"], row["status"]) for row in _rows(tmp_path / "chain_log.csv")
            if row["status"] != "started"] == [
                ("selector", "done"), ("conditional", "skipped"), ("dependent", "skipped")]


def test_consumer_of_path_with_only_skipped_producer_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    stale_run = tmp_path / "runs" / "skipped"
    stale_run.mkdir(parents=True)
    (stale_run / "result.json").write_text(json.dumps({"status": "converged"}),
                                            encoding="utf-8")
    steps = [
        {"id": "selector", "cmd": "compare", "args": ["a", "b", "--out", "selector.json"]},
        {"id": "skipped_run", "cmd": "run_case", "args": ["C5", "--out", "runs/skipped"],
         "when": {"step": "selector", "gate": "warm_start", "verdict": "pass"}},
        {"id": "consumer", "cmd": "run_case", "args": ["C5", "--seed-from", "runs/skipped",
         "--out", "runs/consumer"]},
    ]
    calls = []

    def runner(command, **kwargs):
        calls.append(command[2])
        if command[2].endswith("compare_runs"):
            output = tmp_path / "selector.json"
            output.write_text(json.dumps({"gates": {"warm_start": {"verdict": "fail"}}}),
                              encoding="utf-8")
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    assert calls == ["b737wing.cfd.solve.compare_runs"]
    assert [(row["step_id"], row["status"]) for row in _rows(tmp_path / "chain_log.csv")
            if row["status"] != "started"] == [
                ("selector", "done"), ("skipped_run", "skipped"), ("consumer", "blocked")]


def test_failed_output_dependency_uses_normalized_path_components(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "bad", "cmd": "run_case", "args": ["C5", "--out", "runs/C1"]},
        {"id": "c10", "cmd": "run_case", "args": ["C5", "--init-data",
         "runs/C10/final.dat.h5", "--out", "runs/C10"]},
        {"id": "equivalent", "cmd": "run_case", "args": ["C5", "--init-data",
         "runs/../runs/C1/final.dat.h5", "--out", "runs/equivalent"]},
    ]
    calls = []

    def runner(command, **kwargs):
        calls.append(command[2])
        args = command[3:]
        if args[args.index("--out") + 1] == "runs/C1":
            return SimpleNamespace(returncode=2)
        output = chain._path(args[args.index("--out") + 1], tmp_path)
        _write_run_artifacts(output, args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    assert calls == ["b737wing.cfd.solve.run_case", "b737wing.cfd.solve.run_case"]
    terminals = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                 if row["status"] != "started"}
    assert terminals == {"bad": "failed", "c10": "done", "equivalent": "blocked"}


def test_resolved_output_placeholder_is_used_for_result_and_dependency_lookup(
        tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    reference = tmp_path / "runs" / "ref"
    reference.mkdir(parents=True)
    (reference / "result.json").write_text(json.dumps({"alpha_deg": 8.0}), encoding="utf-8")
    steps = [
        {"id": "resolved", "cmd": "run_case", "args": ["C5", "--out",
         "{alpha:runs/ref}"]},
        {"id": "consumer", "cmd": "run_case", "args": ["C5", "--init-data",
         "8.0/final.dat.h5", "--out", "runs/consumer"]},
    ]
    calls = []

    def runner(command, **kwargs):
        calls.append(command[2])
        args = command[3:]
        output = chain._path(args[args.index("--out") + 1], tmp_path)
        _write_run_artifacts(output, args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 0
    assert calls == ["b737wing.cfd.solve.run_case", "b737wing.cfd.solve.run_case"]
    assert [(row["step_id"], row["status"]) for row in _rows(tmp_path / "chain_log.csv")
            if row["status"] != "started"] == [("resolved", "done"), ("consumer", "done")]


def test_sweep_without_sweep_json_fails_and_blocks_dependents(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "sweep", "cmd": "sweep", "args": ["C5", "--out", "runs/sweep"]},
        {"id": "dependent", "cmd": "run_case", "args": ["C5", "--interp-from",
         "runs/sweep", "--out", "runs/dependent"]},
    ]
    calls = []

    def runner(command, **kwargs):
        calls.append(command[2])
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    assert calls == ["b737wing.cfd.solve.sweep"]
    assert [(row["step_id"], row["status"]) for row in _rows(tmp_path / "chain_log.csv")
            if row["status"] != "started"] == [("sweep", "failed"), ("dependent", "blocked")]


def test_dry_prints_plan_without_running_or_logging(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    calls = []
    steps = [{"id": "dry", "cmd": "run_case", "args": ["C5", "--out", "runs/dry"]}]
    assert chain.run_chain(_write_plan(tmp_path, steps), dry=True,
                           runner=lambda *args, **kwargs: calls.append(args)) == 0
    assert calls == []
    assert not (tmp_path / "chain_log.csv").exists()
    assert json.loads(capsys.readouterr().out)["dry"] is True


def test_chain_plan_dry_prints_resolved_early_gpu_orders(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    root = Path(__file__).resolve().parents[1]
    seeds_src = root / "research" / "alpha_seeds.json"
    plan_path = root / "research" / "chain_plan_v2.json"
    if not seeds_src.is_file() or not plan_path.is_file():
        pytest.skip("project-only plan file not in repo")
    seeds = tmp_path / "research" / "alpha_seeds.json"
    seeds.parent.mkdir()
    shutil.copyfile(seeds_src, seeds)
    reference = tmp_path / "cfd_v2" / "runs" / "C5" / "iH0"
    reference.mkdir(parents=True)
    (reference / "result.json").write_text(json.dumps({"alpha_deg": 3.25}),
                                           encoding="utf-8")
    summary = tmp_path / "results" / "flow5" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "results" / "flow5" / "summary.csv", summary)

    assert chain.run_chain(plan_path, dry=True) == 0
    preview = json.loads(capsys.readouterr().out)
    assert all(not step["resolution_errors"] for step in preview["steps"])
    orders = preview["branch_orders"]
    passed = orders["early_gpu_pass"]
    failed = orders["early_gpu_fail"]
    passed_ids = [step["id"] for step in passed]
    failed_ids = [step["id"] for step in failed]
    assert passed_ids.index("early_gpu_C2") < passed_ids.index("g400")
    assert "early_gpu_C2" not in failed_ids
    assert failed_ids.index("g400") < failed_ids.index("gpu_direct_C2")
    assert all(not step["resolution_errors"] for steps in orders.values() for step in steps)
    assert all("{" not in arg for steps in orders.values()
               for step in steps for arg in step["args"])


def test_chain_plan_loads_and_uses_parser_options():
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "research" / "chain_plan_v2.json"
    if not plan_path.is_file():
        pytest.skip("project-only plan file not in repo")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    parsers = {"run_case": run_case._parser(), "sweep": sweep._parser(),
               "compare": compare_runs._parser(), "gci": gci_module._parser(),
               "post": case_post._parser()}
    assert plan["steps"]
    for step in plan["steps"]:
        if step["cmd"] == "gpu_status":
            assert step["args"] == ["--out", "cfd_v2/runs/gpu_status.json"]
            continue
        assert step["cmd"] in parsers
        valid_options = parsers[step["cmd"]]._option_string_actions
        for argument in step["args"]:
            if argument.startswith("--"):
                option = argument.split("=", 1)[0]
                assert option in valid_options, f"{step['id']}: unsupported option {option}"


def test_chain_plan_search_routes_use_physics_seed_and_first_slope():
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "research" / "chain_plan_v2.json"
    if not plan_path.is_file():
        pytest.skip("project-only plan file not in repo")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    search_steps = [step for step in plan["steps"]
                    if step["id"].startswith(("early_gpu_", "gpu_direct_", "gpu_warm_",
                                              "cpu_warm_", "cpu_cold_"))
                    and step["cmd"] == "run_case"]
    assert search_steps
    for step in search_steps:
        case = step["args"][0]
        args = step["args"]
        slope = args[args.index("--first-slope") + 1]
        if step["id"].startswith("cpu_warm_"):
            assert "--alpha0" not in args
            assert slope == ("0.09" if case == "C3" else "0.079")
            continue
        alpha0 = args[args.index("--alpha0") + 1]
        if case == "C3":
            assert alpha0 == "{flow5_alpha:C3:cfd_v2/runs/C5/iH0}"
            assert slope == "0.09"
        else:
            assert case in {"C1", "C2", "C4"}
            assert alpha0 == f"{{seed_alpha:{case}:research/alpha_seeds.json}}"
            assert slope == "0.079"


def _run_gci_route(tmp_path, monkeypatch, gci_verdict, warm_verdict="pass",
                   early_verdict="fail"):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "research" / "chain_plan_v2.json"
    seeds_path = root / "research" / "alpha_seeds.json"
    if not plan_path.is_file() or not seeds_path.is_file():
        pytest.skip("project-only plan file not in repo")
    reference = tmp_path / "cfd_v2" / "runs" / "C5" / "iH0"
    reference.mkdir(parents=True)
    (reference / "result.json").write_text(
        json.dumps({"alpha_deg": 3.25}), encoding="utf-8")
    flow5 = tmp_path / "results" / "flow5"
    flow5.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "results" / "flow5" / "summary.csv", flow5 / "summary.csv")
    seeds = tmp_path / "research" / "alpha_seeds.json"
    seeds.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seeds_path, seeds)
    calls, posts = [], []

    def output_path(args):
        return chain._path(args[args.index("--out") + 1], tmp_path)

    def runner(command, **kwargs):
        module, args = command[2], command[3:]
        calls.append(module)
        if module.endswith("compare_runs"):
            output = output_path(args)
            output.parent.mkdir(parents=True, exist_ok=True)
            if "--mode" in args and args[args.index("--mode") + 1] == "early_gpu":
                gates = {"early_gpu": {"verdict": early_verdict,
                                        "reason": "fixture early GPU result"}}
            else:
                gates = {"warm_start": {"verdict": warm_verdict}}
            output.write_text(json.dumps({"gates": gates}), encoding="utf-8")
        elif module.endswith(".gci"):
            output = output_path(args)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"gates": {
                "gpu_within_gci": {"verdict": gci_verdict}}}), encoding="utf-8")
        elif module.endswith("run_case"):
            output = output_path(args)
            _write_run_artifacts(output, args)
        return SimpleNamespace(returncode=0)

    def popen(command, **kwargs):
        args = command[3:]
        posts.append(args[args.index("--run") + 1])
        output = output_path(args)
        output.mkdir(parents=True, exist_ok=True)
        (output / "post_report.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0,
                               wait=lambda timeout=None: 0)

    exit_code = chain.run_chain(root / "research" / "chain_plan_v2.json",
                                runner=runner, popen=popen)
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    return exit_code, terminal, calls, posts


def test_gci_pass_runs_gpu_direct_route_and_skips_warm_routes(tmp_path, monkeypatch):
    exit_code, terminal, calls, posts = _run_gci_route(tmp_path, monkeypatch, "pass")

    assert exit_code == 0
    assert all(terminal[f"gpu_direct_{case}"] == "done" for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"post_gpu_direct_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(f"cfd_v2/runs/{case}/gpu" in posts for case in ("C2", "C4", "C1", "C3"))
    assert terminal["T5"] == terminal["warm_start_gate"] == "skipped"
    assert all(terminal[f"gpu_warm_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_warm_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_cold_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert "b737wing.cfd.solve.gci" in calls
    preserved_run_order = [step_id for step_id, status in terminal.items()
                           if status == "done" and (step_id in {"T2", "g400",
                               "g400_cfl20", "g650"}
                               or step_id.startswith("gpu_direct_"))]
    assert preserved_run_order == ["T2", "g400", "g400_cfl20", "g650",
                                   "gpu_direct_C2", "gpu_direct_C4",
                                   "gpu_direct_C1", "gpu_direct_C3"]
    assert all(terminal[f"early_gpu_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))


def test_early_gpu_pass_and_gci_pass_writes_final_status_and_skips_direct(
        tmp_path, monkeypatch):
    exit_code, terminal, _, _ = _run_gci_route(
        tmp_path, monkeypatch, "pass", early_verdict="pass")

    assert exit_code == 0
    assert all(terminal[f"early_gpu_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"post_early_gpu_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"gpu_direct_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    status = json.loads((tmp_path / "cfd_v2" / "runs" / "gpu_status.json").read_text(
        encoding="utf-8"))
    assert status["early_gpu"]["verdict"] == "pass"
    assert status["gci"]["verdict"] == "pass"
    assert status["early_gpu_runs"] == [f"cfd_v2/runs/{case}/gpu"
                                         for case in ("C2", "C4", "C1", "C3")]
    assert status["status"] == "final"


def test_early_gpu_pass_and_gci_fail_keeps_t5_route_and_reuses_early_gpu(
        tmp_path, monkeypatch):
    exit_code, terminal, _, _ = _run_gci_route(
        tmp_path, monkeypatch, "fail", warm_verdict="pass", early_verdict="pass")

    assert exit_code == 0
    assert terminal["T5"] == terminal["post_T5"] == terminal["warm_start_gate"] == "done"
    assert all(terminal[f"gpu_direct_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"gpu_warm_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_warm_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    status = json.loads((tmp_path / "cfd_v2" / "runs" / "gpu_status.json").read_text(
        encoding="utf-8"))
    assert status["early_gpu"]["verdict"] == "pass"
    assert status["gci"]["verdict"] == "fail"
    assert status["status"] == "provisional_gci_fail"


def test_gci_fail_warm_pass_runs_warm_route_despite_skipped_direct_producer(
        tmp_path, monkeypatch):
    exit_code, terminal, _, posts = _run_gci_route(tmp_path, monkeypatch, "fail", "pass")

    assert exit_code == 0
    assert terminal["T5"] == terminal["warm_start_gate"] == "done"
    assert all(terminal[f"gpu_direct_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"gpu_warm_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"post_gpu_warm_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_warm_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(f"cfd_v2/runs/{case}/gpu" in posts for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_cold_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))


def test_gci_fail_warm_fail_runs_only_cold_cpu_route(tmp_path, monkeypatch):
    exit_code, terminal, _, _ = _run_gci_route(tmp_path, monkeypatch, "fail", "fail")

    assert exit_code == 0
    assert terminal["T5"] == terminal["warm_start_gate"] == "done"
    assert all(terminal[f"gpu_warm_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_warm_{case}"] == "skipped"
               for case in ("C2", "C4", "C1", "C3"))
    assert all(terminal[f"cpu_cold_{case}"] == "done"
               for case in ("C2", "C4", "C1", "C3"))


def test_gci_exits_zero_without_readable_output_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [{"id": "gci", "cmd": "gci", "args": ["--out", "reports/gci.json"]}]
    assert chain.run_chain(_write_plan(tmp_path, steps),
                           runner=lambda *args, **kwargs: SimpleNamespace(returncode=0)) == 1


def test_search_timeout_covers_full_secant_budget_and_fixed_alpha_stays_six_hours():
    from b737wing.cfd.solve.search import MAX_ITERATIONS_PER_ALPHA, MAX_SECANT_STEPS

    search_step = {"cmd": "run_case", "args": ["C2", "--mesh", "mesh.msh.h5"]}
    fixed_step = {"cmd": "run_case", "args": ["C2", "--mesh", "mesh.msh.h5",
                                               "--fixed-alpha", "3.6"]}
    expected = (MAX_SECANT_STEPS + 1) * MAX_ITERATIONS_PER_ALPHA * 4.0 + 60 * 60

    assert chain._timeout_for_step(search_step, search_step["args"]) == expected
    assert chain._timeout_for_step(fixed_step, fixed_step["args"]) == 8 * 60 * 60


def test_history_watchdog_kills_a_step_that_stops_growing(tmp_path, monkeypatch):
    class HungProcess:
        pid = 42

        def poll(self):
            return None

    ticks = iter(range(100))
    killed = []
    monkeypatch.setattr(chain.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(chain.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(chain, "STALL_TIMEOUT_S", 3)
    monkeypatch.setattr(chain, "PROCESS_POLL_S", 1)
    monkeypatch.setattr(chain, "_kill_step_process_and_fluent",
                        lambda output, process: killed.append((output, process.pid)))

    exit_code, status, process = chain._run_managed_process(
        ["python", "solve"], {}, 100, output=tmp_path,
        progress_log=tmp_path / "chain_step.log", watchdog=True,
        popen=lambda *_args, **_kwargs: HungProcess())

    assert exit_code is None
    assert status == "stalled"
    assert process.pid == 42
    assert killed == [(tmp_path, 42)]


def test_pid_wait_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "_pid_exists", lambda _pid: True)

    with pytest.raises(TimeoutError, match="timed out waiting for Fluent PIDs"):
        chain._wait_for_pids([44], timeout_s=0.01)


def test_hung_post_is_killed_at_bound_and_independent_case_continues(
        tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(chain, "POST_TIMEOUT_S", 0.01)
    clock = [0.0]

    def monotonic():
        value = clock[0]
        clock[0] += 0.02
        return value

    monkeypatch.setattr(chain.time, "monotonic", monotonic)
    killed = []

    class HungPost:
        pid = 77
        returncode = None

        def poll(self):
            return None

    class CompletedCase:
        returncode = 0

        def poll(self):
            return 0

    def kill_tree(pid, process=None):
        killed.append(pid)
        process.returncode = -9

    monkeypatch.setattr(chain, "_kill_process_tree", kill_tree)
    steps = [
        {"id": "post", "cmd": "post", "args": ["C5", "--run", "runs/source",
         "--out", "runs/source/post"]},
        {"id": "independent", "cmd": "run_case", "args": ["C5", "--mesh",
         "mesh.msh.h5", "--out", "runs/independent"]},
    ]

    def popen(command, **_kwargs):
        if command[2] == "b737wing.post.case_post":
            return HungPost()
        args = command[3:]
        _write_run_artifacts(tmp_path / "runs" / "independent", args)
        return CompletedCase()

    assert chain.run_chain(_write_plan(tmp_path, steps), popen=popen) == 1
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    assert terminal["post"] == "failed"
    assert terminal["independent"] == "done"
    assert killed == [77]


def test_failed_t2_writes_early_gpu_fail_and_runs_cold_route(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "T2", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/T2"]},
        {"id": "early_gpu_gate", "cmd": "compare", "args": ["runs/ref", "runs/T2",
         "--mode", "early_gpu", "--out", "runs/T2/early.json"]},
        {"id": "early_gpu_C2", "cmd": "run_case", "args": ["C2", "--mesh",
         "mesh.msh.h5", "--out", "runs/early_C2"], "when": {
             "step": "early_gpu_gate", "gate": "early_gpu", "verdict": "pass"}},
        {"id": "gci_gate", "cmd": "gci", "args": ["runs/fine", "runs/medium",
         "runs/coarse", "runs/T2", "--warm-compare", "reports/warm.json",
         "--out", "reports/gci.json"]},
        {"id": "cpu_cold_C2", "cmd": "run_case", "args": ["C2", "--mesh",
         "mesh.msh.h5", "--out", "runs/cold_C2"], "when": [
             {"step": "early_gpu_gate", "gate": "early_gpu", "verdict": "fail"},
             {"step": "gci_gate", "gate": "cpu_cold", "verdict": "pass"}]},
    ]
    calls = []

    def runner(command, **_kwargs):
        args = command[3:]
        calls.append(command[2])
        if args[0] == "C5":
            _write_run_artifacts(tmp_path / "runs" / "T2", args)
            return SimpleNamespace(returncode=1)
        _write_run_artifacts(chain._path(chain._option_value(args, "--out"), tmp_path), args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    report = json.loads((tmp_path / "runs" / "T2" / "early.json").read_text(
        encoding="utf-8"))
    gci = json.loads((tmp_path / "reports" / "gci.json").read_text(encoding="utf-8"))
    assert report["gates"]["early_gpu"]["verdict"] == "fail"
    assert "T2" in report["gates"]["early_gpu"]["reason"]
    assert gci["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert terminal["early_gpu_C2"] == "skipped"
    assert terminal["cpu_cold_C2"] == "done"
    assert "b737wing.cfd.solve.compare_runs" not in calls


def test_failed_g650_and_t5_fail_closed_to_cpu_cold_route(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "T5", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/T5"]},
        {"id": "warm_start_gate", "cmd": "compare", "args": ["runs/ref", "runs/T5",
         "--out", "reports/warm.json"]},
        {"id": "g650", "cmd": "run_case", "args": ["C5", "--mesh", "mesh.msh.h5",
         "--out", "runs/g650"]},
        {"id": "gci_gate", "cmd": "gci", "args": ["runs/g400", "runs/g400_cfl20",
         "runs/g650", "runs/T2", "--warm-compare", "reports/warm.json",
         "--out", "reports/gci.json"]},
        {"id": "cpu_warm_C2", "cmd": "run_case", "args": ["C2", "--mesh",
         "mesh.msh.h5", "--out", "runs/cpu_warm"], "when": {
             "step": "gci_gate", "gate": "cpu_warm", "verdict": "pass"}},
        {"id": "cpu_cold_C2", "cmd": "run_case", "args": ["C2", "--mesh",
         "mesh.msh.h5", "--out", "runs/cpu_cold"], "when": {
             "step": "gci_gate", "gate": "cpu_cold", "verdict": "pass"}},
    ]

    def runner(command, **_kwargs):
        args = command[3:]
        if args[0] == "C5":
            out = chain._option_value(args, "--out")
            _write_run_artifacts(chain._path(out, tmp_path), args)
            return SimpleNamespace(returncode=1)
        _write_run_artifacts(chain._path(chain._option_value(args, "--out"), tmp_path), args)
        return SimpleNamespace(returncode=0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    warm = json.loads((tmp_path / "reports" / "warm.json").read_text(encoding="utf-8"))
    gci = json.loads((tmp_path / "reports" / "gci.json").read_text(encoding="utf-8"))
    assert warm["gates"]["warm_start"]["verdict"] == "fail"
    assert gci["gates"]["gpu_within_gci"]["verdict"] == "fail"
    assert gci["gates"]["cpu_cold"]["verdict"] == "pass"
    assert terminal["cpu_warm_C2"] == "skipped"
    assert terminal["cpu_cold_C2"] == "done"


def test_failed_production_case_does_not_stop_independent_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(chain, "PROJECT_ROOT", tmp_path)
    steps = [
        {"id": "failed_case", "cmd": "run_case", "args": ["C2", "--mesh",
         "mesh.msh.h5", "--out", "runs/failed"]},
        {"id": "other_case", "cmd": "run_case", "args": ["C4", "--mesh",
         "mesh.msh.h5", "--out", "runs/other"]},
    ]
    calls = []

    def runner(command, **_kwargs):
        args = command[3:]
        calls.append(args[0])
        out = chain._path(chain._option_value(args, "--out"), tmp_path)
        _write_run_artifacts(out, args)
        return SimpleNamespace(returncode=2 if args[0] == "C2" else 0)

    assert chain.run_chain(_write_plan(tmp_path, steps), runner=runner) == 1
    terminal = {row["step_id"]: row["status"] for row in _rows(tmp_path / "chain_log.csv")
                if row["status"] != "started"}
    assert calls == ["C2", "C4"]
    assert terminal["failed_case"] == "failed"
    assert terminal["other_case"] == "done"


@pytest.mark.parametrize(("early_done", "run_audit", "run_native"), [
    (False, [], True), (True, ["unsupported"], False)])
def test_gpu_status_is_incomplete_without_current_qualified_early_evidence(
        tmp_path, early_done, run_audit, run_native):
    early = {"id": "early", "cmd": "compare", "args": ["a", "b", "--out", "early.json"]}
    gci = {"id": "gci", "cmd": "gci", "args": ["--out", "gci.json"]}
    run = {"id": "gpu_C2", "cmd": "run_case", "args": ["C2", "--out", "runs/C2/gpu"]}
    status_step = {"id": "status", "cmd": "gpu_status", "args": ["--out", "status.json"],
                   "early_gpu_gate_step": "early", "gci_gate_step": "gci",
                   "early_gpu_run_steps": ["gpu_C2"]}
    plan_steps = {step["id"]: step for step in (early, gci, run, status_step)}
    (tmp_path / "early.json").write_text(json.dumps({"gates": {
        "early_gpu": {"verdict": "pass", "reason": "fixture"}}}), encoding="utf-8")
    (tmp_path / "gci.json").write_text(json.dumps({"gates": {
        "gpu_within_gci": {"verdict": "pass", "reason": "fixture"}}}),
        encoding="utf-8")
    output = chain._path("runs/C2/gpu", tmp_path)
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps({
        "status": "audit_failed" if run_audit else "converged",
        "converged": not run_audit, "solver_audit": run_audit,
        "gpu_native_solver_active": run_native}), encoding="utf-8")

    chain._write_gpu_status(status_step, plan_steps, tmp_path,
                            {"early": ["a", "b", "--out", "early.json"],
                             "gci": ["--out", "gci.json"],
                             "gpu_C2": run["args"]},
                            {"early": early_done, "gci": True, "gpu_C2": True})
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert status["status"] == "incomplete"
    assert status["early_gpu_runs"] == []
    assert status["reasons"]
