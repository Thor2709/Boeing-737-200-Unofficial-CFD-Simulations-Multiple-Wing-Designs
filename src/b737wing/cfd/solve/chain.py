"""Run a planned CFD command chain without overlapping Fluent sessions."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from b737wing.config import REPO_ROOT
    PROJECT_ROOT = REPO_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODULES = {
    "run_case": "b737wing.cfd.solve.run_case",
    "sweep": "b737wing.cfd.solve.sweep",
    "compare": "b737wing.cfd.solve.compare_runs",
    "gci": "b737wing.cfd.solve.gci",
    "post": "b737wing.post.case_post",
}
BUILTIN_COMMANDS = {"gpu_status"}
DEFAULT_TIMEOUTS = {"run_case": 8 * 60 * 60, "sweep": 60 * 60, "gci": 60 * 60}
POST_TIMEOUT_S = 2 * 60 * 60
PID_WAIT_TIMEOUT_S = 5 * 60
STALL_TIMEOUT_S = 45 * 60
PROCESS_POLL_S = 5
SEARCH_ITERATION_SECONDS = 4.0
SEARCH_TIMEOUT_BUFFER_S = 60 * 60
LOG_FIELDS = ("iso_time", "step_id", "cmd", "resolved_args", "status",
              "exit_code", "result_status", "seconds")
PLACEHOLDER = re.compile(r"\{(alpha|flow5_alpha|seed_alpha):([^{}]+)\}")


def _path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _out_arg(args):
    for index, value in enumerate(args):
        if value == "--out" and index + 1 < len(args):
            return args[index + 1]
        if value.startswith("--out="):
            return value.split("=", 1)[1]
    return None


def _run_out(step, root, args=None):
    value = _out_arg(step["args"] if args is None else args)
    return _path(value, root) if value else None


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_placeholder(match, root):
    kind, spec = match.groups()
    if kind == "alpha":
        result_path = _path(spec, root) / "result.json"
        try:
            alpha = _read_json(result_path)["alpha_deg"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"cannot read alpha_deg from {result_path}: {exc}") from exc
        return str(alpha)

    if kind == "seed_alpha":
        case, separator, seed_file = spec.partition(":")
        if not separator or case not in {f"C{number}" for number in range(1, 6)}:
            raise ValueError(f"invalid seed_alpha placeholder: {{{kind}:{spec}}}")
        seed_path = _path(seed_file, root)
        try:
            alpha = _read_json(seed_path)["cases"][case]["alpha0_CFD_deg"]
            if isinstance(alpha, bool):
                raise ValueError("alpha0_CFD_deg is not numeric")
            alpha = float(alpha)
            if not math.isfinite(alpha):
                raise ValueError("alpha0_CFD_deg is not finite")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"cannot read {case} alpha0_CFD_deg from {seed_path}: {exc}") from exc
        return f"{alpha:.4f}"

    case, separator, reference = spec.partition(":")
    if not separator or case not in {f"C{number}" for number in range(1, 6)}:
        raise ValueError(f"invalid flow5_alpha placeholder: {{{kind}:{spec}}}")
    reference_path = _path(reference, root) / "result.json"
    try:
        reference_alpha = float(_read_json(reference_path)["alpha_deg"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read reference alpha from {reference_path}: {exc}") from exc

    summary_path = root / "results" / "flow5" / "summary.csv"
    if not summary_path.is_file() and (root / "flow5_v2" / "summary.csv").is_file():
        summary_path = root / "flow5_v2" / "summary.csv"
    target_case = f"F{case[1:]}"
    alphas = {}
    try:
        with summary_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("mesh") != "fine" or row.get("case") not in {target_case, "F5"}:
                    continue
                try:
                    if float(row.get("wake_spans", "nan")) == 20.0:
                        alphas[row["case"]] = float(row["alpha_deg"])
                except (ValueError, TypeError):
                    continue
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"cannot read flow5 alpha from {summary_path}: {exc}") from exc
    if target_case not in alphas or "F5" not in alphas:
        raise ValueError(f"{summary_path} has no fine, 20.0-span alpha for {target_case} and F5")
    return f"{reference_alpha + (alphas[target_case] - alphas['F5']):.3f}"


def _resolve_arg(value, root):
    resolved = PLACEHOLDER.sub(lambda match: _resolve_placeholder(match, root), value)
    leftover = re.search(r"\{[^{}]+\}", resolved)
    if leftover:
        raise ValueError(f"unsupported placeholder {leftover.group(0)}")
    return resolved


def _condition(plan_steps, step, root, outcomes=None, resolved_args=None):
    condition = step.get("when")
    if condition is None:
        return True, ""

    def matches(item):
        source = plan_steps.get(item["step"])
        if source is None:
            return False, f"when step {item['step']} is unknown"
        if outcomes is not None and outcomes.get(source["id"]) is not True:
            return False, f"when gate {item['gate']} from {item['step']} is unavailable"
        source_args = resolved_args.get(source["id"]) if resolved_args else None
        output = _run_out(source, root, source_args)
        if source["cmd"] == "compare" and output is None:
            output = root / "compare.json"
        try:
            report = _read_json(output) if output else {}
            actual = report["gates"][item["gate"]]["verdict"]
        except (OSError, ValueError, KeyError, TypeError):
            return False, f"when gate {item['gate']} from {item['step']} is unavailable"
        expected = item["verdict"]
        return actual == expected, f"{item['gate']}={actual}; expected {expected}"

    conditions = condition if isinstance(condition, list) else [condition]
    results = [matches(item) for item in conditions]
    return all(result for result, _ in results), "; ".join(reason for _, reason in results)


def _decision_gate_names(plan_steps, step_id, root=None):
    names = []
    source = plan_steps.get(step_id)
    source_output = _run_out(source, root) if source is not None and root is not None else None
    for consumer in plan_steps.values():
        if consumer.get("cmd") == "gci" and source_output is not None:
            warm_compare = _option_value(consumer.get("args", []), "--warm-compare")
            if warm_compare and _path(warm_compare, root).resolve() == source_output.resolve():
                names.append("warm_start")
        condition = consumer.get("when")
        conditions = condition if isinstance(condition, list) else [condition]
        for item in conditions:
            if isinstance(item, dict) and item.get("step") == step_id:
                names.append(item["gate"])
    return list(dict.fromkeys(names))


def _write_failed_decision_report(step, gate_names, reason, root, args):
    output = _run_out(step, root, args)
    if step["cmd"] == "compare" and output is None:
        output = root / "compare.json"
    if output is None:
        raise ValueError(f"decision gate {step['id']} has no output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if step["cmd"] == "gci":
        warm_path = _option_value(args, "--warm-compare")
        warm_pass = False
        if warm_path:
            try:
                warm_report = _read_json(_path(warm_path, root))
                warm_pass = (warm_report["gates"]["warm_start"]["verdict"] == "pass")
            except (OSError, ValueError, KeyError, TypeError):
                warm_pass = False
        gates = {
            "gpu_within_gci": {"verdict": "fail", "reason": reason},
            "gpu_batch": {"verdict": "pass" if warm_pass else "fail",
                          "reason": "warm_start gate passes" if warm_pass else reason},
            "cpu_warm": {"verdict": "pass" if warm_pass else "fail",
                         "reason": "GCI fails and warm_start passes" if warm_pass else reason},
            "cpu_cold": {"verdict": "fail" if warm_pass else "pass",
                         "reason": "requires GCI fail and warm_start fail" if warm_pass
                         else "GCI producer failed and warm_start gate fails"},
        }
        report = {"gates": gates, "quantities": {}, "producer_failure": reason}
    else:
        report = {"gates": {name: {"verdict": "fail", "reason": reason}
                            for name in gate_names},
                  "producer_failure": reason}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _done_artifact(step, root, filename, args=None):
    output = _run_out(step, root, args)
    if output is None:
        return None
    try:
        artifact = _read_json(output / filename)
    except (OSError, ValueError, TypeError):
        return None
    return artifact if isinstance(artifact, dict) else None


def _done_report(step, root, args=None):
    output = _run_out(step, root, args)
    try:
        artifact = _read_json(output) if output else None
    except (OSError, ValueError, TypeError):
        return None
    return artifact if isinstance(artifact, dict) else None


def _gate_snapshot(step, gate, root, args=None):
    output = _run_out(step, root, args)
    if step["cmd"] == "compare" and output is None:
        output = root / "compare.json"
    try:
        report = _read_json(output) if output else {}
        outcome = report["gates"][gate]
        verdict = outcome["verdict"]
        reason = outcome.get("reason", "")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{gate} report from {step['id']} is unavailable: {exc}") from exc
    if verdict not in {"pass", "fail"}:
        raise ValueError(f"{gate} report from {step['id']} has invalid verdict {verdict!r}")
    return {"verdict": verdict, "reason": reason}


def _gpu_result_qualified(result):
    return (isinstance(result, dict) and result.get("status") == "converged"
            and result.get("converged") is True
            and result.get("solver_audit") == []
            and result.get("gpu_native_solver_active") is True)


def _write_gpu_status(step, plan_steps, root, resolved_args, outcomes):
    early_step = plan_steps[step["early_gpu_gate_step"]]
    gci_step = plan_steps[step["gci_gate_step"]]
    reasons = []
    try:
        early_gpu = _gate_snapshot(early_step, "early_gpu", root,
                                   resolved_args.get(early_step["id"]))
    except ValueError as exc:
        early_gpu = {"verdict": "unavailable", "reason": str(exc)}
    try:
        gci = _gate_snapshot(gci_step, "gpu_within_gci", root,
                             resolved_args.get(gci_step["id"]))
    except ValueError as exc:
        gci = {"verdict": "unavailable", "reason": str(exc)}
    if outcomes.get(early_step["id"]) is not True:
        reasons.append(f"early GPU gate {early_step['id']} did not complete in this chain run")
    if outcomes.get(gci_step["id"]) is not True:
        reasons.append(f"GCI gate {gci_step['id']} did not complete in this chain run")
    early_runs = []
    if early_gpu["verdict"] == "pass" and outcomes.get(early_step["id"]) is True:
        for run_id in step["early_gpu_run_steps"]:
            run_step = plan_steps[run_id]
            output = _run_out(run_step, root, resolved_args.get(run_id))
            if output is None:
                reasons.append(f"early GPU run {run_id} has no --out path")
                continue
            result = _done_result(run_step, root, resolved_args.get(run_id))
            if not _gpu_result_qualified(result):
                run_status = result.get("status") if isinstance(result, dict) else "missing result"
                reasons.append(f"early GPU run {run_id} is not converged and audit-passed "
                               f"(status={run_status!r})")
                continue
            early_runs.append(_out_arg(run_step["args"]))
    if early_gpu["verdict"] == "pass" and len(early_runs) != len(step["early_gpu_run_steps"]):
        reasons.append("one or more early GPU runs are not authoritative")
    if early_gpu["verdict"] == "unavailable":
        reasons.append(early_gpu["reason"])
    if gci["verdict"] == "unavailable":
        reasons.append(gci["reason"])
    status = ("incomplete" if reasons else
              "final" if gci["verdict"] == "pass" else "provisional_gci_fail")
    output = _run_out(step, root)
    if output is None:
        raise ValueError("gpu_status step has no --out path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"early_gpu": early_gpu, "gci": gci,
                                  "early_gpu_runs": early_runs,
                                  "status": status, "reasons": reasons},
                      indent=2), encoding="utf-8")


def _dry_branch_order(steps, root, early_verdict):
    preview = []
    for step in steps:
        condition = step.get("when")
        conditions = condition if isinstance(condition, list) else [condition]
        early_conditions = [item for item in conditions if isinstance(item, dict)
                            and item.get("gate") == "early_gpu"]
        if any(item.get("verdict") != early_verdict for item in early_conditions):
            continue
        resolved, errors = [], []
        for arg in step["args"]:
            try:
                resolved.append(_resolve_arg(arg, root))
            except ValueError as exc:
                resolved.append(arg)
                errors.append(str(exc))
        preview.append({"id": step["id"], "cmd": step["cmd"], "args": resolved,
                        "when": condition, "resolution_errors": errors})
    return preview


def _done_result(step, root, args=None):
    return _done_artifact(step, root, "result.json", args)


def _option_value(args, option, default=None):
    for index, argument in enumerate(args):
        if argument == option and index + 1 < len(args):
            return args[index + 1]
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
    return default


def _source_identity(value, root, *, directory_files=()):
    if value is None:
        return None
    path = _path(value, root).resolve()
    paths = [path / name for name in directory_files] if path.is_dir() else [path]
    files = []
    for source in paths:
        identity = {"path": str(source)}
        try:
            identity["size"] = source.stat().st_size
            identity["mtime_ns"] = source.stat().st_mtime_ns
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            identity["sha256"] = digest.hexdigest()
        except OSError:
            identity["missing"] = True
        files.append(identity)
    return {"path": str(path), "files": files}


def _convergence_policy_version():
    from . import run_case

    source = Path(run_case.__file__).read_bytes()
    settle_source = inspect.getsource(run_case.settle_check).encode("utf-8")
    return hashlib.sha256(source + b"\0" + settle_source).hexdigest()


def _run_case_provenance(args, root):
    if not args:
        return None
    case = args[0]
    mesh = _option_value(args, "--mesh")
    if not mesh:
        return None
    from .conditions import condition_for_case

    fixed_alpha = _option_value(args, "--fixed-alpha")
    solver = _option_value(args, "--solver", "cpu-db")
    precision = _option_value(args, "--precision", "single" if solver == "gpu-pb" else "double")
    processor_count = int(_option_value(args, "--np", 4))
    if solver == "gpu-pb":
        processor_count = 1
    from . import run_case

    min_final_iters = int(_option_value(
        args, "--min-final-iters", run_case.MIN_FINAL_ITERATIONS))
    courant_ramp = _option_value(args, "--courant-ramp")
    if courant_ramp is None:
        courant_ramp = run_case._ramp_records(run_case.COURANT_RAMP)
    else:
        try:
            courant_ramp = json.loads(courant_ramp)
        except (TypeError, ValueError):
            courant_ramp = str(courant_ramp)
    if fixed_alpha is not None:
        alpha = {"mode": "fixed", "target_deg": float(fixed_alpha)}
    else:
        alpha = {"mode": "target_cl",
                 "target_cl": float(condition_for_case(case)["target_cl"]),
                 "alpha0_deg": (float(_option_value(args, "--alpha0"))
                                if _option_value(args, "--alpha0") is not None else None),
                 "first_slope_per_deg": (
                     float(_option_value(args, "--first-slope"))
                     if _option_value(args, "--first-slope") is not None else None)}
    return {"mesh": str(_path(mesh, root).resolve()), "case": case,
            "alpha": alpha, "solver": solver, "precision": precision,
            "np": processor_count, "min_final_iterations": min_final_iters,
            "first_slope": _option_value(args, "--first-slope"),
            "courant_ramp": courant_ramp,
            "init_data": _source_identity(_option_value(args, "--init-data"), root),
            "interp_from": _source_identity(
                _option_value(args, "--interp-from"), root,
                directory_files=("final.cas.h5", "final.dat.h5")),
            "seed_from": _source_identity(
                _option_value(args, "--seed-from"), root,
                directory_files=("result.json",)),
            "accel": _option_value(args, "--accel", ""),
            "ih": _option_value(args, "--ih"),
            "convergence_policy_version": _convergence_policy_version()}


def _matching_result(result, expected):
    if not expected or not isinstance(result, dict) or result.get("status") != "converged":
        return False
    if result.get("chain_provenance") != expected:
        return False
    if (result.get("case") != expected["case"]
            or result.get("solver") != expected["solver"]
            or result.get("precision") != expected["precision"]):
        return False
    try:
        if Path(result["mesh"]).expanduser().resolve() != Path(expected["mesh"]).resolve():
            return False
        if int(result.get("np")) != expected["np"]:
            return False
        if int(result.get("min_final_iterations")) != expected["min_final_iterations"]:
            return False
        minimum = max(expected["min_final_iterations"], 600)
        rule = result.get("final_iteration_rule")
        convergence = result.get("convergence_numbers")
        if not isinstance(rule, dict) or not isinstance(convergence, dict):
            return False
        if int(rule.get("minimum_iterations")) != minimum:
            return False
        final_iterations = int(convergence.get("final_alpha_iterations"))
        if final_iterations < minimum:
            return False
        if expected["solver"] == "gpu-pb" and not _gpu_result_qualified(result):
            return False
        if expected["alpha"]["mode"] == "fixed":
            return abs(float(result["alpha_deg"]) - expected["alpha"]["target_deg"]) <= 1e-6
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return expected["alpha"]["mode"] == "target_cl"


def _stale_output(output):
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    stale = output.with_name(f"{output.name}_stale_{stamp}")
    if stale.exists():
        raise FileExistsError(f"stale output already exists: {stale}")
    shutil.move(str(output), str(stale))
    return stale


def _pids_from_file(path):
    try:
        data = _read_json(path)
    except FileNotFoundError:
        return []
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict) and isinstance(data.get("pids"), list):
        values = data["pids"]
    else:
        raise ValueError(f"unrecognized Fluent PID file: {path}")
    if any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in values):
        raise ValueError(f"invalid PID in Fluent PID file: {path}")
    return list(dict.fromkeys(values))


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH or getattr(exc, "winerror", None) in {87, 1168}:
            return False
        if exc.errno == errno.EPERM or getattr(exc, "winerror", None) == 5:
            return True
        raise
    return True


def _kill_pid(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        if exc.errno == errno.ESRCH or getattr(exc, "winerror", None) in {87, 1168}:
            return
        raise


def _wait_for_pids(pids, timeout_s=PID_WAIT_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while any(_pid_exists(pid) for pid in pids):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for Fluent PIDs: {pids}")
        time.sleep(min(0.05, remaining))


def _kill_process_tree(pid, process=None):
    running = process is None or process.poll() is None
    if running:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait(timeout=PID_WAIT_TIMEOUT_S)


def _kill_step_process_and_fluent(output, process=None, pid=None):
    pid_file = output / "fluent_pids.json" if output is not None else None
    pids = _pids_from_file(pid_file) if pid_file is not None else []
    for fluent_pid in pids:
        _kill_pid(fluent_pid)
    process_pid = pid if pid is not None else getattr(process, "pid", None)
    if process_pid is not None:
        _kill_process_tree(process_pid, process)
    try:
        _wait_for_pids(pids)
    except TimeoutError:
        for fluent_pid in pids:
            if _pid_exists(fluent_pid):
                _kill_process_tree(fluent_pid)
        raise


def _run_exports(output):
    if output is None:
        return False, "run output path is unavailable"
    if not (output / "walls.csv").is_file():
        return False, "missing walls.csv"
    ensight = output / "ensight"
    if not ensight.is_dir() or not any(ensight.iterdir()):
        return False, "missing or empty ensight directory"
    return True, ""


def _post_run_path(args, root):
    value = _option_value(args, "--run")
    return _path(value, root).resolve() if value else None


def _post_output_path(args, root):
    value = _out_arg(args)
    return _path(value, root).resolve() if value else None


def _finish_post(item, log_path, root, force_timeout=False):
    step, args, process, started = item["step"], item["args"], item["process"], item["started"]
    try:
        if process.poll() is not None:
            waited = process.returncode
        else:
            remaining = max(0.0, POST_TIMEOUT_S - (time.monotonic() - started))
            if force_timeout or remaining <= 0:
                raise subprocess.TimeoutExpired("post", POST_TIMEOUT_S)
            waited = process.wait(timeout=remaining)
        exit_code = getattr(process, "returncode", None)
        if exit_code is None:
            exit_code = waited
        if exit_code is None:
            exit_code = 1
        output = _post_output_path(args, root)
        report = output / "post_report.json" if output else None
        report_exists = report is not None and report.is_file()
        if exit_code == 0 and report_exists:
            status, reason = "done", "post_report.json present"
        elif exit_code != 0:
            status, reason = "failed", f"post exited with code {exit_code}"
        else:
            status, reason = "failed", "missing or unreadable post_report.json"
    except subprocess.TimeoutExpired:
        try:
            _kill_process_tree(process.pid, process)
            exit_code = getattr(process, "returncode", None)
            status, reason = "failed", f"post timed out after {POST_TIMEOUT_S} seconds"
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            exit_code = getattr(process, "returncode", 1)
            status, reason = "failed", f"post timeout cleanup failed: {exc}"
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        exit_code = getattr(process, "returncode", 1)
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
    _append_row(log_path, step, args, status, exit_code, reason,
                time.monotonic() - started)
    return status == "done"


def _referenced_paths(args, root):
    path_options = {"--init-data", "--interp-from", "--mesh",
                    "--run", "--seed-from"}
    consumed_args = []
    skip_next = False
    for argument in args:
        if skip_next:
            skip_next = False
            continue
        if argument == "--out":
            skip_next = True
            continue
        if argument.startswith("--out="):
            continue
        consumed_args.append(argument)
    values = [consumed_args[index + 1] for index, argument in enumerate(consumed_args[:-1])
              if argument in path_options]
    values.extend(argument.split("=", 1)[1] for argument in consumed_args
                  if "=" in argument and argument.split("=", 1)[0] in path_options)
    values.extend(argument for argument in consumed_args
                  if "/" in argument or "\\" in argument)
    for argument in consumed_args:
        for kind, spec in PLACEHOLDER.findall(argument):
            spec = spec.partition(":")[2] if kind in {"flow5_alpha", "seed_alpha"} else spec
            if "/" in spec or "\\" in spec:
                values.append(spec)
    return [_path(value, root).resolve() for value in values]


def _path_producers(earlier_steps, root, resolved_args, step_args):
    producers_by_path = {}
    for path in _referenced_paths(step_args, root):
        producers = []
        for source in earlier_steps:
            output = _run_out(source, root, resolved_args.get(source["id"]))
            if output is None:
                continue
            try:
                output = output.resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if path == output or output in path.parents:
                producers.append(source)
        if producers:
            producers_by_path[path] = producers
    return producers_by_path


def _append_row(log_path, step, args, status, exit_code=None,
                result_status="", seconds=0.0):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({
            "iso_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step_id": step["id"], "cmd": step["cmd"],
            "resolved_args": json.dumps(args, ensure_ascii=False),
            "status": status, "exit_code": "" if exit_code is None else exit_code,
            "result_status": result_status, "seconds": f"{seconds:.3f}",
        })


def _step_log_path(step, args, log_path, root):
    output = _out_arg(args)
    if output:
        directory = _path(output, root)
        if step["cmd"] in {"compare", "gci", "gpu_status"}:
            directory = directory.parent
    else:
        directory = log_path.parent
    return directory / "chain_step.log"


def _plan_data(plan_path, root):
    plan = _read_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ValueError("plan must be an object with a steps list")
    if not isinstance(plan.get("log"), str):
        raise ValueError("plan must contain a log path")
    steps = {}
    for step in plan["steps"]:
        if (not isinstance(step, dict) or not isinstance(step.get("id"), str)
                or (step.get("cmd") not in MODULES
                    and step.get("cmd") not in BUILTIN_COMMANDS)
                or not isinstance(step.get("args"), list)
                or not all(isinstance(arg, str) for arg in step["args"])):
            raise ValueError("each step needs a unique id, supported cmd, and string args")
        if step["id"] in steps:
            raise ValueError(f"duplicate step id: {step['id']}")
        if "timeout_s" in step:
            timeout = step["timeout_s"]
            if (step["cmd"] not in DEFAULT_TIMEOUTS
                    or isinstance(timeout, bool)
                    or not isinstance(timeout, (int, float))
                    or not math.isfinite(timeout) or timeout <= 0):
                raise ValueError(f"invalid timeout_s on step {step['id']}")
        if "when" in step:
            condition = step["when"]
            conditions = condition if isinstance(condition, list) else [condition]
            if (not conditions or any(
                    not isinstance(item, dict) or item.get("gate") not in
                    {"warm_start", "gpu_only", "early_gpu", "gpu_within_gci",
                     "gpu_batch", "cpu_warm", "cpu_cold"}
                    or item.get("verdict") not in {"pass", "fail"}
                    or not isinstance(item.get("step"), str)
                    for item in conditions)):
                raise ValueError(f"invalid when condition on step {step['id']}")
        if step["cmd"] == "gpu_status":
            run_steps = step.get("early_gpu_run_steps")
            if (not isinstance(step.get("early_gpu_gate_step"), str)
                    or not isinstance(step.get("gci_gate_step"), str)
                    or not isinstance(run_steps, list)
                    or not all(isinstance(run_id, str) for run_id in run_steps)
                    or _out_arg(step["args"]) is None):
                raise ValueError(f"invalid gpu_status step {step['id']}")
        steps[step["id"]] = step
    for step in steps.values():
        if step["cmd"] == "gpu_status":
            references = [step["early_gpu_gate_step"], step["gci_gate_step"],
                          *step["early_gpu_run_steps"]]
            if any(reference not in steps for reference in references):
                raise ValueError(f"gpu_status step {step['id']} references an unknown step")
    return plan, steps


def _timeout_for_step(step, args):
    if "timeout_s" in step:
        return step["timeout_s"]
    if step["cmd"] == "run_case" and _option_value(args, "--fixed-alpha") is None:
        from .search import MAX_ITERATIONS_PER_ALPHA, MAX_SECANT_STEPS

        return ((MAX_SECANT_STEPS + 1) * MAX_ITERATIONS_PER_ALPHA
                * SEARCH_ITERATION_SECONDS + SEARCH_TIMEOUT_BUFFER_S)
    return DEFAULT_TIMEOUTS[step["cmd"]]


def _run_managed_process(command, launch_args, timeout_s, *, output=None,
                         progress_log=None, watchdog=False, monitor=None,
                         popen=subprocess.Popen):
    process = popen(command, **launch_args)
    started = time.monotonic()
    last_growth = started
    progress_path = None
    progress_size = 0
    while True:
        if monitor is not None:
            monitor()
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code, "", process
        now = time.monotonic()
        if watchdog:
            history = output / "history.csv" if output is not None else None
            candidate = history if history is not None and history.is_file() else progress_log
            try:
                size = candidate.stat().st_size if candidate is not None else 0
            except OSError:
                size = 0
            if candidate != progress_path:
                progress_path, progress_size, last_growth = candidate, size, now
            elif size > progress_size:
                progress_size, last_growth = size, now
            if now - last_growth >= STALL_TIMEOUT_S:
                try:
                    _kill_step_process_and_fluent(output, process)
                except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
                    return getattr(process, "returncode", None), f"stalled; cleanup failed: {exc}", process
                return getattr(process, "returncode", None), "stalled", process
        remaining = timeout_s - (now - started)
        if remaining <= 0:
            try:
                _kill_step_process_and_fluent(output, process)
            except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
                return getattr(process, "returncode", None), f"timeout; cleanup failed: {exc}", process
            return getattr(process, "returncode", None), "timeout", process
        time.sleep(min(PROCESS_POLL_S, remaining))


def run_chain(plan_file, dry=False, runner=None, popen=None):
    root = PROJECT_ROOT
    plan_path = Path(plan_file).expanduser().resolve()
    plan, by_id = _plan_data(plan_path, root)
    log_path = _path(plan["log"], root)
    if dry:
        preview = []
        for step in plan["steps"]:
            resolved, errors = [], []
            for arg in step["args"]:
                try:
                    resolved.append(_resolve_arg(arg, root))
                except ValueError as exc:
                    resolved.append(arg)
                    errors.append(str(exc))
            condition, reason = _condition(by_id, step, root)
            preview.append({"id": step["id"], "cmd": step["cmd"], "args": resolved,
                            "when": condition if "when" in step else None,
                            "when_reason": reason if "when" in step else "",
                            "resolution_errors": errors})
        output = {"plan": str(plan_path), "dry": True, "steps": preview}
        if "early_gpu_gate" in by_id:
            output["branch_orders"] = {
                "early_gpu_pass": _dry_branch_order(plan["steps"], root, "pass"),
                "early_gpu_fail": _dry_branch_order(plan["steps"], root, "fail"),
            }
        print(json.dumps(output, indent=2))
        return 0

    custom_runner = runner is not None
    runner = runner or subprocess.run
    popen = popen or subprocess.Popen
    outcomes = {}
    resolved_args = {}
    incomplete_exports = {}
    pending_posts = []
    failed = False

    def finish_pending(predicate=lambda item: True):
        nonlocal failed
        remaining = []
        for item in pending_posts:
            if predicate(item):
                done = _finish_post(item, log_path, root)
                outcomes[item["step"]["id"]] = done
                failed = failed or not done
            else:
                remaining.append(item)
        pending_posts[:] = remaining

    def expire_pending_posts():
        nonlocal failed
        remaining = []
        for item in pending_posts:
            if time.monotonic() - item["started"] >= POST_TIMEOUT_S:
                _finish_post(item, log_path, root, force_timeout=True)
                outcomes[item["step"]["id"]] = False
                failed = True
            else:
                remaining.append(item)
        pending_posts[:] = remaining

    for step_index, step in enumerate(plan["steps"]):
        raw_args = step["args"]
        try:
            args = [_resolve_arg(arg, root) for arg in raw_args]
            resolution_error = None
        except ValueError as exc:
            args = None
            resolution_error = exc
        condition, reason = _condition(by_id, step, root, outcomes, resolved_args)
        if not condition:
            _append_row(log_path, step, raw_args, "skipped", result_status=reason)
            outcomes[step["id"]] = None  # skipped: dependents skip via _condition, not blocked
            resolved_args[step["id"]] = args if args is not None else raw_args
            continue

        if resolution_error is None and step["cmd"] == "post":
            run_path = _post_run_path(args, root)
            finish_pending(lambda item: item["run_path"] == run_path)

        if resolution_error is None:
            consumed_paths = _referenced_paths(args, root)
            finish_pending(lambda item: item["output_path"] is not None and any(
                path == item["output_path"] or item["output_path"] in path.parents
                for path in consumed_paths))

        earlier_steps = plan["steps"][:step_index]
        producers_by_path = _path_producers(earlier_steps, root, resolved_args,
                                            args if args is not None else raw_args)
        dependencies = []
        for producers in producers_by_path.values():
            if not any(outcomes.get(source["id"]) is True for source in producers):
                dependencies.extend(producers)
        dependencies = list(dict.fromkeys(source["id"] for source in dependencies))
        gate_names = _decision_gate_names(by_id, step["id"], root)
        decision_gate = bool(gate_names) and step["cmd"] in {"compare", "gci"}
        dependency_gate_reason = (
            f"decision gate {step['id']} failed closed because producer step(s) "
            f"failed, timed out, or were blocked: {', '.join(dependencies)}"
            if dependencies and decision_gate else None)
        if dependencies and not decision_gate:
            reason = "blocked by " + ", ".join(dependencies)
            _append_row(log_path, step, raw_args, "blocked", result_status=reason)
            outcomes[step["id"]] = False
            resolved_args[step["id"]] = args if args is not None else raw_args
            failed = True
            continue

        if resolution_error is not None and dependency_gate_reason is None:
            _append_row(log_path, step, raw_args, "failed", result_status=str(resolution_error))
            outcomes[step["id"]] = False
            resolved_args[step["id"]] = raw_args
            failed = True
            continue

        resolved_args[step["id"]] = args
        if step["cmd"] == "post":
            run_path = _post_run_path(args, root)
            if run_path in incomplete_exports:
                reason = f"done_incomplete_exports: {incomplete_exports[run_path]}"
                _append_row(log_path, step, args, "skipped", result_status=reason)
                outcomes[step["id"]] = None
                continue

        prior_result = _done_result(step, root, args) if step["cmd"] == "run_case" else None
        expected_provenance = (_run_case_provenance(args, root)
                               if step["cmd"] == "run_case" else None)
        if (step.get("skip_if_done")
                    and _matching_result(prior_result, expected_provenance)):
            output = _run_out(step, root, args)
            exports_complete, export_reason = _run_exports(output)
            if exports_complete:
                _append_row(log_path, step, args, "skipped", result_status="converged")
                outcomes[step["id"]] = True
            else:
                _append_row(log_path, step, args, "done_incomplete_exports",
                            result_status=export_reason)
                if output is not None:
                    incomplete_exports[output.resolve()] = export_reason
                outcomes[step["id"]] = True
                failed = True
            continue

        if step.get("skip_if_done") and step["cmd"] == "run_case":
            output = _run_out(step, root, args)
            if output is not None and output.exists():
                _stale_output(output)

        command = (None if step["cmd"] in BUILTIN_COMMANDS else
                   [sys.executable, "-m", MODULES[step["cmd"]], *args])
        step_log = _step_log_path(step, args, log_path, root)
        step_log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        _append_row(log_path, step, args, "started")
        timed_out = False
        stalled = False
        process = None
        try:
            with step_log.open("ab") as stream:
                if step["cmd"] == "gpu_status":
                    _write_gpu_status(step, by_id, root, resolved_args, outcomes)
                    exit_code = 0
                    result_status = "GPU status written"
                elif dependency_gate_reason is not None:
                    _write_failed_decision_report(
                        step, gate_names, dependency_gate_reason, root,
                        args if args is not None else raw_args)
                    exit_code = 0
                    result_status = dependency_gate_reason
                elif step["cmd"] == "post":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    launch_args = {"stdout": stream, "stderr": subprocess.STDOUT,
                                   "cwd": str(root)}
                    if os.name == "nt":
                        launch_args["creationflags"] = creationflags
                    else:
                        launch_args["start_new_session"] = True
                    process = popen(command, **launch_args)
                    pending_posts.append({"step": step, "args": args, "process": process,
                                          "started": started,
                                          "run_path": _post_run_path(args, root),
                                          "output_path": _post_output_path(args, root)})
                    exit_code = None
                    result_status = "background process started"
                elif step["cmd"] in DEFAULT_TIMEOUTS:
                    timeout_s = _timeout_for_step(step, args)
                    if custom_runner:
                        completed = runner(command, stdout=stream, stderr=subprocess.STDOUT,
                                           cwd=str(root), check=False, timeout=timeout_s)
                        exit_code = completed.returncode
                    else:
                        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        launch_args = {"stdout": stream, "stderr": subprocess.STDOUT,
                                       "cwd": str(root)}
                        if os.name == "nt":
                            launch_args["creationflags"] = creationflags
                        else:
                            launch_args["start_new_session"] = True
                        watchdog = (step["cmd"] == "run_case"
                                    and _option_value(args, "--fixed-alpha") is None)
                        exit_code, process_status, process = _run_managed_process(
                            command, launch_args, timeout_s,
                            output=_run_out(step, root, args),
                            progress_log=step_log, watchdog=watchdog,
                            monitor=expire_pending_posts, popen=popen)
                        timed_out = process_status.startswith("timeout")
                        stalled = process_status.startswith("stalled")
                        result_status = process_status
                    if custom_runner:
                        result_status = "timeout" if timed_out else ""
                else:
                    completed = runner(command, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=str(root), check=False)
                    exit_code = completed.returncode
                    result_status = ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            try:
                _kill_step_process_and_fluent(_run_out(step, root, args),
                                              pid=getattr(exc, "pid", None))
            except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as kill_exc:
                result_status = f"timeout; process cleanup failed: {kill_exc}"
            else:
                result_status = "timeout"
            exit_code = None
        except (OSError, ValueError, KeyError, TypeError, TimeoutError,
                subprocess.SubprocessError) as exc:
            exit_code = getattr(exc, "returncode", None)
            result_status = f"{type(exc).__name__}: {exc}"
            exit_code = 1 if exit_code is None else exit_code

        if step["cmd"] == "post" and process is not None:
            outcomes[step["id"]] = True
            continue

        elapsed = time.monotonic() - started
        artifact_name = {"run_case": "result.json", "sweep": "sweep.json"}.get(step["cmd"])
        if step["cmd"] == "gci":
            artifact = _done_report(step, root, args)
            artifact_missing = artifact is None
        else:
            artifact = (_done_artifact(step, root, artifact_name, args)
                        if artifact_name is not None else None)
            artifact_missing = artifact_name is not None and artifact is None

        gate_report_written = False
        if (decision_gate and (timed_out or stalled or exit_code not in (None, 0)
                               or artifact_missing)):
            gate_reason = (f"decision gate {step['id']} failed to produce a usable report; "
                           f"{result_status or f'exit code {exit_code}'}")
            _write_failed_decision_report(step, gate_names, gate_reason, root, args)
            gate_report_written = True

        if (step["cmd"] == "run_case" and not timed_out and exit_code == 0
                and artifact is not None and expected_provenance is not None):
            artifact["chain_provenance"] = expected_provenance
            output = _run_out(step, root, args)
            (output / "result.json").write_text(json.dumps(artifact, indent=2),
                                                encoding="utf-8")

        if stalled:
            result_status = "stalled"
        elif timed_out:
            result_status = "timeout"
        elif artifact_missing:
            result_status = f"missing or unreadable {artifact_name}"
        elif artifact is not None:
            result_status = artifact.get("status", result_status)
        step_failed = (stalled or timed_out or exit_code not in (None, 0)
                       or artifact_missing)
        status = "stalled" if stalled else "failed" if step_failed else "done"
        if (not step_failed and step["cmd"] == "run_case"
                and artifact.get("status") == "converged"):
            exports_complete, export_reason = _run_exports(_run_out(step, root, args))
            if not exports_complete:
                status = "done_incomplete_exports"
                result_status = export_reason
                output = _run_out(step, root, args)
                if output is not None:
                    incomplete_exports[output.resolve()] = export_reason
                failed = True
        _append_row(log_path, step, args, status, exit_code, result_status, elapsed)
        if decision_gate and not gate_report_written:
            try:
                for gate_name in gate_names:
                    _gate_snapshot(step, gate_name, root, args)
                gate_report_written = True
            except ValueError:
                gate_report_written = False
        outcomes[step["id"]] = not step_failed or gate_report_written
        failed = failed or step_failed

    finish_pending()
    return int(failed)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="JSON chain plan")
    parser.add_argument("--dry", action="store_true", help="print resolved steps without running them")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return run_chain(args.plan, dry=args.dry)
    except (OSError, ValueError, KeyError) as exc:
        print(f"chain: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
