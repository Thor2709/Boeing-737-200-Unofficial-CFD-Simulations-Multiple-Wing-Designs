"""Evaluate the C5 three-grid GCI and choose the GPU or CPU finish."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_run(path):
    directory = Path(path).expanduser().resolve()
    result = _read_json(directory / "result.json")
    if not isinstance(result, dict):
        raise ValueError(f"{directory / 'result.json'} is not a JSON object")
    for key in ("alpha_deg", "CL", "CD", "CM", "status"):
        if key not in result:
            raise ValueError(f"{directory / 'result.json'} is missing {key}")
    if result["status"] != "converged":
        raise ValueError(f"{directory} status is {result['status']!r}, not converged")
    for key in ("alpha_deg", "CL", "CD", "CM"):
        value = float(result[key])
        if not math.isfinite(value):
            raise ValueError(f"{directory / 'result.json'} has non-finite {key}")
        result[key] = value
    return directory, result


def _cell_count(run, override):
    if override is not None:
        value = override
    else:
        mesh = Path(run["mesh"]).expanduser()
        if not mesh.is_absolute():
            mesh = mesh.resolve()
        report = _read_json(mesh.parent / "mesh_report.json")
        value = report["cells"]
    if type(value) is not int or value <= 0:
        raise ValueError(f"invalid mesh cell count: {value!r}")
    return value


def _observed_order(e21, e32, r21, r32):
    if e21 == 0.0 or e32 == 0.0:
        return None
    ratio = abs(e32 / e21)
    sign = 1.0 if e21 * e32 > 0.0 else -1.0
    p = 1.0
    for _ in range(1000):
        try:
            numerator = r21 ** p - sign
            denominator = r32 ** p - sign
            if numerator == 0.0 or denominator == 0.0 or numerator * denominator <= 0.0:
                return None
            next_p = (math.log(ratio) + math.log(numerator / denominator)) / math.log(r21)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if not math.isfinite(next_p):
            return None
        if abs(next_p - p) <= 1e-12:
            return next_p
        p = next_p
    return None


def _quantity(fine, medium, coarse, gpu, r21, r32):
    row = {"cpu_fine": fine, "cpu_medium": medium, "cpu_coarse": coarse,
           "gpu": gpu, "r21": r21, "r32": r32, "epsilon21": None,
           "epsilon32": None, "p": None, "convergence": None,
           "extrapolated": None, "gci95": None, "gci925": None,
           "gpu_gap": None}
    values = (fine, medium, coarse, gpu, r21, r32)
    if any(value is None for value in values) or r21 is None or r32 is None:
        return row
    e21, e32 = medium - fine, coarse - medium
    row["epsilon21"], row["epsilon32"] = e21, e32
    p = _observed_order(e21, e32, r21, r32)
    if e21 * e32 < 0.0:
        convergence = "oscillatory"
    elif e21 == 0.0 and e32 != 0.0:
        convergence = "divergent"
    elif e32 == 0.0 or (e21 == 0.0 and e32 == 0.0):
        convergence = "monotonic"
    else:
        convergence = "monotonic" if p is not None and p > 0.0 else "divergent"
    row["convergence"] = convergence
    row["p"] = p
    if fine != 0.0:
        row["gpu_gap"] = abs(gpu - fine) / abs(fine)
    if p is not None:
        denominator = r21 ** p - 1.0
        if denominator > 0.0:
            extrapolated = (r21 ** p * fine - medium) / denominator
            gci_fraction = abs((fine - medium) / (denominator * fine)) if fine else None
            row["extrapolated"] = extrapolated
            if gci_fraction is not None:
                row["gci95"] = 1.25 * gci_fraction
                row["gci925"] = 1.135 * gci_fraction
    return row


def _gates(gci_pass, warm_pass, reason):
    return {
        "gpu_within_gci": {"verdict": "pass" if gci_pass else "fail", "reason": reason},
        "gpu_batch": {"verdict": "pass" if gci_pass or warm_pass else "fail",
                      "reason": "GPU GCI gate passes" if gci_pass else
                      "warm_start gate passes" if warm_pass else
                      "GPU GCI and warm_start gates fail"},
        "cpu_warm": {"verdict": "pass" if not gci_pass and warm_pass else "fail",
                     "reason": "GCI fails and warm_start passes" if not gci_pass and warm_pass
                     else "requires GCI fail and warm_start pass"},
        "cpu_cold": {"verdict": "pass" if not gci_pass and not warm_pass else "fail",
                     "reason": "GCI and warm_start gates fail" if not gci_pass and not warm_pass
                     else "requires GCI fail and warm_start fail"},
    }


def evaluate_gci(fine, medium, coarse, gpu, warm_compare=None, cells=None):
    errors, runs = [], {}
    for name, path in (("fine", fine), ("medium", medium), ("coarse", coarse), ("gpu", gpu)):
        try:
            runs[name] = _read_run(path)[1]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{name} run: {exc}")
    warm_pass = None
    if warm_compare is not None:
        try:
            warm = _read_json(warm_compare)
            warm_pass = (isinstance(warm, dict)
                         and warm["gates"]["warm_start"]["verdict"] == "pass")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            warm_pass = False
            errors.append(f"warm compare: {exc}")
    counts = [None, None, None]
    for index, name in enumerate(("fine", "medium", "coarse")):
        if name not in runs:
            continue
        try:
            counts[index] = _cell_count(runs[name], cells[index] if cells else None)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{name} mesh cells: {exc}")
    r21 = (counts[0] / counts[1]) ** (1.0 / 3.0) if all(counts[:2]) else None
    r32 = (counts[1] / counts[2]) ** (1.0 / 3.0) if all(counts[1:]) else None
    if r21 is not None and r32 is not None and (r21 <= 1.0 or r32 <= 1.0):
        errors.append("cell counts must descend from fine to coarse")
        r21 = r32 = None
    if len({run["alpha_deg"] for run in runs.values()}) > 1:
        errors.append("run alpha_deg values do not match")
    gpu_run = runs.get("gpu", {})
    audit_empty = gpu_run.get("solver_audit") == []
    native_gpu_confirmed = gpu_run.get("gpu_native_solver_active") is True
    quantities = {}
    for key in ("CL", "CD", "CM"):
        quantities[key] = _quantity(
            runs.get("fine", {}).get(key), runs.get("medium", {}).get(key),
            runs.get("coarse", {}).get(key), runs.get("gpu", {}).get(key), r21, r32)
    reasons = []
    if errors:
        reasons.append("; ".join(errors))
    if not audit_empty:
        reasons.append("GPU solver audit is missing or non-empty")
    if not native_gpu_confirmed:
        reasons.append("native GPU solver activation is missing or unconfirmed")
    for key in ("CL", "CD"):
        row = quantities[key]
        valid_p = (row["convergence"] == "monotonic" and row["p"] is not None
                   and 0.5 <= row["p"] <= 4.0)
        if not valid_p:
            reasons.append(f"{key} observed order is invalid")
        if (row["gpu_gap"] is None or row["gci925"] is None
                or row["gpu_gap"] > row["gci925"]):
            reasons.append(f"{key} GPU gap exceeds GCI92.5 or is unavailable")
    gci_pass = not reasons
    reason = "GPU gaps satisfy GCI92.5 for CL and CD" if gci_pass else "; ".join(reasons)
    gates = (_gates(gci_pass, warm_pass, reason) if warm_pass is not None else {
        "gpu_within_gci": {"verdict": "pass" if gci_pass else "fail", "reason": reason}})
    report = {"gates": gates, "quantities": quantities,
              "cells": {"fine": counts[0], "medium": counts[1], "coarse": counts[2]},
              "gpu_solver_audit_empty": audit_empty,
              "gpu_native_solver_confirmed": native_gpu_confirmed}
    if warm_pass is not None:
        report["warm_start"] = "pass" if warm_pass else "fail"
    return report


def _parse_cells(value):
    try:
        cells = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cells must be F,M,C integers") from exc
    if len(cells) != 3 or any(count <= 0 for count in cells):
        raise argparse.ArgumentTypeError("cells must be three positive F,M,C integers")
    return cells


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", required=True)
    parser.add_argument("--medium", required=True)
    parser.add_argument("--coarse", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--warm-compare")
    parser.add_argument("--cells", type=_parse_cells)
    parser.add_argument("--out", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report = evaluate_gci(args.fine, args.medium, args.coarse, args.gpu,
                          args.warm_compare, args.cells)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
