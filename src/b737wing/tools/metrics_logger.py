"""Append periodic host, process, and optional NVIDIA GPU metrics to a CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

try:
    import pynvml
except ImportError:  # The logger remains useful on machines without NVML.
    pynvml = None


CSV_FIELDS = [
    "iso_time",
    "cpu_pct_total",
    "cpu_pct_per_core",
    "ram_used_gb",
    "ram_pct",
    "proc_count",
    "proc_cpu_pct",
    "proc_rss_gb",
    "gpu_util_pct",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
    "gpu_power_w",
    "gpu_temp_c",
]
SCALAR_METRIC_FIELDS = [
    "cpu_pct_total",
    "ram_used_gb",
    "ram_pct",
    "proc_count",
    "proc_cpu_pct",
    "proc_rss_gb",
    "gpu_util_pct",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
    "gpu_power_w",
    "gpu_temp_c",
]
DEFAULT_MATCH = r"fluent|cx|Discovery|flow5|pvbatch"
GIB = 1024**3
PROCESS_ERRORS = (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number greater than zero") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a number greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _compiled_match(value: str) -> re.Pattern[str]:
    try:
        return re.compile(value, re.IGNORECASE)
    except re.error as error:
        raise argparse.ArgumentTypeError(str(error)) from error


class ResourceSampler:
    def __init__(self, process_match: re.Pattern[str]):
        self.process_match = process_match
        self.gpu_handle: Any = None
        self.nvml_initialized = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def close(self) -> None:
        if self.nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_initialized = False

    def _process_metrics(self) -> tuple[int | None, float | None, int | None]:
        count = 0
        cpu_total = 0.0
        rss_total = 0
        cpu_failed = False
        rss_failed = False
        scan_failed = False
        try:
            for process in psutil.process_iter():
                try:
                    name = process.name()
                except Exception:
                    name = ""
                    scan_failed = True
                try:
                    command_line = process.cmdline()
                except Exception:
                    command_line = []
                    scan_failed = True
                if isinstance(command_line, str):
                    command = command_line
                else:
                    command = " ".join(command_line or [])
                if not self.process_match.search(name or "") and not self.process_match.search(command):
                    continue

                count += 1
                try:
                    cpu_total += process.cpu_percent(interval=None)
                except Exception:
                    cpu_failed = True
                try:
                    rss_total += process.memory_info().rss
                except Exception:
                    rss_failed = True
        except Exception:
            return None, None, None
        if scan_failed:
            return None, None, None
        return count, None if cpu_failed else cpu_total, None if rss_failed else rss_total

    def _gpu_metrics(self) -> tuple[Any, Any, Any, Any, Any]:
        empty = (None, None, None, None, None)
        if self.gpu_handle is None or pynvml is None:
            return empty

        values: list[Any] = [None, None, None, None, None]
        try:
            values[0] = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
        except Exception:
            pass
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            values[1] = memory.used / GIB
            values[2] = memory.total / GIB
        except Exception:
            pass
        try:
            values[3] = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
        except Exception:
            pass
        try:
            sensor = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
            values[4] = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, sensor)
        except Exception:
            pass
        return tuple(values)

    def sample(self) -> dict[str, Any]:
        try:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
        except Exception:
            per_core = None
        try:
            cpu_total = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_total = None
        try:
            memory = psutil.virtual_memory()
            ram_used_gb = memory.used / GIB
            ram_percent = memory.percent
        except Exception:
            ram_used_gb = None
            ram_percent = None
        process_count, process_cpu, process_rss = self._process_metrics()
        gpu_values = self._gpu_metrics()
        return {
            "iso_time": datetime.now().astimezone().isoformat(),
            "cpu_pct_total": cpu_total,
            "cpu_pct_per_core": None if per_core is None else ";".join(f"{value:.2f}" for value in per_core),
            "ram_used_gb": ram_used_gb,
            "ram_pct": ram_percent,
            "proc_count": process_count,
            "proc_cpu_pct": process_cpu,
            "proc_rss_gb": None if process_rss is None else process_rss / GIB,
            "gpu_util_pct": gpu_values[0],
            "gpu_mem_used_gb": gpu_values[1],
            "gpu_mem_total_gb": gpu_values[2],
            "gpu_power_w": gpu_values[3],
            "gpu_temp_c": gpu_values[4],
        }


def _core_values(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        return [float(item) for item in value.split(";") if item]
    except ValueError:
        return []


def _aggregate(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"peak": max(values), "mean": statistics.fmean(values)}


def _write_summary(path: Path, rows: list[dict[str, Any]], duration: float) -> None:
    metrics: dict[str, Any] = {}
    for field in SCALAR_METRIC_FIELDS:
        values = [row[field] for row in rows if row[field] is not None]
        metrics[field] = _aggregate(values)

    max_cores = max((len(_core_values(row["cpu_pct_per_core"])) for row in rows), default=0)
    core_summary = []
    for index in range(max_cores):
        values = [cores[index] for row in rows if len(cores := _core_values(row["cpu_pct_per_core"])) > index]
        core_summary.append(_aggregate(values))
    metrics["cpu_pct_per_core"] = core_summary

    summary = {"duration_seconds": duration, "samples": len(rows), "metrics": metrics}
    with path.open("w", encoding="utf-8", newline="") as summary_file:
        json.dump(summary, summary_file, separators=(",", ":"))
        summary_file.write("\n")


def _stop_requested(until_file: Path | None, pid: int | None) -> bool:
    if until_file is not None and until_file.exists():
        return True
    if pid is not None:
        try:
            return not psutil.pid_exists(pid)
        except (psutil.Error, OSError):
            return False
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="CSV output path")
    parser.add_argument("--interval", type=_positive_float, default=5.0, help="sampling interval in seconds")
    parser.add_argument("--until-file", type=Path, help="stop when this file exists")
    parser.add_argument("--pid", type=_positive_int, help="stop when this process exits")
    parser.add_argument("--match", type=_compiled_match, default=_compiled_match(DEFAULT_MATCH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(str(args.out) + ".summary.txt")
    sampler = ResourceSampler(args.match)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        needs_header = not args.out.exists() or args.out.stat().st_size == 0
        with args.out.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
                csv_file.flush()

            first_sample = True
            while first_sample or not _stop_requested(args.until_file, args.pid):
                first_sample = False
                row = sampler.sample()
                rows.append(row)
                writer.writerow(row)
                csv_file.flush()
                if _stop_requested(args.until_file, args.pid):
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.close()
        _write_summary(summary_path, rows, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
