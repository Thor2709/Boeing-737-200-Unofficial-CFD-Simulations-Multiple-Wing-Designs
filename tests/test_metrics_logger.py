from __future__ import annotations

import csv
from io import StringIO
import subprocess
import sys
import time
import uuid

import pytest
from pathlib import Path

from b737wing.tools import metrics_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "aerofoils_v2"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
LOGGER_MODULE = "b737wing.tools.metrics_logger"
GPU_FIELDS = [
    "gpu_util_pct",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
    "gpu_power_w",
    "gpu_temp_c",
]


def test_until_file_writes_rows_and_final_summary():
    prefix = f".metrics_until_{uuid.uuid4().hex}"
    output = OUTPUT_DIRECTORY / f"{prefix}.csv"
    until_file = OUTPUT_DIRECTORY / f"{prefix}.stop"
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                LOGGER_MODULE,
                "--out",
                str(output),
                "--interval",
                "0.5",
                "--until-file",
                str(until_file),
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(3.0)
            until_file.write_text("stop", encoding="utf-8")
            _, stderr = process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

        assert process.returncode == 0, stderr
        with output.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) >= 2
        assert output.with_name(output.name + ".summary.txt").is_file()
    finally:
        output.unlink(missing_ok=True)
        until_file.unlink(missing_ok=True)
        output.with_name(output.name + ".summary.txt").unlink(missing_ok=True)


def test_nvml_init_failure_keeps_rows_with_empty_gpu_fields(monkeypatch):
    class FailedNVML:
        @staticmethod
        def nvmlInit():
            raise RuntimeError("no GPU available")

    monkeypatch.setattr(metrics_logger, "pynvml", FailedNVML())
    prefix = f".metrics_nvml_{uuid.uuid4().hex}"
    output = OUTPUT_DIRECTORY / f"{prefix}.csv"
    until_file = OUTPUT_DIRECTORY / f"{prefix}.stop"
    try:
        until_file.write_text("stop", encoding="utf-8")

        assert metrics_logger.main(
            ["--out", str(output), "--interval", "1", "--until-file", str(until_file)]
        ) == 0

        with output.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) == 1
        assert all(rows[0][field] == "" for field in GPU_FIELDS)
        assert output.with_name(output.name + ".summary.txt").is_file()
    finally:
        output.unlink(missing_ok=True)
        until_file.unlink(missing_ok=True)
        output.with_name(output.name + ".summary.txt").unlink(missing_ok=True)


def test_cpu_access_denied_returns_empty_cpu_values_and_continues(monkeypatch):
    monkeypatch.setattr(metrics_logger, "pynvml", None)
    monkeypatch.setattr(metrics_logger.psutil, "process_iter", lambda: iter(()))

    def denied(*args, **kwargs):
        raise metrics_logger.psutil.AccessDenied(pid=123)

    monkeypatch.setattr(metrics_logger.psutil, "cpu_percent", denied)
    row = metrics_logger.ResourceSampler(metrics_logger._compiled_match(metrics_logger.DEFAULT_MATCH)).sample()
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=metrics_logger.CSV_FIELDS)
    writer.writeheader()
    writer.writerow(row)
    csv_row = next(csv.DictReader(StringIO(output.getvalue(), newline="")))

    assert csv_row["cpu_pct_total"] == ""
    assert csv_row["cpu_pct_per_core"] == ""


def test_ram_access_denied_returns_empty_ram_values_and_continues(monkeypatch):
    monkeypatch.setattr(metrics_logger, "pynvml", None)
    monkeypatch.setattr(metrics_logger.psutil, "process_iter", lambda: iter(()))

    def denied(*args, **kwargs):
        raise metrics_logger.psutil.AccessDenied(pid=123)

    monkeypatch.setattr(metrics_logger.psutil, "virtual_memory", denied)
    row = metrics_logger.ResourceSampler(metrics_logger._compiled_match(metrics_logger.DEFAULT_MATCH)).sample()
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=metrics_logger.CSV_FIELDS)
    writer.writeheader()
    writer.writerow(row)
    csv_row = next(csv.DictReader(StringIO(output.getvalue(), newline="")))

    assert csv_row["ram_used_gb"] == ""
    assert csv_row["ram_pct"] == ""


@pytest.mark.skipif(sys.platform != "win32", reason="PID-exit detection is exercised on Windows only")
def test_pid_exit_stops_logger():
    output = OUTPUT_DIRECTORY / f".metrics_pid_{uuid.uuid4().hex}.csv"
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1.2)"], cwd=PROJECT_ROOT
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                LOGGER_MODULE,
                "--out",
                str(output),
                "--interval",
                "0.1",
                "--pid",
                str(child.pid),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=8,
        )
        assert result.returncode == 0, result.stderr
        assert output.is_file()
        assert output.with_name(output.name + ".summary.txt").is_file()
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        output.unlink(missing_ok=True)
        output.with_name(output.name + ".summary.txt").unlink(missing_ok=True)
