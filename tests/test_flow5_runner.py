from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from b737wing.flow5 import runner
from b737wing.flow5.parse import parse_foil_polar


def test_run_case_removes_existing_solver_output_before_launch(monkeypatch, tmp_path):
    output_root = tmp_path / "flow5_v2"
    output_dir = output_root / "F1" / "coarse" / "pass1" / "flow5-output"
    output_dir.mkdir(parents=True)
    (output_dir / "old-success.log").write_text("old result", encoding="utf-8")
    foil = tmp_path / "A0012.dat"
    foil.write_text("A0012\n0.0 0.0\n", encoding="utf-8")
    observed = []

    def fake_run_script(script, *, pass_number, exe, timeout):
        observed.append(not output_dir.exists())
        return runner.RunResult(
            pass_number=pass_number, returncode=0, stdout="", stderr="", elapsed=0.0,
            launched_at=time.time(), script=Path(script),
            stdout_log=Path(script).with_name("stdout.log"),
            stderr_log=Path(script).with_name("stderr.log"), success=True,
        )

    monkeypatch.setattr(runner, "run_script", fake_run_script)
    monkeypatch.setattr(runner, "stage_foil_polars", lambda *args, **kwargs: ([], []))
    result = runner.run_case("F1", mesh="coarse", pass_mode="1", output_root=output_root,
                             foil_sources={"A0012": foil})

    assert observed == [True]
    assert result["pass1"].success


def test_run_script_ignores_stale_project_log(monkeypatch, tmp_path):
    script = tmp_path / "pass1" / "script.xml"
    log_dir = tmp_path / "pass1" / "flow5-output" / "F1_coarse_pass1"
    log_dir.mkdir(parents=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"<output_dir>{tmp_path / 'pass1' / 'flow5-output'}</output_dir>"
        "<project_file_name>F1_coarse_pass1</project_file_name>", encoding="utf-8")
    log = log_dir / "old.log"
    log.write_text("flow5 v7.57\nFoil analysis completed\n----- Script completed -----\n",
                   encoding="utf-8")
    old_time = time.time() - 60.0
    os.utime(log, (old_time, old_time))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="", stderr="", returncode=0))

    result = runner.run_script(script, pass_number=1, timeout=1800.0)

    assert not result.success
    assert "did not confirm that it read the script" in result.reason


def test_find_plane_polar_rejects_stale_csv(tmp_path):
    run_root = tmp_path / "run"
    project = run_root / "pass2" / "flow5-output" / "F1_coarse_pass2"
    plane_dir = project / "B737-200 F1 Wing"
    plane_dir.mkdir(parents=True)
    stale = plane_dir / "F1.csv"
    stale.write_text("old polar", encoding="utf-8")
    old_time = time.time() - 60.0
    os.utime(stale, (old_time, old_time))

    with pytest.raises(RuntimeError, match="cannot identify the T1 plane polar"):
        runner.find_plane_polar(run_root, "F1", "coarse", launched_at=time.time())


def test_staging_reports_alpha_gaps_wider_than_two_and_a_half_steps(tmp_path):
    produced = tmp_path / "produced" / "Foil_polars" / "B737A"
    produced.mkdir(parents=True)
    source = produced / "B737A_Re20M.txt"
    source.write_text(
        "Calculated polar for: B737A\nRe = 2.000 e 7\nNcrit = 9\n"
        "alpha cl cd cdp cm\n"
        "-8 -0.8 0.08 0.02 -0.01\n"
        "0 0.0 0.01 0.005 0.00\n"
        "16 1.6 0.08 0.02 -0.02\n", encoding="utf-8")

    staged, gaps = runner.stage_foil_polars(produced.parent.parent, tmp_path / "staged",
                                              alpha_step=4.0)

    assert len(staged) == 1
    assert all(item["foil"] == "B737A" and item["re"] == pytest.approx(20_000_000)
               for item in gaps)
    assert [item["gap_deg"] for item in gaps] == [16.0]
    assert "16 deg" in gaps[0]["reason"]
    assert parse_foil_polar(staged[0]).alpha == [-8.0, 0.0, 16.0]


def test_discarded_op_points_are_a_warning_not_fatal():
    classify_output = runner.classify_output
    log = ("flow5 v7.57\nViscous interpolation failures: 3\nError generating the operating point... discarding\n"
           "Panel analysis completed ... Errors encountered\nBoat analyses completed\n")
    ok, reason = classify_output(log, 0, 2)
    assert ok and reason.startswith("warning:")
