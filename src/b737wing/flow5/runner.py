"""Run flow5's isolated foil and plane analyses and stage validated polars."""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import math
import re
import sys

from .cases import case_config
from .parse import alpha_gap_diagnostics, parse_foil_polar, parse_op_points, parse_plane_polar
from .xmlgen import project_root, script_xml, write_case_files

from b737wing.config import B737_FLOW5_EXE

FLOW5_EXE = B737_FLOW5_EXE
DEFAULT_TIMEOUT = 1800.0


@dataclass
class RunResult:
    pass_number: int
    returncode: int
    stdout: str
    stderr: str
    elapsed: float
    launched_at: float
    script: Path
    stdout_log: Path
    stderr_log: Path
    success: bool
    reason: str | None = None


def classify_output(text: str, returncode: int, pass_number: int) -> tuple[bool, str | None]:
    if returncode != 0:
        return False, f"flow5 exited with status {returncode}"
    fatal_markers = [
        "Error reading script...aborting", "The file is not an xml readable script",
        "Expected character data.", "foils not found ...discarding this plane",
        "error: reference chord length is 0m", "error: reference span length is 0m",
        "error: reference area is 0m²", "LLT analysis completed ... Errors encountered",
        "OTF failures:",
    ]
    # Opus F3 decision (2026-10-03): op points flow5 discards (strip alpha beyond the
    # foil polar, i.e. past CLmax) are not fatal; post.py checks the target bracket and
    # flags a CLmax-limited result. They count as completion with a warning.
    partial_markers = ("Panel analysis completed ... Errors encountered",
                       "Error generating the operating point... discarding",
                       "Viscous interpolation failures")
    if pass_number == 2:
        fatal_markers.append("Made 0 valid analysis pairs (plane, polar)")
    found = next((marker for marker in fatal_markers if marker in text), None)
    if found:
        return False, f"flow5 reported failure marker: {found}"
    # Windows (verified 2026-10-03): flow5 writes nothing to stdout; the project log
    # <output_dir>/<project>/<project>.log carries the progress text without the
    # "Script imported"/"Script completed" lines. Accept either form.
    if "Script imported, no parsing error" not in text and "flow5 v" not in text:
        return False, "flow5 output/log did not confirm that it read the script"
    if pass_number == 1 and "Foil analysis completed" not in text:
        return False, "flow5 log did not report foil analysis completion"
    if pass_number == 2 and not ("Panel analysis completed successfully" in text or
                                 "Panel analysis completed ... Errors encountered" in text or
                                 "LLT analysis completed successfully" in text):
        return False, "flow5 output/log did not report successful panel or LLT analysis"
    if "----- Script completed -----" not in text and "Boat analyses completed" not in text:
        return False, "flow5 output/log did not report script completion"
    partial = next((marker for marker in partial_markers if marker in text), None)
    return True, (f"warning: {partial}" if partial else None)


def project_log_text(script: Path, launched_at: float) -> str:
    """Text of flow5's project log(s) for a script (the Windows progress channel)."""
    text = script.read_text(encoding="utf-8", errors="replace")
    out_dir = re.search(r"<output_dir>([^<]*)</output_dir>", text, re.I)
    project = re.search(r"<project_file_name>([^<]*)</project_file_name>", text, re.I)
    if not out_dir:
        return ""
    root = Path(out_dir.group(1).strip())
    if project:
        root = root / project.group(1).strip()
    logs = sorted(path for path in root.glob("*.log")
                  if path.is_file() and path.stat().st_mtime > launched_at) if root.exists() else []
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)


def run_script(script: str | Path, *, pass_number: int, exe: str | Path = FLOW5_EXE,
               timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    script = Path(script).resolve()
    if timeout < 1800.0:
        raise ValueError("real flow5 runs require a timeout of at least 1800 seconds")
    if not script.is_file():
        raise FileNotFoundError(f"flow5 script does not exist: {script}")
    stdout_log = script.parent / f"pass{pass_number}.stdout.log"
    stderr_log = script.parent / f"pass{pass_number}.stderr.log"
    started = time.monotonic()
    launch_options = {}
    if sys.platform == "win32":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        launch_options["startupinfo"] = startup
        launch_options["creationflags"] = subprocess.CREATE_NO_WINDOW
    launched_at = time.time()
    try:
        completed = subprocess.run([str(exe), "-p", "-s", str(script)], stdin=subprocess.DEVNULL,
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout,
                                   **launch_options)
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        raise TimeoutError(f"flow5 pass {pass_number} exceeded {timeout:g}s; logs: {stdout_log}, {stderr_log}") from error
    elapsed = time.monotonic() - started
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")
    combined = stdout + "\n" + stderr + "\n" + project_log_text(script, launched_at)
    success, reason = classify_output(combined, returncode, pass_number)
    return RunResult(pass_number, returncode, stdout, stderr, elapsed, launched_at,
                     script, stdout_log, stderr_log, success, reason)


def stage_foil_polars(produced_root: str | Path, staged_dir: str | Path,
                      *, alpha_step: float = 0.25) -> tuple[list[Path], list[dict]]:
    produced_root, staged_dir = Path(produced_root), Path(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)
    source_roots = sorted(produced_root.rglob("Foil_polars")) if produced_root.exists() else []
    if not source_roots:
        raise RuntimeError(f"pass 1 produced no Foil_polars directory under {produced_root}")
    staged: list[Path] = []
    gaps: list[dict] = []
    for source_root in source_roots:
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or source.suffix.casefold() not in (".txt", ".csv"):
                continue
            target = staged_dir / f"{source.parent.name}__{source.stem}.txt"
            shutil.copyfile(source, target)
            polar = parse_foil_polar(target)
            if len(polar.alpha) < 2:
                raise RuntimeError(f"staged polar has fewer than two converged points: {target.name}")
            gaps.extend(alpha_gap_diagnostics(polar, alpha_step))
            staged.append(target)
    if not staged:
        raise RuntimeError(f"pass 1 produced no 2D foil polar text files under {produced_root}")
    return staged, gaps


def _foil_files(case_name: str, run_root: Path, foil_sources: dict[str, Path] | None = None) -> list[str]:
    case = case_config(case_name)
    source_dir = project_root() / "aerofoils_v2"
    target_dir = run_root / "foils"
    target_dir.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(section["foil"] for section in case["sections"]))
    if case.get("flap"):
        names = ["AOPT", "AOPT_F"]
    result = []
    for name in names:
        source = Path((foil_sources or {}).get(name, source_dir / f"{name}.dat"))
        if not source.is_file():
            raise FileNotFoundError(f"airfoil {name} is not available at {source}")
        content_name = source.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
        if content_name != name:
            raise ValueError(f"airfoil file {source} identifies itself as {content_name!r}, expected {name!r}")
        shutil.copyfile(source, target_dir / f"{name}.dat")
        result.append(f"{name}.dat")
    return result


def _raise_failed(result: RunResult) -> None:
    if not result.success:
        tail = (result.stdout + "\n" + result.stderr)[-1600:]
        raise RuntimeError(f"flow5 pass {result.pass_number} failed: {result.reason}\n"
                           f"stdout log: {result.stdout_log}\nstderr log: {result.stderr_log}\n{tail}")


def find_plane_polar(run_root: str | Path, case_name: str, mesh: str,
                     run_label: str | None = None, *, launched_at: float) -> tuple[Path, Path]:
    case = case_config(case_name)
    label = run_label or mesh
    project = Path(run_root) / "pass2" / "flow5-output" / f"{case['name']}_{label}_pass2"
    plane_dir = project / case["plane_name"]
    expected = plane_dir / f"{case['polar_name']}.csv"
    if expected.is_file() and expected.stat().st_mtime > launched_at:
        return project, expected
    candidates = sorted(path for path in plane_dir.glob("*.csv")
                        if path.is_file() and path.stat().st_mtime > launched_at) if plane_dir.is_dir() else []
    if len(candidates) == 1:
        return project, candidates[0]
    raise RuntimeError(f"cannot identify the T1 plane polar under {plane_dir}; found {len(candidates)} candidates")


def verify_fluid_output(case_name: str, project_dir: str | Path, polar_path: str | Path) -> dict:
    """Cross-check the supplied fluid values against the header and local strip Re."""
    case = case_config(case_name)
    flight = case["flight"]
    polar_path = Path(polar_path)
    table = parse_plane_polar(polar_path)
    density_echo = None
    for key, value in table.header.items():
        if re.search(r"rho|ρ|density", key, re.I):
            numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value.replace(",", " "))
            if numbers:
                density_echo = float(numbers[-1])
                break
    plane_dir = Path(project_dir) / case["plane_name"]
    op_files = [path for path in plane_dir.rglob("*.csv") if path.parent != plane_dir]
    points = parse_op_points(op_files, expected_polar=case["polar_name"])
    if not points:
        raise ValueError("fluid verification found no operating point for the generated polar")
    strip = next(iter(points[0].strips.values()))
    y_index = next((i for i, label in enumerate(strip.columns) if label.casefold() == "y(m)"), None)
    re_index = next((i for i, label in enumerate(strip.columns) if label.casefold() == "re"), None)
    if y_index is None or re_index is None:
        raise ValueError(f"fluid verification requires y(m) and Re strip columns, found {strip.columns}")
    sections = case["sections"]
    relative_errors = []
    for row in strip.rows:
        y = abs(row[y_index])
        left, right = sections[0], sections[-1]
        for candidate_left, candidate_right in zip(sections, sections[1:]):
            if candidate_left["y"] <= y <= candidate_right["y"]:
                left, right = candidate_left, candidate_right
                break
        fraction = min(1.0, max(0.0, (y - left["y"]) / (right["y"] - left["y"])))
        chord = left["chord"] + fraction * (right["chord"] - left["chord"])
        expected_re = chord * float(flight["velocity"]) / float(flight["nu"])
        relative_errors.append(abs(row[re_index] - expected_re) / expected_re)
    max_re_error = max(relative_errors, default=math.inf)
    density_error = math.inf if density_echo is None else abs(density_echo - flight["rho"]) / flight["rho"]
    return {"density_echo": density_echo, "density_relative_error": density_error,
            "max_re_relative_error": max_re_error,
            "ok": density_error < 0.01 and max_re_error < 0.01,
            "strip_count": len(relative_errors)}


def run_case(case_name: str, *, mesh: str = "medium", pass_mode: str = "all",
             alpha: tuple[float, float, float] | None = None, output_root: str | Path | None = None,
             method: str = "TRIUNIFORM", wake_spans: float = 20.0,
             reynolds: list[float] | None = None, foil_sources: dict[str, Path] | None = None,
             exe: str | Path = FLOW5_EXE, timeout: float = DEFAULT_TIMEOUT,
             viscosity: float | None = None, post: bool = True,
             run_label: str | None = None) -> dict:
    """Run pass 1, pass 2, or both for one case and mesh."""
    case = case_config(case_name)
    if pass_mode not in ("1", "2", "all"):
        raise ValueError("pass_mode must be 1, 2, or all")
    output_root = Path(output_root) if output_root is not None else project_root() / "flow5_v2"
    case_output = output_root / case["name"]
    run_root = case_output / (run_label or mesh)
    run_root.mkdir(parents=True, exist_ok=True)
    foil_files = _foil_files(case_name, run_root, foil_sources)
    _, _, pass1_script, pass2_script = write_case_files(case_name, run_root, mesh,
                                                       method=method, wake_spans=wake_spans,
                                                       viscosity=viscosity)
    pass1_dir = run_root / "pass1"
    pass2_dir = run_root / "pass2"
    alpha_step = alpha[2] if alpha is not None else 0.25
    alpha_gaps: list[dict] = []
    if pass_mode in ("1", "all"):
        pass1_script.write_text(script_xml(case_name, 1, run_root=run_root, foil_files=foil_files,
                                           mesh=mesh, reynolds=reynolds,
                                           project_name=f"{case['name']}_{run_label or mesh}_pass1"), encoding="utf-8")
        output_dir = pass1_dir / "flow5-output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        first = run_script(pass1_script, pass_number=1, exe=exe, timeout=timeout)
        _raise_failed(first)
        _, alpha_gaps = stage_foil_polars(pass1_dir / "flow5-output", run_root / "xfoil_polars",
                                          alpha_step=alpha_step)
    if pass_mode in ("2", "all"):
        staged_dir = run_root / "xfoil_polars"
        if not any(staged_dir.glob("*.txt")):
            if pass_mode == "2":
                _, alpha_gaps = stage_foil_polars(pass1_dir / "flow5-output", staged_dir,
                                                  alpha_step=alpha_step)
            else:
                raise RuntimeError("pass 1 completed without staging any XFoil text polars")
        elif pass_mode == "2":
            alpha_gaps = [diagnostic for path in sorted(staged_dir.glob("*.txt"))
                          for diagnostic in alpha_gap_diagnostics(parse_foil_polar(path), alpha_step)]
        pass2_script.write_text(script_xml(case_name, 2, run_root=run_root, foil_files=foil_files,
                                           mesh=mesh, alpha=alpha,
                                           project_name=f"{case['name']}_{run_label or mesh}_pass2"), encoding="utf-8")
        output_dir = pass2_dir / "flow5-output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        second = run_script(pass2_script, pass_number=2, exe=exe, timeout=timeout)
        _raise_failed(second)
        result = {"pass1": first if pass_mode == "all" else None, "pass2": second}
        if post:
            from .post import postprocess
            project_dir, polar_path = find_plane_polar(run_root, case_name, mesh, run_label,
                                                       launched_at=second.launched_at)
            result["post"] = postprocess(case_name, polar_path, project_dir, case_output,
                                         run_root / "xfoil_polars", alpha_gaps=alpha_gaps,
                                         alpha_step=alpha_step)
        result["alpha_gap_diagnostics"] = alpha_gaps
        return result
    return {"pass1": first, "alpha_gap_diagnostics": alpha_gaps}
