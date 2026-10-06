"""Build a Fluent 26.1 Student mesh for one CFD case."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


try:
    from b737wing.config import REPO_ROOT
    ROOT = REPO_ROOT
except ImportError:
    ROOT = Path(__file__).resolve().parents[4]
PYTHON = os.environ.get("PYTHON") or sys.executable
CAP = 1_048_576
CELL_RANGES = {"prod": (800_000, 999_000), "grid": (300_000, 999_000), "sbes": (900_000, 990_000)}
CASE_NAMES = {f"C{i}" for i in range(1, 6)}
TAG_PATTERN = re.compile(r"[A-Za-z0-9_]+")
WALLS = ["mainwing", "horstab", "vertstab", "fuselage", "engine", "nacelle_duct", "pylon", "fairing"]


def validate_case_tag(case, tag):
    if case not in CASE_NAMES:
        raise ValueError(f"invalid case: {case!r}")
    if not isinstance(tag, str) or TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"invalid tag: {tag!r}")


def safe_output_target(root, case, tag):
    """Resolve a case/tag output and reject targets outside that case directory."""
    validate_case_tag(case, tag)
    case_root = (Path(root).resolve() / "cfd_v2" / "mesh" / case).resolve()
    target = (case_root / tag).resolve()
    if target == case_root or case_root not in target.parents:
        raise ValueError(f"output target is not inside {case_root}: {target}")
    return target


def cell_count_in_range(kind, cells):
    bounds = CELL_RANGES.get(kind)
    return bounds is not None and type(cells) is int and bounds[0] <= cells <= bounds[1] and cells < CAP


def require_cell_count(kind, cells):
    if not cell_count_in_range(kind, cells):
        bounds = CELL_RANGES.get(kind)
        raise RuntimeError(f"final interior cell count {cells!r} is outside {kind!r} range {bounds} and below {CAP}")


def make_parameters(scale):
    return dict(
        cmin=0.04 * scale,
        cmax=0.35 * scale,
        ang=10,
        smax=8 * scale,
        hex=6 * scale,
        fh=4e-4,
        nl=12,
        rate=1.2,
        wing_face=0.10 * scale,
        boi_size=0.30 * scale,
        boi_box=(9.5, 0.0, -2.0, 32.0, 15.5, 2.5),
    )


SBES_JUNCTION_BOX = (10.9, 4.6, -0.70, 12.2, 5.5, 0.05)
SBES_JUNCTION_BOI_SIZE = 0.025
SBES_PYLON_FACE_SIZE = 0.04


def sbes_sizing_controls():
    """Extra Add Local Sizing argument sets for the `sbes` mesh kind (fixed sizes, not scaled)."""
    return [
        {"AddChild": "yes", "BOIControlName": "junction_boi", "BOIExecution": "Body Of Influence",
         "BOIFaceLabelList": ["junction_box"], "BOISize": SBES_JUNCTION_BOI_SIZE, "BOIZoneorLabel": "label"},
        {"AddChild": "yes", "BOIControlName": "pylon_face", "BOIExecution": "Face Size",
         "BOIFaceLabelList": ["pylon"], "BOISize": SBES_PYLON_FACE_SIZE, "BOIGrowthRate": 1.2,
         "BOIZoneorLabel": "label"},
    ]


def write_box(path, box):
    x0, y0, z0, x1, y1, z1 = box
    vertices = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    triangles = []
    for a, b, c, d in quads:
        triangles.extend([(a, b, c), (a, c, d)])
    with open(path, "w", encoding="utf-8") as output:
        output.write('(0 "boi box")\n(2 3)\n(10 (0 1 %x 0 3))\n(10 (1 1 %x 1 3)(\n' % (8, 8))
        for point in vertices:
            output.write("%g %g %g\n" % point)
        output.write("))\n(13 (0 1 %x 0))\n(13 (2 1 %x 3 3)(\n" % (len(triangles), len(triangles)))
        for triangle in triangles:
            output.write("%x %x %x 0 0\n" % tuple(index + 1 for index in triangle))
        output.write("))\n(39 (2 wall boi_box)())\n")


def write_boxes(path, named_boxes):
    """Write several axis-aligned boxes as one Fluent .msh surface mesh, one wall face zone (label) per box."""
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    count = len(named_boxes)
    lines = ['(0 "boi boxes")', "(2 3)", "(10 (0 1 %x 0 3))" % (8 * count), "(10 (1 1 %x 1 3)(" % (8 * count)]
    for _, (x0, y0, z0, x1, y1, z1) in named_boxes:
        lines.extend("%g %g %g" % (x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1))
    lines.append("))")
    lines.append("(13 (0 1 %x 0))" % (12 * count))
    for index, _ in enumerate(named_boxes):
        first, last = 12 * index + 1, 12 * index + 12
        lines.append("(13 (%x %x %x 3 3)(" % (index + 2, first, last))
        for a, b, c, d in quads:
            for triangle in ((a, b, c), (a, c, d)):
                lines.append("%x %x %x 0 0" % tuple(8 * index + i + 1 for i in triangle))
        lines.append("))")
    for index, (name, _) in enumerate(named_boxes):
        lines.append("(39 (%x wall %s)())" % (index + 2, name))
    with open(path, "w", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines) + "\n")


def parse(txt, report):
    matches = re.findall(r"(\d+) cells were created", txt)
    report["cells_created_last"] = int(matches[-1]) if matches else None
    report["cells_created"] = report["cells_created_last"]
    for key, pattern in (
        ("cells", r"number of interior cells\s*=\s*(\d+)"),
        ("faces_interior", r"number of interior faces\s*=\s*(\d+)"),
        ("nodes_interior", r"number of interior nodes\s*=\s*(\d+)"),
        ("faces_boundary", r"number of boundary faces\s*=\s*(\d+)"),
        ("nodes_boundary", r"number of boundary nodes\s*=\s*(\d+)"),
    ):
        values = re.findall(pattern, txt)
        report[key] = int(values[-1]) if values else None
    for key, pattern in (
        ("min_orth_quality", r"Minimum Orthogonal Quality\s*=\s*([0-9.eE+-]+)"),
        ("max_aspect_ratio", r"Maximum Aspect Ratio\s*=\s*([0-9.eE+-]+)"),
        ("prism_coverage", r"prism coverage\s*[:=]\s*([0-9.eE+-]+)\s*%?"),
    ):
        values = re.findall(pattern, txt, re.IGNORECASE)
        report[key] = float(values[-1]) if values else None
    values = re.findall(
        r"quality limits \(min max ave\) = \(([0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)\)\s*\n"
        r"count\s*=\s*\d+\s*\n\[Quality Measure : Skewness\]",
        txt,
    )
    report["max_skewness"] = float(values[-1][1]) if values else None
    values = re.findall(
        r"quality limits \(min max ave\) = \(([0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)\)\s*\n"
        r"count\s*=\s*\d+\s*\n\[Quality Measure : Orthogonal",
        txt,
    )
    report["orth_quality_avg"] = float(values[-1][2]) if values else None
    tables = re.findall(
        r"cells \(quality < ([0-9.]+)\)[^\n]*\n-+[^\n]*\n(?:[^\n]*\n)*?\s*Overall Summary\s+none\s+(\d+)\s+"
        r"([0-9.eE+-]+)\s+(\d+)",
        txt,
    )
    report["tables"] = [
        {"thr": float(a), "n_below": int(b), "minq": float(c), "cells": int(d)} for a, b, c, d in tables
    ]
    start = txt.rfind("NLT_0.05_BEGIN")
    if start >= 0:
        counts = [int(value) for value in re.findall(r"(\d+) cells: \(", txt[start:])]
        if len(counts) >= 2:
            report["_lcq_n_lt_0.05"], report["_lcq_n_lt_0.1"] = counts[0], counts[1]
        elif counts:
            report["_lcq_n_lt_0.05"], report["_lcq_n_lt_0.1"] = 0, counts[0]
        else:
            report["_lcq_n_lt_0.05"] = report["_lcq_n_lt_0.1"] = 0
    minimum = report["min_orth_quality"]
    counts = {}
    for threshold in (0.05, 0.1):
        count = 0 if minimum is not None and minimum >= threshold else None
        if count is None:
            count = next((table["n_below"] for table in reversed(report["tables"])
                          if abs(table["thr"] - threshold) < 1e-9), None)
        counts[threshold] = count
    report["n_lt_0.05"] = report.get("_lcq_n_lt_0.05", counts[0.05])
    report["n_lt_0.1"] = report.get("_lcq_n_lt_0.1", counts[0.1])
    warning_pattern = r"(prism|layer).*(fail|warn|could not|collaps|stair|not grown|invalid|skip)"
    warnings = sorted({line.strip()[:200] for line in txt.splitlines()
                       if re.search(warning_pattern, line, re.IGNORECASE) and "subscribe" not in line.lower()})
    report["prism_warnings"] = warnings[:20]
    report["boundary_layer_warnings"] = warnings[:20]
    report["stair_step_lines"] = sorted({line.strip()[:200] for line in txt.splitlines()
                                         if re.search(r"stair", line, re.IGNORECASE)})[:10]


def read_transcript(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def save_report(path, report):
    Path(path).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


def parent_main(case, scale, tag, kind, run_id, reparse=False):
    out = safe_output_target(ROOT, case, tag)
    out.mkdir(parents=True, exist_ok=True)
    transcript = out / "transcript.log"
    report_path = out / "mesh_report.json"
    mesh_path = out / "mesh.msh.h5"
    metrics_path = out / "metrics.csv"
    stop_path = out / f"STOP_METRICS_{run_id}"
    params = make_parameters(scale)
    logger = None
    child_rc = None
    start = time.time()
    env = dict(os.environ, MC_CHILD="1", MC_RUN_ID=run_id)
    try:
        if not reparse:
            logger = subprocess.Popen(
                [PYTHON, str(ROOT / "pipeline" / "tools" / "metrics_logger.py"), "--out", str(metrics_path),
                 "--interval", "5", "--until-file", str(stop_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with transcript.open("w", buffering=1, encoding="utf-8") as output:
                child_rc = subprocess.run(
                    [PYTHON, str(Path(__file__).resolve()), case, str(scale), tag, kind, run_id],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                ).returncode
    finally:
        if logger:
            stop_path.write_text("stop", encoding="utf-8")
            logger.wait(timeout=60)

    report = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if loaded.get("run_id") == run_id:
                report = loaded
        except (OSError, json.JSONDecodeError):
            pass
    report.update(case=case, tag=tag, scale=scale, kind=kind, run_id=run_id, params=params)
    if not reparse:
        report.update(wall_time_s=round(time.time() - start, 1), child_rc=child_rc)
    parse(read_transcript(transcript), report)
    artifact = report.get("mesh_artifact")
    mesh_written = False
    if not reparse and child_rc == 0 and report.get("run_id") == run_id and not report.get("aborted"):
        try:
            stat = mesh_path.stat()
            mesh_written = (
                stat.st_size > 0
                and isinstance(artifact, dict)
                and artifact.get("size_bytes") == stat.st_size
                and artifact.get("mtime_ns") == stat.st_mtime_ns
                and report.get("mesh_written") is True
            )
        except OSError:
            pass
    report["mesh_written"] = mesh_written
    if reparse:
        report["reparsed"] = True
    save_report(report_path, report)
    summary_keys = (
        "case", "tag", "scale", "kind", "cells", "min_orth_quality", "n_lt_0.05", "max_skewness",
        "prism_coverage", "wall_time_s", "mesh_written",
    )
    print(json.dumps({key: report.get(key) for key in summary_keys}))
    if reparse:
        return 0
    if child_rc != 0 or report.get("aborted") or not mesh_written:
        return 1
    try:
        require_cell_count(kind, report.get("cells"))
    except RuntimeError:
        return 1
    if report.get("min_orth_quality") is None or report.get("max_skewness") is None:
        return 1
    # Fluent 26.1 prints no prism-coverage percentage; the gate is "no prism/layer warnings"
    # (prism_coverage stays reported when a transcript provides it).
    if report.get("n_lt_0.05") is None or report.get("prism_warnings"):
        return 1
    return 0


def child_main(case, scale, tag, kind, run_id):
    params = make_parameters(scale)
    out = safe_output_target(ROOT, case, tag)
    out.mkdir(parents=True, exist_ok=True)
    transcript = out / "transcript.log"
    report_path = out / "mesh_report.json"
    mesh_path = out / "mesh.msh.h5"
    staged_mesh = out / f"mesh.{run_id}.msh.h5"
    report = {"steps": {}, "notes": [], "case": case, "tag": tag, "kind": kind, "scale": scale,
              "run_id": run_id, "mesh_written": False}
    started = time.time()

    def save():
        save_report(report_path, report)

    def log(*values):
        print("[%5ds]" % (time.time() - started), *values, flush=True)

    def step(name, function):
        start = time.time()
        try:
            result = function()
            report["steps"][name] = "ok %.0fs" % (time.time() - start)
            log("STEP OK", name)
            save()
            return result
        except Exception as exc:
            report["steps"][name] = f"FAILED {type(exc).__name__}: {str(exc)[:300]}"
            log("STEP FAILED", name, repr(exc)[:500])
            save()
            raise

    save()
    os.environ.setdefault("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")
    import ansys.fluent.core as pyfluent

    fluent = None
    try:
        log("case", case, "scale", scale, "tag", tag, "kind", kind, params)
        box_path = out / "boi_box.msh"
        if kind == "sbes":
            write_boxes(box_path, [("boi_box", params["boi_box"]), ("junction_box", SBES_JUNCTION_BOX)])
        else:
            write_box(box_path, params["boi_box"])
        fluent = pyfluent.launch_fluent(
            mode="meshing",
            precision="double",
            processor_count=int(os.environ.get("MC_NP", "4")),
            product_version="26.1.0",
            cleanup_on_exit=True,
            start_watchdog=False,
            cwd=str(out),
        )
        workflow = fluent.workflow
        workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
        tasks = workflow.TaskObject
        fluent.meshing.GlobalSettings.LengthUnit.set_state("m")

        def import_geometry():
            tasks["Import Geometry"].Arguments.set_state(
                {"FileName": str(ROOT / "cfd_v2" / "geom" / f"{case}_domain.dsco"), "LengthUnit": "m"}
            )
            tasks["Import Geometry"].Execute()

        step("import", import_geometry)

        def import_boi():
            tasks["Import Geometry"].InsertNextTask(CommandName="ImportBodyOfInfluenceGeometry")
            task = tasks["Import Body of Influence Geometry"]
            task.Arguments.set_state({"MeshFileName": str(box_path).replace("\\", "/"), "LengthUnit": "m", "Type": "Mesh"})
            task.Execute()

        step("boi_import", import_boi)

        def curvature_sizing():
            task = tasks["Add Local Sizing"]
            task.Arguments.set_state(
                {"AddChild": "yes", "BOIControlName": "curv_all", "BOIExecution": "Curvature", "BOIFaceLabelList": WALLS,
                 "BOIMinSize": params["cmin"], "BOIMaxSize": params["cmax"], "BOICurvatureNormalAngle": params["ang"],
                 "BOIGrowthRate": 1.2, "BOICellsPerGap": 1, "BOIZoneorLabel": "label"}
            )
            task.AddChildAndUpdate()

        step("curvature_sizing", curvature_sizing)

        if kind == "sbes":
            def sbes_sizing():
                for arguments in sbes_sizing_controls():
                    task = tasks["Add Local Sizing"]
                    task.Arguments.set_state(arguments)
                    task.AddChildAndUpdate()

            step("sbes_sizing", sbes_sizing)

        def wing_face_sizing():
            task = tasks["Add Local Sizing"]
            task.Arguments.set_state(
                {"AddChild": "yes", "BOIControlName": "wing_face", "BOIExecution": "Face Size",
                 "BOIFaceLabelList": ["mainwing", "horstab"], "BOISize": params["wing_face"],
                 "BOIGrowthRate": 1.2, "BOIZoneorLabel": "label"}
            )
            task.AddChildAndUpdate()

        step("wing_face_sizing", wing_face_sizing)

        def wake_boi():
            task = tasks["Add Local Sizing"]
            task.Arguments.set_state(
                {"AddChild": "yes", "BOIControlName": "wake_boi", "BOIExecution": "Body Of Influence",
                 "BOIFaceLabelList": ["boi_box"], "BOISize": params["boi_size"], "BOIZoneorLabel": "label"}
            )
            task.AddChildAndUpdate()

        step("wake_boi", wake_boi)

        def surface_mesh():
            task = tasks["Generate the Surface Mesh"]
            task.Arguments.set_state(
                {"CFDSurfaceMeshControls": {"MinSize": params["cmin"], "MaxSize": params["smax"],
                                             "CurvatureNormalAngle": params["ang"], "GrowthRate": 1.2}}
            )
            task.Execute()

        step("surface_mesh", surface_mesh)

        def describe_geometry():
            task = tasks["Describe Geometry"]
            task.Arguments.set_state({"NonConformal": "No", "SetupType": "The geometry consists of only fluid regions with no voids"})
            task.UpdateChildTasks(SetupTypeChanged=True)
            task.Execute()

        step("describe", describe_geometry)

        def update_boundaries():
            task = tasks["Update Boundaries"]
            task.Arguments.set_state(
                {"BoundaryLabelList": ["symmetry", "farfield"], "BoundaryLabelTypeList": ["symmetry", "pressure-far-field"],
                 "OldBoundaryLabelList": ["symmetry", "farfield"], "OldBoundaryLabelTypeList": ["wall", "wall"]}
            )
            task.Execute()

        step("boundaries", update_boundaries)
        step("regions", lambda: tasks["Update Regions"].Execute())

        def boundary_layers():
            task = tasks["Add Boundary Layers"]
            task.Arguments.set_state(
                {"AddChild": "yes", "BLControlName": "bl_all", "OffsetMethodType": "uniform",
                 "NumberOfLayers": params["nl"], "FirstHeight": params["fh"], "Rate": params["rate"]}
            )
            task.AddChildAndUpdate()

        step("boundary_layers", boundary_layers)

        def volume_mesh():
            task = tasks["Generate the Volume Mesh"]
            task.Arguments.set_state(
                {"VolumeFill": "poly-hexcore", "VolumeFillControls": {"HexMaxCellLength": params["hex"], "GrowthRate": 1.2}}
            )
            task.Execute()

        step("volume_mesh", volume_mesh)
        created = re.findall(r"(\d+) cells were created", read_transcript(transcript))
        report["cells_created"] = int(created[-1]) if created else None
        log("cells created", report["cells_created"])
        if os.environ.get("MC_DRY") == "1":
            report["aborted"] = "dry run (cell-count tuning only)"
            log(report["aborted"])
            save()
            return 2

        def improve_volume_mesh():
            tasks["Generate the Volume Mesh"].InsertNextTask(CommandName="ImproveVolumeMesh")
            task = tasks["Improve Volume Mesh"]
            task.Arguments.set_state(
                {"QualityMethod": "Orthogonal", "CellQualityLimit": float(os.environ.get("MC_IMPROVE", "0.1"))}
            )
            task.Execute()

        step("improve_volume_mesh", improve_volume_mesh)

        def quality_report():
            fluent.tui.report.mesh_size()
            fluent.tui.mesh.check_quality()
            fluent.tui.report.quality_method("skewness")
            fluent.tui.report.cell_quality_limits("*")
            fluent.tui.report.quality_method("orthogonal-quality")
            fluent.tui.report.cell_quality_limits("*")
            print("NLT_0.05_BEGIN", flush=True)
            fluent.tui.report.list_cell_quality("0", "0.05", "5", "*")
            print("NLT_END", flush=True)
            print("NLT_0.1_BEGIN", flush=True)
            fluent.tui.report.list_cell_quality("0", "0.1", "5", "*")
            print("NLT_END", flush=True)

        step("quality_report", quality_report)
        parse(read_transcript(transcript), report)
        require_cell_count(kind, report.get("cells"))
        if report.get("min_orth_quality") is None or report.get("max_skewness") is None:
            raise RuntimeError("Fluent quality report did not provide orthogonal quality and skewness")
        if report.get("n_lt_0.05") is None:
            raise RuntimeError("Fluent quality report did not provide the count below orthogonal quality 0.05")
        if report.get("prism_warnings"):
            raise RuntimeError("prism/boundary-layer warnings: " + "; ".join(report["prism_warnings"][:3]))

        step("write_mesh", lambda: fluent.tui.file.write_mesh('"' + str(staged_mesh).replace("\\", "/") + '"'))
        if not staged_mesh.is_file() or staged_mesh.stat().st_size <= 0:
            raise RuntimeError("Fluent did not write a non-empty mesh for this run")
        os.replace(staged_mesh, mesh_path)
        stat = mesh_path.stat()
        report["mesh_artifact"] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        report["mesh_written"] = True
        save()
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        save()
        raise
    finally:
        if fluent is not None:
            try:
                fluent.exit()
            except Exception as exc:
                log("exit", repr(exc)[:200])
        if staged_mesh.exists():
            staged_mesh.unlink()
        report["run_time_s"] = round(time.time() - started, 1)
        save()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("scale", type=float)
    parser.add_argument("tag")
    parser.add_argument("kind", choices=sorted(CELL_RANGES))
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("mode", nargs="?", choices=("reparse",))
    args = parser.parse_args(argv)
    validate_case_tag(args.case, args.tag)
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be a finite positive number")
    run_id = args.run_id or os.environ.get("MC_RUN_ID") or uuid.uuid4().hex
    if os.environ.get("MC_CHILD") == "1":
        if args.mode == "reparse":
            raise SystemExit("reparse is only supported by the parent process")
        return child_main(args.case, args.scale, args.tag, args.kind, run_id)
    return parent_main(args.case, args.scale, args.tag, args.kind, run_id, reparse=args.mode == "reparse")


if __name__ == "__main__":
    raise SystemExit(main())
