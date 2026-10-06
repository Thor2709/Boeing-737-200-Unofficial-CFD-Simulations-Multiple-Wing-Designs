"""Command line entry point for one flow5 case."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from b737wing.flow5.cases import case_names
    from b737wing.flow5.runner import FLOW5_EXE, find_plane_polar, run_case, verify_fluid_output
    from b737wing.flow5.xmlgen import project_root, script_xml, write_case_files
else:
    from .cases import case_names
    from .runner import FLOW5_EXE, find_plane_polar, run_case, verify_fluid_output
    from .xmlgen import project_root, script_xml, write_case_files


def _alpha(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("alpha must be 'min,max,inc'") from error
    if len(parts) != 3 or parts[2] <= 0.0 or parts[1] < parts[0]:
        raise argparse.ArgumentTypeError("alpha must be 'min,max,inc' with max>=min and inc>0")
    return parts


def run_smoke() -> dict:
    """Run and physically cross-check the bounded NACA 0012 coarse case."""
    from b737wing.aero.foils import naca4, repanel
    from b737wing.flow5.cases import case_config
    from b737wing.flow5.parse import parse_plane_polar
    from b737wing.flow5.post import ASPECT_RATIO, postprocess

    output_root = project_root() / "flow5_v2" / "_smoke"
    foil_path = output_root / "source_foils" / "A0012.dat"
    foil_path.parent.mkdir(parents=True, exist_ok=True)
    points = repanel(naca4(0.0, 0.0, 0.12), n_panels=200)
    foil_lines = ["A0012"] + [f"{x:.9f} {y:.9f}" for x, y in points]
    foil_path.write_text("\n".join(foil_lines) + "\n", encoding="utf-8")
    sources = {"A0012": foil_path}
    case = case_config("F1")
    alpha = (0.0, 6.0, 1.0)
    Reynolds = [8.0e6, 2.0e7, 4.0e7]
    run_label = "coarse_smoke_201pt"
    while (output_root / "F1" / run_label / "pass1" / "flow5-output" /
           f"F1_{run_label}_pass1").exists():
        run_label += "_retry"
    run = run_case("F1", mesh="coarse", pass_mode="all", alpha=alpha,
                   output_root=output_root, reynolds=Reynolds, foil_sources=sources,
                   run_label=run_label, post=False)
    run_root = output_root / "F1" / run_label
    project_dir, polar_path = find_plane_polar(run_root, "F1", "coarse", run_label,
                                               launched_at=run["pass2"].launched_at)
    fluid = verify_fluid_output("F1", project_dir, polar_path)
    if not fluid["ok"]:
        retry_label = f"{run_label}_dynamic_viscosity"
        retry_root = output_root / "F1" / retry_label
        source_polars = run_root / "xfoil_polars"
        target_polars = retry_root / "xfoil_polars"
        if target_polars.exists():
            raise RuntimeError(f"dynamic-viscosity retry output already exists: {target_polars}")
        shutil.copytree(source_polars, target_polars)
        dynamic = float(case["flight"]["rho"]) * float(case["flight"]["nu"])
        retry = run_case("F1", mesh="coarse", pass_mode="2", alpha=alpha,
                         output_root=output_root, foil_sources=sources, viscosity=dynamic,
                         run_label=retry_label, post=False)
        run["pass2_dynamic"] = retry["pass2"]
        project_dir, polar_path = find_plane_polar(retry_root, "F1", "coarse", retry_label,
                                                   launched_at=retry["pass2"].launched_at)
        dynamic_fluid = verify_fluid_output("F1", project_dir, polar_path)
        if not dynamic_fluid["ok"]:
            raise RuntimeError(
                "NEEDS_OPUS_DECISION: neither Fluid/Viscosity interpretation matched. "
                f"Kinematic input echoed density={fluid['density_echo']!r}, "
                f"max strip Re error={fluid['max_re_relative_error']:.3%}; "
                f"dynamic input echoed density={dynamic_fluid['density_echo']!r}, "
                f"max strip Re error={dynamic_fluid['max_re_relative_error']:.3%}."
            )
        fluid = dynamic_fluid
    table = parse_plane_polar(polar_path)
    alpha_index, cl_index = table.index("alpha"), table.index("cl")
    cdi_index, cdv_index, cd_index = (table.index(label) for label in ("cdi", "cdv", "cd"))
    rows = [row for row in table.rows if 0.0 <= row[alpha_index] <= 6.0 and
            all(math.isfinite(row[index]) for index in (alpha_index, cl_index, cdi_index, cdv_index, cd_index))]
    if len(rows) < 3:
        raise RuntimeError(f"smoke polar returned only {len(rows)} finite points in alpha 0..6")
    mean_alpha = sum(row[alpha_index] for row in rows) / len(rows)
    mean_cl = sum(row[cl_index] for row in rows) / len(rows)
    slope = sum((row[alpha_index] - mean_alpha) * (row[cl_index] - mean_cl) for row in rows) / sum(
        (row[alpha_index] - mean_alpha) ** 2 for row in rows)
    if not 0.06 <= slope <= 0.11:
        raise RuntimeError(f"smoke CL-alpha slope {slope:.5f}/deg is outside 0.06..0.11")
    for row in rows:
        cl, cdi = row[cl_index], row[cdi_index]
        if cl > 0.0 and not (0.0 < cdi < cl * cl / (math.pi * ASPECT_RATIO * 0.6)):
            raise RuntimeError(f"smoke CDi {cdi:g} is implausible at alpha {row[alpha_index]:g}, CL={cl:g}")
    post = postprocess("F1", polar_path, project_dir, output_root / "F1", run_root / "xfoil_polars")
    report = {"CL_alpha_per_degree": slope, "fluid_check": fluid,
              "table": [{"alpha": row[alpha_index], "CL": row[cl_index], "CDi": row[cdi_index],
                         "CDv": row[cdv_index], "CD": row[cd_index]} for row in rows],
              "post": post}
    (output_root / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a flow5 v7.57 B737-200 wing case")
    parser.add_argument("--case", required=True, choices=case_names())
    parser.add_argument("--mesh", required=True, choices=("coarse", "medium", "fine"))
    parser.add_argument("--pass", dest="pass_mode", default="all", choices=("1", "2", "all"))
    parser.add_argument("--alpha", type=_alpha, help="plane alpha sweep as min,max,inc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        output_root = project_root() / "flow5_v2"
        run_root = output_root / args.case / args.mesh
        case_root = project_root() / "aerofoils_v2"
        from b737wing.flow5.cases import case_config
        case = case_config(args.case)
        foils = list(dict.fromkeys(section["foil"] for section in case["sections"]))
        if case.get("flap"):
            foils = ["AOPT", "AOPT_F"]
        foil_dir = run_root / "foils"
        foil_dir.mkdir(parents=True, exist_ok=True)
        for foil in foils:
            source = case_root / f"{foil}.dat"
            if not source.is_file():
                raise FileNotFoundError(f"airfoil {foil} is not available at {source}")
            (foil_dir / source.name).write_bytes(source.read_bytes())
        plane_path, polar_path, pass1_script, pass2_script = write_case_files(args.case, run_root, args.mesh)
        if args.pass_mode in ("1", "all"):
            pass1_script.write_text(script_xml(args.case, 1, run_root=run_root,
                                               foil_files=[f"{foil}.dat" for foil in foils], mesh=args.mesh), encoding="utf-8")
        if args.pass_mode in ("2", "all"):
            pass2_script.write_text(script_xml(args.case, 2, run_root=run_root,
                                               foil_files=[f"{foil}.dat" for foil in foils], mesh=args.mesh,
                                               alpha=args.alpha), encoding="utf-8")
        print(f"flow5 executable: {FLOW5_EXE}")
        print(f"plane XML: {plane_path}")
        print(f"polar XML: {polar_path}")
        if args.pass_mode in ("1", "all"):
            print(f"pass 1 script: {pass1_script}")
        if args.pass_mode in ("2", "all"):
            print(f"pass 2 script: {pass2_script}")
        return 0
    result = run_case(args.case, mesh=args.mesh, pass_mode=args.pass_mode,
                      alpha=args.alpha, output_root=project_root() / "flow5_v2")
    for pass_name in ("pass1", "pass2"):
        run = result.get(pass_name)
        if run is not None:
            print(f"{pass_name}: exit={run.returncode} elapsed={run.elapsed:.1f}s stdout={run.stdout_log} stderr={run.stderr_log}")
    if "post" in result:
        report = result["post"]
        print(f"target CL={report['target_cl']:.5f} alpha={report['alpha_deg']:.3f} CD={report['CD']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
