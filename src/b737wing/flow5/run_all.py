"""Run the full case set and the F2 mesh/wake sensitivity matrix."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from b737wing.flow5.cases import case_names
    from b737wing.flow5.runner import run_case
    from b737wing.flow5.xmlgen import project_root
else:
    from .cases import case_names
    from .runner import run_case
    from .xmlgen import project_root


def _row(case: str, mesh: str, wake_spans: float, result: dict) -> dict:
    post = result["post"]
    return {"case": case, "mesh": mesh, "wake_spans": wake_spans,
            "alpha_deg": post["alpha_deg"], "CL": post["CL"], "CDi": post["CDi"],
            "CDv": post["CDv"], "CD": post["CD"], "Cm": post["Cm"],
            "L_over_D": post["L_over_D"], "e": post["span_efficiency_e"],
            "coverage_violations": len(post["coverage_violations"])}


def run_all(output_root: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    output_root = Path(output_root) if output_root is not None else project_root() / "flow5_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    done_flag, failed_flag = output_root / "RUN_ALL.done", output_root / "RUN_ALL.failed"
    summary: list[dict] = []
    convergence: list[dict] = []
    try:
        # Mesh/wake study first: run_case post-processes into <case>/, so the
        # production (fine, 20 spans) runs must come last to own <case>/result.json.
        study: list[dict] = []
        for mesh in ("coarse", "medium"):
            result = run_case("F2", mesh=mesh, output_root=output_root)
            study.append(_row("F2", mesh, 20.0, result))
        result = run_case("F2", mesh="fine", run_label="fine_wake40", wake_spans=40.0,
                          output_root=output_root)
        study.append(_row("F2", "fine", 40.0, result))
        for case in case_names():
            result = run_case(case, mesh="fine", output_root=output_root)  # production = fine (Opus)
            summary.append(_row(case, "fine", 20.0, result))
            if case == "F2":
                convergence.append(_row(case, "fine", 20.0, result))
        summary.extend(study)
        convergence.extend(study)
        fields = list(summary[0])
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        with (output_root / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(convergence)
        if failed_flag.exists():
            failed_flag.unlink()
        done_flag.write_text("DONE\n", encoding="utf-8")
        return summary, convergence
    except Exception as error:
        if done_flag.exists():
            done_flag.unlink()
        failed_flag.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


def main() -> int:
    summary, convergence = run_all()
    print(f"wrote {len(summary)} summary rows and {len(convergence)} convergence rows")
    print(f"outputs: {project_root() / 'flow5_v2'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
