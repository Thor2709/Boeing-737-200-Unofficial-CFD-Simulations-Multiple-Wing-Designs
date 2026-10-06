"""Build the checked 2-D airfoil set and plots for the B737-200 study."""

from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .foils import _interpolate_surface, blunt_te, flap, geometry_metrics, naca4, normalise, read_dat, repanel, self_intersects, smooth_digitised
except ImportError:  # Direct execution by the validation command.
    from foils import _interpolate_surface, blunt_te, flap, geometry_metrics, naca4, normalise, read_dat, repanel, self_intersects, smooth_digitised


try:
    from ..config import AEROFOIL_DIR, REPO_ROOT
    ROOT = REPO_ROOT
except (ImportError, ValueError):
    ROOT = Path(__file__).resolve().parents[3]
    AEROFOIL_DIR = ROOT / "data" / "aerofoils"

OUTPUT = ROOT / "aerofoils_v2"
HINGE_XC = 0.6483810
FLAP_DEFL_DEG = 21.35900
BLUNT_GAP = 0.0025
CSV_FIELDS = (
    "name",
    "n_points",
    "t_c",
    "x_t",
    "camber",
    "x_camber",
    "te_gap",
    "te_x",
    "te_y",
    "camber_signed_max_abs",
    "le_radius",
    "curvature_sign_changes_upper",
    "curvature_sign_changes_lower",
    "self_intersect",
)


def _sharp_te(xy: np.ndarray) -> np.ndarray:
    result = normalise(xy)
    result[0, 0] = result[-1, 0] = 1.0
    result[0, 1] = result[-1, 1] = 0.0
    return result


def _format_dat(name: str, xy: np.ndarray) -> str:
    lines = [name]
    lines.extend(f"{x:.8f} {y:.8f}" for x, y in xy)
    return "\n".join(lines) + "\n"


def _check_metrics(name: str, shape: np.ndarray, *, blunt: bool) -> dict[str, float | int | bool]:
    if len(shape) != 201:
        raise ValueError(f"{name} has {len(shape)} points; expected 201")
    metrics = geometry_metrics(shape, normalize_input=False)
    if metrics["self_intersect"]:
        raise ValueError(f"{name} contains a self-intersection")
    endpoint_gap = float(np.linalg.norm(shape[0] - shape[-1]))
    if blunt and abs(endpoint_gap - BLUNT_GAP) > 1e-5:
        raise ValueError(f"{name} trailing-edge gap is {endpoint_gap:.8f}, expected {BLUNT_GAP:.4f}")
    if not blunt and abs(float(metrics["te_gap"])) > 1e-8:
        raise ValueError(f"{name} sharp trailing edge is not closed")
    return metrics


def _plot_overlay(records: list[dict[str, object]]) -> bytes:
    figure, axes = plt.subplots(4, 2, figsize=(11, 13), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, record in zip(flat_axes, records):
        original = np.asarray(record["original"], dtype=float)
        panelled = np.asarray(record["sharp"], dtype=float)
        axis.plot(original[:, 0], original[:, 1], "o", ms=2.3, label="original")
        axis.plot(panelled[:, 0], panelled[:, 1], "-", lw=1.1, label="repanelled")
        axis.set_title(str(record["name"]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize="small")
    for axis in flat_axes[len(records):]:
        axis.set_visible(False)
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160)
    plt.close(figure)
    return output.getvalue()


def _plot_flap(original: np.ndarray, flapped: np.ndarray) -> bytes:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(original[:, 0], original[:, 1], label="AOPT sharp")
    axis.plot(flapped[:, 0], flapped[:, 1], label="AOPT_F, 21.359 deg down")
    le = int(np.argmin(original[:, 0]))
    upper = original[:le + 1][::-1]
    lower = original[le:]
    hinge_y = 0.5 * (
        _interpolate_surface(upper, np.array((HINGE_XC,)))[0]
        + _interpolate_surface(lower, np.array((HINGE_XC,)))[0]
    )
    axis.plot((HINGE_XC,), (hinge_y,), "x", label="hinge")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160)
    plt.close(figure)
    return output.getvalue()


def build() -> list[dict[str, object]]:
    source_map = (
        ("AOPT", AEROFOIL_DIR / "LatestOptimised.dat"),
        ("B737A", AEROFOIL_DIR / "737-200 Root Refined.dat"),
        ("B737B", AEROFOIL_DIR / "737-200 33% Span Refined.dat"),
        ("B737C", AEROFOIL_DIR / "737-200 66% Span Refined.dat"),
        ("B737D", AEROFOIL_DIR / "737-200 Tip Refined.dat"),
    )
    source_shapes: dict[str, np.ndarray] = {"A0012": naca4(0.0, 0.0, 0.12)}
    for name, source in source_map:
        try:
            source_shapes[name] = normalise(read_dat(source))
        except Exception as exc:
            raise ValueError(f"failed to parse or normalise {source.name}: {exc}") from exc

    original_aopt = source_shapes["AOPT"]
    sharp_shapes: dict[str, np.ndarray] = {}
    for name, original in source_shapes.items():
        sharp_shapes[name] = repanel(_sharp_te(original), 200)
    # Digitising noise in the 33 % section stalls XFoil at high Re (see smooth_digitised).
    sharp_shapes["B737B"] = smooth_digitised(sharp_shapes["B737B"])
    sharp_shapes["AOPT_F"] = flap(sharp_shapes["AOPT"], HINGE_XC, FLAP_DEFL_DEG)

    ordered_names = ("A0012", "AOPT", "AOPT_F", "B737A", "B737B", "B737C", "B737D")
    if set(sharp_shapes) != set(ordered_names):
        raise ValueError("internal foil set is incomplete")

    records: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    files_to_write: dict[str, str | bytes] = {}
    for name in ordered_names:
        sharp = sharp_shapes[name]
        sharp_metrics = _check_metrics(f"{name}.dat", sharp, blunt=False)
        if name == "A0012":
            if abs(float(sharp_metrics["t_c"]) - 0.12) > 0.0005:
                raise ValueError(f"A0012 t/c {sharp_metrics['t_c']:.7f} is outside 0.1200 +/- 0.0005")
            if abs(float(sharp_metrics["camber"])) >= 1e-6:
                raise ValueError(f"A0012 camber {sharp_metrics['camber']:.8g} is not symmetric")
        if name == "AOPT":
            if abs(float(sharp_metrics["camber"]) - 0.0262) > 0.001:
                raise ValueError(f"AOPT camber {sharp_metrics['camber']:.7f} does not match its header claim")
            if abs(float(sharp_metrics["x_camber"]) - 0.356) > 0.02:
                raise ValueError(f"AOPT x/camber {sharp_metrics['x_camber']:.7f} does not match its header claim")
            if abs(float(sharp_metrics["t_c"]) - 0.12) > 0.002:
                raise ValueError(f"AOPT t/c {sharp_metrics['t_c']:.7f} does not match its header claim")

        sharp_filename = f"{name}.dat"
        files_to_write[sharp_filename] = _format_dat(name, sharp)
        rows.append({"name": sharp_filename, "n_points": len(sharp), **sharp_metrics})

        if name == "AOPT_F":
            blunt_aopt = blunt_te(sharp_shapes["AOPT"], BLUNT_GAP)
            blunt = flap(blunt_aopt, HINGE_XC, FLAP_DEFL_DEG)
        else:
            blunt = blunt_te(sharp, BLUNT_GAP)
        blunt_filename = f"{name}_bluntTE.dat"
        blunt_metrics = _check_metrics(blunt_filename, blunt, blunt=True)
        files_to_write[blunt_filename] = _format_dat(f"{name}_bluntTE", blunt)
        rows.append({"name": blunt_filename, "n_points": len(blunt), **blunt_metrics})

        original = source_shapes.get(name, original_aopt)
        records.append({"name": name, "original": original, "sharp": sharp})

    for name, source in source_map[1:]:
        normalized = source_shapes[name]
        le_index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
        te_midpoint = 0.5 * (normalized[0] + normalized[-1])
        if abs(float(np.linalg.norm(te_midpoint - normalized[le_index])) - 1.0) > 1e-6:
            raise ValueError(f"{source.name} does not normalize to unit chord")
        if self_intersects(normalized):
            raise ValueError(f"{source.name} contains a self-intersection")

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    files_to_write["foil_checks.csv"] = csv_buffer.getvalue()
    files_to_write["overlay.png"] = _plot_overlay(records)
    files_to_write["flap.png"] = _plot_flap(sharp_shapes["AOPT"], sharp_shapes["AOPT_F"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, content in files_to_write.items():
        target = OUTPUT / filename
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8", newline="")
    return rows


if __name__ == "__main__":
    built_rows = build()
    print(f"Wrote {len(built_rows)} foil checks to {OUTPUT / 'foil_checks.csv'}")
