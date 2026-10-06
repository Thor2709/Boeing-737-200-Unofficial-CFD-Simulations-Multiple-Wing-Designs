"""Defensive readers for flow5 v7.57 text output."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?(?:inf|nan)$", re.I)
_LABELS = re.compile(r"\s{2,}")


def _number(token: str) -> bool:
    return bool(_NUMBER.fullmatch(token.strip()))


def _floats(tokens: list[str]) -> list[float]:
    return [float(token) for token in tokens]


def _header(lines: list[str], end: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines[:end]:
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _find_table_header(lines: list[str], predicate) -> int:
    for index, line in enumerate(lines):
        if predicate(line):
            return index
    raise ValueError("flow5 file has no recognizable table header")


@dataclass
class NumericTable:
    header: dict[str, str]
    columns: list[str]
    rows: list[list[float]]
    path: Path | None = None
    nonfinite: list[tuple[int, str]] = field(default_factory=list)

    def index(self, name: str) -> int:
        aliases = {
            "alpha": ("α", "alpha", "α (°)", "alpha (°)"),
            "cl": ("CL", "Cl"),
            "cd": ("CD",),
            "cdi": ("CD_induced", "CDi", "Cd_i"),
            "cdv": ("CD_viscous", "CDv", "Cd_v"),
            "cm": ("Cm",),
            "lift": ("Lift (N)", "Lift"),
            "drag": ("Drag (N)", "Drag"),
        }
        candidates = aliases.get(name.lower(), (name,))
        normalized = {candidate.casefold() for candidate in candidates}
        for index, label in enumerate(self.columns):
            clean = label.strip()
            if clean.casefold() in normalized or clean.split(" (")[0].strip().casefold() in normalized:
                return index
        raise KeyError(f"column {name!r} missing from {self.path or 'flow5 table'}; found {self.columns}")

    def values(self, name: str) -> list[float]:
        index = self.index(name)
        return [row[index] for row in self.rows]


def parse_plane_polar(path: str | Path) -> NumericTable:
    """Read fixed-width plane text, including the row welded to the label line."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_plane_polar_lines(lines, path)


def parse_plane_polar_text(text: str, name: str = "<text>") -> NumericTable:
    """Parse plane-polar text directly, useful for fixture-based checks."""
    return _parse_plane_polar_lines(text.splitlines(), Path(name))


def _parse_plane_polar_lines(lines: list[str], path: Path) -> NumericTable:
    header_index = _find_table_header(lines, lambda line: "Ctrl" in line and "CL" in line and "CD" in line)
    header_line = lines[header_index]
    standalone: list[tuple[str, list[str]]] = []
    for line in lines[header_index + 1:]:
        tokens = line.split()
        if tokens and all(_number(token) for token in tokens):
            standalone.append((line, tokens))
            break
    rows: list[list[float]] = []
    if standalone:
        data_line, first_tokens = standalone[0]
        count = len(first_tokens)
        label_region = header_line
        embedded = header_line[-len(data_line):] if len(header_line) > len(data_line) else ""
        embedded_tokens = embedded.split()
        if len(embedded_tokens) == count and all(_number(token) for token in embedded_tokens):
            rows.append(_floats(embedded_tokens))
            label_region = header_line[:-len(data_line)]
        columns = [label for label in _LABELS.split(label_region.strip()) if label]
        for line in lines[header_index + 1:]:
            tokens = line.split()
            if len(tokens) == count and all(_number(token) for token in tokens):
                rows.append(_floats(tokens))
            elif tokens and any(_number(token) for token in tokens):
                raise ValueError(f"{path.name}: malformed numeric plane-polar row: {line[:100]!r}")
    else:
        # T1 has 57 documented 13-character data fields. A one-point polar has no
        # standalone row, so its only numeric row must be split off the label line.
        field_count = 57
        width = field_count * 13
        if len(header_line) < width:
            raise ValueError(f"{path.name}: no standalone data row and no complete welded row")
        label_region = header_line[:-width]
        columns = [label for label in _LABELS.split(label_region.strip()) if label]
        tail = header_line[-width:]
        tokens = [tail[index:index + 13].strip() for index in range(0, width, 13)]
        if len(tokens) != field_count or len(columns) != field_count or not all(_number(token) for token in tokens):
            raise ValueError(f"{path.name}: welded single-point row does not match its labels")
        rows.append(_floats(tokens))
    if not rows or len(columns) != len(rows[0]):
        raise ValueError(f"{path.name}: recovered {len(columns)} labels for {len(rows[0]) if rows else 0} values")
    header = _header(lines, header_index)
    claimed = next((value for key, value in header.items() if key.lower().startswith("nbr. of data points")), None)
    if claimed:
        match = re.search(r"\d+", claimed)
        if match and int(match.group()) != len(rows):
            raise ValueError(f"{path.name}: parsed {len(rows)} polar points but header declares {match.group()}")
    nonfinite = [(row_index, columns[column_index]) for row_index, row in enumerate(rows)
                 for column_index, value in enumerate(row) if not math.isfinite(value)]
    return NumericTable(header, columns, rows, path, nonfinite)


@dataclass
class StripTable:
    surface: str
    columns: list[str]
    rows: list[list[float]]

    def values(self, name: str) -> list[float]:
        lookup = {column.casefold(): index for index, column in enumerate(self.columns)}
        if name.casefold() not in lookup:
            raise KeyError(f"strip column {name!r} missing from {self.surface}; found {self.columns}")
        index = lookup[name.casefold()]
        return [row[index] for row in self.rows]


@dataclass
class CpTable:
    surface: str
    columns: list[str]
    rows: list[list[float]]
    strip_number: int | None = None
    strip_y_m: float | None = None


@dataclass
class OperatingPoint:
    path: Path
    polar_name: str
    alpha: float | None
    coefficients: dict[str, float]
    strips: dict[str, StripTable]
    cp_tables: list[CpTable]
    content_hash: str


def parse_strips(path: str | Path) -> dict[str, StripTable]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _parse_strips_lines(lines, path.name)


def _parse_strips_lines(lines: list[str], source_name: str) -> dict[str, StripTable]:
    result: dict[str, StripTable] = {}
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("y(m)"):
            index += 1
            continue
        surface = next((lines[row].strip() for row in range(index - 1, max(-1, index - 6), -1)
                        if lines[row].strip()), "?")
        columns = lines[index].split()
        if len(columns) < 2:
            raise ValueError(f"{source_name}: strip table has fewer than two column labels")
        rows: list[list[float]] = []
        row_index = index + 1
        while row_index < len(lines):
            tokens = lines[row_index].split()
            if not tokens:
                break
            if len(tokens) != len(columns) or not all(_number(token) for token in tokens):
                if any(_number(token) for token in tokens):
                    raise ValueError(f"{source_name}: malformed strip row: {lines[row_index][:100]!r}")
                break
            rows.append(_floats(tokens))
            row_index += 1
        if not rows:
            raise ValueError(f"{source_name}: strip table for {surface!r} contains no data rows")
        result[surface] = StripTable(surface, columns, rows)
        index = row_index
    if not result:
        raise ValueError(f"{source_name}: no spanwise strip table found")
    return result


def _column_index(columns: list[str], *names: str) -> int | None:
    wanted = {name.casefold() for name in names}
    return next((index for index, column in enumerate(columns)
                 if column.strip().casefold() in wanted), None)


def _parse_cp_tables(lines: list[str], strips: dict[str, StripTable], source_name: str) -> list[CpTable]:
    tables: list[CpTable] = []
    expected_columns = {"panel", "ctrlpt.x", "ctrlpt.y", "ctrlpt.z", "n.x", "n.y", "n.z", "area", "cp"}
    header_index = next((index for index, line in enumerate(lines)
                         if expected_columns.issubset({token.casefold() for token in line.split()})), None)
    if header_index is not None:
        columns = lines[header_index].split()
        surface = next((line.strip().split(" - Cp", 1)[0]
                        for line in reversed(lines[:header_index]) if " - Cp Coefficients" in line), "Main Wing")
        strip_table = next(iter(strips.values()), None)
        if strip_table is None:
            raise ValueError(f"{source_name}: Cp panel blocks have no associated strip table")
        y_index = _column_index(strip_table.columns, "y(m)")
        cp_y_index = _column_index(columns, "CtrlPt.y")
        if y_index is None or cp_y_index is None:
            raise ValueError(f"{source_name}: cannot associate Cp blocks with strip y coordinates")
        strip_ys = [row[y_index] for row in strip_table.rows]
        index = header_index + 1
        while index < len(lines):
            match = re.fullmatch(r"\s*Strip\s+(\d+)\s*", lines[index], re.I)
            if not match:
                index += 1
                continue
            strip_number = int(match.group(1))
            if not 1 <= strip_number <= len(strip_ys):
                raise ValueError(f"{source_name}: Cp Strip {strip_number} has no matching strip row")
            rows: list[list[float]] = []
            row_index = index + 1
            while row_index < len(lines):
                if re.fullmatch(r"\s*Strip\s+\d+\s*", lines[row_index], re.I):
                    break
                tokens = lines[row_index].split()
                if not tokens:
                    row_index += 1
                    if rows:
                        break
                    continue
                if len(tokens) != len(columns) or not all(_number(token) for token in tokens):
                    if rows:
                        break
                    row_index += 1
                    continue
                rows.append(_floats(tokens))
                row_index += 1
            if not rows:
                raise ValueError(f"{source_name}: Cp Strip {strip_number} contains no panel rows")
            expected_y = strip_ys[strip_number - 1]
            mean_y = sum(row[cp_y_index] for row in rows) / len(rows)
            adjacent = []
            if strip_number > 1:
                adjacent.append(abs(expected_y - strip_ys[strip_number - 2]))
            if strip_number < len(strip_ys):
                adjacent.append(abs(strip_ys[strip_number] - expected_y))
            half_width = (sum(adjacent) / len(adjacent)) / 2.0 if adjacent else 0.0
            if abs(mean_y - expected_y) > half_width + 1e-9:
                raise ValueError(
                    f"{source_name}: Cp Strip {strip_number} mean CtrlPt.y={mean_y:.6g} "
                    f"does not match strip y={expected_y:.6g} within half-width {half_width:.6g}"
                )
            tables.append(CpTable(surface, columns, rows, strip_number, expected_y))
            index = row_index
        if tables:
            if len(tables) != len(strip_ys):
                raise ValueError(f"{source_name}: parsed {len(tables)} Cp strip blocks for {len(strip_ys)} strip rows")
            return tables

    # Older exports can contain a standalone x/c, Cp table instead of panel blocks.
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        lowered = header.casefold()
        if "cp" not in lowered or not ("x/c" in lowered or "x / c" in lowered or "x/c(%)" in lowered):
            index += 1
            continue
        columns = header.split()
        rows: list[list[float]] = []
        row_index = index + 1
        while row_index < len(lines):
            tokens = lines[row_index].replace(",", " ").split()
            if len(tokens) < 2 or not all(_number(token) for token in tokens):
                if rows:
                    break
                row_index += 1
                continue
            rows.append(_floats(tokens))
            row_index += 1
        if rows:
            surface = next((lines[row].strip() for row in range(index - 1, max(-1, index - 5), -1)
                            if lines[row].strip()), "Main Wing")
            tables.append(CpTable(surface, columns, rows))
            index = row_index
        else:
            index += 1
    return tables


def owning_polar(path: str | Path) -> str | None:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[1].strip() if len(lines) > 1 and lines[1].strip() else None


def _alpha_from_path(path: Path) -> float | None:
    match = re.search(r"([+-]?\d+(?:_\d+)?)\s*°", path.name)
    return float(match.group(1).replace("_", ".")) if match else None


def parse_op_point(path: str | Path) -> OperatingPoint:
    path = Path(path)
    raw = path.read_bytes()
    return _parse_op_point_bytes(raw, path)


def parse_op_point_text(text: str, name: str) -> OperatingPoint:
    return _parse_op_point_bytes(text.encode("utf-8"), Path(name))


def _parse_op_point_bytes(raw: bytes, path: Path) -> OperatingPoint:
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if len(lines) < 2 or not lines[1].strip():
        raise ValueError(f"{path.name}: missing owning-polar line")
    strips = _parse_strips_lines(lines, path.name)
    coefficients = _parse_op_coefficients(lines, path.name)
    alpha = _alpha_from_header(lines)
    if alpha is None:
        alpha = _alpha_from_path(path)
    if alpha is None:
        for line in lines[:12]:
            match = re.search(r"(?:alpha|α)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", line, re.I)
            if match:
                alpha = float(match.group(1))
                break
    return OperatingPoint(path, lines[1].strip(), alpha, coefficients, strips,
                          _parse_cp_tables(lines, strips, path.name),
                          hashlib.sha256(raw).hexdigest())


def _parse_op_coefficients(lines: list[str], source_name: str) -> dict[str, float]:
    required = {"cl", "cx", "cy", "cd_inviscid", "cd_viscous", "cm_inviscid"}
    for index, line in enumerate(lines[:-1]):
        columns = line.split()
        if not required.issubset({column.casefold() for column in columns}):
            continue
        for data_line in lines[index + 1:]:
            tokens = data_line.split()
            if not tokens:
                continue
            if len(tokens) == len(columns) and all(_number(token) for token in tokens):
                return dict(zip(columns, _floats(tokens)))
            break
    return {}


def _alpha_from_header(lines: list[str]) -> float | None:
    for index, line in enumerate(lines[:-1]):
        columns = line.split()
        alpha_index = next((column for column, label in enumerate(columns)
                            if label.casefold() in {"α", "î±", "alpha"}), None)
        if alpha_index is None:
            continue
        for data_line in lines[index + 1:]:
            tokens = data_line.split()
            if not tokens:
                continue
            if len(tokens) == len(columns) and all(_number(token) for token in tokens):
                return float(tokens[alpha_index])
            break
    return None


def parse_op_points(paths: list[str | Path], *, expected_polar: str | None = None) -> list[OperatingPoint]:
    """Deduplicate identical flow5 copies and filter by the polar named in line 2."""
    points: list[OperatingPoint] = []
    seen: set[str] = set()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        point = parse_op_point(path)
        if expected_polar is not None and point.polar_name.casefold() != expected_polar.casefold():
            continue
        if point.content_hash in seen:
            continue
        seen.add(point.content_hash)
        points.append(point)
    if expected_polar is not None and not points:
        raise ValueError(f"no operating-point files declare polar {expected_polar!r} on line 2")
    return points


def parse_op_point_texts(texts: list[str], names: list[str], *,
                         expected_polar: str | None = None) -> list[OperatingPoint]:
    if len(texts) != len(names):
        raise ValueError("each operating-point text needs a source name")
    points: list[OperatingPoint] = []
    seen: set[str] = set()
    for text, name in zip(texts, names):
        point = parse_op_point_text(text, name)
        if expected_polar is not None and point.polar_name.casefold() != expected_polar.casefold():
            continue
        if point.content_hash in seen:
            continue
        seen.add(point.content_hash)
        points.append(point)
    if expected_polar is not None and not points:
        raise ValueError(f"no operating-point files declare polar {expected_polar!r} on line 2")
    return points


@dataclass
class FoilPolar:
    foil: str
    reynolds: float
    ncrit: float
    alpha: list[float]
    cl: list[float]
    cd: list[float]
    cm: list[float]
    path: Path


def parse_foil_polar(path: str | Path) -> FoilPolar:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    foil, reynolds, ncrit = path.stem, None, 9.0
    for line in lines[:20]:
        if "Calculated polar for:" in line:
            foil = line.split(":", 1)[1].strip()
        match = re.search(r"Re\s*=\s*([0-9.]+)\s*e\s*([+-]?\d+)", line, re.I)
        if match:
            reynolds = float(match.group(1)) * 10.0 ** int(match.group(2))
        elif re.search(r"\bRe\s*=", line):
            match = re.search(r"\bRe\s*=\s*([0-9.eE+-]+)", line)
            if match:
                reynolds = float(match.group(1))
        match = re.search(r"Ncrit\s*=\s*([0-9.]+)", line, re.I)
        if match:
            ncrit = float(match.group(1))
    header_index = _find_table_header(lines, lambda line: "alpha" in line.casefold() and "cl" in line.casefold() and "cd" in line.casefold())
    delimiter = "," if "," in lines[header_index] else None
    alpha: list[float] = []
    cl: list[float] = []
    cd: list[float] = []
    cm: list[float] = []
    for line in lines[header_index + 1:]:
        if not line.strip() or set(line.strip()) <= {"-", " "}:
            continue
        tokens = [token.strip() for token in line.split(delimiter)] if delimiter else line.split()
        if len(tokens) < 5 or not all(_number(token) for token in tokens[:5]):
            if any(_number(token) for token in tokens):
                raise ValueError(f"{path.name}: malformed numeric 2D-polar row: {line[:100]!r}")
            continue
        alpha.append(float(tokens[0]))
        cl.append(float(tokens[1]))
        cd.append(float(tokens[2]))
        cm.append(float(tokens[4]))
    if not alpha:
        raise ValueError(f"{path.name}: 2D polar has no converged alpha points")
    if reynolds is None:
        match = re.search(r"(?:Re|re)[-_]?([0-9]+(?:\.[0-9]+)?)", path.stem)
        if match:
            reynolds = float(match.group(1))
    if reynolds is None or reynolds <= 0.0:
        raise ValueError(f"{path.name}: could not identify the polar Reynolds number")
    return FoilPolar(foil, reynolds, ncrit, alpha, cl, cd, cm, path)


def alpha_gap_diagnostics(polar: FoilPolar, requested_step: float) -> list[dict]:
    """Describe missing alpha intervals wider than 2.5 requested steps."""
    if not math.isfinite(requested_step) or requested_step <= 0.0:
        raise ValueError("requested alpha step must be finite and positive")
    alphas = sorted({alpha for alpha in polar.alpha if math.isfinite(alpha)})
    diagnostics = []
    for lower, upper in zip(alphas, alphas[1:]):
        gap = upper - lower
        if gap > 2.5 * requested_step:
            diagnostics.append({
                "kind": "alpha_gap",
                "foil": polar.foil,
                "re": polar.reynolds,
                "alpha_from_deg": lower,
                "alpha_to_deg": upper,
                "gap_deg": gap,
                "requested_step_deg": requested_step,
                "reason": (f"alpha gap {gap:g} deg exceeds 2.5 requested steps "
                           f"({2.5 * requested_step:g} deg)"),
            })
    return diagnostics


def nonfinite(table: NumericTable) -> list[tuple[int, str]]:
    return list(table.nonfinite)
