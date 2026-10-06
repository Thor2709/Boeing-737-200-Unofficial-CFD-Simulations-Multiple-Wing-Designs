#!/bin/bash
# usage: tune_case.sh <case> <scale0> <tag> <prod|grid|sbes> <target_cells> <lo> <hi> [skipdry]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$(command -v python)}"
export MC_NP="${MC_NP-4}"
MESH_CASE="$SCRIPT_DIR/mesh_case.py"

if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
  echo "usage: tune_case.sh <case> <scale0> <tag> <prod|grid|sbes> <target_cells> <lo> <hi> [skipdry]" >&2
  exit 2
fi
c=$1; scale=$2; tag=$3; kind=$4; target=$5; lo=$6; hi=$7; mode=${8:-}
case "$c" in C1|C2|C3|C4|C5) ;; *) echo "invalid case: $c" >&2; exit 2 ;; esac
[[ "$tag" =~ ^[A-Za-z0-9_]+$ ]] || { echo "invalid tag: $tag" >&2; exit 2; }
case "$kind" in prod|grid|sbes) ;; *) echo "invalid mesh kind: $kind" >&2; exit 2 ;; esac
[[ "$scale" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid scale: $scale" >&2; exit 2; }
for value in "$target" "$lo" "$hi"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "cell targets must be positive integers" >&2; exit 2; }
done
[ "$target" -gt 0 ] && [ "$lo" -gt 0 ] && [ "$lo" -le "$hi" ] && [ "$lo" -le "$target" ] && [ "$target" -le "$hi" ] \
  || { echo "target must be within the positive lo..hi interval" >&2; exit 2; }
case "$kind" in prod) min_allowed=800000 ;; sbes) min_allowed=900000 ;; *) min_allowed=300000 ;; esac
[ "$lo" -ge "$min_allowed" ] && [ "$hi" -le 999000 ] \
  || { echo "$kind range must stay within $min_allowed..999000" >&2; exit 2; }
[ -z "$mode" ] || [ "$mode" = skipdry ] || { echo "invalid option: $mode" >&2; exit 2; }

case_root="$ROOT/cfd_v2/mesh/$c"
rm_target=$("$PY" - "$ROOT" "$c" "$tag" <<'PY'
from pathlib import Path
import re
import sys

root, case, tag = sys.argv[1:]
if case not in {f"C{i}" for i in range(1, 6)} or re.fullmatch(r"[A-Za-z0-9_]+", tag) is None:
    raise SystemExit("invalid case or tag")
base = (Path(root).resolve() / "cfd_v2" / "mesh" / case).resolve()
target = (base / tag).resolve()
if target == base or base not in target.parents:
    raise SystemExit("resolved deletion target escapes the case output directory")
print(target)
PY
)

dry() {
  local candidate=$1 dry_tag="d$(printf '%s' "$1" | tr -d .)"
  local report="$case_root/$dry_tag/mesh_report.json" rc
  if MC_DRY=1 "$PY" "$MESH_CASE" "$c" "$candidate" "$dry_tag" "$kind" >/dev/null 2>&1; then
    rc=0
  else
    rc=$?
  fi
  "$PY" - "$report" "$rc" <<'PY'
import json
import sys
from pathlib import Path

path, rc = Path(sys.argv[1]), int(sys.argv[2])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit("dry run did not produce a readable mesh report")
cells = report.get("cells_created")
if rc == 0 or not str(report.get("aborted", "")).startswith("dry run") or type(cells) is not int:
    raise SystemExit("dry run did not fail closed with an explicit cell count")
print(cells)
PY
}

if [ "$mode" != skipdry ]; then
  cells=$(dry "$scale")
  echo "$c scale $scale cells $cells"
  if [ "$cells" -lt "$lo" ] || [ "$cells" -gt "$hi" ]; then
    scale=$("$PY" - "$scale" "$cells" "$target" <<'PY'
import math
import sys

scale, cells, target = map(float, sys.argv[1:])
if cells <= 0 or target <= 0:
    raise SystemExit("cell counts must be positive")
print(round(scale * (cells / target) ** (1 / 2.6), 4))
PY
)
    cells=$(dry "$scale")
    echo "$c scale $scale cells $cells (confirm)"
    if [ "$cells" -lt "$lo" ] || [ "$cells" -gt "$hi" ]; then
      echo "confirmed cell count $cells is outside $lo..$hi" >&2
      exit 1
    fi
  fi
fi

rm -rf -- "$rm_target"
"$PY" "$MESH_CASE" "$c" "$scale" "$tag" "$kind" | tail -1
