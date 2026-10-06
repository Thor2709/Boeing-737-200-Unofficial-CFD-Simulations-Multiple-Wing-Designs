#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
LOG="$ROOT/cfd_v2/mesh/run_prod_all2.log"
{
bash "$SCRIPT_DIR/tune_case.sh" C3 0.9778 prod prod 990000 982000 998000
bash "$SCRIPT_DIR/tune_case.sh" C5 1.733 g400 grid 400000 385000 415000
bash "$SCRIPT_DIR/tune_case.sh" C5 1.246 g650 grid 650000 635000 665000
} > "$LOG" 2>&1
echo ALLDONE >> "$LOG"
