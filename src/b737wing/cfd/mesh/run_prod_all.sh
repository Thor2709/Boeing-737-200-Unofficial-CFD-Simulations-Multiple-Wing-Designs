#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"
LOG="$ROOT/cfd_v2/mesh/run_prod_all.log"
{
bash "$SCRIPT_DIR/tune_case.sh" C2 0.9525 prod prod 990000 982000 998000 skipdry
bash "$SCRIPT_DIR/tune_case.sh" C4 0.95 prod prod 990000 982000 998000
bash "$SCRIPT_DIR/tune_case.sh" C1 0.95 prod prod 990000 982000 998000
bash "$SCRIPT_DIR/tune_case.sh" C3 0.95 prod prod 990000 982000 998000
bash "$SCRIPT_DIR/tune_case.sh" C5 1.33 g400 grid 400000 380000 420000
bash "$SCRIPT_DIR/tune_case.sh" C5 1.12 g650 grid 650000 625000 675000
} > "$LOG" 2>&1
echo ALLDONE >> "$LOG"
