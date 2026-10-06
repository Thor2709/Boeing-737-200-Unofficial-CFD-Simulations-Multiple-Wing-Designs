#!/bin/bash
# usage: run.sh file.py [timeout_s] -> submit to the persistent Discovery session
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
Q="$ROOT/geom_v2/_work/q"
src="$(realpath "$1")"
n="cmd_$(date +%s%N).py"
cp "$src" "$Q/$n.tmp"
mv "$Q/$n.tmp" "$Q/$n"
o="$Q/${n%.py}.out"
if ! timeout "${2:-500}" bash -c "until [ -f '$o' ] && { grep -q '<<DONE>>' '$o' || grep -q '<<FAILED>>' '$o'; }; do sleep 2; done"; then
  [ ! -f "$o" ] || cat "$o"
  exit 1
fi
cat "$o"
grep -q '<<DONE>>' "$o"
