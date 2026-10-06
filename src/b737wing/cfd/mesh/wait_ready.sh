#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
DOMAIN="$ROOT/cfd_v2/geom/C5_domain.dsco"
while :; do
  processes="$(tasklist)" || { echo "could not inspect running processes" >&2; exit 1; }
  if [ -f "$DOMAIN" ] && ! grep -qi "Discovery.exe" <<< "$processes"; then
    sleep 30
    processes="$(tasklist)" || { echo "could not inspect running processes" >&2; exit 1; }
    if [ -f "$DOMAIN" ] && ! grep -qi "Discovery.exe" <<< "$processes"; then
      ls -l "$DOMAIN"
      echo READY
      exit 0
    fi
  fi
  sleep 20
done
