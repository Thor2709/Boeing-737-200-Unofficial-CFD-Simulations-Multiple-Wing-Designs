#!/usr/bin/env bash
# Blocking watcher (run with run_in_background so the orchestrator is woken ONCE, on completion).
# Usage:
#   wait_for.sh pid   <PID>            [max_s] [summary_file]   # wait until process exits
#   wait_for.sh file  <PATH>           [max_s] [summary_file]   # wait until file exists (e.g. *.result / DONE flag)
#   wait_for.sh match <LOG> <REGEX>    [max_s] [summary_file]   # wait until regex appears in log (also stops on Error/Traceback)
# Exits before max_s (default 6900 s, under the 2 h background limit) with STATUS=STILL_RUNNING so it can be re-armed.
# Prints a short summary only: STATUS, elapsed, last 15 lines of summary_file (or the log).
mode=$1; shift
case $mode in
  pid)   target=$1; shift ;;
  file)  target=$1; shift ;;
  match) target=$1; regex=$2; shift 2 ;;
  *) echo "usage: wait_for.sh pid|file|match ..."; exit 2 ;;
esac
max=${1:-6900}; summary=${2:-}
t0=$(date +%s); status=STILL_RUNNING
while :; do
  case $mode in
    pid)   if ! tasklist //FI "PID eq $target" 2>/dev/null | grep -q " $target "; then status=DONE; break; fi ;;
    file)  [ -e "$target" ] && { status=DONE; break; } ;;
    match) if [ -f "$target" ]; then
             grep -E -q "$regex" "$target" && { status=DONE; break; }
             grep -E -q "Traceback|Error Object|FATAL|Segmentation|exceeded.*limit" "$target" && { status=ERROR_IN_LOG; break; }
           fi ;;
  esac
  [ $(( $(date +%s) - t0 )) -ge "$max" ] && break
  sleep 20
done
echo "STATUS=$status mode=$mode target=$target elapsed_s=$(( $(date +%s) - t0 ))"
f=${summary:-$([ "$mode" = match ] && echo "$target")}
[ -n "$f" ] && [ -f "$f" ] && tail -n 15 "$f"
exit 0
