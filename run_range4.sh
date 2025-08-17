#!/usr/bin/env bash
set -euo pipefail

START="2025-05-02"
END="2025-06-01"      # inclusive
FIELD="cs"
DEST="./download"
SCRIPT="examples/daily_cs_crawling.py"

mkdir -p "$DEST" logs

# Increment YYYY-MM-DD by one day (GNU date -> gdate -> BSD date)
inc_day() {
  local d="$1"
  if date -d "$d + 1 day" +%F >/dev/null 2>&1; then
    date -d "$d + 1 day" +%F
  elif command -v gdate >/dev/null 2>&1; then
    gdate -d "$d + 1 day" +%F
  else
    date -j -f "%Y-%m-%d" "$d" -v+1d "+%Y-%m-%d"
  fi
}

d="$START"
while :; do
  LOG="./logs/graph_log_${d}_${d}.log"
  DONE="${LOG}.done"

  echo "=== $d ==="

  if [[ -f "$DONE" && "${FORCE:-0}" != "1" ]]; then
    echo "Already completed, skipping (set FORCE=1 to re-run)." | tee -a "$LOG"
  else
    # Run and continue even if the Python command fails
    set +e
    python -u "$SCRIPT" \
      --start-date "$d" --end-date "$d" \
      --field "$FIELD" --dest "$DEST" \
      2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}  # exit code of python
    set -e

    if (( status == 0 )); then
      echo "OK $d" | tee -a "$LOG"
      touch "$DONE"
    else
      echo "⚠️  FAILED $d (exit $status) — continuing" | tee -a "$LOG"
    fi
  fi

  [[ "$d" == "$END" ]] && break
  d="$(inc_day "$d")"
done
