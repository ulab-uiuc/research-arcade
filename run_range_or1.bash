#!/usr/bin/env bash
set -euo pipefail

DEST="./download"
SCRIPT="examples/openreview_list_crawling.py"

mkdir -p "$DEST" ./openreview_log

for i in {1..10}; do
  SRC="./arxiv_id_openreview/arxiv_id_list${i}.json"
  LOG="./openreview_log/arxiv_id_list${i}.log"
  DONE="${LOG}.done"
  
  echo "=== Processing $SRC ==="

  if [[ -f "$DONE" && "${FORCE:-0}" != "1" ]]; then
    echo "Already completed, skipping (set FORCE=1 to re-run)." | tee -a "$LOG"
  else
    # Run and continue even if the Python command fails
    set +e
    python -u "$SCRIPT" \
      --source "$SRC" --dest "$DEST" \
      2>&1 | tee -a "$LOG"
    status=${PIPESTATUS[0]}  # exit code of python
    set -e

    if (( status == 0 )); then
      echo "OK $SRC" | tee -a "$LOG"
      touch "$DONE"
    else
      echo "⚠️  FAILED $SRC (exit $status) — continuing" | tee -a "$LOG"
    fi
  fi
done
