#!/usr/bin/env bash
set -euo pipefail

DEST="./download"
SCRIPT="examples/citation_processing_batch.py"
LOG_DIR="./citation_log"
SRC_DIR="./csv"

mkdir -p "$DEST" "$LOG_DIR"

for i in {11..20}; do
  SRC="${SRC_DIR}/arxiv_id_list${i}.csv"
  LOG="${LOG_DIR}/citation_list${i}.log"
  DONE="${LOG}.done"

  echo "=== Processing $SRC ==="

  if [[ ! -f "$SRC" ]]; then
    echo "Missing source: $SRC" | tee -a "$LOG"
    continue
  fi

  if [[ -f "$DONE" && "${FORCE:-0}" != "1" ]]; then
    echo "Already completed, skipping (set FORCE=1 to re-run)." | tee -a "$LOG"
    continue
  fi

  set +e
  python -u "$SCRIPT" --source "$SRC" --dest "$DEST" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  if (( status == 0 )); then
    echo "OK $SRC" | tee -a "$LOG"
    : > "$DONE"
  else
    echo "⚠️  FAILED $SRC (exit $status) — continuing" | tee -a "$LOG"
  fi
done
