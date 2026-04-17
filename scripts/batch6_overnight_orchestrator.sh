#!/bin/bash
# Overnight Batch 6 driver: tile all 4 Planck cleaned maps at HEALPix Nside=32
# (~9500 patches per map after common-mask filter at unmasked-fraction >= 0.5),
# then run the 3-policy cross-map recalibration analysis at paper-grade
# statistical power.
#
# Schedule: SMICA + NILC on GPU 0 sequentially, SEVEM + Commander on GPU 1
# sequentially. Pair-parallel; total wall clock ~5h for 4 maps. Analysis runs
# after all 4 tiles complete.
#
# Usage: bash scripts/batch6_overnight_orchestrator.sh
#        (run from project root)

set -u  # fail on undefined var; do NOT use -e because we want to keep going
        # even if one map fails

PROJECT_ROOT=/data/william/CMB-Collision-Bubbles
LOG_DIR="$PROJECT_ROOT/work/batch6_logs"
RUN_ROOT="$PROJECT_ROOT/runs/phase3_unet"
PYTHON=/home/mtsu/miniconda3/envs/cmb/bin/python

mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/orchestrator.log"

echo "===== Batch 6 overnight driver started: $(date -Iseconds) =====" > "$ORCH_LOG"

run_map() {
  local map=$1
  local gpu=$2
  local out="$RUN_ROOT/batch6_fullsky_nside32_$map"
  local log="$LOG_DIR/${map}_nside32.log"
  echo "[$(date -Iseconds)] [GPU$gpu] start $map -> $out" >> "$ORCH_LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$PROJECT_ROOT/scripts/phase3_fullsky_tile.py" \
    --map "$map" \
    --tile-nside 32 \
    --mask-threshold 0.5 \
    --output-dir "$out" \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date -Iseconds)] [GPU$gpu] done  $map (exit $rc)" >> "$ORCH_LOG"
  return $rc
}

(
  for m in smica nilc; do
    run_map "$m" 0
  done
  echo "[$(date -Iseconds)] GPU0 lane finished" >> "$ORCH_LOG"
) &
GPU0_PID=$!

(
  for m in sevem commander; do
    run_map "$m" 1
  done
  echo "[$(date -Iseconds)] GPU1 lane finished" >> "$ORCH_LOG"
) &
GPU1_PID=$!

wait $GPU0_PID
wait $GPU1_PID
echo "[$(date -Iseconds)] ALL TILE JOBS COMPLETE" >> "$ORCH_LOG"

# Run analysis. Failure here is non-fatal; user can re-run by hand.
ANALYSIS_LOG="$LOG_DIR/analysis_nside32.log"
echo "[$(date -Iseconds)] starting analysis" >> "$ORCH_LOG"
"$PYTHON" "$PROJECT_ROOT/scripts/batch6_overnight_analysis.py" \
  > "$ANALYSIS_LOG" 2>&1
echo "[$(date -Iseconds)] analysis exit $?" >> "$ORCH_LOG"

echo "===== Batch 6 overnight driver done:    $(date -Iseconds) =====" >> "$ORCH_LOG"
