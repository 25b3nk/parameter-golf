#!/usr/bin/env bash
# autoresearch/run.sh — Run one experiment for parameter golf autoresearch loop.
#
# Usage:
#   ./autoresearch/run.sh                        # 1xGPU, 5-min budget (cheap iteration)
#   ./autoresearch/run.sh --gpus 8 --time 600    # 8xGPU, 10-min budget (leaderboard run)
#
# Output is written to autoresearch/run.log

set -euo pipefail

GPUS=1
TIME=300  # seconds (5 min default for cheap 1xH100 testing)
RUN_ID="exp_$(date +%s)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --id)   RUN_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOGFILE="autoresearch/run.log"
echo "============================="
echo "  Parameter Golf Experiment"
echo "  RUN_ID  : $RUN_ID"
echo "  GPUs    : $GPUS"
echo "  Budget  : ${TIME}s"
echo "  Log     : $LOGFILE"
echo "============================="

RUN_ID="$RUN_ID" \
DATA_PATH="./data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS="$TIME" \
torchrun --standalone --nproc_per_node="$GPUS" train_gpt.py > "$LOGFILE" 2>&1

echo ""
echo "=== Results ==="
grep -E "final_int8_zlib_roundtrip val|Total submission size int8|peak memory" "$LOGFILE" | tail -5
echo ""
echo "Log saved to $LOGFILE"
