#!/usr/bin/env bash
# Lambda sweep runner with a system-RAM guard (review P1-4 rerun, 2026-10-07).
#
# Memory strategy (math-neutral only -- scientific hyperparameters untouched):
#   * one training process at a time, sequential
#   * before each run, poll free physical RAM until >= 3000 MB (max 30 min)
#   * cap CPU thread pools (OMP/MKL) and tokenizer threads to shrink
#     per-process CPU-side buffers; expandable_segments for the CUDA allocator
#   * nothing else heavy should run concurrently with this script
set -u
cd "$(dirname "$0")/.." || exit 1
PY="D:/Users/27682/anaconda3/envs/tinytoolrl/python.exe"

free_mb() {
  powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tr -d '\r\n '
}

run_one() {
  reward="$1"; lam="$2"; tag="$3"
  local f=0 i
  for i in $(seq 1 60); do
    f="$(free_mb)"
    [ "${f:-0}" -ge 3000 ] && break
    sleep 30
  done
  echo "=== START reward=$reward lam=$lam tag=$tag free=${f}MB at $(date +%H:%M:%S) ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" train_grpo.py --reward "$reward" --lam "$lam" --tag "$tag" 2>&1 | tail -3
  echo "=== END reward=$reward lam=$lam tag=$tag at $(date +%H:%M:%S) ==="
}

run_one fixed 0.3 lam03
run_one cap   0.3 lam03
run_one fixed 0.5 lam05
run_one cap   0.5 lam05
echo "=== SWEEP ALL DONE at $(date +%H:%M:%S) ==="
