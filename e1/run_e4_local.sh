#!/usr/bin/env bash
# Local (Apple MPS) smoke/quick pass for the E4 high-frequency experiments.
# Small subsets only -- full-1319 runs live on RunPod (see runpod_e4.sh).
#
# Usage:
#   bash e1/run_e4_local.sh                 # defaults: sfm, subset 32
#   SUBSET=64 MODEL=sfm-dit bash e1/run_e4_local.sh
set -euo pipefail

PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-sfm}"
SUBSET="${SUBSET:-32}"
STEPS="${STEPS:-8}"
K="${K:-2}"
OUT="${OUT:-e1/results}"
mkdir -p "$OUT"

echo "== E4 local ($MODEL, subset=$SUBSET, steps=$STEPS, K=$K) =="

echo "==> E4.0 spectrum diagnostic"
"$PY" e1/run_e4_spectrum.py --model "$MODEL" --steps "$STEPS" --K "$K" \
  --length 256 --subset 8 --batch 8 --blocks 0,-1 \
  --out "$OUT/e4_spectrum_${MODEL}.json"

echo "==> E4.1 membrane shortcut (lambda sweep)"
"$PY" e1/run_e4_membrane.py --model "$MODEL" --steps "$STEPS" --subset "$SUBSET" \
  --batch 8 --K "$K" --bypass 0,0.25,0.5,0.75,1 --workers 8 \
  --out "$OUT/e4_membrane_${MODEL}_K${K}.json"

echo "==> E4.2 WTA / top-k velocity"
"$PY" e1/run_e4_wta.py --model "$MODEL" --steps "$STEPS" --subset "$SUBSET" \
  --batch 8 --K "$K" --topks=-1,1,2,4 --modes float,sigma_delta --workers 8 \
  --out "$OUT/e4_wta_${MODEL}_K${K}.json"

echo "==> E4.3 max-style early token mixer"
"$PY" e1/run_e4_mixer.py --model "$MODEL" --steps "$STEPS" --subset "$SUBSET" \
  --batch 8 --K "$K" --blocks 2 --mode center --betas 0,0.25,0.5,1 --workers 8 \
  --out "$OUT/e4_mixer_${MODEL}_K${K}.json"

echo "==> E4.4 frequency-selective bypass"
"$PY" e1/run_e4_freqbypass.py --model "$MODEL" --steps "$STEPS" --subset "$SUBSET" \
  --batch 8 --K "$K" --alpha 0.5,0.8 --gamma 0,0.5,1,1.5,2 --workers 8 \
  --out "$OUT/e4_freqbypass_${MODEL}_K${K}.json"

echo "== DONE. results in $OUT =="
ls -la "$OUT"/e4_*.json 2>/dev/null || true
