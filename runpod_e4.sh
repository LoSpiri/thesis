#!/usr/bin/env bash
# RunPod driver for the E4 "high-frequency information" experiments (full 1319).
#
# Motivated by "Spiking Neural Networks Need High-Frequency Information"
# (Fang et al., 2024). Training-free levers on the S-FLM flow + sigma-delta
# spiking forward pass:
#   E4.0 spectrum     -- is the sigma-delta LIF low-passing the flow? (diagnostic)
#   E4.1 membrane     -- continuous bypass of the quantizer (membrane shortcut)
#   E4.2 WTA          -- top-k / argmax (Max) velocity under spiking
#   E4.3 mixer        -- high-pass token-mixing residual in the early blocks
#   E4.4 freqbypass   -- frequency-selective bypass (reject / boost the shaped
#                        high-frequency quantization noise)
#
# Tuned for RTX PRO 4000 (24 GB, 12 vCPU): batch 16 + 12 sandbox workers.
#
# NOTE: run on the CONTAINER disk (e.g. /root/s-flm), not the network volume
# (its FS rejects chmod and git fails). Results are written DIRECTLY to the
# network volume (/workspace/results) so each section survives a spot
# preemption; if the volume is not writable it falls back to e1/results.
# The pod is best-effort stopped at the end (set RUNPOD_API_KEY as a Pod env).
#
# K_LIST selects the quantization resolutions swept (default "2"; set
# K_LIST="1 2" to re-enable the coarse K=1 rows).
set -euo pipefail

set -a; source e1/env.runpod; set +a

PERSIST_DIR="${PERSIST_DIR:-/workspace/results}"
if mkdir -p "$PERSIST_DIR" 2>/dev/null && touch "$PERSIST_DIR/.wtest" 2>/dev/null; then
  rm -f "$PERSIST_DIR/.wtest"
  OUT_DIR="$PERSIST_DIR"
else
  echo "==> ${PERSIST_DIR} not writable; falling back to e1/results (container disk)"
  OUT_DIR="e1/results"
fi
mkdir -p "$OUT_DIR"
K_LIST="${K_LIST:-2}"
echo "==> outputs -> ${OUT_DIR}; K_LIST='${K_LIST}'"

echo "==> installing deps"
pip install -q hydra-core==1.3.2 omegaconf==2.3.0 lightning==2.5.1 \
  einops==0.8.1 fancy-einsum==0.0.3 tqdm==4.67.1 rich==13.9.4 termcolor==3.0.1 \
  pyyaml==6.0.2 fsspec blobfile==3.0.0 transformers==4.45.0 tokenizers==0.20.3 \
  datasets==3.5.0 huggingface-hub==0.30.2 safetensors==0.5.2 sentencepiece==0.2.0 \
  evaluate==0.4.3 matplotlib torchmetrics==1.7.1 pandas==2.2.1 scikit-learn==1.4.0 \
  jinja2==3.1.5 timm==1.0.15 scipy hf_transfer

echo "==> downloading checkpoints (~13 GB)"
if ! python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
python - <<'PY'
from huggingface_hub import hf_hub_download
files = [
  "tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt",
  "tinygsm/sfm/sphere_dit_truncated_adaptive_no_renorm.ckpt",
]
for f in files:
    hf_hub_download("jdeschena/s-flm", f, local_dir="checkpoints")
    print("ok", f, flush=True)
PY

echo "==> (E4.0) frequency spectrum diagnostic (small subset; cheap)"
for M in sfm sfm-dit; do
  python e1/run_e4_spectrum.py --model "$M" --steps 8 --K 2 --length 512 \
    --subset 16 --batch 8 --blocks 0,-1 \
    --out "${OUT_DIR}/e4_spectrum_${M}.json"
done

echo "==> (E4.1) membrane shortcut: lambda sweep, K in {${K_LIST}}, full 1319"
for M in sfm sfm-dit; do
  for KK in $K_LIST; do
    python e1/run_e4_membrane.py --model "$M" --steps 8 --subset 1319 \
      --K "$KK" --bypass 0,0.25,0.5,0.75,1 --workers 12 \
      --out "${OUT_DIR}/e4_membrane_${M}_K${KK}.json"
  done
done

echo "==> (E4.2) WTA / top-k velocity, K in {${K_LIST}}, full 1319"
for M in sfm sfm-dit; do
  for KK in $K_LIST; do
    python e1/run_e4_wta.py --model "$M" --steps 8 --subset 1319 \
      --K "$KK" --topks=-1,1,2,4 --modes float,sigma_delta --workers 12 \
      --out "${OUT_DIR}/e4_wta_${M}_K${KK}.json"
  done
done

echo "==> (E4.3) max-style early token mixer, K=2, full 1319"
for M in sfm sfm-dit; do
  python e1/run_e4_mixer.py --model "$M" --steps 8 --subset 1319 \
    --K 2 --blocks 2 --mode center --betas 0,0.25,0.5,1 --workers 12 \
    --out "${OUT_DIR}/e4_mixer_center_${M}_K2.json"
done
python e1/run_e4_mixer.py --model sfm --steps 8 --subset 1319 \
  --K 2 --blocks 2 --mode diff --betas 0,0.25,0.5,1 --workers 12 \
  --out "${OUT_DIR}/e4_mixer_diff_sfm_K2.json"

echo "==> (E4.4) frequency-selective bypass, K in {${K_LIST}}, full 1319"
for M in sfm sfm-dit; do
  for KK in $K_LIST; do
    python e1/run_e4_freqbypass.py --model "$M" --steps 8 --subset 1319 \
      --K "$KK" --alpha 0.5,0.8 --gamma 0,0.5,1,1.5,2 --workers 12 \
      --out "${OUT_DIR}/e4_freqbypass_${M}_K${KK}.json"
  done
done

echo "==> DONE. results in ${OUT_DIR} :"
ls -la "${OUT_DIR}" || true

# Best-effort stop so the GPU is released (results persist on the volume).
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "==> stopping pod ${RUNPOD_POD_ID} (GPU released; volume persists)"
  curl -s -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
else
  echo "==> RUNPOD_API_KEY / RUNPOD_POD_ID not set: NOT auto-stopping."
  echo "    Stop the pod manually to release the GPU (results persist on the volume)."
fi
