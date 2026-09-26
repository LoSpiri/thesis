#!/usr/bin/env bash
# RunPod Phase 0: flow-vs-masked under spiking + adaptive-threshold experiments.
#
# Tuned for RTX PRO 4000 (24 GB, 12 vCPU): batch 16 + 12 sandbox workers.
# Deploy with the network volume attached at /workspace so results persist.
# At the end it best-effort stops the pod to release the GPU -- set
# RUNPOD_API_KEY as a Pod environment variable at deploy time to enable it.
set -euo pipefail

set -a; source e1/env.runpod; set +a
mkdir -p e1/results

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
  "tinygsm/mdlm.ckpt",
  "tinygsm/duo.ckpt",
  "tinygsm/flm/default.ckpt",
]
for f in files:
    hf_hub_download("jdeschena/s-flm", f, local_dir="checkpoints")
    print("ok", f, flush=True)
PY

echo "==> (1) flow vs masked under spiking @ NFE in {16,32}, K=1,2,4 (full 1319)"
for NFE in 16 32; do
  for M in mdlm duo sfm sfm-dit; do
    echo "---- $M @ NFE=$NFE ----"
    python e1/run_e2.py --model "$M" --steps "$NFE" --subset 1319 --K 1,2,4 \
      --out "e1/results/e2_${M}_s${NFE}.json"
  done
done

echo "==> (2) adaptive threshold: bucket schedule (sphere-arch, K=2, NFE=8)"
python e1/run_e2_adaptive.py --model sfm --steps 8 --subset 1319 --K 2 \
  --mode bucket --bucket-b 0,0.5,1,2 --out e1/results/e2a_bucket_sfm.json

echo "==> (3) adaptive threshold: magnitude-adaptive tau (sphere-arch, K=2, NFE=8)"
python e1/run_e2_adaptive.py --model sfm --steps 8 --subset 1319 --K 2 \
  --mode magnitude --out e1/results/e2a_magnitude_sfm.json

echo "==> DONE. results in e1/results/ (persisted on the network volume):"
ls -la e1/results/ || true

# Best-effort stop so the GPU is released (results persist on the volume).
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "==> stopping pod ${RUNPOD_POD_ID} (GPU released; volume persists)"
  curl -s -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
else
  echo "==> RUNPOD_API_KEY / RUNPOD_POD_ID not set: NOT auto-stopping."
  echo "    Stop the pod manually to release the GPU (results persist on the volume)."
fi
