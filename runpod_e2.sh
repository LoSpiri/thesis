#!/usr/bin/env bash
# RunPod setup for the E2/E3 follow-up experiments (full 1319, both models).
#
# Tuned for RTX PRO 4000 (24 GB, 12 vCPU): batch 16 + 12 sandbox workers
# (parallel_eval.py). The scripts default to these values.
#
# Run from the cloned repo root. Installs deps, downloads checkpoints, then
# runs: (1) full-scale E2.3, (2) layer-group hybrid, (3) adaptive threshold.
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

echo "==> (1) full-scale E2.3 (both models, K=1,2,4)"
python e1/run_e2.py --model sfm --steps 8 --subset 1319 --K 1,2,4 --out e1/results/e2_sfm_full.json
python e1/run_e2.py --model sfm-dit --steps 8 --subset 1319 --K 1,2,4 --out e1/results/e2_sfm-dit_full.json

echo "==> (2) layer-group hybrid (both models, K=2)"
python e1/run_e2_groups.py --model sfm --subset 1319 --K 2 --out e1/results/e2g_sfm.json
python e1/run_e2_groups.py --model sfm-dit --subset 1319 --K 2 --out e1/results/e2g_sfm-dit.json

echo "==> (3) adaptive threshold (both models, K=2, a=0.5,1,2,4)"
python e1/run_e2_adaptive.py --model sfm --steps 8 --subset 1319 --K 2 --A 0.5,1,2,4 --out e1/results/e2a_sfm.json
python e1/run_e2_adaptive.py --model sfm-dit --steps 8 --subset 1319 --K 2 --A 0.5,1,2,4 --out e1/results/e2a_sfm-dit.json

echo "==> done"
