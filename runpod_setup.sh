#!/usr/bin/env bash
# RunPod setup script for the E1 spiking-flow experiment.
#
# Run this from the cloned repo root (after `git clone ... s-flm; cd s-flm`).
# torch/CUDA already come from the RunPod PyTorch template; this installs the
# remaining deps, downloads the TinyGSM checkpoints, and launches the sweep.
set -euo pipefail

# Source RunPod env (QUALITY_BATCH=32)
set -a; source e1/env.runpod; set +a

echo "==> installing deps"
pip install -q hydra-core==1.3.2 omegaconf==2.3.0 lightning==2.5.1 \
  einops==0.8.1 fancy-einsum==0.0.3 tqdm==4.67.1 rich==13.9.4 termcolor==3.0.1 \
  pyyaml==6.0.2 fsspec blobfile==3.0.0 transformers==4.45.0 tokenizers==0.20.3 \
  datasets==3.5.0 huggingface-hub==0.30.2 safetensors==0.5.2 sentencepiece==0.2.0 \
  evaluate==0.4.3 matplotlib torchmetrics==1.7.1 pandas==2.2.1 scikit-learn==1.4.0 \
  jinja2==3.1.5 timm==1.0.15 scipy hf_transfer

# If lightning 2.5.1 errors against the template's torch 2.8, uncomment:
# pip install -U lightning

echo "==> downloading checkpoints (~13 GB)"
# Some RunPod templates set HF_HUB_ENABLE_HF_TRANSFER=1 but lack hf_transfer.
# Fall back to plain HTTP if hf_transfer isn't importable.
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

echo "==> launching T=0.1 sweep (resume-safe)"
nohup python e1/run_all.py --temp 0.1 > e1/run_runpod.log 2>&1 &
echo "started — watch with:  tail -f e1/run_runpod.log"
