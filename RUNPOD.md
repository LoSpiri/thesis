# Running the E1 experiment on RunPod

Fast CUDA execution of the sigma-delta sparsity sweep + full-GSM8K quality
sweep (vs ~6 h on Apple MPS).

## 1. Deploy a pod

1. `runpod.io` → **Pods** → **Deploy**.
2. GPU: **RTX PRO 4000 (24 GB)** (or any 24 GB+ card).
3. Template: **RunPod PyTorch 2.8.0** (any recent PyTorch template works).
4. **Container disk: 40 GB** (13 GB checkpoints + code + deps).
5. Cloud type: Community/spot is fine — the runner resumes.

## 2. Connect

Pod → **Connect** → **Connect to Web Terminal**.

## 3. Run

```bash
cd /workspace
git clone https://github.com/LoSpiri/thesis.git s-flm
cd s-flm
bash runpod_setup.sh
tail -f e1/run_runpod.log
```

`runpod_setup.sh` installs deps, downloads the 5 TinyGSM checkpoints from HF,
and launches the sweep under `nohup`. It sources `e1/env.runpod`
(`QUALITY_BATCH=32`).

## 4. Monitor

- ~20 sigma-delta runs (fast), then ~26 full-GSM8K quality runs (~40–60 min).
- `e1/REPORT.md` is auto-generated when done.
- `Ctrl-C` on `tail` anytime; the run keeps going (it's `nohup`-ed).
- Resume: re-running `bash runpod_setup.sh` (or `python e1/run_all.py --temp 0.1`)
  skips already-finished results.

## 5. Stop

Pod → **Stop/Terminate** when finished so you stop being billed.

## Local (Mac MPS) equivalent

```bash
set -a; source e1/env.local; set +a   # QUALITY_BATCH=16
python e1/run_all.py --temp 0.1
```
