"""E4.4 -- frequency-selective bypass of the sigma-delta reconstruction.

E4.0 showed sigma-delta is signal-transparent with its quantization error
shaped to HIGH frequencies (y = a + (1 - z^-1) e). This script splits the spike
reconstruction into a causal low band and a high band and recombines:

    y_lp   = alpha * y_lp + (1 - alpha) * y     # causal EMA low-pass
    y_freq = y_lp + gamma * (y - y_lp)          # gamma = high-frequency gain

  gamma = 0  -> pure low-pass: reject the shaped high-frequency noise
               (the sigma-delta "decimation filter" direction);
  gamma = 1  -> exact validation (no change);
  gamma > 1  -> amplify high frequencies (the paper's "restore high-frequency"
               direction).

alpha=0 also collapses to the baseline for any gamma (validation row).

Usage:
  python run_e4_freqbypass.py --model sfm --steps 8 --subset 1319 --K 2 \
      --alpha 0.5,0.8 --gamma 0,0.5,1,1.5,2 --workers 12 \
      --out results/e4_freqbypass_sfm_K2.json
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch

import dataloader
from loader import build
from spike_inject import SigmaDeltaInjector
from parallel_eval import score_all


def pad_prefix_batch(id_lists, device):
    lengths = torch.tensor([len(ids) for ids in id_lists], device=device)
    max_len = int(lengths.max())
    padded = torch.zeros(len(id_lists), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(id_lists):
        padded[i, :len(ids)] = ids.to(device)
    return padded, lengths


def run(injector, model, all_ids, dataset, K, alpha, gamma, steps, batch,
        workers):
    injector.mode = "sigma_delta"
    injector.K = K
    injector.freq_alpha = alpha
    injector.freq_gamma = gamma
    torch.manual_seed(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    n = len(all_ids)
    pairs = []
    for start in range(0, n, batch):
        idxs = list(range(start, min(start + batch, n)))
        b_ids = [all_ids[i] for i in idxs]
        padded, lengths = pad_prefix_batch(b_ids, model.device)
        injector.reset()
        with torch.no_grad():
            samples, _ = model.generate_samples(
                num_samples=len(idxs), num_steps=steps,
                prefix_tokens=padded, prefix_lengths=lengths)
        for j, i in enumerate(idxs):
            pl = int(lengths[j])
            resp = model.tokenizer.decode(
                samples[j, pl:].cpu().tolist(), skip_special_tokens=True)
            pairs.append((resp, dataset[i]["response_ground_truth"]))
    scores = score_all(pairs, workers=workers)
    return sum(scores) / max(n, 1), injector.total()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit"], required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--subset", type=int, default=1319)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--K", type=float, default=2.0)
    p.add_argument("--alpha", type=str, default="0.5,0.8")
    p.add_argument("--gamma", type=str, default="0,0.5,1,1.5,2")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    alphas = [float(x) for x in args.alpha.split(",")]
    gammas = [float(x) for x in args.gamma.split(",")]
    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    injector = SigmaDeltaInjector(model, mode="sigma_delta", K=args.K,
                                  steps=args.steps).attach()
    results = {"model": args.model, "steps": args.steps, "subset": n,
               "K": args.K, "temperature": args.temperature, "runs": []}

    t0 = time.time()
    acc_f, _ = run(injector, model, all_ids, dataset, args.K, 0.0, 1.0,
                   args.steps, args.batch, args.workers)
    results["float_accuracy"] = acc_f
    print(f"[{args.model}] float: acc={acc_f:.4f} ({time.time()-t0:.0f}s)",
          flush=True)

    # validation: alpha=0,gamma=1 must reproduce the pure sigma-delta baseline
    t0 = time.time()
    acc0, sp0 = run(injector, model, all_ids, dataset, args.K, 0.0, 1.0,
                    args.steps, args.batch, args.workers)
    results["runs"].append({"alpha": 0.0, "gamma": 1.0, "acc": acc0,
                            "spikes": sp0, "validation": "baseline"})
    print(f"[{args.model}] alpha=0 gamma=1: acc={acc0:.4f} spikes={sp0} "
          f"({time.time()-t0:.0f}s) (validation=baseline)", flush=True)

    for alpha in alphas:
        for gamma in gammas:
            t0 = time.time()
            acc, sp = run(injector, model, all_ids, dataset, args.K, alpha,
                          gamma, args.steps, args.batch, args.workers)
            results["runs"].append({"alpha": alpha, "gamma": gamma, "acc": acc,
                                    "spikes": sp})
            vtag = " (validation)" if gamma == 1.0 else ""
            print(f"[{args.model}] alpha={alpha} gamma={gamma}: acc={acc:.4f} "
                  f"spikes={sp} ({time.time()-t0:.0f}s){vtag}", flush=True)

    injector.detach()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
