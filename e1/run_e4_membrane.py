"""E4.1 -- membrane shortcut: continuous bypass of the sigma-delta quantizer.

The paper's "membrane shortcut" connects the continuous membrane potential
rather than only the spikes, preserving (high-frequency) information. We mirror
it as a bypass of the spiking reconstruction:

    y_eff = y + lambda * (a - y)          (lambda in [0, 1])

lambda=0 reproduces the pure sigma-delta result and lambda=1 reproduces the
float model -- both are self-validating rows. Intermediate lambda tests whether
leaking a fraction of the sub-threshold error (the "membrane") through a
continuously-valued shortcut recovers accuracy at little/no extra spike cost.
Only the quantized part is counted as spikes; the bypass is analog.

Usage:
  python run_e4_membrane.py --model sfm --steps 8 --subset 1319 --K 2 \
      --bypass 0,0.25,0.5,0.75,1 --workers 12 --out results/e4_membrane_sfm.json
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


def run(injector, model, all_ids, dataset, mode, K, bypass, steps, batch,
        workers):
    injector.mode = mode
    injector.K = K
    injector.bypass = bypass
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
    p.add_argument("--bypass", type=str, default="0,0.25,0.5,0.75,1")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    lams = [float(x) for x in args.bypass.split(",")]
    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    injector = SigmaDeltaInjector(model, mode="sigma_delta", K=args.K, steps=args.steps).attach()
    results = {"model": args.model, "steps": args.steps, "subset": n,
               "K": args.K, "temperature": args.temperature, "runs": []}

    t0 = time.time()
    acc_f, _ = run(injector, model, all_ids, dataset, "float", args.K, 0.0,
                   args.steps, args.batch, args.workers)
    results["float_accuracy"] = acc_f
    print(f"[{args.model}] float: acc={acc_f:.4f} ({time.time()-t0:.0f}s)",
          flush=True)

    for lam in lams:
        t0 = time.time()
        acc, sp = run(injector, model, all_ids, dataset, "sigma_delta",
                      args.K, lam, args.steps, args.batch, args.workers)
        results["runs"].append({"lambda": lam, "acc": acc, "spikes": sp})
        vtag = " (validation=baseline)" if lam == 0.0 else (
            " (validation=float)" if lam == 1.0 else "")
        print(f"[{args.model}] lambda={lam}: acc={acc:.4f} spikes={sp} "
              f"({time.time()-t0:.0f}s){vtag}", flush=True)

    injector.detach()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
