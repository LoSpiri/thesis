"""E4.2 -- WTA / max decoding under spiking.

The paper reads: spiking neurons suppress high-frequency information; *max*
operators restore it. In S-FLM, `sampler.top_k_velocity = k` restricts the
flow velocity to the top-k tokens and renormalizes -- k=1 is a pure argmax
(Max / winner-take-all) velocity, the discrete analogue of Max-Pool. We sweep
k under float and under sigma-delta spiking to test whether leaning on the max
recovers the accuracy that spiking smooths away.

k=-1 disables top-k (exact velocity) and is the self-validating baseline.

Usage:
  python run_e4_wta.py --model sfm --steps 8 --subset 1319 --K 2 \
      --topks -1,1,2,4 --workers 12 --out results/e4_wta_sfm.json
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


def run(model, injector, all_ids, dataset, mode, K, topk, steps, batch,
        workers):
    model.sampler.top_k_velocity = topk
    injector.mode = mode
    injector.K = K
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
    p.add_argument("--topks", type=str, default="-1,1,2,4")
    p.add_argument("--modes", type=str, default="float,sigma_delta")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    topks = [int(x) for x in args.topks.split(",")]
    modes = [m for m in args.modes.split(",") if m]
    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    injector = SigmaDeltaInjector(model, mode="float", K=args.K,
                                  steps=args.steps).attach()
    results = {"model": args.model, "steps": args.steps, "subset": n,
               "K": args.K, "temperature": args.temperature, "runs": []}

    for topk in topks:
        for mode in modes:
            t0 = time.time()
            acc, sp = run(model, injector, all_ids, dataset, mode, args.K,
                          topk, args.steps, args.batch, args.workers)
            results["runs"].append({"topk_velocity": topk, "mode": mode,
                                    "acc": acc, "spikes": sp})
            vtag = " (baseline)" if topk == -1 else ""
            print(f"[{args.model}] topk={topk} {mode}: acc={acc:.4f} "
                  f"spikes={sp} ({time.time()-t0:.0f}s){vtag}", flush=True)

    injector.detach()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
