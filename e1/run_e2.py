"""E2.3 headline: in-the-loop spiking reconstruction vs float accuracy.

Runs the GSM8K sampler with each nn.Linear input replaced by its spiking
reconstruction (sigma-delta or stateless), and measures end-to-end accuracy
plus total spike count. Sweeps quantization resolution K. Sandbox scoring is
parallelized across --workers processes.

Usage:
  python run_e2.py --model sfm --steps 8 --subset 1319 --K 1,2,4 --workers 12
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


def run_mode(model, injector, all_ids, dataset, mode, K, steps, batch, workers):
    injector.mode = mode
    injector.K = K
    injector.reset()
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
    correct = sum(scores)
    return correct / max(n, 1), injector.total()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit", "mdlm", "duo", "flm"],
                   required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--subset", type=int, default=1319)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--K", type=str, default="1,2,4,8")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    Ks = [int(x) for x in args.K.split(",")]

    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  top_k_velocity=args.topk,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    injector = SigmaDeltaInjector(model, mode="float", K=1.0).attach()
    results = {"model": args.model, "steps": args.steps, "subset": n,
               "temperature": args.temperature, "runs": []}

    t0 = time.time()
    acc, _ = run_mode(model, injector, all_ids, dataset, "float", 1,
                      args.steps, args.batch, args.workers)
    results["float_accuracy"] = acc
    print(f"[{args.model}] float: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    for K in Ks:
        t0 = time.time()
        acc_sd, sp_sd = run_mode(model, injector, all_ids, dataset,
                                 "sigma_delta", K, args.steps, args.batch, args.workers)
        acc_st, sp_st = run_mode(model, injector, all_ids, dataset,
                                 "stateless", K, args.steps, args.batch, args.workers)
        results["runs"].append({"K": K,
                                "sigma_delta_acc": acc_sd, "sigma_delta_spikes": sp_sd,
                                "stateless_acc": acc_st, "stateless_spikes": sp_st})
        print(f"[{args.model}] K={K}: sd acc={acc_sd:.4f} spikes={sp_sd} | "
              f"st acc={acc_st:.4f} spikes={sp_st} ({time.time()-t0:.0f}s)", flush=True)

    injector.detach()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
