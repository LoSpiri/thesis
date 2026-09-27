"""E4.3 -- max-style high-pass token mixer in the early blocks.

Softmax attention is a weighted average (low-pass); the paper's Max-Pool / DWC
are high-pass restorers. We add a cheap, training-free high-pass residual to
the output of the first N transformer blocks:

    x <- x + beta * highpass(x)

highpass modes:
  center -- x - local_mean(x)   (depth-wise conv over the token axis; DWC analogue)
  diff   -- x[:, i] - x[:, i-1] (first-order difference along tokens)
  global -- x - mean_tokens(x)  (remove the low-frequency DC component)

beta=0 reproduces the unmodified model (validation). The mixer is evaluated
under sigma-delta spiking (K) to test whether restoring high-frequency content
recovers accuracy; a float+beta row isolates the mixer's effect without spiking.

Usage:
  python run_e4_mixer.py --model sfm --steps 8 --subset 1319 --K 2 \
      --blocks 2 --mode center --betas 0,0.25,0.5,1 --workers 12 \
      --out results/e4_mixer_sfm.json
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
import torch.nn.functional as F

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


def highpass(x, mode, window):
    if mode == "global":
        return x - x.mean(dim=1, keepdim=True)
    if mode == "diff":
        hp = torch.zeros_like(x)
        hp[:, 1:] = x[:, 1:] - x[:, :-1]
        return hp
    if mode == "center":
        xt = x.transpose(1, 2)                       # [B, d, L]
        pad = window // 2
        xtp = F.pad(xt, (pad, window - 1 - pad), mode="replicate")
        mean = F.avg_pool1d(xtp, kernel_size=window, stride=1)
        return (xt - mean).transpose(1, 2)
    raise ValueError(mode)


def attach_mixers(model, blocks, beta, mode, window):
    hooks = []

    def make(i):
        def hook(module, inp, out):
            if not isinstance(out, torch.Tensor):
                return out
            return out + beta * highpass(out, mode, window)
        return hook

    for i in blocks:
        hooks.append(model.backbone.blocks[i].register_forward_hook(make(i)))
    return hooks


def run(model, injector, all_ids, dataset, mode_spike, K, mixer, steps, batch,
        workers):
    injector.mode = mode_spike
    injector.K = K
    hooks = attach_mixers(model, *mixer) if mixer is not None else []
    torch.manual_seed(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    n = len(all_ids)
    pairs = []
    try:
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
    finally:
        for h in hooks:
            h.remove()
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
    p.add_argument("--blocks", type=int, default=2,
                   help="number of leading blocks to modify")
    p.add_argument("--mode", choices=["center", "diff", "global"],
                   default="center")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--betas", type=str, default="0,0.25,0.5,1")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    betas = [float(x) for x in args.betas.split(",")]
    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]
    blocks = list(range(min(args.blocks, len(model.backbone.blocks))))

    injector = SigmaDeltaInjector(model, mode="float", K=args.K,
                                  steps=args.steps).attach()
    results = {"model": args.model, "steps": args.steps, "subset": n,
               "K": args.K, "temperature": args.temperature, "mode": args.mode,
               "window": args.window, "blocks": blocks, "runs": []}

    t0 = time.time()
    acc_f, _ = run(model, injector, all_ids, dataset, "float", args.K, None,
                   args.steps, args.batch, args.workers)
    results["float_accuracy"] = acc_f
    print(f"[{args.model}] float: acc={acc_f:.4f} ({time.time()-t0:.0f}s)",
          flush=True)

    for beta in betas:
        mixer = (blocks, beta, args.mode, args.window)
        t0 = time.time()
        acc, sp = run(model, injector, all_ids, dataset, "sigma_delta",
                      args.K, mixer, args.steps, args.batch, args.workers)
        results["runs"].append({"beta": beta, "spike_mode": "sigma_delta",
                                "acc": acc, "spikes": sp})
        vtag = " (validation=baseline)" if beta == 0.0 else ""
        print(f"[{args.model}] {args.mode} beta={beta} sigma_delta: "
              f"acc={acc:.4f} spikes={sp} ({time.time()-t0:.0f}s){vtag}",
              flush=True)

    injector.detach()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
