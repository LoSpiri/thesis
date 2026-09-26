"""E2.6: which layer *groups* tolerate spiking? Spike one group at a time
(sigma-delta, K=2) and measure the accuracy drop vs float. Informs the
hybrid "spike the MLP, keep attention float" design.

Includes a negated group "not_attn_head" = spike everything EXCEPT attention
and the output head (the proposed hybrid config).

Usage:
  python run_e2_groups.py --model sfm --subset 1319 --K 2 --workers 12 --out results/x.json
"""
import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch
import dataloader
from loader import build
from spike_inject import SigmaDeltaInjector
from parallel_eval import score_all

_ATTN = lambda n: ("query" in n or "key" in n or "value" in n
                   or "attn_qkv" in n or "att_out" in n)
_HEAD = lambda n: ("lm_head" in n or "output_layer" in n)

GROUPS = {
    "attention": _ATTN,
    "mlp":       lambda n: ("c_fc" in n or "mlp_out" in n or ".mlp." in n),
    "head":      _HEAD,
    "time":      lambda n: ("sigma_map" in n or "alpha_modulation" in n
                            or "time_token" in n),
    "not_attn_head": lambda n: not (_ATTN(n) or _HEAD(n)),
}


def acc_with_filter(model, all_ids, dataset, steps, batch, filt, K, workers):
    inj = SigmaDeltaInjector(model, mode="sigma_delta", K=K, layer_filter=filt)
    inj.attach()
    torch.manual_seed(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    n = len(all_ids)
    pairs = []
    for start in range(0, n, batch):
        idxs = list(range(start, min(start + batch, n)))
        lens = torch.tensor([len(all_ids[i]) for i in idxs], device=model.device)
        ml = int(lens.max())
        pad = torch.zeros(len(idxs), ml, dtype=torch.long, device=model.device)
        for j, i in enumerate(idxs):
            pad[j, :len(all_ids[i])] = all_ids[i].to(model.device)
        inj.reset()
        with torch.no_grad():
            samples, _ = model.generate_samples(
                num_samples=len(idxs), num_steps=steps,
                prefix_tokens=pad, prefix_lengths=lens)
        for j, i in enumerate(idxs):
            pl = int(lens[j])
            resp = model.tokenizer.decode(samples[j, pl:].cpu().tolist(),
                                          skip_special_tokens=True)
            pairs.append((resp, dataset[i]["response_ground_truth"]))
    scores = score_all(pairs, workers=workers)
    inj.detach()
    return sum(scores) / max(n, 1), inj.total()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit"], default="sfm")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--subset", type=int, default=1319)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--K", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    ds = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(ds))
    ids = [torch.tensor(ds[i]["input_ids"]) for i in range(n)]

    results = {"model": args.model, "subset": n, "K": args.K, "groups": {}}
    print(f"[{args.model}] subset={n} K={args.K}", flush=True)
    print(f"{'group':>14} {'acc':>8} {'spikes':>12}")
    for gname, filt in GROUPS.items():
        acc, sp = acc_with_filter(model, ids, ds, args.steps, args.batch, filt,
                                  args.K, args.workers)
        results["groups"][gname] = {"acc": acc, "spikes": sp}
        print(f"{gname:>14} {acc:8.4f} {sp:12,}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
