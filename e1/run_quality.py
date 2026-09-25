"""Full GSM8K accuracy for a single (model, NFE) config.

Replicates main.py's `_sample_gsm8k` + sandbox scoring without Fabric,
single-process on the auto-detected device (CUDA > MPS > CPU).

Usage:
  python run_quality.py --model sfm --steps 32 --out results/q_sfm_32.json
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
import sandbox_gsm8k
from loader import build
from util import auto_device


def pad_prefix_batch(id_lists, device):
    lengths = torch.tensor([len(ids) for ids in id_lists], device=device)
    max_len = int(lengths.max())
    padded = torch.zeros(len(id_lists), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(id_lists):
        padded[i, :len(ids)] = ids.to(device)
    return padded, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit", "mdlm", "duo", "flm"],
                   required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-examples", type=int, default=0)  # 0 = all
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    model, tokenizer, cfg = build(args.model, steps=args.steps, length=512,
                                  top_k_velocity=args.topk,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = len(dataset) if args.max_examples <= 0 else min(args.max_examples, len(dataset))

    device = auto_device()
    correct = 0
    total = 0
    t0 = time.time()
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    for start in range(0, n, args.batch):
        idxs = list(range(start, min(start + args.batch, n)))
        batch_ids = [all_ids[i] for i in idxs]
        padded, lengths = pad_prefix_batch(batch_ids, device)
        with torch.no_grad():
            samples, _ = model.generate_samples(
                num_samples=len(idxs), num_steps=args.steps,
                prefix_tokens=padded, prefix_lengths=lengths)
        for j, i in enumerate(idxs):
            pl = int(lengths[j])
            gen_ids = samples[j, pl:].cpu().tolist()
            resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gold = dataset[i]["response_ground_truth"]
            try:
                ok = bool(sandbox_gsm8k.evaluate_samples(resp, gold, timeout_s=5.0))
            except Exception:
                ok = False
            correct += int(ok)
            total += 1
        if (start // args.batch) % 5 == 0:
            print(f"  {start}/{n}  acc={correct/max(total,1):.3f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    acc = correct / max(total, 1)
    result = {"model": args.model, "steps": args.steps, "topk": args.topk,
              "temperature": args.temperature, "n": total, "correct": correct,
              "accuracy": acc, "elapsed_s": time.time() - t0}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.model}] steps={args.steps}: accuracy={acc:.4f} "
          f"({correct}/{total}) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
