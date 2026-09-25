"""E1: sigma-delta sparsity of inter-step activation changes.

Captures the input activation of every linear layer in the backbone at
each sampler step, then simulates sigma-delta coding across steps and
reports firing rates vs. quantization resolution K.

Usage:
  python run_e1.py --model sfm --steps 8 --length 64 --batch 2
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")
import torch

from loader import build
from sigma_delta import sigma_delta


def register_hooks(model):
    record = {}

    def make_hook(name):
        def hook(module, args):
            x = args[0]
            if x is None:
                return
            record.setdefault(name, []).append(x.detach().cpu().float())
        return hook

    hooks = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    return record, hooks


def sample_and_capture(model, batch, steps):
    record, hooks = register_hooks(model)
    with torch.no_grad():
        state = model.sampler.init_state(model, num_samples=batch, num_steps=steps)
        while not state.done:
            state = model.sampler.step(model, state)
    for h in hooks:
        h.remove()
    return record


def layer_stats(record, K):
    rows = []
    total_fired = 0
    total_neurons = 0
    for name, acts_list in record.items():
        n_steps = len(acts_list)
        if n_steps < 2:
            continue
        shapes = {tuple(a.shape) for a in acts_list}
        if len(shapes) > 1:
            continue
        acts = torch.stack(acts_list, dim=0).reshape(n_steps, -1)
        fr, per_step, emitted = sigma_delta(acts, K=K)
        N = acts.shape[1]
        total_fired += emitted.sum().item()
        total_neurons += emitted.numel()
        rows.append({"layer": name, "firing_rate": fr,
                     "n_steps": n_steps, "n_neurons": N})
    overall = total_fired / max(total_neurons, 1)
    return rows, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "mdlm", "duo", "flm"], required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model, tokenizer, cfg = build(args.model, steps=args.steps, length=args.length)
    print(f"[{args.model}] sampling {args.batch} seqs x {args.steps} steps "
          f"(len {args.length})...", flush=True)
    record = sample_and_capture(model, args.batch, args.steps)
    print(f"captured {len(record)} linear layers", flush=True)

    result = {"model": args.model, "steps": args.steps, "length": args.length,
              "batch": args.batch, "overall_firing_rate": {}, "layers": {}}
    for K in (1, 2, 4, 8):
        rows, overall = layer_stats(record, K)
        result["overall_firing_rate"][str(K)] = overall
        result["layers"][str(K)] = rows
        print(f"  K={K}: overall sigma-delta firing rate = {overall:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
