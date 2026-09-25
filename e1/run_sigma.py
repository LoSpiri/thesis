"""Full-scope sigma-delta analysis for a single (model, NFE) config.

Captures the input activation of every linear layer at each sampler step,
then reports: sigma-delta firing rate vs K, raw activation-delta magnitude,
per-step firing trajectory, and a total-spikes-per-token energy proxy.

Usage:
  python run_sigma.py --model sfm --steps 8 [--topk 1] [--out results/x.json]
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


def capture(model, batch, steps):
    record, hooks = register_hooks(model)
    with torch.no_grad():
        state = model.sampler.init_state(model, num_samples=batch, num_steps=steps)
        while not state.done:
            state = model.sampler.step(model, state)
    for h in hooks:
        h.remove()
    return record


def analyze(record, K_values=(1, 2, 4, 8), n_tokens=64):
    layer_info = []
    # accumulate per-K totals and per-step (global, weighted by neurons)
    totals = {K: 0.0 for K in K_values}      # total firing (neuron,step) counts
    total_neurons_steps = 0                  # sum over layers of N*(S-1)
    per_step_num = None                      # [S-1] firing neurons (global)
    per_step_den = None                      # [S-1] total neurons (global)
    raw_delta_num = 0.0
    raw_delta_den = 0.0

    for name, acts_list in record.items():
        S = len(acts_list)
        if S < 2:
            continue
        shapes = {tuple(a.shape) for a in acts_list}
        if len(shapes) > 1:
            continue
        acts = torch.stack(acts_list, dim=0).reshape(S, -1)  # [S, N]
        N = acts.shape[1]
        # raw delta
        d = (acts[1:] - acts[:-1]).abs()
        raw_delta_num += d.sum().item()
        raw_delta_den += d.numel()

        layer_row = {"layer": name, "n_steps": S, "n_neurons": N,
                     "firing_rate": {}}
        for K in K_values:
            fr, per_step, emitted = sigma_delta(acts, K=K)
            totals[K] += emitted.sum().item()
            layer_row["firing_rate"][str(K)] = fr
            if K == K_values[0]:
                if per_step_num is None:
                    per_step_num = emitted.float().sum(dim=1)
                    per_step_den = torch.full_like(per_step_num, float(N))
                else:
                    per_step_num = per_step_num + emitted.float().sum(dim=1)
                    per_step_den = per_step_den + float(N)
        layer_info.append(layer_row)

    n_trans = S - 1 if S >= 1 else 0
    total_neurons_steps = sum(r["n_neurons"] * (r["n_steps"] - 1) for r in layer_info)
    result = {
        "overall_firing_rate": {str(K): totals[K] / max(total_neurons_steps, 1)
                                for K in K_values},
        "overall_raw_delta": raw_delta_num / max(raw_delta_den, 1),
        "per_step_firing_rate": (
            (per_step_num / per_step_den.clamp(min=1)).tolist()
            if per_step_num is not None else []),
        "total_spikes_per_token": sum(totals.values()) / max(n_tokens, 1),
        "layers": layer_info,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit", "mdlm", "duo", "flm"],
                   required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model, tokenizer, cfg = build(args.model, steps=args.steps, length=args.length,
                                  top_k_velocity=args.topk,
                                  temperature=args.temperature)
    print(f"[{args.model}] steps={args.steps} topk={args.topk} "
          f"temp={args.temperature}: sampling...", flush=True)
    record = capture(model, args.batch, args.steps)
    result = analyze(record, n_tokens=args.length)
    result.update({"model": args.model, "steps": args.steps,
                   "length": args.length, "batch": args.batch,
                   "topk": args.topk, "temperature": args.temperature,
                   "n_layers": len(record)})
    print(f"  overall firing rate (K=1): "
          f"{result['overall_firing_rate'].get('1'):.4f} | "
          f"raw delta: {result['overall_raw_delta']:.5f} | "
          f"spikes/token: {result['total_spikes_per_token']:.1f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
