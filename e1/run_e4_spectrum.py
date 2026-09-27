"""E4.0 -- frequency diagnostic: does the sigma-delta LIF low-pass the flow?

Motivated by "Spiking Neural Networks Need High-Frequency Information"
(Fang et al.): spiking neurons suppress high-frequency components. This script
tests that claim on the S-FLM flow model, along two axes:

  temporal -- how activations evolve across sampler steps (the axis sigma-delta
              actually compresses);
  hidden   -- across the embedding dimension at a fixed step (the paper's
              feature-map analogue).

Three measurements per captured layer:
  (A) in-the-loop: float vs sigma_delta forward passes (with feedback);
  (B) offline: apply the sigma-delta / stateless operator to the float series
      (isolates the operator's transfer function, no feedback);
  (C) synthetic: feed unit sinusoids through the operators and report
      |Y(f)| / |X(f)| -- the clean frequency response (paper Fig. 6 analogue).

Writes JSON; prints a per-layer high-frequency energy table.

Usage:
  python run_e4_spectrum.py --model sfm --steps 8 --K 2 --length 128 \
      --subset 4 --out results/e4_spectrum_sfm.json
"""
import argparse
import json
import math
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch

import dataloader
from loader import build
from spike_inject import SigmaDeltaInjector
from sigma_delta import (sigma_delta_reconstruct, stateless_reconstruct,
                         hard_reset_reconstruct)


def pad_prefix_batch(id_lists, device):
    lengths = torch.tensor([len(ids) for ids in id_lists], device=device)
    max_len = int(lengths.max())
    padded = torch.zeros(len(id_lists), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(id_lists):
        padded[i, :len(ids)] = ids.to(device)
    return padded, lengths


def temporal_energy(A):
    """A: [S, ...] -> mean power spectrum over the step axis [S//2+1]."""
    Af = torch.fft.rfft(A.float(), dim=0)
    e = Af.abs().pow(2)
    dims = tuple(range(1, e.ndim))
    return e.mean(dim=dims) if dims else e


def hidden_energy(A):
    """A: [..., d] -> mean power spectrum over the hidden axis [d//2+1]."""
    Af = torch.fft.rfft(A.float(), dim=-1)
    e = Af.abs().pow(2)
    dims = tuple(range(Af.ndim - 1))
    return e.mean(dim=dims) if dims else e


def hf_fraction(e, lo=None):
    """Fraction of (non-DC) energy above the midpoint of the spectrum."""
    e = e[1:]
    if e.numel() <= 1:
        return 0.0
    half = lo if lo is not None else e.numel() // 2
    return float(e[half:].sum() / e.sum().clamp_min(1e-12))


def band_gain(e_num, e_den, band="low"):
    """Mean spectral gain in the low/high half of the shared band."""
    n = min(e_num.numel(), e_den.numel())
    e_num, e_den = e_num[:n][1:], e_den[:n][1:]
    half = e_num.numel() // 2
    sl = slice(0, half) if band == "low" else slice(half, None)
    r = torch.sqrt(e_num[sl].sum().clamp_min(1e-12)
                   / e_den[sl].sum().clamp_min(1e-12))
    return float(r)


def flatten(A):
    return A.reshape(A.shape[0], -1)


def capture_run(model, all_ids, steps, batch, mode, K, cap_filter, max_batch):
    inj = SigmaDeltaInjector(model, mode=mode, K=K, steps=steps,
                             capture=True, capture_filter=cap_filter,
                             capture_max_batch=max_batch).attach()
    torch.manual_seed(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    n = len(all_ids)
    inj.reset()
    with torch.no_grad():
        for start in range(0, n, batch):
            idxs = list(range(start, min(start + batch, n)))
            b_ids = [all_ids[i] for i in idxs]
            padded, lengths = pad_prefix_batch(b_ids, model.device)
            inj.reset()
            model.generate_samples(num_samples=len(idxs), num_steps=steps,
                                   prefix_tokens=padded, prefix_lengths=lengths)
    cap = {k: torch.stack([d["a"] for d in v], dim=0)
           for k, v in inj.capture.items()}
    inj.detach()
    return cap


def synthetic_response(K, S=64):
    """Characterise the reconstructors on a two-tone signal (low + high).

    Reports, per operator: the gain on the low and high tone (does the
    high-frequency *signal* survive?) and the high-frequency fraction of the
    quantization *error* spectrum (is the error noise-shaped?).

    Expected: all reconstructors track the signal (gains ~1), so no
    high-frequency signal is lost; sigma-delta pushes its error to high
    frequencies (err_hf ~1), stateless/hard-reset leave it white (err_hf ~0.5).
    """
    f_lo, f_hi = 3, 24
    steps = torch.arange(S, dtype=torch.float32)
    x = (torch.sin(2 * math.pi * f_lo * steps / S)
         + torch.sin(2 * math.pi * f_hi * steps / S)).unsqueeze(1)
    Xe = torch.fft.rfft(x, dim=0).abs()

    ops = {
        "sigma_delta": lambda z: sigma_delta_reconstruct(z.clone(), K=K)[0],
        "stateless": lambda z: stateless_reconstruct(z.clone(), K=K),
        "hard_reset": lambda z: hard_reset_reconstruct(z.clone(), K=K),
    }
    out = {}
    for name, fn in ops.items():
        y = fn(x)
        Ye = torch.fft.rfft(y, dim=0).abs()
        err = y - x
        Ee = torch.fft.rfft(err, dim=0).abs().pow(2)
        out[name] = {
            "low_gain": float((Ye[f_lo] / Xe[f_lo].clamp_min(1e-12)).mean()),
            "high_gain": float((Ye[f_hi] / Xe[f_hi].clamp_min(1e-12)).mean()),
            "err_hf_frac": hf_fraction(Ee.mean(dim=1)),
        }
    return {"f_lo": f_lo, "f_hi": f_hi, **out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sfm", "sfm-dit"], default="sfm")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--subset", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--K", type=float, default=2.0)
    p.add_argument("--blocks", type=str, default="0,-1",
                   help="block indices to capture ('-1' = last)")
    p.add_argument("--ops", type=str, default="query,c_fc,attn_qkv,mlp")
    p.add_argument("--capture-max-batch", type=int, default=1)
    p.add_argument("--synth-steps", type=int, default=64)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    model, tokenizer, cfg = build(args.model, steps=args.steps,
                                  length=args.length,
                                  temperature=args.temperature,
                                  eval_batch_size=args.batch)
    dataset = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = min(args.subset, len(dataset))
    all_ids = [torch.tensor(dataset[i]["input_ids"]) for i in range(n)]

    n_blocks = len(model.backbone.blocks)
    blocks = [b if b >= 0 else n_blocks + b for b in
              (int(x) for x in args.blocks.split(","))]
    ops = tuple(args.ops.split(","))

    def cap_filter(name):
        if not any(f"blocks.{b}." in name for b in blocks):
            return False
        return any(o in name for o in ops)

    results = {"model": args.model, "steps": args.steps, "subset": n,
               "length": args.length, "K": args.K, "blocks": blocks,
               "ops": list(ops), "layers": {}}

    t0 = time.time()
    float_cap = capture_run(model, all_ids, args.steps, args.batch, "float",
                            args.K, cap_filter, args.capture_max_batch)
    print(f"[{args.model}] float capture: {len(float_cap)} layers "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    sd_cap = capture_run(model, all_ids, args.steps, args.batch, "sigma_delta",
                         args.K, cap_filter, args.capture_max_batch)
    print(f"[{args.model}] sigma_delta capture: {len(sd_cap)} layers "
          f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'layer':>28} {'hf_hid f/sd':>14} {'hf_time f/sd':>16} "
          f"{'offline gain sd / hard':>26}")
    for name in sorted(float_cap):
        A = float_cap[name]
        if name not in sd_cap:
            continue
        Ad = sd_cap[name]
        if A.shape != Ad.shape:
            continue
        eA_h, eAd_h = hidden_energy(A), hidden_energy(Ad)
        eA_t, eAd_t = temporal_energy(A), temporal_energy(Ad)
        row = {
            "shape": list(A.shape),
            "inloop": {
                "float_hf_hidden": hf_fraction(eA_h),
                "sd_hf_hidden": hf_fraction(eAd_h),
                "ratio_hidden": (hf_fraction(eAd_h)
                                 / max(hf_fraction(eA_h), 1e-12)),
                "float_hf_time": hf_fraction(eA_t),
                "sd_hf_time": hf_fraction(eAd_t),
                "ratio_time": (hf_fraction(eAd_t)
                               / max(hf_fraction(eA_t), 1e-12)),
            },
        }
        # (B) offline operator response on the float series
        flat = flatten(A)
        y_sd = sigma_delta_reconstruct(flat.clone(), K=args.K)[0].reshape(A.shape)
        y_st = stateless_reconstruct(flat.clone(), K=args.K).reshape(A.shape)
        y_hd = hard_reset_reconstruct(flat.clone(), K=args.K).reshape(A.shape)
        e_sd_t, e_st_t, e_hd_t = (temporal_energy(y_sd), temporal_energy(y_st),
                                  temporal_energy(y_hd))
        row["offline"] = {
            "sd_hf_time": hf_fraction(e_sd_t),
            "stateless_hf_time": hf_fraction(e_st_t),
            "hard_hf_time": hf_fraction(e_hd_t),
            "float_hf_time": hf_fraction(eA_t),
            "sd_gain_low": band_gain(e_sd_t, eA_t, "low"),
            "sd_gain_high": band_gain(e_sd_t, eA_t, "high"),
            "hard_gain_low": band_gain(e_hd_t, eA_t, "low"),
            "hard_gain_high": band_gain(e_hd_t, eA_t, "high"),
        }
        results["layers"][name] = row
        print(f"{name:>28} "
              f"{row['inloop']['float_hf_hidden']:.3f}/"
              f"{row['inloop']['sd_hf_hidden']:.3f}"
              f"{'':>4}"
              f"{row['inloop']['float_hf_time']:.3f}/"
              f"{row['inloop']['sd_hf_time']:.3f}"
              f"{'':>6}"
              f"{row['offline']['sd_gain_low']:.2f}@lo "
              f"{row['offline']['sd_gain_high']:.2f}@hi | "
              f"hard {row['offline']['hard_gain_low']:.2f}@lo "
              f"{row['offline']['hard_gain_high']:.2f}@hi", flush=True)

    results["synthetic_response"] = synthetic_response(args.K, args.synth_steps)
    sr = results["synthetic_response"]
    print(f"\nsynthetic two-tone (S={args.synth_steps}, "
          f"f_lo={sr['f_lo']}, f_hi={sr['f_hi']}):", flush=True)
    for name in ("sigma_delta", "stateless", "hard_reset"):
        d = sr[name]
        print(f"  {name:>12}: low_gain={d['low_gain']:.3f} "
              f"high_gain={d['high_gain']:.3f} "
              f"err_hf_frac={d['err_hf_frac']:.3f}", flush=True)
    print("  (high_gain~1 = high-freq signal preserved; "
          "err_hf_frac~1 = error noise-shaped to high freq)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
