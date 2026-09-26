"""E3: Level-3 checks.

E3.1: the MDLM/masked sampler is a CTMC; verify its per-step token-jump
      (unmask) count follows the transition rate denoise_prob = (a_s-a_t)/(1-a_t).
E3.2: stochastic velocity (velocity=sample = a categorical "spike" per position)
      samples the correct marginal -- compare accuracy to deterministic exact.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch

import dataloader
import sandbox_gsm8k
from loader import build
from util import auto_device


def e3_1_ctmc(model, steps=32, length=64, batch=1):
    """Log masked count and unmask events per step; check against the rate."""
    model.sampler.init_state(model, num_samples=batch, num_steps=steps)
    state = model.sampler.init_state(model, num_samples=batch, num_steps=steps)
    mask = model.mask_index
    rows = []
    with torch.no_grad():
        for k in range(steps):
            before = int((state.xt == mask).sum())
            t = state.timesteps[state.step_idx] * state.ones
            s = state.timesteps[state.step_idx + 1] * state.ones
            _, alpha_t = model.noise(t)
            _, alpha_s = model.noise(s)
            denoise_prob = float(((alpha_s - alpha_t) / (1 - alpha_t + 1e-12)).mean())
            state = model.sampler.step(model, state)
            after = int((state.xt == mask).sum())
            unmasked = before - after
            rows.append({"step": k, "masked_before": before,
                         "denoise_prob": denoise_prob,
                         "expected": denoise_prob * before,
                         "unmasked": unmasked})
    return rows


def e3_2_velocity(model, all_ids, dataset, steps, batch):
    accs = {}
    for vel in ("exact", "sample"):
        model.sampler.velocity = vel
        n = len(all_ids)
        correct = total = 0
        for start in range(0, n, batch):
            idxs = list(range(start, min(start + batch, n)))
            padded, lengths = [], torch.tensor(
                [len(all_ids[i]) for i in idxs], device=model.device)
            ml = int(lengths.max())
            pad = torch.zeros(len(idxs), ml, dtype=torch.long, device=model.device)
            for j, i in enumerate(idxs):
                pad[j, :len(all_ids[i])] = all_ids[i].to(model.device)
            with torch.no_grad():
                samples, _ = model.generate_samples(
                    num_samples=len(idxs), num_steps=steps,
                    prefix_tokens=pad, prefix_lengths=lengths)
            for j, i in enumerate(idxs):
                pl = int(lengths[j])
                resp = model.tokenizer.decode(
                    samples[j, pl:].cpu().tolist(), skip_special_tokens=True)
                try:
                    ok = bool(sandbox_gsm8k.evaluate_samples(
                        resp, dataset[i]["response_ground_truth"], timeout_s=5.0))
                except Exception:
                    ok = False
                correct += int(ok); total += 1
        accs[vel] = correct / max(total, 1)
    return accs


def main():
    print("=== E3.1: MDLM sampler = CTMC (token-jump rate) ===", flush=True)
    model, tokenizer, cfg = build("mdlm", steps=32, length=64)
    rows = e3_1_ctmc(model, steps=32, length=64)
    print(f"{'step':>5} {'masked':>8} {'denoise_p':>10} {'expected':>9} {'unmasked':>9}")
    for r in rows[:8]:
        print(f"{r['step']:5d} {r['masked_before']:8d} {r['denoise_prob']:10.4f} "
              f"{r['expected']:9.1f} {r['unmasked']:9d}")

    print("\n=== E3.2: stochastic velocity (spike) vs exact ===", flush=True)
    model2, tokenizer2, cfg2 = build("sfm", steps=16, length=512,
                                     temperature=0.1)
    ds = dataloader.get_dataset(cfg2, tokenizer2, mode="valid")
    n = 128
    ids = [torch.tensor(ds[i]["input_ids"]) for i in range(n)]
    accs = e3_2_velocity(model2, ids, ds, steps=16, batch=8)
    print(f"  exact (deterministic) : {accs['exact']:.4f}")
    print(f"  sample (stochastic)   : {accs['sample']:.4f}")


if __name__ == "__main__":
    main()
