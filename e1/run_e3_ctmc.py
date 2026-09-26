"""E3.1: MDLM sampler is a CTMC; verify token-jump count matches the rate."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch
from loader import build


def main():
    model, tokenizer, cfg = build("mdlm", steps=32, length=64)
    mask = model.mask_index
    state = model.sampler.init_state(model, num_samples=1, num_steps=32)
    total_expected = total_actual = 0
    print(f"{'step':>4} {'masked':>7} {'rate':>7} {'expected':>8} {'unmasked':>8}")
    with torch.no_grad():
        for k in range(32):
            before = int((state.xt == mask).sum())
            t = state.timesteps[state.step_idx] * state.ones
            s = state.timesteps[state.step_idx + 1] * state.ones
            _, at = model.noise(t)
            _, as_ = model.noise(s)
            rate = float(((as_ - at) / (1 - at + 1e-12)).mean())
            state = model.sampler.step(model, state)
            after = int((state.xt == mask).sum())
            unmasked = before - after
            total_expected += rate * before
            total_actual += unmasked
            print(f"{k:4d} {before:7d} {rate:7.4f} {rate*before:8.1f} {unmasked:8d}")
    print(f"\n  total expected jumps = {total_expected:.1f} | actual = {total_actual}")


if __name__ == "__main__":
    main()
