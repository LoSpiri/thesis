"""E3.2: stochastic velocity (velocity=sample = a categorical spike per position)
vs deterministic exact velocity. Confirms the stochastic spiking update samples
the correct marginal (accuracy ~ exact).
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import torch
import dataloader, sandbox_gsm8k
from loader import build


def main():
    model, tokenizer, cfg = build("sfm", steps=8, length=512, temperature=0.1,
                                  eval_batch_size=8)
    ds = dataloader.get_dataset(cfg, tokenizer, mode="valid")
    n = 128
    ids = [torch.tensor(ds[i]["input_ids"]) for i in range(n)]
    device = model.device

    for vel in ("exact", "sample"):
        model.sampler.velocity = vel
        torch.manual_seed(0)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(0)
        correct = total = 0
        for start in range(0, n, 8):
            idxs = list(range(start, min(start + 8, n)))
            lens = torch.tensor([len(ids[i]) for i in idxs], device=device)
            ml = int(lens.max())
            pad = torch.zeros(len(idxs), ml, dtype=torch.long, device=device)
            for j, i in enumerate(idxs):
                pad[j, :len(ids[i])] = ids[i].to(device)
            with torch.no_grad():
                samples, _ = model.generate_samples(
                    num_samples=len(idxs), num_steps=8,
                    prefix_tokens=pad, prefix_lengths=lens)
            for j, i in enumerate(idxs):
                pl = int(lens[j])
                resp = tokenizer.decode(samples[j, pl:].cpu().tolist(),
                                        skip_special_tokens=True)
                try:
                    ok = bool(sandbox_gsm8k.evaluate_samples(
                        resp, ds[i]["response_ground_truth"], timeout_s=5.0))
                except Exception:
                    ok = False
                correct += int(ok); total += 1
        print(f"velocity={vel}: acc={correct/max(total,1):.4f} ({correct}/{total})",
              flush=True)


if __name__ == "__main__":
    main()
