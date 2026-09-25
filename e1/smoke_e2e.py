"""Minimal end-to-end smoke test: load S-FLM checkpoint, sample on MPS.

Mirrors scripts/sample/tinygsm/sfm_sphere_arch_truncated_adaptive_no_renorm.sh
but bypasses the full GSM8K harness. Confirms the MPS patch works through the
full sampler loop.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
import torch
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("cwd", os.getcwd)
OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

import dataloader
import algo
from util import device_ready


def main():
    ckpt = "checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt"
    hydra.initialize(config_path="../configs", version_base=None)
    overrides = [
        "mode=gsm8k_eval",
        f"eval.checkpoint_path={ckpt}",
        "eval.strict_loading=false",
        "data=gsm8k-test",
        "data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M",
        "model=small-sphere-arch",
        "model.length=64",
        "model.normalize_input_embed=False",
        "algo=sfm",
        "algo.renormalize_weights=True",
        "algo.invert_time_convention=false",
        "algo.slerp_precision=float32",
        "noise=log-linear-adaptive",
        "noise.alpha_max=0.121",
        "noise.adaptive_refit_every=50",
        "noise.adaptive_buffer_size=25600",
        "noise.adaptive_ema=0.9",
        "noise.adaptive_uniform_mix=1e-3",
        "sampler=sfm",
        "sampler.noise_removal=greedy",
        "sampler.velocity=exact",
        "sampler.top_k_velocity=-1",
        "sampler.use_float64=false",
        "sampler.steps=8",
        "loader.eval_batch_size=2",
        "loader.num_workers=0",
        "trainer.num_nodes=1",
        "trainer.devices=1",
    ]
    cfg = hydra.compose(config_name="config", overrides=overrides)

    print("loading tokenizer...")
    tokenizer = dataloader.get_tokenizer(cfg)
    print("loading checkpoint...")
    model = algo.SFM.load_from_checkpoint(
        cfg.eval.checkpoint_path, tokenizer=tokenizer, config=cfg, strict=False
    )
    model = device_ready(model)
    model.eval()
    model._eval_mode()
    model.ema = None

    print("sampling (2 samples, 8 steps)...")
    with torch.no_grad():
        samples, metadata = model.generate_samples(num_samples=2, num_steps=8)
    print("samples shape:", tuple(samples.shape), "dtype:", samples.dtype)
    print("metadata:", {k: v for k, v in metadata.items() if k != "tokens"})
    text = tokenizer.batch_decode(samples.cpu())
    print("decoded sample 0 (first 80 chars):", text[0][:80])
    print("SMOKE E2E PASSED")


if __name__ == "__main__":
    main()
