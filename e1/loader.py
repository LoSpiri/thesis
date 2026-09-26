"""Model recipes + loading helpers for E1 experiments."""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra

OmegaConf.register_new_resolver("cwd", os.getcwd)
OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

# torch.compile (inductor) is not supported / very slow on MPS.
torch.compile = lambda f, *a, **k: f

import dataloader
import algo
from util import device_ready, auto_device

CKPT = {
    "sfm": "checkpoints/tinygsm/sfm/sphere_arch_truncated_adaptive_no_renorm.ckpt",
    "sfm-dit": "checkpoints/tinygsm/sfm/sphere_dit_truncated_adaptive_no_renorm.ckpt",
    "mdlm": "checkpoints/tinygsm/mdlm.ckpt",
    "duo": "checkpoints/tinygsm/duo.ckpt",
    "flm": "checkpoints/tinygsm/flm/default.ckpt",
}

_BASE = [
    "mode=gsm8k_eval",
    "data=gsm8k-test",
    "data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M",
    "data.data_path=data/gsm8k_test.json",
    "data.cache_dir=data_cache",
    "loader.num_workers=0",
    "trainer.num_nodes=1",
    "trainer.devices=1",
]

_SFM_SAMPLER = [
    "sampler=sfm",
    "sampler.noise_removal=greedy",
    "sampler.velocity=exact",
    "sampler.use_float64=false",
]

_RECIPES = {
    "sfm": [
        "model=small-sphere-arch",
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
    ] + _SFM_SAMPLER,
    "sfm-dit": [
        "model=small-sphere-dit",
        "model.init=ngpt",
        "algo=sfm",
        "algo.renormalize_weights=False",
        "algo.invert_time_convention=false",
        "algo.slerp_precision=float32",
        "noise=log-linear-adaptive",
        "noise.alpha_max=0.121",
        "noise.adaptive_refit_every=50",
        "noise.adaptive_buffer_size=25600",
        "noise.adaptive_ema=0.9",
        "noise.adaptive_uniform_mix=1e-3",
    ] + _SFM_SAMPLER,
    "mdlm": [
        "model=small",
        "algo=mdlm",
        "sampler=ancestral",
        "sampler.use_float64=false",
        "sampler.noise_removal=ancestral",
    ],
    "duo": [
        "model=small",
        "algo=duo-base",
        "sampler=ancestral",
        "sampler.use_float64=false",
        "sampler.noise_removal=ancestral",
    ],
    "flm": [
        "model=small-flm",
        "noise=log-linear",
        "algo=flm",
        "sampler=flm_euler",
        "sampler.use_float64=false",
    ],
}

_MODEL_CLASS = {
    "sfm": algo.SFM,
    "sfm-dit": algo.SFM,
    "mdlm": algo.MDLM,
    "duo": algo.DUO_BASE,
    "flm": algo.FLM,
}


def build(name, *, steps, length, device=None, top_k_velocity=None,
          eval_batch_size=2, temperature=None):
    """Load a TinyGSM checkpoint onto the best device (CUDA > MPS > CPU)."""
    device = device or auto_device()
    ckpt = CKPT[name]
    overrides = _BASE + _RECIPES[name] + [
        f"eval.checkpoint_path={ckpt}",
        "eval.strict_loading=false",
        f"model.length={length}",
        f"sampler.steps={steps}",
        f"loader.eval_batch_size={eval_batch_size}",
    ]
    if top_k_velocity is not None and name in ("sfm", "sfm-dit"):
        overrides.append(f"sampler.top_k_velocity={top_k_velocity}")
    if temperature is not None:
        overrides.append(f"sampler.temperature={temperature}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="../configs", version_base=None)
    cfg = hydra.compose(config_name="config", overrides=overrides)
    tokenizer = dataloader.get_tokenizer(cfg)
    cls = _MODEL_CLASS[name]
    model = cls.load_from_checkpoint(cfg.eval.checkpoint_path,
                                     tokenizer=tokenizer, config=cfg,
                                     strict=False)
    model = device_ready(model, device=device)
    model.eval()
    model._eval_mode()
    model.ema = None
    return model, tokenizer, cfg
