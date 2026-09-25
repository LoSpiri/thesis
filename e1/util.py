"""Device helpers for the S-FLM codebase.

MPS does not support float64, but the codebase keeps float64 buffers in the
adaptive noise schedule and float64 torchmetrics. These helpers cast
everything to float32 and move a loaded model to the best available device
(CUDA > MPS > CPU). The float32 cast is harmless on CUDA and required on MPS.
"""
import torch


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cast_float64(model):
    for p in model.parameters():
        if p.dtype == torch.float64:
            p.data = p.data.float()
    for b in model.buffers():
        if b.dtype == torch.float64:
            b.data = b.data.float()


def _cast_metrics_float32(model):
    m = getattr(model, "metrics", None)
    if m is None:
        return
    for attr in ("train_nlls", "train_aux", "valid_nlls", "valid_aux",
                 "gen_ppl", "sample_entropy"):
        obj = getattr(m, attr, None)
        if obj is not None and hasattr(obj, "set_dtype"):
            try:
                obj.set_dtype(torch.float32)
            except Exception:
                pass


def device_ready(model, device=None):
    device = device or auto_device()
    _cast_float64(model)
    _cast_metrics_float32(model)
    return model.to(device)
