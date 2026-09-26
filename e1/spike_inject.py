"""In-the-loop sigma-delta injection: replace each nn.Linear input with its
spiking reconstruction, with a persistent per-layer membrane carried across
sampler steps. Training-free; simulates a spiking forward pass.

Modes:
  float       -- identity (no-op)
  sigma_delta -- soft-reset LIF (error feedback across steps)
  stateless   -- per-step rounding (no feedback; the naive-quantization baseline)

Adaptive threshold (optional), selected by `tau_mode`:
  tau_mode="fixed"     -> tau = tau_base (default). If `adaptive_a != 1` this
                          becomes the exponential schedule below.
  tau_mode="exp"       -> tau(t) = tau_base * adaptive_a^(1 - 2t/(S-1))
                          (a=1 == fixed; a>1 coarse early, a<1 fine early)
  tau_mode="bucket"    -> tau(t) = tau_base * (1 + bucket_b * sin(pi t/(S-1)))
                          (fine at t=0 and t=S-1, coarser in the middle;
                           bucket_b=0 == fixed)
  tau_mode="magnitude" -> tau(t) = mean(|a_t|) / K at each step (adapts to the
                          current activation magnitude instead of the frozen
                          first-step scale)

Spike counts are accumulated on-device (no per-hook sync) and materialized
only by calling .total().
"""
import math

import torch


class SigmaDeltaInjector:
    def __init__(self, model, mode="sigma_delta", K=1.0, layer_filter=None,
                 steps=None, adaptive_a=1.0, tau_mode="exp", bucket_b=0.0):
        assert mode in ("float", "sigma_delta", "stateless")
        self.model = model
        self.mode = mode
        self.K = K
        self.layer_filter = layer_filter  # callable(name)->bool, or None = all
        self.steps = steps
        self.adaptive_a = adaptive_a
        self.tau_mode = tau_mode
        self.bucket_b = bucket_b
        self.state = {}
        self._spike_acc = None
        self.hooks = []

    def reset(self):
        self.state = {}
        self._spike_acc = None

    def total(self):
        if self._spike_acc is None:
            return 0
        return int(self._spike_acc.item())

    def _tau(self, a, tau_base, step):
        if self.tau_mode == "magnitude":
            return max(float(a.detach().abs().mean().item()) / self.K, 1e-12)
        if self.tau_mode == "bucket" and self.steps and self.steps > 1:
            return tau_base * (1.0 + self.bucket_b
                               * math.sin(math.pi * step / (self.steps - 1)))
        # exp (default / back-compat); fires only when adaptive_a != 1
        if (self.adaptive_a is None or self.adaptive_a == 1.0
                or self.steps is None or self.steps <= 1):
            return tau_base
        return tau_base * (self.adaptive_a ** (1.0 - 2.0 * step / (self.steps - 1)))

    def _hook(self, name):
        def f(module, args):
            a = args[0]
            if a is None or self.mode == "float":
                return None
            if self.layer_filter is not None and not self.layer_filter(name):
                return None
            st = self.state.get(name)
            if st is None:
                tau_base = max(float(a.detach().abs().mean().item()) / self.K, 1e-12)
                st = {"m": torch.zeros_like(a), "y": torch.zeros_like(a),
                      "tau_base": tau_base, "step": 0}
                self.state[name] = st
            tau = self._tau(a, st["tau_base"], st["step"])
            st["step"] += 1
            if self.mode == "sigma_delta":
                st["m"] = st["m"] + (a - st["y"])
                q = torch.round(st["m"] / tau) * tau
                spikes = (q.abs() / tau).sum()
                st["y"] = st["y"] + q
                st["m"] = st["m"] - q
            else:  # stateless
                y = torch.round(a / tau) * tau
                spikes = (y.abs() / tau).sum()
                st["y"] = y
            if self._spike_acc is None:
                self._spike_acc = spikes.detach()
            else:
                self._spike_acc = self._spike_acc + spikes.detach()
            return (st["y"],)
        return f

    def attach(self):
        for name, module in self.model.backbone.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.hooks.append(
                    module.register_forward_pre_hook(self._hook(name)))
        return self

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
