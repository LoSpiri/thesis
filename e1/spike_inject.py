"""In-the-loop sigma-delta injection: replace each nn.Linear input with its
spiking reconstruction, with a persistent per-layer membrane carried across
sampler steps. Training-free; simulates a spiking forward pass.

Modes:
  float       -- identity (no-op); still captures activations if capture=True
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

Membrane shortcut (`bypass`, E4.1): a continuous residual bypass that leaks a
fraction lambda of the sub-threshold error through un-quantized, mirroring the
paper's "membrane shortcut" (connect the membrane, not only the spikes):

    y_eff = y + lambda * (a - y)

lambda=0 is the pure spiking reconstruction; lambda=1 is the float identity.
This is the analog shortcut path; only the quantized part `q` is counted as
spikes (the bypass carries no spike events -- that is the point).

Frequency-selective bypass (`freq_alpha`, `freq_gamma`, E4.4): sigma-delta
preserves the signal but shapes its quantization error to HIGH frequencies
(y_k = a_k + (e_k - e_{k-1}), so NTF = 1 - z^-1). A causal EMA splits the
spike reconstruction into a low band and a high band and recombines them with
an adjustable high-frequency gain gamma:

    y_lp = alpha * y_lp + (1 - alpha) * y       # causal low-pass (EMA)
    y_hp = y - y_lp                             # shaped-noise band
    y_freq = y_lp + gamma * y_hp
    y_eff  = y_freq + bypass * (a - y_freq)     # then the membrane shortcut

gamma=1 (or alpha=0) is an exact validation (no change). gamma=0 is a pure
low-pass that rejects the shaped high-frequency noise (the "decimation filter"
direction); gamma>1 amplifies high frequencies (the paper's "restore
high-frequency" direction). Swept by run_e4_freqbypass.py.

Activation capture (`capture=True`, E4.0): records, for every layer selected by
`capture_filter`, the float input `a` and the injected output `y` at each
sampler step (first `capture_max_batch` batch elements, moved to CPU). Used by
run_e4_spectrum.py to measure the operator's frequency response in the loop.

Spike counts are accumulated on-device (no per-hook sync) and materialized
only by calling .total().
"""
import math

import torch


class SigmaDeltaInjector:
    def __init__(self, model, mode="sigma_delta", K=1.0, layer_filter=None,
                 steps=None, adaptive_a=1.0, tau_mode="exp", bucket_b=0.0,
                 bypass=0.0, freq_alpha=0.0, freq_gamma=1.0,
                 capture=False, capture_filter=None, capture_max_batch=1):
        assert mode in ("float", "sigma_delta", "stateless")
        self.model = model
        self.mode = mode
        self.K = K
        self.layer_filter = layer_filter  # callable(name)->bool, or None = all
        self.steps = steps
        self.adaptive_a = adaptive_a
        self.tau_mode = tau_mode
        self.bucket_b = bucket_b
        self.bypass = bypass
        self.freq_alpha = freq_alpha
        self.freq_gamma = freq_gamma
        self.capture_filter = capture_filter
        self.capture_max_batch = capture_max_batch
        self.capture = {} if capture else None
        self.state = {}
        self._spike_acc = None
        self.hooks = []

    def reset(self):
        self.state = {}
        self._spike_acc = None
        if self.capture is not None:
            self.capture = {}

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

    def _cap(self, t):
        t = t.detach().float().cpu()
        mb = self.capture_max_batch
        if mb is not None and t.ndim >= 1 and t.shape[0] > mb:
            t = t[:mb]
        return t

    def _record(self, name, step, a, y):
        if self.capture is None:
            return
        if self.capture_filter is not None and not self.capture_filter(name):
            return
        self.capture.setdefault(name, []).append(
            {"step": step, "a": self._cap(a), "y": self._cap(y)})

    def _hook(self, name):
        def f(module, args):
            a = args[0]
            if a is None:
                return None
            if self.layer_filter is not None and not self.layer_filter(name):
                return None

            st = self.state.get(name)
            if st is None:
                st = {"step": 0}
                self.state[name] = st
            step_now = st["step"]
            st["step"] = step_now + 1

            if self.mode == "float":
                self._record(name, step_now, a, a)
                return None

            if "m" not in st:
                tau_base = max(float(a.detach().abs().mean().item()) / self.K,
                               1e-12)
                st.update({"m": torch.zeros_like(a), "y": torch.zeros_like(a),
                           "tau_base": tau_base})
            tau = self._tau(a, st["tau_base"], step_now)
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

            y_spike = st["y"]
            if self.freq_alpha and self.freq_alpha > 0.0:
                # causal EMA low-pass; init to y_spike so step 0 is transient-free
                if "y_lp" not in st:
                    st["y_lp"] = y_spike.clone()
                else:
                    st["y_lp"] = (self.freq_alpha * st["y_lp"]
                                  + (1.0 - self.freq_alpha) * y_spike)
                y_hp = y_spike - st["y_lp"]
                y_out = st["y_lp"] + self.freq_gamma * y_hp
            else:
                y_out = y_spike
            if self.bypass:
                y_out = y_out + self.bypass * (a - y_out)
            self._record(name, step_now, a, y_out)
            return (y_out,)
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
