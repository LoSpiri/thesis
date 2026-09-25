"""Sigma-delta (delta-sigma) coding across sampler steps.

Approach 3: "spike time = sampler time". Keep a per-neuron membrane state
across sampler steps, transmit only the quantized *change* between steps,
and carry the quantization residual forward so error does not accumulate.

    m_k = m_{k-1} + (v_k - y_{k-1})
    q_k = round(m_k / tau) * tau
    y_k = y_{k-1} + q_k
    m_k = m_k - q_k

`tau` is the quantization step; following N-MDLM's adaptive step we set
tau = mean(|v_0|) / K over neurons at the first step.
"""
import torch


def _threshold(v0, K):
    tau = v0.abs().mean() / K
    return max(float(tau), 1e-12)


def sigma_delta(acts, K=1.0):
    """acts: [S, N] (steps x neurons). Returns (firing_rate, per_step, emitted)."""
    S, N = acts.shape
    S = S - 1  # number of transitions
    tau = _threshold(acts[0], K)
    m = torch.zeros(N, dtype=acts.dtype)
    y = torch.zeros(N, dtype=acts.dtype)
    emitted = torch.zeros(S, N, dtype=torch.bool)
    y = y + torch.round(acts[0] / tau) * tau
    for k in range(1, S + 1):
        m = m + (acts[k] - y)
        q = torch.round(m / tau) * tau
        emitted[k - 1] = (q != 0)
        y = y + q
        m = m - q
    firing_rate = emitted.float().mean().item()
    per_step = emitted.float().mean(dim=1)
    return firing_rate, per_step, emitted


def raw_delta_stats(acts):
    """Mean absolute inter-step change, and fraction below a threshold."""
    d = (acts[1:] - acts[:-1]).abs()
    mean_delta = d.mean().item()
    return mean_delta
