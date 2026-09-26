"""Sigma-delta (delta-sigma) coding across sampler steps.

Level-2 primitives. A soft-reset LIF neuron IS a first-order sigma-delta
modulator: it integrates the tracking error, emits quantized *changes*, and
carries the sub-threshold residual forward (soft reset). For the
slowly-varying activations of a flow sampler, the number of emitted spikes
scales with the signal's *variation*, not its magnitude -- so a fine
quantization step costs few spikes, and accuracy is preserved.

    m_k = m_{k-1} + (v_k - y_{k-1})   # integrate tracking error
    q_k = round(m_k / tau) * tau      # graded spike (quantized change)
    y_k = y_{k-1} + q_k               # running reconstruction
    m_k = m_k - q_k                   # soft reset (residual stays)
"""
import torch


def _threshold(v0, K):
    tau = v0.abs().mean() / K
    return max(float(tau), 1e-12)


# --------------------------------------------------------------------------
# Reconstructions
# --------------------------------------------------------------------------
def sigma_delta_reconstruct(acts, tau=None, K=1.0):
    """Soft-reset LIF = sigma-delta. Returns (y_hist, q_hist).

    y_hist: [S, N] running reconstruction.
    q_hist: [S, N] graded emission at each step (value units, multiples of tau).
    """
    S, N = acts.shape
    if tau is None:
        tau = _threshold(acts[0], K)
    m = torch.zeros(N, dtype=acts.dtype)
    y = torch.zeros(N, dtype=acts.dtype)
    y_hist = torch.zeros_like(acts)
    q_hist = torch.zeros_like(acts)
    for k in range(S):
        m = m + (acts[k] - y)
        q = torch.round(m / tau) * tau
        y = y + q
        m = m - q
        y_hist[k] = y
        q_hist[k] = q
    return y_hist, q_hist


def hard_reset_reconstruct(acts, tau=None, K=1.0):
    """Hard-reset LIF: discards the sub-threshold residual after each emit."""
    S, N = acts.shape
    if tau is None:
        tau = _threshold(acts[0], K)
    V = torch.zeros(N, dtype=acts.dtype)
    y = torch.zeros(N, dtype=acts.dtype)
    y_hist = torch.zeros_like(acts)
    for k in range(S):
        V = V + (acts[k] - y)
        q = torch.round(V / tau) * tau
        y = y + q
        V = torch.zeros(N, dtype=acts.dtype)
        y_hist[k] = y
    return y_hist


def stateless_reconstruct(acts, tau=None, K=1.0):
    """Per-step rounding, no feedback (flat-error baseline)."""
    if tau is None:
        tau = _threshold(acts[0], K)
    return torch.round(acts / tau) * tau


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def reconstruction_error(y_hist, acts):
    """Per-step mean absolute error ||y_k - v_k|| over neurons. [S]"""
    return (y_hist - acts).abs().mean(dim=1)


def graded_spike_count(q_hist, tau):
    """Total graded spikes (sum of |q|/tau over steps and neurons)."""
    return int((q_hist.abs() / tau).sum().item())


def unary_spike_count(y_hist, tau):
    """Spike count if the quantized value is re-emitted in unary each step."""
    return int((y_hist.abs() / tau).sum().item())


# --------------------------------------------------------------------------
# E1 firing-rate interface
# --------------------------------------------------------------------------
def sigma_delta(acts, K=1.0):
    """acts: [S, N]. Returns (firing_rate, per_step, emitted)."""
    y_hist, q_hist = sigma_delta_reconstruct(acts, K=K)
    emitted = (q_hist != 0)
    firing_rate = emitted.float().mean().item()
    per_step = emitted.float().mean(dim=1)
    return firing_rate, per_step, emitted


def raw_delta_stats(acts):
    """Mean absolute inter-step change."""
    d = (acts[1:] - acts[:-1]).abs()
    return d.mean().item()
