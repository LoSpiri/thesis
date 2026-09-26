"""E2.1 offline: Level-2 identity + spikes-vs-error Pareto on smooth signals.

1. Confirm soft-reset LIF == sigma-delta modulator (independent formulations).
2. On a smooth (low-frequency) signal -- representative of flow activations --
   compare sigma-delta vs stateless rounding: at equal spike count, sigma-delta
   achieves lower reconstruction error, and its spike count scales with the
   signal's variation rather than its magnitude.
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np

from sigma_delta import (sigma_delta_reconstruct, stateless_reconstruct,
                         reconstruction_error, graded_spike_count,
                         unary_spike_count)


def literal_lif(u, tau):
    """Independent LIF(soft-reset) formulation to cross-check the identity."""
    S = u.shape[0]
    V = torch.zeros_like(u[0])
    y = torch.zeros_like(u[0])
    y_hist = torch.zeros_like(u)
    for k in range(S):
        V = V + (u[k] - y)          # integrate current u_k minus feedback y
        s = torch.round(V / tau)    # graded spike count
        y = y + s * tau
        V = V - s * tau             # soft reset
        y_hist[k] = y
    return y_hist


def smooth_signal(S, N, periods=(40.0, 13.0), noise=0.0):
    """Low-frequency deterministic signal (sum of sinusoids + optional noise)."""
    k = torch.arange(S).float()
    v = torch.zeros(S, N)
    for i, T in enumerate(periods):
        v += (0.5 / (i + 1)) * torch.sin(2 * np.pi * k[:, None] / T + i)
    if noise > 0:
        v += torch.randn(S, N) * noise
    return v


def main():
    torch.manual_seed(0)
    S, N = 64, 2000
    tau = 0.1

    # 1) identity check
    u = smooth_signal(S, N) + 0.01 * torch.randn(S, N)
    y1, _ = sigma_delta_reconstruct(u, tau=tau)
    y2 = literal_lif(u, tau)
    print("== identity: soft-reset LIF == sigma-delta ==")
    print(f"  max |diff| = {(y1 - y2).abs().max().item():.3e}")

    # 2) spikes vs error, sweep tau (finer tau = more spikes, lower error)
    u = smooth_signal(S, N)
    print("\n== smooth signal: error vs spikes (sigma-delta vs stateless) ==")
    print(f"{'tau':>8} {'sd_err':>10} {'sd_spikes':>10} {'st_err':>10} {'st_spikes':>10}")
    for tau in [0.2, 0.1, 0.05, 0.02, 0.01]:
        ys, qs = sigma_delta_reconstruct(u, tau=tau)
        yst = stateless_reconstruct(u, tau=tau)
        sd_err = reconstruction_error(ys, u).mean().item()
        st_err = reconstruction_error(yst, u).mean().item()
        sd_spikes = graded_spike_count(qs, tau)
        st_spikes = unary_spike_count(yst, tau)
        print(f"{tau:8.3f} {sd_err:10.5f} {sd_spikes:10d} {st_err:10.5f} {st_spikes:10d}")

    # 3) error at matched spike budget: interpolate implicitly by choosing
    #    a sigma-delta tau that uses the same spikes as a stateless tau.
    print("\n== at matched spike count (~6000), what error does each reach? ==")
    tau_st = 0.05
    yst = stateless_reconstruct(u, tau=tau_st)
    target = unary_spike_count(yst, tau_st)
    # binary search tau for sigma-delta to match target spikes
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        _, qs = sigma_delta_reconstruct(u, tau=mid)
        if graded_spike_count(qs, mid) > target:
            lo = mid
        else:
            hi = mid
    tau_sd = (lo + hi) / 2
    ys, qs = sigma_delta_reconstruct(u, tau=tau_sd)
    print(f"  stateless: tau={tau_st}, spikes={target}, error="
          f"{reconstruction_error(yst, u).mean().item():.5f}")
    print(f"  sigma-delta: tau={tau_sd:.4f}, spikes={graded_spike_count(qs, tau_sd)}, "
          f"error={reconstruction_error(ys, u).mean().item():.5f}")


if __name__ == "__main__":
    main()
