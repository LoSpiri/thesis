# E4 Report — High-frequency information in spiking flow matching

_Motivated by "Spiking Neural Networks Need High-Frequency Information"
(Fang et al.). Training-free experiments on the S-FLM flow models + sigma-delta
spiking forward pass (`e1/spike_inject.py`)._

## The hypothesis (from the paper)

Spiking neurons suppress high-frequency components; restoring high-frequency
information recovers accuracy. If true for our sigma-delta LIF, then adding
high-pass structure (bypass / Max-Pool-like token mixer / argmax decoding)
should help under spiking.

## E4.0 — frequency diagnostic (DONE, local)

`run_e4_spectrum.py` measures the operator's effect along the **temporal**
(across sampler steps) and **hidden** axes, and characterises the
reconstructors on a two-tone signal.

**Result: the paper's "low-pass" claim does _not_ transfer to sigma-delta.**

- Synthetic two-tone (K=2, f_lo=3, f_hi=24, S=64):

  | operator | low-tone gain | high-tone gain | error HF fraction |
  |---|---|---|---|
  | sigma_delta | 1.000 | 1.000 | **0.784** |
  | stateless | 1.000 | 1.000 | 0.519 |
  | hard_reset | 1.000 | 1.000 | 0.373 |

  All reconstructors preserve the high-frequency **signal** (gain ≈ 1). The
  sigma-delta instead pushes its quantization **error** to high frequencies
  (noise shaping) -- the opposite of losing high-frequency signal.

- On real activations (offline operator gain vs float): the high-band gain is
  **1.0–8×** for sigma-delta (extra HF from shaped noise), while the
  hidden-axis spectrum is essentially unchanged (float ≈ sd).

**Implication.** The lever is *not* "restore lost high-frequency signal"; if
anything sigma-delta *injects* high-frequency quantization noise. That reframes
E4.1–E4.3 as tests of whether (a) a continuous bypass reduces the noise
(membrane shortcut), (b) argmax decoding is more robust to the noise, and
(c) adding more high-frequency structure helps or hurts.

## E4.1 — membrane shortcut (continuous bypass)

`y_eff = y + lambda*(a - y)`; lambda=0 is the sigma-delta baseline, lambda=1 is
float. `run_e4_membrane.py` sweeps lambda per K (both are validation rows).
Measures the accuracy/spike Pareto of leaking the sub-threshold residual
through a continuous shortcut.

## E4.2 — WTA / max decoding

`sampler.top_k_velocity = k`; k=1 is a pure argmax (Max) velocity, k=-1 is the
exact baseline. `run_e4_wta.py` sweeps k under float and sigma-delta.

## E4.3 — max-style early token mixer

Adds `x <- x + beta*highpass(x)` to the first N blocks (`center` / `diff` modes;
beta=0 is the validation baseline). `run_e4_mixer.py` sweeps beta under spiking.

## E4.4 — frequency-selective bypass (the result-driven lever)

Since the diagnostic shows sigma-delta is signal-transparent with its error
shaped to high frequencies (`y = a + (1 - z^-1) e`), the textbook counterpart is
a *decimation low-pass*, not a high-pass restoration. We split the spike
reconstruction into a causal low band and a high band and recombine:

    y_lp   = alpha * y_lp + (1 - alpha) * y     # EMA low-pass
    y_freq = y_lp + gamma * (y - y_lp)

- `gamma = 0` -> reject the shaped high-frequency noise (noise-rejection);
- `gamma = 1` (or `alpha = 0`) -> exact validation (no change);
- `gamma > 1` -> amplify high frequencies (the paper's direction).

`run_e4_freqbypass.py` sweeps (alpha, gamma) at K=2 (both models) and K=1 (sfm).
Local smoke (sfm, subset 16, K=2): alpha=0.8 gives spikes 1.71B at gamma=0,
2.75B at gamma=1, and 14.5B at gamma=2 — the high-frequency gain directly
controls the membrane activity, as expected.

## How to run

```bash
# local (Apple MPS), small subsets
SUBSET=64 bash e1/run_e4_local.sh          # or MODEL=sfm-dit

# RunPod, full 1319 (container disk; results written straight to /workspace/results)
bash runpod_e4.sh                 # sweeps K=2
K_LIST="1 2" bash runpod_e4.sh    # re-enable the coarse K=1 rows
```

`runpod_e4.sh` writes each result JSON directly to the network volume
(`/workspace/results`, override with `PERSIST_DIR`) so sections survive a spot
preemption, and best-effort stops the pod at the end (needs `RUNPOD_API_KEY`
and `RUNPOD_POD_ID`).

**Caveat on the default K=2.** Sigma-delta already matches float at K=2
(E2.3), so the levers may hit a ceiling there — the coarse K=1 regime is where
spiking hurts and the restorations should show the most headroom. `K_LIST="1 2"`
restores those rows (~1/3 more GPU time).

Results land in `e1/results/e4_*.json` (local) / `e1/results/` on the pod.

## Full-1319 results

_Pending RunPod run (`runpod_e4.sh`)._
