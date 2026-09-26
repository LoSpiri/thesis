"""Parallel sandbox scoring (uses multiple vCPUs on RunPod).

The GSM8K sandbox `evaluate_samples` is CPU-bound and single-threaded in the
naive loop; this scores a list of (response, gold) pairs in a process pool.
Spawn context is used so workers never inherit the CUDA/MPS context.
"""
import multiprocessing as mp
import os
import sys

# Ensure the worker can import sandbox_gsm8k (repo root) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _score_one(args):
    import sandbox_gsm8k
    resp, gold = args
    try:
        return bool(sandbox_gsm8k.evaluate_samples(resp, gold, timeout_s=5.0))
    except Exception:
        return False


def score_all(pairs, workers=8):
    if workers <= 1 or len(pairs) <= 1:
        return [_score_one(p) for p in pairs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(_score_one, pairs, chunksize=16)
