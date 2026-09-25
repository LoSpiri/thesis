"""Unattended E1 experiment runner (temperature-parameterized).

Runs the sigma-delta sweep (fast) then the full-GSM8K quality sweep (slow)
at a fixed temperature, with resume.

Usage: python run_all.py [--temp 0.1]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PY = sys.executable

MODELS = ["sfm", "sfm-dit", "flm", "duo", "mdlm"]
STEPS = [4, 8, 16, 32]
WTA = [(m, s, 1) for m in ("sfm", "sfm-dit") for s in STEPS]


def log(msg, logfile):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(logfile, "a") as f:
        f.write(line + "\n")


def tag(model, steps, topk, temp):
    t = f"t{temp}".replace(".", "_")
    return f"{model}_s{steps}_{t}" + (f"_topk{topk}" if topk else "")


def run(cmd, out_path, timeout_s, logfile):
    if os.path.exists(out_path):
        return "skip"
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=os.path.dirname(HERE), timeout=timeout_s,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        if r.returncode == 0 and os.path.exists(out_path):
            log(f"  OK {os.path.basename(out_path)} ({time.time()-t0:.0f}s)", logfile)
            for l in tail:
                log(f"      {l.strip()[:120]}", logfile)
            return "ok"
        else:
            log(f"  FAIL {os.path.basename(out_path)} rc={r.returncode}", logfile)
            for l in tail:
                log(f"      {l.strip()[:200]}", logfile)
            return "fail"
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT {os.path.basename(out_path)}", logfile)
        return "timeout"
    except Exception as e:
        log(f"  ERROR {os.path.basename(out_path)}: {e}", logfile)
        return "error"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--quality-only", action="store_true")
    p.add_argument("--sigma-only", action="store_true")
    p.add_argument("--batch", type=int,
                   default=int(os.environ.get("QUALITY_BATCH", "16")),
                   help="quality eval batch size (default: $QUALITY_BATCH or 16)")
    args = p.parse_args()
    temp = args.temp
    ttag = f"t{temp}".replace(".", "_")
    logfile = os.path.join(HERE, f"run_all_{ttag}.log")

    log(f"=== E1 run (T={temp}, batch={args.batch}) start ===", logfile)

    if not args.quality_only:
        log("--- sigma-delta sweep ---", logfile)
        jobs = [(m, s, None) for m in MODELS for s in STEPS] + WTA
        for m, s, topk in jobs:
            out = os.path.join(RESULTS, f"sigma_{tag(m, s, topk, temp)}.json")
            cmd = [PY, os.path.join(HERE, "run_sigma.py"), "--model", m,
                   "--steps", str(s), "--length", "64", "--temperature",
                   str(temp), "--out", out]
            if topk:
                cmd += ["--topk", str(topk)]
            run(cmd, out, timeout_s=600, logfile=logfile)

    if not args.sigma_only:
        log("--- full-GSM8K quality sweep ---", logfile)
        jobs = [(m, s, None) for m in MODELS for s in STEPS] + WTA
        for m, s, topk in jobs:
            out = os.path.join(RESULTS, f"quality_{tag(m, s, topk, temp)}.json")
            cmd = [PY, os.path.join(HERE, "run_quality.py"), "--model", m,
                   "--steps", str(s), "--batch", str(args.batch), "--temperature",
                   str(temp), "--out", out]
            if topk:
                cmd += ["--topk", str(topk)]
            timeout_s = 3600 + 600 * s
            run(cmd, out, timeout_s=timeout_s, logfile=logfile)

    log(f"=== E1 run (T={temp}) complete ===", logfile)
    subprocess.run([PY, os.path.join(HERE, "summarize.py")],
                   cwd=os.path.dirname(HERE))


if __name__ == "__main__":
    main()
