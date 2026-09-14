#!/usr/bin/env python3
"""Run a concurrency sweep against a container-local fake-decode server."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

from analyze import save_json, summarize, write_csv, write_summary

HERE = Path(__file__).resolve().parent


def get(base, endpoint):
    with urllib.request.urlopen(base + endpoint, timeout=10) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 40, 64])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--label", default="sweep")
    parser.add_argument("--min-requests", type=int, default=128)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--benchmark-timeout", type=int, default=2400)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
    default_out = Path(os.environ.get("RESULTS_DIR", HERE / "results")) / stamp
    parser.add_argument("--out-dir", type=Path, default=os.environ.get("OUT_DIR", default_out))
    args = parser.parse_args()
    isl, osl = int(os.environ.get("ISL", 10000)), int(os.environ.get("OSL", 500))
    if min(*args.concurrencies, args.repeats, args.min_requests, args.waves,
           args.ready_timeout, args.benchmark_timeout, isl) < 1 or osl < 2:
        parser.error("Counts must be positive; OSL must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, underscores or hyphens")
    if len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("concurrencies must be unique")

    run = args.out_dir.resolve()
    points = [(f"{args.label}_c{c:02d}_r{r:02d}", c, r)
              for r in range(1, args.repeats + 1) for c in args.concurrencies]
    for point, _, _ in points:
        if (run / "rounds" / point).exists():
            parser.error(f"Round already exists: {point}; choose a new label or output directory")

    base = f"http://127.0.0.1:{os.environ.get('PORT', '31832')}"
    print(f"Waiting for {base}; results: {run}", flush=True)
    deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            get(base, "/health")
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Server readiness timed out; check the server terminal")
            time.sleep(5)
    info = get(base, "/server_info")
    if (info["disaggregation_mode"] != "decode"
            or info["disaggregation_transfer_backend"] != "fake"
            or info.get("enable_profile_cuda_graph")):
        raise RuntimeError("Expected a fake-decode server without graph profiling; use up.sh")
    if max(args.concurrencies) > info["max_running_requests"]:
        parser.error(f"Concurrency exceeds server capacity {info['max_running_requests']}; restart with CONC=...")

    # Keep boundary repeats in one directory only while server settings and lengths match.
    config = {"server": {k: v for k, v in info.items() if k != "internal_states"},
              "input_len": isl, "output_len": osl}
    config_path = run / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Server settings or token lengths changed; choose a fresh output directory")
    run.mkdir(parents=True, exist_ok=True)
    save_json(config_path, config)
    save_json(run / f"server-info-{args.label}.json", info)

    # Patch a local copy: leave the installed SGLang benchmark untouched.
    source = Path(os.environ.get("SGLANG_DIR", "/sglang")) / "python/sglang/benchmark/serving.py"
    client = run / "bench_with_details.py"
    shutil.copyfile(source, client)
    subprocess.run(["patch", "--batch", "--fuzz=0", str(client), str(HERE / "client_details.patch")], check=True)
    for point, conc, repeat in points:
        directory = run / "rounds" / point
        directory.mkdir(parents=True)
        n = max(args.min_requests, args.waves * conc)
        n = (n + conc - 1) // conc * conc
        warmup = max(16, conc)
        # Cycle the original 160 prompts; native random/tokenize otherwise truncates large runs.
        prompts = [{"conversations": [
            {"from": "human", "value": f"Example {i % 160}. Describe an integer sequence and its next value."},
            {"from": "gpt", "value": "The next value depends on the rule."}
        ]} for i in range(max(160, n))]
        save_json(directory / "prompts.json", prompts)
        meta = {"point": point, "concurrency": conc, "repeat": repeat, "num_requests": n,
                "warmup_requests": warmup, "input_len": isl, "output_len": osl}
        save_json(directory / "run.json", meta)
        command = [sys.executable, str(client),
            "--backend", "sglang", "--host", "127.0.0.1", "--port", str(info["port"]),
            "--model", info["model_path"], "--served-model-name", info["served_model_name"],
            "--dataset-name", "random", "--dataset-path", str(directory / "prompts.json"),
            "--tokenize-prompt", "--random-input-len", str(isl), "--random-output-len", str(osl),
            "--random-range-ratio", "1", "--num-prompts", str(n),
            "--max-concurrency", str(conc), "--warmup-requests", str(warmup),
            "--fake-prefill", "--output-details", "--output-file", str(directory / "benchmark.jsonl")]
        (directory / "command.txt").write_text(shlex.join(command) + "\n")
        print(f"{point}: {n} measured + {warmup} warmup; log: {directory / 'benchmark.log'}", flush=True)
        with (directory / "benchmark.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.benchmark_timeout)
        data = json.loads((directory / "benchmark.jsonl").read_text())
        row, requests = summarize(data, meta)
        save_json(directory / "server-info-after.json", get(base, "/server_info"))
        write_csv(directory / "requests.csv", requests)
        save_json(directory / "metrics.json", row)
        print(f"  output={row['output_tokens_s']:.2f} tok/s; "
              f"TPOT P50/P90={row['tpot_ms_p50']:.3f}/{row['tpot_ms_p90']:.3f} ms", flush=True)
    write_summary(run)


if __name__ == "__main__":
    main()
