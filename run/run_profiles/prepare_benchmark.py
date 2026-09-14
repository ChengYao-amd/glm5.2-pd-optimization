#!/usr/bin/env python3
"""Generate synthetic seed prompts and wait for the benchmark server to be ready."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("prompts_path", type=Path)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=os.environ.get("PORT", "31832"))
    parser.add_argument("--base-url")
    parser.add_argument("--ready-check-timeout-sec", type=float, default=3600)
    # Other options are forwarded unchanged to the native benchmark by bench.sh.
    args, _ = parser.parse_known_args()

    with args.prompts_path.open("w") as stream:
        json.dump([{"conversations": [
            {"from": "human", "value": f"Example {i % 160}. Describe an integer sequence and its next value."},
            {"from": "gpt", "value": "The next value depends on the rule."}
        ]} for i in range(max(160, args.num_prompts))], stream)

    # /v1/models can respond before the server's own PD warmup has completed.
    base = (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")
    deadline = time.monotonic() + args.ready_check_timeout_sec
    print(f"Waiting for {base}/health", flush=True)
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=10):
                break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Server readiness timed out; check the server log")
            time.sleep(5)


if __name__ == "__main__":
    main()
