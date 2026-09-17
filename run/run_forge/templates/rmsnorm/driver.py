"""Frozen driver for the residual RMSNorm component of the 025 norm/staging task.

Default: complete correctness suite, nonzero exit on any output/input violation.
--bench-mode: scored workload only, measured exclusively through graph replay.
--profile-run [--profile-case ID]: initialization plus target dispatches only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-mode", action="store_true")
    mode.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", help="exact scored CASE_ID; only with --profile-run")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.repeat) < 1:
        parser.error("--warmup, --iters and --repeat must be positive")
    if args.profile_case and not args.profile_run:
        parser.error("--profile-case requires --profile-run")
    return args


def main():
    args = arguments()
    # --help stays usable without a GPU or optional AITER dependency.
    import torch
    from graph_harness import capture_verified, graph_bench
    from kernel import fused_add_rmsnorm
    from reference import compare, make_inputs, reference

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA GPU is required")
    contract = json.loads(Path(__file__).with_name("cases.json").read_text())
    scored = contract["scored_cases"]
    eps = contract["epsilon"]
    torch.set_grad_enabled(False)

    def setup(case):
        x, residual, weight = make_inputs(case, contract)
        # Guard allocations catch row overrun while retaining contiguous outputs.
        n = contract["hidden"]
        out_storage = torch.full((case["rows"] + 2, n), 31.0, dtype=x.dtype, device=x.device)
        res_storage = torch.full_like(out_storage, -29.0)
        out, residual_out = out_storage[1:-1], res_storage[1:-1]
        inputs_before = tuple(t.clone() for t in (x, residual, weight))
        expected = reference(x, residual, weight, eps)

        def step():
            fused_add_rmsnorm(x, residual, weight, out, residual_out, eps)

        def dirty():
            out.fill_(float("nan"))
            residual_out.fill_(float("nan"))

        def inspect():
            result = compare(out, residual_out, expected, contract)
            result["inputs_unchanged"] = all(
                torch.equal(t, before) for t, before in zip((x, residual, weight), inputs_before)
            )
            result["guards_unchanged"] = bool(
                (out_storage[[0, -1]] == 31).all().item()
                and (res_storage[[0, -1]] == -29).all().item()
            )
            result["ok"] = result["ok"] and result["inputs_unchanged"] and result["guards_unchanged"]
            return result

        return step, dirty, inspect

    if args.profile_run:
        selected = args.profile_case or contract["profile_case"]
        matches = [case for case in scored if case["id"] == selected]
        if len(matches) != 1:
            raise ValueError(f"unknown scored profile case: {selected}")
        # Do not call setup: reference, guards, comparison, and metrics have no
        # place in hardware profiling. CPU inputs transfer before warmup.
        x, residual, weight = make_inputs(matches[0], contract)
        out, residual_out = torch.empty_like(x), torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                fused_add_rmsnorm(x, residual, weight, out, residual_out, eps)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_add_rmsnorm(x, residual, weight, out, residual_out, eps)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        return 0

    if args.bench_mode:
        case_times = []
        for case in scored:
            step, dirty, inspect = setup(case)
            result = graph_bench(
                step, warmup=args.warmup, iters=args.iters, repeat=args.repeat,
                calls_per_graph=contract["benchmark_calls_per_graph"],
                dirty=dirty, verify=lambda: inspect()["ok"],
            )
            case_times.append(result["median_ms"])
            print("# graph_benchmark " + json.dumps({"case_id": case["id"], **result}, sort_keys=True))
            print(f"case_ms: {case['id']} {result['median_ms']:.9f}")
        print(f"median_ms: {statistics.median(case_times):.9f}")
        print(f"mean_ms: {statistics.mean(case_times):.9f}")
        return 0

    results = []
    for case in [*scored, *contract["correctness_only"]]:
        step, dirty, inspect = setup(case)
        dirty()
        step()
        torch.cuda.synchronize()
        result = inspect()
        result["case_id"] = case["id"]
        result["graph_verified"] = False
        # The empty case is a vacuous wrapper contract, not a GPU workload.
        if result["ok"] and case["rows"]:
            capture_verified(step, warmup=2, dirty=dirty, verify=lambda: inspect()["ok"])
            result["graph_verified"] = True
        results.append(result)
        print("# correctness " + json.dumps(result, sort_keys=True))
    ok = all(result["ok"] for result in results)
    print(f"SNR: {min(result['snr_db'] for result in results):.6f} dB")
    print(f"allclose: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"driver_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
