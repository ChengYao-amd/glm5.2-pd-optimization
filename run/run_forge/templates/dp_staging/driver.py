"""Protected single-GPU DP staging driver: exact correctness, graph timing, profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

import kernel
from graph_harness import capture_graph, checked_replay, graph_bench
from reference import bitwise_equal, gather_reference, make_input, scatter_reference


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "cases.json").read_text())
SCORED_CASES = CONFIG["scored_cases"]
ALL_CASES = {case["id"]: case for case in SCORED_CASES + CONFIG["diagnostics"]}
GUARD_ELEMENTS = 4096
GUARD_BITS = 23130


class Fixture:
    def __init__(self, local_rows: int, global_rows: int, *, profiling=False):
        self.local_rows, self.global_rows = local_rows, global_rows
        self.hidden = CONFIG["hidden"]
        self.local_input = torch.empty((local_rows, self.hidden), dtype=torch.bfloat16, device="cuda")
        self.global_input = torch.empty((global_rows, self.hidden), dtype=torch.bfloat16, device="cuda")
        self.local_backing, self.local_output = self._guarded(self.local_input.shape)
        self.global_backing, self.global_output = self._guarded(self.global_input.shape)
        self.start = torch.tensor(0, dtype=torch.int64, device="cuda")
        self.valid = torch.tensor(local_rows, dtype=torch.int64, device="cuda")
        self.reset_inputs(variant=0, seed=CONFIG["seed"], preserve=not profiling)
        self.set_metadata(0, local_rows, reference=False)
        if not profiling:
            self.dirty()

    @staticmethod
    def _guarded(shape):
        count = shape.numel()
        backing = torch.empty(count + 2 * GUARD_ELEMENTS, dtype=torch.bfloat16, device="cuda")
        output = backing[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(shape)
        assert output.is_contiguous()
        return backing, output

    def reset_inputs(self, *, variant: int, seed: int, preserve=True):
        self.local_input.copy_(make_input(self.local_rows, self.hidden, variant, seed))
        self.global_input.copy_(make_input(self.global_rows, self.hidden, variant, seed + 103))
        if preserve:
            self.local_original = self.local_input.clone()
            self.global_original = self.global_input.clone()

    def set_metadata(self, start: int, valid: int, *, reference: bool = True):
        if not (0 <= valid <= self.local_rows and 0 <= start <= self.global_rows - valid):
            raise ValueError("invalid staging range")
        self.start.fill_(start)
        self.valid.fill_(valid)
        self.start_host, self.valid_host = start, valid
        if reference:
            self.expected_global = gather_reference(self.local_original, self.global_input.shape, start, valid)
            self.expected_local = scatter_reference(self.global_original, self.local_input.shape, start, valid)

    def dirty(self):
        self.local_output.fill_(-13)
        self.global_output.fill_(19)
        for backing in (self.local_backing, self.global_backing):
            backing[:GUARD_ELEMENTS].view(torch.int16).fill_(GUARD_BITS)
            backing[-GUARD_ELEMENTS:].view(torch.int16).fill_(GUARD_BITS)

    def gather(self):
        kernel.gather(self.global_output, self.local_input, self.start, self.valid)

    def scatter(self):
        kernel.scatter(self.local_output, self.global_input, self.start, self.valid)

    def pair(self):
        self.gather()
        self.scatter()

    def step(self, case):
        if case["kind"] == "gather":
            self.gather()
        elif case["kind"] == "scatter":
            self.scatter()
        elif case["kind"] == "sequence":
            # Fixed per-rank production frequency, with omitted norm/collective
            # boundaries. Each side has independent immutable input and output.
            for _ in range(case["scatter_calls"]):
                self.gather()
                self.scatter()
            for _ in range(case["gather_calls"] - case["scatter_calls"]):
                self.gather()
        else:
            raise ValueError(f"unknown case kind {case['kind']}")

    def verify(self, kind="sequence"):
        checks = [
            bitwise_equal(self.local_input, self.local_original),
            bitwise_equal(self.global_input, self.global_original),
            self.start.item() == self.start_host,
            self.valid.item() == self.valid_host,
        ]
        if kind != "scatter":
            checks.append(bitwise_equal(self.global_output, self.expected_global))
        if kind != "gather":
            checks.append(bitwise_equal(self.local_output, self.expected_local))
        for backing in (self.local_backing, self.global_backing):
            checks.append(bool((backing[:GUARD_ELEMENTS].view(torch.int16) == GUARD_BITS).all()))
            checks.append(bool((backing[-GUARD_ELEMENTS:].view(torch.int16) == GUARD_BITS).all()))
        return all(checks)


def correctness():
    checked = 0
    validation = CONFIG["validation"]
    for layout_index, layout in enumerate(validation["layouts"]):
        fixture = Fixture(layout["local_rows"], layout["global_rows"])
        # Capture once per allocation shape. Metadata and tensor values change
        # in place afterwards, catching host-specialized/stale graph results.
        graph = capture_graph(fixture.pair, warmup=3)
        for variant in validation["input_variants"]:
            fixture.reset_inputs(variant=variant, seed=CONFIG["seed"] + variant * 59)
            for rank in validation["ranks"]:
                for valid in layout["valid_rows"]:
                    fixture.set_metadata(rank * layout["local_rows"], valid)
                    fixture.dirty()
                    fixture.pair()
                    torch.cuda.synchronize()
                    if not fixture.verify():
                        raise RuntimeError(f"eager mismatch: rows={layout['local_rows']} rank={rank} valid={valid} variant={variant}")
                    checked_replay(graph, dirty=fixture.dirty, verify=fixture.verify)
                    checked_replay(graph, dirty=fixture.dirty, verify=fixture.verify)
                    checked += 3
            # Extra observed variable-count layouts from node 168 exercise
            # cumsum-derived offsets rather than only fixed padded rank slots.
            counts = validation["packed_counts"][layout_index]
            offset = 0
            for rank, valid in enumerate(counts):
                fixture.set_metadata(offset, valid)
                checked_replay(graph, dirty=fixture.dirty, verify=fixture.verify)
                offset += valid
                checked += 1
    for case in SCORED_CASES:
        fixture = Fixture(case["local_rows"], case["global_rows"])
        fixture.set_metadata(case["rank"] * case["local_rows"], case["valid_rows"])
        graph = capture_graph(lambda: fixture.step(case), warmup=3)
        for cycle in range(4):
            fixture.reset_inputs(variant=cycle % 2, seed=CONFIG["seed"] + 71 * cycle)
            fixture.set_metadata(case["rank"] * case["local_rows"], case["valid_rows"])
            checked_replay(graph, dirty=fixture.dirty, verify=fixture.verify)
            checked += 1
    print(f"verified_checks: {checked}")
    print("comparison: bitwise BF16, including input preservation, padding, guards, and mutable graph metadata")
    print("scope: single GPU local staging; ranks 0..3 simulated; no collective or norm validation")
    print("SNR: 100.00 dB")
    print("allclose: True")
    return 0


def benchmark(case, args, *, diagnostic=False):
    fixture = Fixture(case["local_rows"], case["global_rows"])
    fixture.set_metadata(case["rank"] * case["local_rows"], case["valid_rows"])
    result = graph_bench(
        lambda: fixture.step(case), warmup=args.warmup, iters=args.iters,
        repeat=args.repeat, dirty=fixture.dirty,
        verify=lambda: fixture.verify(case["kind"]),
        calls_per_graph=case.get("graph_invocations", 1),
    )
    print("# mode: cudagraph; scope: single_gpu_local_staging")
    if diagnostic:
        print(f"diagnostic_ms: {case['id']} {result['median_ms']:.9f}")
        print("diagnostic_samples_ms: " + json.dumps(result["samples_ms"]))
    else:
        for sample in result["samples_ms"]:
            print(f"wall_ms: {sample:.9f}")
        print(f"case_ms: {case['id']} {result['median_ms']:.9f}")
        print(f"median_ms: {result['median_ms']:.9f}")


def profile(case, args):
    # Initialization uses CPU fixtures; no reference result or correctness path
    # runs in profile mode. Captured/replayed regions contain only the target.
    fixture = Fixture(case["local_rows"], case["global_rows"], profiling=True)
    fixture.set_metadata(case["rank"] * case["local_rows"], case["valid_rows"], reference=False)
    graph = capture_graph(lambda: fixture.step(case), warmup=args.warmup)
    for _ in range(args.iters):
        graph.replay()
    torch.cuda.synchronize()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bench-mode", action="store_true")
    mode.add_argument("--profile-run", action="store_true")
    mode.add_argument("--diagnostic-case", choices=[c["id"] for c in CONFIG["diagnostics"]])
    parser.add_argument("--profile-case", choices=list(ALL_CASES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.profile_case and not args.profile_run:
        parser.error("--profile-case requires --profile-run")
    if args.warmup < 0 or args.iters <= 0 or args.repeat <= 0:
        parser.error("warmup must be nonnegative; iters and repeat must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA GPU is required")
    if args.profile_run:
        return profile(ALL_CASES[args.profile_case or SCORED_CASES[0]["id"]], args)
    if args.diagnostic_case:
        benchmark(ALL_CASES[args.diagnostic_case], args, diagnostic=True)
        return 0
    if args.bench_mode:
        for case in SCORED_CASES:
            benchmark(case, args)
        return 0
    return correctness()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--profile-run" not in sys.argv:
            print("SNR: 0.00 dB")
            print("allclose: False")
        sys.exit(1)
