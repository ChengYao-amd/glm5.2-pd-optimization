# SPDX-License-Identifier: MIT
"""Frozen GLM TP4 BF16 [192, 6144] raw all-reduce measurement contract.

No arguments: all three cases, all numerical modes, eager and real graph replay.
--bench-mode: graph-only timing, per-case medians of slowest-rank GPU timings.
--profile-run [--profile-case ID]: warmup then a few real graph replays, no oracle.
No RANK self-launches torchrun; external torchrun workers use their existing RANK.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

# Measurement/source snapshots are immutable during evaluation.
sys.dont_write_bytecode = True

import torch
import torch.distributed as dist

from graph_harness import (
    benchmark, candidate, capture, close_worker, init_worker, reduce_scalar, self_launch,
)
from reference import compare, make_input, passed, reference_sum

SENTINEL = "__FORGE_DISTRIBUTED_RESULT__"
CASE_IDS = ("gather_192x6144", "moe_out_192x6144", "dense_out_192x6144")


def load_contract():
    data = json.loads(Path(__file__).with_name("cases.json").read_text())
    if tuple(item["id"] for item in data["scored_cases"]) != CASE_IDS:
        raise ValueError("the complete ordered GLM case suite must be present")
    if data["shape"] != [192, 6144] or data["world_size"] != 4 or data["dtype"] != "bfloat16":
        raise ValueError("this driver only implements TP4 BF16 [192,6144]")
    return data


def check_case(ctx, case_id, contract):
    gates = contract["correctness"]
    result = {"snr_db": 200.0, "max_diff": 0.0, "passed": True, "checks": 0, "failures": []}

    def record(output, ref, before, x, label, previous=None):
        metrics = compare(ref, output, before, x, case_id, gates)
        if previous is not None and not torch.equal(previous, output):
            metrics["deterministic"] = False
        valid = passed(metrics, gates) and metrics.get("deterministic", True)
        result["snr_db"] = min(result["snr_db"], metrics["snr_db"])
        result["max_diff"] = max(result["max_diff"], metrics["max_diff"])
        result["passed"] = result["passed"] and valid
        result["checks"] += 1
        if not valid:
            result["failures"].append({"rank": ctx.rank, "check": label, **metrics})

    for mode_idx, mode in enumerate(gates["modes"]):
        seed = contract["seed"] + 7919 * mode_idx
        x = make_input(case_id, ctx.rank, ctx.device, seed, mode)
        before = x.clone()
        ref = reference_sum(x, ctx.group.device_group)
        eager = candidate(ctx, x)
        torch.cuda.synchronize()
        record(eager, ref, before, x, f"{mode}/eager")
        again = candidate(ctx, x)
        torch.cuda.synchronize()
        record(again, ref, before, x, f"{mode}/eager_repeat", eager)

        graph, outputs = capture(ctx, x)
        output = outputs[0]
        # Probe two different input payloads without recapturing. Dirty every
        # output first, so an empty/stale graph cannot pass even on zero inputs.
        for update in range(2):
            replay_input = make_input(case_id, ctx.rank, ctx.device, seed + 104729 * update, mode)
            x.copy_(replay_input)
            ref = reference_sum(replay_input, ctx.group.device_group)
            previous = None
            for replay in range(gates["graph_replays"]):
                output.fill_(float("nan") if replay % 2 else 0.0)
                torch.cuda.synchronize()
                # Skew rank arrival while all ranks eventually reach the same
                # collective. This is a correctness stress, outside all timing.
                if replay % 4 == ctx.rank:
                    time.sleep(0.001 * (ctx.rank + 1))
                graph.replay()
                torch.cuda.synchronize()
                record(output, ref, replay_input, x, f"{mode}/graph{update}/{replay}", previous)
                previous = output.clone()
            # A queue of replays tests signal reuse without host synchronization.
            output.fill_(float("nan"))
            for _ in range(gates["graph_replays"]):
                graph.replay()
            torch.cuda.synchronize()
            record(output, ref, replay_input, x, f"{mode}/graph{update}/queued", previous)
        del graph, outputs

    result["snr_db"] = reduce_scalar(result["snr_db"], ctx, dist.ReduceOp.MIN)
    result["max_diff"] = reduce_scalar(result["max_diff"], ctx)
    result["passed"] = bool(reduce_scalar(int(result["passed"]), ctx, dist.ReduceOp.MIN))
    failures = [None] * 4
    dist.all_gather_object(failures, result["failures"][:4], group=ctx.group.cpu_group)
    result["failures"] = [item for rank_failures in failures for item in rank_failures]
    return result


def emit(payload, dump_json):
    if dump_json:
        Path(dump_json).write_text(json.dumps(payload, indent=2) + "\n")
    # Compact protocol excludes detailed identities/checks, which stay in JSON.
    short = {key: payload[key] for key in (
        "kind", "world_size", "source_hash", "custom_ar_active", "graph_replay"
    )}
    if "case_ms" in payload:
        short["case_ms"] = payload["case_ms"]
    if "passed" in payload:
        short["passed"] = payload["passed"]
    print(SENTINEL + json.dumps(short, separators=(",", ":")) + SENTINEL, flush=True)


def worker_main(args, contract):
    ctx = init_worker()
    try:
        payload = {"world_size": 4, "source_hash": ctx.identity["source_hash"],
                   "identity": ctx.identity, "custom_ar_active": True, "graph_replay": True}
        if args.profile_run:
            # A default profile is one representative case, never a mixed suite.
            case_id = args.profile_case or CASE_IDS[0]
            x = make_input(case_id, ctx.rank, ctx.device, contract["seed"])
            for _ in range(args.warmup):
                candidate(ctx, x)
            graph, outputs = capture(ctx, x)
            for _ in range(max(1, args.iters)):
                graph.replay()
            torch.cuda.synchronize()
            del graph, outputs
            return 0

        if args.bench_mode:
            rounds = []
            sample_rounds = []
            for _ in range(args.repeat):
                values, samples = {}, {}
                for case_id in CASE_IDS:
                    x = make_input(case_id, ctx.rank, ctx.device, contract["seed"])
                    values[case_id], samples[case_id] = benchmark(ctx, x, args.warmup, args.iters)
                rounds.append(values)
                sample_rounds.append(samples)
            case_ms = {cid: statistics.median(r[cid] for r in rounds) for cid in CASE_IDS}
            payload.update(kind="integrated_bench", case_ms=case_ms, rounds_ms=rounds,
                           samples_ms=sample_rounds, mean_ms=statistics.mean(case_ms.values()),
                           repeat=args.repeat, iters=args.iters, warmup=args.warmup,
                           statistic="per-case median of repeated medians of five slowest-rank GPU event samples")
            if ctx.rank == 0:
                for cid, value in case_ms.items():
                    print(f"case_ms: {cid} {value:.9f}")
                print(f"mean_ms: {payload['mean_ms']:.9f}")
                emit(payload, args.dump_json)
            return 0

        results = {cid: check_case(ctx, cid, contract) for cid in CASE_IDS}
        worst_snr = min(value["snr_db"] for value in results.values())
        worst_diff = max(value["max_diff"] for value in results.values())
        all_passed = all(value["passed"] for value in results.values())
        payload.update(kind="correctness", passed=all_passed, snr_db=worst_snr,
                       max_diff=worst_diff, cases=results, gates=contract["correctness"])
        if ctx.rank == 0:
            print(f"SNR: {worst_snr:.6f} dB")
            print(f"allclose: {all_passed}")
            print(f"max_diff: {worst_diff:.9e}")
            emit(payload, args.dump_json)
            if not all_passed:
                print(json.dumps(results, indent=2), file=sys.stderr)
        return 0 if all_passed else 1
    finally:
        close_worker()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bench-mode", action="store_true")
    group.add_argument("--profile-run", action="store_true")
    parser.add_argument("--profile-case", choices=CASE_IDS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dump-json", default="")
    # Forge's preparer may issue --shape default and mode probes. These never
    # narrow the correctness contract or introduce a different workload.
    parser.add_argument("--shape", default="", choices=("", "default"))
    parser.add_argument("--mode", choices=("smoke", "stability", "determinism"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.repeat < 1:
        parser.error("warmup must be >=0; iters and repeat must be >=1")
    if args.profile_case and not args.profile_run:
        parser.error("--profile-case is valid only with --profile-run")
    contract = load_contract()
    if "RANK" not in os.environ:
        return self_launch(__file__, sys.argv[1:])
    return worker_main(args, contract)


if __name__ == "__main__":
    raise SystemExit(main())
