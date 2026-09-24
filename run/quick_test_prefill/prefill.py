#!/usr/bin/env python3
"""Single-node, fixed-batch GLM-5.2 prefill. Extra flags go to SGLang ServerArgs."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def cached_prefix_length(input_len, hit_rate, page_size):
    """Match a page-aligned prefix, leaving a suffix to produce next-token logits."""
    if not 0 <= hit_rate <= 1:
        raise ValueError("cache hit rate must be finite and between 0 and 1")
    if input_len < 1 or page_size < 1:
        raise ValueError("input length and KV page size must be positive")
    return min(int(input_len * hit_rate), input_len - 1) // page_size * page_size


def prefill_range(runner, reqs, start, end, chunk_len):
    """Extend real KV from start to end without resetting the cached prefix."""
    logits = None
    for chunk_start in range(start, end, chunk_len):
        chunk_end = min(chunk_start + chunk_len, end)
        for req in reqs:
            req.set_extend_range(chunk_start, chunk_end)
        _, logits, _ = runner.extend(reqs)
        for req in reqs:
            req.prefix_indices = runner.torch_runner.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, :chunk_end
            ].to(req.prefix_indices.dtype)
    return logits


def save_profile(profiler, directory, rank):
    profiler.export_chrome_trace(str(directory / f"prefill-TP-{rank}.trace.json.gz"))
    events = profiler.key_averages(group_by_input_shape=True)
    for kind in ("operators", "kernels"):
        rows = []
        for event in events:
            gpu = str(event.device_type).endswith("CUDA")
            if event.is_user_annotation or gpu != (kind == "kernels"):
                continue
            rows.append({"name": event.key, "count": event.count,
                         "self_cpu_us": event.self_cpu_time_total,
                         "self_gpu_us": event.self_device_time_total,
                         "total_gpu_us": event.device_time_total,
                         "input_shapes": str(event.input_shapes)})
        rows.sort(key=lambda row: row["self_gpu_us"], reverse=True)
        fields = ("name", "count", "self_cpu_us", "self_gpu_us", "total_gpu_us", "input_shapes")
        with (directory / f"{kind}_rank_{rank}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if kind == "kernels" and not rows:
            raise RuntimeError("Profiler captured no GPU kernels")


def run_rank(rank, server_args, port_args, args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from sglang.benchmark.one_batch import load_model, prepare_synthetic_inputs_for_latency_test
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.utils import configure_logger

    torch.cuda.set_device(rank)
    publish(server_args, role="scheduler")
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    configure_logger(server_args, prefix=f" TP{rank}")
    cfg = resolving_view(server_args)
    runner, _ = load_model(server_args, port_args, rank, rank)
    model = runner.torch_runner
    local_bs = args.batch_size // cfg.dp_size
    page = model.token_to_kv_pool_allocator.page_size
    cached_len = cached_prefix_length(args.input_len, args.cache_hit_rate, page)
    extend_len = args.input_len - cached_len
    capacity = local_bs * math.ceil(args.input_len / page) * page
    if capacity > model.max_total_num_tokens or args.input_len > model.model_config.context_len:
        raise ValueError("Batch exceeds KV capacity or model context length")
    if local_bs > model.req_to_token_pool.size:
        raise ValueError("Batch exceeds request pool capacity")
    # Equal chunks per request, page-aligned except the final chunk.
    chunk_len = args.input_len
    if cfg.chunked_prefill_size > 0 and local_bs * chunk_len > cfg.chunked_prefill_size:
        chunk_len = cfg.chunked_prefill_size // local_bs // page * page
        if chunk_len == 0:
            raise ValueError("Chunk budget must fit at least one KV page per local request")
    dp_rank = rank // (cfg.tp_size // cfg.dp_size)
    np.random.seed(args.seed + dp_rank)
    inputs = np.random.randint(0, 10000, (local_bs, args.input_len), dtype=np.int32)
    directory = Path(args.result_dir)

    @torch.inference_mode()
    def prefill(reqs, start=cached_len, end=args.input_len):
        return prefill_range(runner, reqs, start, end, chunk_len)

    def prepare():
        runner.clear()
        reqs = prepare_synthetic_inputs_for_latency_test(local_bs, args.input_len, list(inputs))
        for i, req in enumerate(reqs):
            req.rid = dp_rank * local_bs + i
            req.sampling_params.max_new_tokens = 1
        # Materialize every layer's real prefix KV, including DSA indexer state.
        # Setup is outside both timing and profiling; every iteration starts
        # with exactly cached_len resident tokens per request.
        if cached_len:
            prefill(reqs, 0, cached_len)
        return reqs

    if rank == 0:
        print(f"Cache hit rate: requested={args.cache_hit_rate:.6f}, "
              f"effective={cached_len / args.input_len:.6f}; "
              f"cached={cached_len}, new={extend_len} tokens/request, KV page={page}", flush=True)
        print(f"Warmup {args.warmup_steps} batches; {math.ceil(extend_len / chunk_len)} measured chunks/request", flush=True)
    for _ in range(args.warmup_steps):
        logits = prefill(prepare())
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Non-finite prefill logits after warmup")
    times = []
    for step in range(args.steps):
        reqs = prepare()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        prefill(reqs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if rank == 0:
            print(f"Prefill {step + 1}/{args.steps}: rank0={elapsed:.4f}s", flush=True)
    write_json(directory / f"rank_{rank}.json", {"rank": rank, "latency_seconds": times})
    slowest = torch.tensor(times, dtype=torch.float64, device="cuda")
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    if rank == 0:
        latencies = slowest.tolist()
        mean = statistics.mean(latencies)
        result = {"is_performance_measurement": True, "batch_size": args.batch_size,
                  "input_len": args.input_len, "tp": cfg.tp_size, "dp": cfg.dp_size, "ep": cfg.ep_size,
                  "local_batch_size": local_bs, "chunk_budget_per_dp": cfg.chunked_prefill_size,
                  "chunk_tokens_per_request": chunk_len,
                  "chunks_per_request": math.ceil(extend_len / chunk_len),
                  "requested_cache_hit_rate": args.cache_hit_rate,
                  "effective_cache_hit_rate": cached_len / args.input_len,
                  "cached_tokens_per_request": cached_len,
                  "new_tokens_per_request": extend_len, "kv_page_size": page,
                  "cache_hit_rate_basis": "token fraction per request; GPU-resident prefix KV",
                  "latency_seconds": latencies, "mean_latency_seconds": mean,
                  "median_latency_seconds": statistics.median(latencies),
                  "input_tokens_per_second": args.batch_size * args.input_len / mean,
                  "input_tokens_per_second_per_gpu": args.batch_size * args.input_len / mean / cfg.tp_size,
                  "new_tokens_per_second": args.batch_size * extend_len / mean,
                  "new_tokens_per_second_per_gpu": args.batch_size * extend_len / mean / cfg.tp_size,
                  "scope": "suffix prefill: batch preparation, forward and sampling; "
                           "excludes prefix KV setup, scheduler, HiCache and KV transfer",
                  "profile_separate_from_measurement": args.profile}
        write_json(directory / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)

    if args.profile:
        trace_dir = directory / "traces"
        trace_dir.mkdir(exist_ok=True)
        emit = args.profile_ranks == "all" or rank in [int(x) for x in args.profile_ranks.split(",")]
        for step in range(args.profile_steps):
            reqs = prepare()
            torch.cuda.synchronize()
            dist.barrier()
            profiler = None
            if emit:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=args.profile_record_shapes, with_stack=args.profile_with_stack)
                profiler.start()
            prefill(reqs)
            torch.cuda.synchronize()
            if profiler is not None:
                profiler.stop()
                step_dir = trace_dir if args.profile_steps == 1 else trace_dir / f"step_{step + 1}"
                step_dir.mkdir(exist_ok=True)
                save_profile(profiler, step_dir, rank)
            dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=8, help="global batch, divided across attention DP ranks")
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--cache-hit-rate", type=float, default=0.0,
                        help="cached token fraction per request (0..1); real GPU prefix KV is "
                             "prepared outside timing; rounded down to KV pages, leaving a suffix")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--profile", action="store_true", help="capture extra iterations AFTER clean timing")
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--profile-ranks", default="0", help="comma-separated TP ranks, or all")
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if min(args.batch_size, args.input_len, args.warmup_steps, args.steps, args.profile_steps) < 1:
        parser.error("batch size, input length and iteration counts must be positive")
    if not 0 <= args.cache_hit_rate <= 1:
        parser.error("cache-hit-rate must be finite and between 0 and 1")
    config = {"benchmark": vars(args), "server_cli": extra,
              "environment": {k: v for k, v in os.environ.items() if k.startswith(
                  ("SGLANG_", "AITER_", "HIP_", "HSA_", "NCCL_", "SAFETENSORS_"))},
              "image": os.environ.get("IMAGE"), "image_id": os.environ.get("EXPECTED_IMAGE_ID")}
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    import torch.multiprocessing as mp
    import sglang
    import torch
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_parser = argparse.ArgumentParser(allow_abbrev=False)
    ServerArgs.add_cli_args(server_parser)
    server_args = ServerArgs.from_cli_args(server_parser.parse_args(extra))
    server_args.resolve_once()
    cfg = resolving_view(server_args)
    config["runtime"] = {"sglang": sglang.__version__, "torch": torch.__version__,
                         "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0),
                         "sglang_commit": subprocess.check_output(
                             ["git", "-C", str(Path(sglang.__file__).resolve().parents[2]),
                              "rev-parse", "HEAD"], text=True).strip()}
    if cfg.nnodes != 1 or any(getattr(cfg, k) != 1 for k in ("pp_size", "attn_cp_size", "dcp_size", "moe_dp_size")):
        parser.error("requires single-node PP1/CP1/DCP1/MoEDP1")
    if cfg.tp_size % cfg.dp_size or args.batch_size % cfg.dp_size or (cfg.dp_size > 1 and not cfg.enable_dp_attention):
        parser.error("TP and global batch must be divisible by DP; DP>1 requires DP attention")
    if cfg.load_format == "dummy" or cfg.speculative_algorithm or cfg.enable_hierarchical_cache:
        parser.error("requires real target weights, no speculation and no HiCache")
    if cfg.disaggregation_mode != "null":
        parser.error("KV transfer requires a server; keep disaggregation-mode=null")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("use plain python, not torchrun")
    if args.profile_ranks != "all":
        ranks = [int(x) for x in args.profile_ranks.split(",")]
        if not ranks or min(ranks) < 0 or max(ranks) >= cfg.tp_size:
            parser.error("profile-ranks must name existing TP ranks")
    directory = Path(args.result_dir).resolve()
    directory.mkdir(parents=True)  # Require fresh output, including after a failed run.
    args.result_dir = str(directory)
    print(f"Results: {directory}", flush=True)
    write_json(directory / "config.json", config)
    write_json(directory / "server_args.json", server_args.resolved_dict())
    (directory / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    status = {"exit_code": 1}
    write_json(directory / "status.json", status)
    start = time.perf_counter()
    try:
        _set_envs_and_config(server_args)
        mp.spawn(run_rank, args=(server_args, PortArgs.init_new(server_args), args), nprocs=cfg.tp_size, join=True)
        status["exit_code"] = 0
    finally:
        status["wall_seconds"] = time.perf_counter() - start
        write_json(directory / "status.json", status)


if __name__ == "__main__":
    main()
