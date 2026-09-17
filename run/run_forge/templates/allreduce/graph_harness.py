# SPDX-License-Identifier: MIT
# Rank launch, communicator lifecycle and graph timing adapted from KernelForge
# examples/aiter-allreduce-forge-loop/driver.py (AMD, MIT).
"""Frozen TP4 launcher, candidate identity checks and real IPC graph harness."""
from __future__ import annotations

import atexit
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist


JIT_MODULE = "module_custom_all_reduce"
_child = None


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.origin:
        raise RuntimeError("AITER is unavailable; put the candidate source root on PYTHONPATH")
    root = Path(spec.origin).resolve().parent.parent
    expected = os.environ.get("FORGE_EXPECTED_AITER_ROOT") or os.environ.get("FORGE_AITER_ROOT")
    if expected and root != Path(expected).resolve():
        raise RuntimeError(f"candidate AITER root mismatch: imported {root}, expected {expected}")
    if os.environ.get("AITER_META_DIR") and Path(os.environ["AITER_META_DIR"]).resolve() != root:
        raise RuntimeError("AITER_META_DIR points outside the imported candidate source tree")
    required = [
        root / "csrc/kernels/custom_all_reduce.cu",
        root / "csrc/pybind/custom_all_reduce_pybind.cu",
        root / "csrc/include/custom_all_reduce.cuh",
        root / "csrc/include/custom_all_reduce.h",
        root / "aiter/jit/core.py",
        root / "aiter/jit/optCompilerConfig.json",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"candidate source dependency missing: {path}")
    paths = set(required)
    paths.update(p for p in (root / "csrc/include").rglob("*") if p.is_file())
    paths.update((root / "aiter/dist").rglob("*.py"))
    paths.update((root / "aiter/ops").glob("*all_reduce*.py"))
    digest = hashlib.sha256()
    # Cache identity also includes build configuration and toolchain identity.
    config = {key: os.environ.get(key, "") for key in (
        "GPU_ARCHS", "ENABLE_CK", "AITER_DISABLE_KERNARG_PRELOAD", "HIP_PATH", "ROCM_PATH"
    )}
    config.update(torch_version=torch.__version__, hip_version=torch.version.hip, source_root=str(root))
    digest.update(json.dumps(config, sort_keys=True).encode())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return {"aiter_root": str(root), "source_hash": digest.hexdigest(), "build_config": config}


def ensure_candidate_built():
    """Serialize rebuilds for both self-launch and external torchrun workers.

    AITER_JIT_DIR must be a task-local cache, outside the frozen source tree.
    A full rebuild on a changed fingerprint avoids stale transitive headers.
    Both the fingerprint and the resulting .so digest are checked on reuse.
    """
    identity = source_identity()
    cache_value = os.environ.get("AITER_JIT_DIR")
    if not cache_value:
        root_hash = hashlib.sha256(identity["aiter_root"].encode()).hexdigest()[:16]
        cache_value = f"/tmp/forge-tp4-allreduce-{os.getuid()}/{root_hash}"
        os.environ["AITER_JIT_DIR"] = cache_value
    cache = Path(cache_value).resolve()
    source_root = Path(identity["aiter_root"])
    if cache == source_root or source_root in cache.parents:
        raise RuntimeError("AITER_JIT_DIR must be outside the candidate source tree")
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / ".forge-tp4-allreduce.json"
    module = cache / f"{JIT_MODULE}.so"
    with (cache / ".forge-tp4-allreduce.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached = json.loads(stamp.read_text())
        except (OSError, json.JSONDecodeError):
            cached = {}
        ready = (cached.get("source_hash") == identity["source_hash"] and module.is_file()
                 and cached.get("module_sha256") == file_hash(module))
        if not ready:
            env = dict(os.environ, AITER_REBUILD="1", AITER_META_DIR=str(source_root),
                       PYTHONDONTWRITEBYTECODE="1")
            print(f"[forge-ar] build source={identity['source_hash'][:16]} cache={cache}", file=sys.stderr, flush=True)
            result = subprocess.run(
                [sys.executable, "-c", "import aiter; print(aiter.meta_size())"],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if result.returncode or not module.is_file():
                raise RuntimeError(f"candidate JIT build failed (exit {result.returncode}):\n{result.stdout[-12000:]}")
            identity.update(module_path=str(module), module_sha256=file_hash(module))
            temporary_stamp = stamp.with_suffix(f".tmp.{os.getpid()}")
            temporary_stamp.write_text(json.dumps(identity, indent=2) + "\n")
            temporary_stamp.replace(stamp)
        else:
            identity = cached
    os.environ["AITER_REBUILD"] = "0"
    os.environ["AITER_META_DIR"] = str(source_root)
    return identity


def _stop_child(*_args):
    global _child
    proc, _child = _child, None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def self_launch(driver, argv):
    global _child
    requested = int(os.environ.get("FORGE_NPROC_PER_NODE", "4"))
    if requested != 4:
        raise ValueError(f"this frozen workload requires TP4; FORGE_NPROC_PER_NODE={requested}")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"need exactly four visible GPUs; found {torch.cuda.device_count()}")
    ensure_candidate_built()
    env = dict(os.environ, FORGE_NPROC_PER_NODE="4", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
    # Keep the caller's process group, so Forge timeout group-kills reach every rank.
    _child = subprocess.Popen([
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        str(Path(driver).resolve()), *argv,
    ], env=env)
    atexit.register(_stop_child)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop_child)
    try:
        return _child.wait()
    finally:
        _stop_child()


@dataclass
class Worker:
    rank: int
    device: torch.device
    group: object
    comm: object
    identity: dict


def init_worker():
    rank = int(os.environ["RANK"])
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("external torchrun must set WORLD_SIZE=4")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    identity = ensure_candidate_built()
    import aiter
    from aiter.jit import core
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized, get_tp_group,
        init_distributed_environment, set_custom_all_reduce,
    )
    if Path(aiter.__file__).resolve().parent.parent != Path(identity["aiter_root"]):
        raise RuntimeError("AITER imported outside verified candidate root")
    if Path(core.AITER_CSRC_DIR).resolve() != Path(identity["aiter_root"]) / "csrc":
        raise RuntimeError("AITER JIT includes point outside candidate source snapshot")
    aiter.meta_size()
    loaded = sys.modules.get(JIT_MODULE)
    if loaded is None or Path(loaded.__file__).resolve() != Path(identity["module_path"]):
        raise RuntimeError("custom allreduce loaded a stale/unverified extension")
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=4, rank=rank, local_rank=local_rank)
    ensure_model_parallel_initialized(4, 1)
    group = get_tp_group()
    comm = getattr(group.device_communicator, "ca_comm", None)
    if comm is None or comm.disabled:
        raise RuntimeError("custom allreduce unavailable; RCCL fallback is not a valid candidate")
    if not comm.enable_register_for_capturing:
        raise RuntimeError("production graph IPC input registration is disabled")
    torch.cuda.synchronize()
    dist.barrier(group=group.device_group)
    identity["registered_input_graph_capture"] = True
    return Worker(rank, torch.device(f"cuda:{local_rank}"), group, comm, identity)


def close_worker():
    from aiter.dist.parallel_state import destroy_distributed_environment, destroy_model_parallel
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()


def candidate(ctx, x):
    # This is the raw public communicator ABI used by the production wrapper.
    # No caller-side global_tokens[:] writeback belongs inside this operation.
    output = ctx.comm.custom_all_reduce(x)
    if output is None:
        raise RuntimeError("custom_all_reduce returned None; fallback is forbidden")
    return output


def capture(ctx, x, chain=1):
    from aiter.dist.parallel_state import graph_capture
    if not ctx.comm.should_custom_ar(x):
        raise RuntimeError("candidate communicator rejects the fixed 2,359,296-byte payload")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(chain):
                outputs.append(candidate(ctx, x))
    return graph, outputs


def reduce_scalar(value, ctx, op=dist.ReduceOp.MAX):
    tensor = torch.tensor(value, device=ctx.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op, group=ctx.group.device_group)
    return tensor.item()


def benchmark(ctx, x, warmup, iters):
    for _ in range(warmup):
        candidate(ctx, x)
    graph, outputs = capture(ctx, x, chain=iters)
    for _ in range(max(2, warmup)):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        dist.barrier(group=ctx.group.device_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(reduce_scalar(start.elapsed_time(end) / iters, ctx))
    # Keep outputs alive through every replay and timing synchronization.
    del graph, outputs
    return statistics.median(samples), samples
