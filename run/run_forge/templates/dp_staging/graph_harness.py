"""Protected CUDA/HIP graph capture and GPU-event timing, with mandatory replay checks."""

from __future__ import annotations

import math
import statistics

import torch


def capture_graph(step, *, warmup: int):
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.synchronize()
    return graph


def checked_replay(graph, *, dirty, verify):
    dirty()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    if not verify():
        raise RuntimeError("graph replay failed dirty-output or input-preservation verification")


def graph_bench(step, *, warmup: int, iters: int, repeat: int, dirty, verify, calls_per_graph=1):
    if iters <= 0 or repeat <= 0 or calls_per_graph <= 0:
        raise ValueError("iters, repeat, and calls_per_graph must be positive")

    def graph_step():
        for _ in range(calls_per_graph):
            step()

    graph = capture_graph(graph_step, warmup=warmup)
    checked_replay(graph, dirty=dirty, verify=verify)
    samples = []
    for _ in range(repeat):
        for _ in range(max(1, warmup)):
            graph.replay()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed = begin.elapsed_time(end) / (iters * calls_per_graph)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError(f"invalid GPU-event time: {elapsed}")
        samples.append(elapsed)
        checked_replay(graph, dirty=dirty, verify=verify)
    return {"mode": "cudagraph", "samples_ms": samples, "median_ms": statistics.median(samples)}
