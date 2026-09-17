"""Frozen graph-only timing: capture/dirty/replay verification, no eager fallback."""

from __future__ import annotations

import math
import statistics

import torch


def capture_verified(step, *, warmup, dirty, verify, calls_per_graph=1):
    if warmup < 1 or calls_per_graph < 1:
        raise ValueError("warmup and calls_per_graph must be positive")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            step()
    # Check multiple replays with dirty outputs, including all-zero references.
    # NaN dirt cannot coincidentally satisfy a valid output, unlike zero filling.
    for _ in range(3):
        dirty()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if not verify():
            raise RuntimeError("graph replay failed dirty-output verification")
    return graph


def graph_bench(step, *, warmup, iters, repeat, calls_per_graph, dirty, verify):
    if iters < 1 or repeat < 1:
        raise ValueError("iters and repeat must be positive")
    graph = capture_verified(
        step, warmup=warmup, dirty=dirty, verify=verify, calls_per_graph=calls_per_graph,
    )
    # Capture many calls so host submission gaps/event overhead are amortized.
    # Every normalized sample covers exactly calls_per_graph actual invocations.
    # Keep CUDAGraph.replay visible to KernelForge's sitecustomize graph probe.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    all_samples = []
    repeats = []
    for _ in range(repeat):
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if elapsed <= 0 or not math.isfinite(elapsed):
                raise RuntimeError(f"invalid graph event timing: {elapsed}")
            samples.append(elapsed / calls_per_graph)
        if not verify():
            raise RuntimeError("graph output verification failed after timing")
        repeats.append(statistics.median(samples))
        all_samples.extend(samples)
    return {
        "mode": "cudagraph", "times_ms": all_samples,
        "calls_per_graph": calls_per_graph,
        "repeat_medians_ms": repeats,
        "median_ms": statistics.median(all_samples),
        "mean_ms": statistics.mean(all_samples),
        "min_ms": min(all_samples), "max_ms": max(all_samples),
    }
