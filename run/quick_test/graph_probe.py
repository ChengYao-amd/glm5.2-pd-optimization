#!/usr/bin/env python3
"""Check whether torch.profiler sees kernels inside graph replay, using one GPU."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.size, args.iters) < 1:
        parser.error("size and iters must be positive")
    import torch
    torch.manual_seed(1234)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.zeros(args.size, device="cuda", dtype=torch.bfloat16)

    def body():
        c.copy_(torch.tanh((a @ b) * 1.5 + 0.25).sum(dim=0))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    run = body
    if args.mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        run = graph.replay
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()
    kernels = [{"name": event.key, "count": event.count, "self_gpu_us": event.self_device_time_total}
               for event in profiler.key_averages() if str(event.device_type).endswith("CUDA")]
    report = {"mode": args.mode, "torch": torch.__version__, "hip": torch.version.hip,
              "device": torch.cuda.get_device_name(), "size": args.size, "iterations": args.iters,
              "DEBUG_CLR_GRAPH_PACKET_CAPTURE": os.environ.get("DEBUG_CLR_GRAPH_PACKET_CAPTURE"),
              "distinct_kernels": len(kernels),
              "kernels": sorted(kernels, key=lambda row: row["self_gpu_us"], reverse=True)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
