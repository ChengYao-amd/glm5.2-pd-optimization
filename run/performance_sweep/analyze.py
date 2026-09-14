#!/usr/bin/env python3
"""Summarize completed sweep rounds and select the largest measured SLO-safe concurrency."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["point", "concurrency"])
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def distribution(values, prefix):
    return {f"{prefix}_{key}": func(values) if values else None for key, func in (
        ("mean", statistics.mean), ("p50", lambda v: percentile(v, .5)),
        ("p90", lambda v: percentile(v, .9)), ("p95", lambda v: percentile(v, .95)),
        ("p99", lambda v: percentile(v, .99)), ("max", max))}


def summarize(data, meta):
    n = meta["num_requests"]
    keys = ("input_lens", "output_lens", "ttfts", "itls", "latencies", "start_times",
            "successes", "raw_chunk_gaps", "chunk_token_counts", "errors")
    if any(len(data[key]) != n for key in keys):
        raise ValueError("Request count mismatch")
    if (data["max_concurrency"] != meta["concurrency"] or data["completed"] != n
            or not all(data["successes"]) or any(data["errors"])):
        raise ValueError("Incomplete benchmark or client concurrency mismatch")
    if not math.isfinite(data["duration"]) or data["duration"] <= 0:
        raise ValueError("Invalid benchmark duration")
    for key, expected in (("output_throughput", n * meta["output_len"] / data["duration"]),
                          ("request_throughput", n / data["duration"])):
        if not math.isclose(data[key], expected, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"Native {key} mismatch")
    requests, all_itls, all_gaps, all_sizes = [], [], [], []
    for i in range(n):
        output = data["output_lens"][i]
        ttft, latency = data["ttfts"][i], data["latencies"][i]
        if (data["input_lens"][i] != meta["input_len"] or output != meta["output_len"]
                or output < 2 or not (math.isfinite(ttft) and math.isfinite(latency) and 0 <= ttft < latency)):
            raise ValueError(f"Invalid token lengths or timing for request {i}")
        tpot = (latency - ttft) / (output - 1) * 1000
        itls = [x * 1000 for x in data["itls"][i]]
        gaps = [x * 1000 for x in data["raw_chunk_gaps"][i]]
        sizes = data["chunk_token_counts"][i]
        if (len(gaps) != len(sizes) or sum(sizes) != len(itls)
                or len(itls) > output - 1
                or any(type(s) is not int or s <= 0 for s in sizes)
                or any(not math.isfinite(x) or x < 0 for x in itls + gaps)):
            raise ValueError(f"Invalid SSE chunk timing for request {i}")
        offset = 0
        for gap, size in zip(gaps, sizes):
            if any(not math.isclose(itl, gap / size, rel_tol=1e-9, abs_tol=1e-7)
                   for itl in itls[offset:offset + size]):
                raise ValueError(f"SSE gap/token timing mismatch for request {i}")
            offset += size
        all_itls.extend(itls)
        all_gaps.extend(gaps)
        all_sizes.extend(sizes)
        requests.append({"request_index": i, "input_tokens": data["input_lens"][i],
            "output_tokens": output, "start_time_monotonic_s": data["start_times"][i],
            "ttft_ms": ttft * 1000, "e2e_ms": latency * 1000, "tpot_ms": tpot,
            "decode_tokens_s": 1000 / tpot, "e2e_tokens_s": output / latency,
            "pass_70": 1000 / tpot >= 70, "pass_80": 1000 / tpot >= 80})
    tpots = [r["tpot_ms"] for r in requests]
    # Compare independent request timing with native SGLang percentiles.
    for metric, values in (("tpot", tpots), ("itl", all_itls)):
        for name, q in (("median", .5), ("p90", .9), ("p99", .99)):
            expected = percentile(values, q) if values else 0
            if not math.isclose(expected, data[f"{name}_{metric}_ms"], rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError(f"Native {metric} {name} mismatch")
    row = {**meta, "completed": n, "duration_s": data["duration"],
           "output_tokens_s": data["output_throughput"], "request_throughput": data["request_throughput"],
           "observed_client_concurrency": data["concurrency"],
           "native_accept_length_cumulative": data.get("accept_length")}
    for name, values in (("tpot_ms", tpots), ("itl_ms", all_itls), ("raw_chunk_gap_ms", all_gaps),
                         ("ttft_ms", [r["ttft_ms"] for r in requests]), ("e2e_ms", [r["e2e_ms"] for r in requests])):
        row.update(distribution(values, name))
    row["mean_tokens_per_chunk_after_first"] = statistics.mean(all_sizes) if all_sizes else None
    row["e2e_tokens_s_p50"] = percentile([r["e2e_tokens_s"] for r in requests], .5)
    for name, q in (("p10", .1), ("p50", .5), ("p90", .9)):
        row[f"decode_tokens_s_{name}"] = percentile([r["decode_tokens_s"] for r in requests], q)
    for target in (70, 80):
        passed = [r for r in requests if r[f"pass_{target}"]]
        row[f"slo_{target}_pass_fraction"] = len(passed) / n
        row[f"slo_{target}_goodput_tokens_s"] = sum(r["output_tokens"] for r in passed) / data["duration"]
        for q in ("p50", "p90"):
            row[f"slo_{target}_{q}_pass"] = row[f"tpot_ms_{q}"] <= 1000 / target
    return row, requests


def write_summary(run):
    # metrics.json is written only after the entire round passes validation.
    rows = [json.loads(p.read_text()) for p in run.glob("rounds/*/metrics.json")]
    incomplete = [json.loads(p.read_text()) for p in run.glob("rounds/*/run.json")
                  if not (p.parent / "metrics.json").exists()]
    if not rows and not incomplete:
        raise ValueError(f"No rounds in {run}")
    incomplete_concurrencies = {r["concurrency"] for r in incomplete}
    rows.sort(key=lambda r: (r["concurrency"], r["point"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["concurrency"]].append(row)
    decisions = {}
    for target in (70, 80):
        decisions[target] = {"tpot_limit_ms": 1000 / target}
        rules = {"p50": lambda r: r[f"slo_{target}_p50_pass"],
                 "p90": lambda r: r[f"slo_{target}_p90_pass"],
                 "90pct_requests": lambda r: r[f"slo_{target}_pass_fraction"] >= .9}
        for name, passes in rules.items():
            decisions[target][f"max_tested_concurrency_{name}"] = max(
                (c for c, rounds in groups.items()
                 if c not in incomplete_concurrencies and all(passes(r) for r in rounds)), default=None)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    write_csv(out / "metrics.csv", rows)
    save_json(out / "summary.json", {"points": rows, "incomplete_rounds": incomplete, "decisions": decisions})
    print(json.dumps(decisions, indent=2))
    print(f"Summary: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    write_summary(parser.parse_args().run_dir)
