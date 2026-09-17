#!/usr/bin/env python3
"""Check one decode result or collect a sweep into CSV; fail on incomplete runs."""
import argparse
import csv
import json
from pathlib import Path

ACCEPTANCE_FIELDS = ("input_len", "output_len", "seed", "accept_length", "accept_method", "accept_token_mode")


def summarize(point):
    row = {"point": str(point)}
    try:
        result = json.loads((point / "result.json").read_text())
        config = json.loads((point / "config.json").read_text())["benchmark"]
        status = json.loads((point / "status.json").read_text())
        for key in ("batch_size", "tp_size", "ep_size", "dp_size", "local_batch_size", "input_len", "output_len",
                    "verify_iterations", "realized_accept_length", "output_tokens_per_second",
                    "output_tokens_per_second_per_gpu", "effective_token_latency_ms_per_user"):
            row[key] = result[key]
        row.update({key: config[key] for key in ACCEPTANCE_FIELDS})
        dp = config["tp_size"] if config["enable_dp_attention"] else 1
        config_fields = ("batch_size", "tp_size", "ep_size", "input_len", "output_len", "enable_dp_attention",
                         "accept_method", "accept_token_mode")
        checks = {
            "driver failed": status["exit_code"] == 0,
            "incomplete decode": result["complete"],
            "profiled run": result["is_performance_measurement"],
            "token count mismatch": result["useful_output_tokens"] == config["batch_size"] * config["output_len"],
            "DP topology mismatch": result["dp_size"] == dp and result["local_batch_size"] * dp == config["batch_size"],
            "configuration mismatch": all(result[key] == config[key] for key in config_fields),
            "acceptance target mismatch": result["expected_accept_length"] == config["accept_length"],
            "invalid acceptance length": 1 <= result["realized_accept_length"] <= 6,
        }
        row["problems"] = "; ".join(name for name, passed in checks.items() if not passed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        row["problems"] = str(error)
    row["verdict"] = "fail" if row["problems"] else "pass"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="one result directory, or the parent of sweep points")
    parser.add_argument("--output", type=Path, help="optional CSV output")
    args = parser.parse_args()
    if (args.directory / "config.json").is_file():
        points = [args.directory]
    else:
        points = sorted(path for path in args.directory.iterdir() if path.is_dir())
    rows = [summarize(point) for point in points]
    acceptance = {}
    for row in rows:
        if row["verdict"] != "pass":
            continue
        key = tuple(row[field] for field in ACCEPTANCE_FIELDS)
        expected = acceptance.setdefault(key, row["realized_accept_length"])
        if row["realized_accept_length"] != expected:
            row.update(verdict="fail", problems="acceptance differs across equivalent sweep points")
    rows.sort(key=lambda row: (row.get("batch_size", 0), row["point"]))
    print(json.dumps(rows, indent=2))
    if args.output and rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return int(not rows or any(row["verdict"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
