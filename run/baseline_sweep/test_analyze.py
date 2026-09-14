import copy
import json
from pathlib import Path
import tempfile
import unittest

from analyze import percentile, save_json, summarize, write_summary


def sample():
    # 100, 76.9 and 50 decode tok/s; each post-first SSE chunk carries 1 or 2 tokens.
    tpots = [10., 13., 20.]
    sizes = [2] * 249 + [1]
    data = {"max_concurrency": 3, "completed": 3, "duration": 10.,
            "input_lens": [10000] * 3, "output_lens": [500] * 3,
            "ttfts": [.1] * 3, "latencies": [.1 + t / 1000 * 499 for t in tpots],
            "itls": [[t / 1000] * 499 for t in tpots],
            "raw_chunk_gaps": [[t / 1000 * s for s in sizes] for t in tpots],
            "chunk_token_counts": [sizes] * 3, "start_times": [0.] * 3,
            "successes": [True] * 3, "errors": [""] * 3,
            "output_throughput": 150., "request_throughput": .3, "concurrency": 3.}
    flat = [v * 1000 for request in data["itls"] for v in request]
    for name, q in (("median", .5), ("p90", .9), ("p99", .99)):
        data[f"{name}_tpot_ms"] = percentile(tpots, q)
        data[f"{name}_itl_ms"] = percentile(flat, q)
    meta = {"point": "test", "num_requests": 3, "concurrency": 3, "repeat": 1,
            "input_len": 10000, "output_len": 500}
    return data, meta


class AnalyzeTest(unittest.TestCase):
    def test_request_slo_and_mtp_chunk_timing(self):
        row, requests = summarize(*sample())
        self.assertAlmostEqual(requests[0]["decode_tokens_s"], 100)
        self.assertLess(requests[0]["e2e_tokens_s"], 100)
        self.assertEqual(row["slo_70_pass_fraction"], 2 / 3)
        self.assertEqual(row["slo_80_pass_fraction"], 1 / 3)
        self.assertEqual(row["slo_70_goodput_tokens_s"], 100)
        self.assertEqual(row["slo_80_goodput_tokens_s"], 50)
        self.assertFalse(row["slo_70_p90_pass"])
        self.assertGreater(row["raw_chunk_gap_ms_p50"], row["itl_ms_p50"])

    def test_invalid_rounds_are_rejected(self):
        data, meta = sample()
        for key, value in {"completed": 2, "input_lens": [10000] * 2,
                           "output_lens": [499] * 3, "errors": ["failed", "", ""],
                           "successes": [False] * 3, "median_tpot_ms": 100,
                           "p90_itl_ms": 100, "duration": 0,
                           "latencies": [float("nan")] * 3,
                           "chunk_token_counts": [[1]] * 3}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize({**data, key: value}, meta)

    def test_custom_lengths_and_single_chunk(self):
        data, meta = sample()
        data.update(input_lens=[8] * 3, output_lens=[2] * 3,
                    itls=[[]] * 3, raw_chunk_gaps=[[]] * 3, chunk_token_counts=[[]] * 3)
        for name, q in (("median", .5), ("p90", .9), ("p99", .99)):
            data[f"{name}_tpot_ms"] = percentile([(x - .1) * 1000 for x in data["latencies"]], q)
            data[f"{name}_itl_ms"] = 0
        row, _ = summarize(data, {**meta, "input_len": 8, "output_len": 2})
        self.assertIsNone(row["itl_ms_p50"])

    def test_every_repeat_must_pass(self):
        row, _ = summarize(*sample())
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for point, conc, passing in (("c1", 1, True), ("c2_r1", 2, True),
                                         ("c2_r2", 2, False), ("c4_r1", 4, True)):
                directory = run / "rounds" / point
                directory.mkdir(parents=True)
                metrics = copy.deepcopy(row)
                metrics.update(point=point, concurrency=conc)
                for target in (70, 80):
                    metrics.update({f"slo_{target}_p50_pass": passing,
                                    f"slo_{target}_p90_pass": passing,
                                    f"slo_{target}_pass_fraction": float(passing)})
                save_json(directory / "metrics.json", metrics)
            failed = run / "rounds/c4_r2"
            failed.mkdir()
            save_json(failed / "run.json", {"point": "c4_r2", "concurrency": 4})
            write_summary(run)
            result = json.loads((run / "analysis/summary.json").read_text())
            self.assertEqual(len(result["incomplete_rounds"]), 1)
            for decision in result["decisions"].values():
                for rule in ("p50", "p90", "90pct_requests"):
                    self.assertEqual(decision[f"max_tested_concurrency_{rule}"], 1)


if __name__ == "__main__":
    unittest.main()
