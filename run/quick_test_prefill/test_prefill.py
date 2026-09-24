"""CPU checks for cache-hit boundaries before an expensive model launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from prefill import cached_prefix_length


DIRECTORY = Path(__file__).resolve().parent


class CacheHitTests(unittest.TestCase):
    def test_realized_prefix(self):
        # Cold, exact, rounded, and all-hit requests must still leave a suffix.
        for length, rate, page, expected in (
            (32768, 0, 64, 0),
            (32768, 0.75, 64, 24576),
            (1000, 0.75, 64, 704),
            (32768, 1, 64, 32704),
            (1000, 1, 64, 960),
            (64, 1, 64, 0),
            (1, 1, 64, 0),
            (1000, 0.75, 1, 750),
        ):
            with self.subTest(length=length, rate=rate, page=page):
                self.assertEqual(cached_prefix_length(length, rate, page), expected)

    def test_invalid_rates_fail_before_loading_model(self):
        for rate in ("-0.01", "1.01", "75", "nan", "inf", "-inf"):
            with self.subTest(rate=rate):
                result = subprocess.run(
                    [sys.executable, str(DIRECTORY / "prefill.py"),
                     "--result-dir", "/unused", f"--cache-hit-rate={rate}", "--dry-run"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("cache-hit-rate must be finite and between 0 and 1", result.stderr)

    def test_shell_environment_and_cli_override(self):
        for extra, expected in (([], 0.75), (["--cache-hit-rate", "0.5"], 0.5)):
            with self.subTest(extra=extra):
                result = subprocess.run(
                    ["bash", str(DIRECTORY / "bench.sh"), "--dry-run", *extra],
                    env={**os.environ, "CACHE_HIT_RATE": "0.75"},
                    check=True, capture_output=True, text=True,
                )
                config = json.loads(result.stdout)
                self.assertEqual(config["benchmark"]["cache_hit_rate"], expected)
                self.assertNotIn("--cache-hit-rate", config["server_cli"])


if __name__ == "__main__":
    unittest.main()
