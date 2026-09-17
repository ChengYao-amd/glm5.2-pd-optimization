"""Profile a decode window and time target/draft stages after graph capture."""
import argparse
import json
import os
from pathlib import Path


def add_profile_cli_args(parser):
    group = parser.add_argument_group("profiling")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--profile-start-step", type=int, default=20)
    group.add_argument("--profile-num-steps", type=int, default=12)
    group.add_argument("--profile-ranks", default="0", help="comma-separated TP ranks, or all")
    group.add_argument("--profile-activities", default="CPU,GPU", help="comma-separated CPU,GPU,MEM,CUDA_PROFILER")
    group.add_argument("--profile-dir", help="default: <result-dir>/traces")
    group.add_argument("--profile-top-n", type=int, default=10)
    group.add_argument("--profile-with-stack", action=argparse.BooleanOptionalAction,
                       default=os.environ.get("SGLANG_PROFILE_WITH_STACK", "false").lower() == "true")
    group.add_argument("--profile-record-shapes", action=argparse.BooleanOptionalAction,
                       default=os.environ.get("SGLANG_PROFILE_RECORD_SHAPES", "false").lower() == "true")
    group.add_argument("--device-timer", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--profile-graph-capture-trace", action="store_true",
                       help="also capture graph construction as a kernel inventory")


def profile_ranks(args):
    ranks = list(range(args.tp_size)) if args.profile_ranks == "all" else [int(x) for x in args.profile_ranks.split(",")]
    if not ranks or any(rank < 0 or rank >= args.tp_size for rank in ranks):
        raise ValueError("profile-ranks must name existing TP ranks")
    return ranks


def validate_profile_args(args):
    if args.profile_graph_capture_trace and not args.profile:
        raise ValueError("profile-graph-capture-trace requires --profile")
    if not args.profile:
        return
    if args.profile_start_step < 0 or min(args.profile_num_steps, args.profile_top_n) < 1:
        raise ValueError("Profile start must be nonnegative; step count and top-n must be positive")
    activities = {x.strip().upper() for x in args.profile_activities.split(",")}
    if not activities or activities - {"CPU", "GPU", "MEM", "CUDA_PROFILER"}:
        raise ValueError("profile-activities must contain CPU, GPU, MEM or CUDA_PROFILER")
    profile_ranks(args)
    if args.max_steps and args.max_steps < args.profile_start_step + args.profile_num_steps:
        raise ValueError("max-steps ends before the profile window closes")


def configure_profile_env(args):
    args.profile_dir = str(Path(args.profile_dir or Path(args.result_dir) / "traces").resolve())
    Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = args.profile_dir
    if args.profile_graph_capture_trace:
        os.environ["SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"] = "1"


class ProfileSession:
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.emitting = rank in profile_ranks(args)
        self.directory = Path(args.profile_dir)
        self.activities = {x.strip().upper() for x in args.profile_activities.split(",")}
        self.profiler = None
        self.started = self.stopped = None
        self.by_category = {}
        self.by_runner = {}
        self.runners = []
        self.window_before = {}
        self.window_delta = {}

    def reporter(self, label):
        def report(t, category="unlabelled"):
            for table in (self.by_category, self.by_runner.setdefault(label, {})):
                entry = table.setdefault(category, {"seconds": 0.0, "count": 0})
                entry["seconds"] += t
                entry["count"] += 1
        return report

    def attach(self, runners):
        if not self.args.device_timer:
            return
        from sglang.srt.utils.device_timer import DeviceTimer
        # Separate timers avoid nested target/draft wraps; attach after warmup.
        for label, runner in runners.items():
            if runner.device_timer is not None:
                raise RuntimeError(f"{label} already has a device timer")
            runner.device_timer = DeviceTimer(reporter=self.reporter(label))
            self.runners.append(runner)

    def flush_timers(self):
        import torch
        torch.cuda.synchronize()
        for runner in self.runners:
            runner.device_timer._report()

    def step_boundary(self, completed):
        if not self.emitting or self.stopped is not None:
            return
        if self.started is None and completed == self.args.profile_start_step:
            self.start(completed)
        elif self.started is not None and completed >= self.args.profile_start_step + self.args.profile_num_steps:
            self.stop(completed)

    def start(self, completed):
        import torch
        self.flush_timers()
        self.window_before = {key: value.copy() for key, value in self.by_category.items()}
        if "MEM" in self.activities:
            from sglang.srt.environ import envs
            torch.cuda.memory._record_memory_history(max_entries=envs.SGLANG_MEM_PROFILE_MAX_ENTRIES.get())
        if "CUDA_PROFILER" in self.activities:
            torch.cuda.cudart().cudaProfilerStart()
        activity_map = {"CPU": torch.profiler.ProfilerActivity.CPU, "GPU": torch.profiler.ProfilerActivity.CUDA}
        selected = [value for key, value in activity_map.items() if key in self.activities]
        if selected:
            # SGLang's profiler stop has a collective; direct torch profiling also
            # works when only rank 0 emits a trace.
            self.profiler = torch.profiler.profile(
                activities=selected, with_stack=self.args.profile_with_stack,
                record_shapes=self.args.profile_record_shapes,
            )
            self.profiler.start()
        self.started = completed

    def stop(self, completed):
        import torch
        self.flush_timers()
        if self.profiler is not None:
            self.profiler.stop()
        if "CUDA_PROFILER" in self.activities:
            torch.cuda.cudart().cudaProfilerStop()
        if "MEM" in self.activities:
            torch.cuda.memory._dump_snapshot(str(self.directory / f"memory_rank_{self.rank}.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        self.window_delta = {
            key: {field: value[field] - self.window_before.get(key, {}).get(field, 0)
                  for field in ("seconds", "count")}
            for key, value in self.by_category.items()
        }
        self.stopped = completed

    def write_top_ops(self):
        # User annotations include nested kernel time and must not be summed with kernels.
        annotations = {event.key for event in self.profiler.events() if event.is_user_annotation}
        rows = [{"name": event.key, "count": event.count,
                 "self_device_time_us": event.self_device_time_total,
                 "cpu_time_us": event.cpu_time_total,
                 "device_kernel": str(event.device_type).endswith("CUDA") and event.key not in annotations}
                for event in self.profiler.key_averages()]
        kernels = sorted((row for row in rows if row["device_kernel"]),
                         key=lambda row: row["self_device_time_us"], reverse=True)
        payload = {
            "kernel_time_us": sum(row["self_device_time_us"] for row in kernels),
            "top_kernels": kernels[:self.args.profile_top_n],
            "top_cpu_ops": sorted(rows, key=lambda row: row["cpu_time_us"], reverse=True)[:self.args.profile_top_n],
        }
        self.write_json(f"top_ops_rank_{self.rank}.json", payload)
        lines = ["self GPU (us)  count  kernel"]
        lines += [f"{row['self_device_time_us']:13.2f} {row['count']:6d}  {row['name']}" for row in payload["top_kernels"]]
        (self.directory / f"top_ops_rank_{self.rank}.txt").write_text("\n".join(lines) + "\n")

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=2) + "\n")

    def finish(self, completed):
        if self.started is not None and self.stopped is None:
            self.stop(completed)
        self.flush_timers()
        for runner in self.runners:
            runner.device_timer = None
        if self.profiler is not None:
            self.profiler.export_chrome_trace(str(self.directory / f"decode-TP-{self.rank}.trace.json.gz"))
            self.write_top_ops()
        if self.args.device_timer:
            self.write_json(f"device_timer_rank_{self.rank}.json", {
                "scope": "whole measured decode loop", "by_category": self.by_category,
                "by_runner": self.by_runner, "profile_window": self.window_delta,
            })
        window_end = self.args.profile_start_step + self.args.profile_num_steps
        self.write_json(f"profile_meta_rank_{self.rank}.json", {
            "is_performance_measurement": False, "rank": self.rank, "emitting_rank": self.emitting,
            "cuda_graph_enabled": not self.args.disable_cuda_graph,
            "window_requested": [self.args.profile_start_step, window_end],
            "window_actual": [self.started, self.stopped], "window_complete": completed >= window_end,
            "capture_profile_note": "Graph construction traces are a kernel inventory, not execution timing.",
            "env": {key: value for key, value in os.environ.items()
                    if key.startswith(("SGLANG_", "DEBUG_CLR_", "HIP_", "AITER_"))},
        })
        if completed < window_end:
            raise RuntimeError(f"Decode ended at step {completed}, before profile window end {window_end}; increase OSL")
