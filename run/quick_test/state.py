"""Parallel layout, sequence progress and decode metrics."""
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecodeTopology:
    tp_size: int
    ep_size: int
    batch_size: int  # Global request concurrency, never per-DP concurrency.
    enable_dp_attention: bool = False

    def __post_init__(self):
        if min(self.tp_size, self.ep_size, self.batch_size) <= 0:
            raise ValueError("TP, EP and global batch sizes must be positive")
        if self.tp_size % self.ep_size:
            raise ValueError("EP size must divide TP size")
        if self.batch_size % self.dp_size:
            raise ValueError("Global batch size must divide evenly across attention DP shards")

    @property
    def dp_size(self):
        return self.tp_size if self.enable_dp_attention else 1

    @property
    def local_batch_size(self):
        return self.batch_size // self.dp_size

    @property
    def representative_ranks(self):
        return list(range(self.tp_size)) if self.enable_dp_attention else [0]

    def parallel_state_kwargs(self, rank):
        if not 0 <= rank < self.tp_size:
            raise ValueError("Rank is outside the TP process group")
        return dict(
            tp_rank=rank, tp_size=self.tp_size, pp_rank=0, pp_size=1,
            dp_rank=rank if self.enable_dp_attention else None, dp_size=self.dp_size,
            attn_tp_rank=0 if self.enable_dp_attention else rank,
            attn_tp_size=self.tp_size // self.dp_size,
            attn_cp_rank=0, attn_cp_size=1, attn_dcp_rank=0, attn_dcp_size=1,
            attn_dp_rank=rank if self.enable_dp_attention else 0, attn_dp_size=self.dp_size,
            moe_ep_rank=rank // (self.tp_size // self.ep_size), moe_ep_size=self.ep_size,
            moe_dp_rank=0, moe_dp_size=1, gpu_id=rank,
        )

    def request_id_offset(self, rank):
        return self.parallel_state_kwargs(rank)["attn_dp_rank"] * self.local_batch_size

    def summary(self):
        return dict(tp_size=self.tp_size, ep_size=self.ep_size, dp_size=self.dp_size,
                    enable_dp_attention=self.enable_dp_attention,
                    global_batch_size=self.batch_size, local_batch_size=self.local_batch_size)


def prepare_dp_metadata(batch, topology, *, is_extend, disable_cuda_graph):
    # ScheduleBatch counts are BASE requests. The pinned ForwardBatch scales them
    # using spec_info separately for draft, target verify and draft extension.
    batch.global_num_tokens = [topology.local_batch_size] * topology.dp_size
    batch.global_num_tokens_for_logprob = batch.global_num_tokens[:]
    batch.is_extend_in_batch = is_extend
    batch.can_run_dp_cuda_graph = not is_extend and not disable_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = False


def rank_progress_signature(iteration, previous_lens, accept_lens, new_lens, final_len):
    """Never raise locally before the collective: encode invalid state instead."""
    valid = bool(previous_lens) and len(previous_lens) == len(accept_lens) == len(new_lens)
    valid = valid and len(set(accept_lens)) == 1 and len(set(previous_lens)) == 1
    valid = valid and all(1 <= count <= 6 for count in accept_lens)
    valid = valid and all(new == old + count for old, count, new in zip(previous_lens, accept_lens, new_lens))
    return [int(valid), iteration, len(previous_lens), previous_lens[0] if previous_lens else -1,
            accept_lens[0] if accept_lens else -1, new_lens[0] if new_lens else -1,
            int(bool(new_lens) and all(length >= final_len for length in new_lens))]


def validate_rank_progress(signatures):
    if not signatures or any(not row[0] or list(row) != list(signatures[0]) for row in signatures):
        raise ValueError(f"Cross-rank acceptance/progress/completion divergence: {signatures}")


def aggregate_rank_summaries(reports, topology):
    if len(reports) != topology.tp_size or sorted(report["rank"] for report in reports) != list(range(topology.tp_size)):
        raise ValueError("Expected exactly one report from every TP process")
    reports = sorted(reports, key=lambda report: report["rank"])
    first = reports[0]
    # Uniform shared coins keep every process in the same collective sequence.
    fields = ("complete", "verify_iterations", "batch_size", "input_len", "output_len",
              "emitted_per_request", "final_seq_lens", "raw_accept_tokens", "accept_histogram")
    for report in reports:
        if report["batch_size"] != topology.local_batch_size or any(report[key] != first[key] for key in fields):
            raise ValueError("Rank reports disagree on completion or acceptance exposure")
    replicas = [reports[rank] for rank in topology.representative_ranks]
    result = dict(first)
    result.update(topology.summary())
    result.update(batch_size=topology.batch_size, report_scope="global",
                  representative_ranks=topology.representative_ranks,
                  elapsed_seconds=max(report["elapsed_seconds"] for report in reports))
    for key in ("useful_output_tokens", "raw_accept_tokens", "num_correct_drafts", "terminal_extra_tokens"):
        result[key] = sum(report[key] for report in replicas)
    for key in ("emitted_per_request", "final_seq_lens"):
        result[key] = [value for report in replicas for value in report[key]]
    histogram = Counter()
    for report in replicas:
        histogram.update({int(key): value for key, value in report["accept_histogram"].items()})
    result["accept_histogram"] = dict(sorted(histogram.items()))
    exposure = topology.batch_size * result["verify_iterations"]
    result["realized_accept_length"] = result["raw_accept_tokens"] / exposure if exposure else 0.0
    useful, elapsed = result["useful_output_tokens"], result["elapsed_seconds"]
    result["output_tokens_per_second"] = useful / elapsed if elapsed > 0 else 0.0
    result["output_tokens_per_second_per_gpu"] = result["output_tokens_per_second"] / topology.tp_size
    result["effective_token_latency_ms_per_user"] = elapsed * 1000 * topology.batch_size / useful if useful else 0.0
    result["graph_execution_counts_scope"] = "rank_0"
    result["graph_execution_counts_by_rank"] = {str(report["rank"]): report["graph_execution_counts"] for report in reports}
    result["per_dp_useful_output_tokens"] = [report["useful_output_tokens"] for report in replicas]
    if "bootstrap_tokens_excluded" in first:
        result["bootstrap_tokens_excluded"] = sum(report["bootstrap_tokens_excluded"] for report in replicas)
    if "phase_seconds" in first:
        result["phase_seconds_max_rank"] = {
            key: max(report["phase_seconds"][key] for report in reports) for key in first["phase_seconds"]
        }
    result.pop("rank", None)
    result.pop("parallel_state", None)
    return result


def validate_kv_layout(cache_dim, kv_lora_rank, rope_dim, packed_fp8):
    if packed_fp8 or cache_dim != kv_lora_rank + rope_dim:
        raise ValueError("Physical initializer requires raw MLA nope+rope layout, not packed FP8/scales/BF16 rope")


def count_graph_executions(runners):
    """Attach host-side counters after graph capture; preserve execute results."""
    counts = {name: 0 for name in runners}
    for name, runner in runners.items():
        if runner is None:
            continue
        execute = runner.execute

        def counted(*args, _name=name, _execute=execute, **kwargs):
            result = _execute(*args, **kwargs)
            counts[_name] += 1
            return result

        runner.execute = counted
    return counts


def required_token_capacity(batch_size, input_len, output_len, page_size, reserve):
    if min(batch_size, input_len, output_len, page_size) <= 0 or reserve < 0:
        raise ValueError("lengths and batch/page sizes must be positive; reserve nonnegative")
    per_request = input_len + output_len + reserve
    return batch_size * ((per_request + page_size - 1) // page_size * page_size)


@dataclass
class DecodeAccounting:
    batch_size: int
    input_len: int
    output_len: int
    max_accept_len: int = 6
    emitted: list = field(init=False)
    verify_ct: int = 0
    raw_accept_tokens: int = 0
    accept_histogram: Counter = field(default_factory=Counter)

    def __post_init__(self):
        if min(self.batch_size, self.input_len, self.output_len, self.max_accept_len) <= 0:
            raise ValueError("batch and lengths must be positive")
        self.emitted = [0] * self.batch_size

    @property
    def seq_lens(self):
        return [self.input_len + count for count in self.emitted]

    @property
    def complete(self):
        return all(count == self.output_len for count in self.emitted)

    def record(self, accept_lens):
        if self.complete:
            raise ValueError("Cannot record another iteration after output completion")
        if len(accept_lens) != self.batch_size or any(
            not isinstance(count, int) or not 1 <= count <= self.max_accept_len
            for count in accept_lens
        ):
            raise ValueError("accept_lens must have one valid bonus-inclusive integer per request")
        useful = [min(count, self.output_len - done) for count, done in zip(accept_lens, self.emitted)]
        self.emitted = [done + count for done, count in zip(self.emitted, useful)]
        self.verify_ct += 1
        self.raw_accept_tokens += sum(accept_lens)
        self.accept_histogram.update(accept_lens)
        return {"iteration": self.verify_ct, "accept_lens": list(accept_lens),
                "useful_accept_lens": useful, "seq_lens": self.seq_lens,
                "complete": self.complete}

    def summary(self, elapsed_seconds):
        useful = sum(self.emitted)
        exposure = self.batch_size * self.verify_ct
        return {
            "complete": self.complete,
            "batch_size": self.batch_size,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "emitted_per_request": self.emitted[:],
            "final_seq_lens": self.seq_lens,
            "verify_iterations": self.verify_ct,
            "useful_output_tokens": useful,
            "raw_accept_tokens": self.raw_accept_tokens,
            "num_correct_drafts": self.raw_accept_tokens - exposure,
            "terminal_extra_tokens": self.raw_accept_tokens - useful,
            "realized_accept_length": self.raw_accept_tokens / exposure if exposure else 0.0,
            "accept_histogram": dict(sorted(self.accept_histogram.items())),
            "elapsed_seconds": elapsed_seconds,
            "output_tokens_per_second": useful / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "effective_token_latency_ms_per_user": elapsed_seconds * 1000 * self.batch_size / useful if useful else 0.0,
        }
