# Trace → KernelForge analysis handoffs

One Python Codex SDK thread runs three prompts: hotspot analysis, source/input
recovery (with optional diagnostics), then YAML handoff. Analysis scripts remain
editable; no fixed kernel taxonomy or optimization loop is included.

Run in a dedicated experiment container with the actual SGLang/AITER sources.

```bash
python3 -m pip install -r /run/analyze_profiles_by_agent/requirements.txt
python3 /run/analyze_profiles_by_agent/run.py \
  --trace-dir /work/run_profiles/profile-001/traces \
  --output-dir /work/analysis-001 \
  --kernelforge /KernelForge \
  --model gpt-5.6-sol --effort max --max-minutes 40 \
  --allow-reprofile --context-file /work/experiment-context.md
```

`--trace-dir` also accepts one trace file. Plain JSON and gzip Chrome/Kineto
traces are supported. The context file is optional: use it to identify the
dedicated server, source trees, existing caches and approved profiling commands.
Adjacent benchmark/log files are discovered by the agent.

`--allow-reprofile` authorizes diagnostics on that instance; otherwise the prompt
limits work to existing evidence. `--max-reprofiles` is a prompt-level limit
(default 2). The runner enforces per-stage timeouts within the overall time
budget, reserves final packaging time and interrupts the SDK on timeout.

For a staged run, add `--stop-after 01_hotspots`. Continue using the same command
with `--resume` and without `--stop-after`. Resume uses the recorded thread ID and
next stage, requires the same trace/model/effort, reloads the saved prompts, and grants the new
invocation's time budget. The container's Codex session files must still exist.
An already complete run is revalidated without calling the model.

A run marked `partial` can still have a complete set of validated YAMLs: an
earlier investigation stage may have reached its cap. Once all three stages
are finished, resume only revalidates/publishes the existing drafts. To investigate
remaining evidence gaps, start a new output directory with the previous findings
in the context file.

The agent uses full filesystem access inside the dedicated container, with no
interactive approvals. Its prompt permits diagnostic instrumentation, and keeps
kernel optimization and unrelated processes outside the task. Run it on the
experiment instance you intend to let it modify.

## Gateway configuration

Use normal Codex provider configuration and API credential environment variables.
For this repository's container launcher, optionally set `LLM_GATEWAY_DIR` to a
host directory containing:

```text
config.toml       # Codex model_provider/model_providers configuration
credentials.json  # environment-variable-name → credential value
```

`run/start_container.sh` mounts the directory at `/llm_gateway` and the config at
`/root/.codex/config.toml`, read-only. The runner reads credentials.json if present
(or `--credentials-file PATH`), passes the environment to the local runtime and
redacts those values from its event log. Credential values never belong in the
context file, prompts, command line or experiment record. Keep that directory
private and outside versioned files; do not mount the whole personal Codex home.

## Output

```text
run.json             # model/SDK, thread, stages, status, task results
analysis.md          # analysis and limitations
candidates.yaml      # every candidate and task/merge relation
drafts/*.yaml        # agent-authored tasks before publication
tasks/*.yaml         # tasks published only after validation
contract.json        # schema used by this run
prompts.json         # frozen prompts used when resuming this run
validation.json      # validation errors or task summary
evidence/            # measurements, source excerpts, probe logs, diagnostic traces
scratch/             # small scripts the agent can modify
events.jsonl         # SDK tool/progress events
```

`ready_for_handoff` means enough evidence for KernelForge to prepare a driver;
this tool does not claim the kernel is optimized or the driver passes. Other
states are `needs_evidence` and `not_recommended`. KernelForge does not enforce
this custom readiness field: a downstream submission script must select tasks.

Each candidate has `id`, `name`, and `task: tasks/<id>.yaml`, or `merged_into`.
Each handoff follows `handoff.schema.json`; custom analysis fields live under
`x-handoff`. Unknown values remain explicit. Independent benchmark baselines are
not inferred from trace kernel averages.

Validate again in the source container (source paths are container paths):

```bash
python3 /run/analyze_profiles_by_agent/tools/validate_handoff.py \
  /work/analysis-001 --kernelforge /KernelForge
```

The validator checks structure, readiness requirements, evidence paths/references,
the actual KernelForge backend registry and candidate coverage. It does not
certify the scientific correctness of a roofline estimate or source mapping.

Use the raw reader directly, or import `iter_events` from it:

```bash
python3 /run/analyze_profiles_by_agent/tools/trace_io.py rank0.trace.json.gz
python3 /run/analyze_profiles_by_agent/tools/trace_io.py rank0.trace.json.gz --category kernel --limit 3
```

Summary mode reads and validates the entire document. Sample mode stops after
the requested events; it does not validate the unread remainder. Truncated JSON
raises an error when fully consumed.

Run the small parser/contract test suite with
`python3 -m unittest discover -s /run/analyze_profiles_by_agent -p test_tools.py`.
The design and two-node experiment are recorded in
`docs/analyze_profiles_by_agent_design.md` and `.record/20260914-kernel-analyze-poc/`.
