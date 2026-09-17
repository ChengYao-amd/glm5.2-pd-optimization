Stage 1: identify worthwhile operators, representative inputs and preliminary
optimization space. Finish this stage by writing analysis.md and candidates.yaml.

1. Inspect the supplied capture and adjacent benchmark/logs. Identify workload,
   ranks, GPU, graph mode, phase markers, ramp-up or cold-path contamination, and
   missing metadata. Multiple capture directories are separate measurements.
2. Write/run a short streaming analysis. Summarize phase/rank-aware hotspots,
   counts and duration distributions; save quantitative evidence in evidence/.
   The supplied trace_io.py only reads raw events: choose your own aggregation.
3. Form operator candidates, including communication, vendor kernels, unknown
   symbols and fusion chains. Do not silently filter unknown names out.
4. Record any known/derived input shapes and provisional roofline. State what
   requires source inspection or further capture; do not require complete shapes
   to advance to stage 2.
5. Prioritize by likely contribution to the target e2e workload and plausible
   headroom. Report the denominator of all time shares and coverage calculations.

candidates.yaml format (choose descriptive stable ids; id equals task_id):

```yaml
candidates:
  - id: operator_family_001
    name: Human-readable operator description
    task: tasks/operator_family_001.yaml
```

A merged candidate instead has id, name, and merged_into: <another candidate id>.
Never delete a considered candidate silently. If there are no candidates, emit
an empty candidates list and a substantive no_candidates_reason.

Keep the work bounded. Source browsing is allowed, but reserve deep source and
input recovery for stage 2. End this turn after the two stage artifacts exist.
