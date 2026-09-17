Stage 2: resolve actual source and input semantics, updating the stage-1 analysis.
Read your current analysis.md/candidates.yaml and spend the available time on the
highest-impact uncertainties. Do not just repeat the kernel-name ranking.

For each candidate, trace device symbol/code object/JIT cache to implementation,
wrapper, SGLang callsite and actual runtime dispatch. Verify the installed source
version. A generic main_kernel or mfma prefix is not an implementation identity.
Preserve evidence with path, revision and function/line locators.

Recover representative per-invocation inputs. Consider graph padding, local DP
batch versus gathered tokens, speculative verify versus draft, expert token
distribution, quantization scales/packing, KV lengths/page tables/sparse indices,
and communication topology. Inspect cache IR and source metadata when useful.
Record exact knowledge or derivations and remaining uncertainty for each input.

If reprofiling is authorized and would answer a concrete question, state that
question in analysis.md, then use existing experiment scripts or short probes.
Add stack/shape/instrumentation only as needed. Python hooks may need graph
capture rather than replay. Check effective profiler settings; separate all
diagnostic timings from baseline. Serialize GPU measurements on this instance.
Save exact commands, logs, instrumentation diffs and configuration changes.

You may write a small original-kernel probe to establish input behavior/counters;
you do not owe a production driver or a full correctness benchmark suite. Do not
change compute algorithms or start an optimization search.

Refine roofline using recovered semantics: explicit FLOPs, memory bytes and
memory level, elapsed-time source, attainable-versus-specification reference,
and assumptions. If counters are unavailable, use an honest estimate or mark
unavailable. Do not apply an HBM roofline to communication waiting time.

Update analysis.md and candidate priorities/boundaries. Save enough evidence for
stage 3 to produce every task without rediscovering source. Unresolved candidates
remain visible with actionable missing information. End the turn when this
investigation is complete, leaving final YAML packaging to stage 3.
