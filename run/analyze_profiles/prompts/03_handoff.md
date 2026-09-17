Stage 3: write drafts/<id>.yaml for every unmerged candidate in candidates.yaml.
Read the supplied handoff.schema.json and KernelForge task-definition reference.
These are analysis handoffs; driver_owner is kernelforge. Do not launch profiling
or optimization now. Use established evidence and honestly record gaps.

Each task includes the standard KernelForge fields plus x-handoff:
- readiness: ready_for_handoff, needs_evidence, or not_recommended.
- inputs: [{name, knowledge: observed|derived|unknown, spec: {...}, evidence_refs: [E1]}].
  spec holds concrete dtype/shape/stride/layout/scalars and data semantics.
- measurements: [{name, value, unit, scope, evidence_refs: [E1]}]. Values are
  nonnegative numbers; units are us/ms/s/count/percent/bytes/flops/GB/s/TFLOP/s.
- roofline: {status: measured|estimated|unavailable, explanation: ..., evidence_refs: [...]}.
  Add explicit quantitative assumptions/calculations as extra fields when known.
- source: an object recording revisions/callsites/provenance, or null if unknown.
- evidence: [{id: E1, path: existing-file, locator: precise-location}].
- workload: phase/rank/graph/input distribution and original/diagnostic identity.
- missing_information: explicit remaining gaps, not TODO placeholders.

kernels_to_review entries require kernel, source_path (existing file or null),
bottleneck_hypothesis and tunable_surfaces. Include all kernels in the intended
operator boundary. Use only actual KernelForge registry backend names. Do not
fill baseline_wall_ms with a trace mean; omit it unless independently measured.
Set top-level status: pending. Define semantic correctness requirements for
indices, cache writes, multiple outputs and quantization, not an arbitrary SNR.

ready_for_handoff requires actual implementation/callsite, representative input
specifications, evidence and a useful optimization hypothesis. Derived shapes
are acceptable when justified and representative. Unresolved critical input or
source information means needs_evidence. not_recommended must explain why.

Run the supplied validator against --task-dir drafts. Fix all structural errors
without editing the validator/schema, weakening readiness, or inventing evidence.
All candidates must map to a draft or a valid merged_into chain. Update the final
summary in analysis.md and report readiness counts, key opportunities and gaps.
The runner alone publishes validated drafts to tasks/.
