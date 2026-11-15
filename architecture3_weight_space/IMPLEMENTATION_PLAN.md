# Architecture 3 controlling implementation plan

This file is the controlling plan for the Architecture 3 workspace. The longer
Tinker-oriented plan supplied separately is advisory only: this implementation
borrows its reproducibility and leakage safeguards but does not adopt Tinker,
Qwen3.8-27B, reinforcement learning, or its larger experimental matrix.

## Frozen architecture

The actor is the pinned `Qwen/Qwen3.5-0.8B` checkpoint. Modal hosts actor
inference, LoRA training, restricted numerical execution, and adapter storage.
OpenAI is the fixed teacher, Exa supplies historical research, past.dev supplies
scoped research memory, and the repository's unchanged scorer is the only
forecasting evaluator. Only LoRA adapter parameters may be updated.

No model, provider, target population, origin block, scorer, acceptance rule,
resource ceiling, or remote-action gate may change silently.

### Approved hardware amendment — 2026-11-15

After the L4 sanity and G0 episode passed, repeated L4 queue delays blocked the
remaining run. The user explicitly approved `H100` with `B200` fallback for
Phase 5 training and Phase 6 adapter inference. The model, LoRA configuration,
data, evaluator, admission rule, and resource ceilings are unchanged. The exact
authorization and reason are frozen in
`config/hardware_amendment_2026-11-15.json`.

## Phases and gates

1. **Scope and preflight:** hash the controlling configuration, frozen protocol,
   target/origin manifests, evaluator, dependency lock, and pack manifests.
2. **Typed contracts and journal:** reject malformed actions, evidence, memory,
   forecasts, examples, checkpoints, and decisions; preserve every event in a
   per-worker hash-chained JSONL shard.
3. **Information boundary:** export origin-eligible actor views while keeping
   protected truth inside the evaluator boundary.
4. **Evidence and memory:** capture Exa results and past.dev ingests/recalls with
   cutoff checks, arm/generation isolation, provenance, and local replay copies.
5. **Frozen G0 episode:** run one pinned actor episode through forecast commitment
   and official scoring before producing training data.
6. **Verified teaching:** separate teacher diagnosis from correction construction,
   re-execute corrected actions, and compile only verified assistant targets.
7. **LoRA sanity and training:** after explicit approval, prove one Modal update,
   base-weight immutability, adapter save/reload, and then run one bounded training
   candidate with pre-approved hyperparameters.
8. **Admission and recursion:** apply the existing frozen gate honestly. Only an
   admitted child may become active or propose an executed successor plan.
9. **Report and replay:** regenerate metrics and reports from frozen numerical
   artifacts, preserve failures, and distinguish replay from fresh discovery.

Remote provider calls, model download, Modal GPU use, training, Git commits, and
Git pushes remain separate user-approval gates.

## Execution outcome — 2026-11-15

The authorized pipeline completed through reporting. The pinned actor passed
model sanity and G0 evaluation; one verified example produced immutable LoRA
checkpoint `g1-6ccddfe047031e9c` on an NVIDIA H100 80GB HBM3. Base weights were
unchanged and adapter save/reload parity was exact. The frozen admission gate
rejected the child because it was worse than `F_0` on both development and
checkpoint. In accordance with the plan, protected truth was not scored and the
run does not establish recursive self-improvement.

## Reproduction standard

- The source pack remains unchanged.
- Configurations, inputs, outputs, and checkpoints are content-addressed.
- Service responses are frozen locally with timestamps and eligibility decisions.
- Secrets never enter manifests, events, prompts, or tracked files.
- Empty or unexecuted results are reported as `not_run`, never as success.
- Numerical replay must not require OpenAI, Exa, past.dev, or Modal calls.
