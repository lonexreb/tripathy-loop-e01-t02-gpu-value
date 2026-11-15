# Experiment protocol

`research/PROTOCOL.md` is the frozen, authoritative evaluation contract. This document records the repository-level workflow around it.

Before a cycle:

1. Start from the current accepted lineage head.
2. Freeze the hypothesis, candidate budget, origin blocks, baseline, metric, and rejection condition.
3. Restrict every feature to information published and effective on or before its simulated origin.
4. Keep instrument classes separate and respect `DATA.md` source restrictions.

During evaluation:

1. Score each candidate once on development and once on checkpoint.
2. Accept only when checkpoint improvement has a 90% origin-block bootstrap interval below zero and development loss does not increase.
3. Never use the protected block to rescue or tune a candidate.

After a cycle:

1. Write accepted and rejected lineage artifacts.
2. Save metrics, hashes, failures, elapsed time, and budget use.
3. Update `artifacts/session-summary.json` and the existing Linear/Slack lane.
4. Run `bash research/reproduce.sh` before reporting success.
