# Architecture

The active system has three intentionally separate layers:

- Immutable contract: the local `gpu-price-pack-v0.1.1-handover/` and its official starter scorer/baselines. It is ignored by Git and never edited.
- Fixed evaluator: `research/PROTOCOL.md`, `origins.json`, `adapted_series.txt`, `paired.py`, and `scientist/evaluate.py`. Candidate code cannot modify these files during a cycle.
- Evolvable scientist: `research/scientist/forecaster.py` and the versioned improvers that propose bounded candidates.

`research/run_baselines.py`, `run_cycle1.py`, and `reproduce.sh` orchestrate the layers. Machine-readable decisions live in `research/lineage/`; generated bulk outputs live in ignored `research/out/`.

GitHub records code and review evidence. Linear records ownership and next actions. Slack broadcasts concise, linked results. External adapters do not participate in scoring unless their as-of evidence is frozen and admissible under the protocol.
