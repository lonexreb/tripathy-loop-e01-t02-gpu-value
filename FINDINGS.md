# Findings — Track 2, Direction A: a self-improving forecaster of GPU rental prices

Tripathy Loop E01 · Pacific "AI Scientist for GPU Value" · written 2026-11-15

## 1. Question

Can a system improve its own short-horizon forecasts of posted GPU rental prices, on origins it had not seen when it changed itself — and does it matter **where** the improvement is written: in the forecasting code, in the model's instructions, or in the model's weights?

## 2. What we forecast

Seven headline H100/A100 posted-rental series with daily truth (`research/adapted_series.txt`), each a per-collector median in USD per GPU-hour. Three instrument classes, never pooled or spliced:

| Instrument class | Collector | Series |
|---|---|---|
| Hyperscaler spot | afhubbard | A100 40GB, A100 SXM 80GB |
| Marketplace ask | gpurentalprices | H100 80GB |
| Neocloud rate card | gpurentalprices | A100 40GB, A100 SXM 80GB, H100 80GB, H100 SXM 80GB |

Horizons 1, 3 and 7 days; quantiles q10, q50, q90. All prices are posted, not transacted. Every other headline series in the submission carries the disclosed starter `drift` baseline.

## 3. Evaluation (frozen before any comparison was run)

`research/PROTOCOL.md`, frozen 2026-11-14:

- **Blocks of daily origins:** development 09-01..09-30 (the improver may read errors), checkpoint 10-01..10-15 (acceptance test), protected 10-16..10-30 (scored once), official 11-06.
- **Scorer:** the starter `scorer.py`, unchanged. Mean pinball loss over q10/q50/q90.
- **Acceptance rule:** a candidate is accepted only if its paired mean pinball difference against its parent is negative on checkpoint with the 90% block-bootstrap interval excluding zero, and not positive on development. Rejected candidates are kept.
- **Leakage:** every input is cut off at the origin; frozen-cutoff mode refuses anything later (verified: exit code 2).

Baseline F_0 is starter `drift`:

| Block | Cells | Skill vs last value | 80% interval coverage | Pinball diff vs last value (90% CI) |
|---|---:|---:|---:|---|
| Development | 626 | +0.163 | 0.588 | −0.004581 [−0.005881, −0.003372] |
| Checkpoint | 290 | +0.089 | 0.666 | −0.002892 [−0.005056, −0.001043] |
| Protected | 255 | +0.196 | 0.663 | −0.008192 [−0.011236, −0.005372] |

Drift beats last value everywhere, but its intervals cover 59–67% of outcomes against a target of 80%. **Interval width, not the median, is the weakness.**

## 4. Results: the same gate, six attempts at self-improvement

Each row is one candidate change, scored once on checkpoint against its parent. Negative is better.

| Where the change lives | Change | Checkpoint diff (90% CI) | Decision |
|---|---|---|---|
| Code | `conformal_trailing`, 45-day window: interval width from recent realised moves | −0.000957 [−0.001539, −0.000315] | **Accepted → F_1 (submitted)** |
| Code | `repricing_mixture`: width from repricing frequency | +0.000124 [−0.000141, +0.000395] | Rejected |
| Instructions (text) | LLM judge rewrites the 0.8B actor's prompt and skill file | −0.002143 [−0.004295, −0.000455] vs its frozen control | Accepted on checkpoint; **ties drift on protected** (+0.000003) |
| Weights, RL | One GRPO-LoRA step on the 0.8B actor | −0.000307 [−0.000854, +0.000122] | Rejected |
| Weights, supervised | One LoRA step on a teacher-confirmed example | +0.002892 [+0.001043, +0.005056] | Rejected: behaviour did not change |
| Weights, programmed | Direct gradient-free edit of 58,293 MLP weights (section 5) | +0.000037 [−0.001051, +0.001053] | Rejected: behaviour changed, ties drift, coverage 0.821 |

F_1 lifts checkpoint skill against last value from +0.089 to +0.119 and coverage from 0.666 to 0.714.

**What the system learned across iterations**

1. *The error is in the width.* The first improver read development errors: 80% intervals covered 58.8% overall and only 44.3% at seven days. Both of its candidates changed width, not the median.
2. *Recent volatility beats full-history volatility.* Calibrating width on the last 45 days of realised moves was accepted; modelling repricing frequency was not.
3. *A text-space gain can vanish out of sample.* The rewritten instructions passed checkpoint and then added nothing on the protected block. One accepted checkpoint result is not a durable gain.
4. *A weight update is only as good as its label.* The supervised LoRA step trained on the model's own answer, confirmed by a teacher on a single origin. The model emitted the identical action before and after. That action compiles to a zero-width interval (coverage 0.375) and was correctly rejected.
5. *The same action space held a better answer all along.* A development-only sweep shows `repricing_interval` at width scales up to 1.0 beats drift by 0.0013–0.0018; the actor never chose it. That is the obstacle section 5 goes after.

## 5. Weight-space programming on the forecasting actor

**Technique.** Weight-space programming (Takehi, Teoh, Zhang and Khattab, 2 November 2026) treats weights as a program: an LLM reads a network's weights and activations and writes new weight values directly, with no gradient descent. Their released example edits the same base model we use, `Qwen/Qwen3.5-0.8B` at the same revision, to stop it looping on math problems. We applied the idea to one decision of our forecasting actor. Editor: Claude with a local Python kernel over the weights. No optimizer, no backward pass, no remote compute.

**The failure.** Asked to choose a forecasting patch, the unedited actor always emits `drift, 60, 1`, which our compiler turns into a zero-width interval. The supervised LoRA step did not change that.

**Reading the weights (24 development-origin prompts).**
- At the token where the method is chosen, the model prefers `dr` over `repr` by 1.61 logits, almost identically on every origin (range 1.53–1.66). The choice does not depend on the price data.
- Swapping the order of the two allowed actions in the prompt does not change the choice. It is not a position effect, and this prompt-space fix does not work.
- Projecting each MLP's output onto the direction that separates the two tokens: layers 14, 16 and 21 write toward `drift` (−0.55, −0.61, −0.52); the last MLP, layer 23, writes toward `repricing_interval` (+2.59) and is outvoted.
- The read is exact: the direction's dot product with the final residual reproduces the logit gap to four decimals.

**Writing the edit.** A rank-one, closed-form update of layer 23's `down_proj`: `W' = W + s · d · kᵀ`, with `d` the read direction, `k` the activation pattern that distinguishes the decision position from the rest of the sequence, restricted to 32 columns.

| Strength (× predicted) | Mean logit gap | Prompts choosing `repricing_interval` |
|---:|---:|---:|
| 0 | −1.606 | 0 / 24 |
| 0.5 | −0.803 | 0 / 24 |
| 1.0 | −0.000 | 13 / 24 |
| 1.25 | +0.402 | 24 / 24 |

The strength computed from the read closes the gap to zero, as predicted.

**A second obstacle.** Once the method flipped, the model began quoting its numbers (`"window_days":"60"`), which the frozen patch contract rejects, and on some prompts copied the template text instead of a number. The cause was one token: after a numeric key it chose colon-quote over colon. A second edit of the same form on the same tensor (32 more columns) fixed it; all 24 prompts then emitted a valid patch. This amendment was recorded before any checkpoint score.

**The frozen edit.** 58,293 weights in one tensor (57 columns), 0.0077% of the model, 0.14% relative change to that tensor, zero gradient steps. Stored as sparse indices and values with SHA-256 of the base and edited tensor.

**Checks on inputs not used to choose the edit.**

| Check | Result |
|---|---|
| 6 held-out development origins, edited model | 6 / 6 valid `repricing_interval` patches (unedited: 0 / 6) |
| Random-direction control, same columns, same norm | 0 / 6 flipped; output identical to the unedited model |
| Unrelated prompts, next-token top-1 agreement with the unedited model | 99.4% (mean KL 0.0001 nats) |
| Unrelated prompts, 48-token greedy generations identical | 5 of 8 |
| Edit rebuilt from the recipe equals the stored edit | Yes |
| Same patch scored with no model | Identical forecasts |

**Through the gate.** The edited actor emitted `repricing_interval, window 60, width scale 1.5`. The width is the model's own choice; we did not steer it.

| Block | Pinball diff vs F_0 (90% CI) | Coverage |
|---|---|---:|
| Development | +0.000652 [+0.000130, +0.001164] | 0.859 |
| Checkpoint | +0.000037 [−0.001051, +0.001053] | 0.821 |

**Rejected**, on both conditions. The protected block was not scored and the submission stays F_1.

**What this teaches.**
- The edit did what gradient training had not: it changed the actor's behaviour, moving it from 0.0029 worse than drift on checkpoint to a tie, and from 50% to 82% interval coverage there, above F_0 (0.666) and F_1 (0.714) and at the 80% target.
- It did not produce a better forecaster, because the model's own width (1.5) is too wide. On development, widths of 0.75 to 1.0 beat drift by 0.0013–0.0018; the model did not pick them and nothing in its weights was edited to make it.
- This matches the original authors' limit: an edit can remove an obstacle, but cannot add an ability the model lacks. In everything we observed, the 0.8B actor's choice did not respond to the price data, so there is no data-dependent skill for an edit to release.
- Honest caveats: one tensor, one editor, one run; the edit sits in the last MLP, close to the readout; three of eight unrelated generations changed somewhere in 48 tokens; the target method came from a development sweep of 19 actions.

**Replay** (local, no remote calls; `architecture3_weight_space/`):

```bash
uv run python -m tripathy_arch3.wsp_gate prompts
.venv-wsp/bin/python src/tripathy_arch3/wsp.py diagnose
.venv-wsp/bin/python src/tripathy_arch3/wsp.py edit --layer model.layers.23.mlp.down_proj
.venv-wsp/bin/python src/tripathy_arch3/wsp.py validate
.venv-wsp/bin/python src/tripathy_arch3/wsp.py actor
uv run python -m tripathy_arch3.wsp_gate gate
```

Rules were fixed in `config/wsp_preregistration_2026-11-15.json` before any checkpoint score. Sources: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), [released checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming), [Nusrat and Nusrat, arXiv 2611.05341](https://arxiv.org/abs/2611.05341).

## 6. An autonomous weight-programming loop, and the recursion test

Section 5 was one edit made by hand that wrote a constant. This section is the unattended version: one command runs several cycles in which an LLM editor decides each edit, a deterministic kernel writes it, and the frozen gate decides. No human input during a run.

```
brief (development results, actor's choices, notebook)
   → EDITOR   headless Claude (Sonnet), no tools, structured reply:
              {hypothesis, class of series, band to promote}
   → KERNEL   closed-form conditional edit, dose-response, random-direction control
   → GATE     official scorer → paired bootstrap → acceptance rule vs the current actor
   → accept: stack the edit / reject: revert → lineage entry + notebook
```

**The actor was redesigned so an edit can install a rule.** It decides per (origin, series) from that series' own as-of summary: a `narrow`, `medium` or `wide` band (width scale 0.5, 1.0, 2.0). An edit promotes one band for a class of series only, keyed on what distinguishes that class's activations from the others.

**Can a rule be programmed into the weights? Sometimes.**

| Class the editor asked for | Class on target | Other series moved | Programmable |
|---|---:|---:|---|
| Marketplace ask → narrow | 100% | 0% | Yes |
| Hyperscaler spot → wide | 100% | 0% | Yes |
| Daily move ≥ 15% of price → narrow | 100% | 6% | Yes |
| Unchanged days < 50% → narrow | 100% | 60% | No |
| GPU model H100 80GB → narrow | 100% | 60% | No |
| GPU model A100 SXM 80GB → wide | 81% | 21% | No |
| GPU model A100 40GB → wide | 100% | 32% | No |

Rates are at the chosen strength for programmable edits and at the strongest strength tried otherwise. Classes that follow the instrument type are cleanly addressable in the last MLP, with the same rates on six held-out origins. Classes that cut across instrument types (a GPU model sold as both spot and rate card) are not: the edit spills onto other series. On the real weights each programmable edit changes about 32,700 weights and reproduces the predicted logits to 0.00001; agreement with the unedited model on unrelated prompts is 97.6–100%.

**The loop ran twice from the same unedited actor, three cycles each** (rules fixed beforehand in `config/wsp_loop_preregistration_2026-11-15.json`). I_0 started with an empty notebook; I_1 started with the notebook I_0 produced. All nine editor proposals came from the headless editor; the scripted fallback was never used.

| Run | Cycle | Proposal | Development vs parent (90% CI) | Checkpoint vs parent (90% CI) | Outcome |
|---|---:|---|---|---|---|
| I_0 | 1 | Marketplace ask → narrow | −0.000795 [−0.001266, −0.000297] | −0.000099 [−0.001086, +0.001173] | Rejected |
| I_0 | 2 | Hyperscaler spot → wide | −0.000438 [−0.000744, −0.000167] | +0.000258 [+0.000050, +0.000472] | Rejected |
| I_0 | 3 | Unchanged < 50% → narrow; then H100 80GB → narrow | — | — | Not programmable (both) |
| I_1 | 1 | A100 SXM 80GB → wide; then A100 40GB → wide | — | — | Not programmable (both) |
| I_1 | 2 | Daily move ≥ 10% of price → narrow | −0.000795 [−0.001266, −0.000297] | −0.000099 [−0.001086, +0.001173] | Rejected |
| I_1 | 3 | Daily move ≥ 0.1% → narrow (not programmable); then ≥ 15% → narrow | −0.000795 [−0.001266, −0.000297] | −0.000143 [−0.001145, +0.001139] | Rejected |

**No edit was admitted in either run.** Every edit that could be programmed improved the development block with the interval excluding zero, and none carried to checkpoint.

**Recursion test: not supported.** Both runs ended on the same unedited actor, so the protected-block difference between I_1's and I_0's final actors is exactly zero. That final actor against drift on the protected block: −0.000610 [−0.001891, +0.000607], coverage 0.663.

**What the loop learned.**
- *The method works; the forecasting gain does not transfer.* Programming a class rule into the weights is reliable for instrument-level classes. What fails is the same thing that failed for the text-space arm: a development gain that is not there on later origins. Two weeks of development data is too little to tell which band a series needs next.
- *The notebook changed the editor's behaviour but did not make it better.* I_1 avoided repeating I_0's exact proposals and tried new classes. But in cycle 2 it re-proposed I_0's rejected marketplace edit under a different feature test (daily move ≥ 10% selects exactly the marketplace series), producing the identical forecast and the identical rejection. The notebook recorded proposals by their wording, not by the decisions they select. That is a concrete defect in the improver, and the next version should key the notebook on the selected decision set.
- *Descriptive, not an admission:* the redesigned actor's unedited default (medium band everywhere) was the loop's starting point, not a gated candidate. On checkpoint it is −0.001109 [−0.001826, −0.000424] against drift with coverage 0.755, and ties F_1 (−0.000152 [−0.000995, +0.000618]). The submission stays F_1.
- *For someone pricing these instruments:* on development, the volatile marketplace ask wanted a **narrower** band than one typical daily move and the near-constant hyperscaler spot series wanted a **wider** one — rare jumps in quiet series cost more than routine churn in noisy ones. Neither held on the following two weeks, so treat both as hypotheses, not rules.

**Cost and replay.** Each run took a few minutes on a laptop: no GPU rental, nine editor calls costing $0.51 in total. Re-running an arm replays the stored editor replies, so the numbers reproduce with no LLM call:

```bash
cd architecture3_weight_space
.venv-wsp/bin/python -m tripathy_arch3.wsp_loop run --arm i0 --cycles 3
.venv-wsp/bin/python -m tripathy_arch3.wsp_loop run --arm i1 --cycles 3
.venv-wsp/bin/python -m tripathy_arch3.wsp_loop protected
```

**Disclosures.** The 15 checkpoint origins had already been scored for at least seven earlier candidates (the pre-registration file says six; it undercounts the text-and-weight arm); the loop scored five more forecasts on them (the unedited start and four candidates, one of them a duplicate). All edits sit in the last MLP, close to the readout. The actor's decision is read as the highest of three band-token logits, not free-form text.

## 7. What a lessor, lender or capacity buyer can take from this

- **Rate cards move in steps.** In the 60 days to 30 September, five of the six neocloud and hyperscaler-spot medians were unchanged on 75–93% of days. The sixth, the H100 80GB rate card, moved on 68% of days but by under one cent at the 80th percentile, on a $3.25 price. For a one-day horizon the best point forecast is today's price; the forecasting value is entirely in how wide a band to hold around it.
- **Marketplace asks are a different instrument.** The H100 marketplace ask changed on 88% of days, with an 80th-percentile daily move of $0.27 on a $1.57 price. Its official 7-day band is $1.47–$2.39 around $1.875, against $3.26–$3.87 around $3.55 for the H100 SXM rate card. Anyone marking collateral or pricing short leases off marketplace asks needs roughly three times the relative margin.
- **Uncertainty grows faster than the starter baseline assumes.** On the development block, drift's seven-day intervals cover 44% of outcomes. A reserve or haircut sized from full-history volatility is too thin; size it from the last six weeks.
- **Stale intervals are the cheap fix.** The accepted improvement changed nothing about the median and came entirely from recalibrating width on recent data.

## 8. Limits

- Seven series, 45 development and checkpoint origins, daily origins that overlap: intervals are block-bootstrapped over origins and are wide.
- All prices are posted, not paid.
- **No recursive-self-improvement claim.** The common-start comparison of an improver with and without its own history was run for the weight-programming loop (section 6) and came out at exactly zero: neither improver produced an admitted change.
- Official truth for 7–13 November is withheld, so nothing here is a claim about the scored window.
- The 2–3 year resale and rental view is not addressed.

## 9. Reproduce

```bash
bash research/reproduce.sh
```

Rebuilds the baseline and Cycle 1 from the unchanged pack, validates the submission with the official scorer, runs the leakage refusal check, and prints SHA-256 hashes. The submission file is `submission/submission_head_2026-11-06.csv` (111 rows: 37 headline series × 3 horizons; the 7 adapted series from F_1, the rest from drift). The pack's `ATTRIBUTION.md` travels with this repository.

The weight-space arms live in `architecture3_weight_space/` and replay locally with no remote calls: see sections 5 and 6 for commands.
