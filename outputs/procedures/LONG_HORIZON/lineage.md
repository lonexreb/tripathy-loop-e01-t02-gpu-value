# Improvement-procedure lineage (I_g)

| version | parent | status | component | change | frozen | candidate | admitted |
|---|---|---|---|---|---|---|---|
| v000 | - | superseded | - | seed | - | - | - |
| v001 | v000 | candidate | research_policy | Add one mechanical submission gate to the Method section: the exact candidate code submitted to Holdout must have comple | - | - | - |
| v002 | v000 | active | research_policy | Add one mechanical submission gate to the Method section: the exact candidate code committed to Holdout must have comple | 0.002779 | 0.005975 | True |

## Research episodes

| episode | procedure | tools used | model class | locked score | Holdout shot |
|---|---|---|---|---|---|
| g001_frozen | v000 | inspect_current_statex1, recall_memoryx1, search_exa_signalsx1, analyze_discoveryx1, write_candidatex1, dry_run_discoveryx3, save_variantx2, edit_candidatex12, backtest_candidatex1, view_candidatex3, show_candidate_diffx1, commit_forecastx1 | regime_switching | 0.0 | -  |
| g002_frozen | v000 | recall_memoryx1, inspect_current_statex1, view_candidatex1, profile_packx1, search_exa_signalsx2, analyze_discoveryx3, write_candidatex1, dry_run_discoveryx3, edit_candidatex3, backtest_candidatex2, commit_forecastx1 | hierarchical_pooling | 0.022505 | PROMOTED 0.288617018988914 |
| g003_frozen | v000 | inspect_current_statex2, recall_memoryx1, search_exa_signalsx1, analyze_discoveryx6, write_candidatex1, dry_run_discoveryx2, edit_candidatex1, backtest_candidatex1 | - | 0.0 | -  |
| g003_candidate | v002 | inspect_current_statex1, recall_memoryx1, analyze_discoveryx4, write_candidatex6, dry_run_discoveryx3, save_variantx3, load_variantx2, backtest_candidatex3, commit_forecastx1 | hierarchical_pooling | 0.004049 | -  |
| g003_candidate_r2 | v002 | inspect_current_statex1, recall_memoryx1, search_exa_signalsx1, analyze_discoveryx7, edit_candidatex6, view_candidatex1, dry_run_discoveryx2, backtest_candidatex2, commit_forecastx1 | hierarchical_pooling | 0.009128 | REVERTED 0.24750342396533664 |
| g003_frozen_r2 | v000 | inspect_current_statex1, recall_memoryx1, search_exa_signalsx1, analyze_discoveryx7, edit_candidatex9, dry_run_discoveryx3, view_candidatex2, backtest_candidatex4, commit_forecastx1 | hierarchical_pooling | 0.004804 | -  |
| g003_frozen_r3 | v000 | inspect_current_statex1, recall_memoryx1, search_exa_signalsx1, analyze_discoveryx6, write_candidatex1, dry_run_discoveryx2, edit_candidatex1, backtest_candidatex1, commit_forecastx1 | hierarchical_pooling | 0.003532 | -  |
| g003_candidate_r3 | v002 | inspect_current_statex1, recall_memoryx1, search_exa_signalsx1, write_candidatex1, dry_run_discoveryx2, backtest_candidatex4, analyze_discoveryx4, edit_candidatex3, commit_forecastx1 | hierarchical_pooling | 0.004748 | -  |

## Scientist diagnoses (what the model says limits it)

- **d003** (gen 3, v000): h180 dominates the loss (46% of champion Discovery loss; q90 breach 32.4%, mean bias -0.23): the drift is family-heterogeneous (p5/A100->H100 premium families ROSE 2023-24 while p4d fell ~-0.35 log), and the champion's single pooled 730-day family change-quantile set cannot represent both directions - its worst series are exactly the rising p5 AZs (coverage 0.52-0.67, share_of_total_loss up to 14%
  - data_quality_issues: 116 of 133 target series have NO history at Discovery origins (p4de/p5e/p5en/p6-b200/p6-b300 and most AZs start after 2024-09): at Holdout origins 2025-03+ these families appear with short histories, so per-series levers must guard min-pairs (>=30 change pairs) with pooled/last-value fallbacks or they will produce noise or crashes (cf. PACK_PLUS_EXA iter-5 ZeroDivisionError on short card series).;
  - data_needed: The 53 non-target AWS spot series (other AZs / p6) that exist in the source tick data but are absent from this stage's series_history: they would enlarge family cross-sections (better csmed, more pooled change pairs) and give early history for the p6 families that appear at Holdout; daily close, point-in-time truncated at origin.; Raw aws_spot_ticks (intraday) or n_obs-weighted daily statistics, t
  - tools_needed: None blocking: numpy/pandas suffice and runtime is trivial (champion runs 0.02-0.3 s/origin). Note: the 6-analysis budget for this episode is exhausted (6/6 used); the pseudo-origin harness above is reusable but needs a fresh budget next episode.; A per-series residual/breach diagnostic in the Discovery dry-run output (per-series q10/q90 breach rates, not just per-slice) would let us attribute the

## Capability requests (model-proposed; lab-served, redirected or queued)

| id | gen | kind | request | status | times | note |
|---|---|---|---|---|---|---|
| cr001 | 3 | diagnostic | Add per-series q10/q90 breach rates (not just per-slice) to the Discovery dry-run output, so breach attribution to insta | queued_for_maintainer | 1 | needs a lab change (scoring, partitions, network policy or new code): recorded as a spec for the maintainer; the lab nev |
