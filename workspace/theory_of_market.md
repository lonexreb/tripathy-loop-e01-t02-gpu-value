# Theory of the market (append-only lab notebook)

Entries are appended by the RSI controller after every experiment (promoted, reverted or crashed).
Do not rewrite earlier entries; add corrections as new entries.

## Step 0 facts (from the pack and the starter kit)
- Scored set: 97 headline series of the starter catalog (as_of 2026-10-24): 80 AWS EC2 spot AZ/composite series
  (`hyperscaler_spot`, administered smoothed spot prices), 4 afhubbard hyperscaler_spot medians,
  12 gpurentalprices neocloud rate cards, 1 marketplace_ask median.
- Baseline: flat last value. Starter `drift` scores +0.2004 pooled at 2026-10-24 and +0.1734 over 15 weekly
  origins (verified by re-running starter/run_demo.sh).
- Rate-card medians move in steps (menu costs); AWS spot prices move most days (>80% of cells change).
- Do-not-splice rule: instrument classes are different instruments; never chain AWS spot with rate cards.
