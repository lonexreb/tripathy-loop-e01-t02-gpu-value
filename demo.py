#!/usr/bin/env python3
"""Two-minute demo for the judges. Reads only files committed to this repository.

    python3 demo.py            # the story, with every number read from the recorded evidence
    bash research/reproduce.sh # rebuilds the submission from the data pack (needs the pack)

No GPU, no network, no model download, standard library only.
"""

import csv
import glob
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINEAGE = ROOT / "research" / "lineage"
ARCH3 = ROOT / "architecture3_weight_space" / "artifacts"


def load(path):
    return json.loads(Path(path).read_text())


def say(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def gate(paired):
    low, high = paired["bootstrap_90ci_mean_diff"]
    return f"{paired['overall']['mean_diff']:+.6f}  [{low:+.6f}, {high:+.6f}]"


say("1. WHAT WE SUBMIT")
submission = ROOT / "submission" / "submission_head_2026-11-06.csv"
recorded = (ROOT / "submission" / "SHA256SUMS").read_text().split()[0]
actual = hashlib.sha256(submission.read_bytes()).hexdigest()
rows = list(csv.DictReader(submission.open()))
print(f"{submission.name}: {len(rows)} forecasts, origin 2026-11-06, horizons 1/3/7 days")
print(f"SHA-256 matches the recorded hash: {actual == recorded}")
print("Seven-day bands, USD per GPU-hour (q10 / q50 / q90):")
for row in rows:
    if row["horizon_days"] == "7" and ".H100_" in row["series_id"] and "gpurentalprices" in row["series_id"]:
        q10, q50, q90 = (float(row[k]) for k in ("q10", "q50", "q90"))
        width = (q90 - q10) / q50
        print(f"  {row['series_id'].split('.', 1)[1]:<48} {q10:.2f} / {q50:.2f} / {q90:.2f}   width {width:.0%}")
print("-> The marketplace ask needs a far wider band than the rate cards.")

say("2. ONE FROZEN GATE, EVERY ATTEMPT  (checkpoint pinball difference vs parent; negative is better)")
print("Rule, fixed before any comparison: accept only if the 90% interval is entirely below zero.\n")
table = [
    ("Code: width from last 45 days (F_1)", "F_1_c2"),
    ("Code: repricing mixture", "F_1_c1"),
    ("Weights: one LoRA step", "F_arch3_g1-6ccddfe047031e9c"),
    ("Weights: direct edit, by hand", "F_arch3_wsp1-866f2dc0fc30a2df"),
]
for label, name in table:
    entry = load(LINEAGE / f"{name}.json")
    print(f"  {label:<38} {gate(entry['paired_vs_parent']['checkpoint'])}   {entry['status'].upper()}")
loop_gated = []
outcomes = Counter()
for path in glob.glob(str(LINEAGE / "F_arch3_loop-*.json")):
    for attempt in load(path)["attempts"]:
        outcomes[re.sub(r"duplicate_of_.*", "duplicate", attempt["outcome"])] += 1
        if "checkpoint_vs_parent" in attempt:
            loop_gated.append(attempt["checkpoint_vs_parent"])
best = min(loop_gated, key=lambda c: c["diff"])
print(
    f"  {'Weights: autonomous loop, 26 cycles':<38} {len(loop_gated)} candidates gated, "
    f"best {best['diff']:+.6f} [{best['ci90'][0]:+.6f}, {best['ci90'][1]:+.6f}]   ALL REJECTED"
)
print("-> One small gain admitted. Nothing in weight space passed.")

say("3. DIRECT WEIGHT EDITS WORK MECHANICALLY")
edit = load(LINEAGE / "F_arch3_wsp1-866f2dc0fc30a2df.json")["weight_edit"]
print(
    f"Hand edit: {edit['changed_weights']:,} weights in one tensor "
    f"({edit['fraction_of_model']:.4%} of the model), {edit['gradient_steps']} gradient steps."
)
print("It switched the actor's choice on every prompt, including held-out ones.")
print(f"\nAutonomous loop, {sum(outcomes.values())} proposals from a headless LLM editor, no human input:")
for outcome, count in outcomes.most_common():
    print(f"  {count:>3}  {outcome.replace('_', ' ')}")
protected = load(ARCH3 / "wsp_loop" / "protected_result.json")
print(
    "\nRecursion test (editor with its own history vs without, same start, protected block): "
    f"difference {protected['i1_head_vs_i0_head']['diff']}, supported = {protected['recursion_claim_supported']}"
)

say("4. THE ORIGINAL AUTHORS' RECIPE, ON OUR TASK")
diagnose = load(ARCH3 / "wsp_author" / "diagnose.json")["arms"]
sampled = load(ARCH3 / "wsp_author" / "diagnose_sampled.json")
print("Failure: with thinking on, the actor never finishes. Items hitting the token limit, of 60:")
print(f"  unedited model, greedy                      {diagnose['base_thinking']['classes'].get('token_limit_in_thinking', 0)}")
print(f"  released third-party edit, greedy           {diagnose['author_thinking']['classes'].get('token_limit_in_thinking', 0)}")
print(f"  unedited, the release's sampling settings   {sampled['base_thinking_sampled']['classes'].get('token_limit_in_thinking', 0)}")
print(f"  released edit, its own sampling settings    {sampled['author_thinking_sampled']['classes'].get('token_limit_in_thinking', 0)}")
print("\nOur own edit of the looping direction (layers 15 and 19, the same two the authors found):")
for strength, arm in load(ARCH3 / "wsp_author" / "work" / "step3.json").items():
    stuck = arm["classes"].get("token_limit_in_thinking", 0)
    print(f"  strength {strength}: {stuck} of {arm['n']} still hit the limit")
tail = json.loads((ARCH3 / "wsp_author" / "work" / "B_s4.jsonl").read_text().splitlines()[0])["text"]
print("\nWhat the edited model writes instead of looping verbatim:")
for line in [x.strip() for x in tail.split("\n") if "cannot" in x or "Wait" in x][:3]:
    print(f"  | {line[:150]}")
off = diagnose["base_no_thinking"]["pinball"]
print(
    f"\nWhen it does answer (thinking off): pinball {off['actor']:.4f} "
    f"vs {off['last_value']:.4f} for simply repeating the last price."
)
print("-> There is no forecasting ability behind the obstacle, so removing it releases nothing.")

say("5. BOTTOM LINE")
print("We did not build a recursively self-improving forecaster, and the evidence above shows why.")
print("  Supported:     one small calibration gain over drift (F_1), submitted.")
print("  Not supported: lasting text-space gains, LoRA gains, direct-edit gains, an improver")
print("                 that improves from its own history, ability released by removing a loop.")
print("  For a lessor or lender: size short-horizon bands from the last six weeks, not full history,")
print("  and hold about three times the relative margin on marketplace asks as on rate cards.")
print("\nFull write-up: FINDINGS.md    Rebuild the submission: bash research/reproduce.sh")
