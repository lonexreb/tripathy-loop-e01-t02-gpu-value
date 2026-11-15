#!/usr/bin/env python3
"""Cycle 1: I_0 diagnoses F_0 on the development block, proposes <= 2 candidates, the fixed evaluator scores each once on
development and once on checkpoint, accepted/rejected artifacts are kept, the best accepted candidate becomes F_1, and the
official-origin submission is recomposed from the lineage head (adapted series) plus the drift fallback (everything else).
Requires research/run_baselines.py to have run (F_0 and its cells)."""
import json, os, subprocess, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, OUT, L = f"{ROOT}/research", f"{ROOT}/research/out", f"{ROOT}/research/lineage"
S = f"{ROOT}/gpu-price-pack-v0.1.1-handover/starter"
PY = [f"{ROOT}/.venv/bin/python", "-I", "-B"]
sys.path.insert(0, f"{R}/scientist")
import improver_i0, evaluate  # noqa: E402
OFFICIAL, IDS, TRUTH = "2026-11-06", f"@{R}/adapted_series.txt", f"{OUT}/series_daily.csv"


def main():
    parent = json.load(open(f"{L}/F_0.json"))
    blocks = json.load(open(f"{R}/origins.json"))["blocks"]
    i0 = improver_i0.artifact(ROOT)
    diag = improver_i0.diagnose(parent, OUT, ROOT)
    cands, fired = improver_i0.propose(diag)
    i0["cycle1"] = {"diagnosis": diag, "rules_fired": fired, "n_proposed": len(cands)}
    json.dump(i0, open(f"{L}/I_0.json", "w"), indent=2)
    print(f"I_0 diagnosis: coverage {diag['coverage_within']} (by h {diag['coverage_by_horizon']}), median_signal {diag['median_signal']:+.6f}; rules {fired}; {len(cands)} candidates")

    entries = [evaluate.evaluate(f"F_1_c{i+1}", c, parent, blocks, TRUTH, IDS, "I_0") for i, c in enumerate(cands)]
    for e in entries:
        print(f"{e['id']} {e['method']} {e['params']}: {e['status']} | {e['decision_reason']} | "
              f"dev skill {e['scores']['development']['skill_pooled']:+.4f} cov {e['scores']['development']['coverage']['within_q10_q90']:.3f} | "
              f"ckpt skill {e['scores']['checkpoint']['skill_pooled']:+.4f} cov {e['scores']['checkpoint']['coverage']['within_q10_q90']:.3f}")
    acc = sorted([e for e in entries if e["status"] == "accepted"], key=lambda e: e["paired_vs_parent"]["checkpoint"]["overall"]["mean_diff"])
    head = parent
    if acc:
        head = acc[0]
        for e in acc[1:]:
            e["status"] = "accepted_not_selected"; json.dump(e, open(f"{L}/{e['id']}.json", "w"), indent=2)
        head["selected_as"] = "F_1"; json.dump(head, open(f"{L}/{head['id']}.json", "w"), indent=2)

    # submission from the head: adapted series by the head forecaster, every other headline cell by the drift fallback
    base = pd.read_csv(f"{OUT}/submission_F0_drift_{OFFICIAL}.csv", dtype=str, keep_default_na=False)
    ids = [l.strip() for l in open(IDS[1:]) if l.strip() and not l.startswith("#")]
    if head is not parent:
        hf = f"{OUT}/official_{head['id']}.csv"
        subprocess.run([*PY, f"{R}/scientist/forecaster.py", "--history", f"{OUT}/history_{OFFICIAL}.csv", "--origin", OFFICIAL, "--frozen",
                        "--series-ids", IDS, "--method", head["method"], "--params", json.dumps(head["params"]), "--out", hf], check=True)
        base = pd.concat([base[~base.series_id.isin(ids)], pd.read_csv(hf, dtype=str, keep_default_na=False)])
    sub = f"{OUT}/submission_head_{OFFICIAL}.csv"; base.to_csv(sub, index=False)
    rc = subprocess.run([*PY, f"{S}/scorer.py", "--forecast", sub, "--truth", TRUTH, "--history", f"{OUT}/history_{OFFICIAL}.csv", "--frozen-cutoff", OFFICIAL,
                         "--catalog", f"{OUT}/catalog_{OFFICIAL}.csv", "--tier", "headline", "--horizons", "1,3,7", "--out", f"{OUT}/validate_submission_head.json"],
                        stdout=subprocess.DEVNULL).returncode
    v = json.load(open(f"{OUT}/validate_submission_head.json"))
    summary = {"cycle": 1, "improver": "I_0", "proposed": [e["id"] for e in entries], "statuses": {e["id"]: e["status"] for e in entries},
               "head": head["id"], "submission": os.path.relpath(sub, ROOT), "submission_validation": {"exit": rc, "status": v["status"], **v["forecast_validation"]}}
    json.dump(summary, open(f"{L}/cycle1.json", "w"), indent=2)
    print(f"head: {head['id']} | submission {os.path.basename(sub)} rows {len(base)} | frozen validation exit {rc} status {v['status']} problems {v['forecast_validation']['n_problems']}")


if __name__ == "__main__":
    main()
