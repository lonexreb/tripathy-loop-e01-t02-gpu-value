"""Passes 10-13: paired generation on fresh choose items; untouched vs our edit at 3 strengths."""
import sys, json, random, torch; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel as K
from tripathy_arch3 import wsp, wsp_practice as P
tok, model = wsp._load(); d = K.read(); base = K.base_weights(model)
items = [i for i in P.practice_set() if i["split"] == "choose"]
used = {(json.loads(l)["series_id"], json.loads(l)["origin"]) for l in open(P.OUT / "diagnose_base_thinking.jsonl")}
fresh = [i for i in items if (i["series_id"], i["origin"]) not in used]
B = random.Random(1).sample(fresh, 60)
edit = K.build(model, d, ref=19)
out = {}
for s in (0, 1, 2, 4):
    K.apply(model, edit, s, base)
    n = K.charge("generate", f"sample B (60 fresh choose items), greedy 1024, our edit strength {s}")
    recs = P.generate(tok, model, B, thinking=True, max_new_tokens=1024, batch_size=30)
    out[s] = P.score(recs); print("pass", n, "strength", s, json.dumps(out[s]), flush=True)
    with open(K.WORK / f"B_s{s}.jsonl", "w") as h:
        for r in recs: h.write(json.dumps({"series_id": r["item"]["series_id"], "origin": r["item"]["origin"], "class": r["class"], "tokens": r["tokens"], "text": r["text"]}) + "\n")
(K.WORK / "step3.json").write_text(json.dumps(out, indent=1) + "\n")
