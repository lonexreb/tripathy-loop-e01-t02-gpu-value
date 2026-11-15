"""Recursive rounds: read the current model's own loop, stack a direction-removal edit, repeat."""
import sys, json, random, statistics, torch; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel as K
from tripathy_arch3 import wsp, wsp_practice as P
tok, model = wsp._load()
items = {(i["series_id"], i["origin"]): i for i in P.practice_set()}
layers = wsp._mlp_layers(model); L = len(layers)
def W(l): return model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight
original = {l: W(l).clone() for l in range(L)}
def onset_char(text):
    seen, pos = set(), 0
    for line in text.split("\n"):
        k = line.strip()
        if len(k) > 15 and k in seen: return pos
        if len(k) > 15: seen.add(k)
        pos += len(line) + 1
    return None
def read(rows):
    cap = {}
    hooks = [m.register_forward_hook(lambda _m, a, o, i=i: cap.__setitem__(i, (a[0][0], o[0]))) for i, (_, m) in enumerate(layers)]
    acc = {k: [] for k in ("out_R", "out_P", "in_R", "in_P", "res_R", "res_P")}
    for r in rows:
        text = r["text"].removeprefix("<think>"); start = onset_char(text)
        if start is None: continue
        chat = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": P.prompt(items[(r["series_id"], r["origin"])])}]
        pid = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=True)["input_ids"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.cat([pid, torch.tensor([enc["input_ids"]])], dim=1).to(model.device); n0 = pid.shape[1]
        offs = torch.tensor([o[0] for o in enc["offset_mapping"]])
        R = torch.zeros(ids.shape[1], dtype=torch.bool); R[n0:] = offs >= start
        Pm = torch.zeros(ids.shape[1], dtype=torch.bool); Pm[n0:] = offs < start
        if R.sum() < 50 or Pm.sum() < 50: continue
        with torch.no_grad(): hs = model(ids, output_hidden_states=True).hidden_states
        acc["out_R"].append(torch.stack([cap[i][1][R].mean(0) for i in range(L)]).cpu()); acc["out_P"].append(torch.stack([cap[i][1][Pm].mean(0) for i in range(L)]).cpu())
        acc["in_R"].append(torch.stack([cap[i][0][R].mean(0) for i in range(L)]).cpu()); acc["in_P"].append(torch.stack([cap[i][0][Pm].mean(0) for i in range(L)]).cpu())
        acc["res_R"].append(torch.stack([h[0][R].mean(0) for h in hs]).cpu()); acc["res_P"].append(torch.stack([h[0][Pm].mean(0) for h in hs]).cpu())
    for h in hooks: h.remove()
    return {k: torch.stack(v) for k, v in acc.items()}
def text_stats(recs):
    dup, uniq = [], []
    for r in recs:
        lines = [x.strip() for x in r["text"].split("\n") if len(x.strip()) > 15]
        dup.append(1 - len(set(lines)) / max(len(lines), 1)); uniq.append(len(set(lines)))
    return statistics.mean(dup), statistics.mean(uniq)
def run(B, note):
    n = K.charge("generate", note)
    recs = P.generate(tok, model, B, thinking=True, max_new_tokens=1024, batch_size=30)
    sc = P.score(recs); dup, uniq = text_stats(recs)
    done = sum("token_limit" not in r["class"] for r in recs)
    return recs, {"pass": n, "terminated": done, "ok": sc["classes"].get("ok", 0), "duplicate_share": round(dup, 3), "distinct_lines": round(uniq, 1), "classes": sc["classes"], "pinball": sc["pinball"]}
def rows_of(recs): return [{"series_id": r["item"]["series_id"], "origin": r["item"]["origin"], "text": r["text"]} for r in recs]
choose = [i for i in P.practice_set() if i["split"] == "choose"]
usedA = {(json.loads(l)["series_id"], json.loads(l)["origin"]) for l in open(P.OUT / "diagnose_base_thinking.jsonl")}
B = random.Random(1).sample([i for i in choose if (i["series_id"], i["origin"]) not in usedA], 60)
# Round 1 = the edit already tested (read1 direction, layers 15 and 19, strength 4).
d1 = K.read(); e1 = K.build(model, d1, ref=19); K.apply(model, e1, 4, {l: original[l] for l in (15, 19)})
stack = [{"round": 1, "layers": [15, 19], "strength": 4, "u": e1[15]["u"].tolist(), "columns": {str(l): e1[l]["columns"].tolist() for l in (15, 19)}}]
rows = [json.loads(l) for l in open(K.WORK / "B_s4.jsonl")]
current = {"duplicate_share": 0.191, "distinct_lines": 20.6, "terminated": 0}
log = []
for rnd in range(2, 8):
    K.charge("teacher_forced", f"round {rnd}: read looping vs productive on the current model's own outputs")
    d = read(rows)
    if d["res_R"].shape[0] < 10: log.append({"round": rnd, "stop": "fewer than 10 looping texts"}); break
    u = K.direction(d, 19)
    writes = ((d["out_R"] - d["out_P"]) @ u).mean(0)
    top = sorted(writes.topk(2).indices.tolist())
    state = {l: W(l).clone() for l in top}
    edit = {}
    for l in top:
        sc = K.neuron_scores(d, state[l], l, u); J = sc.topk(64).indices.sort().values
        edit[l] = {"columns": J, "u": u}
    trials = {}
    for s in (2, 4, 8):
        K.apply(model, edit, s, state)
        recs, st = run(B, f"round {rnd}: stacked edit on layers {top}, strength {s}")
        trials[s] = (recs, st); print("round", rnd, "layers", top, "strength", s, json.dumps(st), flush=True)
    ok = {s: t for s, t in trials.items() if t[1]["terminated"] > current["terminated"] or (t[1]["duplicate_share"] < current["duplicate_share"] and t[1]["distinct_lines"] >= 0.8 * current["distinct_lines"])}
    entry = {"round": rnd, "texts_read": int(d["res_R"].shape[0]), "writer_layers": top, "writes": [round(float(x), 3) for x in writes], "cos_with_round1_u": round(float(u @ torch.tensor(stack[0]["u"])), 3), "trials": {s: t[1] for s, t in trials.items()}}
    if not ok:
        K.apply(model, edit, 0, state); entry["kept"] = None; log.append(entry); print("round", rnd, "no strength helped; stopping", flush=True); break
    best = max(ok, key=lambda s: (ok[s][1]["terminated"], -ok[s][1]["duplicate_share"]))
    K.apply(model, edit, best, state); rows = rows_of(trials[best][0]); current = trials[best][1]
    entry["kept"] = best; log.append(entry)
    stack.append({"round": rnd, "layers": top, "strength": best, "u": u.tolist(), "columns": {str(l): edit[l]["columns"].tolist() for l in top}})
    with open(K.WORK / f"round{rnd}_kept.jsonl", "w") as h:
        for r in rows: h.write(json.dumps(r) + "\n")
    (K.WORK / "step4.json").write_text(json.dumps({"log": log, "current": current}, indent=1) + "\n")
    if current["terminated"] >= 6: print("termination reached 10%; stopping to confirm", flush=True); break
(K.WORK / "step4.json").write_text(json.dumps({"log": log, "current": current}, indent=1) + "\n")
torch.save({"stack": stack, "weights": {l: W(l).cpu() for l in range(L) if not torch.equal(W(l), original[l])}}, K.WORK / "stack.pt")
print("final", json.dumps(current), "rounds kept", len(stack))
