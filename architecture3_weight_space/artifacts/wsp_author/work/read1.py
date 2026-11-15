"""Pass 9 (teacher-forced): where is the repetition written? Productive vs looping tokens."""
import json, torch
from tripathy_arch3 import wsp, wsp_practice as P
tok, model = wsp._load()
rows = [json.loads(l) for l in open(P.OUT / "diagnose_base_thinking.jsonl")]
items = {(i["series_id"], i["origin"]): i for i in P.practice_set()}
layers = wsp._mlp_layers(model)
L = len(layers)
def onset_char(text):
    seen, pos = set(), 0
    for line in text.split("\n"):
        k = line.strip()
        if len(k) > 15 and k in seen: return pos
        if len(k) > 15: seen.add(k)
        pos += len(line) + 1
    return None
cap = {}
hooks = [m.register_forward_hook(lambda _m, a, o, i=i: cap.__setitem__(i, (a[0][0], o[0]))) for i, (_, m) in enumerate(layers)]
out_R, out_P, in_R, in_P, res_R, res_P = [], [], [], [], [], []
for r in rows:
    text = r["text"].removeprefix("<think>")
    start = onset_char(text)
    if start is None: continue
    chat = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": P.prompt(items[(r["series_id"], r["origin"])])}]
    prompt_ids = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=True)["input_ids"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.cat([prompt_ids, torch.tensor([enc["input_ids"]])], dim=1).to(model.device)
    n0 = prompt_ids.shape[1]
    offs = torch.tensor([o[0] for o in enc["offset_mapping"]])
    R = torch.zeros(ids.shape[1], dtype=torch.bool); R[n0:] = offs >= start
    Pm = torch.zeros(ids.shape[1], dtype=torch.bool); Pm[n0:] = offs < start
    if R.sum() < 50 or Pm.sum() < 50: continue
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    out_R.append(torch.stack([cap[i][1][R].mean(0) for i in range(L)]).cpu()); out_P.append(torch.stack([cap[i][1][Pm].mean(0) for i in range(L)]).cpu())
    in_R.append(torch.stack([cap[i][0][R].mean(0) for i in range(L)]).cpu()); in_P.append(torch.stack([cap[i][0][Pm].mean(0) for i in range(L)]).cpu())
    res_R.append(torch.stack([h[0][R].mean(0) for h in hs]).cpu()); res_P.append(torch.stack([h[0][Pm].mean(0) for h in hs]).cpu())
for h in hooks: h.remove()
S = lambda x: torch.stack(x)
d = {k: S(v) for k, v in dict(out_R=out_R, out_P=out_P, in_R=in_R, in_P=in_P, res_R=res_R, res_P=res_P).items()}
torch.save(d, P.OUT / "work" / "read1.pt")
n = d["out_R"].shape[0]; print("texts used", n)
res_diff = d["res_R"] - d["res_P"]          # [n, 25, 1024]
out_diff = d["out_R"] - d["out_P"]          # [n, 24, 1024]
def consistency(x):                         # mean cosine of each text's diff with the mean diff
    m = x.mean(0, keepdim=True); return torch.nn.functional.cosine_similarity(x, m.expand_as(x), dim=-1).mean(0)
rc, oc = consistency(res_diff), consistency(out_diff)
print("layer | resid diff norm / resid norm | resid consistency | mlp-out diff norm | mlp consistency")
for l in range(L):
    print(l, round(float(res_diff[:, l + 1].mean(0).norm() / d["res_P"][:, l + 1].mean(0).norm()), 3), round(float(rc[l + 1]), 3), round(float(out_diff[:, l].mean(0).norm()), 3), round(float(oc[l]), 3))
# shared direction: mean residual difference after each layer; how much does each MLP write along it?
for ref in (15, 19, 23):
    u = res_diff[:, ref + 1].mean(0); u = u / u.norm()
    w = (out_diff @ u).mean(0)
    print("ref layer", ref, "MLP writes along u:", [round(float(x), 2) for x in w])
