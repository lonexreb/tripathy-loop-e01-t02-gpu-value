"""Editor's kernel helpers: build and apply direction-removal edits from the read in read1.pt."""
import json, torch
from tripathy_arch3 import wsp, wsp_practice as P
WORK = P.OUT / "work"
def read(): return torch.load(WORK / "read1.pt")
def direction(d, ref=19):
    u = (d["res_R"] - d["res_P"])[:, ref + 1].mean(0)
    return u / u.norm()
def neuron_scores(d, W, layer, u):
    """How much more each MLP neuron writes along u when looping than when productive."""
    proj = u @ W.cpu()                                   # [3584] projection of each output column
    return (d["in_R"][:, layer].mean(0) - d["in_P"][:, layer].mean(0)) * proj
def build(model, d, layers=(15, 19), k=64, ref=19, u=None):
    u = direction(d, ref) if u is None else u
    edit = {}
    for l in layers:
        W = model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight
        sc = neuron_scores(d, W, l, u)
        J = sc.topk(k).indices.sort().values
        edit[l] = {"columns": J, "u": u, "coverage": float(sc[J].sum() / sc.clamp(min=0).sum()), "net": float(sc.sum())}
    return edit
def apply(model, edit, s, base):
    with torch.no_grad():
        for l, e in edit.items():
            W = model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight
            W.copy_(base[l]); u = e["u"].to(W.device); J = e["columns"].to(W.device)
            W[:, J] -= s * torch.outer(u, u @ W[:, J])
def base_weights(model, layers=(15, 19)):
    return {l: model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight.clone() for l in layers}
def charge(kind, note):
    p = P.OUT / "forward_passes.json"; led = json.loads(p.read_text())
    led.append({"n": len(led) + 1, "kind": kind, "note": note}); p.write_text(json.dumps(led, indent=1) + "\n")
    assert len(led) <= 60, "forward-pass budget exhausted"; return len(led)
