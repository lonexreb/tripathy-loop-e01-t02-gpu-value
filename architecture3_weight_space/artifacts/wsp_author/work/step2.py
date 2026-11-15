"""No forward pass: compare our direction with the released edit's, and size the candidate edit."""
import sys, torch; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel as K
from tripathy_arch3 import wsp, wsp_practice as P
tok, model = wsp._load(); d = K.read(); base = K.base_weights(model)
u15, u19 = K.direction(d, 15), K.direction(d, 19)
print("cos(u@15, u@19) =", round(float(u15 @ u19), 3))
P.apply_author_edit(model)
for l in (15, 19):
    W2 = model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight
    delta = (W2 - base[l]).cpu(); U, S, V = torch.linalg.svd(delta, full_matrices=False)
    ua = U[:, 0]; cols = (delta.abs().sum(0) > 0).nonzero().flatten()
    print(f"layer {l}: released edit rank-1 share {float(S[0]**2 / (S**2).sum()):.3f}; |cos| with our u@15 {abs(float(ua @ u15)):.3f}, u@19 {abs(float(ua @ u19)):.3f}")
    sc = K.neuron_scores(d, base[l], l, u19); ours = set(sc.topk(64).indices.tolist())
    print(f"   released columns also in our top-64: {len(ours & set(cols.tolist()))}/64; our score mass on released columns {float(sc[cols].sum() / sc.clamp(min=0).sum()):.3f}")
    with torch.no_grad(): W2.copy_(base[l])
for ref in (15, 19):
    e = K.build(model, d, ref=ref)
    print("ref", ref, {l: (round(v["coverage"], 3), round(v["net"], 3)) for l, v in e.items()})
