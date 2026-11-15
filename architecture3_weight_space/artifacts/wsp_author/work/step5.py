"""Is the 1,024-token cap the binding constraint? Final stack vs untouched at 4,096 tokens, 20 items."""
import sys, json, random, torch; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import kernel as K
from tripathy_arch3 import wsp, wsp_practice as P
tok, model = wsp._load()
choose = [i for i in P.practice_set() if i["split"] == "choose"]
C = random.Random(2).sample(choose, 20)
out = {}
def arm(name):
    K.charge("generate", f"{name}: 20 choose items, greedy, 4096 tokens")
    recs = P.generate(tok, model, C, thinking=True, max_new_tokens=4096, batch_size=20)
    sc = P.score(recs); out[name] = {"classes": sc["classes"], "mean_tokens": sc["mean_tokens"], "pinball": sc["pinball"]}
    print(name, json.dumps(out[name]), flush=True)
    return recs
arm("untouched")
st = torch.load(K.WORK / "stack.pt")
with torch.no_grad():
    for l, w in st["weights"].items(): model.get_submodule(f"model.layers.{l}.mlp.down_proj").weight.copy_(w.to(model.device))
recs = arm("stacked_rounds_1_2")
print("tail:", recs[0]["text"][-600:])
(K.WORK / "step5.json").write_text(json.dumps(out, indent=1) + "\n")
