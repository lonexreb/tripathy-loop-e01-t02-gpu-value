"""Weight-space programming kernel: read one decision in Qwen3.5-0.8B and write a direct edit.

No optimizer and no backward pass anywhere in this file. Runs locally with the
torch environment (`.venv-wsp`), as a script:

    .venv-wsp/bin/python src/tripathy_arch3/wsp.py diagnose|edit|validate|actor

Inputs are the development-origin prompts written by `wsp_gate.py prompts`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / "artifacts" / "wsp"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SYSTEM_PROMPT = (  # identical to modal_app.run_actor
    "You are the frozen G0 forecasting actor. Use only supplied data, "
    "choose one allowed action, and emit one JSON object with no markdown."
)
ANSWER_PREFIX = '{"action":"propose_forecaster_patch","method":"'
BAD, GOOD = "drift", "repricing_interval"
CHOOSING_END = "2026-09-24"  # origins after this are held out
MAX_COLUMNS = 32  # per decision; two decisions on one tensor = 64 columns
UNRELATED_PROMPTS = (
    "What is the capital of France?",
    "Write a haiku about autumn rain.",
    "Explain what a hash function is in two sentences.",
    "Translate 'good morning' into Spanish and German.",
    "List three prime numbers greater than 50.",
    'Return a JSON object with keys "name" and "age" for a 30 year old named Ada.',
    "What is 17 times 23?",
    "Give one reason why the sky is blue.",
)


def _load() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": MODEL_REVISION, "cache_dir": str(WORKSPACE / ".hf-cache")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, **kwargs)
    model.to("mps" if torch.backends.mps.is_available() else "cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _prompts() -> dict[str, str]:
    return dict(sorted(json.loads((OUT / "prompts.json").read_text()).items()))


def _encode(tokenizer: Any, prompt: str, *, system: str | None = SYSTEM_PROMPT) -> Any:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )["input_ids"]


def _split(tokenizer: Any, bad: str, good: str) -> tuple[list[int], int, int]:
    """Ids shared by two answer continuations and the first token where they diverge."""
    bad_ids = tokenizer(bad, add_special_tokens=False)["input_ids"]
    good_ids = tokenizer(good, add_special_tokens=False)["input_ids"]
    shared = 0
    while bad_ids[shared] == good_ids[shared]:
        shared += 1
    return bad_ids[:shared], bad_ids[shared], good_ids[shared]


Decision = tuple[list[int], int, int]


def _decision_tokens(tokenizer: Any) -> Decision:
    return _split(tokenizer, ANSWER_PREFIX + BAD + '"', ANSWER_PREFIX + GOOD + '"')


def _specs(tokenizer: Any) -> list[tuple[str, list[Decision]]]:
    """The two decisions the edit rewrites: (prefix ids, token to demote, token to promote)."""
    head = ANSWER_PREFIX + GOOD + '","width_scale'
    return [
        ("method", [_decision_tokens(tokenizer)]),
        (
            "bare_number",  # after a numeric key the edited model writes colon-quote; want colon
            [
                _split(tokenizer, head + '":"1', head + '":1'),
                _split(tokenizer, head + '":1,"window_days":"6', head + '":1,"window_days":6'),
            ],
        ),
    ]


def _mlp_layers(model: Any) -> list[tuple[str, Any]]:
    layers = [(n, m) for n, m in model.named_modules() if n.endswith(".mlp.down_proj")]
    if not layers:
        raise RuntimeError("no mlp.down_proj modules found")
    return layers


def _generate(tokenizer: Any, model: Any, input_ids: Any, max_new_tokens: int = 64) -> str:
    import torch

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)
    return str(tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip())


def _record(
    tokenizer: Any, model: Any, prompt: str, decision: Decision | None = None
) -> dict[str, Any]:
    """One teacher-forced pass up to the decision token, with MLP reads at every position."""
    import torch

    shared, bad_id, good_id = decision or _decision_tokens(tokenizer)
    ids = torch.cat([_encode(tokenizer, prompt), torch.tensor([shared])], dim=1).to(model.device)
    captured: dict[str, Any] = {}
    hooks = []
    for name, module in _mlp_layers(model):
        hooks.append(
            module.register_forward_hook(
                lambda _m, args, output, name=name: captured.__setitem__(
                    name, (args[0][0].clone(), output[0, -1].clone())
                )
            )
        )
    head = model.get_output_embeddings()
    hooks.append(
        head.register_forward_hook(
            lambda _m, args, _o: captured.__setitem__("head_in", args[0][0, -1].clone())
        )
    )
    last_block = _mlp_layers(model)[-1][0].removesuffix(".mlp.down_proj")
    hooks.append(
        model.get_submodule(last_block).register_forward_hook(
            lambda _m, _a, output: captured.__setitem__(
                "resid", (output[0] if isinstance(output, tuple) else output)[0, -1].clone()
            )
        )
    )
    try:
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
    finally:
        for hook in hooks:
            hook.remove()
    captured["gap"] = float(logits[good_id] - logits[bad_id])
    captured["top"] = tokenizer.decode([int(logits.argmax())])
    captured["tokens"] = (bad_id, good_id)
    return captured


def _direction(model: Any, tokenizer: Any, record: dict[str, Any]) -> Any:
    """Residual-space direction that raises logit(GOOD) - logit(BAD), through the final norm."""
    bad_id, good_id = record["tokens"]
    unembed = model.get_output_embeddings().weight
    resid = record["resid"]
    # An RMS-style norm maps x to gain * x / rms(x); its gain is what it returns for all-ones.
    gain = _final_norm(model)(resid.new_ones(1, 1, resid.shape[0]))[0, 0]
    return (unembed[good_id] - unembed[bad_id]) * gain / resid.pow(2).mean().sqrt()


def _final_norm(model: Any) -> Any:
    parent = _mlp_layers(model)[-1][0].rsplit(".layers.", 1)[0]
    return model.get_submodule(parent + ".norm")


def cmd_diagnose(_: argparse.Namespace) -> None:
    import torch

    tokenizer, model = _load()
    prompts = _prompts()
    shared, bad_id, good_id = _decision_tokens(tokenizer)
    proposal = prompts["2026-09-30"]
    anchor = _generate(tokenizer, model, _encode(tokenizer, proposal))
    layers = [name for name, _ in _mlp_layers(model)]
    per_layer = torch.zeros(len(layers))
    gaps: dict[str, float] = {}
    for origin, prompt in prompts.items():
        if origin > CHOOSING_END:
            continue
        record = _record(tokenizer, model, prompt)
        direction = _direction(model, tokenizer, record)
        gaps[origin] = record["gap"]
        per_layer += torch.stack([record[name][1] @ direction for name in layers]).cpu()
    per_layer /= len(gaps)

    swapped: dict[str, str] = {}
    for origin in ("2026-09-01", "2026-09-15", "2026-09-30"):
        episode = json.loads(prompts[origin])
        assert json.dumps(episode, sort_keys=True, separators=(",", ":")) == prompts[origin]
        episode["allowed_actions"].reverse()
        text = json.dumps(episode, sort_keys=True, separators=(",", ":"))
        swapped[origin] = _generate(tokenizer, model, _encode(tokenizer, text))

    record = _record(tokenizer, model, proposal)
    with torch.no_grad():
        ids = _encode(tokenizer, proposal).to(model.device)
        own = model.generate(ids, do_sample=False, max_new_tokens=len(shared))[0][ids.shape[-1] :]
        norm_error = float(
            (
                _final_norm(model)(record["resid"][None, None])[0, 0] - record["head_in"]
            ).abs().max()
        )
        predicted_gap = float(
            record["head_in"] @ (model.get_output_embeddings().weight[good_id]
                                 - model.get_output_embeddings().weight[bad_id])
        )
    result = {
        "anchor_output_2026-09-30": anchor,
        "prefix_tokenization_matches_own_generation": own.tolist() == shared,
        "final_norm_reconstruction_max_abs_error": norm_error,
        "gap_from_head_input_vs_logits": [predicted_gap, record["gap"]],
        "direction_dot_resid_vs_gap": [
            float(_direction(model, tokenizer, record) @ record["resid"]), record["gap"]
        ],
        "shared_prefix_tokens": [tokenizer.decode([t]) for t in shared],
        "bad_token": tokenizer.decode([bad_id]),
        "good_token": tokenizer.decode([good_id]),
        "logit_gap_good_minus_bad": gaps,
        "mean_gap": sum(gaps.values()) / len(gaps),
        "mlp_direct_contribution_to_gap": {
            n: round(float(v), 4) for n, v in zip(layers, per_layer, strict=True)
        },
        "order_swap_control_outputs": swapped,
        "down_proj_shape": list(_mlp_layers(model)[-1][1].weight.shape),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnose.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def _mean_gap(tokenizer: Any, model: Any, prompts: dict[str, str], spec: list[Decision]) -> float:
    gaps = [_record(tokenizer, model, p, d)["gap"] for p in prompts.values() for d in spec]
    return float(sum(gaps) / len(gaps))


def _build_edit(
    tokenizer: Any,
    model: Any,
    layer: str,
    prompts: dict[str, str],
    spec: list[Decision],
    *,
    random_seed: int | None,
) -> dict[str, Any]:
    """Closed-form rank-one write per unit strength: delta_W[:, J] = unit(d) c_J^T / (c_J . k_J)."""
    import torch

    keys, others, directions = [], [], []
    for prompt in prompts.values():
        for decision in spec:
            record = _record(tokenizer, model, prompt, decision)
            mlp_in = record[layer][0]  # [positions, 3584]
            keys.append(mlp_in[-1])
            others.append(mlp_in[:-1].mean(dim=0))
            directions.append(_direction(model, tokenizer, record))
    key = torch.stack(keys).mean(dim=0)
    other = torch.stack(others).mean(dim=0)
    contrast = key - other
    columns = contrast.abs().topk(MAX_COLUMNS).indices.sort().values
    direction = torch.stack(directions).mean(dim=0)
    unit = direction / direction.norm()
    if random_seed is not None:  # negative control: same norm, orthogonal to the read direction
        noise = torch.randn(unit.shape, generator=torch.Generator().manual_seed(random_seed))
        noise -= (noise @ unit.cpu()) * unit.cpu()
        unit = (noise / noise.norm()).to(direction.device)
    row = contrast[columns] / (contrast[columns] @ key[columns])
    return {
        "columns": columns,
        "delta": torch.outer(unit, row),
        "gain": float(direction.norm()),  # logit-gap change per unit of strength
        "other_position_leak": float((row @ other[columns]).abs()),
    }


def _apply(model: Any, layer: str, edit: dict[str, Any], strength: float, base: Any) -> None:
    weight = model.get_submodule(layer).weight
    if not layer.endswith(".mlp.down_proj") or weight.shape != base.shape:
        raise ValueError("edits are restricted to MLP down_proj weights")
    weight.copy_(base)
    weight[:, edit["columns"]] += strength * edit["delta"]


def _behaviour(tokenizer: Any, model: Any, prompts: dict[str, str]) -> dict[str, Any]:
    outputs = {o: _generate(tokenizer, model, _encode(tokenizer, p)) for o, p in prompts.items()}
    method, valid = 0, 0
    for text in outputs.values():
        try:
            patch = json.loads(text)
        except json.JSONDecodeError:
            continue
        method += patch.get("method") == GOOD
        width = patch.get("width_scale")
        valid += (
            patch.get("method") == GOOD
            and patch.get("action") == "propose_forecaster_patch"
            and patch.get("window_days") in (30, 60, 90)
            and isinstance(width, (int, float))
            and 0.5 <= width <= 2
        )
    return {"outputs": outputs, "method_flipped": method, "valid": valid, "n": len(outputs)}


def cmd_edit(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    tokenizer, model = _load()
    prompts = _prompts()
    choosing = {o: p for o, p in prompts.items() if o <= CHOOSING_END}
    weight = model.get_submodule(args.layer).weight
    original = weight.clone()
    stages: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, spec in _specs(tokenizer):
            base = weight.clone()
            base_gap = _mean_gap(tokenizer, model, choosing, spec)
            edit = _build_edit(tokenizer, model, args.layer, choosing, spec, random_seed=None)
            goal = "method_flipped" if name == "method" else "valid"
            dose: list[dict[str, Any]] = []
            chosen: float | None = None
            for multiple in (0.5, 1.0, 1.25, 1.5, 2.0, 3.0):
                strength = multiple * -base_gap / edit["gain"]  # 1.0 = predicted to close the gap
                _apply(model, args.layer, edit, strength, base)
                behaviour = _behaviour(tokenizer, model, choosing)
                dose.append(
                    {
                        "multiple_of_predicted": multiple,
                        "strength": strength,
                        "mean_gap": _mean_gap(tokenizer, model, choosing, spec),
                        "method_flipped": behaviour["method_flipped"],
                        "valid": behaviour["valid"],
                        "n": behaviour["n"],
                        "outputs": sorted(set(behaviour["outputs"].values())),
                    }
                )
                print(name, json.dumps(dose[-1]))
                if behaviour[goal] == behaviour["n"]:
                    chosen = strength
                    break
            stages.append(
                {
                    "decision": name,
                    "columns": edit["columns"].tolist(),
                    "strength": chosen,
                    "base_mean_logit_gap": base_gap,
                    "predicted_gap_gain_per_unit_strength": edit["gain"],
                    "other_position_leak_per_unit_strength": edit["other_position_leak"],
                    "dose_response": dose,
                }
            )
            if chosen is None:
                weight.copy_(original)
                (OUT / "dose_response.json").write_text(json.dumps(stages, indent=2) + "\n")
                raise SystemExit(f"no strength on the grid achieved the goal for decision {name}")
        edited = weight.clone()
    changed = (edited != original).nonzero()
    flat = (changed[:, 0] * original.shape[1] + changed[:, 1]).cpu().numpy()
    edit_dir = OUT / "edit"
    edit_dir.mkdir(parents=True, exist_ok=True)
    np.savez(edit_dir / "edit.npz", idx=flat, val=edited[edited != original].cpu().numpy())
    total = sum(p.numel() for p in model.parameters())
    manifest = {
        "base_model": f"{MODEL_ID}@{MODEL_REVISION}",
        "tensor": args.layer + ".weight",
        "form": "sparse float32 overwrite; per decision W + s * unit(d) c_J^T / (c_J . k_J)",
        "shape": list(original.shape),
        "decisions": [{k: v for k, v in s.items() if k != "dose_response"} for s in stages],
        "changed_columns": len({c for s in stages for c in s["columns"]}),
        "changed_weights": int(flat.size),
        "fraction_of_model": flat.size / total,
        "relative_change_frobenius": float((edited - original).norm() / original.norm()),
        "base_tensor_sha256": hashlib.sha256(original.cpu().numpy().tobytes()).hexdigest(),
        "edited_tensor_sha256": hashlib.sha256(edited.cpu().numpy().tobytes()).hexdigest(),
        "gradient_steps": 0,
    }
    (edit_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "dose_response.json").write_text(json.dumps(stages, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def _load_edited() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Rebuild the edited model from the frozen sparse edit, verifying both tensor hashes."""
    import numpy as np
    import torch

    tokenizer, model = _load()
    manifest = json.loads((OUT / "edit" / "manifest.json").read_text())
    stored = np.load(OUT / "edit" / "edit.npz")
    weight = model.get_submodule(manifest["tensor"].removesuffix(".weight")).weight
    base = weight.clone()
    if hashlib.sha256(base.cpu().numpy().tobytes()).hexdigest() != manifest["base_tensor_sha256"]:
        raise RuntimeError("base tensor does not match the edit manifest")
    with torch.no_grad():
        weight.view(-1)[torch.from_numpy(stored["idx"]).to(weight.device)] = torch.from_numpy(
            stored["val"]
        ).to(weight.device)
    if (
        hashlib.sha256(weight.cpu().numpy().tobytes()).hexdigest()
        != manifest["edited_tensor_sha256"]
    ):
        raise RuntimeError("rebuilt tensor does not match the edit manifest")
    return tokenizer, model, base, manifest


def _log_probs(model: Any, texts: list[Any]) -> list[Any]:
    import torch

    with torch.no_grad():
        return [model(ids.to(model.device)).logits[0].log_softmax(dim=-1) for ids in texts]


def cmd_validate(_: argparse.Namespace) -> None:
    import torch

    tokenizer, model, base, manifest = _load_edited()
    layer = manifest["tensor"].removesuffix(".weight")
    weight = model.get_submodule(layer).weight
    edited = weight.clone()
    prompts = _prompts()
    choosing = {o: p for o, p in prompts.items() if o <= CHOOSING_END}
    held_out = {o: p for o, p in prompts.items() if o > CHOOSING_END}
    unrelated = [_encode(tokenizer, text, system=None) for text in UNRELATED_PROMPTS]
    own = [_encode(tokenizer, p) for p in list(held_out.values())[:3]]

    result: dict[str, Any] = {"held_out_edited": _behaviour(tokenizer, model, held_out)}
    edited_lp = _log_probs(model, unrelated + own)
    edited_text = [_generate(tokenizer, model, ids, 48) for ids in unrelated]

    with torch.no_grad():
        weight.copy_(base)
    result["held_out_original"] = _behaviour(tokenizer, model, held_out)
    base_lp = _log_probs(model, unrelated + own)
    base_text = [_generate(tokenizer, model, ids, 48) for ids in unrelated]
    agree = [
        float((a.argmax(-1) == b.argmax(-1)).float().mean())
        for a, b in zip(base_lp, edited_lp, strict=True)
    ]
    pairs = list(zip(base_lp, edited_lp, strict=True))
    kl = [float((a.exp() * (a - b)).sum(-1).mean()) for a, b in pairs]
    result["locality"] = {
        "unrelated_prompts_top1_agreement": sum(agree[: len(unrelated)]) / len(unrelated),
        "unrelated_prompts_mean_kl_nats": sum(kl[: len(unrelated)]) / len(unrelated),
        "own_prompt_positions_top1_agreement": sum(agree[len(unrelated) :]) / len(own),
        "own_prompt_positions_mean_kl_nats": sum(kl[len(unrelated) :]) / len(own),
        "unrelated_generations_identical": sum(
            a == b for a, b in zip(base_text, edited_text, strict=True)
        ),
        "unrelated_generations_n": len(unrelated),
    }

    # Negative control: same columns, keys and strengths, random directions orthogonal to the
    # read ones. Keys for the second decision are measured with the real first edit in place.
    with torch.no_grad():
        control = base.clone()
        for seed, ((_name, spec), stage) in enumerate(
            zip(_specs(tokenizer), manifest["decisions"], strict=True)
        ):
            current = weight.clone()
            random_edit = _build_edit(tokenizer, model, layer, choosing, spec, random_seed=seed)
            control[:, random_edit["columns"]] += stage["strength"] * random_edit["delta"]
            real_edit = _build_edit(tokenizer, model, layer, choosing, spec, random_seed=None)
            _apply(model, layer, real_edit, stage["strength"], current)
        rebuilt_matches = bool(torch.allclose(weight, edited, atol=1e-6))
        weight.copy_(control)
    control_behaviour = _behaviour(tokenizer, model, held_out)
    result["edit_rebuilds_from_recipe"] = rebuilt_matches
    result["random_direction_control"] = {
        "relative_change_frobenius": float((control - base).norm() / base.norm()),
        "method_flipped": control_behaviour["method_flipped"],
        "valid": control_behaviour["valid"],
        "n": control_behaviour["n"],
        "outputs": sorted(set(control_behaviour["outputs"].values())),
    }
    with torch.no_grad():
        weight.copy_(edited)
    (OUT / "validate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def cmd_actor(_: argparse.Namespace) -> None:
    tokenizer, model, _, manifest = _load_edited()
    prompt = _prompts()["2026-09-30"]
    response = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "edit_manifest_sha256": hashlib.sha256(
            (OUT / "edit" / "manifest.json").read_bytes()
        ).hexdigest(),
        "edited_tensor_sha256": manifest["edited_tensor_sha256"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output": _generate(tokenizer, model, _encode(tokenizer, prompt), 256),
        "device": f"local {model.device.type} float32",
        "seed": 0,
    }
    (OUT / "actor_response.json").write_text(json.dumps(response, indent=2) + "\n")
    print(json.dumps(response, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("diagnose").set_defaults(run=cmd_diagnose)
    edit = commands.add_parser("edit")
    edit.add_argument("--layer", required=True, help="module name ending in .mlp.down_proj")
    edit.set_defaults(run=cmd_edit)
    commands.add_parser("validate").set_defaults(run=cmd_validate)
    commands.add_parser("actor").set_defaults(run=cmd_actor)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
