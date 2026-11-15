"""Modal boundary for Architecture 3. No function is invoked on import."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tripathy-e01-arch3-weight-space"
MODEL_VOLUME_NAME = "tripathy-e01-arch3-weights"
MODEL_ROOT = Path("/models")
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_PATH = MODEL_ROOT / "qwen3.5-0.8b" / MODEL_REVISION
TRAINING_GPU_PRIORITY = ["H100", "B200"]

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True, version=2)

training_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12"
).uv_pip_install(
    "accelerate==1.15.0",
    "datasets==4.8.5",
    "peft==0.21.3",
    "safetensors==0.8.0",
    "torch==2.15.0",
    "transformers==5.23.0",
    "trl==0.29.1",
)


@app.function(
    image=training_image,
    gpu="L4",
    timeout=2 * 60,
    volumes={MODEL_ROOT.as_posix(): model_volume},
)
def environment_sanity() -> dict[str, object]:
    """Check the remote runtime only; do not download or train a model."""
    import peft
    import torch
    import transformers

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_manifest() -> tuple[list[dict[str, object]], str]:
    import hashlib
    import json

    files = [
        path
        for path in sorted(MODEL_PATH.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    rows = [
        {
            "path": path.relative_to(MODEL_PATH).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    rendered = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return rows, hashlib.sha256(rendered).hexdigest()


def _ensure_model_snapshot() -> tuple[list[dict[str, object]], str]:
    from huggingface_hub import snapshot_download

    marker = MODEL_PATH / ".complete"
    if not marker.is_file():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=MODEL_PATH,
        )
        marker.write_text(MODEL_REVISION + "\n", encoding="utf-8")
        model_volume.commit()
    if marker.read_text(encoding="utf-8").strip() != MODEL_REVISION:
        raise RuntimeError("model snapshot marker does not match the frozen revision")
    return _snapshot_manifest()


def _load_model() -> tuple[object, object, list[dict[str, object]], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files, manifest_sha = _ensure_model_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    return tokenizer, model, files, manifest_sha


@app.function(
    image=training_image,
    gpu="L4",
    timeout=5 * 60,
    volumes={MODEL_ROOT.as_posix(): model_volume},
)
def model_sanity() -> dict[str, object]:
    """Download the exact snapshot once and verify its architecture on an L4."""
    import torch

    tokenizer, model, files, manifest_sha = _load_model()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "parameters": parameters,
        "trainable_before_lora": trainable,
        "snapshot_file_count": len(files),
        "snapshot_manifest_sha256": manifest_sha,
        "cuda_device": torch.cuda.get_device_name(0),
        "dtype": str(next(model.parameters()).dtype),
    }


@app.function(
    image=training_image,
    gpu="L4",
    timeout=3 * 60,
    volumes={MODEL_ROOT.as_posix(): model_volume},
)
def run_actor(prompt: str, seed: int = 0, max_new_tokens: int = 256) -> dict[str, object]:
    """Run one deterministic text-only action proposal from the frozen G0 actor."""
    import hashlib

    import torch

    if not prompt.strip():
        raise ValueError("actor prompt may not be empty")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")
    tokenizer, model, _, manifest_sha = _load_model()
    torch.manual_seed(seed)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the frozen G0 forecasting actor. Use only supplied data, "
                "choose one allowed action, and emit one JSON object with no markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    output_ids = generated[0][prompt_tokens:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot_manifest_sha256": manifest_sha,
        "seed": seed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": int(output_ids.shape[-1]),
        "output": output,
    }


@app.function(
    image=training_image,
    gpu=TRAINING_GPU_PRIORITY,
    timeout=3 * 60,
    volumes={MODEL_ROOT.as_posix(): model_volume},
)
def run_adapter_actor(
    run_id: str,
    prompt: str,
    seed: int = 0,
    max_new_tokens: int = 256,
) -> dict[str, object]:
    """Run deterministic inference from one immutable LoRA child checkpoint."""
    import hashlib
    import re

    import torch
    from peft import PeftModel

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", run_id):
        raise ValueError("run_id must be a lowercase filesystem-safe identifier")
    if not prompt.strip():
        raise ValueError("actor prompt may not be empty")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")
    adapter_root = MODEL_ROOT / "adapters" / run_id
    if not adapter_root.is_dir():
        raise RuntimeError("the requested immutable adapter checkpoint does not exist")

    tokenizer, base_model, _, base_manifest_sha = _load_model()
    adapter_files, adapter_manifest_sha = _directory_manifest(adapter_root)
    model = PeftModel.from_pretrained(base_model, adapter_root).to("cuda").eval()
    torch.manual_seed(seed)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Architecture 3 forecasting actor. Use only supplied data and "
                "emit one action JSON object with no markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    output_ids = generated[0][prompt_tokens:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_snapshot_manifest_sha256": base_manifest_sha,
        "adapter_run_id": run_id,
        "adapter_manifest_sha256": adapter_manifest_sha,
        "adapter_file_count": len(adapter_files),
        "cuda_device": torch.cuda.get_device_name(0),
        "seed": seed,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": int(output_ids.shape[-1]),
        "output": output,
    }


def _tensor_digest(model: object, *, trainable_only: bool) -> str:
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        is_adapter = "lora_" in name
        if trainable_only != is_adapter:
            continue
        digest.update(name.encode())
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _directory_manifest(root: Path) -> tuple[list[dict[str, object]], str]:
    import hashlib
    import json

    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    rendered = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return rows, hashlib.sha256(rendered).hexdigest()


@app.function(
    image=training_image,
    gpu=TRAINING_GPU_PRIORITY,
    timeout=9 * 60,
    volumes={MODEL_ROOT.as_posix(): model_volume},
)
def train_lora_sanity(
    run_id: str,
    prompt: str,
    target: str,
    seed: int = 0,
    steps: int = 1,
    learning_rate: float = 0.0001,
) -> dict[str, object]:
    """Perform a bounded LoRA-only update, persist it, and prove save/reload parity."""
    import gc
    import json
    import math
    import re

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", run_id):
        raise ValueError("run_id must be a lowercase filesystem-safe identifier")
    if not prompt.strip() or not target.strip():
        raise ValueError("training prompt and target may not be empty")
    if not 1 <= steps <= 3:
        raise ValueError("sanity training steps must be between 1 and 3")
    if not 0 < learning_rate <= 0.001:
        raise ValueError("learning rate is outside the frozen sanity range")
    torch.manual_seed(seed)

    tokenizer, base_model, _, base_files_before = _load_model()
    system = (
        "You are the Architecture 3 forecasting actor. Use only supplied data and emit "
        "one action JSON object with no markdown."
    )
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    full_messages = [*prompt_messages, {"role": "assistant", "content": target}]
    prompt_encoding = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_encoding = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_encoding["input_ids"]
    full_ids = full_encoding["input_ids"]
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if full_ids.ndim == 1:
        full_ids = full_ids.unsqueeze(0)
    prompt_length = int(prompt_ids.shape[-1])
    if prompt_length >= int(full_ids.shape[-1]):
        raise RuntimeError("assistant target produced no trainable tokens")
    if not torch.equal(prompt_ids[0], full_ids[0, :prompt_length]):
        raise RuntimeError("chat template prompt is not a prefix of the training sequence")
    input_ids = full_ids.to("cuda")
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    attention_mask = torch.ones_like(input_ids)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("a non-LoRA parameter is trainable")
    frozen_before = _tensor_digest(model, trainable_only=False)
    adapter_before = _tensor_digest(model, trainable_only=True)
    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=learning_rate,
    )
    model.train()
    model.config.use_cache = False
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("training loss is missing or non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [value for value in model.parameters() if value.requires_grad], max_norm=1.0
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("adapter gradient norm is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm.detach().cpu()))

    frozen_after = _tensor_digest(model, trainable_only=False)
    adapter_after = _tensor_digest(model, trainable_only=True)
    if frozen_after != frozen_before:
        raise RuntimeError("a frozen base-model tensor changed during LoRA training")
    if adapter_after == adapter_before:
        raise RuntimeError("the LoRA adapter did not change")

    model.eval()
    with torch.inference_mode():
        reference_logits = (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            .logits[:, -1, :]
            .float()
            .cpu()
        )

    adapter_root = MODEL_ROOT / "adapters" / run_id
    temporary_root = MODEL_ROOT / "adapters" / ("." + run_id + ".tmp")
    if adapter_root.exists() or temporary_root.exists():
        raise RuntimeError("adapter run_id already exists; checkpoints are immutable")
    temporary_root.mkdir(parents=True)
    model.save_pretrained(temporary_root, safe_serialization=True)
    torch.save(optimizer.state_dict(), temporary_root / "optimizer.pt")
    provisional_files, provisional_sha = _directory_manifest(temporary_root)
    training_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cuda_device": torch.cuda.get_device_name(0),
        "base_snapshot_manifest_sha256": base_files_before,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "rank": 8,
        "alpha": 16,
        "target_modules": "all-linear",
        "prompt_tokens": prompt_length,
        "target_tokens": int(full_ids.shape[-1]) - prompt_length,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "base_tensor_sha256_before": frozen_before,
        "base_tensor_sha256_after": frozen_after,
        "adapter_tensor_sha256_before": adapter_before,
        "adapter_tensor_sha256_after": adapter_after,
        "provisional_adapter_sha256": provisional_sha,
        "provisional_files": provisional_files,
    }
    (temporary_root / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_root.rename(adapter_root)
    model_volume.commit()
    adapter_files, adapter_sha = _directory_manifest(adapter_root)

    del optimizer, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    _, reloaded_base, _, base_files_after = _load_model()
    if base_files_after != base_files_before:
        raise RuntimeError("persisted base-model snapshot changed during adapter training")
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_root).to("cuda").eval()
    with torch.inference_mode():
        reloaded_logits = (
            reloaded(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            .logits[:, -1, :]
            .float()
            .cpu()
        )
    reload_max_abs_diff = float((reference_logits - reloaded_logits).abs().max())
    if not math.isfinite(reload_max_abs_diff) or reload_max_abs_diff > 1e-5:
        raise RuntimeError("saved and reloaded adapter logits differ")

    loss_mask = [0] * prompt_length + [1] * (int(full_ids.shape[-1]) - prompt_length)
    return {
        "status": "passed",
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cuda_device": torch.cuda.get_device_name(0),
        "base_snapshot_manifest_sha256": base_files_after,
        "base_tensor_sha256": frozen_after,
        "adapter_tensor_sha256_before": adapter_before,
        "adapter_tensor_sha256_after": adapter_after,
        "adapter_artifact_sha256": adapter_sha,
        "adapter_files": adapter_files,
        "trainable_parameter_names": trainable_names,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "reload_max_abs_diff": reload_max_abs_diff,
        "token_ids": full_ids[0].tolist(),
        "loss_mask": loss_mask,
        "assistant_target_start": prompt_length,
    }


@app.local_entrypoint()
def main(operation: str, input_path: str, output_path: str) -> None:
    """Local file boundary for reproducible Modal invocations."""
    import json

    request = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if operation == "model_sanity":
        result = model_sanity.remote()
    elif operation == "actor":
        result = run_actor.remote(
            str(request["prompt"]),
            seed=int(request.get("seed", 0)),
            max_new_tokens=int(request.get("max_new_tokens", 256)),
        )
    elif operation == "actor_adapter":
        result = run_adapter_actor.remote(
            run_id=str(request["run_id"]),
            prompt=str(request["prompt"]),
            seed=int(request.get("seed", 0)),
            max_new_tokens=int(request.get("max_new_tokens", 256)),
        )
    elif operation == "train_lora_sanity":
        result = train_lora_sanity.remote(
            run_id=str(request["run_id"]),
            prompt=str(request["prompt"]),
            target=str(request["target"]),
            seed=int(request.get("seed", 0)),
            steps=int(request.get("steps", 1)),
            learning_rate=float(request.get("learning_rate", 0.0001)),
        )
    else:
        raise ValueError(f"unknown operation: {operation}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
