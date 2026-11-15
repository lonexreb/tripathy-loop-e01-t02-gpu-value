"""Practice harness for author-style weight-space programming of the forecasting actor.

The actor forecasts the quantiles itself (optionally with thinking on). This module builds a
practice set from development-period data only, runs generation passes, classifies failures and
scores pinball loss. It is the editor's forward-pass tool, and `diagnose` is the first read:

    .venv-wsp/bin/python -m tripathy_arch3.wsp_practice diagnose

Practice data: headline posted-rental series, origins 2026-07-01..2026-09-30. Checkpoint and
protected origins (October) never enter this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any

from . import wsp
from .config import WORKSPACE_ROOT

REPO_ROOT = WORKSPACE_ROOT.parent
OUT = WORKSPACE_ROOT / "artifacts" / "wsp_author"
PRACTICE_START, PRACTICE_END = "2026-07-01", "2026-09-30"
CHOOSE_END = "2026-09-15"  # later practice origins confirm edits; they never choose them
HORIZONS = (1, 3, 7)
HISTORY_DAYS = 30
MIN_HISTORY = 20
SYSTEM = "You forecast posted GPU rental prices. Answer with one JSON object and nothing else."
AUTHOR_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
AUTHOR_TENSORS = (15, 19)


def _series() -> dict[str, list[tuple[str, float]]]:
    with (REPO_ROOT / "research" / "out" / "catalog_2026-11-06.csv").open() as handle:
        headline = {r["series_id"] for r in csv.DictReader(handle) if r["tier"] == "headline"}
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with (REPO_ROOT / "research" / "out" / "series_daily.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["series_id"] in headline:
                series[row["series_id"]].append((row["date"], float(row["value"])))
    return {name: sorted(rows) for name, rows in sorted(series.items())}


def practice_set() -> list[dict[str, Any]]:
    """One item per (series, origin) with at least MIN_HISTORY days of history and some truth."""
    from datetime import date, timedelta

    items: list[dict[str, Any]] = []
    for series_id, rows in _series().items():
        by_date = dict(rows)
        for origin, _ in rows:
            if not PRACTICE_START <= origin <= PRACTICE_END:
                continue
            history = [(d, v) for d, v in rows if d <= origin]
            if len(history) < MIN_HISTORY:
                continue
            start = date.fromisoformat(origin)
            truth = {h: by_date.get((start + timedelta(days=h)).isoformat()) for h in HORIZONS}
            # Truth must also lie inside the practice period: nothing from October leaks in.
            truth = {
                h: (y if (start + timedelta(days=h)).isoformat() <= PRACTICE_END else None)
                for h, y in truth.items()
            }
            if not any(y is not None for y in truth.values()):
                continue
            items.append(
                {
                    "series_id": series_id,
                    "origin": origin,
                    "history": history,
                    "truth": truth,
                    "split": "choose" if origin <= CHOOSE_END else "confirm",
                }
            )
    return items


def prompt(item: dict[str, Any]) -> str:
    recent = item["history"][-HISTORY_DAYS:]
    return json.dumps(
        {
            "task": (
                "Forecast this posted price 1, 3 and 7 days after the origin date. "
                "Give the 10%, 50% and 90% quantiles for each horizon."
            ),
            "series_id": item["series_id"],
            "instrument_class": item["series_id"].split(".")[1],
            "unit": "USD per GPU-hour",
            "origin_date": item["origin"],
            "daily_prices_oldest_first": [round(v, 4) for _, v in recent],
            "output_contract": {
                "h1": "[q10, q50, q90]",
                "h3": "[q10, q50, q90]",
                "h7": "[q10, q50, q90]",
            },
        },
        separators=(",", ":"),
    )


def pinball(y: float, q10: float, q50: float, q90: float) -> float:
    def one(q: float, t: float) -> float:
        return max(t * (y - q), (t - 1) * (y - q))

    return (one(q10, 0.1) + one(q50, 0.5) + one(q90, 0.9)) / 3


def drift_quantiles(item: dict[str, Any], horizon: int) -> tuple[float, float, float]:
    """Starter-style drift: last value times empirical quantiles of h-day log changes."""
    values = [v for _, v in item["history"]]
    last = values[-1]
    changes = sorted(
        math.log(b / a) for a, b in zip(values, values[horizon:], strict=False) if a > 0 and b > 0
    )
    if len(changes) < MIN_HISTORY:
        return last, last, last

    def quantile(p: float) -> float:
        position = p * (len(changes) - 1)
        low = int(position)
        high = min(low + 1, len(changes) - 1)
        return changes[low] + (changes[high] - changes[low]) * (position - low)

    return tuple(last * math.exp(quantile(p)) for p in (0.1, 0.5, 0.9))


def parse(text: str, hit_limit: bool) -> tuple[str, dict[int, tuple[float, float, float]]]:
    """Failure class and the valid per-horizon quantiles found in one answer."""
    if "<think>" in text and "</think>" not in text:
        return ("token_limit_in_thinking" if hit_limit else "unclosed_thinking"), {}
    answer = text.split("</think>")[-1]
    match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
    if not match:
        return ("token_limit_no_answer" if hit_limit else "no_json"), {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ("token_limit_cut_json" if hit_limit else "invalid_json"), {}
    quantiles: dict[int, tuple[float, float, float]] = {}
    for horizon in HORIZONS:
        triple = payload.get(f"h{horizon}") if isinstance(payload, dict) else None
        if (
            isinstance(triple, list)
            and len(triple) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in triple)
            and 0 < triple[0] <= triple[1] <= triple[2]
        ):
            quantiles[horizon] = (float(triple[0]), float(triple[1]), float(triple[2]))
    if len(quantiles) == len(HORIZONS):
        return "ok", quantiles
    return ("partial_quantiles" if quantiles else "bad_quantiles"), quantiles


def generate(
    tokenizer: Any,
    model: Any,
    items: list[dict[str, Any]],
    *,
    thinking: bool,
    max_new_tokens: int,
    batch_size: int,
    sampling: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Batched generation; one record per item with raw text and token count.

    Greedy by default. `sampling` uses the released checkpoint's evaluation settings:
    temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    class PresencePenalty(LogitsProcessor):  # type: ignore[misc]
        def __init__(self, prompt_length: int) -> None:
            self.prompt_length = prompt_length

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, input_ids[:, self.prompt_length :], True)
            return scores - 1.5 * seen

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        chats = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(item)}]
            for item in batch
        ]
        encoded = tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=thinking,
        ).to(model.device)
        options: dict[str, Any] = {"do_sample": False}
        if sampling:
            torch.manual_seed(seed + start)
            options = {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "logits_processor": LogitsProcessorList(
                    [PresencePenalty(encoded["input_ids"].shape[1])]
                ),
            }
        with torch.no_grad():
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, **options)
        new = output[:, encoded["input_ids"].shape[1] :]
        for item, ids in zip(batch, new, strict=True):
            kept = ids[ids != tokenizer.pad_token_id]
            ended = bool((ids == tokenizer.eos_token_id).any()) or len(kept) < max_new_tokens
            text = tokenizer.decode(kept, skip_special_tokens=False)
            text = re.sub(r"<\|[a-z_]+\|>", "", text).strip()
            if thinking and "</think>" not in text:
                text = "<think>" + text
            records.append(
                {"item": item, "text": text, "tokens": int(len(kept)), "hit_limit": not ended}
            )
    return records


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Failure shares and pinball loss; invalid cells fall back to the last value, as the
    official scorer does for missing cells."""
    classes: Counter[str] = Counter()
    losses: dict[str, list[float]] = defaultdict(list)
    anchored, widths, inside = [], [], []
    for record in records:
        item = record["item"]
        kind, quantiles = parse(record["text"], record["hit_limit"])
        record["class"] = kind
        classes[kind] += 1
        last = item["history"][-1][1]
        for horizon, y in item["truth"].items():
            if y is None:
                continue
            q10, q50, q90 = quantiles.get(horizon, (last, last, last))
            losses["actor"].append(pinball(y, q10, q50, q90))
            losses["last_value"].append(pinball(y, last, last, last))
            losses["drift"].append(pinball(y, *drift_quantiles(item, horizon)))
            if horizon in quantiles:
                anchored.append(abs(q50 / last - 1))
                widths.append((q90 - q10) / last)
                inside.append(q10 <= y <= q90)
                losses["actor_valid_cells"].append(pinball(y, q10, q50, q90))
                losses["drift_same_cells"].append(pinball(y, *drift_quantiles(item, horizon)))

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    def median(values: list[float]) -> float | None:
        return round(sorted(values)[len(values) // 2], 5) if values else None

    return {
        "n": len(records),
        "classes": dict(classes.most_common()),
        "mean_tokens": mean([float(r["tokens"]) for r in records]),
        "hit_limit_share": mean([float(r["hit_limit"]) for r in records]),
        "pinball": {name: mean(values) for name, values in losses.items()},
        "valid_cells": len(anchored),
        "median_abs_q50_move_vs_last": median(anchored),
        "median_relative_width": median(widths),
        "coverage_on_valid_cells": mean([float(v) for v in inside]),
    }


def apply_author_edit(model: Any) -> dict[str, Any]:
    """Patch the two tensors the released checkpoint changed; report what differs."""
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    cache = str(WORKSPACE_ROOT / ".hf-cache")
    files = [f for f in list_repo_files(AUTHOR_REPO) if f.endswith(".safetensors")]
    report: dict[str, Any] = {}
    for layer in AUTHOR_TENSORS:
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        theirs = None
        for file in files:
            with safe_open(hf_hub_download(AUTHOR_REPO, file, cache_dir=cache), "pt") as handle:
                if name in handle.keys():  # noqa: SIM118 - safetensors handle, not a dict
                    theirs = handle.get_tensor(name).to(torch.float32)
        if theirs is None:
            raise RuntimeError(f"released checkpoint has no tensor {name}")
        weight = model.get_submodule(f"model.layers.{layer}.mlp.down_proj").weight
        diff = theirs.to(weight.device) - weight
        report[f"layer{layer}"] = {
            "changed_columns": int((diff.abs().sum(dim=0) > 0).sum()),
            "changed_weights": int((diff != 0).sum()),
            "relative_change": round(float(diff.norm() / weight.norm()), 4),
        }
        with torch.no_grad():
            weight.copy_(theirs.to(weight.device))
    return report


def diagnose(n: int, max_new_tokens: int, batch_size: int) -> None:
    items = [item for item in practice_set() if item["split"] == "choose"]
    sample = random.Random(0).sample(items, n)
    tokenizer, model = wsp._load()
    result: dict[str, Any] = {
        "practice_items": len(practice_set()),
        "choose_items": len(items),
        "sample": n,
        "max_new_tokens": max_new_tokens,
        "arms": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)

    def arm(name: str, thinking: bool) -> None:
        records = generate(
            tokenizer,
            model,
            sample,
            thinking=thinking,
            max_new_tokens=max_new_tokens if thinking else 200,
            batch_size=batch_size,
        )
        result["arms"][name] = score(records)
        with (OUT / f"diagnose_{name}.jsonl").open("w") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "series_id": record["item"]["series_id"],
                            "origin": record["item"]["origin"],
                            "class": record["class"],
                            "tokens": record["tokens"],
                            "text": record["text"],
                        }
                    )
                    + "\n"
                )
        print(name, json.dumps(result["arms"][name]))

    arm("base_no_thinking", False)
    arm("base_thinking", True)
    result["author_edit"] = apply_author_edit(model)
    print("author_edit", json.dumps(result["author_edit"]))
    arm("author_no_thinking", False)
    arm("author_thinking", True)
    (OUT / "diagnose.json").write_text(json.dumps(result, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    diag = commands.add_parser("diagnose")
    diag.add_argument("--n", type=int, default=60)
    diag.add_argument("--max-new-tokens", type=int, default=1200)
    diag.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    diagnose(args.n, args.max_new_tokens, args.batch_size)


if __name__ == "__main__":
    main()
