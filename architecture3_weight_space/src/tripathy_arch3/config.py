"""Typed, redaction-safe runtime configuration for Architecture 3."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = WORKSPACE_ROOT / ".env"
UNPINNED_REVISION = "PIN_BEFORE_FIRST_MODEL_DOWNLOAD"


def _optional(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


@dataclass(frozen=True)
class Architecture3Settings:
    model_id: str
    model_revision: str
    modal_profile: str
    modal_gpu: str
    modal_volume: str
    max_sequence_length: int
    critic_provider: str | None
    critic_model: str | None
    hf_token: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    exa_api_key: str | None
    past_api_key: str | None

    def validate(self, *, require_pinned_model: bool = False, require_critic: bool = False) -> None:
        errors: list[str] = []
        if not self.model_id:
            errors.append("ARCH3_MODEL_ID is required")
        if require_pinned_model and self.model_revision == UNPINNED_REVISION:
            errors.append("ARCH3_MODEL_REVISION must be pinned before downloading weights")
        if self.max_sequence_length <= 0:
            errors.append("ARCH3_MAX_SEQUENCE_LENGTH must be positive")
        if require_critic:
            if self.critic_provider not in {"openai", "anthropic"}:
                errors.append("ARCH3_CRITIC_PROVIDER must be openai or anthropic")
            if not self.critic_model:
                errors.append("ARCH3_CRITIC_MODEL is required for critic calls")
            if self.critic_provider == "openai" and not self.openai_api_key:
                errors.append("OPENAI_API_KEY is required for the selected critic")
            if self.critic_provider == "anthropic" and not self.anthropic_api_key:
                errors.append("ANTHROPIC_API_KEY is required for the selected critic")
        if errors:
            raise ValueError("; ".join(errors))

    def redacted_summary(self) -> dict[str, str | int | bool | None]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "modal_profile": self.modal_profile,
            "modal_gpu": self.modal_gpu,
            "modal_volume": self.modal_volume,
            "max_sequence_length": self.max_sequence_length,
            "critic_provider": self.critic_provider,
            "critic_model": self.critic_model,
            "hf_token_present": bool(self.hf_token),
            "openai_api_key_present": bool(self.openai_api_key),
            "anthropic_api_key_present": bool(self.anthropic_api_key),
            "exa_api_key_present": bool(self.exa_api_key),
            "past_api_key_present": bool(self.past_api_key),
        }


def load_settings(
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> Architecture3Settings:
    file_values = {key: value or "" for key, value in dotenv_values(env_file).items()}
    environment = dict(os.environ if environ is None else environ)
    values = {**file_values, **environment}
    settings = Architecture3Settings(
        model_id=values.get("ARCH3_MODEL_ID", "Qwen/Qwen3.5-0.8B").strip(),
        model_revision=values.get("ARCH3_MODEL_REVISION", UNPINNED_REVISION).strip(),
        modal_profile=values.get("ARCH3_MODAL_PROFILE", "reach2shubhankar").strip(),
        modal_gpu=values.get("ARCH3_MODAL_GPU", "L4").strip(),
        modal_volume=values.get("ARCH3_MODAL_VOLUME", "tripathy-e01-arch3-weights").strip(),
        max_sequence_length=int(values.get("ARCH3_MAX_SEQUENCE_LENGTH", "2048")),
        critic_provider=_optional(values.get("ARCH3_CRITIC_PROVIDER")),
        critic_model=_optional(values.get("ARCH3_CRITIC_MODEL")),
        hf_token=_optional(values.get("HF_TOKEN")),
        openai_api_key=_optional(values.get("OPENAI_API_KEY")),
        anthropic_api_key=_optional(values.get("ANTHROPIC_API_KEY")),
        exa_api_key=_optional(values.get("EXA_API_KEY")),
        past_api_key=_optional(values.get("PAST_API_KEY")),
    )
    settings.validate()
    return settings
