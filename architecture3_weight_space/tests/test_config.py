from pathlib import Path

import pytest

from tripathy_arch3.config import UNPINNED_REVISION, load_settings


def test_load_settings_and_redact_credentials(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ARCH3_MODEL_ID=Qwen/Qwen3.5-0.8B",
                "ARCH3_MODEL_REVISION=abc123",
                "ARCH3_MAX_SEQUENCE_LENGTH=2048",
                "HF_TOKEN=secret-value",
                "PAST_API_KEY=past-secret-value",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file=env_file, environ={})
    summary = settings.redacted_summary()

    assert settings.hf_token == "secret-value"
    assert summary["hf_token_present"] is True
    assert summary["past_api_key_present"] is True
    assert "secret-value" not in str(summary)
    assert "past-secret-value" not in str(summary)


def test_critic_validation_requires_matching_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ARCH3_CRITIC_PROVIDER=openai\nARCH3_CRITIC_MODEL=gpt-example\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_file, environ={})

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate(require_critic=True)


def test_model_revision_must_be_pinned_before_download(tmp_path: Path) -> None:
    settings = load_settings(env_file=tmp_path / "missing", environ={})

    assert settings.model_revision == UNPINNED_REVISION
    with pytest.raises(ValueError, match="must be pinned"):
        settings.validate(require_pinned_model=True)
