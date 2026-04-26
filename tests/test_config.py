from __future__ import annotations

import pytest

from aibridge_houdini.config import ConfigError, Settings


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for name in [
        "PROVIDERS",
        "DEFAULT_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "LMSTUDIO_BASE_URL",
        "LMSTUDIO_MODEL",
        "LMSTUDIO_API_KEY",
        "MODE",
        "HOUDINI_HOST",
        "HOUDINI_PORT",
        "HYTHON_PATH",
        "AIBRIDGE_LOG_LEVEL",
        "AIBRIDGE_LOG_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)


def _load_no_dotenv() -> Settings:
    return Settings.load(env_file=None)


def test_defaults_require_provider_key():
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_openai_provider_requires_openai_key(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_loads_with_anthropic_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    s = _load_no_dotenv()
    assert s.default_provider == "anthropic"
    assert s.mode == "safe"
    assert s.houdini_host == "127.0.0.1"
    assert s.houdini_port == 18861
    assert s.hython_path is None


def test_invalid_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("MODE", "yolo")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "mode" in str(exc.value).lower()


def test_invalid_provider(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "groq")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "default_provider" in str(exc.value).lower()


def test_invalid_port_non_integer(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("HOUDINI_PORT", "not-a-number")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "HOUDINI_PORT" in str(exc.value)


def test_invalid_port_out_of_range(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("HOUDINI_PORT", "70000")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "houdini_port" in str(exc.value).lower()


def test_provider_value_is_normalized(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "OpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    s = _load_no_dotenv()
    assert s.default_provider == "openai"


def test_secrets_are_not_in_repr_or_summary(monkeypatch):
    secret = "sk-ant-supersecret-shouldnotappear"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    s = _load_no_dotenv()

    rep = repr(s)
    summary = str(s.safe_summary())
    assert secret not in rep
    assert secret not in summary
    # masked form should still hint at the value
    assert s.safe_summary()["anthropic_api_key"].startswith("sk-a")
    assert s.safe_summary()["openai_api_key"] == "<unset>"
    # actual secret is still retrievable when explicitly requested
    assert s.anthropic_api_key.get_secret_value() == secret


def test_hython_path_parsed(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("HYTHON_PATH", str(tmp_path / "hython"))
    s = _load_no_dotenv()
    assert s.hython_path == tmp_path / "hython"


# ---- LM Studio + PROVIDERS ----------------------------------------------


def test_default_providers_includes_lmstudio(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    s = _load_no_dotenv()
    assert s.providers == ["openai", "anthropic", "lmstudio"]


def test_providers_csv_parsed_and_normalized(monkeypatch):
    monkeypatch.setenv("PROVIDERS", " OpenAI , Anthropic , LMStudio , openai ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    s = _load_no_dotenv()
    assert s.providers == ["openai", "anthropic", "lmstudio"]


def test_default_provider_must_be_in_providers(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "openai,anthropic")
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    msg = str(exc.value)
    assert "DEFAULT_PROVIDER=lmstudio" in msg
    assert "PROVIDERS=openai,anthropic" in msg


def test_lmstudio_unknown_value_in_providers_rejected(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "openai,llama")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "providers" in str(exc.value).lower()


def test_lmstudio_loads_with_defaults(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder-7b")
    s = _load_no_dotenv()
    assert s.default_provider == "lmstudio"
    assert s.lmstudio_base_url == "http://localhost:1234/v1"
    assert s.lmstudio_model == "qwen2.5-coder-7b"
    assert s.lmstudio_api_key.get_secret_value() == "lm-studio"


def test_lmstudio_requires_model(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "LMSTUDIO_MODEL" in str(exc.value)


def test_lmstudio_invalid_base_url(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "localhost:1234/v1")
    with pytest.raises(ConfigError) as exc:
        _load_no_dotenv()
    assert "lmstudio_base_url" in str(exc.value).lower()


def test_lmstudio_base_url_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1/")
    s = _load_no_dotenv()
    assert s.lmstudio_base_url == "http://localhost:1234/v1"


def test_lmstudio_empty_api_key_rejected(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "   ")
    # whitespace-only is treated as absent and falls back to the default
    s = _load_no_dotenv()
    assert s.lmstudio_api_key.get_secret_value() == "lm-studio"


def test_lmstudio_api_key_is_masked_in_summary(monkeypatch):
    secret = "lms-supersecret-shouldnotappear"
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder")
    monkeypatch.setenv("LMSTUDIO_API_KEY", secret)
    s = _load_no_dotenv()
    summary = s.safe_summary()
    assert secret not in str(summary)
    assert secret not in repr(s)
    assert summary["lmstudio_api_key"] != secret
    assert summary["lmstudio_model"] == "qwen2.5-coder"
    assert s.lmstudio_api_key.get_secret_value() == secret


def test_existing_openai_path_still_works(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    s = _load_no_dotenv()
    assert s.default_provider == "openai"
    assert "openai" in s.providers
