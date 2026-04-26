from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)


ProviderName = Literal["openai", "anthropic", "lmstudio"]
Mode = Literal["dev", "safe", "direct"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


ALL_PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic", "lmstudio")
LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_DEFAULT_API_KEY = "lm-studio"


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid. Safe to display."""


def _mask(secret: SecretStr | None) -> str:
    if secret is None:
        return "<unset>"
    raw = secret.get_secret_value()
    if len(raw) <= 8:
        return "***"
    return f"{raw[:4]}…{raw[-4:]}"


class Settings(BaseModel):
    providers: list[ProviderName] = Field(default_factory=lambda: list(ALL_PROVIDERS))
    default_provider: ProviderName = "anthropic"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-4-7"
    lmstudio_base_url: str = LMSTUDIO_DEFAULT_BASE_URL
    lmstudio_model: str | None = None
    lmstudio_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(LMSTUDIO_DEFAULT_API_KEY)
    )
    mode: Mode = "safe"
    houdini_host: str = "127.0.0.1"
    houdini_port: int = 18861
    hython_path: Path | None = None
    log_level: LogLevel = "INFO"
    log_dir: Path = Field(default_factory=lambda: Path("logs"))

    model_config = {"frozen": True}

    @field_validator("houdini_port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("must be between 1 and 65535")
        return v

    @field_validator("houdini_host")
    @classmethod
    def _host_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be non-empty")
        return v

    @field_validator("lmstudio_base_url")
    @classmethod
    def _lmstudio_url_shape(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be non-empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("providers")
    @classmethod
    def _providers_nonempty_and_known(
        cls, v: list[ProviderName]
    ) -> list[ProviderName]:
        if not v:
            raise ValueError(
                f"PROVIDERS must list at least one of {list(ALL_PROVIDERS)}"
            )
        # de-dupe while preserving order
        seen: list[ProviderName] = []
        for name in v:
            if name not in seen:
                seen.append(name)
        return seen

    @model_validator(mode="after")
    def _validate_active_provider(self) -> "Settings":
        if self.default_provider not in self.providers:
            raise ValueError(
                f"DEFAULT_PROVIDER={self.default_provider} is not in "
                f"PROVIDERS={','.join(self.providers)}"
            )
        if self.default_provider == "openai" and self.openai_api_key is None:
            raise ValueError(
                "DEFAULT_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
        if self.default_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError(
                "DEFAULT_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set"
            )
        if self.default_provider == "lmstudio":
            if not self.lmstudio_model:
                raise ValueError(
                    "DEFAULT_PROVIDER=lmstudio but LMSTUDIO_MODEL is not set"
                )
            if not self.lmstudio_api_key.get_secret_value():
                raise ValueError(
                    "DEFAULT_PROVIDER=lmstudio but LMSTUDIO_API_KEY is empty"
                )
        return self

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)

        raw = {
            "providers": _parse_csv(os.getenv("PROVIDERS"), list(ALL_PROVIDERS)),
            "default_provider": _norm_lower(os.getenv("DEFAULT_PROVIDER"), "anthropic"),
            "openai_api_key": _secret(os.getenv("OPENAI_API_KEY")),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "anthropic_api_key": _secret(os.getenv("ANTHROPIC_API_KEY")),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            "lmstudio_base_url": os.getenv(
                "LMSTUDIO_BASE_URL", LMSTUDIO_DEFAULT_BASE_URL
            ),
            "lmstudio_model": (os.getenv("LMSTUDIO_MODEL") or None),
            "lmstudio_api_key": _secret(os.getenv("LMSTUDIO_API_KEY"))
            or SecretStr(LMSTUDIO_DEFAULT_API_KEY),
            "mode": _norm_lower(os.getenv("MODE"), "safe"),
            "houdini_host": os.getenv("HOUDINI_HOST", "127.0.0.1"),
            "houdini_port": _parse_int("HOUDINI_PORT", os.getenv("HOUDINI_PORT"), 18861),
            "hython_path": _parse_path(os.getenv("HYTHON_PATH")),
            "log_level": _norm_upper(os.getenv("AIBRIDGE_LOG_LEVEL"), "INFO"),
            "log_dir": Path(os.getenv("AIBRIDGE_LOG_DIR", "logs")),
        }

        try:
            return cls(**raw)
        except ValidationError as e:
            raise ConfigError(_format_validation_error(e)) from e

    def safe_summary(self) -> dict[str, Any]:
        """Dict suitable for logging — secrets are masked."""
        return {
            "providers": ",".join(self.providers),
            "default_provider": self.default_provider,
            "openai_api_key": _mask(self.openai_api_key),
            "openai_model": self.openai_model,
            "anthropic_api_key": _mask(self.anthropic_api_key),
            "anthropic_model": self.anthropic_model,
            "lmstudio_base_url": self.lmstudio_base_url,
            "lmstudio_model": self.lmstudio_model or "<unset>",
            "lmstudio_api_key": _mask(self.lmstudio_api_key),
            "mode": self.mode,
            "houdini_host": self.houdini_host,
            "houdini_port": self.houdini_port,
            "hython_path": str(self.hython_path) if self.hython_path else "<unset>",
            "log_level": self.log_level,
            "log_dir": str(self.log_dir),
        }

    def __repr__(self) -> str:  # never leak secrets via repr/print
        return f"Settings({self.safe_summary()})"

    __str__ = __repr__


def _secret(value: str | None) -> SecretStr | None:
    if value is None:
        return None
    value = value.strip()
    return SecretStr(value) if value else None


def _norm_lower(value: str | None, default: str) -> str:
    return (value or default).strip().lower()


def _norm_upper(value: str | None, default: str) -> str:
    return (value or default).strip().upper()


def _parse_int(name: str, value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from e


def _parse_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return default
    items = [v.strip().lower() for v in value.split(",")]
    return [v for v in items if v]


def _format_validation_error(err: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for issue in err.errors():
        loc = ".".join(str(p) for p in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']}")
    return "\n".join(lines)
