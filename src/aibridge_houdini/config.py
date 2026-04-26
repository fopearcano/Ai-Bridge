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


ProviderName = Literal["openai", "anthropic"]
Mode = Literal["dev", "safe", "direct"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


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
    default_provider: ProviderName = "anthropic"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-4-7"
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

    @model_validator(mode="after")
    def _provider_key_present(self) -> "Settings":
        if self.default_provider == "openai" and self.openai_api_key is None:
            raise ValueError(
                "DEFAULT_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
        if self.default_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError(
                "DEFAULT_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set"
            )
        return self

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)

        raw = {
            "default_provider": _norm_lower(os.getenv("DEFAULT_PROVIDER"), "anthropic"),
            "openai_api_key": _secret(os.getenv("OPENAI_API_KEY")),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "anthropic_api_key": _secret(os.getenv("ANTHROPIC_API_KEY")),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
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
            "default_provider": self.default_provider,
            "openai_api_key": _mask(self.openai_api_key),
            "openai_model": self.openai_model,
            "anthropic_api_key": _mask(self.anthropic_api_key),
            "anthropic_model": self.anthropic_model,
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


def _format_validation_error(err: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for issue in err.errors():
        loc = ".".join(str(p) for p in issue["loc"]) or "<root>"
        lines.append(f"  - {loc}: {issue['msg']}")
    return "\n".join(lines)
