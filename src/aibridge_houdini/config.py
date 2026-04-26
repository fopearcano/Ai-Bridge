from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


ProviderName = Literal["openai", "anthropic"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseModel):
    provider: ProviderName = "anthropic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-7"
    log_level: LogLevel = "INFO"
    log_dir: Path = Field(default_factory=lambda: Path("logs"))

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        return cls(
            provider=os.getenv("AIBRIDGE_PROVIDER", "anthropic"),  # type: ignore[arg-type]
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            log_level=os.getenv("AIBRIDGE_LOG_LEVEL", "INFO"),  # type: ignore[arg-type]
            log_dir=Path(os.getenv("AIBRIDGE_LOG_DIR", "logs")),
        )
