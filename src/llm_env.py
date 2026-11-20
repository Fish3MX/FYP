#!/usr/bin/env python3
"""Resolve LLM endpoint configuration from the environment or a .env file.

Why this exists
---------------
Credentials must never enter the repository. The convention is therefore:

  1. `export LLM_API_KEY=...` in the shell (highest priority, nothing on disk);
  2. or a `.env` file in the project root, which is git-ignored.

Both work without touching any code. This module is the single place that reads
them, so no other module has to care where the values came from.

Precedence: a real environment variable always wins over `.env`. That lets a
`.env` hold defaults while a shell export overrides them for one run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# The three settings every OpenAI-compatible endpoint needs.
ENV_KEYS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

# Sensible defaults so only the key is strictly required.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

ENV_FILENAME = ".env"


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from `start` looking for a `.env` file."""
    directory = (start or Path(__file__)).resolve()
    if directory.is_file():
        directory = directory.parent

    for candidate_dir in (directory, *directory.parents):
        candidate = candidate_dir / ENV_FILENAME
        if candidate.is_file():
            return candidate
    return None


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a minimal KEY=VALUE file.

    Supports blank lines, `#` comments, an optional leading `export`, single or
    double quotes, and trailing comments after an unquoted value. Deliberately
    small: no interpolation, no multiline values, no includes.
    """
    values: Dict[str, str] = {}
    # utf-8-sig transparently strips a BOM, which Windows editors (and
    # PowerShell's Set-Content) frequently prepend. Without this the first
    # key silently becomes "\ufeffLLM_API_KEY" and never matches.
    text = path.read_text(encoding="utf-8-sig")
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif "#" in value:
            # Unquoted values may carry a trailing comment.
            value = value.split("#", 1)[0].strip()

        if key:
            values[key] = value
    return values


def load_env_file(path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    """Load a `.env` file into os.environ.

    Returns the path that was used, or None when no file was found. Existing
    environment variables are preserved unless `override` is True.
    """
    env_path = path or find_env_file()
    if env_path is None or not env_path.is_file():
        return None

    for key, value in parse_env_file(env_path).items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


@dataclass
class LLMConfig:
    """Resolved endpoint settings."""

    api_key: str
    base_url: str
    model: str
    source: str = "environment"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def masked_key(self) -> str:
        """Never print a full secret; show enough to tell keys apart."""
        key = self.api_key
        if not key:
            return "(not set)"
        if len(key) <= 8:
            return "*" * len(key)
        return f"{key[:3]}{'*' * 6}{key[-4:]}"

    def describe(self) -> str:
        return (
            f"  source   : {self.source}\n"
            f"  base_url : {self.base_url}\n"
            f"  model    : {self.model}\n"
            f"  api_key  : {self.masked_key()}"
        )


def resolve_config(load_file: bool = True) -> LLMConfig:
    """Read the three settings, loading `.env` first when present."""
    source = "environment"
    if load_file:
        used = load_env_file()
        if used is not None:
            source = f"{ENV_FILENAME} ({used})"

    return LLMConfig(
        api_key=os.environ.get("LLM_API_KEY", "").strip(),
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip()
        or DEFAULT_BASE_URL,
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        source=source,
    )


if __name__ == "__main__":
    config = resolve_config()
    print("LLM configuration")
    print(config.describe())
    if not config.configured:
        print("\nLLM_API_KEY is not set. Either:")
        print("  export LLM_API_KEY=sk-xxxx")
        print("  or copy .env.example to .env and fill it in.")
        raise SystemExit(1)
    print("\nconfiguration looks complete")