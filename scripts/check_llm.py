#!/usr/bin/env python3
"""Check the LLM endpoint configuration with one minimal request.

Purpose
-------
Before running a compilation, confirm three things: the key is present, the base
URL is reachable, and the model name is accepted. A five-token request costs
almost nothing and turns a confusing failure inside the compiler into a clear
diagnosis here.

Configuration comes from `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`, either as
real environment variables or from a `.env` file in the project root. See
`.env.example`.

Usage
-----
    python scripts/check_llm.py
    python scripts/check_llm.py -v
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_compiler import call_chat_api  # noqa: E402
from llm_env import resolve_config  # noqa: E402

LOG = logging.getLogger("check_llm")

DIAGNOSIS = {
    "401": "the API key was rejected. Re-copy it from the console; watch for a "
           "trailing space or a truncated paste.",
    "403": "the key is valid but not allowed to use this model or endpoint.",
    "404": "the base URL or the model name is wrong. Check that the URL ends "
           "with the /v1 style prefix and that the model exists.",
    "429": "rate limited or out of quota. Check the account balance.",
    "500": "the provider had a server error; retry shortly.",
    "503": "the provider is temporarily unavailable; retry shortly.",
}


def diagnose(message: str) -> str:
    """Turn an exception message into an actionable hint."""
    for code, hint in DIAGNOSIS.items():
        if f"HTTP {code}" in message:
            return hint
    if "cannot reach" in message or "timed out" in message:
        return (
            "the endpoint is unreachable from this machine. Check the network, "
            "and set https_proxy / http_proxy if the lab network needs a proxy."
        )
    if "unexpected LLM response payload" in message:
        return (
            "the endpoint answered, but not in OpenAI chat-completions format. "
            "The base URL probably points at a non-compatible API."
        )
    return "no specific hint available; read the error above."


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the LLM endpoint configuration")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = resolve_config()

    print("LLM configuration")
    print(config.describe())
    print()

    if not config.configured:
        print("LLM_API_KEY is not set. Either:")
        print("  export LLM_API_KEY=sk-xxxx")
        print("  or: cp .env.example .env  and fill in the key")
        return 2

    messages = [
        {
            "role": "user",
            "content": "Reply with exactly one word: ok",
        }
    ]

    print("sending a minimal request ...")
    started = time.time()
    try:
        reply = call_chat_api(
            messages,
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=30.0,
            temperature=0.0,
        )
    except RuntimeError as exc:
        elapsed = time.time() - started
        print(f"FAILED after {elapsed:.2f} s")
        print(f"  {exc}")
        print()
        print("what this usually means:")
        print(f"  {diagnose(str(exc))}")
        return 1

    elapsed = time.time() - started
    snippet = " ".join(reply.split())[:120]
    print(f"OK  responded in {elapsed:.2f} s")
    print(f"  reply: {snippet!r}")
    print()
    print("the endpoint is configured correctly; you can now run:")
    print('  python src/llm_compiler.py --request "<your test request>" --out specs/x.json')
    return 0


if __name__ == "__main__":
    sys.exit(main())