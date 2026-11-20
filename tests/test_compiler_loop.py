#!/usr/bin/env python3
"""Tests for the compile-and-retry loop and credential loading.

These run entirely offline: the LLM transport is replaced by a fake, so the
feedback loop can be exercised deterministically and for free.

What is being proven here
-------------------------
The central claim of the project is that generation is *compiled*, not sampled:
a bad scenario is detected and sent back to the model rather than accepted. The
tests below assert exactly that behaviour, including the case where the model
never converges.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import llm_compiler  # noqa: E402
import llm_env  # noqa: E402


# --------------------------------------------------------------------------- #
# Deterministic transport
# --------------------------------------------------------------------------- #
class FakeTransport:
    """Replaces call_chat_api with a scripted sequence of replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        # Snapshot the conversation: compile_request appends to the live list,
        # so storing a reference would show the final state in every call.
        self.calls.append([dict(message) for message in messages])
        if not self.replies:
            raise AssertionError("the transport was called more often than scripted")
        return self.replies.pop(0)


def sealed_room_spec() -> dict:
    """A room with no opening: structurally valid, geometrically unsolvable."""

    def box(obj_id, pos, scale, role="structure"):
        return {
            "id": obj_id,
            "asset": "Cube",
            "pos": pos,
            "scale": scale,
            "physics": True,
            "role": role,
        }

    return {
        "version": "0.1",
        "name": "sealed_room",
        "seed": 1,
        "origin": [0.0, 0.0, 0.0],
        "objects": [
            box("ground", [0, 0, -0.4], [30, 30, 1.2]),
            box("floor", [0, 0, 0.1], [10, 10, 0.2]),
            box("ceiling", [0, 0, 3.1], [10, 10, 0.2]),
            box("wall_north", [0, 5, 1.6], [10, 0.2, 3]),
            box("wall_south", [0, -5, 1.6], [10, 0.2, 3]),
            box("wall_east", [5, 0, 1.6], [0.2, 10, 3]),
            box("wall_west", [-5, 0, 1.6], [0.2, 10, 3]),
            {
                "id": "goal",
                "asset": "BP_ma_construction_worker_01",
                "pos": [0, 0, 0.3],
                "physics": True,
                "role": "target",
            },
        ],
        "target": {"object_id": "goal", "landing_radius_m": 2.0},
        "spawn": {"pos": [-20, 0, 8], "yaw_deg": 0},
        "requirements": [],
    }


def reference_spec() -> dict:
    """The known-good reference scenario, which passes every gate."""
    path = PROJECT_ROOT / "specs" / "fire_floor2.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Compile-and-retry loop
# --------------------------------------------------------------------------- #
class TestGeometryFeedbackLoop(unittest.TestCase):
    def test_retries_after_geometric_failure(self):
        """A sealed scenario must be rejected and sent back, not accepted."""
        transport = FakeTransport(
            [json.dumps(sealed_room_spec()), json.dumps(reference_spec())]
        )
        with patch.object(llm_compiler, "call_chat_api", transport):
            spec = llm_compiler.compile_request(
                "search for a survivor in a building",
                base_url="http://test/v1",
                api_key="test",
                model="test-model",
                max_attempts=3,
            )

        self.assertEqual(spec.name, "fire_floor2")
        self.assertEqual(len(transport.calls), 2, "the loop should have retried once")

        # The second request must carry a concrete diagnostic, not a generic retry.
        second_conversation = "\n".join(
            message["content"] for message in transport.calls[1]
        )
        self.assertIn("no feasible path", second_conversation)
        self.assertIn("Hints for the geometric failures", second_conversation)

    def test_gives_up_and_reports_problems(self):
        """When the model never converges the failure must be explicit."""
        transport = FakeTransport(
            [json.dumps(sealed_room_spec()) for _ in range(3)]
        )
        with patch.object(llm_compiler, "call_chat_api", transport):
            with self.assertRaises(RuntimeError) as context:
                llm_compiler.compile_request(
                    "search for a survivor in a building",
                    base_url="http://test/v1",
                    api_key="test",
                    model="test-model",
                    max_attempts=3,
                )

        self.assertEqual(len(transport.calls), 3)
        self.assertIn("no feasible path", str(context.exception))

    def test_geometry_can_be_disabled_for_the_ablation(self):
        """Disabling verification accepts the sealed scenario on the first pass.

        This is the ablation condition: it measures what verification is worth.
        """
        transport = FakeTransport([json.dumps(sealed_room_spec())])
        with patch.object(llm_compiler, "call_chat_api", transport):
            spec = llm_compiler.compile_request(
                "search for a survivor in a building",
                base_url="http://test/v1",
                api_key="test",
                model="test-model",
                max_attempts=3,
                verify_geometry=False,
            )

        self.assertEqual(spec.name, "sealed_room")
        self.assertEqual(len(transport.calls), 1)

    def test_structural_failure_is_also_fed_back(self):
        """Missing fields are reported before geometry is even reached."""
        broken = reference_spec()
        broken["objects"] = []                     # structurally invalid
        transport = FakeTransport([json.dumps(broken), json.dumps(reference_spec())])
        with patch.object(llm_compiler, "call_chat_api", transport):
            spec = llm_compiler.compile_request(
                "search for a survivor",
                base_url="http://test/v1",
                api_key="test",
                model="test-model",
                max_attempts=3,
            )

        self.assertEqual(spec.name, "fire_floor2")
        second = "\n".join(m["content"] for m in transport.calls[1])
        self.assertIn("failed validation", second)


# --------------------------------------------------------------------------- #
# Credential loading
# --------------------------------------------------------------------------- #
class TestLLMEnv(unittest.TestCase):
    def test_parses_export_prefix_and_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                'export LLM_API_KEY="sk-abc"\n'
                "LLM_BASE_URL='https://example.test/v1'\n"
                "LLM_MODEL=my-model   # trailing comment\n",
                encoding="utf-8",
            )
            parsed = llm_env.parse_env_file(path)

        self.assertEqual(parsed["LLM_API_KEY"], "sk-abc")
        self.assertEqual(parsed["LLM_BASE_URL"], "https://example.test/v1")
        self.assertEqual(parsed["LLM_MODEL"], "my-model")

    def test_ignores_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# a comment\n\n   \nLLM_MODEL=m\n", encoding="utf-8"
            )
            parsed = llm_env.parse_env_file(path)

        self.assertEqual(parsed, {"LLM_MODEL": "m"})

    def test_strips_windows_bom(self):
        """Editors on Windows silently prepend a BOM; the first key must survive."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_bytes(
                "\ufeffLLM_API_KEY=sk-bom\n".encode("utf-8")
            )
            parsed = llm_env.parse_env_file(path)

        self.assertEqual(parsed, {"LLM_API_KEY": "sk-bom"})

    def test_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"LLM_MODEL": "from-env"}, clear=False):
                os.environ.pop("LLM_MODEL", None)
                os.environ["LLM_MODEL"] = "from-env"
                llm_env.load_env_file(path, override=False)
                self.assertEqual(os.environ["LLM_MODEL"], "from-env")

    def test_env_file_fills_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
            os.environ.pop("LLM_MODEL", None)
            llm_env.load_env_file(path, override=False)
            self.assertEqual(os.environ["LLM_MODEL"], "from-file")

    def test_masked_key_hides_the_secret(self):
        config = llm_env.LLMConfig(
            api_key="sk-1234567890abcdef", base_url="u", model="m"
        )
        masked = config.masked_key()
        self.assertNotIn("1234567890", masked)
        self.assertTrue(masked.startswith("sk-"))
        self.assertTrue(masked.endswith("cdef"))

    def test_missing_key_is_reported_as_not_configured(self):
        config = llm_env.LLMConfig(api_key="", base_url="u", model="m")
        self.assertFalse(config.configured)
        self.assertEqual(config.masked_key(), "(not set)")

    def test_find_env_file_walks_upward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("LLM_MODEL=m\n", encoding="utf-8")
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)
            found = llm_env.find_env_file(nested / "module.py")
            self.assertEqual(found, root / ".env")

    def test_find_env_file_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A temp directory whose ancestors are outside the repo.
            self.assertIsNone(llm_env.find_env_file(Path(tmp) / "module.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)