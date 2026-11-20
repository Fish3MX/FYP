#!/usr/bin/env python3
"""Compile a natural-language test request into a scenario specification.

Core idea
---------
The task is not "let a large model generate a 3D world". It is "let a large
model write a constrained configuration file". The format is defined by
`scene_spec.py`, and the permitted asset names by `asset_catalog.py`. The model
plans (what to place, where, and what the task is); it does not render.

Three layers of constraint, from soft to hard
---------------------------------------------
1. **Prompt layer**: states the coordinate system, the field meanings, the
   available assets and the object-count limits. The model can still violate it.
2. **Validation layer**: `scene_spec.parse_spec()` rejects missing fields, wrong
   types, duplicate ids, and a target referencing a non-existent object.
3. **Feedback layer**: when validation fails, the concrete problem list is sent
   back to the model for a rewrite, for up to N rounds. This is the module's
   most important design point: the model is not expected to be right first
   time, it is driven to correctness by the validator.

Asset whitelisting runs on top of that: any asset name not present in
`asset_catalog` counts as a hallucination and also triggers a rewrite.

Offline mode
------------
Without an API key, `--offline` produces a specification from a deterministic
template. This is not a fallback hack; it is a useful control condition. The
effect of the LLM compiler should be reported against template generation,
otherwise there is no way to say what the model contributed.

Usage
-----
    # Needs an API key (any OpenAI-compatible endpoint)
    export LLM_API_KEY=sk-xxxx
    export LLM_BASE_URL=https://api.deepseek.com/v1      # or another provider
    export LLM_MODEL=deepseek-chat

    python src/llm_compiler.py \\
        --request "test the drone searching for a survivor in a burning two-storey flat" \\
        --out specs/generated_fire.json

    # No API: use the template and still run the whole chain
    python src/llm_compiler.py --request "burning two-storey flat" \\
        --offline --out specs/generated_fire.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from asset_catalog import prompt_text, whitelist
from scene_spec import SceneSpec, SpecError, parse_spec

LOG = logging.getLogger("llm_compiler")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_ATTEMPTS = 3

# Field reference embedded in the prompt.
SCHEMA_BRIEF = """
{
  "version": "0.1",
  "name": "snake_case_scenario_name",
  "seed": 42,
  "origin": [x_east, y_north, z_up],
  "objects": [
    {
      "id": "unique_id",
      "asset": "<an asset name from the catalogue>",
      "pos": [x_east, y_north, z_up],
      "yaw_deg": 0.0,
      "scale": [sx, sy, sz],
      "physics": true,
      "role": "structure | prop | target | hazard"
    }
  ],
  "target": { "object_id": "id_of_the_goal_object", "landing_radius_m": 4.0 },
  "spawn": { "pos": [x_east, y_north, z_up], "yaw_deg": 0.0 },
  "requirements": [ { "kind": "max_time_s", "value": 300 } ],
  "instruction": "one or two sentences telling the drone what to do"
}
"""


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def system_prompt() -> str:
    return textwrap.dedent(
        f"""\
        You are a scenario compiler for a UAV simulation benchmark.

        Your job: turn a natural-language test request into ONE JSON object that
        describes a flyable 3D test scenario for Unreal Engine / AirSim.

        ## Coordinate system (critical)
        All positions use ENU, in METRES:
          - x = east, y = north, z = UP
          - yaw_deg = rotation about the vertical axis, counter-clockwise from east
        The scene is built around `origin`. Object `pos` values are RELATIVE to
        `origin`; `spawn.pos` is also relative to `origin`.
        You never write AirSim's NED frame. A converter handles that.

        ## Scale
        The base primitives (Cube, Plane) are 1 m across at scale 1.
        To make a wall 12 m long, 0.2 m thick and 3 m tall:
          "asset": "Cube", "scale": [12.0, 0.2, 3.0]
        Non-primitive assets ignore this and use their modelled size.

        ## Asset catalogue
        You may ONLY use asset names listed below. Inventing a name is a hard
        failure and the scenario will be rejected.

        {prompt_text()}

        ## Build rules
        1. 12 to 40 objects total. Prefer `Cube` for walls and floors: it is the
           only asset whose dimensions you control exactly.
        2. Walls MUST have "physics": true, otherwise the drone flies through
           them. Decorative props should have "physics": false or omitted.
        3. Every building needs an opening. If you enclose a room completely with
           walls, the scenario is unsolvable. Leave a doorway at least 4 m wide,
           because a 0.6 m safety clearance shrinks it considerably.
        4. Always include a ground slab under the structure. A floor modelled as
           a floating slab leaves the space underneath open, so the drone can
           fly under the building and reach the target from below.
        5. Exactly one object must have "role": "target", and
           `target.object_id` must equal that object's id.
        6. The target must be hidden from `spawn`: place it behind a wall or
           inside the structure, not in open line of sight.
        7. Multi-storey interiors need an opening in the upper floor
           (a stairwell), otherwise the upper floor is unreachable.
        8. `spawn.pos` must be OUTSIDE the structure and at least 10 m away,
           with z of 5-15 m (i.e. airborne). The drone starts already flying.
        9. Keep the whole scene within roughly 40 m x 40 m around `origin`.

        ## Output format
        Return ONLY the JSON object. No prose, no markdown fences, no comments.
        The object must match this schema:

        {SCHEMA_BRIEF}
        """
    )


def user_prompt(request: str) -> str:
    return (
        "Compile this test request into a scenario specification:\n\n"
        f"{request}\n\n"
        "Return only the JSON object."
    )


# --------------------------------------------------------------------------- #
# HTTP call (standard library only, to avoid a hard dependency)
# --------------------------------------------------------------------------- #
def call_chat_api(
    messages: Sequence[Dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    temperature: float = 0.2,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return the text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach the LLM endpoint {url}: {exc.reason}\n"
            f"Check LLM_BASE_URL and your network or proxy settings."
        ) from exc

    try:
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"unexpected LLM response payload: {body[:500]}"
        ) from exc


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models are frequently disobedient: markdown fences, prose before and after,
    or trailing commas. Try progressively more permissive strategies.
    """
    candidates: List[str] = []

    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1))

    candidates.append(text)

    # Slice from the first '{' to the last '}'.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Optional[Exception] = None
    for raw in candidates:
        cleaned = raw.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\s*//.*$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(
        f"no parsable JSON object in the model output. First 600 characters:\n"
        f"{text[:600]}\nparse error: {last_error}"
    )


# --------------------------------------------------------------------------- #
# Constraint checks
# --------------------------------------------------------------------------- #
def check_assets_whitelisted(spec: SceneSpec) -> List[str]:
    """Return assets used by the spec that are absent from the catalogue."""
    allowed = whitelist()
    return sorted({o.asset for o in spec.objects if o.asset not in allowed})


def check_scenario_sanity(spec: SceneSpec) -> List[str]:
    """Checks that go beyond structural validity.

    Structural validation only proves the fields are well formed. These rules
    ask whether the scenario is meaningful *as a test*. In the proposal they
    correspond to RQ2's non-triviality and initial-occlusion conditions.
    """
    problems: List[str] = []

    if len(spec.objects) < 5:
        problems.append(
            f"only {len(spec.objects)} objects; the scenario is too simple "
            f"(at least 5 required)"
        )

    if spec.target is None:
        problems.append("missing target: a search task must name the goal object")
    if spec.spawn is None:
        problems.append("missing spawn: the drone start pose is required")

    targets = [o for o in spec.objects if o.role == "target"]
    if spec.target is not None:
        if len(targets) != 1:
            problems.append(
                f"exactly one object must have role=target, found {len(targets)}"
            )
        elif spec.target.object_id != targets[0].id:
            problems.append(
                f"target.object_id='{spec.target.object_id}' does not match the "
                f"role=target object '{targets[0].id}'"
            )

    structures = [o for o in spec.objects if o.role == "structure"]
    if not structures:
        problems.append("no role=structure objects; the scene has no solid environment")
    for obj in structures:
        if not obj.physics:
            problems.append(
                f"structure '{obj.id}' has physics=false, the drone would fly "
                f"straight through it"
            )

    if spec.spawn is not None and spec.target is not None:
        goal = spec.object_by_id(spec.target.object_id)
        if goal is not None:
            if spec.spawn.pos[2] < 3.0:
                problems.append(
                    f"spawn altitude {spec.spawn.pos[2]:.1f} m is too low; "
                    f"the drone should start at 5-15 m"
                )
            horizontal = (
                (spec.spawn.pos[0] - goal.pos[0]) ** 2
                + (spec.spawn.pos[1] - goal.pos[1]) ** 2
            ) ** 0.5
            if horizontal < 8.0:
                problems.append(
                    f"spawn is only {horizontal:.1f} m from the target; "
                    f"the task is too short (at least 8 m)"
                )

    return problems


def validate_all(spec: SceneSpec) -> List[str]:
    """Combine whitelist and sanity checks into a single problem list."""
    missing = check_assets_whitelisted(spec)
    problems = [
        f"asset '{asset}' is not in the permitted catalogue; "
        f"use a name that exists there"
        for asset in missing
    ]
    problems += check_scenario_sanity(spec)
    return problems


# --------------------------------------------------------------------------- #
# Offline template (deterministic baseline)
# --------------------------------------------------------------------------- #
def template_spec(request: str) -> Dict[str, Any]:
    """Build a deterministic specification from keywords in the request.

    Two purposes:
      1. It keeps the whole pipeline runnable without an API key.
      2. It is the control condition: reporting what the LLM adds requires a
         rule-based baseline to compare against.
    """
    text = request.lower()

    objects: List[Dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        objects.append(kwargs)

    # Ground base: a floating floor slab leaves the space underneath open, so
    # the drone could enter from below and the room would not be sealed.
    add(id="ground_base", asset="Cube", pos=[0, 0, -0.4], scale=[24, 20, 1.2],
        physics=True, role="structure")

    add(id="floor_ground", asset="Cube", pos=[0, 0, 0.1], scale=[12, 8, 0.2],
        physics=True, role="structure")

    # The first floor is split into four slabs, leaving a 3x3 m stairwell,
    # otherwise the upper floor cannot be reached.
    for tag, pos, scale in [
        ("north", [0, 3.0, 3.1], [12, 2.0, 0.2]),
        ("south", [0, -2.5, 3.1], [12, 3.0, 0.2]),
        ("west", [-5.5, 0.5, 3.1], [1.0, 3.0, 0.2]),
        ("east", [2.0, 0.5, 3.1], [8.0, 3.0, 0.2]),
    ]:
        add(id=f"floor_second_{tag}", asset="Cube", pos=pos, scale=scale,
            physics=True, role="structure")

    add(id="roof", asset="Cube", pos=[0, 0, 6.1], scale=[12, 8, 0.2],
        physics=True, role="structure")

    # Ground floor: a 4 m doorway in the south wall. A 2 m opening is closed
    # off once a 0.6 m safety clearance is rasterised.
    add(id="l1_wall_north", asset="Cube", pos=[0, 4, 1.6], scale=[12, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_south_left", asset="Cube", pos=[-5, -4, 1.6], scale=[2, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_south_right", asset="Cube", pos=[3, -4, 1.6], scale=[6, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_east", asset="Cube", pos=[6, 0, 1.6], scale=[0.2, 8, 3],
        physics=True, role="structure")
    add(id="l1_wall_west", asset="Cube", pos=[-6, 0, 1.6], scale=[0.2, 8, 3],
        physics=True, role="structure")

    # Upper floor: fully enclosed, so the target is genuinely hidden.
    for tag, pos, scale in [
        ("north", [0, 4, 4.6], [12, 0.2, 3]),
        ("south", [0, -4, 4.6], [12, 0.2, 3]),
        ("east", [6, 0, 4.6], [0.2, 8, 3]),
        ("west", [-6, 0, 4.6], [0.2, 8, 3]),
    ]:
        add(id=f"l2_wall_{tag}", asset="Cube", pos=pos, scale=scale,
            physics=True, role="structure")

    add(id="survivor_1", asset="BP_ma_construction_worker_01",
        pos=[2, 2, 3.2], yaw_deg=210, physics=True, role="target")

    # Domain-specific extension driven by keywords.
    if any(k in text for k in ("fire", "burn", "smoke", "flame")):
        add(id="fire_light_2f", asset="PointLightBP", pos=[0, 0, 3.6], role="hazard")
        add(id="fire_marker_2f", asset="Sphere", pos=[0.5, 0.5, 3.5],
            scale=[0.5, 0.5, 0.5], role="hazard")
        add(id="fireplace_1f", asset="SM_Fireplace_01a", pos=[-4, 2, 0.2],
            physics=True, role="hazard")

    if any(k in text for k in ("car", "vehicle", "street", "urban", "traffic")):
        for index, (px, py) in enumerate([(-9, -6), (-9, 6), (9, -5)]):
            add(id=f"car_{index}", asset="SM_Sedan03_Main", pos=[px, py, 0.2],
                yaw_deg=90 * index, physics=True, role="prop")

    if any(k in text for k in ("tree", "forest", "outdoor", "wilderness")):
        for index, (px, py) in enumerate([(-10, 9), (10, 9), (-11, -9)]):
            add(id=f"tree_{index}", asset="SM_tree_pine_a", pos=[px, py, 0.2],
                physics=True, role="prop")

    if any(k in text for k in ("rubble", "collapse", "disaster", "earthquake")):
        for index, (px, py) in enumerate([(-3, -6), (4, 5)]):
            add(id=f"rubble_{index}", asset="SM_Rubble01", pos=[px, py, 0.3],
                physics=True, role="prop")

    if any(k in text for k in ("inspection", "infrastructure", "pipe", "industrial")):
        for index in range(4):
            add(id=f"pillar_{index}", asset="Cube",
                pos=[-4 + index * 2.5, -2, 1.6], scale=[0.4, 0.4, 3],
                physics=True, role="structure")

    # Decoration.
    add(id="door_entry", asset="SM_build_b_door_wood_single",
        pos=[-3, -3.9, 1.0], role="prop")
    add(id="window_north", asset="SM_build_b_mod_lvl2_doublewindow",
        pos=[2, 3.85, 4.6], yaw_deg=90, role="prop")

    # Task and metrics.
    name = "scenario"
    for key, slug in [
        ("fire", "fire"), ("burn", "fire"),
        ("search", "search"), ("rescue", "search"),
        ("inspection", "inspection"), ("inspect", "inspection"),
    ]:
        if key in text:
            name = f"{slug}_building"
            break

    return {
        "version": "0.1",
        "name": name,
        "seed": 42,
        "origin": [25.0, 0.0, 0.0],
        "objects": objects,
        "target": {"object_id": "survivor_1", "landing_radius_m": 4.0},
        "spawn": {"pos": [-28.0, 0.0, 8.0], "yaw_deg": 0.0},
        "requirements": [
            {"kind": "max_time_s", "value": 600},
            {"kind": "min_altitude_m", "value": 1.5},
            {"kind": "no_collision", "value": True},
        ],
        "instruction": (
            "Take off, fly east toward the two-storey building, enter through the "
            "ground-floor doorway, and locate the person on the second floor. "
            "Land within 4 metres of them."
        ),
    }


# --------------------------------------------------------------------------- #
# Main compile loop
# --------------------------------------------------------------------------- #
def compile_request(
    request: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    temperature: float = 0.2,
) -> SceneSpec:
    """Generate, validate, feed back, repeat until a valid spec appears.

    Returns a validated SceneSpec. Raises RuntimeError when the retry budget is
    exhausted, with the full problem list from the last round.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(request)},
    ]

    all_problems: List[str] = []

    for attempt in range(1, max_attempts + 1):
        LOG.info("generation attempt %d/%d", attempt, max_attempts)
        raw = call_chat_api(
            messages,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

        # --- parse JSON ---
        try:
            data = extract_json(raw)
        except ValueError as exc:
            problems = [str(exc)]
            LOG.warning("the reply was not valid JSON; retrying")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    "Your reply was not valid JSON. Problems:\n- "
                    + "\n- ".join(problems)
                    + "\n\nReturn ONLY the JSON object, no fences, no prose."},
            ]
            all_problems = problems
            continue

        # --- structural validation ---
        try:
            spec = parse_spec(data)
        except SpecError as exc:
            problems = exc.problems
            LOG.warning(
                "structural validation failed with %d problem(s); sending them back",
                len(problems),
            )
            messages += [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {"role": "user", "content":
                    "That specification failed validation. Fix ALL of these and "
                    "return the complete corrected JSON object:\n- "
                    + "\n- ".join(problems)},
            ]
            all_problems = problems
            continue

        # --- semantic and asset validation ---
        problems = validate_all(spec)
        if problems:
            LOG.warning(
                "sanity checks found %d problem(s); sending them back", len(problems)
            )
            messages += [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {"role": "user", "content":
                    "The scenario is structurally valid but not usable as a test. "
                    "Fix ALL of these and return the complete corrected JSON:\n- "
                    + "\n- ".join(problems)},
            ]
            all_problems = problems
            continue

        LOG.info("attempt %d passed every check", attempt)
        return spec

    raise RuntimeError(
        f"no valid specification after {max_attempts} attempts. "
        f"Problems from the last round:\n"
        + "\n".join(f"  - {p}" for p in all_problems)
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Natural-language request -> scenario specification JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--request", required=True, help="natural-language test request")
    parser.add_argument("--out", default="", help="output path (empty prints to stdout)")
    parser.add_argument(
        "--offline", action="store_true", help="skip the LLM and use the template"
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip geometric verification (solvability, occlusion, non-triviality)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- offline template ----
    if args.offline:
        LOG.info("offline mode: generating from the deterministic template")
        data = template_spec(args.request)
        try:
            spec = parse_spec(data)
        except SpecError as exc:
            LOG.error("the template produced an invalid spec (this is a bug):\n%s", exc)
            return 1
        problems = validate_all(spec)
        if problems:
            LOG.error("the template failed the sanity checks (this is a bug):")
            for problem in problems:
                LOG.error("  - %s", problem)
            return 1
        if not args.no_verify and not _verify_spec(spec):
            LOG.error("the template failed geometric verification (this is a bug)")
            return 1
        return _emit(spec, data, args.out)

    # ---- online mode ----
    if not args.api_key:
        LOG.error(
            "no API key. Two options:\n"
            "  1) export LLM_API_KEY=sk-xxxx  (optionally LLM_BASE_URL / LLM_MODEL)\n"
            "  2) add --offline to run the chain with the template"
        )
        return 2

    try:
        spec = compile_request(
            args.request,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
        )
    except RuntimeError as exc:
        LOG.error("compilation failed: %s", exc)
        return 3

    if not args.no_verify and not _verify_spec(spec):
        LOG.error(
            "the model produced a scenario that fails geometric verification. "
            "The validate-and-retry loop does not cover the verifier yet; "
            "wiring it in is the next step."
        )
        return 4

    return _emit(spec, spec_to_dict(spec), args.out)


def spec_to_dict(spec: SceneSpec) -> Dict[str, Any]:
    """Serialise a SceneSpec back to a dict, omitting default values."""
    return {
        "version": spec.version,
        "name": spec.name,
        "seed": spec.seed,
        "origin": list(spec.origin),
        "objects": [
            {
                "id": obj.id,
                "asset": obj.asset,
                "pos": list(obj.pos),
                **({"yaw_deg": obj.yaw_deg} if obj.yaw_deg else {}),
                **(
                    {"scale": list(obj.scale)}
                    if tuple(obj.scale) != (1.0, 1.0, 1.0)
                    else {}
                ),
                **({"physics": obj.physics} if obj.physics else {}),
                "role": obj.role,
            }
            for obj in spec.objects
        ],
        **(
            {"target": {
                "object_id": spec.target.object_id,
                "landing_radius_m": spec.target.landing_radius_m,
            }}
            if spec.target else {}
        ),
        **(
            {"spawn": {"pos": list(spec.spawn.pos), "yaw_deg": spec.spawn.yaw_deg}}
            if spec.spawn else {}
        ),
        "requirements": spec.requirements,
        "instruction": spec.instruction,
    }


def _verify_spec(spec: SceneSpec) -> bool:
    """Run geometric verification and print the summary.

    Imported locally so callers that only compile are unaffected by changes in
    the verifier's dependencies.
    """
    import verifier as _verifier

    result = _verifier.verify(spec)
    print(result.summary())
    print()
    return result.usable


def _emit(spec: SceneSpec, data: Dict[str, Any], out: str) -> int:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        LOG.info("specification written to %s", out)
    else:
        print(text)

    print()
    print(spec.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())