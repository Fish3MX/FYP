#!/usr/bin/env python3
"""Scene specification: 
    -definition
    -validation
    -coordinate conversion

Design rules
------------
1. **Specifications are written in ENU metres.** x=east, y=north, z=up, and
   yaw measured counter-clockwise from east. This is what a language model
   produces naturally ("5 m north, 3 m east, 2 m up"), and it removes a whole
   class of sign errors.

2. **Conversion happens in exactly one place.** AirSim uses NED (x=north,
   y=east, z=down, yaw clockwise from north). Every ENU->NED conversion goes
   through `enu_to_ned()` and `yaw_enu_to_ned()`. No other module may compute
   one. Scattered conversions are the single most common bug in this kind of
   system.

3. **A specification is pure data.** It references no AirSim objects, so it can
   be generated, validated, verified and unit-tested on a CPU with no engine.

Specification layout (v0.1)
---------------------------
{
  "version": "0.1",
  "name": "fire_floor2",
  "seed": 42,
  "origin": {"enu": [40.0, -20.0, 0.0]},   # local origin in world ENU
  "objects": [
    {
      "id": "wall_north",                    # unique within the scenario
      "asset": "Cube",                       # must exist in simListAssets()
      "pos": [0.0, 5.0, 1.5],               # ENU, relative to origin (metres)
      "yaw_deg": 90.0,                       # CCW from east
      "scale": [1.0, 1.0, 1.0],
      "physics": true,                       # participates in collision
      "role": "structure"                    # structure/prop/target/hazard
    }
  ],
  "target": {"object_id": "survivor_1", "landing_radius_m": 5.0},
  "spawn": {"pos": [-30.0, 0.0, 10.0], "yaw_deg": 0.0},
  "requirements": [
    {"kind": "max_time_s", "value": 300},
    {"kind": "min_altitude_m", "value": 3.0}
  ],
  "instruction": "Fly to the two-storey building and find the person upstairs."
}

Usage
-----
    from scene_spec import load_spec, enu_to_ned
    spec = load_spec("specs/fire_floor2.json")   # raises SpecError if invalid
    print(spec.summary())
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SPEC_VERSION = "0.1"

# Roles decide which checks apply to an object: structure must collide,
# a target is what the agent searches for, a hazard is a domain marker.
VALID_ROLES = {"structure", "prop", "target", "hazard"}


class SpecError(ValueError):
    """Raised when a specification fails validation.

    Carries every problem found, not just the first one, so a generator can be
    told to fix all of them in a single retry instead of one per round.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            "specification validation failed with %d problem(s):\n" % len(self.problems)
            + "\n".join("  - " + p for p in self.problems)
        )


# --------------------------------------------------------------------------- #
# Coordinate conversion
# --------------------------------------------------------------------------- #
def enu_to_ned(enu: Sequence[float]) -> Tuple[float, float, float]:
    """ENU (east, north, up) -> NED (north, east, down), all in metres.

        x_ned =  y_enu   (north)
        y_ned =  x_enu   (east)
        z_ned = -z_enu   (up becomes down)
    """
    e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
    return (n, e, -u)


def ned_to_enu(ned: Sequence[float]) -> Tuple[float, float, float]:
    """NED -> ENU. Used to write AirSim poses back in human-readable form."""
    n, e, d = float(ned[0]), float(ned[1]), float(ned[2])
    return (e, n, -d)


def yaw_enu_to_ned(yaw_deg: float) -> float:
    """Convert a yaw angle from the ENU convention to the AirSim convention.

    ENU: counter-clockwise from east. AirSim/NED: clockwise from north.

        yaw_ned = 90 - yaw_enu   (normalised to [0, 360))

    Self-check:
        yaw_enu = 0   (east)  -> yaw_ned = 90  (east in NED)
        yaw_enu = 90  (north) -> yaw_ned = 0   (north in NED)
        yaw_enu = 180 (west)  -> yaw_ned = 270
    """
    return (90.0 - float(yaw_deg)) % 360.0


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class SceneObject:
    id: str
    asset: str
    pos: Tuple[float, float, float] # ENU, relative to origin
    yaw_deg: float = 0.0
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    physics: bool = False
    role: str = "prop"


@dataclass
class Target:
    object_id: str
    landing_radius_m: float = 5.0


@dataclass
class Spawn:
    pos: Tuple[float, float, float]  # ENU, relative to origin
    yaw_deg: float = 0.0


@dataclass
class SceneSpec:
    version: str
    name: str
    seed: int
    origin: Tuple[float, float, float] # world ENU
    objects: List[SceneObject]
    target: Optional[Target] = None
    spawn: Optional[Spawn] = None
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    instruction: str = ""

    # ---------- convenience ----------
    def object_by_id(self, obj_id: str) -> Optional[SceneObject]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def world_position_enu(self, obj: SceneObject) -> Tuple[float, float, float]:
        """Object position in world ENU coordinates."""
        return (
            self.origin[0] + obj.pos[0],
            self.origin[1] + obj.pos[1],
            self.origin[2] + obj.pos[2],
        )

    def world_point_enu(self, local: Sequence[float]) -> Tuple[float, float, float]:
        """Convert any local point to world ENU.

        `spawn.pos` and every object `pos` are local to `origin`. Any
        computation that mixes them with world-space bounding boxes (line of
        sight, rasterisation, path planning) must go through here first. Mixing
        the two frames silently yields "never occluded" and implausible path
        lengths.
        """
        return (
            self.origin[0] + float(local[0]),
            self.origin[1] + float(local[1]),
            self.origin[2] + float(local[2]),
        )

    def world_spawn_enu(self) -> Optional[Tuple[float, float, float]]:
        """Drone start position in world ENU, or None if spawn is unset."""
        if self.spawn is None:
            return None
        return self.world_point_enu(self.spawn.pos)

    def asset_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for obj in self.objects:
            hist[obj.asset] = hist.get(obj.asset, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        lines = [
            f"scenario '{self.name}'  v{self.version}  seed={self.seed}",
            f"  origin (ENU) : {self.origin}",
            f"  objects      : {len(self.objects)}",
            f"  roles        : "
            + ", ".join(
                f"{role}={sum(1 for o in self.objects if o.role == role)}"
                for role in sorted(VALID_ROLES)
            ),
            f"  target       : {self.target.object_id if self.target else '(none)'}",
            f"  spawn        : {self.spawn.pos if self.spawn else '(none)'}",
            f"  requirements : {len(self.requirements)}",
        ]
        if self.instruction:
            lines.append(f"  instruction  : {self.instruction[:70]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #
def _as_vec3(value: Any, where: str, problems: List[str]) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        problems.append(f"{where}: expected an array of 3 numbers, got {value!r}")
        return (0.0, 0.0, 0.0)
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        problems.append(f"{where}: elements must be numbers, got {value!r}")
        return (0.0, 0.0, 0.0)


def _as_float(
    value: Any, where: str, problems: List[str], default: float = 0.0
) -> float:
    """Convert to float leniently.

    Language models frequently write numbers as strings ("4.0") or with units
    ("4 m"). Accept plain numbers and numeric strings; record a problem for
    anything else and keep going, so validation reports every issue at once
    instead of dying on the first type error.
    """
    if isinstance(value, bool):  # bool is a subclass of int, reject it first
        problems.append(f"{where}: expected a number, got boolean {value!r}")
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            problems.append(f"{where}: expected a number, got string {value!r}")
            return default
    problems.append(f"{where}: expected a number, got {value!r}")
    return default


def parse_spec(data: Dict[str, Any]) -> SceneSpec:
    """Parse a dict into a SceneSpec, collecting every problem before raising."""
    problems: List[str] = []

    if not isinstance(data, dict):
        raise SpecError(["the specification must be a JSON object"])

    version = str(data.get("version", ""))
    if version != SPEC_VERSION:
        problems.append(f"version: expected '{SPEC_VERSION}', got '{version}'")

    name = str(data.get("name", "")).strip()
    if not name:
        problems.append("name: must not be empty")

    try:
        seed = int(data.get("seed", 0))
    except (TypeError, ValueError):
        problems.append(f"seed: must be an integer, got {data.get('seed')!r}")
        seed = 0

    origin = _as_vec3(data.get("origin", [0, 0, 0]), "origin", problems)

    # ---- objects ----
    raw_objects = data.get("objects", [])
    objects: List[SceneObject] = []
    seen_ids: set[str] = set()

    if not isinstance(raw_objects, list) or not raw_objects:
        problems.append("objects: must be a non-empty array")
        raw_objects = []

    for idx, raw in enumerate(raw_objects):
        where = f"objects[{idx}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be an object")
            continue

        obj_id = str(raw.get("id", "")).strip()
        if not obj_id:
            problems.append(f"{where}.id: must not be empty")
            continue
        if obj_id in seen_ids:
            problems.append(f"{where}.id: duplicate id '{obj_id}'")
        seen_ids.add(obj_id)

        asset = str(raw.get("asset", "")).strip()
        if not asset:
            problems.append(
                f"{where}.asset: must not be empty "
                f"(use a name returned by simListAssets)"
            )

        pos = _as_vec3(raw.get("pos", [0, 0, 0]), f"{where}.pos", problems)
        scale = _as_vec3(raw.get("scale", [1, 1, 1]), f"{where}.scale", problems)

        if any(s <= 0 for s in scale):
            problems.append(f"{where}.scale: all components must be > 0, got {scale}")

        role = str(raw.get("role", "prop"))
        if role not in VALID_ROLES:
            problems.append(
                f"{where}.role: must be one of {sorted(VALID_ROLES)}, got '{role}'"
            )

        try:
            yaw = float(raw.get("yaw_deg", 0.0))
        except (TypeError, ValueError):
            problems.append(f"{where}.yaw_deg: must be a number")
            yaw = 0.0

        objects.append(
            SceneObject(
                id=obj_id,
                asset=asset,
                pos=pos,
                yaw_deg=yaw,
                scale=scale,
                physics=bool(raw.get("physics", False)),
                role=role,
            )
        )

    # ---- target ----
    target: Optional[Target] = None
    raw_target = data.get("target")
    if raw_target is not None:
        if not isinstance(raw_target, dict):
            problems.append("target: must be an object")
        else:
            tid = str(raw_target.get("object_id", "")).strip()
            if not tid:
                problems.append("target.object_id: must not be empty")
            elif tid not in seen_ids:
                problems.append(
                    f"target.object_id: '{tid}' is not one of the object ids"
                )
            radius = _as_float(
                raw_target.get("landing_radius_m", 5.0),
                "target.landing_radius_m",
                problems,
                default=5.0,
            )
            if radius <= 0:
                problems.append("target.landing_radius_m: must be > 0")
            target = Target(object_id=tid, landing_radius_m=radius)

    # ---- spawn ----
    spawn: Optional[Spawn] = None
    raw_spawn = data.get("spawn")
    if raw_spawn is not None:
        if not isinstance(raw_spawn, dict):
            problems.append("spawn: must be an object")
        else:
            spos = _as_vec3(raw_spawn.get("pos", [0, 0, 0]), "spawn.pos", problems)
            syaw = _as_float(
                raw_spawn.get("yaw_deg", 0.0), "spawn.yaw_deg", problems, default=0.0
            )
            spawn = Spawn(pos=spos, yaw_deg=syaw)

    # ---- requirements ----
    requirements: List[Dict[str, Any]] = []
    raw_reqs = data.get("requirements", [])
    if not isinstance(raw_reqs, list):
        problems.append("requirements: must be an array")
    else:
        for idx, req in enumerate(raw_reqs):
            if not isinstance(req, dict) or "kind" not in req:
                problems.append(
                    f"requirements[{idx}]: must be an object with a 'kind' field"
                )
                continue
            requirements.append(dict(req))

    if problems:
        raise SpecError(problems)

    return SceneSpec(
        version=version,
        name=name,
        seed=seed,
        origin=origin,
        objects=objects,
        target=target,
        spawn=spawn,
        requirements=requirements,
        instruction=str(data.get("instruction", "")),
    )


def load_spec(path: str | Path) -> SceneSpec:
    """Load and validate a specification file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"specification file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise SpecError([f"invalid JSON: {exc}"]) from exc
    return parse_spec(data)


def check_assets(spec: SceneSpec, available: Sequence[str]) -> List[str]:
    """Return the assets used by the spec that the scene cannot provide.

    An empty list means every object can be instantiated.
    """
    available_set = set(available)
    return sorted({o.asset for o in spec.objects if o.asset not in available_set})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a scenario specification")
    parser.add_argument("spec", help="path to the specification JSON")
    args = parser.parse_args()

    try:
        loaded = load_spec(args.spec)
    except SpecError as exc:
        print(exc)
        raise SystemExit(1)

    print(loaded.summary())
    print("\nassets used:")
    for asset_name, count in loaded.asset_histogram().items():
        print(f"  {count:3d} x {asset_name}")