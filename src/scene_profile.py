#!/usr/bin/env python3
"""Scene profiles: what a specific simulator scene actually provides.

Why this exists
---------------
Different packaged scenes expose different vehicle names, different camera
names, and wildly different asset lists. Hard-coding any of them means the code
works on exactly one scene.

A profile is the small JSON that a smoke test produces once per scene:

    {
      "scene_id": "openuav_moderncity",
      "port": 30001,
      "vehicle": "Drone_1",
      "cameras": ["FrontCamera", "RearCamera", "LeftCamera", "RightCamera",
                  "DownCamera"],
      "assets": ["Cube", "Sphere", ...]
    }

Everything downstream reads the profile instead of assuming CityStreet's
conventions. Switching scenes then means switching profiles, not editing code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# AirSim camera names are arbitrary strings chosen in settings.json. These
# heuristics map a name onto the role it probably plays, so image capture can
# label its output sensibly without knowing the scene beforehand.
_NAME_HINTS: List[tuple] = [
    ("front", "front"),
    ("rear", "back"),
    ("back", "back"),
    ("left", "left"),
    ("right", "right"),
    ("down", "down"),
    ("bottom", "down"),
    ("up", "up"),
    ("top", "up"),
]

# Cameras whose names end like this are recording duplicates at a different
# resolution; keep them out of the default capture set.
_RECORD_SUFFIXES = ("record", "rec")


@dataclass
class SceneProfile:
    """What one simulator scene provides."""

    scene_id: str
    port: int
    vehicle: str
    cameras: Dict[str, str] = field(default_factory=dict)   # camera_name -> role
    assets: List[str] = field(default_factory=list)
    primitives: List[str] = field(default_factory=list)
    source: str = ""

    @property
    def asset_count(self) -> int:
        return len(self.assets)

    def has_asset(self, name: str) -> bool:
        return not self.assets or name in self.assets

    def pick_capture_cameras(self, exclude_record: bool = True) -> Dict[str, str]:
        """Camera name -> role, without the duplicate recording cameras."""
        if not exclude_record:
            return dict(self.cameras)
        kept = {
            name: role
            for name, role in self.cameras.items()
            if not name.lower().endswith(_RECORD_SUFFIXES)
        }
        return kept or dict(self.cameras)

    def describe(self) -> str:
        lines = [
            f"scene    : {self.scene_id}",
            f"port     : {self.port}",
            f"vehicle  : {self.vehicle}",
            f"cameras  : {list(self.cameras)}",
            f"assets   : {self.asset_count}",
        ]
        if self.primitives:
            lines.append(f"primitives: {self.primitives}")
        if self.source:
            lines.append(f"source   : {self.source}")
        return "\n".join(lines)


def role_for_camera(name: str) -> str:
    """Guess the viewing direction from a camera name."""
    lowered = name.lower()
    for needle, role in _NAME_HINTS:
        if needle in lowered:
            return role
    # Fall back to the name itself so nothing is silently dropped.
    return lowered


def _normalise_cameras(raw: Any) -> Dict[str, str]:
    """Accept either a list of names or a name->role mapping."""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(name): role_for_camera(str(name)) for name in raw}
    return {}


def from_dict(data: Dict[str, Any], source: str = "") -> SceneProfile:
    cameras = _normalise_cameras(data.get("cameras", []))
    assets = [str(a) for a in data.get("assets", [])]
    primitives = [str(p) for p in data.get("primitives_present", data.get("primitives", []))]
    return SceneProfile(
        scene_id=str(data.get("scene_id") or data.get("scene") or "unknown"),
        port=int(data.get("port", 30100)),
        vehicle=str(data.get("vehicle") or data.get("vehicle_used") or "Drone_1"),
        cameras=cameras,
        assets=assets,
        primitives=primitives,
        source=source,
    )


def load(path: str | Path) -> SceneProfile:
    """Load a profile file written by the smoke test.

    Read with utf-8-sig so a profile saved by a Windows editor (which may
    prepend a BOM) does not fail to parse. Harmless for BOM-free files.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"scene profile not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    return from_dict(data, source=str(p))


def save(profile: SceneProfile, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_id": profile.scene_id,
        "port": profile.port,
        "vehicle": profile.vehicle,
        "cameras": profile.cameras,
        "primitives_present": profile.primitives,
        "asset_count": profile.asset_count,
        "assets": profile.assets,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def choose_probe_asset(profile: SceneProfile, preferred: Sequence[str] = ()) -> Optional[str]:
    """Pick the most predictable asset available for a smoke probe."""
    for name in preferred:
        if name in profile.assets:
            return name
    for name in profile.primitives:
        if name in profile.assets:
            return name
    return profile.assets[0] if profile.assets else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a scene profile")
    parser.add_argument("profile", help="path to profile_*.json")
    args = parser.parse_args()

    prof = load(args.profile)
    print(prof.describe())
    print()
    print("capture cameras:")
    for name, role in prof.pick_capture_cameras().items():
        print(f"  {name:<18} -> {role}")
    print()
    print("first 20 assets:")
    for asset in prof.assets[:20]:
        print("  ", asset)