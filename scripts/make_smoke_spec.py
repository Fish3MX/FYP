#!/usr/bin/env python3
"""Write a minimal scenario spec that is guaranteed to match a real scene.

The reference spec (`specs/fire_floor2.json`) was written for CityStreet and
names assets that may not exist elsewhere. This script instead reads a scene
profile produced by `scripts/smoke_test_scene.py` and emits a spec using only
assets that the scene actually reported.

The result is deliberately trivial: one object, one spawn pose, no requirements.
Its purpose is to prove the transport path (spec -> spawn -> capture) on a new
scene before anything interesting is attempted.

Usage
-----
    python scripts/make_smoke_spec.py \
        --profile runs/smoke_openuav/profile_openuav_moderncity.json \
        --out specs/openuav_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import scene_profile


def build_spec(profile: scene_profile.SceneProfile, asset: str, scale: float) -> dict:
    """A one-object scenario, positioned relative to the world origin."""
    return {
        "version": "0.1",
        "name": f"{profile.scene_id}_smoke",
        "seed": 1,
        "origin": [0.0, 0.0, 0.0],
        "objects": [
            {
                "id": "smoke_object",
                "asset": asset,
                "pos": [8.0, 0.0, 2.0],
                "scale": [scale, scale, scale],
                "physics": False,
                "role": "prop",
            }
        ],
        "target": {"object_id": "smoke_object", "landing_radius_m": 4.0},
        "spawn": {"pos": [-14.0, 0.0, 8.0], "yaw_deg": 0.0},
        "requirements": [],
        "instruction": "Fly forward and inspect the test object.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a minimal spec matching a scene profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", required=True, help="profile_*.json from the smoke test")
    parser.add_argument("--out", default="specs/openuav_smoke.json")
    parser.add_argument("--asset", default="", help="force a specific asset name")
    parser.add_argument("--scale", type=float, default=3.0, help="uniform scale for the object")
    args = parser.parse_args(argv)

    try:
        profile = scene_profile.load(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot load the profile: {exc}")
        return 2

    print(profile.describe())
    print()

    if args.asset:
        if not profile.has_asset(args.asset):
            print(f"warning: '{args.asset}' was not in the profile's asset list")
        asset = args.asset
    else:
        asset = scene_profile.choose_probe_asset(profile, preferred=("Cube", "Sphere", "Plane"))
        if asset is None:
            print("the profile lists no assets; run the smoke test first")
            return 3

    print(f"using asset: {asset}")

    spec = build_spec(profile, asset, args.scale)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"spec written to {out_path}")
    print()
    print("next command (note --no-verify: the base map's geometry is not modelled,")
    print("so the pure-geometry verifier cannot judge occlusion here):")
    print(
        f"  python scripts/generate_and_render.py \\\n"
        f"      --spec {out_path} \\\n"
        f"      --scene-profile {args.profile} \\\n"
        f"      --no-verify \\\n"
        f"      --out runs/loop1"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())