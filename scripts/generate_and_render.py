#!/usr/bin/env python3
"""Closed-loop generator: a sentence becomes an inspected, rendered scene.

Four stages, each one independently switchable so a failure can be attributed:

     stage 1  compile   natural language -> specification       (llm_compiler)
     stage 2  verify    is it a usable test?                    (verifier)
     stage 3  build     spawn the objects into AirSim           (scene_builder)
     stage 4  render    place the drone, capture camera views   (this file)

Why the stages are switchable
-----------------------------
The chain touches a language model, a geometric reasoner and a game engine. If
it is only ever run as one opaque command, any failure looks the same. The flags
let you climb a ladder: check the script offline, then the builder with a known
good spec, then rendering, and only then the model.

What "empty scene" means here (and does not)
--------------------------------------------
The packaged simulator is a city street containing thousands of pre-existing
objects. This tool adds a generated building on top of it. It cannot start from
a blank world; that would need a separate UE project and a new package.

Camera convention
-----------------
The four cameras are bolted to the airframe and defined in settings.json:
0=front, 1=left, 2=right, 3=down. Orientation is therefore controlled by yawing
the vehicle, not by pitching it: the dedicated downward camera already gives the
overhead view. Keeping pitch at zero avoids attitude gymnastics and the physics
that comes with them.

Usage
-----
    # Ladder, cheapest first
    python scripts/generate_and_render.py --request "fire in a two-storey flat" \\
        --offline --no-build
    python scripts/generate_and_render.py --spec specs/fire_floor2.json --no-render
    python scripts/generate_and_render.py --spec specs/fire_floor2.json
    python scripts/generate_and_render.py --request "a drone searching for a
        survivor in a burning two-storey apartment"
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _path in (SRC_DIR, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import llm_compiler  # noqa: E402
import verifier  # noqa: E402
from scene_spec import (  # noqa: E402
    SceneSpec,
    SpecError,
    check_assets,
    enu_to_ned,
    load_spec,
    parse_spec,
    yaw_enu_to_ned,
)

LOG = logging.getLogger("generate_and_render")

# Bumping this string makes it obvious in the logs which version ran.
VERSION = "0.3"

# Fallback camera mapping for the CityStreet scene, used when no scene profile
# is supplied. A profile always wins, because camera names are scene-specific.
CAMERAS: Dict[str, str] = {"0": "front", "1": "left", "2": "right", "3": "down"}


# --------------------------------------------------------------------------- #
# Stage 1: compile
# --------------------------------------------------------------------------- #
def stage_compile(args, out_dir: Path) -> Optional[SceneSpec]:
    """Produce a specification from a file, the template, or the model."""
    if args.spec:
        LOG.info("[stage 1] loading specification: %s", args.spec)
        try:
            return load_spec(args.spec)
        except (SpecError, FileNotFoundError) as exc:
            LOG.error("cannot load the specification: %s", exc)
            return None

    if args.offline:
        LOG.info("[stage 1] offline template (no model call)")
        data = llm_compiler.template_spec(args.request)
        try:
            return parse_spec(data)
        except SpecError as exc:
            LOG.error("the template produced an invalid spec: %s", exc)
            return None

    LOG.info("[stage 1] compiling with the model")
    from llm_env import resolve_config

    config = resolve_config()
    if not config.configured:
        LOG.error(
            "no API key. Put LLM_API_KEY in .env, or pass --offline or --spec."
        )
        return None

    LOG.info("[stage 1] endpoint %s, model %s", config.base_url, config.model)
    try:
        return llm_compiler.compile_request(
            args.request,
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            verify_geometry=not args.no_verify,
        )
    except RuntimeError as exc:
        LOG.error("compilation failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Stage 2: verify
# --------------------------------------------------------------------------- #
def stage_verify(spec: SceneSpec, args, out_dir: Path) -> bool:
    """Verify geometry and persist the spec, unless verification is disabled."""
    if args.no_verify:
        LOG.warning("[stage 2] verification skipped (--no-verify)")
        return True

    LOG.info("[stage 2] geometric verification")
    result = verifier.verify(spec)
    print(result.summary())
    print()

    (out_dir / "verification.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not result.usable:
        LOG.error("the scenario is not usable as a test; refusing to build")
        for failure in result.failures:
            LOG.error("  - %s", failure)
        return False
    return True


# --------------------------------------------------------------------------- #
# Shot list
# --------------------------------------------------------------------------- #
def structure_centroid_enu(spec: SceneSpec) -> Tuple[float, float, float]:
    """Centre of the solid part of the scene, in world ENU metres."""
    structures = [o for o in spec.objects if o.role == "structure"]
    if not structures:
        return (0.0, 0.0, 2.0)
    xs, ys, zs = [], [], []
    for obj in structures:
        x, y, z = spec.world_position_enu(obj)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)


def shot_list(spec: SceneSpec) -> List[Dict[str, Any]]:
    """Vantage points, each a position plus the point the nose should face."""
    cx, cy, cz = structure_centroid_enu(spec)
    spawn = spec.world_spawn_enu() or (cx - 30.0, cy, 8.0)

    return [
        {
            "label": "01_spawn_view",
            "pos": spawn,
            "look_at": (cx, cy, cz),
            "note": "the declared start pose, facing the building",
        },
        {
            "label": "02_approach",
            "pos": (cx - 22.0, cy - 12.0, max(cz + 10.0, 8.0)),
            "look_at": (cx, cy, cz),
            "note": "three-quarter approach, best overview",
        },
        {
            "label": "03_overhead",
            "pos": (cx, cy, cz + 26.0),
            "look_at": (cx, cy, cz),
            "note": "top-down; the down-facing camera does the work",
        },
    ]


def heading_enu_deg(from_enu: Sequence[float], to_enu: Sequence[float]) -> float:
    """Compass-style heading in the ENU frame, degrees CCW from east."""
    dx = to_enu[0] - from_enu[0]
    dy = to_enu[1] - from_enu[1]
    return math.degrees(math.atan2(dy, dx))


# --------------------------------------------------------------------------- #
# Stage 3: build
# --------------------------------------------------------------------------- #
def stage_build(spec: SceneSpec, args) -> bool:
    """Spawn every object and confirm the assets exist first."""
    import scene_builder  # lazy: this module imports airsim

    try:
        client = scene_builder.connect(args.port, args.vehicle)
    except RuntimeError as exc:
        LOG.error("[stage 3] %s", exc)
        return False

    try:
        available = list(client.simListAssets())
    except Exception as exc:  # noqa: BLE001
        LOG.error("[stage 3] cannot read the asset list: %s", exc)
        return False

    missing = check_assets(spec, available)
    if missing:
        LOG.error("[stage 3] these assets do not exist in this scene:")
        for name in missing:
            LOG.error("  - %s", name)
        return False

    LOG.info("[stage 3] building %d objects", len(spec.objects))
    scene_builder.cleanup(client, spec, args.prefix)
    report = scene_builder.spawn_all(client, spec, args.prefix)

    if report["failed"]:
        LOG.error("[stage 3] %d object(s) failed to spawn", len(report["failed"]))
        for entry in report["failed"]:
            LOG.error("  - %s (%s): %s", entry["id"], entry["asset"], entry["error"])
        return False

    LOG.info("[stage 3] all %d objects spawned", len(report["spawned"]))
    return True


# --------------------------------------------------------------------------- #
# Stage 4: render
# --------------------------------------------------------------------------- #
def stage_render(
    spec: SceneSpec, args, out_dir: Path, cameras: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Place the vehicle at each vantage point and capture every camera."""
    import airsim
    import scene_builder

    client = scene_builder.connect(args.port, args.vehicle)

    rendered: List[Dict[str, Any]] = []

    for shot in shot_list(spec):
        label = shot["label"]
        shot_dir = out_dir / "images" / label
        shot_dir.mkdir(parents=True, exist_ok=True)

        yaw_ned = math.radians(
            yaw_enu_to_ned(heading_enu_deg(shot["pos"], shot["look_at"]))
        )
        x, y, z = enu_to_ned(shot["pos"])

        pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.to_quaternion(0.0, 0.0, yaw_ned),
        )
        client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=args.vehicle)

        # Capture immediately: the vehicle is teleported, not flown, so any delay
        # would let physics move it away from the intended vantage point.
        requests = [
            airsim.ImageRequest(
                camera_name=camera,
                image_type=airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=True,
            )
            for camera in cameras
        ]
        responses = client.simGetImages(requests, vehicle_name=args.vehicle)

        entry: Dict[str, Any] = {
            "label": label,
            "note": shot["note"],
            "pose_enu": list(shot["pos"]),
            "look_at_enu": list(shot["look_at"]),
            "images": [],
        }

        for (camera, view), response in zip(cameras.items(), responses):
            if response is None or len(response.image_data_uint8) == 0:
                LOG.warning("[%s] camera %s returned no image", label, camera)
                continue
            path = shot_dir / f"{view}.png"
            airsim.write_file(str(path), response.image_data_uint8)
            entry["images"].append({"camera": camera, "view": view, "path": str(path)})

        LOG.info("[%s] %d views -> %s", label, len(entry["images"]), shot_dir)
        rendered.append(entry)

    return rendered


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Closed loop: request -> specification -> verification -> scene -> images"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--request", default="", help="natural-language test request")
    source.add_argument("--spec", default="", help="use an existing specification")

    parser.add_argument("--offline", action="store_true", help="template instead of a model")
    parser.add_argument("--no-verify", action="store_true", help="skip geometric verification")
    parser.add_argument(
        "--no-build", action="store_true",
        help="stop after verification; does not touch AirSim",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="build the scene but do not capture images",
    )
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument(
        "--scene-profile", default="",
        help="profile_*.json from scripts/smoke_test_scene.py; sets port, "
             "vehicle and camera names for this scene",
    )
    parser.add_argument("--prefix", default="gen")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    LOG.info("generate_and_render v%s", VERSION)

    if not args.request and not args.spec:
        parser.error("one of --request or --spec is required")

    # A scene profile carries the camera names, port and vehicle that this
    # particular scene actually uses. Without one we fall back to the
    # CityStreet defaults, which will produce empty images elsewhere.
    cameras: Dict[str, str] = dict(CAMERAS)
    if args.scene_profile:
        import scene_profile

        profile = scene_profile.load(args.scene_profile)
        LOG.info("scene profile loaded:\n%s", profile.describe())
        args.port = profile.port
        args.vehicle = profile.vehicle
        cameras = profile.pick_capture_cameras()
        LOG.info("capture cameras: %s", list(cameras))
    else:
        LOG.warning(
            "no --scene-profile given; using CityStreet camera names %s. "
            "On another scene this yields empty images.",
            list(cameras),
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (Path("runs") / f"render_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("output directory: %s", out_dir)

    # ---- stage 1 ----
    spec = stage_compile(args, out_dir)
    if spec is None:
        return 1

    # ---- stage 2 ----
    if not stage_verify(spec, args, out_dir):
        return 1

    (out_dir / "spec.json").write_text(
        json.dumps(llm_compiler.spec_to_dict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("specification saved to %s", out_dir / "spec.json")

    if args.no_build:
        LOG.info("--no-build given; stopping before AirSim")
        print(spec.summary())
        return 0

    # ---- stage 3 ----
    if not stage_build(spec, args):
        return 2

    if args.no_render:
        LOG.info("--no-render given; stopping before capture")
        return 0

    # ---- stage 4 ----
    LOG.info("[stage 4] rendering")
    try:
        rendered = stage_render(spec, args, out_dir, cameras)
    except Exception as exc:  # noqa: BLE001
        LOG.error("[stage 4] rendering failed: %s", exc)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.exception("traceback:")
        return 3

    report = {
        "version": VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "request": args.request,
        "spec_name": spec.name,
        "objects": len(spec.objects),
        "shots": rendered,
    }
    (out_dir / "render_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = sum(len(shot["images"]) for shot in rendered)
    print()
    LOG.info("done: %d images under %s", total, out_dir / "images")
    for shot in rendered:
        LOG.info("  %-14s %s", shot["label"], shot["note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())