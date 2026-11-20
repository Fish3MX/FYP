#!/usr/bin/env python3
"""Instantiate a scenario specification inside AirSim.

Scope
-----
This module does exactly one thing: read a validated SceneSpec and place the
objects it describes into the running simulator. It does not call a language
model, does not generate scenarios, and does not verify geometry; those live
elsewhere.

That boundary matters. It means the whole chain can be debugged with a
hand-written JSON file, before any model is involved. Uncertainties get removed
one at a time instead of all at once.

Pipeline
--------
1. Connect to AirSim and read the list of assets the scene can spawn.
2. Pre-check: every asset named in the spec must exist, or fail before building.
3. Clean up objects left by a previous build of the same scenario.
4. Spawn each object with simSpawnObject.
5. Read each object's pose back to confirm it really exists.
6. Write a structured build report.

Usage
-----
    # Dry run: check the plan and the assets, change nothing
    python src/scene_builder.py --spec specs/fire_floor2.json --dry-run

    # Build it
    python src/scene_builder.py --spec specs/fire_floor2.json

    # Build and drop the drone at the specified start pose
    python src/scene_builder.py --spec specs/fire_floor2.json --move-drone

    # Remove a previous build without rebuilding
    python src/scene_builder.py --spec specs/fire_floor2.json --cleanup-only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import airsim
except ImportError:
    sys.exit(
        "cannot import airsim. Activate the environment that has the client, e.g.\n"
        "    source /home/SIGPhysics/venvs/airsim/bin/activate"
    )

from scene_spec import (
    SceneObject,
    SceneSpec,
    SpecError,
    check_assets,
    enu_to_ned,
    load_spec,
    yaw_enu_to_ned,
)

LOG = logging.getLogger("scene_builder")

# Prefix for every spawned object. Cleanup removes only names with this prefix,
# so the thousands of objects already in the base scene are never touched.
DEFAULT_PREFIX = "gen"


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect(port: int, vehicle: str) -> "airsim.MultirotorClient":
    """Connect and confirm the requested vehicle exists."""
    kwargs: Dict[str, Any] = {"port": port}
    try:
        client = airsim.MultirotorClient(**kwargs)
        client.confirmConnection()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cannot connect to AirSim on port {port}. Check:\n"
            f"  1) the UE scene is running\n"
            f"  2) ss -lntp | grep {port}\n"
            f"underlying error: {exc}"
        ) from exc

    vehicles = client.listVehicles()
    if vehicle not in vehicles:
        raise RuntimeError(f"the scene has no vehicle '{vehicle}'. Available: {vehicles}")
    LOG.info("connected; vehicles: %s", vehicles)
    return client


# --------------------------------------------------------------------------- #
# Naming and poses
# --------------------------------------------------------------------------- #
def object_name(spec: SceneSpec, obj_id: str, prefix: str) -> str:
    """Build a unique object name: <prefix>_<scenario>_<object id>."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj_id)
    return f"{prefix}_{spec.name}_{safe}"


def build_pose(spec: SceneSpec, obj: SceneObject) -> "airsim.Pose":
    """Convert a local ENU pose from the spec into a world NED AirSim pose."""
    world_enu = spec.world_position_enu(obj)
    x, y, z = enu_to_ned(world_enu)
    yaw_ned = yaw_enu_to_ned(obj.yaw_deg)
    return airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, math.radians(yaw_ned)),
    )


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def cleanup(client, spec: SceneSpec, prefix: str, dry_run: bool = False) -> int:
    """Delete objects from a previous build; returns how many were removed."""
    try:
        names = client.simListSceneObjects()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("cannot list scene objects, skipping cleanup: %s", exc)
        return 0

    my_prefix = f"{prefix}_{spec.name}_"
    targets = [name for name in names if name.startswith(my_prefix)]

    if not targets:
        LOG.info("nothing to clean up (prefix %s)", my_prefix)
        return 0

    LOG.info("found %d objects from a previous build", len(targets))
    removed = 0
    for name in targets:
        if dry_run:
            removed += 1
            continue
        try:
            client.simDestroyObject(name)
            removed += 1
        except Exception as exc:  # noqa: BLE001
            LOG.warning("failed to delete %s: %s", name, exc)

    LOG.info("%s %d objects", "would remove" if dry_run else "removed", removed)
    return removed


# --------------------------------------------------------------------------- #
# Spawning
# --------------------------------------------------------------------------- #
def spawn_all(
    client, spec: SceneSpec, prefix: str, dry_run: bool = False
) -> Dict[str, Any]:
    """Spawn every object in the spec and return a structured report."""
    spawned: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for obj in spec.objects:
        name = object_name(spec, obj.id, prefix)
        pose = build_pose(spec, obj)
        world_enu = spec.world_position_enu(obj)
        ned = enu_to_ned(world_enu)

        entry = {
            "id": obj.id,
            "name": name,
            "asset": obj.asset,
            "role": obj.role,
            "enu_world": list(world_enu),
            "ned_world": list(ned),
            "yaw_enu_deg": obj.yaw_deg,
        }

        if dry_run:
            LOG.info(
                "[dry-run] would spawn %-22s %-32s NED=(%.1f, %.1f, %.1f)",
                obj.id, obj.asset, *ned,
            )
            spawned.append(entry)
            continue

        try:
            ok = client.simSpawnObject(
                object_name=name,
                asset_name=obj.asset,
                pose=pose,
                scale=airsim.Vector3r(*obj.scale),
                physics_enabled=obj.physics,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error("spawning %s (%s) raised: %s", obj.id, obj.asset, exc)
            entry["error"] = str(exc)
            failed.append(entry)
            continue

        if not ok:
            LOG.error(
                "spawning %s (%s) returned False: the asset cannot be created "
                "at runtime",
                obj.id, obj.asset,
            )
            entry["error"] = "simSpawnObject returned False"
            failed.append(entry)
            continue

        # Read the pose back to confirm the object really exists in the world.
        try:
            actual = client.simGetObjectPose(name)
            entry["actual_ned"] = [
                actual.position.x_val,
                actual.position.y_val,
                actual.position.z_val,
            ]
            drift = math.dist(entry["actual_ned"], list(ned))
            entry["drift_m"] = round(drift, 3)
            if drift > 1.0:
                LOG.warning("  %s ended up %.2f m from the requested pose", obj.id, drift)
        except Exception as exc:  # noqa: BLE001
            entry["pose_readback_error"] = str(exc)

        LOG.info("ok %-22s %-32s %.1f,%.1f,%.1f", obj.id, obj.asset, *ned)
        spawned.append(entry)

    return {"spawned": spawned, "failed": failed}


# --------------------------------------------------------------------------- #
# Drone placement
# --------------------------------------------------------------------------- #
def move_drone_to_spawn(client, spec: SceneSpec, vehicle: str) -> None:
    """Teleport the drone to the start pose declared by the spec."""
    if spec.spawn is None:
        LOG.info("the spec has no spawn field, skipping")
        return

    world_enu = spec.world_point_enu(spec.spawn.pos)
    x, y, z = enu_to_ned(world_enu)
    yaw = math.radians(yaw_enu_to_ned(spec.spawn.yaw_deg))

    pose = airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, yaw),
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)
    LOG.info("drone moved to spawn NED=(%.1f, %.1f, %.1f)", x, y, z)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a scenario from a spec in AirSim")
    parser.add_argument("--spec", required=True, help="path to the specification JSON")
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="spawned object name prefix")
    parser.add_argument("--dry-run", action="store_true", help="plan only, change nothing")
    parser.add_argument("--cleanup-only", action="store_true", help="remove and stop")
    parser.add_argument("--no-cleanup", action="store_true", help="skip cleanup before building")
    parser.add_argument("--move-drone", action="store_true", help="place the drone at spawn")
    parser.add_argument("--report", default="", help="build report output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- 1. load the spec ----
    try:
        spec = load_spec(args.spec)
    except (SpecError, FileNotFoundError) as exc:
        LOG.error("cannot load the specification:\n%s", exc)
        return 2

    LOG.info("specification loaded")
    print(spec.summary())

    # ---- 2. connect ----
    try:
        client = connect(args.port, args.vehicle)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 3

    # ---- 3. asset pre-check ----
    try:
        available = list(client.simListAssets())
    except Exception as exc:  # noqa: BLE001
        LOG.error("cannot read the available asset list: %s", exc)
        return 3

    missing = check_assets(spec, available)
    if missing:
        LOG.error(
            "these assets are not available in the current scene (%d):", len(missing)
        )
        for name in missing:
            LOG.error("  - %s", name)
        LOG.error("use names present in simListAssets(), or check the spelling.")
        return 4
    LOG.info(
        "asset pre-check passed: all %d objects use available assets", len(spec.objects)
    )

    # ---- 4. cleanup ----
    if not args.no_cleanup:
        cleanup(client, spec, args.prefix, dry_run=args.dry_run)

    if args.cleanup_only:
        LOG.info("--cleanup-only given, stopping")
        return 0

    # ---- 5. spawn ----
    result = spawn_all(client, spec, args.prefix, dry_run=args.dry_run)

    # ---- 6. move the drone ----
    if args.move_drone and not args.dry_run:
        move_drone_to_spawn(client, spec, args.vehicle)

    # ---- 7. report ----
    report = {
        "built_at": datetime.now().astimezone().isoformat(),
        "spec": {"name": spec.name, "version": spec.version, "path": str(args.spec)},
        "dry_run": args.dry_run,
        "totals": {
            "planned": len(spec.objects),
            "spawned": len(result["spawned"]),
            "failed": len(result["failed"]),
        },
        **result,
    }

    report_path = args.report or f"build_report_{spec.name}.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("report written to %s", report_path)

    if result["failed"]:
        LOG.error("%d object(s) failed to spawn, see the report", len(result["failed"]))
        return 5

    LOG.info("all %d objects spawned successfully", len(result["spawned"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())