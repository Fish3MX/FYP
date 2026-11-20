#!/usr/bin/env python3
"""Probe whether the current UE scene supports runtime object placement.

Why probe first
---------------
The core action of this project is "read a JSON spec and place objects in the
scene". That depends on AirSim's simSpawnObject API, which can only instantiate
assets that already exist inside the packaged UE project.

Packaged projects vary enormously in what they expose:

  - some expose hundreds of spawnable assets -> runtime placement is viable
  - some expose one or three placeholders     -> offline editing and repackaging
                                                 is the only route

This script answers which case applies, in about thirty seconds, and therefore
decides the technical approach for everything that follows.

Usage
-----
    python scripts/probe_assets.py
    python scripts/probe_assets.py --port 30100 --vehicle Drone_1
    python scripts/probe_assets.py --spawn        # also try spawning one object
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence

try:
    import airsim
except ImportError:
    sys.exit(
        "cannot import airsim. Activate the environment that has the client, e.g.\n"
        "    source /home/SIGPhysics/venvs/airsim/bin/activate"
    )

LOG = logging.getLogger("probe_assets")


def connect(port: int, vehicle: str) -> "airsim.MultirotorClient":
    client = airsim.MultirotorClient(port=port)
    client.confirmConnection()
    LOG.info("connected to port %d; vehicles: %s", port, client.listVehicles())
    return client


def list_assets(client: "airsim.MultirotorClient") -> List[str]:
    """List every asset name that can be spawned at runtime."""
    try:
        assets = client.simListAssets()
    except Exception as exc:  # noqa: BLE001
        LOG.error(
            "simListAssets failed: %s\n"
            "This usually means the AirSim build does not implement the API, "
            "or the scene forbids asset queries.",
            exc,
        )
        return []
    return list(assets) if assets else []


def existing_objects(client: "airsim.MultirotorClient") -> int:
    """Count objects already present, to compare before and after a spawn."""
    try:
        objects = client.simListSceneObjects()
        return len(objects) if objects else 0
    except Exception as exc:  # noqa: BLE001
        LOG.warning("simListSceneObjects unavailable: %s", exc)
        return -1


def try_spawn(
    client: "airsim.MultirotorClient", asset: str, vehicle: str
) -> Optional[str]:
    """Spawn one instance of the asset next to the drone.

    Returns the object name on success and None on failure.
    """
    name = "probe_object_1"
    # 5 m ahead, 3 m right, 1 m below: offset from the airframe.
    pose = airsim.Pose(
        airsim.Vector3r(5.0, 3.0, -1.0),
        airsim.to_quaternion(0, 0, 0),
    )
    try:
        ok = client.simSpawnObject(
            object_name=name,
            asset_name=asset,
            pose=pose,
            scale=airsim.Vector3r(1, 1, 1),
            physics_enabled=False,   # no physics yet, we only test spawnability
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("simSpawnObject raised: %s", exc)
        return None

    if not ok:
        LOG.error("simSpawnObject returned False: this asset cannot be spawned")
        return None

    try:
        actual = client.simGetObjectPose(name)
        LOG.info("spawn succeeded, pose read back: %s", actual.position)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("spawned but the pose could not be read back: %s", exc)

    return name


def cleanup(client: "airsim.MultirotorClient", name: str) -> None:
    try:
        client.simDestroyObject(name)
        LOG.info("probe object %s destroyed", name)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("destroy failed (harmless): %s", exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe the runtime asset capability of a UE scene"
    )
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument("--spawn", action="store_true", help="actually spawn one object")
    parser.add_argument("--json", type=str, default="", help="write the report here")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report: Dict[str, Any] = {"port": args.port, "vehicle": args.vehicle}
    client = connect(args.port, args.vehicle)

    # 1. objects already in the scene
    before = existing_objects(client)
    report["scene_object_count_before"] = before
    LOG.info("objects already in the scene: %s", before)

    # 2. spawnable assets
    assets = list_assets(client)
    report["available_assets"] = assets
    report["available_asset_count"] = len(assets)

    if assets:
        LOG.info("%d assets available:", len(assets))
        for index, name in enumerate(assets[:50]):
            print(f"   [{index:3d}] {name}")
        if len(assets) > 50:
            print(f"   ... {len(assets) - 50} more not shown")
    else:
        LOG.warning(
            "no assets reported; the scene probably does not support runtime placement"
        )

    # 3. optional: spawn one for real
    if args.spawn and assets:
        target = assets[0]
        LOG.info("attempting to spawn '%s' ...", target)
        spawned = try_spawn(client, target, args.vehicle)
        report["spawn_attempted"] = target
        report["spawn_succeeded"] = spawned is not None
        if spawned:
            after = existing_objects(client)
            report["scene_object_count_after"] = after
            LOG.info("scene object count %s -> %s", before, after)
            cleanup(client, spawned)

    # 4. verdict
    if len(assets) >= 50:
        verdict = "runtime_placement_viable"
        message = (
            "plenty of assets: runtime placement is viable "
            "(fast, no repackaging needed)"
        )
    elif assets:
        verdict = "runtime_placement_limited"
        message = (
            f"only {len(assets)} assets: runtime placement works but is "
            f"expressively limited; offline assets may be needed"
        )
    else:
        verdict = "runtime_placement_unavailable"
        message = (
            "runtime placement is unavailable: offline editing and repackaging "
            "is the only route"
        )
    report["verdict"] = verdict
    LOG.info("verdict: %s -- %s", verdict, message)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        LOG.info("report written to %s", args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())