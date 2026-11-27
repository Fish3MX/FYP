#!/usr/bin/env python3
"""Measure whether whole-building blueprints can be spawned at runtime.

For each asset name: capture a depth frame as background, spawn the asset at a
known pose, capture depth again, then diff the two frames. The changed pixels are
exactly the object, so its pixel extent and distance give its real size without
any assumptions about the mesh's units.

Usage
-----
    python scripts/probe_prefabs.py --port 30001
    python scripts/probe_prefabs.py --port 30001 --assets BP_MBuilding01 SM_MBuilding01_Wall
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Imported lazily: --help must work on a machine without the AirSim client.
try:
    import airsim
except ImportError:  # pragma: no cover
    airsim = None  # type: ignore[assignment]

# Camera sits this far south of the object, at this height, facing north.
DISTANCE_M = 40.0
CAMERA_Z = 10.0
FOV_DEG = 90.0

DEFAULT_ASSETS = [f"BP_MBuilding0{i}" for i in range(1, 6)]


def enu_to_ned(enu: Sequence[float]) -> "airsim.Vector3r":
    return airsim.Vector3r(float(enu[1]), float(enu[0]), -float(enu[2]))


def set_view(client, vehicle: str, distance: float, height: float) -> None:
    """Place the drone at the measuring vantage point, facing north."""
    pose = airsim.Pose(
        enu_to_ned([0.0, -distance, height]),
        airsim.to_quaternion(0.0, 0.0, 0.0),   # yaw 0 in NED = north
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)


def grab_depth(client, vehicle: str, camera: str) -> np.ndarray:
    """One planar depth frame, in metres."""
    response = client.simGetImages(
        [airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False)],
        vehicle_name=vehicle,
    )[0]
    return np.asarray(response.image_data_float, dtype=np.float32).reshape(
        response.height, response.width
    )


def grab_rgb(client, vehicle: str, camera: str, path: Path) -> bool:
    response = client.simGetImages(
        [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True)],
        vehicle_name=vehicle,
    )[0]
    if response is None or len(response.image_data_uint8) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    airsim.write_file(str(path), response.image_data_uint8)
    return True


def measure(background: np.ndarray, current: np.ndarray, fov: float = FOV_DEG) -> Optional[Dict[str, float]]:
    """Object extent from the difference between two depth frames.

    A pixel belongs to the object when its depth changed by more than a metre.
    Size follows from the pinhole relation: size_m = distance * pixels / focal,
    with focal = (width / 2) / tan(fov / 2).
    """
    valid = np.isfinite(background) & np.isfinite(current)
    changed = np.abs(current - background) > 1.0
    mask = valid & changed
    if mask.sum() < 20:
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pixel_w = int(cols[-1] - cols[0] + 1)
    pixel_h = int(rows[-1] - rows[0] + 1)

    depths = current[mask]
    depths = depths[np.isfinite(depths) & (depths > 0)]
    if depths.size == 0:
        return None
    median_depth = float(np.median(depths))

    focal = (current.shape[1] / 2.0) / math.tan(math.radians(fov) / 2.0)
    return {
        "pixels_w": pixel_w,
        "pixels_h": pixel_h,
        "depth_m": round(median_depth, 2),
        "width_m": round(pixel_w * median_depth / focal, 2),
        "height_m": round(pixel_h * median_depth / focal, 2),
        "pixels": int(mask.sum()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether building blueprints can be spawned and how big they are",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument("--camera", default="FrontCamera")
    parser.add_argument("--assets", nargs="*", default=DEFAULT_ASSETS)
    parser.add_argument("--out", type=Path, default=Path("runs/prefabs"))
    parser.add_argument("--distance", type=float, default=DISTANCE_M,
                        help="camera distance south of the object, metres")
    parser.add_argument("--height", type=float, default=CAMERA_Z,
                        help="camera height, metres")
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="camera horizontal FOV")
    parser.add_argument("--keep", action="store_true", help="leave the last probe in place")
    args = parser.parse_args(argv)

    if airsim is None:
        print("the airsim Python client is not installed in this environment.")
        return 2

    client = airsim.MultirotorClient(port=args.port)
    client.confirmConnection()
    print(f"connected; vehicles={client.listVehicles()}")
    print(f"vantage: {args.distance:.0f} m south of origin, {args.height:.0f} m up, facing north\n")

    set_view(client, args.vehicle, args.distance, args.height)
    background = grab_depth(client, args.vehicle, args.camera)

    print(f"  {'asset':<26} {'spawn':<7} {'drift':<7} {'px':>9} {'depth':>8} "
          f"{'width':>7} {'height':>7}")
    print("  " + "-" * 82)

    results: List[Dict[str, object]] = []

    for index, asset in enumerate(args.assets):
        name = f"probe_building_{index}"
        pose = airsim.Pose(enu_to_ned([0.0, 0.0, 0.0]), airsim.to_quaternion(0, 0, 0))

        try:
            ok = client.simSpawnObject(
                object_name=name, asset_name=asset, pose=pose,
                scale=airsim.Vector3r(1, 1, 1), physics_enabled=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {asset:<26} ERROR   {exc}")
            results.append({"asset": asset, "spawned": False, "error": str(exc)})
            continue

        if not ok:
            print(f"  {asset:<26} False   -- not spawnable at runtime")
            results.append({"asset": asset, "spawned": False, "error": "returned False"})
            continue

        try:
            actual = client.simGetObjectPose(name)
            drift = math.dist(
                [actual.position.x_val, actual.position.y_val, actual.position.z_val],
                [0.0, 0.0, 0.0],
            )
        except Exception:  # noqa: BLE001
            drift = float("nan")

        grab_rgb(client, args.vehicle, args.camera, args.out / f"{asset}.png")
        current = grab_depth(client, args.vehicle, args.camera)
        stats = measure(background, current, args.fov)

        if stats is None:
            print(f"  {asset:<26} True    {drift:<7.2f} {'-- no pixels changed (invisible or tiny)'}")
            results.append({"asset": asset, "spawned": True, "drift": drift, "measured": None})
        else:
            print(
                f"  {asset:<26} True    {drift:<7.2f} "
                f"{stats['pixels']:>9} {stats['depth_m']:>8} "
                f"{stats['width_m']:>7} {stats['height_m']:>7}"
            )
            results.append({"asset": asset, "spawned": True, "drift": drift, "measured": stats})

        client.simDestroyObject(name)

    if args.keep and args.assets:
        # Re-spawn the last one so it can be inspected by hand.
        name = "probe_building_kept"
        client.simSpawnObject(name, args.assets[-1], airsim.Pose(
            enu_to_ned([0.0, 0.0, 0.0]), airsim.to_quaternion(0, 0, 0)),
            airsim.Vector3r(1, 1, 1), physics_enabled=True)
        print(f"\nkept '{name}' ({args.assets[-1]}) in the scene")

    print()
    usable = [r for r in results if r.get("spawned")]
    sized = [r for r in usable if r.get("measured")]
    print(f"spawnable: {len(usable)}/{len(results)}   measurable: {len(sized)}/{len(results)}")
    print(f"images: {args.out}")

    import json
    (args.out / "probe_report.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "probe_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if sized else 1


if __name__ == "__main__":
    sys.exit(main())