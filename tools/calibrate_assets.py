#!/usr/bin/env python3
"""Calibrate the real-world size of the primitive assets.

Why this is needed
------------------
Every dimension in a specification is in metres, but a primitive's final size
depends on the mesh's own units inside the packaged UE project. The code assumes
`Cube` is 1 m at scale 1. If that assumption is wrong -- by a factor of 100, or
by 0.01 -- every generated wall is wrong by the same factor, and the scene looks
like sheets of card floating over the street.

This script removes the guesswork. It places three primitives with known scale
values at a known distance from the camera, prints the pixel width each should
occupy if the 1 m assumption holds, and captures an image. Comparing the real
picture with the prediction settles the question in one run.

Geometry of the test
--------------------
    camera : ENU (0, 0, 5), yaw north, front camera, 90 deg FOV, 512 px
    targets: 12 m north of the camera, all at z = 5 m

      #1  east -4 m, scale [1, 1, 1]   -> expected ~27 px wide if Cube is 1 m
      #2  east  0 m, scale [2, 1, 1]   -> expected ~54 px wide
      #3  east  4 m, scale [1, 2, 1]   -> expected ~27 px, twice as tall

If the image shows three small blocks of roughly those sizes, the 1 m assumption
is right and the geometry problem lies elsewhere. If one block fills the frame,
the mesh is 100x too large for our use.

Usage
-----
    python scripts/calibrate_assets.py
    python scripts/calibrate_assets.py --out runs/calib --keep   # keep the probes
"""
from __future__ import annotations

import argparse
import math
import sys
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

import airsim  # noqa: E402

# Test setup, in world ENU metres.
CAMERA_POS_ENU = (0.0, 0.0, 5.0)
CAMERA_YAW_ENU_DEG = 90.0        # facing north
DISTANCE_M = 12.0
FOV_DEG = 90.0
RESOLUTION_PX = 512

PROBES: List[Dict[str, Any]] = [
    {"name": "calib_cube_1m", "scale": (1.0, 1.0, 1.0), "offset_east": -4.0},
    {"name": "calib_cube_2m_wide", "scale": (2.0, 1.0, 1.0), "offset_east": 0.0},
    {"name": "calib_cube_2m_tall", "scale": (1.0, 2.0, 1.0), "offset_east": 4.0},
]

PREFIX = "calib_"


def enu_to_ned(enu: Sequence[float]) -> Tuple[float, float, float]:
    """Local copy so the calibration runs without importing scene_spec."""
    return (float(enu[1]), float(enu[0]), -float(enu[2]))


def yaw_enu_to_ned(yaw_deg: float) -> float:
    return (90.0 - float(yaw_deg)) % 360.0


def expected_pixels(size_m: float) -> float:
    """Apparent width in pixels if the object really is `size_m` metres wide."""
    half_angle = math.atan((size_m / 2.0) / DISTANCE_M)
    full_angle_deg = math.degrees(2.0 * half_angle)
    return RESOLUTION_PX * (full_angle_deg / FOV_DEG)


def connect(port: int, vehicle: str) -> "airsim.MultirotorClient":
    client = airsim.MultirotorClient(port=port)
    client.confirmConnection()
    vehicles = client.listVehicles()
    if vehicle not in vehicles:
        raise RuntimeError(f"no vehicle '{vehicle}'; available: {vehicles}")
    print(f"connected, vehicles: {vehicles}")
    return client


def clear_probes(client: "airsim.MultirotorClient") -> int:
    """Remove probe objects left by a previous run."""
    try:
        names = client.simListSceneObjects()
    except Exception as exc:  # noqa: BLE001
        print(f"cannot list scene objects: {exc}")
        return 0

    removed = 0
    for name in names:
        if name.startswith(PREFIX):
            try:
                client.simDestroyObject(name)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    if removed:
        print(f"removed {removed} leftover probes")
    return removed


def spawn_probes(client: "airsim.MultirotorClient") -> None:
    """Spawn the three test cubes at their known positions."""
    print()
    print("spawning probes (no physics, so they stay where placed)")
    print(f"  camera at ENU {CAMERA_POS_ENU}, yaw {CAMERA_YAW_ENU_DEG} deg (north)")
    print(f"  probes {DISTANCE_M:.0f} m ahead, all at z = {CAMERA_POS_ENU[2]:.0f} m")
    print()
    print(f"  {'name':<22} {'scale':<16} {'ENU position':<26} expected px")
    print("  " + "-" * 78)

    for probe in PROBES:
        enu = (
            probe["offset_east"],
            DISTANCE_M,
            CAMERA_POS_ENU[2],
        )
        x, y, z = enu_to_ned(enu)
        pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        ok = client.simSpawnObject(
            object_name=probe["name"],
            asset_name="Cube",
            pose=pose,
            scale=airsim.Vector3r(*probe["scale"]),
            physics_enabled=False,
        )
        px = expected_pixels(probe["scale"][0] * 1.0)
        status = "ok " if ok else "FAILED"
        print(
            f"  {probe['name']:<22} {str(probe['scale']):<16} "
            f"{str(enu):<26} {px:5.1f}  [{status}]"
        )


def place_camera(client: "airsim.MultirotorClient", vehicle: str) -> None:
    """Put the vehicle at the calibration viewpoint and let it settle."""
    x, y, z = enu_to_ned(CAMERA_POS_ENU)
    yaw = math.radians(yaw_enu_to_ned(CAMERA_YAW_ENU_DEG))
    pose = airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, yaw),
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)
    print(f"\ncamera placed at NED ({x:.1f}, {y:.1f}, {z:.1f}), yaw {math.degrees(yaw):.0f} deg")


def capture(client: "airsim.MultirotorClient", vehicle: str, out_dir: Path) -> None:
    """Capture RGB and depth from the front camera."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)],
        vehicle_name=vehicle,
    )[0]
    if rgb is not None and len(rgb.image_data_uint8) > 0:
        path = out_dir / "front_rgb.png"
        airsim.write_file(str(path), rgb.image_data_uint8)
        print(f"rgb   saved: {path}  ({rgb.width}x{rgb.height})")
    else:
        print("rgb   capture returned nothing")

    depth = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)],
        vehicle_name=vehicle,
    )[0]
    if depth is not None and len(depth.image_data_float) > 0:
        try:
            import numpy as np

            array = np.asarray(depth.image_data_float, dtype=np.float32).reshape(
                depth.height, depth.width
            )
            path = out_dir / "front_depth.npy"
            np.save(path, array)
            centre = array[array.shape[0] // 2, array.shape[1] // 2]
            print(f"depth saved: {path}  (centre pixel: {float(centre):.2f} m)")
        except Exception as exc:  # noqa: BLE001
            print(f"depth saved failed: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the real size of the Cube primitive"
    )
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--vehicle", default="Drone_1")
    parser.add_argument("--out", type=Path, default=Path("runs/calib"))
    parser.add_argument("--keep", action="store_true", help="leave the probes in the scene")
    args = parser.parse_args(argv)

    try:
        client = connect(args.port, args.vehicle)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot connect: {exc}")
        return 2

    clear_probes(client)
    spawn_probes(client)
    place_camera(client, args.vehicle)
    capture(client, args.vehicle, args.out)

    if not args.keep:
        clear_probes(client)
        print("probes removed")

    print()
    print("=" * 78)
    print("how to read the result")
    print("=" * 78)
    print("  Compare the blocks in front_rgb.png with the 'expected px' column.")
    print("  - three small blocks, roughly those widths -> Cube really is 1 m,")
    print("    and the geometry problem is somewhere else (materials or assets).")
    print("  - one block fills the frame                    -> Cube is far larger")
    print("    than 1 m; every dimension in every spec is scaled by that factor.")
    print("  - no blocks at all                             -> the Cube asset did")
    print("    not spawn, or is not visible from this angle.")
    print()
    print(f"  send {args.out / 'front_rgb.png'} back for analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())