#!/usr/bin/env python3
"""Minimal control-link check for the AirSim + UE4 stack.

Purpose
-------
Prove that Python -> AirSim -> UE is wired end to end, by running the smallest
complete loop: connect, unlock, take off, climb, fly forward, capture four views
with depth, land.

This depends on nothing but the official AirSim Python client, so when something
breaks later it can be used to decide whether the fault is in the environment or
in our own code.

Outputs
-------
    <out>/cam<name>_rgb.png      one RGB image per camera (512x512 by default)
    <out>/cam<name>_depth.npy    planar depth in metres, float32
    <out>/metadata.json          pose, timestamps, camera setup, conventions

Usage
-----
    python scripts/hello_uav.py
    python scripts/hello_uav.py --port 30100 --vehicle Drone_1
    python scripts/hello_uav.py --altitude 8 --forward-time 8 --out runs/demo1
    python scripts/hello_uav.py --connect-only
    python scripts/hello_uav.py -v

Coordinate note
---------------
AirSim uses NED: x forward, y right, z **down**, so "climb 5 m" is z = -5.
This script exposes --altitude as "up is positive" and converts once, rather
than forcing every call site to remember the minus sign.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import airsim
except ImportError:  # pragma: no cover - environment problem, give the fix
    sys.exit(
        "cannot import airsim. Activate the environment that has the client, e.g.\n"
        "    source /home/SIGPhysics/venvs/airsim/bin/activate"
    )

LOG = logging.getLogger("hello_uav")

# Image type constants, named so the code reads without a lookup.
IMG_RGB = airsim.ImageType.Scene          # 0
IMG_DEPTH = airsim.ImageType.DepthPlanar  # 2


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FlightConfig:
    """Everything one verification flight needs."""

    host: str = ""                       # empty means loopback
    port: int = 30100                    # from settings.json ApiServerPort
    vehicle: str = "Drone_1"             # from settings.json Vehicles keys
    altitude: float = 5.0                # metres, up is positive
    forward_speed: float = 3.0           # m/s
    forward_time: float = 4.0            # simulated seconds (see ClockSpeed)
    out_dir: Path = Path("shots")
    want_depth: bool = True
    connect_only: bool = False
    takeoff_timeout: float = 20.0

    # camera_name -> human label; names come from settings.json Cameras keys
    cameras: Dict[str, str] = field(
        default_factory=lambda: {"0": "front", "1": "left", "2": "right", "3": "down"}
    )

    @property
    def ned_altitude(self) -> float:
        """Altitude as an AirSim NED z value (negative is up)."""
        return -abs(self.altitude)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("airsim").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )


def vec3(value: Any) -> List[float]:
    """Convert an AirSim Vector3r into a plain list for JSON."""
    return [float(value.x_val), float(value.y_val), float(value.z_val)]


def quat(value: Any) -> List[float]:
    return [float(value.w_val), float(value.x_val), float(value.y_val), float(value.z_val)]


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def connect(config: FlightConfig) -> "airsim.MultirotorClient":
    """Open the connection and confirm the vehicle exists."""
    LOG.info("connecting to AirSim %s:%d ...", config.host or "127.0.0.1", config.port)
    # Never pass an empty string as `ip`: airsim would build an invalid RPC address.
    kwargs: Dict[str, Any] = {"port": config.port}
    if config.host:
        kwargs["ip"] = config.host
    try:
        client = airsim.MultirotorClient(**kwargs)
        client.confirmConnection()
    except Exception as exc:
        raise RuntimeError(
            f"cannot connect to AirSim ({config.host or '127.0.0.1'}:{config.port}). "
            f"Check, in order:\n"
            f"  1) the UE scene is actually running (it prints the vehicle name)\n"
            f"  2) the port is listening: ss -lntp | grep {config.port}\n"
            f"  3) settings.json ApiServerPort equals {config.port}\n"
            f"underlying error: {exc}"
        ) from exc

    vehicles = client.listVehicles()
    LOG.info("connected. vehicles: %s", vehicles)
    if config.vehicle not in vehicles:
        raise RuntimeError(
            f"the scene has no vehicle named '{config.vehicle}'; "
            f"available: {vehicles}. Pass one with --vehicle."
        )
    return client


def print_state(client: "airsim.MultirotorClient", config: FlightConfig, note: str) -> None:
    """Log the position so the flight can be followed by eye."""
    state = client.getMultirotorState(vehicle_name=config.vehicle)
    position = state.kinematics_estimated.position
    LOG.info(
        "%s -> position (NED) x=%.2f y=%.2f z=%.2f",
        note, position.x_val, position.y_val, position.z_val,
    )


def take_off(client: "airsim.MultirotorClient", config: FlightConfig) -> None:
    """Unlock, take off, and climb to the target altitude."""
    client.enableApiControl(True, vehicle_name=config.vehicle)
    client.armDisarm(True, vehicle_name=config.vehicle)
    LOG.info("API control enabled")

    client.takeoffAsync(
        timeout_sec=config.takeoff_timeout, vehicle_name=config.vehicle
    ).join()
    LOG.info("airborne")

    # The second argument of moveToZAsync is the maximum speed in m/s.
    client.moveToZAsync(config.ned_altitude, 2.0, vehicle_name=config.vehicle).join()
    LOG.info("climbed to %.1f m", config.altitude)


def fly_forward(client: "airsim.MultirotorClient", config: FlightConfig) -> None:
    """Hold altitude and fly forward at constant speed for a while."""
    client.moveByVelocityZAsync(
        vx=config.forward_speed,
        vy=0.0,
        z=config.ned_altitude,
        duration=config.forward_time,
        vehicle_name=config.vehicle,
    ).join()
    client.hoverAsync(vehicle_name=config.vehicle).join()

    distance = config.forward_speed * config.forward_time
    LOG.info("flew forward about %.0f m and hovered", distance)


def capture_views(
    client: "airsim.MultirotorClient", config: FlightConfig
) -> Dict[str, Any]:
    """Capture RGB and depth from every camera and write the metadata."""
    config.out_dir.mkdir(parents=True, exist_ok=True)

    camera_items = list(config.cameras.items())
    rgb_requests = [
        airsim.ImageRequest(
            camera_name=camera, image_type=IMG_RGB, pixels_as_float=False, compress=True
        )
        for camera, _ in camera_items
    ]
    rgb_responses = client.simGetImages(rgb_requests, vehicle_name=config.vehicle)

    depth_responses: Sequence[Any] = []
    if config.want_depth:
        depth_requests = [
            airsim.ImageRequest(
                camera_name=camera,
                image_type=IMG_DEPTH,
                pixels_as_float=True,
                compress=False,
            )
            for camera, _ in camera_items
        ]
        depth_responses = client.simGetImages(
            depth_requests, vehicle_name=config.vehicle
        )

    frames: List[Dict[str, Any]] = []

    for index, ((camera, label), rgb) in enumerate(zip(camera_items, rgb_responses)):
        frame: Dict[str, Any] = {"camera": camera, "label": label}

        if rgb is not None and len(rgb.image_data_uint8) > 0:
            rgb_path = config.out_dir / f"cam{label}_rgb.png"
            airsim.write_file(str(rgb_path), rgb.image_data_uint8)
            frame["rgb"] = str(rgb_path)
            frame["width"], frame["height"] = int(rgb.width), int(rgb.height)
            LOG.info("[%s] RGB saved to %s (%dx%d)", label, rgb_path, rgb.width, rgb.height)
        else:
            frame["rgb_error"] = "empty image"
            LOG.warning("[%s] RGB is empty; check that camera '%s' exists", label, camera)

        if config.want_depth and index < len(depth_responses):
            depth = depth_responses[index]
            if depth is not None and len(depth.image_data_float) > 0:
                array = np.asarray(depth.image_data_float, dtype=np.float32).reshape(
                    depth.height, depth.width
                )
                depth_path = config.out_dir / f"cam{label}_depth.npy"
                np.save(depth_path, array)
                frame["depth"] = str(depth_path)
                frame["depth_min_m"] = float(np.nanmin(array))
                frame["depth_max_m"] = float(np.nanmax(array))
                LOG.info(
                    "[%s] depth saved to %s, range %.2f to %.2f m",
                    label, depth_path, frame["depth_min_m"], frame["depth_max_m"],
                )
            else:
                frame["depth_error"] = "empty image"

        frames.append(frame)

    state = client.getMultirotorState(vehicle_name=config.vehicle)
    kinematics = state.kinematics_estimated
    metadata: Dict[str, Any] = {
        "schema_version": "0.1",
        "created_at": datetime.now().astimezone().isoformat(),
        "sim": {
            "host": config.host or "127.0.0.1",
            "port": config.port,
            "vehicle": config.vehicle,
        },
        "flight": {
            "altitude_m": config.altitude,
            "forward_speed_mps": config.forward_speed,
            "forward_time_sim_s": config.forward_time,
        },
        "pose_ned": {
            "position": vec3(kinematics.position),
            "orientation_xyzw": quat(kinematics.orientation),
            "linear_velocity": vec3(kinematics.linear_velocity),
        },
        "frames": frames,
        # Record the convention explicitly so later modules do not re-invent it.
        "conventions": {
            "frame": "NED",
            "length_unit": "meter",
            "z_axis": "down (negative z is up)",
        },
    }

    metadata_path = config.out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("metadata written to %s", metadata_path)
    return metadata


def land(client: "airsim.MultirotorClient", config: FlightConfig) -> None:
    """Land and release control."""
    client.landAsync(vehicle_name=config.vehicle).join()
    LOG.info("landed")
    client.armDisarm(False, vehicle_name=config.vehicle)
    client.enableApiControl(False, vehicle_name=config.vehicle)
    LOG.info("control released")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> FlightConfig:
    parser = argparse.ArgumentParser(
        description="AirSim minimal loop: connect, take off, fly, capture, land",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=30100, help="AirSim API port")
    parser.add_argument("--vehicle", default="Drone_1", help="vehicle name")
    parser.add_argument(
        "--altitude", type=float, default=5.0, help="target altitude in metres, up positive"
    )
    parser.add_argument("--forward-speed", type=float, default=3.0, help="m/s")
    parser.add_argument("--forward-time", type=float, default=4.0, help="simulated seconds")
    parser.add_argument("--out", type=Path, default=Path("shots"), help="output directory")
    parser.add_argument("--no-depth", action="store_true", help="skip depth capture")
    parser.add_argument(
        "--connect-only", action="store_true", help="test the connection without flying"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    config = FlightConfig(
        port=args.port,
        vehicle=args.vehicle,
        altitude=args.altitude,
        forward_speed=args.forward_speed,
        forward_time=args.forward_time,
        out_dir=args.out,
        want_depth=not args.no_depth,
        connect_only=args.connect_only,
    )
    setup_logging(args.verbose)
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    LOG.debug("configuration: %s", asdict(config))

    client: Optional["airsim.MultirotorClient"] = None
    airborne = False
    exit_code = 0

    try:
        client = connect(config)
        print_state(client, config, "initial")

        if config.connect_only:
            LOG.info("--connect-only given, skipping the flight")
            return 0

        take_off(client, config)
        airborne = True
        print_state(client, config, "after take-off")

        fly_forward(client, config)
        print_state(client, config, "after flying forward")

        capture_views(client, config)

    except KeyboardInterrupt:
        LOG.warning("interrupted (Ctrl+C)")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 - top level turns any failure into a code
        LOG.error("failed: %s", exc)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.exception("traceback:")
        exit_code = 1
    finally:
        # Whatever happened, a drone that left the ground must come back down.
        if airborne and client is not None:
            try:
                land(client, config)
            except Exception as exc:  # noqa: BLE001
                LOG.error("landing failed, check the scene manually: %s", exc)
                exit_code = exit_code or 1

    LOG.info("finished with exit code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())