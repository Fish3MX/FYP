#!/usr/bin/env python3
"""Smoke test for a newly launched AirSim scene.

Answers one question: can this scene serve as a base for the project? It runs the
six checks below in order, reports PASS/FAIL for each, and writes a scene profile
that later stages can reuse.

    1. the airsim Python package imports
    2. a client can connect on the given port
    3. listVehicles() returns something, and which names
    4. simListAssets() returns something, and whether primitives exist
    5. simSpawnObject() puts one object into the world and it can be read back
    6. simGetImages() returns an image from at least one camera

Nothing here assumes the CityStreet asset set. It detects what the scene actually
provides, so it works against an OpenUAV environment, UAV-ON, or anything else.

Usage
-----
    # defaults: port 30100, auto-detect vehicle, auto-detect cameras
    python scripts/smoke_test_scene.py

    # name the scene so the generated profile is easy to find
    python scripts/smoke_test_scene.py --scene modular_european --out runs/smoke

    # force a probe asset instead of the auto-chosen primitive
    python scripts/smoke_test_scene.py --asset SM_WoodenCrate_01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

LOG = logging.getLogger("smoke_test")

# Primitives that AirSim's UE plugin normally provides. Used only to prefer a
# probe asset with predictable size over an arbitrary mesh.
PRIMITIVE_CANDIDATES = ["Cube", "Sphere", "Plane", "Cylinder", "Cone"]

# Camera names worth trying. OpenUAV and UAV-ON name their cameras differently,
# so the test probes rather than assumes.
CAMERA_CANDIDATES = [
    "FrontCamera", "DownCamera", "LeftCamera", "RightCamera", "RearCamera",
    "0", "1", "2", "3", "4",
]


class Checks:
    """Collects named pass/fail results and prints them at the end."""

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "passed": passed, "detail": detail})
        mark = "PASS" if passed else "FAIL"
        LOG.info("[%s] %s%s", mark, name, f"  -- {detail}" if detail else "")
        return passed

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    def render(self) -> None:
        print()
        print("=" * 72)
        print("smoke test summary")
        print("=" * 72)
        for result in self.results:
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"  [{mark}] {result['check']}")
            if result["detail"]:
                print(f"         {result['detail']}")
        print("=" * 72)
        print("VERDICT:", "usable as a base scene" if self.all_passed else "NOT usable yet")
        print("=" * 72)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_import(checks: Checks):
    try:
        import airsim  # noqa: F401
    except ImportError as exc:
        checks.record("1 airsim package imports", False, str(exc))
        return None
    checks.record("1 airsim package imports", True, getattr(airsim, "__file__", ""))
    return airsim


def check_connect(checks: Checks, airsim, port: int):
    try:
        client = airsim.MultirotorClient(port=port)
        client.confirmConnection()
    except Exception as exc:  # noqa: BLE001
        checks.record(
            f"2 client connects on port {port}", False,
            f"{exc}  (is the environment running? ss -lntp | grep {port})",
        )
        return None
    checks.record(f"2 client connects on port {port}", True, "connection confirmed")
    return client


def check_vehicles(
    checks: Checks, client, requested: str
) -> Tuple[Optional[str], List[str]]:
    try:
        vehicles = list(client.listVehicles())
    except Exception as exc:  # noqa: BLE001
        checks.record("3 listVehicles()", False, str(exc))
        return None, []

    if not vehicles:
        checks.record("3 listVehicles()", False, "no vehicles reported")
        return None, []

    if requested and requested in vehicles:
        chosen = requested
        detail = f"requested '{requested}' found among {vehicles}"
    else:
        chosen = vehicles[0]
        detail = f"vehicles = {vehicles}; using '{chosen}'"
        if requested:
            detail += f" (requested '{requested}' not present)"

    checks.record("3 listVehicles()", True, detail)
    return chosen, vehicles


def check_assets(checks: Checks, client) -> List[str]:
    try:
        assets = list(client.simListAssets())
    except Exception as exc:  # noqa: BLE001
        checks.record("4 simListAssets()", False, str(exc))
        return []

    if not assets:
        checks.record(
            "4 simListAssets()", False,
            "empty list: this build may not support runtime spawning",
        )
        return []

    primitives = [a for a in PRIMITIVE_CANDIDATES if a in assets]
    detail = f"{len(assets)} assets"
    if primitives:
        detail += f"; primitives present: {primitives}"
    else:
        detail += f"; NO primitives found among {PRIMITIVE_CANDIDATES}"
    checks.record("4 simListAssets()", True, detail)
    return assets


def check_spawn(
    checks: Checks, airsim, client, vehicle: str, asset: str
) -> Optional[str]:
    import math

    def count_objects() -> int:
        try:
            return len(list(client.simListSceneObjects()))
        except Exception:  # noqa: BLE001
            return -1

    before = count_objects()
    name = "smoke_probe_1"

    # 5 m ahead, 2 m above the origin, so it is in front of the default camera.
    pose = airsim.Pose(
        airsim.Vector3r(5.0, 0.0, -2.0),
        airsim.to_quaternion(0.0, 0.0, 0.0),
    )
    try:
        ok = client.simSpawnObject(
            object_name=name,
            asset_name=asset,
            pose=pose,
            scale=airsim.Vector3r(1.0, 1.0, 1.0),
            physics_enabled=False,
        )
    except Exception as exc:  # noqa: BLE001
        checks.record("5 simSpawnObject()", False, f"{asset}: {exc}")
        return None

    if not ok:
        checks.record("5 simSpawnObject()", False, f"{asset} returned False")
        return None

    try:
        actual = client.simGetObjectPose(name)
        got = [actual.position.x_val, actual.position.y_val, actual.position.z_val]
    except Exception as exc:  # noqa: BLE001
        checks.record("5 simSpawnObject()", False, f"spawned but readback failed: {exc}")
        return name

    drift = math.dist(got, [5.0, 0.0, -2.0])
    after = count_objects()
    delta = after - before if before >= 0 and after >= 0 else "n/a"
    checks.record(
        "5 simSpawnObject()", True,
        f"'{asset}' at {tuple(round(v, 2) for v in got)}, drift {drift:.2f} m, "
        f"scene objects {before} -> {after} (delta {delta})",
    )
    return name


def check_images(
    checks: Checks, airsim, client, vehicle: str, out_dir: Path
) -> List[str]:
    """Probe candidate camera names and save the first image that arrives."""
    working: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Optional[Path] = None

    for camera in CAMERA_CANDIDATES:
        try:
            responses = client.simGetImages(
                [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True)],
                vehicle_name=vehicle,
            )
        except Exception:  # noqa: BLE001
            continue
        if not responses:
            continue
        response = responses[0]
        if response is None or len(response.image_data_uint8) == 0:
            continue

        working.append(camera)
        if saved is None:
            saved = out_dir / f"smoke_{camera}.png"
            airsim.write_file(str(saved), response.image_data_uint8)

    if not working:
        checks.record("6 simGetImages()", False, "no camera returned an image")
        return []

    detail = f"working cameras: {working}"
    if saved is not None:
        detail += f"; saved {saved}"
    checks.record("6 simGetImages()", True, detail)
    return working


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that an AirSim scene can serve as a project base",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=30100, help="AirSim API port")
    parser.add_argument("--vehicle", default="", help="vehicle name; empty means auto")
    parser.add_argument("--asset", default="", help="probe asset; empty means auto")
    parser.add_argument("--scene", default="", help="scene id, for the saved profile")
    parser.add_argument("--out", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--keep", action="store_true", help="leave the probe object")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    report: Dict[str, Any] = {
        "tested_at": datetime.now().astimezone().isoformat(),
        "port": args.port,
        "scene": args.scene or "unnamed",
    }

    # ---- 1. import ----
    airsim = check_import(checks)
    if airsim is None:
        checks.render()
        return 2

    # ---- 2. connect ----
    client = check_connect(checks, airsim, args.port)
    if client is None:
        checks.render()
        return 3

    # ---- 3. vehicles ----
    vehicle, vehicles = check_vehicles(checks, client, args.vehicle)
    report["vehicles"] = vehicles
    report["vehicle_used"] = vehicle
    if vehicle is None:
        checks.render()
        return 4

    # ---- 4. assets ----
    assets = check_assets(checks, client)
    report["asset_count"] = len(assets)
    report["primitives_present"] = [a for a in PRIMITIVE_CANDIDATES if a in assets]
    report["asset_sample"] = assets[:40]
    if not assets:
        checks.render()
        return 5

    # ---- profile draft, written before spawning ----
    # Written here on purpose: even if object spawning is unsupported, the
    # vehicle, port and asset list are already valuable and worth keeping.
    def write_profile(camera_names: List[str]) -> Path:
        payload = {
            "scene_id": args.scene or "unnamed",
            "port": args.port,
            "vehicle": vehicle,
            "cameras": camera_names,
            "primitives_present": report["primitives_present"],
            "asset_count": len(assets),
            "assets": assets,
        }
        target = args.out / f"profile_{args.scene or 'unnamed'}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    profile_path = write_profile([])
    LOG.info("scene profile written early: %s", profile_path)

    # ---- 5. spawn ----
    probe_asset = args.asset or ("Cube" if "Cube" in assets else assets[0])
    report["probe_asset"] = probe_asset
    spawned = check_spawn(checks, airsim, client, vehicle, probe_asset)
    if spawned is None:
        checks.render()
        print(f"\nprofile written anyway: {profile_path}")
        print("spawning failed, so the scene cannot host generated objects yet.")
        return 6

    # ---- 6. images ----
    cameras = check_images(checks, airsim, client, vehicle, args.out)
    report["working_cameras"] = cameras

    # ---- cleanup ----
    if not args.keep and spawned:
        try:
            client.simDestroyObject(spawned)
            LOG.info("probe object removed")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not remove the probe object: %s", exc)

    # ---- profile, now with the cameras that actually returned images ----
    if cameras:
        profile_path = write_profile(cameras)
        LOG.info("scene profile updated with %d camera(s): %s", len(cameras), profile_path)

    report["checks"] = checks.results
    report["profile_path"] = str(profile_path)
    (args.out / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checks.render()
    print()
    print(f"scene profile written to {profile_path}")
    print(f"full report written to {args.out / 'smoke_report.json'}")
    print()
    print("next step: send profile_*.json back so the asset catalogue can be")
    print("rebuilt against this scene instead of the CityStreet one.")

    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())