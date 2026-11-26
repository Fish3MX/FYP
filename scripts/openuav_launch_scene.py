#!/usr/bin/env python3
"""Launch an OpenUAV environment through its own RPC server, then find its port.

Background
----------
`AirVLNSimulatorServerTool.py` does not start any scene at startup. It listens on
a control port (for example 30000) and waits for a client to ask for scenes. On
request it:

  1. assigns AirSim ports starting at `control_port + 1`
  2. writes a generated settings.json into `airsim_plugin/settings/<port>/`
  3. launches `bash <root>/closeloop_envs/<Env>.sh -RenderOffscreen ...`
  4. the environment then listens for AirSim clients on that assigned port

So "the server is listening on 30000" means the launcher is ready, not that a
simulation exists. This script performs step 1-4 and reports the port that
actually answers.

Caveats about the generated settings
------------------------------------
The tool's built-in template sets `LockStep: true` and `ViewMode: NoDisplay`.
With LockStep the simulation clock only advances when the client steps it, so
`takeoffAsync().join()` will wait forever unless the client calls
`simContinueForTime()`. For plain flight testing prefer launching the `.sh`
directly with our own settings (`scenes/settings/*.json`), where LockStep is off.

Usage
-----
    # start ModularEuropean on the first GPU
    python scripts/openuav_launch_scene.py --env ModularEuropean --gpu 0

    # just probe which AirSim port is already answering
    python scripts/openuav_launch_scene.py --probe-only

    # shut every scene down
    python scripts/openuav_launch_scene.py --close
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import List, Optional, Sequence

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    """Is something listening on this TCP port?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def find_airsim_ports(control_port: int, scan: int = 8) -> List[int]:
    """The server assigns control_port+1, +2, ... so scan that range."""
    found = []
    for offset in range(1, scan + 1):
        candidate = control_port + offset
        if port_open(candidate):
            found.append(candidate)
    return found


def wait_for_port(
    control_port: int, scan: int = 8, timeout: float = 90.0
) -> Optional[int]:
    """Poll until at least one scene port answers, or the timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ports = find_airsim_ports(control_port, scan)
        if ports:
            return ports[0]
        time.sleep(2.0)
    return None


def connect_control(control_port: int, timeout: float = 180.0):
    try:
        import msgpackrpc
    except ImportError:
        print(
            "msgpackrpc is not installed in this Python environment.\n"
            "OpenUAV's server needs it, so use the same environment that runs\n"
            "AirVLNSimulatorServerTool.py, for example:\n"
            "    source /home/SIGPhysics/venvs/airsim/bin/activate\n"
            "or the conda env where openuav was installed."
        )
        return None

    address = msgpackrpc.Address("127.0.0.1", control_port)
    return msgpackrpc.Client(address, timeout=timeout)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch an OpenUAV scene through its RPC server and find its port",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--control-port", type=int, default=30000,
                        help="port the server tool listens on")
    parser.add_argument("--env", default="ModularEuropean",
                        help="environment id as listed in env_exec_path_dict")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index passed to -GraphicsAdapter")
    parser.add_argument("--probe-only", action="store_true",
                        help="do not launch anything, only report open ports")
    parser.add_argument("--close", action="store_true",
                        help="ask the server to close all scenes")
    parser.add_argument("--scan", type=int, default=8,
                        help="how many ports above the control port to scan")
    parser.add_argument("--wait", type=float, default=90.0,
                        help="seconds to wait for the scene to come up")
    args = parser.parse_args(argv)

    print(f"control server : 127.0.0.1:{args.control_port}")
    print(f"control port open: {port_open(args.control_port)}")

    if args.probe_only:
        ports = find_airsim_ports(args.control_port, args.scan)
        if ports:
            print(f"AirSim ports answering: {ports}")
            print(f"\nnext: python scripts/smoke_test_scene.py --port {ports[0]}")
            return 0
        print(
            "no AirSim port is answering above the control port.\n"
            "The launcher is up but no scene has been started yet.\n"
            f"run: python scripts/openuav_launch_scene.py --env {args.env} --gpu {args.gpu}"
        )
        return 1

    client = connect_control(args.control_port)
    if client is None:
        return 2

    try:
        if args.close:
            print("asking the server to close all scenes ...")
            client.call("close_scenes", "127.0.0.1")
            print("close_scenes returned")
            return 0

        print("ping ...")
        client.call("ping")
        print("ping ok")

        request = [[args.env, args.gpu]]
        print(f"reopen_scenes({request}) ...")
        result = client.call("reopen_scenes", "127.0.0.1", request)
        print(f"reopen_scenes returned: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"RPC call failed: {exc}")
        print(
            "\ncommon causes:\n"
            "  - the server was started with --gpus that do not include this index\n"
            "  - the environment id is not a key in env_exec_path_dict\n"
            "  - env_exec_path_dict paths do not match --root_path, so the .sh is missing"
        )
        return 3

    print(f"\nwaiting up to {args.wait:.0f}s for an AirSim port to answer ...")
    port = wait_for_port(args.control_port, args.scan, args.wait)

    if port is None:
        print(
            "timed out: no AirSim port came up.\n"
            "check the server's own stdout for the bash command it ran and any\n"
            "error from the environment, and confirm the paths in\n"
            "env_exec_path_dict resolve under --root_path."
        )
        return 4

    print(f"\nAirSim is answering on port {port}")
    print("\nnext steps:")
    print(f"  python scripts/smoke_test_scene.py --port {port} --scene {args.env}")
    print("\nnote: the server-generated settings use LockStep and NoDisplay.")
    print("      connections and image capture work, but async flight commands may")
    print("      block until the clock is stepped with simContinueForTime().")
    print("      for flight experiments, launch the .sh directly with")
    print("      scenes/settings/modular_european.json instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())