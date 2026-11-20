#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hello_uav.py —— 无人机仿真链路验证脚本 (AirSim + UE4)

用途
----
验证 "Python -> AirSim -> UE 场景" 这条链路是否完全打通，并完成一次最小闭环：
连接 -> 解锁/起飞 -> 爬升 -> 前飞 -> 四视角采集(RGB + 深度) -> 降落。

这是整个毕设的地基脚本。它刻意**不依赖任何项目代码**，只依赖 AirSim 官方
Python API，因此当后面出问题时，你可以用它来判断故障在"环境"还是在"自己的代码"。

输出目录
--------
    <out>/cam<name>_rgb.png      各视角 RGB (默认 512x512)
    <out>/cam<name>_depth.npy    各视角 DepthPlanar (单位: 米, float32)
    <out>/metadata.json          位姿、时间戳、相机配置 —— 供后续场景规格书使用

用法
----
    # 最简
    python hello_uav.py

    # 指定端口和机型
    python hello_uav.py --port 30100 --vehicle Drone_1

    # 飞高一点、飞远一点，输出到独立目录
    python hello_uav.py --altitude 8 --forward-time 8 --out runs/demo1

    # 只测连接，不飞行
    python hello_uav.py --connect-only

    # 调试用：打印详细日志
    python hello_uav.py -v

坐标系约定（重要）
------------------
AirSim 使用 NED 右手系：x 向前、y 向右、z **向下**。
因此"上升 5 米"要写成 z = -5。本脚本对外暴露的 --altitude 是"向上为正"的
人类习惯值，内部统一转换成 -altitude，避免每个调用点各写一遍负号。
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
except ImportError:  # pragma: no cover - 环境问题，给出可执行的修复提示
    sys.exit(
        "找不到 airsim 包。请先激活虚拟环境：\n"
        "    source /home/SIGPhysics/venvs/airsim/bin/activate"
    )

LOG = logging.getLogger("hello_uav")

# AirSim 的图像类型常量（用常量而不是魔法数字，读代码时不用查文档）
IMG_RGB = airsim.ImageType.Scene          # 0
IMG_DEPTH = airsim.ImageType.DepthPlanar  # 2


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class FlightConfig:
    """一次飞行验证的全部参数。"""

    host: str = ""                       # 空字符串 = 本机回环
    port: int = 30100                    # 由 settings.json 的 ApiServerPort 决定
    vehicle: str = "Drone_1"             # 由 settings.json 的 Vehicles 键名决定
    altitude: float = 5.0                # 米，向上为正
    forward_speed: float = 3.0           # m/s
    forward_time: float = 4.0            # 仿真秒（受 settings.json 的 ClockSpeed 影响）
    out_dir: Path = Path("shots")
    want_depth: bool = True
    connect_only: bool = False
    takeoff_timeout: float = 20.0

    # camera_name -> 人类可读标签；来自 settings.json 里的 Cameras 键
    cameras: Dict[str, str] = field(
        default_factory=lambda: {"0": "front", "1": "left", "2": "right", "3": "down"}
    )

    @property
    def ned_altitude(self) -> float:
        """把"向上为正"的高度转成 AirSim 的 NED z 值。"""
        return -abs(self.altitude)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # airsim 的库日志比较吵，只在 -v 时放出来
    logging.getLogger("airsim").setLevel(logging.DEBUG if verbose else logging.WARNING)


def vec3(v: Any) -> List[float]:
    """把 AirSim 的 Vector3r 转成普通 list，方便写 JSON。"""
    return [float(v.x_val), float(v.y_val), float(v.z_val)]


def quat(q: Any) -> List[float]:
    return [float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)]


# --------------------------------------------------------------------------- #
# 核心步骤
# --------------------------------------------------------------------------- #
def connect(cfg: FlightConfig) -> "airsim.MultirotorClient":
    """建立连接并做一次握手。失败时抛出带排查建议的异常。"""
    LOG.info("正在连接 AirSim %s:%d ...", cfg.host or "127.0.0.1", cfg.port)
    # 注意：不要把空字符串传给 ip，airsim 会拿它去建 RPC 地址而失败
    kwargs: Dict[str, Any] = {"port": cfg.port}
    if cfg.host:
        kwargs["ip"] = cfg.host
    try:
        client = airsim.MultirotorClient(**kwargs)
        client.confirmConnection()
    except Exception as exc:
        raise RuntimeError(
            f"连接 AirSim 失败 ({cfg.host or '127.0.0.1'}:{cfg.port})。请依次检查：\n"
            f"  1) UE 场景是否真的启动了（终端里应能看到 'Drone_1' 字样）\n"
            f"  2) 端口是否在监听：ss -lntp | grep {cfg.port}\n"
            f"  3) settings.json 的 ApiServerPort 是否等于 {cfg.port}\n"
            f"原始错误: {exc}"
        ) from exc

    vehicles = client.listVehicles()
    LOG.info("已连接。当前遥控对象: %s", vehicles)
    if cfg.vehicle not in vehicles:
        raise RuntimeError(
            f"场景里没有名为 '{cfg.vehicle}' 的遥控对象，实际有: {vehicles}。\n"
            f"请用 --vehicle 指定其中一个。"
        )
    return client


def print_state(client: "airsim.MultirotorClient", cfg: FlightConfig, note: str) -> None:
    """打印当前位置，方便肉眼确认飞行动作真的生效了。"""
    state = client.getMultirotorState(vehicle_name=cfg.vehicle)
    p = state.kinematics_estimated.position
    LOG.info("%s -> 位置 (NED) x=%.2f y=%.2f z=%.2f", note, p.x_val, p.y_val, p.z_val)


def take_off(client: "airsim.MultirotorClient", cfg: FlightConfig) -> None:
    """解锁并起飞，然后爬升到目标高度。"""
    client.enableApiControl(True, vehicle_name=cfg.vehicle)
    client.armDisarm(True, vehicle_name=cfg.vehicle)
    LOG.info("已解锁 API 控制")

    client.takeoffAsync(timeout_sec=cfg.takeoff_timeout, vehicle_name=cfg.vehicle).join()
    LOG.info("已起飞")

    # moveToZAsync 的第二个参数是最大速度(m/s)
    client.moveToZAsync(
        cfg.ned_altitude, 2.0, vehicle_name=cfg.vehicle
    ).join()
    LOG.info("已爬升到 %.1f 米", cfg.altitude)


def fly_forward(client: "airsim.MultirotorClient", cfg: FlightConfig) -> None:
    """保持当前高度，以恒定速度向前飞一段仿真时间。"""
    client.moveByVelocityZAsync(
        vx=cfg.forward_speed,
        vy=0.0,
        z=cfg.ned_altitude,
        duration=cfg.forward_time,
        vehicle_name=cfg.vehicle,
    ).join()
    client.hoverAsync(vehicle_name=cfg.vehicle).join()

    distance = cfg.forward_speed * cfg.forward_time
    LOG.info("已前飞约 %.0f 米并悬停", distance)


def capture_views(
    client: "airsim.MultirotorClient", cfg: FlightConfig
) -> Dict[str, Any]:
    """采集所有视角的 RGB + 深度，并把元数据落盘。返回元数据字典。"""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    cam_items = list(cfg.cameras.items())
    rgb_requests = [
        airsim.ImageRequest(
            camera_name=cam, image_type=IMG_RGB, pixels_as_float=False, compress=True
        )
        for cam, _ in cam_items
    ]
    rgb_responses = client.simGetImages(rgb_requests, vehicle_name=cfg.vehicle)

    depth_responses: Sequence[Any] = []
    if cfg.want_depth:
        depth_requests = [
            airsim.ImageRequest(
                camera_name=cam,
                image_type=IMG_DEPTH,
                pixels_as_float=True,
                compress=False,
            )
            for cam, _ in cam_items
        ]
        depth_responses = client.simGetImages(depth_requests, vehicle_name=cfg.vehicle)

    frames: List[Dict[str, Any]] = []

    for idx, ((cam, label), rgb) in enumerate(zip(cam_items, rgb_responses)):
        frame: Dict[str, Any] = {"camera": cam, "label": label}

        # ---- RGB ----
        if rgb is not None and len(rgb.image_data_uint8) > 0:
            rgb_path = cfg.out_dir / f"cam{label}_rgb.png"
            airsim.write_file(str(rgb_path), rgb.image_data_uint8)
            frame["rgb"] = str(rgb_path)
            frame["width"], frame["height"] = int(rgb.width), int(rgb.height)
            LOG.info("[%s] RGB 已保存 %s (%dx%d)", label, rgb_path, rgb.width, rgb.height)
        else:
            frame["rgb_error"] = "empty image"
            LOG.warning("[%s] RGB 为空，检查相机名 '%s' 是否存在", label, cam)

        # ---- Depth ----
        if cfg.want_depth and idx < len(depth_responses):
            dep = depth_responses[idx]
            if dep is not None and len(dep.image_data_float) > 0:
                arr = np.asarray(dep.image_data_float, dtype=np.float32).reshape(
                    dep.height, dep.width
                )
                dep_path = cfg.out_dir / f"cam{label}_depth.npy"
                np.save(dep_path, arr)
                frame["depth"] = str(dep_path)
                frame["depth_min_m"] = float(np.nanmin(arr))
                frame["depth_max_m"] = float(np.nanmax(arr))
                LOG.info(
                    "[%s] 深度已保存 %s  范围 %.2f ~ %.2f 米",
                    label,
                    dep_path,
                    frame["depth_min_m"],
                    frame["depth_max_m"],
                )
            else:
                frame["depth_error"] = "empty image"

        frames.append(frame)

    # ---- 位姿元数据：后续生成场景规格书时要用 ----
    state = client.getMultirotorState(vehicle_name=cfg.vehicle)
    kin = state.kinematics_estimated
    metadata: Dict[str, Any] = {
        "schema_version": "0.1",
        "created_at": datetime.now().astimezone().isoformat(),
        "sim": {"host": cfg.host or "127.0.0.1", "port": cfg.port, "vehicle": cfg.vehicle},
        "flight": {
            "altitude_m": cfg.altitude,
            "forward_speed_mps": cfg.forward_speed,
            "forward_time_sim_s": cfg.forward_time,
        },
        "pose_ned": {
            "position": vec3(kin.position),
            "orientation_xyzw": quat(kin.orientation),
            "linear_velocity": vec3(kin.linear_velocity),
        },
        "frames": frames,
        # 显式记录坐标系约定，避免后续模块各写各的转换
        "conventions": {
            "frame": "NED",
            "length_unit": "meter",
            "z_axis": "down (negative z is up)",
        },
    }

    meta_path = cfg.out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("元数据已保存 %s", meta_path)
    return metadata


def land(client: "airsim.MultirotorClient", cfg: FlightConfig) -> None:
    """降落并释放控制权。"""
    client.landAsync(vehicle_name=cfg.vehicle).join()
    LOG.info("已降落")
    client.armDisarm(False, vehicle_name=cfg.vehicle)
    client.enableApiControl(False, vehicle_name=cfg.vehicle)
    LOG.info("已释放控制权")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> FlightConfig:
    p = argparse.ArgumentParser(
        description="AirSim 最小闭环验证：连接-起飞-前飞-采集-降落",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=30100, help="AirSim API 端口")
    p.add_argument("--vehicle", default="Drone_1", help="遥控对象名")
    p.add_argument("--altitude", type=float, default=5.0, help="目标高度(米，向上为正)")
    p.add_argument("--forward-speed", type=float, default=3.0, help="前飞速度(m/s)")
    p.add_argument("--forward-time", type=float, default=4.0, help="前飞时长(仿真秒)")
    p.add_argument("--out", type=Path, default=Path("shots"), help="输出目录")
    p.add_argument("--no-depth", action="store_true", help="不采集深度图")
    p.add_argument("--connect-only", action="store_true", help="只测试连接，不飞行")
    p.add_argument("-v", "--verbose", action="store_true", help="打印详细日志")

    a = p.parse_args(argv)
    cfg = FlightConfig(
        port=a.port,
        vehicle=a.vehicle,
        altitude=a.altitude,
        forward_speed=a.forward_speed,
        forward_time=a.forward_time,
        out_dir=a.out,
        want_depth=not a.no_depth,
        connect_only=a.connect_only,
    )
    setup_logging(a.verbose)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    LOG.debug("配置: %s", asdict(cfg))

    client: Optional["airsim.MultirotorClient"] = None
    airborne = False
    exit_code = 0

    try:
        client = connect(cfg)
        print_state(client, cfg, "初始")

        if cfg.connect_only:
            LOG.info("--connect-only 模式，跳过飞行")
            return 0

        take_off(client, cfg)
        airborne = True
        print_state(client, cfg, "起飞后")

        fly_forward(client, cfg)
        print_state(client, cfg, "前飞后")

        capture_views(client, cfg)

    except KeyboardInterrupt:
        LOG.warning("收到中断信号（Ctrl+C）")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 - 顶层需要把任何失败转成非零退出码
        LOG.error("执行失败: %s", exc)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.exception("详细堆栈：")
        exit_code = 1
    finally:
        # 无论成功失败，只要上过天就必须落下来，避免无人机悬在半空
        if airborne and client is not None:
            try:
                land(client, cfg)
            except Exception as exc:  # noqa: BLE001
                LOG.error("降落失败，请手动检查场景状态: %s", exc)
                exit_code = exit_code or 1

    LOG.info("结束，退出码 %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())