#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_assets.py —— 探测当前 UE 场景是否支持"运行时动态摆放物体"

为什么先测这个
--------------
毕设的核心动作是"读 JSON -> 在场景里摆出物体"。这依赖于 AirSim 的
simSpawnObject API，而该 API 只能生成 **UE 工程里已经存在的资产**。

不同 UE 打包工程暴露的资产差异极大：
  - 有的工程暴露了成百上千个可用资产  -> 可以走"运行时摆放"路线
  - 有的只暴露 1~3 个占位物            -> 必须走"离线编辑+打包"路线

这个脚本用 30 秒告诉你属于哪一种，从而决定后面所有方案。

用法
----
    python probe_assets.py
    python probe_assets.py --port 30100 --vehicle Drone_1
    python probe_assets.py --spawn        # 额外尝试真的生成一个物体
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
    sys.exit("找不到 airsim 包，请先激活正确环境。")

LOG = logging.getLogger("probe_assets")


def connect(port: int, vehicle: str) -> "airsim.MultirotorClient":
    client = airsim.MultirotorClient(port=port)
    client.confirmConnection()
    LOG.info("已连接 %s，遥控对象: %s", port, client.listVehicles())
    return client


def list_assets(client: "airsim.MultirotorClient") -> List[str]:
    """列出场景里所有可在运行时生成的资产名。"""
    try:
        assets = client.simListAssets()
    except Exception as exc:  # noqa: BLE001
        LOG.error(
            "调用 simListAssets 失败: %s\n"
            "这通常意味着该 AirSim 版本不支持此 API，或场景禁止资产查询。",
            exc,
        )
        return []
    return list(assets) if assets else []


def existing_objects(client: "airsim.MultirotorClient") -> int:
    """统计场景里已存在的物体数量（用于对比 spawn 前后的变化）。"""
    try:
        objs = client.simListSceneObjects()
        return len(objs) if objs else 0
    except Exception as exc:  # noqa: BLE001
        LOG.warning("simListSceneObjects 不可用: %s", exc)
        return -1


def try_spawn(client: "airsim.MultirotorClient", asset: str, vehicle: str) -> Optional[str]:
    """
    在无人机旁边尝试生成一个资产实例。

    返回物体名表示成功，None 表示失败。
    """
    name = "probe_object_1"
    # 放在无人机东南方 3 米处，避免和机身重叠
    pose = airsim.Pose(
        airsim.Vector3r(5.0, 3.0, -1.0),  # NED: 向前5m, 向右3m, 向下1m
        airsim.to_quaternion(0, 0, 0),
    )
    try:
        ok = client.simSpawnObject(
            object_name=name,
            asset_name=asset,
            pose=pose,
            scale=airsim.Vector3r(1, 1, 1),
            physics_enabled=False,   # 先不带物理，验证"能不能生成"
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("simSpawnObject 抛异常: %s", exc)
        return None

    if not ok:
        LOG.error("simSpawnObject 返回 False —— 该资产无法在运行时生成")
        return None

    # 回读位姿，确认物体真的在世界里
    try:
        actual = client.simGetObjectPose(name)
        LOG.info("生成成功，回读位姿: %s", actual.position)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("已生成但回读位姿失败: %s", exc)

    return name


def cleanup(client: "airsim.MultirotorClient", name: str) -> None:
    try:
        client.simDestroyObject(name)
        LOG.info("已销毁探测物体 %s", name)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("销毁失败（可忽略）: %s", exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="探测 UE 场景的运行时资产能力")
    p.add_argument("--port", type=int, default=30100)
    p.add_argument("--vehicle", default="Drone_1")
    p.add_argument("--spawn", action="store_true", help="尝试真的生成一个物体")
    p.add_argument("--json", type=str, default="", help="把结果写入指定 JSON 文件")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report: Dict[str, Any] = {"port": a.port, "vehicle": a.vehicle}
    client = connect(a.port, a.vehicle)

    # 1. 已存在的场景物体数量
    before = existing_objects(client)
    report["scene_object_count_before"] = before
    LOG.info("场景内已有物体数量: %s", before)

    # 2. 可生成的资产列表
    assets = list_assets(client)
    report["available_assets"] = assets
    report["available_asset_count"] = len(assets)

    if assets:
        LOG.info("可用资产 %d 个:", len(assets))
        for i, name in enumerate(assets[:50]):
            print(f"   [{i:3d}] {name}")
        if len(assets) > 50:
            print(f"   ... 其余 {len(assets) - 50} 个省略")
    else:
        LOG.warning("没有查到任何可用资产 —— 该场景很可能不支持运行时摆放")

    # 3. 可选：真的生成一个
    if a.spawn and assets:
        target = assets[0]
        LOG.info("尝试生成资产 '%s' ...", target)
        spawned = try_spawn(client, target, a.vehicle)
        report["spawn_attempted"] = target
        report["spawn_succeeded"] = spawned is not None
        if spawned:
            after = existing_objects(client)
            report["scene_object_count_after"] = after
            LOG.info("场景物体数 %s -> %s", before, after)
            cleanup(client, spawned)

    # 4. 结论
    if len(assets) >= 50:
        verdict = "runtime_placement_viable"
        msg = "资产充足，可以走【运行时可摆放】路线（快，无需重新打包）"
    elif assets:
        verdict = "runtime_placement_limited"
        msg = f"仅有 {len(assets)} 个资产，运行时可摆放但表达力受限，可能需要离线补资产"
    else:
        verdict = "runtime_placement_unavailable"
        msg = "不支持运行时摆放，必须走【离线编辑器 + 重新打包】路线"
    report["verdict"] = verdict
    LOG.info("结论: %s —— %s", verdict, msg)

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        LOG.info("报告已写入 %s", a.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())