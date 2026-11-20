#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_builder.py —— 把场景规格书实例化成 AirSim 里真实存在的物体

职责边界
--------
本模块**只做一件事**：读 validated 的 SceneSpec，调用 AirSim API 摆物体。
它不做 LLM 调用，不做场景生成，不做几何验证 —— 那些是别的模块的事。

这个边界很重要：它意味着你可以用**手写的 JSON** 调试整个链路，
而不必先把大模型接进来。把不确定性一个一个消掉，而不是一起消。

核心流程
--------
1. 连接 AirSim，读取可用资产列表
2. 预检：规格书里每个 asset 是否真的可用（不可用就提前报告，而不是中途炸掉）
3. 清理：删掉上一次构建的同前缀物体（保证可重复构建）
4. 生成：逐个 simSpawnObject
5. 校验：回读每个物体位姿，确认真的在世界里
6. 落盘：写一份 build_report.json

用法
----
    # 先干跑，只看计划和资产是否齐全，不动场景
    python scene_builder.py --spec specs/fire_floor2.json --dry-run

    # 真的搭出来
    python scene_builder.py --spec specs/fire_floor2.json

    # 搭完顺便把无人机扔到起点
    python scene_builder.py --spec specs/fire_floor2.json --move-drone

    # 只清理，不重建
    python scene_builder.py --spec specs/fire_floor2.json --cleanup-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import math

try:
    import airsim
except ImportError:
    sys.exit("找不到 airsim 包，请先激活正确环境。")

from scene_spec import (
    SceneSpec,
    SpecError,
    check_assets,
    enu_to_ned,
    load_spec,
    yaw_enu_to_ned,
)

LOG = logging.getLogger("scene_builder")

# 生成物体的命名前缀。清理时只删这个前缀，避免误删场景原有物体。
DEFAULT_PREFIX = "gen"


# --------------------------------------------------------------------------- #
# 连接
# --------------------------------------------------------------------------- #
def connect(port: int, vehicle: str) -> "airsim.MultirotorClient":
    kwargs: Dict[str, Any] = {"port": port}
    try:
        client = airsim.MultirotorClient(**kwargs)
        client.confirmConnection()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"连接 AirSim 失败 (port={port})。检查:\n"
            f"  1) UE 场景是否在运行\n"
            f"  2) ss -lntp | grep {port}\n"
            f"原始错误: {exc}"
        ) from exc

    vehicles = client.listVehicles()
    if vehicle not in vehicles:
        raise RuntimeError(f"场景里没有 '{vehicle}'，实际有: {vehicles}")
    LOG.info("已连接，遥控对象: %s", vehicles)
    return client


# --------------------------------------------------------------------------- #
# 名称与坐标
# --------------------------------------------------------------------------- #
def object_name(spec: SceneSpec, obj_id: str, prefix: str) -> str:
    """生成唯一的 AirSim 物体名。格式: gen_<场景名>_<物体id>"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj_id)
    return f"{prefix}_{spec.name}_{safe}"


def build_pose(spec: SceneSpec, obj) -> "airsim.Pose":
    """把规格书里的局部 ENU 位姿转成 AirSim 世界 NED 位姿。"""
    world_enu = spec.world_position_enu(obj)
    x, y, z = enu_to_ned(world_enu)
    yaw_ned = yaw_enu_to_ned(obj.yaw_deg)
    return airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, math.radians(yaw_ned)),
    )


# --------------------------------------------------------------------------- #
# 清理
# --------------------------------------------------------------------------- #
def cleanup(client, spec: SceneSpec, prefix: str, dry_run: bool = False) -> int:
    """删除上次构建留下的物体，返回删除数量。"""
    try:
        names = client.simListSceneObjects()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("无法列出场景物体，跳过清理: %s", exc)
        return 0

    my_prefix = f"{prefix}_{spec.name}_"
    targets = [n for n in names if n.startswith(my_prefix)]

    if not targets:
        LOG.info("没有需要清理的旧物体（前缀 %s）", my_prefix)
        return 0

    LOG.info("发现 %d 个上次构建的物体，准备清理", len(targets))
    removed = 0
    for name in targets:
        if dry_run:
            removed += 1
            continue
        try:
            client.simDestroyObject(name)
            removed += 1
        except Exception as exc:  # noqa: BLE001
            LOG.warning("删除 %s 失败: %s", name, exc)

    LOG.info("%s %d 个物体", "计划清理" if dry_run else "已清理", removed)
    return removed


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #
def spawn_all(
    client, spec: SceneSpec, prefix: str, dry_run: bool = False
) -> Dict[str, Any]:
    """逐个生成物体，返回结构化报告。"""
    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

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
            LOG.info("[dry-run] 将生成 %-22s %-32s NED=(%.1f, %.1f, %.1f)",
                     obj.id, obj.asset, *ned)
            successes.append(entry)
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
            LOG.error("生成 %s (%s) 抛异常: %s", obj.id, obj.asset, exc)
            entry["error"] = str(exc)
            failures.append(entry)
            continue

        if not ok:
            LOG.error("生成 %s (%s) 返回 False —— 该资产无法在运行时生成",
                      obj.id, obj.asset)
            entry["error"] = "simSpawnObject returned False"
            failures.append(entry)
            continue

        # 回读位姿，确认物体真的存在
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
                LOG.warning("  %s 实际位置偏离计划 %.2f 米", obj.id, drift)
        except Exception as exc:  # noqa: BLE001
            entry["pose_readback_error"] = str(exc)

        LOG.info("✅ %-22s %-32s %.1f,%.1f,%.1f",
                 obj.id, obj.asset, *ned)
        successes.append(entry)

    return {"spawned": successes, "failed": failures}


# --------------------------------------------------------------------------- #
# 无人机落位
# --------------------------------------------------------------------------- #
def move_drone_to_spawn(client, spec: SceneSpec, vehicle: str) -> None:
    """把无人机瞬移到规格书指定的起点（方便直接开始跑 episode）。"""
    if spec.spawn is None:
        LOG.info("规格书没有 spawn 字段，跳过")
        return

    world_enu = (
        spec.origin[0] + spec.spawn.pos[0],
        spec.origin[1] + spec.spawn.pos[1],
        spec.origin[2] + spec.spawn.pos[2],
    )
    x, y, z = enu_to_ned(world_enu)
    yaw = math.radians(yaw_enu_to_ned(spec.spawn.yaw_deg))

    pose = airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(0.0, 0.0, yaw),
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle)
    LOG.info("无人机已移动到起点 NED=(%.1f, %.1f, %.1f)", x, y, z)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="按规格书在 AirSim 中生成场景")
    ap.add_argument("--spec", required=True, help="规格书 JSON 路径")
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--vehicle", default="Drone_1")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="生成物体名前缀")
    ap.add_argument("--dry-run", action="store_true", help="只演练，不改动场景")
    ap.add_argument("--cleanup-only", action="store_true", help="只清理，不重建")
    ap.add_argument("--no-cleanup", action="store_true", help="构建前不清理旧物体")
    ap.add_argument("--move-drone", action="store_true", help="构建后把无人机移到起点")
    ap.add_argument("--report", default="", help="构建报告输出路径")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- 1. 读规格书 ----
    try:
        spec = load_spec(a.spec)
    except (SpecError, FileNotFoundError) as exc:
        LOG.error("规格书加载失败:\n%s", exc)
        return 2

    LOG.info("规格书加载成功")
    print(spec.summary())

    # ---- 2. 连接 ----
    try:
        client = connect(a.port, a.vehicle)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 3

    # ---- 3. 资产预检 ----
    try:
        available = list(client.simListAssets())
    except Exception as exc:  # noqa: BLE001
        LOG.error("无法读取可用资产列表: %s", exc)
        return 3

    missing = check_assets(spec, available)
    if missing:
        LOG.error("以下资产在当前场景中不可用（共 %d 个）:", len(missing))
        for m in missing:
            LOG.error("  - %s", m)
        LOG.error("请改用 simListAssets() 里存在的资产名，或核对拼写。")
        return 4
    LOG.info("资产预检通过：%d 个物体使用的资产全部可用", len(spec.objects))

    # ---- 4. 清理 ----
    if not a.no_cleanup:
        cleanup(client, spec, a.prefix, dry_run=a.dry_run)

    if a.cleanup_only:
        LOG.info("--cleanup-only 模式，结束")
        return 0

    # ---- 5. 生成 ----
    result = spawn_all(client, spec, a.prefix, dry_run=a.dry_run)

    # ---- 6. 移动无人机 ----
    if a.move_drone and not a.dry_run:
        move_drone_to_spawn(client, spec, a.vehicle)

    # ---- 7. 报告 ----
    report = {
        "built_at": datetime.now().astimezone().isoformat(),
        "spec": {"name": spec.name, "version": spec.version, "path": str(a.spec)},
        "dry_run": a.dry_run,
        "totals": {
            "planned": len(spec.objects),
            "spawned": len(result["spawned"]),
            "failed": len(result["failed"]),
        },
        **result,
    }

    report_path = a.report or f"build_report_{spec.name}.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("报告已写入 %s", report_path)

    if result["failed"]:
        LOG.error("有 %d 个物体生成失败，详见报告", len(result["failed"]))
        return 5

    LOG.info("全部 %d 个物体生成成功", result["spawned"].__len__())
    return 0


if __name__ == "__main__":
    sys.exit(main())