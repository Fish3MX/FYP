#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scene_spec.py —— 场景规格书的定义、校验与坐标转换

设计原则
--------
1. **规格书用 ENU（东-北-天，米）书写**。这是人类和 LLM 都容易理解的坐标系，
   也是地理/机器人领域的通行约定：x=东, y=北, z=上, yaw 从东开始逆时针。
2. **转换只在一处发生**。AirSim 用 NED（x=北, y=东, z=下, yaw 从北顺时针）。
   所有 ENU→NED 的转换集中在 `enu_to_ned()` / `yaw_enu_to_ned()` 两个函数里，
   任何其他模块都不许自己算。这是工程上最容易出错、也最容易避免的一类 bug。
3. **规格书是纯数据**。它不引用任何 AirSim 对象，因此可以在没有引擎的
   CPU 笔记本上被生成、校验、验证、单元测试。

规格书结构（v0.1）
------------------
{
  "version": "0.1",
  "name": "fire_floor2",
  "seed": 42,
  "origin": {"enu": [40.0, -20.0, 0.0]},   # 场景局部原点在世界中的 ENU 位置
  "objects": [
    {
      "id": "wall_north",                    # 场景内唯一，用于引用与清理
      "asset": "SM_build_b_mod_wall_a",      # 必须是 simListAssets() 里的名字
      "pos": [0.0, 5.0, 1.5],               # 相对 origin 的 ENU 位置（米）
      "yaw_deg": 90.0,                       # 相对 ENU 东方向逆时针角度
      "scale": [1.0, 1.0, 1.0],
      "physics": true,                       # 是否参与碰撞（墙应为 true）
      "role": "structure"                    # structure/prop/target/hazard
    }
  ],
  "target": {"object_id": "survivor_1", "landing_radius_m": 5.0},
  "spawn": {"pos": [-30.0, 0.0, 10.0], "yaw_deg": 0.0},
  "requirements": [
    {"kind": "max_time_s", "value": 300},
    {"kind": "min_altitude_m", "value": 3.0}
  ],
  "instruction": "Fly to the two-storey building and find the person on the second floor."
}

用法
----
    from scene_spec import load_spec, enu_to_ned
    spec = load_spec("specs/fire_floor2.json")   # 校验失败会抛出 SpecError
    print(spec.summary())
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SPEC_VERSION = "0.1"

# 允许的 role，用来区分"结构/道具/目标/危险源"，后续验证器会按 role 做不同检查
VALID_ROLES = {"structure", "prop", "target", "hazard"}


class SpecError(ValueError):
    """规格书校验失败。携带所有问题，而不是只报第一个。"""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(
            "规格书校验失败，共 %d 个问题:\n" % len(self.problems)
            + "\n".join("  - " + p for p in self.problems)
        )


# --------------------------------------------------------------------------- #
# 坐标转换：全项目唯一入口
# --------------------------------------------------------------------------- #
def enu_to_ned(enu: Sequence[float]) -> Tuple[float, float, float]:
    """
    ENU (东, 北, 上)  ->  NED (北, 东, 下)，单位统一为米。

        x_ned =  y_enu   (北)
        y_ned =  x_enu   (东)
        z_ned = -z_enu   (上 -> 下)
    """
    e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
    return (n, e, -u)


def ned_to_enu(ned: Sequence[float]) -> Tuple[float, float, float]:
    """NED -> ENU，用于把 AirSim 回读的位姿写回人类可读的规格书。"""
    n, e, d = float(ned[0]), float(ned[1]), float(ned[2])
    return (e, n, -d)


def yaw_enu_to_ned(yaw_deg: float) -> float:
    """
    ENU 下 yaw 从东(+x) 逆时针；AirSim(NED) 下 yaw 从北 顺时针。

    两者关系： yaw_ned = 90 - yaw_enu  （再归一化到 [0,360)）

    自检：
        yaw_enu = 0   (朝东) -> yaw_ned = 90  (NED 中朝东) ✔
        yaw_enu = 90  (朝北) -> yaw_ned = 0   (NED 中朝北) ✔
        yaw_enu = 180 (朝西) -> yaw_ned = 270 ✔
    """
    return (90.0 - float(yaw_deg)) % 360.0


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class SceneObject:
    id: str
    asset: str
    pos: Tuple[float, float, float]          # ENU，相对 origin
    yaw_deg: float = 0.0
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    physics: bool = False
    role: str = "prop"


@dataclass
class Target:
    object_id: str
    landing_radius_m: float = 5.0


@dataclass
class Spawn:
    pos: Tuple[float, float, float]           # ENU
    yaw_deg: float = 0.0


@dataclass
class SceneSpec:
    version: str
    name: str
    seed: int
    origin: Tuple[float, float, float]        # ENU，世界坐标
    objects: List[SceneObject]
    target: Optional[Target] = None
    spawn: Optional[Spawn] = None
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    instruction: str = ""

    # ---------- 便捷查询 ----------
    def object_by_id(self, obj_id: str) -> Optional[SceneObject]:
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None

    def world_position_enu(self, obj: SceneObject) -> Tuple[float, float, float]:
        """把局部坐标加上 origin，得到世界 ENU 坐标。"""
        return (
            self.origin[0] + obj.pos[0],
            self.origin[1] + obj.pos[1],
            self.origin[2] + obj.pos[2],
        )

    def asset_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for o in self.objects:
            hist[o.asset] = hist.get(o.asset, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        lines = [
            f"场景 '{self.name}'  v{self.version}  seed={self.seed}",
            f"  原点(ENU)   : {self.origin}",
            f"  物体数      : {len(self.objects)}",
            f"  role 分布   : "
            + ", ".join(
                f"{r}={sum(1 for o in self.objects if o.role == r)}"
                for r in sorted(VALID_ROLES)
            ),
            f"  目标        : {self.target.object_id if self.target else '(无)'}",
            f"  起点        : {self.spawn.pos if self.spawn else '(无)'}",
            f"  需求断言    : {len(self.requirements)} 条",
        ]
        if self.instruction:
            lines.append(f"  指令        : {self.instruction[:70]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 解析与校验
# --------------------------------------------------------------------------- #
def _as_vec3(value: Any, where: str, problems: List[str]) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        problems.append(f"{where}: 期望长度为 3 的数组，得到 {value!r}")
        return (0.0, 0.0, 0.0)
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        problems.append(f"{where}: 元素必须是数字，得到 {value!r}")
        return (0.0, 0.0, 0.0)


def parse_spec(data: Dict[str, Any]) -> SceneSpec:
    """把 dict 解析成 SceneSpec，收集所有校验问题后一次性抛出。"""
    problems: List[str] = []

    if not isinstance(data, dict):
        raise SpecError(["规格书顶层必须是 JSON 对象"])

    version = str(data.get("version", ""))
    if version != SPEC_VERSION:
        problems.append(
            f"version: 期望 '{SPEC_VERSION}'，得到 '{version}'"
        )

    name = str(data.get("name", "")).strip()
    if not name:
        problems.append("name: 不能为空")

    try:
        seed = int(data.get("seed", 0))
    except (TypeError, ValueError):
        problems.append(f"seed: 必须是整数，得到 {data.get('seed')!r}")
        seed = 0

    origin = _as_vec3(data.get("origin", [0, 0, 0]), "origin", problems)

    # ---- objects ----
    raw_objects = data.get("objects", [])
    objects: List[SceneObject] = []
    seen_ids: set[str] = set()

    if not isinstance(raw_objects, list) or not raw_objects:
        problems.append("objects: 必须是非空数组")
        raw_objects = []

    for idx, raw in enumerate(raw_objects):
        where = f"objects[{idx}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: 必须是对象")
            continue

        obj_id = str(raw.get("id", "")).strip()
        if not obj_id:
            problems.append(f"{where}.id: 不能为空")
            continue
        if obj_id in seen_ids:
            problems.append(f"{where}.id: '{obj_id}' 重复")
        seen_ids.add(obj_id)

        asset = str(raw.get("asset", "")).strip()
        if not asset:
            problems.append(f"{where}.asset: 不能为空（必须是 simListAssets 返回的名字）")

        pos = _as_vec3(raw.get("pos", [0, 0, 0]), f"{where}.pos", problems)
        scale = _as_vec3(raw.get("scale", [1, 1, 1]), f"{where}.scale", problems)

        if any(s <= 0 for s in scale):
            problems.append(f"{where}.scale: 三个分量都必须 > 0，得到 {scale}")

        role = str(raw.get("role", "prop"))
        if role not in VALID_ROLES:
            problems.append(
                f"{where}.role: 必须是 {sorted(VALID_ROLES)} 之一，得到 '{role}'"
            )

        try:
            yaw = float(raw.get("yaw_deg", 0.0))
        except (TypeError, ValueError):
            problems.append(f"{where}.yaw_deg: 必须是数字")
            yaw = 0.0

        objects.append(
            SceneObject(
                id=obj_id,
                asset=asset,
                pos=pos,
                yaw_deg=yaw,
                scale=scale,
                physics=bool(raw.get("physics", False)),
                role=role,
            )
        )

    # ---- target ----
    target: Optional[Target] = None
    raw_target = data.get("target")
    if raw_target is not None:
        if not isinstance(raw_target, dict):
            problems.append("target: 必须是对象")
        else:
            tid = str(raw_target.get("object_id", "")).strip()
            if not tid:
                problems.append("target.object_id: 不能为空")
            elif tid not in seen_ids:
                problems.append(
                    f"target.object_id: '{tid}' 不在 objects 里（必须是某个物体的 id）"
                )
            radius = float(raw_target.get("landing_radius_m", 5.0))
            if radius <= 0:
                problems.append("target.landing_radius_m: 必须 > 0")
            target = Target(object_id=tid, landing_radius_m=radius)

    # ---- spawn ----
    spawn: Optional[Spawn] = None
    raw_spawn = data.get("spawn")
    if raw_spawn is not None:
        if not isinstance(raw_spawn, dict):
            problems.append("spawn: 必须是对象")
        else:
            spos = _as_vec3(raw_spawn.get("pos", [0, 0, 0]), "spawn.pos", problems)
            syaw = float(raw_spawn.get("yaw_deg", 0.0))
            spawn = Spawn(pos=spos, yaw_deg=syaw)

    # ---- requirements ----
    requirements: List[Dict[str, Any]] = []
    raw_reqs = data.get("requirements", [])
    if not isinstance(raw_reqs, list):
        problems.append("requirements: 必须是数组")
    else:
        for idx, r in enumerate(raw_reqs):
            if not isinstance(r, dict) or "kind" not in r:
                problems.append(f"requirements[{idx}]: 必须是含 kind 字段的对象")
                continue
            requirements.append(dict(r))

    if problems:
        raise SpecError(problems)

    return SceneSpec(
        version=version,
        name=name,
        seed=seed,
        origin=origin,
        objects=objects,
        target=target,
        spawn=spawn,
        requirements=requirements,
        instruction=str(data.get("instruction", "")),
    )


def load_spec(path: str | Path) -> SceneSpec:
    """从文件加载并校验规格书。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到规格书文件: {p}")
    with p.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise SpecError([f"JSON 解析失败: {exc}"]) from exc
    return parse_spec(data)


def check_assets(spec: SceneSpec, available: Sequence[str]) -> List[str]:
    """
    交叉检查规格书里的 asset 是否都在场景可用资产列表中。
    返回缺失的资产名列表（空列表表示全部可用）。
    """
    available_set = set(available)
    missing = sorted({o.asset for o in spec.objects if o.asset not in available_set})
    return missing


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="校验一份场景规格书")
    ap.add_argument("spec", help="规格书 JSON 路径")
    a = ap.parse_args()

    try:
        s = load_spec(a.spec)
    except SpecError as exc:
        print(exc)
        raise SystemExit(1)

    print(s.summary())
    print("\n资产使用统计:")
    for asset, count in s.asset_histogram().items():
        print(f"  {count:3d} x {asset}")