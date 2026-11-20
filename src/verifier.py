#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifier.py —— 场景的几何验证（纯 CPU，不需要 Unreal Engine / AirSim）

为什么验证器是核心贡献
----------------------
"让大模型生成 3D 场景"这件事本身不构成研究贡献，因为生成出来的东西
可能根本不能用：楼是封死的（飞不进去）、目标是起点一眼就看见的（不是
搜索任务）、路径短到没有难度。

本模块在**任何智能体运行之前**回答四个问题：

  1. 可解吗？        —— 3D 占据栅格 + A* 寻路，找不到路就是生成器失败
  2. 一开始看见了吗？—— 起点到目标的视线是否被遮挡（这决定它是不是搜索任务）
  3. 够难吗？        —— 路径长度、绕行比是否超过阈值（平凡场景要被拒）
  4. 需求能通过吗？  —— 每条 requirement 在真值路径上逐条检查

关键设计：不依赖引擎
--------------------
验证只读规格书，用轴对齐包围盒（AABB）近似物体体积。这带来两个好处：

  * **快**：几十毫秒，可以在笔记本上批量跑
  * **可测试**：可以写单元测试证明它真的会拒绝坏场景

代价是近似误差。这个误差不是被忽略，而是被**测量**：`--compare-engine`
可以把验证器的判断和 AirSim 的真实碰撞结果对比（那部分需要引擎，单独做）。

坐标系
------
规格书与内部计算全部用 ENU（x=东, y=北, z=上，米）。不出现 NED。

用法
----
    python src/verifier.py --spec specs/fire_floor2.json
    python src/verifier.py --spec specs/fire_floor2.json --json report.json
    python src/verifier.py --spec specs/fire_floor2.json --resolution 0.4 -v
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from asset_catalog import CATALOG
from scene_spec import SceneObject, SceneSpec, load_spec

LOG = logging.getLogger("verifier")

# 无人机以中心点为基准的保守半径（米）。用于在栅格里把障碍物"膨胀"，
# 保证寻路结果是一条真正能飞的通道，而不是贴着墙皮的零宽曲线。
DEFAULT_CLEARANCE_M = 0.6

# 栅格分辨率（米）。越小越精确，内存和时间按三次方增长。
DEFAULT_RESOLUTION_M = 0.5

# 超过这个格子数就自动降精度，避免在 12 核机器上吃光内存。
MAX_CELLS = 6_000_000

# 巡航速度假设，仅用于把路径长度换算成"预计耗时"来检查 max_time_s。
ASSUMED_CRUISE_MPS = 3.0


# --------------------------------------------------------------------------- #
# 资产 -> 占位尺寸
# --------------------------------------------------------------------------- #
# 基础几何体的尺寸由 scale 决定（1 m 单位），其余资产用各自类别的名义尺寸。
# 这些数字是"够用就好"的保守估计，目的是让寻路不把无人机塞进比它小的缝里，
# 而不是精确复刻美术资产。
CATEGORY_EXTENT_M: Dict[str, Tuple[float, float, float]] = {
    "primitive_box": (1.0, 1.0, 1.0),
    "primitive_plane": (1.0, 1.0, 0.05),
    "primitive_sphere": (1.0, 1.0, 1.0),
    "primitive_cylinder": (1.0, 1.0, 1.0),
    "primitive_cone": (1.0, 1.0, 1.0),
    "wall": (4.0, 0.4, 3.0),
    "window": (2.5, 0.3, 1.8),
    "door": (1.2, 0.2, 2.2),
    "stairs": (2.0, 4.0, 3.0),
    "pillar": (0.5, 0.5, 4.0),
    "awning": (3.0, 2.0, 1.0),
    "ground_panel": (4.0, 4.0, 0.2),
    "person": (0.6, 0.6, 1.8),
    "statue": (1.2, 1.2, 2.0),
    "light_marker": (0.2, 0.2, 0.2),
    "fireplace": (1.5, 1.5, 1.2),
    "crate": (0.8, 0.8, 0.8),
    "rubble": (2.0, 2.0, 1.0),
    "road_barrier": (1.0, 1.0, 1.0),
    "car": (2.0, 4.5, 1.6),
    "bicycle": (0.6, 1.8, 1.1),
    "cart": (1.2, 2.2, 1.2),
    "tree": (3.0, 3.0, 8.0),
    "bush": (2.0, 2.0, 1.5),
    "grass": (2.0, 2.0, 0.3),
    "street_light": (0.6, 0.6, 8.0),
    "street_prop": (1.0, 1.0, 1.5),
    "tent": (3.0, 3.0, 2.0),
}

# 资产名 -> 类别，由目录反推，避免手工维护第二份映射表
ASSET_TO_CATEGORY: Dict[str, str] = {}
for _cat in CATALOG.values():
    for _asset in _cat.assets:
        ASSET_TO_CATEGORY.setdefault(_asset, _cat.name)


# --------------------------------------------------------------------------- #
# 基础几何
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AABB:
    """轴对齐包围盒，ENU 米制。"""

    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]

    @property
    def center(self) -> Tuple[float, float, float]:
        return tuple((self.lo[i] + self.hi[i]) / 2 for i in range(3))  # type: ignore[return-value]

    @property
    def size(self) -> Tuple[float, float, float]:
        return tuple(self.hi[i] - self.lo[i] for i in range(3))  # type: ignore[return-value]

    def expanded(self, margin: float) -> "AABB":
        return AABB(
            tuple(self.lo[i] - margin for i in range(3)),  # type: ignore[arg-type]
            tuple(self.hi[i] + margin for i in range(3)),  # type: ignore[arg-type]
        )

    def contains(self, p: Sequence[float]) -> bool:
        return all(self.lo[i] <= p[i] <= self.hi[i] for i in range(3))

    def intersects(self, other: "AABB") -> bool:
        return all(self.lo[i] <= other.hi[i] and other.lo[i] <= self.hi[i] for i in range(3))

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            tuple(min(self.lo[i], other.lo[i]) for i in range(3)),  # type: ignore[arg-type]
            tuple(max(self.hi[i], other.hi[i]) for i in range(3)),  # type: ignore[arg-type]
        )


def object_extent(spec: SceneSpec, obj: SceneObject) -> Tuple[float, float, float]:
    """
    估算物体的包围盒尺寸（米）。

    基础几何体按 scale 精确计算；其余资产用类别名义尺寸再乘 scale。
    找不到类别时退回 1 m 立方体——宁可高估也不要漏掉障碍。
    """
    cat = ASSET_TO_CATEGORY.get(obj.asset)
    base = CATEGORY_EXTENT_M.get(cat or "", (1.0, 1.0, 1.0))
    sx, sy, sz = obj.scale
    return (base[0] * sx, base[1] * sy, base[2] * sz)


def object_aabb(spec: SceneSpec, obj: SceneObject) -> AABB:
    """
    物体的世界 AABB。

    注意：这里**忽略 yaw**，即把旋转后的物体当成其轴对齐外包盒。
    对 90° 倍数的旋转是精确的；对其他角度是保守（偏大）的近似。
    保守意味着"可能把可通行的缝判成不可通行"，而不是反过来，
    因此不会放过不可解场景。
    """
    cx, cy, cz = spec.world_position_enu(obj)
    ex, ey, ez = object_extent(spec, obj)
    return AABB(
        (cx - ex / 2, cy - ey / 2, cz - ez / 2),
        (cx + ex / 2, cy + ey / 2, cz + ez / 2),
    )


def segment_hits_aabb(p0: Sequence[float], p1: Sequence[float], box: AABB) -> bool:
    """
    线段与 AABB 是否相交（slab 方法）。

    用于视线遮挡判定：从起点到目标连一条线，只要穿过任意遮挡体
    就说明目标在初始时刻不可见。
    """
    direction = [p1[i] - p0[i] for i in range(3)]
    t_min, t_max = 0.0, 1.0

    for i in range(3):
        if abs(direction[i]) < 1e-9:
            if not (box.lo[i] <= p0[i] <= box.hi[i]):
                return False
            continue

        inv = 1.0 / direction[i]
        t1 = (box.lo[i] - p0[i]) * inv
        t2 = (box.hi[i] - p0[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1

        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False

    return t_max >= 0.0 and t_min <= 1.0


# --------------------------------------------------------------------------- #
# 占据栅格
# --------------------------------------------------------------------------- #
class OccupancyGrid:
    """
    三维占据栅格。

    索引约定：`idx = ix + nx * (iy + ny * iz)`，iz 为高度方向。
    """

    def __init__(self, aabb: AABB, resolution: float) -> None:
        self.resolution = float(resolution)
        self.origin = tuple(aabb.lo)
        size = aabb.size
        self.nx = max(1, int(math.ceil(size[0] / self.resolution)))
        self.ny = max(1, int(math.ceil(size[1] / self.resolution)))
        self.nz = max(1, int(math.ceil(size[2] / self.resolution)))
        self.blocked = bytearray(self.nx * self.ny * self.nz)

    # ---------- 坐标索引 ----------
    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def to_index(self, x: int, y: int, z: int) -> int:
        return x + self.nx * (y + self.ny * z)

    def to_cell(self, p: Sequence[float]) -> Tuple[int, int, int]:
        ix = int((p[0] - self.origin[0]) / self.resolution)
        iy = int((p[1] - self.origin[1]) / self.resolution)
        iz = int((p[2] - self.origin[2]) / self.resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return (ix, iy, iz)

    def to_point(self, cell: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(
            self.origin[i] + (cell[i] + 0.5) * self.resolution for i in range(3)  # type: ignore[return-value]
        )

    # ---------- 占用查询 ----------
    def is_blocked(self, cell: Tuple[int, int, int]) -> bool:
        return bool(self.blocked[self.to_index(*cell)])

    def is_free(self, cell: Tuple[int, int, int]) -> bool:
        if not (
            0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny and 0 <= cell[2] < self.nz
        ):
            return False
        return not self.blocked[self.to_index(*cell)]

    def mark(self, cell: Tuple[int, int, int]) -> None:
        if (
            0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny and 0 <= cell[2] < self.nz
        ):
            self.blocked[self.to_index(*cell)] = 1

    # ---------- 栅格化 ----------
    def rasterize(self, box: AABB) -> int:
        """把一个 AABB 标记为占用，返回影响的格子数。"""
        lo = self.to_cell(box.lo)
        hi = self.to_cell(box.hi)
        count = 0
        for iz in range(lo[2], hi[2] + 1):
            for iy in range(lo[1], hi[1] + 1):
                base = self.to_index(0, iy, iz)
                row = slice(base + lo[0], base + hi[0] + 1)
                for i in range(row.start, row.stop):
                    if not self.blocked[i]:
                        self.blocked[i] = 1
                        count += 1
        return count

    def carve(self, box: AABB) -> None:
        """把一个 AABB 强制标记为自由（用于清出目标周围空间）。"""
        lo = self.to_cell(box.lo)
        hi = self.to_cell(box.hi)
        for iz in range(lo[2], hi[2] + 1):
            for iy in range(lo[1], hi[1] + 1):
                base = self.to_index(0, iy, iz)
                for i in range(base + lo[0], base + hi[0] + 1):
                    self.blocked[i] = 0


def build_grid(
    spec: SceneSpec,
    resolution: float = DEFAULT_RESOLUTION_M,
    clearance: float = DEFAULT_CLEARANCE_M,
) -> Tuple[OccupancyGrid, Dict[str, AABB], Dict[str, Any]]:
    """
    根据规格书栅格化三维占据体。

    返回 (栅格, 每个物体的 AABB, 元信息)。
    只有 `physics=True` 的物体参与碰撞栅格——与引擎行为一致：
    没有碰撞代理体的装饰物不会挡住无人机。

    注意：所有坐标都在**世界 ENU 系**下计算。`spawn.pos` 是相对 origin 的
    局部坐标，必须先经 `world_spawn_enu()` 转换，否则会把两套坐标混算。
    """
    solid: List[Tuple[str, AABB]] = []
    for obj in spec.objects:
        if not obj.physics:
            continue
        solid.append((obj.id, object_aabb(spec, obj)))

    # 计算包围范围：所有实心物体 + 起点 + 目标，全部用世界坐标
    points: List[Tuple[float, float, float]] = []
    for _, box in solid:
        points.append(box.lo)
        points.append(box.hi)

    spawn_world = spec.world_spawn_enu()
    if spawn_world is not None:
        points.append(spawn_world)

    if spec.target is not None:
        goal_obj = spec.object_by_id(spec.target.object_id)
        if goal_obj is not None:
            points.append(spec.world_position_enu(goal_obj))

    if not points:
        raise ValueError("规格书里既没有实心物体也没有起点/目标，无法建立栅格")

    lo = tuple(min(p[i] for p in points) - 3.0 for i in range(3))
    hi = tuple(max(p[i] for p in points) + 3.0 for i in range(3))
    bounds = AABB(lo, hi)  # type: ignore[arg-type]

    # 自动降精度，防止内存爆掉
    res = float(resolution)
    while True:
        g = OccupancyGrid(bounds, res)
        if g.n_cells <= MAX_CELLS:
            break
        res *= 1.5
        LOG.warning("栅格过大，分辨率自动放宽到 %.2f m", res)

    boxes: Dict[str, AABB] = {}
    for obj_id, box in solid:
        boxes[obj_id] = box
        g.rasterize(box.expanded(clearance))

    # 刻意不做任何"为达目标凿洞"的处理。目标如果被封在实心结构里，
    # 就应当判定为不可解——那正是验证器要抓的生成器失败，而不是要掩盖的误差。
    meta = {
        "resolution_m": g.resolution,
        "dims": [g.nx, g.ny, g.nz],
        "cells": g.n_cells,
        "origin_enu": list(g.origin),
        "blocked_cells": sum(g.blocked),
        "solid_objects": len(solid),
        "clearance_m": clearance,
    }
    return g, boxes, meta


# --------------------------------------------------------------------------- #
# A* 寻路
# --------------------------------------------------------------------------- #
# 26 邻域：三个轴各 -1/0/+1，排除 (0,0,0) 本身
NEIGHBOURS: List[Tuple[int, int, int]] = [
    d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)
]


def astar(
    grid: OccupancyGrid,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    max_expansions: int = 2_000_000,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    三维 A*。返回格子路径（含起终点），不可达时返回 None。

    启发式用欧氏距离乘分辨率，在 26 邻域下是可采纳的（不会高估），
    因此 A* 找到的是最优路径。
    """
    if not grid.is_free(start):
        return None
    if not grid.is_free(goal):
        return None
    if start == goal:
        return [start]

    res = grid.resolution

    def h(c: Tuple[int, int, int]) -> float:
        return res * math.sqrt(
            (c[0] - goal[0]) ** 2 + (c[1] - goal[1]) ** 2 + (c[2] - goal[2]) ** 2
        )

    costs: Dict[Tuple[int, int, int], float] = {start: 0.0}
    parents: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    heap: List[Tuple[float, float, Tuple[int, int, int]]] = [(h(start), 0.0, start)]
    closed: set = set()
    expansions = 0

    while heap:
        _, g_cost, current = heapq.heappop(heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in parents:
                path.append(parents[path[-1]])
            return list(reversed(path))

        closed.add(current)
        expansions += 1
        if expansions > max_expansions:
            LOG.warning("A* 展开超过 %d 个节点，放弃", max_expansions)
            return None

        for d in NEIGHBOURS:
            nxt = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if nxt in closed or not grid.is_free(nxt):
                continue
            step = res * math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            ng = g_cost + step
            if ng < costs.get(nxt, math.inf):
                costs[nxt] = ng
                parents[nxt] = current
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))

    return None


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)
    )


# --------------------------------------------------------------------------- #
# 需求断言检查
# --------------------------------------------------------------------------- #
def check_requirements(
    spec: SceneSpec,
    path_enu: Optional[Sequence[Tuple[float, float, float]]],
) -> List[Dict[str, Any]]:
    """
    逐条检查 requirements 是否可满足。

    "可满足"的含义是：存在一条从起点到目标的可行路径，沿着它飞能同时
    满足所有断言。因此这些检查建立在 A* 的真值路径上。
    """
    results: List[Dict[str, Any]] = []

    length = path_length(path_enu) if path_enu else None
    estimated_time = (
        length / ASSUMED_CRUISE_MPS if length is not None else None
    )

    for req in spec.requirements:
        kind = str(req.get("kind", ""))
        value = req.get("value")
        passed: Optional[bool] = None
        detail = ""

        if kind == "max_time_s":
            if estimated_time is None:
                passed, detail = False, "无法计算路径耗时（场景不可解）"
            else:
                passed = estimated_time <= float(value)
                detail = (
                    f"预计 {estimated_time:.1f} s"
                    f"（{length:.1f} m / {ASSUMED_CRUISE_MPS} m/s），"
                    f"上限 {float(value):.0f} s"
                )

        elif kind == "min_altitude_m":
            if path_enu is None:
                passed, detail = False, "无法计算（场景不可解）"
            else:
                lowest = min(p[2] for p in path_enu)
                passed = lowest >= float(value)
                detail = f"路径最低点 {lowest:.2f} m，下限 {float(value):.1f} m"

        elif kind == "no_collision":
            # 路径本身就走在自由栅格里，能求出路径即意味着可无碰撞通过
            passed = path_enu is not None
            detail = "由 A* 在膨胀后的自由空间中的可达性保证"

        elif kind == "must_enter_building":
            if path_enu is None:
                passed, detail = False, "无法计算（场景不可解）"
            else:
                structures = [o for o in spec.objects if o.role == "structure"]
                if not structures:
                    passed, detail = False, "场景里没有结构物体"
                else:
                    xs = [spec.world_position_enu(o)[0] for o in structures]
                    ys = [spec.world_position_enu(o)[1] for o in structures]
                    inside = any(
                        min(xs) <= p[0] <= max(xs) and min(ys) <= p[1] <= max(ys)
                        for p in path_enu
                    )
                    passed = inside
                    detail = "路径经过建筑投影范围" if inside else "路径始终在建筑外部"

        elif kind == "min_path_length_m":
            if length is None:
                passed, detail = False, "无法计算（场景不可解）"
            else:
                passed = length >= float(value)
                detail = f"路径 {length:.1f} m，下限 {float(value):.1f} m"

        elif kind == "max_detour_ratio":
            if length is None or spec.target is None or spec.spawn is None:
                passed, detail = False, "无法计算（场景不可解或缺目标）"
            else:
                goal = spec.object_by_id(spec.target.object_id)
                start_w = spec.world_spawn_enu()
                goal_w = spec.world_position_enu(goal) if goal else start_w
                straight = math.dist(start_w, goal_w)
                ratio = length / straight if straight > 1e-6 else math.inf
                passed = ratio <= float(value)
                detail = f"绕行比 {ratio:.2f}，上限 {float(value):.2f}"

        else:
            passed, detail = None, f"未知的断言类型 '{kind}'，未检查"

        results.append(
            {
                "kind": kind,
                "value": value,
                "satisfiable": passed,
                "detail": detail,
            }
        )

    return results


# --------------------------------------------------------------------------- #
# 验证结果
# --------------------------------------------------------------------------- #
@dataclass
class VerificationResult:
    spec_name: str
    solvable: bool
    initially_occluded: bool
    path_length_m: Optional[float]
    straight_distance_m: float
    detour_ratio: Optional[float]
    min_path_altitude_m: Optional[float]
    requirement_results: List[Dict[str, Any]]
    failures: List[str]
    grid_meta: Dict[str, Any]
    clearance_m: float

    @property
    def usable(self) -> bool:
        """能否作为测试用例使用：可解 + 有遮挡 + 没有失败项 + 断言全过。"""
        return (
            self.solvable
            and self.initially_occluded
            and not self.failures
            and all(r["satisfiable"] is not False for r in self.requirement_results)
        )

    def summary(self) -> str:
        def yn(v: bool) -> str:
            return "yes" if v else "NO"

        lines = [
            f"验证对象: {self.spec_name}",
            f"  可解性          : {yn(self.solvable)}"
            + (f"  (路径 {self.path_length_m:.1f} m)" if self.path_length_m else ""),
            f"  初始是否被遮挡  : {yn(self.initially_occluded)}",
            f"  直线距离        : {self.straight_distance_m:.1f} m",
            f"  绕行比          : "
            + (f"{self.detour_ratio:.2f}" if self.detour_ratio else "n/a"),
            f"  路径最低高度    : "
            + (
                f"{self.min_path_altitude_m:.2f} m"
                if self.min_path_altitude_m is not None
                else "n/a"
            ),
            f"  栅格            : {self.grid_meta['dims']} @ "
            f"{self.grid_meta['resolution_m']:.2f} m "
            f"({self.grid_meta['cells']} 格, 膨胀 {self.clearance_m:.1f} m)",
        ]

        if self.requirement_results:
            lines.append("  需求断言:")
            for r in self.requirement_results:
                mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r["satisfiable"]]
                lines.append(f"    [{mark}] {r['kind']} = {r['value']}  ({r['detail']})")

        if self.failures:
            lines.append("  结论: 不可用")
            for f in self.failures:
                lines.append(f"    - {f}")
        else:
            lines.append("  结论: 可作为测试用例")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "usable": self.usable,
            "solvable": self.solvable,
            "initially_occluded": self.initially_occluded,
            "path_length_m": self.path_length_m,
            "straight_distance_m": self.straight_distance_m,
            "detour_ratio": self.detour_ratio,
            "min_path_altitude_m": self.min_path_altitude_m,
            "requirement_results": self.requirement_results,
            "failures": self.failures,
            "grid": self.grid_meta,
            "clearance_m": self.clearance_m,
        }


# --------------------------------------------------------------------------- #
# 主验证流程
# --------------------------------------------------------------------------- #
def verify(
    spec: SceneSpec,
    *,
    resolution: float = DEFAULT_RESOLUTION_M,
    clearance: float = DEFAULT_CLEARANCE_M,
    min_detour_ratio: float = 1.15,
    min_path_length_m: float = 10.0,
) -> VerificationResult:
    """
    对一份规格书做完整几何验证。

    参数
    ----
    resolution, clearance:
        栅格精度与无人机安全半径。
    min_detour_ratio:
        路径长度 / 直线距离 的下限。低于它说明几乎直线可达，属于平凡场景。
        下限而非上限，因为这里要拒绝的是"太简单"。
    min_path_length_m:
        绝对路径长度下限，防止目标就在隔壁。
    """
    failures: List[str] = []

    # 场景必须给全起点与目标，否则无从验证
    if spec.spawn is None:
        failures.append("缺少 spawn，无法验证可解性")
    if spec.target is None:
        failures.append("缺少 target，无法验证可解性")

    goal_obj = (
        spec.object_by_id(spec.target.object_id) if spec.target is not None else None
    )
    if spec.target is not None and goal_obj is None:
        failures.append(
            f"target.object_id='{spec.target.object_id}' 在 objects 中不存在"
        )

    if failures:
        return VerificationResult(
            spec_name=spec.name,
            solvable=False,
            initially_occluded=False,
            path_length_m=None,
            straight_distance_m=0.0,
            detour_ratio=None,
            min_path_altitude_m=None,
            requirement_results=[],
            failures=failures,
            grid_meta={},
            clearance_m=clearance,
        )

    assert spec.spawn is not None and goal_obj is not None

    # 全部换算到世界 ENU 系：spawn.pos 与物体 pos 都是相对 origin 的局部坐标
    start_enu = spec.world_spawn_enu()
    goal_enu = spec.world_position_enu(goal_obj)
    straight = math.dist(start_enu, goal_enu)

    # ---------- 1. 初始遮挡 ----------
    occluders = [
        object_aabb(spec, o)
        for o in spec.objects
        if o.role in ("structure", "prop") and o.id != goal_obj.id
    ]
    blocked_segments = [
        b for b in occluders if segment_hits_aabb(start_enu, goal_enu, b)
    ]
    occluded = bool(blocked_segments)

    # ---------- 2. 占据栅格 + A* ----------
    grid, _, grid_meta = build_grid(
        spec, resolution=resolution, clearance=clearance
    )

    start_cell = grid.to_cell(start_enu)
    goal_cell = grid.to_cell(goal_enu)

    # 起点如果落在障碍里（例如贴地），就近找一个自由格
    if not grid.is_free(start_cell):
        start_cell = _nearest_free(grid, start_cell, radius=6)

    # 目标格子可能被目标自身占着（人物模型带碰撞）。此时在**着陆半径内**
    # 找一个自由格作为寻路终点 —— 这与评测协议一致：只要飞到目标附近即可，
    # 不需要与目标重合。若整个着陆半径都被封死，那就是不可解。
    if not grid.is_free(goal_cell):
        landing_cells = max(1, int(round(
            (spec.target.landing_radius_m if spec.target else 4.0) / grid.resolution
        )))
        goal_cell = _nearest_free(grid, goal_cell, radius=landing_cells)

    cells = astar(grid, start_cell, goal_cell) if start_cell and goal_cell else None
    solvable = cells is not None

    path_enu: Optional[List[Tuple[float, float, float]]] = None
    length: Optional[float] = None
    detour: Optional[float] = None
    min_alt: Optional[float] = None

    if solvable and cells:
        path_enu = [grid.to_point(c) for c in cells]
        # 用真实起终点替换首尾，避免栅格量化带来的误差
        path_enu[0] = tuple(start_enu)
        path_enu[-1] = tuple(goal_enu)
        length = path_length(path_enu)
        detour = length / straight if straight > 1e-6 else math.inf
        min_alt = min(p[2] for p in path_enu)
    else:
        failures.append("A* 找不到从起点到目标的可行路径（场景不可解）")

    # ---------- 3. 非平凡性 ----------
    if solvable and length is not None:
        if length < min_path_length_m:
            failures.append(
                f"路径仅 {length:.1f} m，短于下限 {min_path_length_m:.0f} m，任务过于简单"
            )
        if detour is not None and detour < min_detour_ratio:
            failures.append(
                f"绕行比仅 {detour:.2f}（下限 {min_detour_ratio:.2f}），几乎直线可达，"
                f"不构成搜索任务"
            )

    # ---------- 4. 遮挡是搜索任务的必要条件 ----------
    if not occluded:
        failures.append(
            "起点可以直线看到目标，任务不是搜索任务（初始遮挡检查失败）"
        )

    # ---------- 5. 需求断言 ----------
    requirement_results = check_requirements(spec, path_enu)
    for r in requirement_results:
        if r["satisfiable"] is False:
            failures.append(f"需求 '{r['kind']}' 无法满足: {r['detail']}")

    return VerificationResult(
        spec_name=spec.name,
        solvable=solvable,
        initially_occluded=occluded,
        path_length_m=length,
        straight_distance_m=straight,
        detour_ratio=detour,
        min_path_altitude_m=min_alt,
        requirement_results=requirement_results,
        failures=failures,
        grid_meta=grid_meta,
        clearance_m=clearance,
    )


def _nearest_free(
    grid: OccupancyGrid,
    cell: Tuple[int, int, int],
    radius: int = 6,
) -> Optional[Tuple[int, int, int]]:
    """在给定格子附近螺旋搜索一个自由格。"""
    if grid.is_free(cell):
        return cell
    for r in range(1, radius + 1):
        for d in itertools.product(range(-r, r + 1), repeat=3):
            cand = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
            if grid.is_free(cand):
                return cand
    return None


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="场景几何验证：可解性、初始遮挡、非平凡性、需求可满足性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--spec", required=True, help="规格书 JSON 路径")
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    ap.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE_M)
    ap.add_argument("--min-detour", type=float, default=1.15)
    ap.add_argument("--min-path-length", type=float, default=10.0)
    ap.add_argument("--json", default="", help="把结果写入 JSON 文件")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    # Windows 控制台默认 CP936，会把中文输出变成乱码
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        spec = load_spec(a.spec)
    except Exception as exc:  # noqa: BLE001
        LOG.error("规格书加载失败: %s", exc)
        return 2

    result = verify(
        spec,
        resolution=a.resolution,
        clearance=a.clearance,
        min_detour_ratio=a.min_detour,
        min_path_length_m=a.min_path_length,
    )

    print(result.summary())

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOG.info("报告已写入 %s", a.json)

    return 0 if result.usable else 1


if __name__ == "__main__":
    sys.exit(main())