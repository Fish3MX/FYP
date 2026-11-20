#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asset_catalog.py —— 场景可用的语义资产目录

为什么需要这一层
----------------
`simListAssets()` 返回 487 个名字，其中大量是招牌、邮筒、下水道盖这类
对测试场景没有意义的装饰物。如果把这些原样丢给大模型，会产生两个问题：

  1. **信噪比太低**：模型要在 487 个字符串里挑，容易挑错或拼错；
  2. **语义失配**：模型看到 `SM_signs_poster_smoothies` 并不知道它多大、
     能不能当墙用。

所以本模块把原始资产名收敛成**带语义的类别**：每个类别给出
  - 一组可互换的具体资产名（实际生成时随机或顺序取用）
  - 该类别默认的物理属性（结构必须带碰撞，装饰物不需要）
  - 该类别在规格书里的 role（structure / prop / target / hazard）
  - 大致的真实尺寸，供模型估算排版

规格书里只允许写本目录里出现过的资产名。这条约束在
`llm_compiler.py` 里强制执行，也被 `scene_builder.py` 的资产预检兜底。

单位说明
--------
所有长度单位为米。基础几何体（Cube / Plane / Sphere / Cylinder / Cone）
在 UE 默认工程中边长/直径约 1 m，因此 scale=1 表示 1 米；其他资产按
其原始建模尺寸，scale 只做整体缩放。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class AssetCategory:
    """一个语义类别及其可用资产。"""

    name: str
    assets: Sequence[str]
    role: str = "prop"          # structure / prop / target / hazard
    physics: bool = False       # 是否默认带碰撞
    unit_size_m: str = "n/a"    # 大致尺寸描述，给模型做排版参考
    note: str = ""              # 类别用途说明

    def to_prompt_line(self) -> str:
        assets = ", ".join(self.assets)
        bits = [f"`{self.name}`", f"role={self.role}"]
        if self.physics:
            bits.append("physics=true")
        if self.unit_size_m != "n/a":
            bits.append(f"size≈{self.unit_size_m}")
        head = " | ".join(bits)
        line = f"- {head}\n    assets: {assets}"
        if self.note:
            line += f"\n    usage: {self.note}"
        return line


# --------------------------------------------------------------------------- #
# 目录本身
# --------------------------------------------------------------------------- #
# 顺序按"搭场景时的常用程度"排，越靠前越常用。
CATALOG: Dict[str, AssetCategory] = {
    # ============ 基础几何体：最可靠，永远能生成 ============
    "primitive_box": AssetCategory(
        name="primitive_box",
        assets=["Cube"],
        role="structure",
        physics=True,
        unit_size_m="1 m 立方体，靠 scale 拉伸",
        note="搭墙体、楼板、台阶、障碍物的首选。scale=[长,宽,高] 以米为单位。",
    ),
    "primitive_plane": AssetCategory(
        name="primitive_plane",
        assets=["Plane"],
        role="structure",
        physics=True,
        unit_size_m="1 m 平板",
        note="地面、屋顶、薄板。",
    ),
    "primitive_sphere": AssetCategory(
        name="primitive_sphere",
        assets=["Sphere"],
        role="hazard",
        physics=False,
        unit_size_m="1 m 直径",
        note="做标记点、火源示意、目标占位。",
    ),
    "primitive_cylinder": AssetCategory(
        name="primitive_cylinder",
        assets=["Cylinder"],
        role="structure",
        physics=True,
        unit_size_m="1 m 直径 x 1 m 高",
        note="柱子、管道、圆形障碍。",
    ),
    "primitive_cone": AssetCategory(
        name="primitive_cone",
        assets=["Cone"],
        role="hazard",
        physics=True,
        unit_size_m="1 m 高",
        note="路锥、警示标记。",
    ),

    # ============ 建筑结构：模块化构件 ============
    "wall": AssetCategory(
        name="wall",
        assets=[
            "SM_build_e_mod_wall_a",
            "SM_build_cafe_a_mod_wall_a",
            "SM_build_cafe_a_mod_wall_c",
            "SM_build_b_mod_lvl1_storefront_b_wall5m",
            "SM_build_b_mod_lvl1_storefront_b_wall3m",
        ],
        role="structure",
        physics=True,
        unit_size_m="3~5 m 宽，一层高",
        note="整段墙体。需要精确尺寸控制时优先用 primitive_box。",
    ),
    "window": AssetCategory(
        name="window",
        assets=[
            "SM_build_b_mod_lvl2_doublewindow",
            "SM_build_b_mod_lvl2_singlewindow",
            "SM_build_b_mod_lvl2_widewindow",
            "SM_build_b_mod_lvl3_singlewindow",
            "SM_build_b_mod_lvl3_doublewindow",
            "SM_build_e_mod_lvl1_window_a",
            "SM_build_e_mod_wall_lvl2_window",
            "SM_build_d_mod_wall_lvl2_window",
            "SM_build_d_mod_lvl1_window",
        ],
        role="prop",
        physics=False,
        unit_size_m="2~4 m 宽",
        note="贴在墙面上做视觉细节与视线遮挡。lvl1/lvl2/lvl3 对应楼层。",
    ),
    "door": AssetCategory(
        name="door",
        assets=[
            "SM_build_b_door_wood_single",
            "SM_build_b_door_wood_double",
            "SM_build_e_mod_lvl1_door_a",
            "SM_build_e_mod_lvl1_door_b",
            "SM_build_f_mod_lvl1_door_a",
            "SM_Door_02",
        ],
        role="prop",
        physics=False,
        unit_size_m="1~2 m 宽",
        note="入口。注意：门本身不挡路，墙体要留出缺口才飞得进去。",
    ),
    "stairs": AssetCategory(
        name="stairs",
        assets=["SM_stairs_with_loops"],
        role="prop",
        physics=True,
        unit_size_m="一层楼梯",
        note="楼层之间。若担心尺寸不确定，可用 primitive_box 叠出阶梯。",
    ),
    "pillar": AssetCategory(
        name="pillar",
        assets=[
            "SM_build_b_mod_lvl1_pillar_a",
            "SM_build_b_mod_lvl1_pillar_b",
            "SM_build_b_mod_lvl1_pillar_c",
            "SM_build_d_pillar_a",
            "SM_build_d_pillar_b",
            "SM_pillar",
        ],
        role="structure",
        physics=True,
        unit_size_m="0.4 m 见方，3~6 m 高",
        note="承重柱、廊柱，也可做飞行障碍。",
    ),
    "awning": AssetCategory(
        name="awning",
        assets=[
            "SM_building_awning_a",
            "SM_building_awning_b",
            "SM_building_awning_c",
            "SM_building_a_awning_2_5m",
            "SM_building_a_awning_5m",
        ],
        role="prop",
        physics=False,
        unit_size_m="2.5~5 m 宽",
        note="雨棚，做建筑立面细节。",
    ),

    # ============ 地面 ============
    "ground_panel": AssetCategory(
        name="ground_panel",
        assets=[
            "SM_ground_concrete_platform",
            "SM_ground_mod_walkway_a_concrete",
            "SM_ground_mod_walkway_b_concrete",
            "SM_ground_mod_walkway_c_concrete",
            "SM_ground_park_walkways",
        ],
        role="structure",
        physics=True,
        unit_size_m="数米见方",
        note="铺装地面。大范围地面用 primitive_plane 更省事。",
    ),

    # ============ 目标：人物 ============
    "person": AssetCategory(
        name="person",
        assets=[
            "BP_ma_construction_worker_01",
            "BP_ma_construction_worker_02_b",
            "BP_ma_construction_side_overseer_01",
            "BP_fe_casual_02_d",
            "BP_fe_clerk_01",
            "BP_chef_01",
            "BP_ma_waiter_02",
            "BP_Character",
        ],
        role="target",
        physics=True,
        unit_size_m="1.7~1.8 m 高",
        note="被搜索的人。role=target 时必须指定到 target.object_id。",
    ),
    "statue": AssetCategory(
        name="statue",
        assets=[
            "SM_Statue01",
            "SM_Statue05",
            "SM_Statue06",
            "SM_prop_statue_eagle",
            "SM_prop_statue_deer",
            "SM_FountainStatue",
        ],
        role="target",
        physics=True,
        unit_size_m="1~2 m",
        note="人物类资产的静态替补，生成失败时可用。",
    ),

    # ============ 危险源 ============
    "light_marker": AssetCategory(
        name="light_marker",
        assets=["PointLightBP", "SpotLightBP"],
        role="hazard",
        physics=False,
        unit_size_m="点/聚光源",
        note="做火光的视觉示意。这是当前资产库里最接近'着火'效果的东西。",
    ),
    "fireplace": AssetCategory(
        name="fireplace",
        assets=["SM_Fireplace_01a", "SM_CookingStove_01"],
        role="hazard",
        physics=True,
        unit_size_m="1~2 m",
        note="有火焰造型的静态模型，可当作火源实体。",
    ),

    # ============ 障碍与杂物 ============
    "crate": AssetCategory(
        name="crate",
        assets=[
            "SM_WoodenCrate_01",
            "SM_WoodenCrate_02",
            "SM_WoodenCrate_03",
            "SM_Wood_Barrel_01",
            "SM_MetalBox",
            "SM_WaterBox",
        ],
        role="prop",
        physics=True,
        unit_size_m="0.5~1 m",
        note="室内外杂物，做通行障碍与遮挡。",
    ),
    "rubble": AssetCategory(
        name="rubble",
        assets=[
            "SM_Rubble01",
            "SM_Rubble02",
            "SM_Concrete_01",
            "SM_Concrete_02",
            "SM_Concrete_03",
            "SM_TrackLoaderRubble",
        ],
        role="prop",
        physics=True,
        unit_size_m="1~3 m",
        note="坍塌后的废墟堆，适合灾害场景。",
    ),
    "road_barrier": AssetCategory(
        name="road_barrier",
        assets=[
            "SM_Road_Barrier_01",
            "SM_Road_Barrier_02",
            "SM_RoadBarriers_03",
            "SM_WetFloorCone",
        ],
        role="prop",
        physics=True,
        unit_size_m="1~2 m",
        note="路障、锥桶，做地面通行限制。",
    ),

    # ============ 载具 ============
    "car": AssetCategory(
        name="car",
        assets=[
            "SM_Sedan03_Main",
            "SM_SUV_01a",
            "SM_MERGED_SK_NYC_Car01",
        ],
        role="prop",
        physics=True,
        unit_size_m="4~5 m 长",
        note="街道载具。可作为搜救目标或遮挡物。",
    ),
    "bicycle": AssetCategory(
        name="bicycle",
        assets=["SM_Bicycle_01", "SM_Bicycle_02"],
        role="prop",
        physics=True,
        unit_size_m="1.8 m 长",
        note="小型街道道具。",
    ),
    "cart": AssetCategory(
        name="cart",
        assets=["SM_Cart_v1_low", "SM_food_cart"],
        role="prop",
        physics=True,
        unit_size_m="1.5~2.5 m",
        note="推车、餐车，街边障碍。",
    ),

    # ============ 植被 ============
    "tree": AssetCategory(
        name="tree",
        assets=[
            "SM_tree_pine_a",
            "SM_tree_pine_b",
            "SM_tree_pine_c",
            "sm_tree_oak_a",
            "SM_tree_narrowleaf_a",
            "SM_tree_narrowleaf_b",
            "SM_tree_narrowleaf_c",
        ],
        role="prop",
        physics=True,
        unit_size_m="5~12 m 高",
        note="乔木。冠层可作为空中遮挡，影响视线与航线。",
    ),
    "bush": AssetCategory(
        name="bush",
        assets=[
            "SM_Tall_Bush_01",
            "SM_Tall_Bush_02",
            "SM_Tall_Bush_03",
            "SM_bush_a",
            "SM_bush_b",
        ],
        role="prop",
        physics=True,
        unit_size_m="1~3 m",
        note="灌木，地面遮挡。",
    ),
    "grass": AssetCategory(
        name="grass",
        assets=[
            "SM_grass_patch_a",
            "SM_grass_patch_b",
            "SM_grass_patch_c",
            "SM_grass_patch_a_large",
            "SM_flowers_flowers_b",
        ],
        role="prop",
        physics=False,
        unit_size_m="1~3 m",
        note="纯装饰，不影响飞行。",
    ),

    # ============ 街道设施 ============
    "street_light": AssetCategory(
        name="street_light",
        assets=[
            "SM_light_streetlight",
            "SM_light_streetlight_complete",
            "SM_lightpost_light_post_a",
            "SM_lightpost_light_post_b",
        ],
        role="prop",
        physics=True,
        unit_size_m="6~9 m 高",
        note="路灯。高度与无人机飞行高度重叠，是天然障碍。",
    ),
    "street_prop": AssetCategory(
        name="street_prop",
        assets=[
            "SM_bench_wood_a",
            "SM_bench_wood_b",
            "SM_prop_recycle_bin",
            "SM_prop_hydrant_trim",
            "SM_Street_Clock",
            "SM_Busstop_Pole",
            "SM_Mailbox_01",
            "SM_Mailbox_03",
            "SM_TelephoneCabin",
            "SM_VendingMachine",
            "SM_prop_drinking_fountain",
            "SM_NewsPaperStand",
        ],
        role="prop",
        physics=True,
        unit_size_m="1~3 m",
        note="长椅、垃圾桶、消防栓、公交站牌、电话亭等街道细节。",
    ),

    # ============ 户外 / 临时营地 ============
    "tent": AssetCategory(
        name="tent",
        assets=["SM_Tent_01b", "SM_PicnicTable_01", "SM_LawnChair_01", "SM_TableCloth_01a"],
        role="prop",
        physics=True,
        unit_size_m="2~3 m",
        note="户外营地，适合野外搜救场景。",
    ),
}


# --------------------------------------------------------------------------- #
# 查询接口
# --------------------------------------------------------------------------- #
def category_names() -> List[str]:
    return list(CATALOG.keys())


def get(name: str) -> AssetCategory:
    if name not in CATALOG:
        raise KeyError(
            f"未知资产类别 '{name}'。可用类别: {sorted(CATALOG)}"
        )
    return CATALOG[name]


def all_assets() -> List[str]:
    """目录中出现的所有具体资产名（去重并保持顺序）。"""
    seen: Dict[str, None] = {}
    for cat in CATALOG.values():
        for a in cat.assets:
            seen.setdefault(a, None)
    return list(seen.keys())


def whitelist() -> frozenset:
    """允许出现在规格书里的资产名集合。"""
    return frozenset(all_assets())


def asset_role(asset: str) -> Optional[str]:
    for cat in CATALOG.values():
        if asset in cat.assets:
            return cat.role
    return None


def asset_default_physics(asset: str) -> bool:
    for cat in CATALOG.values():
        if asset in cat.assets:
            return cat.physics
    return False


def categories_for_role(role: str) -> List[str]:
    return [c.name for c in CATALOG.values() if c.role == role]


def prompt_text(include_usage: bool = True) -> str:
    """
    生成给大模型看的资产目录文本。

    刻意保持紧凑：类别名 + role + 尺寸 + 资产名，不用 JSON，
    因为自然语言列表对模型的约束效果更好，也更容易被引用。
    """
    lines = ["Available asset categories (use ONLY these names):"]
    for cat in CATALOG.values():
        if not include_usage:
            lines.append(
                f"- `{cat.name}` (role={cat.role}"
                f"{', physics=true' if cat.physics else ''}"
                f"{f', size≈{cat.unit_size_m}' if cat.unit_size_m != 'n/a' else ''})"
                f" -> {', '.join(cat.assets)}"
            )
        else:
            lines.append(cat.to_prompt_line())
    return "\n".join(lines)


def compact_table() -> str:
    """极简表格版，用于在终端打印核对。"""
    rows = []
    for cat in CATALOG.values():
        rows.append(
            f"{cat.name:<18} role={cat.role:<10} physics={str(cat.physics):<5} "
            f"n={len(cat.assets):<3} {cat.unit_size_m}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="查看资产目录")
    ap.add_argument("--prompt", action="store_true", help="打印给大模型看的目录文本")
    ap.add_argument("--assets", action="store_true", help="打印去重后的资产名列表")
    ap.add_argument("--check", nargs="*", metavar="ASSET", help="检查资产是否在目录内")
    a = ap.parse_args()

    if a.prompt:
        print(prompt_text())
    elif a.assets:
        for name in all_assets():
            print(name)
    elif a.check is not None:
        wl = whitelist()
        for name in a.check:
            print(f"{'OK   ' if name in wl else 'MISS '} {name}")
    else:
        print(f"共 {len(CATALOG)} 个类别，{len(all_assets())} 个资产\n")
        print(compact_table())