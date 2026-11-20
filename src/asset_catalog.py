#!/usr/bin/env python3
"""Semantic asset catalogue: the vocabulary a scenario generator may use.

Why this layer exists
---------------------
`simListAssets()` returns around 487 names, most of them street dressing
(posters, mailboxes, sewer caps) that is useless for a test scenario. Passing
that raw list to a language model causes two problems:

  1. **Signal-to-noise**: the model has to pick from 487 opaque strings and
     frequently misspells or invents one.
  2. **No semantics**: a name like `SM_signs_poster_smoothies` says nothing
     about how large the object is or whether it can serve as a wall.

This module collapses the raw names into semantic categories. Each category
provides:

  - a set of interchangeable asset names (instantiated in order or at random)
  - the default physics flag (structure must collide, decoration need not)
  - the role the category should carry in a specification
  - an approximate real-world size, so the model can lay things out sensibly

Only assets listed here may appear in a specification. That constraint is
enforced by `llm_compiler`, and backed up by the asset pre-check in
`scene_builder`.

Units
-----
All lengths are metres. The base primitives (Cube / Plane / Sphere / Cylinder /
Cone) are about 1 m across at scale 1, so scale is effectively a size in metres
for those. Other assets keep their modelled size and scale only multiplies it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class AssetCategory:
    """A semantic category and the concrete assets that can stand in for it."""

    name: str
    assets: Sequence[str]
    role: str = "prop"          # structure / prop / target / hazard
    physics: bool = False       # collides by default
    unit_size_m: str = "n/a"    # approximate size, for layout reasoning
    note: str = ""              # what the category is for

    def to_prompt_line(self) -> str:
        assets = ", ".join(self.assets)
        bits = [f"`{self.name}`", f"role={self.role}"]
        if self.physics:
            bits.append("physics=true")
        if self.unit_size_m != "n/a":
            bits.append(f"size~{self.unit_size_m}")
        line = f"- {' | '.join(bits)}\n    assets: {assets}"
        if self.note:
            line += f"\n    usage: {self.note}"
        return line


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #
# Ordered by how often a scenario builder needs the category.
CATALOG: Dict[str, AssetCategory] = {
    # ======== primitives: always available, exact dimensions ========
    "primitive_box": AssetCategory(
        name="primitive_box",
        assets=["Cube"],
        role="structure",
        physics=True,
        unit_size_m="1 m cube, stretched by scale",
        note="First choice for walls, floors and obstacles. scale=[length,width,height] in metres.",
    ),
    "primitive_plane": AssetCategory(
        name="primitive_plane",
        assets=["Plane"],
        role="structure",
        physics=True,
        unit_size_m="1 m quad",
        note="Ground, roof, thin slabs.",
    ),
    "primitive_sphere": AssetCategory(
        name="primitive_sphere",
        assets=["Sphere"],
        role="hazard",
        physics=False,
        unit_size_m="1 m diameter",
        note="Marker point, fire indication, target placeholder.",
    ),
    "primitive_cylinder": AssetCategory(
        name="primitive_cylinder",
        assets=["Cylinder"],
        role="structure",
        physics=True,
        unit_size_m="1 m diameter x 1 m tall",
        note="Columns, pipes, round obstacles.",
    ),
    "primitive_cone": AssetCategory(
        name="primitive_cone",
        assets=["Cone"],
        role="hazard",
        physics=True,
        unit_size_m="1 m tall",
        note="Traffic cone, warning marker.",
    ),

    # ======== building structure: modular kit ========
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
        unit_size_m="3-5 m wide, one storey tall",
        note="Full wall segments. Prefer primitive_box when exact dimensions matter.",
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
        unit_size_m="2-4 m wide",
        note="Mounted on a wall for visual detail and occlusion. lvl1/lvl2/lvl3 select the storey.",
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
        unit_size_m="1-2 m wide",
        note="Entrance. The door itself does not block flight; the wall must have a real gap.",
    ),
    "stairs": AssetCategory(
        name="stairs",
        assets=["SM_stairs_with_loops"],
        role="prop",
        physics=True,
        unit_size_m="one storey of steps",
        note="Between floors. If the size is uncertain, stack primitive_box steps instead.",
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
        unit_size_m="0.4 m square, 3-6 m tall",
        note="Load-bearing columns, also useful as flight obstacles.",
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
        unit_size_m="2.5-5 m wide",
        note="Facade detail.",
    ),

    # ======== ground ========
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
        unit_size_m="a few metres square",
        note="Paving. For large areas, primitive_plane is cheaper.",
    ),

    # ======== targets: people ========
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
        unit_size_m="1.7-1.8 m tall",
        note="The person to be found. role=target must be referenced by target.object_id.",
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
        unit_size_m="1-2 m",
        note="Static fallback if a character asset cannot be spawned.",
    ),

    # ======== hazards ========
    "light_marker": AssetCategory(
        name="light_marker",
        assets=["PointLightBP", "SpotLightBP"],
        role="hazard",
        physics=False,
        unit_size_m="point / spot light",
        note="Visual stand-in for fire. The closest thing to flames in this asset set.",
    ),
    "fireplace": AssetCategory(
        name="fireplace",
        assets=["SM_Fireplace_01a", "SM_CookingStove_01"],
        role="hazard",
        physics=True,
        unit_size_m="1-2 m",
        note="Static model with a flame shape, usable as a fire source.",
    ),

    # ======== clutter and obstacles ========
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
        unit_size_m="0.5-1 m",
        note="Indoor and outdoor clutter; blocks movement and sight.",
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
        unit_size_m="1-3 m",
        note="Collapse debris, suited to disaster scenarios.",
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
        unit_size_m="1-2 m",
        note="Barriers and cones restricting ground movement.",
    ),

    # ======== vehicles ========
    "car": AssetCategory(
        name="car",
        assets=[
            "SM_Sedan03_Main",
            "SM_SUV_01a",
            "SM_MERGED_SK_NYC_Car01",
        ],
        role="prop",
        physics=True,
        unit_size_m="4-5 m long",
        note="Street vehicles, usable as a search target or as cover.",
    ),
    "bicycle": AssetCategory(
        name="bicycle",
        assets=["SM_Bicycle_01", "SM_Bicycle_02"],
        role="prop",
        physics=True,
        unit_size_m="1.8 m long",
        note="Small street prop.",
    ),
    "cart": AssetCategory(
        name="cart",
        assets=["SM_Cart_v1_low", "SM_food_cart"],
        role="prop",
        physics=True,
        unit_size_m="1.5-2.5 m",
        note="Hand cart, food cart; street-side obstacle.",
    ),

    # ======== vegetation ========
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
        unit_size_m="5-12 m tall",
        note="Canopy occludes sight lines and flight paths at altitude.",
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
        unit_size_m="1-3 m",
        note="Ground-level occlusion.",
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
        unit_size_m="1-3 m",
        note="Decoration only; does not affect flight.",
    ),

    # ======== street furniture ========
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
        unit_size_m="6-9 m tall",
        note="Street lamps. Their height overlaps typical drone altitude, so they are natural obstacles.",
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
        unit_size_m="1-3 m",
        note="Benches, bins, hydrants, bus stops, phone booths and similar detail.",
    ),

    # ======== outdoor / temporary structures ========
    "tent": AssetCategory(
        name="tent",
        assets=[
            "SM_Tent_01b",
            "SM_PicnicTable_01",
            "SM_LawnChair_01",
            "SM_TableCloth_01a",
        ],
        role="prop",
        physics=True,
        unit_size_m="2-3 m",
        note="Field camp, suited to wilderness search-and-rescue.",
    ),
}


# --------------------------------------------------------------------------- #
# Query interface
# --------------------------------------------------------------------------- #
def category_names() -> List[str]:
    return list(CATALOG.keys())


def get(name: str) -> AssetCategory:
    if name not in CATALOG:
        raise KeyError(f"unknown asset category '{name}'. Available: {sorted(CATALOG)}")
    return CATALOG[name]


def all_assets() -> List[str]:
    """Every concrete asset name in the catalogue, de-duplicated, order kept."""
    seen: Dict[str, None] = {}
    for category in CATALOG.values():
        for asset in category.assets:
            seen.setdefault(asset, None)
    return list(seen.keys())


def whitelist() -> frozenset:
    """The set of asset names allowed in a specification."""
    return frozenset(all_assets())


def asset_role(asset: str) -> Optional[str]:
    for category in CATALOG.values():
        if asset in category.assets:
            return category.role
    return None


def asset_default_physics(asset: str) -> bool:
    for category in CATALOG.values():
        if asset in category.assets:
            return category.physics
    return False


def categories_for_role(role: str) -> List[str]:
    return [c.name for c in CATALOG.values() if c.role == role]


def prompt_text(include_usage: bool = True) -> str:
    """Render the catalogue for a language model prompt.

    Deliberately compact prose rather than JSON: a natural-language list is
    easier for a model to respect ("use only these") and easier to quote back.
    """
    lines = ["Available asset categories (use ONLY these names):"]
    for category in CATALOG.values():
        if not include_usage:
            lines.append(
                f"- `{category.name}` (role={category.role}"
                f"{', physics=true' if category.physics else ''}"
                f"{f', size~{category.unit_size_m}' if category.unit_size_m != 'n/a' else ''})"
                f" -> {', '.join(category.assets)}"
            )
        else:
            lines.append(category.to_prompt_line())
    return "\n".join(lines)


def compact_table() -> str:
    """One line per category, for eyeballing in a terminal."""
    rows = []
    for category in CATALOG.values():
        rows.append(
            f"{category.name:<18} role={category.role:<10} "
            f"physics={str(category.physics):<5} "
            f"n={len(category.assets):<3} {category.unit_size_m}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the asset catalogue")
    parser.add_argument("--prompt", action="store_true", help="print the model prompt text")
    parser.add_argument("--assets", action="store_true", help="print the flat asset list")
    parser.add_argument(
        "--check", nargs="*", metavar="ASSET",
        help="check whether assets are in the catalogue",
    )
    args = parser.parse_args()

    if args.prompt:
        print(prompt_text())
    elif args.assets:
        for name in all_assets():
            print(name)
    elif args.check is not None:
        allowed = whitelist()
        for name in args.check:
            print(f"{'OK   ' if name in allowed else 'MISS '} {name}")
    else:
        print(f"{len(CATALOG)} categories, {len(all_assets())} assets\n")
        print(compact_table())