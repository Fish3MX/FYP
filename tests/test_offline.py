#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_offline.py —— 不需要 UE / AirSim 的全套自检

为什么单独有这一层
------------------
整条流水线里有一半是"纯数据"的：规格书校验、坐标转换、资产白名单、
模板生成、场景合理性检查。这些不需要引擎、不需要 GPU、不需要连接
AirSim，在你的笔记本上就能跑完，几秒钟出结果。

把这一半的测试和"需要引擎"的那一半分开，有两个实际好处：

  1. 改了规格书或编译器之后，先跑这个确认没搞坏逻辑，再上远程机浪费
     时间启动 UE；
  2. 论文附录里可以直接报告"离线组件测试全部通过"，说明工程可靠性，
     而不只是"我跑通了一个 demo"。

运行
----
    python tests/test_offline.py            # 简洁输出
    python tests/test_offline.py -v         # 每个用例都打印
    python -m unittest tests.test_offline   # 等价
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Sequence

# 让测试能 import 到 src/ 下的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Windows 控制台默认用 CP936，而源码是 UTF-8，中文会变乱码。
# 这里把输出流钉死为 UTF-8，避免看起来像测试失败。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # 非标准流或已重定向时忽略
        pass

import asset_catalog  # noqa: E402
import llm_compiler  # noqa: E402
import verifier  # noqa: E402
from scene_spec import (  # noqa: E402
    SpecError,
    enu_to_ned,
    load_spec,
    ned_to_enu,
    parse_spec,
    yaw_enu_to_ned,
)


# --------------------------------------------------------------------------- #
# 测试用的最小合法规格书
# --------------------------------------------------------------------------- #
def minimal_spec(**overrides) -> dict:
    base = {
        "version": "0.1",
        "name": "unit_test",
        "seed": 1,
        "origin": [0.0, 0.0, 0.0],
        "objects": [
            {
                "id": "floor",
                "asset": "Cube",
                "pos": [0.0, 0.0, 0.1],
                "scale": [10.0, 10.0, 0.2],
                "physics": True,
                "role": "structure",
            },
            {
                "id": "goal",
                "asset": "BP_ma_construction_worker_01",
                "pos": [3.0, 0.0, 0.1],
                "physics": True,
                "role": "target",
            },
        ],
        "target": {"object_id": "goal", "landing_radius_m": 4.0},
        "spawn": {"pos": [-20.0, 0.0, 8.0], "yaw_deg": 0.0},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 1. 坐标转换
# --------------------------------------------------------------------------- #
class TestCoordinateConversion(unittest.TestCase):
    """ENU -> NED 是全项目最容易写错的地方，必须钉死。"""

    def test_enu_to_ned_swaps_and_flips(self):
        # ENU (东1, 北2, 上3) -> NED (北2, 东1, 下-3)
        self.assertEqual(enu_to_ned([1.0, 2.0, 3.0]), (2.0, 1.0, -3.0))

    def test_enu_ned_roundtrip(self):
        for v in ([0, 0, 0], [5, -3, 2], [-12.5, 7.25, -1.5]):
            self.assertEqual(ned_to_enu(enu_to_ned(v)), tuple(float(x) for x in v))

    def test_up_is_negative_z_in_ned(self):
        # 无人机在 10 m 高 -> NED z = -10
        self.assertEqual(enu_to_ned([0, 0, 10])[2], -10.0)

    def test_yaw_enu_to_ned_cardinal_directions(self):
        self.assertAlmostEqual(yaw_enu_to_ned(0.0), 90.0)     # 东 -> NED 90
        self.assertAlmostEqual(yaw_enu_to_ned(90.0), 0.0)     # 北 -> NED 0
        self.assertAlmostEqual(yaw_enu_to_ned(180.0), 270.0)  # 西 -> NED 270
        self.assertAlmostEqual(yaw_enu_to_ned(270.0), 180.0)  # 南 -> NED 180

    def test_yaw_wraps_within_range(self):
        for deg in (-90, 0, 45, 359, 360, 720):
            got = yaw_enu_to_ned(deg)
            self.assertTrue(0.0 <= got < 360.0, f"yaw {deg} -> {got} 越界")


# --------------------------------------------------------------------------- #
# 2. 规格书校验：该拒绝的必须拒绝
# --------------------------------------------------------------------------- #
class TestSpecValidationRejects(unittest.TestCase):
    """校验器是质量闸门。闸门失灵比没有闸门更危险。"""

    def test_accepts_minimal_valid_spec(self):
        spec = parse_spec(minimal_spec())
        self.assertEqual(spec.name, "unit_test")
        self.assertEqual(len(spec.objects), 2)

    def test_rejects_wrong_version(self):
        with self.assertRaises(SpecError):
            parse_spec(minimal_spec(version="9.9"))

    def test_rejects_empty_name(self):
        with self.assertRaises(SpecError):
            parse_spec(minimal_spec(name=""))

    def test_rejects_empty_objects(self):
        with self.assertRaises(SpecError):
            parse_spec(minimal_spec(objects=[]))

    def test_rejects_duplicate_object_id(self):
        data = minimal_spec()
        data["objects"].append(dict(data["objects"][0]))
        with self.assertRaises(SpecError) as ctx:
            parse_spec(data)
        self.assertIn("重复", str(ctx.exception))

    def test_rejects_unknown_role(self):
        data = minimal_spec()
        data["objects"][0]["role"] = "wallpaper"
        with self.assertRaises(SpecError):
            parse_spec(data)

    def test_rejects_non_positive_scale(self):
        data = minimal_spec()
        data["objects"][0]["scale"] = [10.0, 0.0, 0.2]
        with self.assertRaises(SpecError):
            parse_spec(data)

    def test_rejects_target_pointing_to_missing_object(self):
        data = minimal_spec()
        data["target"]["object_id"] = "does_not_exist"
        with self.assertRaises(SpecError) as ctx:
            parse_spec(data)
        self.assertIn("does_not_exist", str(ctx.exception))

    def test_rejects_malformed_position(self):
        data = minimal_spec()
        data["objects"][0]["pos"] = [1.0, 2.0]
        with self.assertRaises(SpecError):
            parse_spec(data)

    def test_accepts_numeric_strings_from_llm(self):
        """
        模型常把数字写成字符串（"4.5"、"300"）。这不是错误，应被宽容接受，
        而不是让整个编译循环崩掉。
        """
        data = minimal_spec()
        data["target"]["landing_radius_m"] = "4.5"
        data["spawn"]["yaw_deg"] = "90"
        spec = parse_spec(data)
        self.assertAlmostEqual(spec.target.landing_radius_m, 4.5)
        self.assertAlmostEqual(spec.spawn.yaw_deg, 90.0)

    def test_rejects_non_numeric_radius_without_crashing(self):
        data = minimal_spec()
        data["target"]["landing_radius_m"] = "about four metres"
        with self.assertRaises(SpecError):
            parse_spec(data)

    def test_reports_multiple_problems_at_once(self):
        """一次报全所有问题，而不是修一个报一个。"""
        data = minimal_spec(name="", seed="not-an-int", objects=[])
        with self.assertRaises(SpecError) as ctx:
            parse_spec(data)
        self.assertGreaterEqual(len(ctx.exception.problems), 2)


# --------------------------------------------------------------------------- #
# 3. 资产目录
# --------------------------------------------------------------------------- #
class TestAssetCatalog(unittest.TestCase):

    def test_catalog_is_not_empty(self):
        self.assertGreaterEqual(len(asset_catalog.CATALOG), 10)
        self.assertGreater(len(asset_catalog.all_assets()), 40)

    def test_whitelist_matches_all_assets(self):
        self.assertEqual(
            set(asset_catalog.whitelist()), set(asset_catalog.all_assets())
        )

    def test_every_category_declares_valid_role(self):
        for cat in asset_catalog.CATALOG.values():
            self.assertIn(cat.role, {"structure", "prop", "target", "hazard"})

    def test_structure_categories_have_physics(self):
        """结构件必须带碰撞，否则无人机会穿墙。"""
        for cat in asset_catalog.CATALOG.values():
            if cat.role == "structure":
                self.assertTrue(
                    cat.physics, f"类别 {cat.name} 是结构但 physics=False"
                )

    def test_prompt_text_mentions_every_asset(self):
        text = asset_catalog.prompt_text()
        missing = [a for a in asset_catalog.all_assets() if a not in text]
        self.assertEqual(missing, [], f"提示词漏掉资产: {missing}")

    def test_get_raises_on_unknown_category(self):
        with self.assertRaises(KeyError):
            asset_catalog.get("no_such_category")


# --------------------------------------------------------------------------- #
# 4. 参照规格书必须完全合规
# --------------------------------------------------------------------------- #
class TestReferenceSpec(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = PROJECT_ROOT / "specs" / "fire_floor2.json"
        cls.spec = load_spec(path)
        cls.raw = json.loads(path.read_text(encoding="utf-8"))

    def test_loads(self):
        self.assertEqual(self.spec.name, "fire_floor2")

    def test_assets_all_in_catalogue(self):
        problems = llm_compiler.check_assets_whitelisted(self.spec)
        self.assertEqual(problems, [], f"规格书用了目录外资产: {problems}")

    def test_passes_scenario_sanity(self):
        problems = llm_compiler.check_scenario_sanity(self.spec)
        self.assertEqual(problems, [], f"场景合理性检查失败: {problems}")

    def test_has_single_target_and_spawn(self):
        targets = [o for o in self.spec.objects if o.role == "target"]
        self.assertEqual(len(targets), 1)
        self.assertIsNotNone(self.spec.target)
        self.assertEqual(self.spec.target.object_id, targets[0].id)
        self.assertIsNotNone(self.spec.spawn)

    def test_structure_has_collision(self):
        for o in self.spec.objects:
            if o.role == "structure":
                self.assertTrue(o.physics, f"{o.id} 是结构但没开碰撞")

    def test_target_is_hidden_behind_structure(self):
        """
        目标必须在结构内部：它的水平位置应落在建筑占地范围内。
        这是"搜索任务"而非"直飞任务"的必要条件（粗略版）。
        """
        goal = self.spec.object_by_id(self.spec.target.object_id)
        xs = [o.pos[0] for o in self.spec.objects if o.role == "structure"]
        ys = [o.pos[1] for o in self.spec.objects if o.role == "structure"]
        self.assertGreaterEqual(goal.pos[0], min(xs))
        self.assertLessEqual(goal.pos[0], max(xs))
        self.assertGreaterEqual(goal.pos[1], min(ys))
        self.assertLessEqual(goal.pos[1], max(ys))

    def test_drone_starts_outside_and_airborne(self):
        goal = self.spec.object_by_id(self.spec.target.object_id)
        spawn = self.spec.spawn
        self.assertGreaterEqual(spawn.pos[2], 3.0, "起点高度不足 3 m")
        horizontal = (
            (spawn.pos[0] - goal.pos[0]) ** 2 + (spawn.pos[1] - goal.pos[1]) ** 2
        ) ** 0.5
        self.assertGreaterEqual(horizontal, 8.0, "起点离目标太近，任务过于简单")


# --------------------------------------------------------------------------- #
# 5. 离线模板：多种需求都必须产出合规规格书
# --------------------------------------------------------------------------- #
class TestTemplateGenerator(unittest.TestCase):

    REQUESTS = [
        "着火的二层公寓里搜索伤员",
        "test the drone searching for a survivor inside a burning two-storey apartment",
        "city street search for a lost car",
        "outdoor forest search and rescue",
        "inspection of an industrial facility with pipes",
        "地震废墟中的搜救任务",
    ]

    def test_all_requests_produce_valid_spec(self):
        for req in self.REQUESTS:
            with self.subTest(request=req):
                data = llm_compiler.template_spec(req)
                spec = parse_spec(data)          # 结构必须合法
                problems = llm_compiler.validate_all(spec)
                self.assertEqual(problems, [], f"请求 '{req}' 产出不合规: {problems}")

    def test_template_is_deterministic(self):
        a = llm_compiler.template_spec("着火的二层公寓")
        b = llm_compiler.template_spec("着火的二层公寓")
        self.assertEqual(a, b)

    def test_fire_keyword_adds_hazard_objects(self):
        with_fire = llm_compiler.template_spec("着火的楼")
        without = llm_compiler.template_spec("普通的楼")
        fire_hazards = [o for o in with_fire["objects"] if o["role"] == "hazard"]
        plain_hazards = [o for o in without["objects"] if o["role"] == "hazard"]
        self.assertGreater(len(fire_hazards), len(plain_hazards))

    def test_template_name_reflects_request(self):
        self.assertIn("fire", llm_compiler.template_spec("着火的楼")["name"])
        self.assertIn("inspection", llm_compiler.template_spec("inspection job")["name"])

    def test_template_scenarios_are_geometrically_usable(self):
        """
        端到端约束：模板产出的场景必须**真的能跑**——可解、初始遮挡、断言可满足。

        这条测试把"格式正确"提升为"几何可用"。如果只测 schema 校验，
        一个二楼被封死、根本飞不进去的场景也会被当成通过。
        """
        requests = [
            "着火的二层公寓里搜索伤员",
            "city street search for a lost car",
            "outdoor forest search and rescue",
            "地震废墟中的搜救任务",
        ]
        for req in requests:
            with self.subTest(request=req):
                spec = parse_spec(llm_compiler.template_spec(req))
                result = verifier.verify(spec)
                self.assertTrue(
                    result.usable,
                    f"请求 '{req}' 的场景几何不可用: {result.failures}",
                )


# --------------------------------------------------------------------------- #
# 6. 规格书往返序列化
# --------------------------------------------------------------------------- #
class TestSpecRoundtrip(unittest.TestCase):

    def test_dict_spec_dict_is_stable(self):
        """spec -> dict -> spec -> dict 必须一致，否则写盘会悄悄丢字段。"""
        original = parse_spec(minimal_spec())
        d1 = llm_compiler.spec_to_dict(original)
        again = parse_spec(d1)
        d2 = llm_compiler.spec_to_dict(again)
        self.assertEqual(d1, d2)

    def test_roundtrip_preserves_key_fields(self):
        original = parse_spec(minimal_spec())
        again = parse_spec(llm_compiler.spec_to_dict(original))
        self.assertEqual(again.name, original.name)
        self.assertEqual(again.seed, original.seed)
        self.assertEqual(again.origin, original.origin)
        self.assertEqual(len(again.objects), len(original.objects))
        self.assertEqual(again.target.object_id, original.target.object_id)

    def test_reference_spec_roundtrips(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        d1 = llm_compiler.spec_to_dict(spec)
        d2 = llm_compiler.spec_to_dict(parse_spec(d1))
        self.assertEqual(d1, d2)


# --------------------------------------------------------------------------- #
# 7. JSON 抽取：模型不听话时的容错
# --------------------------------------------------------------------------- #
class TestJsonExtraction(unittest.TestCase):

    def test_plain_json(self):
        self.assertEqual(llm_compiler.extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fenced_json(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
        self.assertEqual(llm_compiler.extract_json(text), {"a": 1})

    def test_json_with_prose_around_it(self):
        text = 'Sure! {"a": 1, "b": [2, 3]} That is the spec.'
        self.assertEqual(llm_compiler.extract_json(text), {"a": 1, "b": [2, 3]})

    def test_json_with_trailing_comma(self):
        self.assertEqual(llm_compiler.extract_json('{"a": 1,}'), {"a": 1})

    def test_raises_when_no_json(self):
        with self.assertRaises(ValueError):
            llm_compiler.extract_json("I cannot help with that.")


# --------------------------------------------------------------------------- #
# 8. 几何验证器
# --------------------------------------------------------------------------- #
def _box(
    obj_id: str,
    pos: Sequence[float],
    scale: Sequence[float],
    role: str = "structure",
    physics: bool = True,
) -> dict:
    return {
        "id": obj_id,
        "asset": "Cube",
        "pos": list(pos),
        "scale": list(scale),
        "physics": physics,
        "role": role,
    }


def _room_spec(with_doorway: bool, **overrides) -> dict:
    """
    一个 10x10x3 的封闭房间，可通过 `with_doorway` 在南墙留 4 m 门洞。

    刻意把 origin 设成远离零点的值，用来抓"局部坐标与世界坐标混用"这类 bug：
    如果实现里漏了 origin 偏移，封闭房间会被误判成可解。
    """
    objects = [
        # 地基：没有它，楼板下方的空间是通的，无人机会从楼底钻进来，
        # 于是"封闭房间"在 3D 上并不封闭。规格书约定必须显式给地基。
        _box("ground_base", [0, 0, -0.4], [30, 30, 1.2]),
        _box("floor", [0, 0, 0.1], [10, 10, 0.2]),
        _box("ceiling", [0, 0, 3.1], [10, 10, 0.2]),
        _box("wall_north", [0, 5, 1.6], [10, 0.2, 3]),
        _box("wall_east", [5, 0, 1.6], [0.2, 10, 3]),
        _box("wall_west", [-5, 0, 1.6], [0.2, 10, 3]),
    ]
    if with_doorway:
        objects += [
            _box("wall_south_left", [-3.5, -5, 1.6], [3, 0.2, 3]),
            _box("wall_south_right", [3.5, -5, 1.6], [3, 0.2, 3]),
        ]
    else:
        objects.append(_box("wall_south", [0, -5, 1.6], [10, 0.2, 3]))

    objects.append(
        {
            "id": "goal",
            "asset": "BP_ma_construction_worker_01",
            "pos": [0, 0, 0.3],
            "physics": True,
            "role": "target",
        }
    )

    spec = {
        "version": "0.1",
        "name": "room_door" if with_doorway else "room_sealed",
        "seed": 1,
        "origin": [100.0, 100.0, 0.0],
        "objects": objects,
        "target": {"object_id": "goal", "landing_radius_m": 2.0},
        "spawn": {"pos": [-12, 0, 6], "yaw_deg": 0},
        "requirements": [],
    }
    spec.update(overrides)
    return spec


class TestSegmentAABB(unittest.TestCase):
    """视线遮挡的基础：线段与包围盒相交判定。"""

    def test_segment_through_box_hits(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertTrue(
            verifier.segment_hits_aabb((-5, 0, 0), (5, 0, 0), box)
        )

    def test_segment_passing_beside_box_misses(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertFalse(
            verifier.segment_hits_aabb((-5, 5, 0), (5, 5, 0), box)
        )

    def test_segment_stopping_short_misses(self):
        box = verifier.AABB((10, -1, -1), (12, 1, 1))
        self.assertFalse(
            verifier.segment_hits_aabb((-5, 0, 0), (5, 0, 0), box)
        )

    def test_segment_above_box_misses(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertFalse(
            verifier.segment_hits_aabb((-5, 0, 5), (5, 0, 5), box)
        )


class TestVerifierOnRooms(unittest.TestCase):
    """验证器必须能区分"封死的房间"和"有门洞的房间"。"""

    def test_sealed_room_is_unsolvable(self):
        spec = parse_spec(_room_spec(with_doorway=False))
        result = verifier.verify(spec, resolution=0.5, clearance=0.6)
        self.assertFalse(result.solvable, "封死的房间不应被判为可解")
        self.assertTrue(result.initially_occluded, "墙体应挡住视线")
        self.assertFalse(result.usable)

    def test_room_with_doorway_is_solvable(self):
        spec = parse_spec(_room_spec(with_doorway=True))
        result = verifier.verify(spec, resolution=0.5, clearance=0.6)
        self.assertTrue(result.solvable, "有 4 m 门洞的房间应当可解")
        self.assertIsNotNone(result.path_length_m)
        self.assertGreater(result.path_length_m, result.straight_distance_m)

    def test_open_scene_is_not_occluded(self):
        """没有任何遮挡物时，验证器必须报告"未被遮挡"并因此拒绝该场景。"""
        data = {
            "version": "0.1",
            "name": "open_field",
            "seed": 1,
            "origin": [0.0, 0.0, 0.0],
            "objects": [
                _box("pad", [0, 0, 0.1], [6, 6, 0.2]),
                {
                    "id": "goal",
                    "asset": "BP_ma_construction_worker_01",
                    "pos": [10, 0, 0.3],
                    "physics": True,
                    "role": "target",
                },
            ],
            "target": {"object_id": "goal", "landing_radius_m": 2.0},
            "spawn": {"pos": [-10, 0, 6], "yaw_deg": 0},
        }
        spec = parse_spec(data)
        result = verifier.verify(spec)
        self.assertFalse(result.initially_occluded)
        self.assertIn(
            "起点可以直线看到目标，任务不是搜索任务（初始遮挡检查失败）",
            result.failures,
        )

    def test_requirement_max_time_can_fail(self):
        """需求断言要真的会失败，否则它就只是装饰。"""
        data = _room_spec(
            with_doorway=True,
            requirements=[{"kind": "max_time_s", "value": 1}],
        )
        spec = parse_spec(data)
        result = verifier.verify(spec)
        self.assertFalse(result.usable)
        self.assertTrue(
            any(r["kind"] == "max_time_s" and r["satisfiable"] is False
                for r in result.requirement_results)
        )

    def test_requirement_min_altitude_can_fail(self):
        data = _room_spec(
            with_doorway=True,
            requirements=[{"kind": "min_altitude_m", "value": 100}],
        )
        spec = parse_spec(data)
        result = verifier.verify(spec)
        self.assertTrue(
            any(r["kind"] == "min_altitude_m" and r["satisfiable"] is False
                for r in result.requirement_results)
        )


class TestVerifierOnReferenceSpec(unittest.TestCase):
    """参照场景必须通过完整验证——它是后面所有对照实验的基线。"""

    def test_reference_scenario_is_usable(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        self.assertTrue(result.solvable, f"不可解: {result.failures}")
        self.assertTrue(result.initially_occluded, "目标必须初始被遮挡")
        self.assertEqual(result.failures, [], f"验证失败项: {result.failures}")
        self.assertTrue(result.usable)

    def test_reference_path_length_is_plausible(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        # 进入门洞再上二楼，路径必须显著长于直线距离
        self.assertGreater(result.path_length_m, result.straight_distance_m * 1.2)
        self.assertLess(result.path_length_m, result.straight_distance_m * 4.0)

    def test_grid_metadata_is_populated(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        self.assertIn("resolution_m", result.grid_meta)
        self.assertIn("cells", result.grid_meta)
        self.assertGreater(result.grid_meta["cells"], 0)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    banner = "=" * 68
    print(banner)
    print("Offline self-test: no Unreal Engine, no AirSim, no GPU required")
    print(f"Project root: {PROJECT_ROOT}")
    print(banner)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent),
        pattern="test_*.py",
        top_level_dir=str(Path(__file__).resolve().parent),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print(banner)
    if result.wasSuccessful():
        print(f"PASS  {result.testsRun} tests, 0 failures, 0 errors")
        print("Offline components are consistent.")
        print(banner)
        return 0

    print(f"FAIL  {result.testsRun} tests run")
    print(f"      failures={len(result.failures)} errors={len(result.errors)}")
    print(banner)
    return 1


if __name__ == "__main__":
    sys.exit(main())