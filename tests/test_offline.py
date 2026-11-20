#!/usr/bin/env python3
"""Offline self-test suite: no Unreal Engine, no AirSim, no GPU required.

Why this layer exists separately
--------------------------------
Half of the pipeline is pure data manipulation: specification validation,
coordinate conversion, the asset whitelist, template generation, and scenario
sanity checks. None of it needs an engine, a GPU, or a connection to AirSim, so
it runs in seconds on a laptop.

Separating those tests from the engine-bound half has two practical benefits:

  1. after touching the spec schema or the compiler, this suite confirms nothing
     broke *before* the expensive round trip to the workstation;
  2. a report or paper appendix can state that the offline components were
     verified by tests, rather than merely claiming that a demo ran once.

Running
-------
    python tests/test_offline.py            # concise
    python tests/test_offline.py -v         # verbose, per test
    python -m unittest tests.test_offline   # equivalent
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Sequence

# Make the src/ modules importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Windows consoles default to a legacy code page; pin the streams to UTF-8 so
# output never looks like a failing test.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
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
# A minimal valid specification used by several test groups
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
# 1. Coordinate conversion
# --------------------------------------------------------------------------- #
class TestCoordinateConversion(unittest.TestCase):
    """ENU -> NED is the easiest thing to get wrong; pin it down."""

    def test_enu_to_ned_swaps_and_flips(self):
        # ENU (east 1, north 2, up 3) -> NED (north 2, east 1, down -3)
        self.assertEqual(enu_to_ned([1.0, 2.0, 3.0]), (2.0, 1.0, -3.0))

    def test_enu_ned_roundtrip(self):
        for value in ([0, 0, 0], [5, -3, 2], [-12.5, 7.25, -1.5]):
            self.assertEqual(ned_to_enu(enu_to_ned(value)), tuple(float(x) for x in value))

    def test_up_is_negative_z_in_ned(self):
        # A drone at 10 m altitude sits at NED z = -10.
        self.assertEqual(enu_to_ned([0, 0, 10])[2], -10.0)

    def test_yaw_enu_to_ned_cardinal_directions(self):
        self.assertAlmostEqual(yaw_enu_to_ned(0.0), 90.0)     # east  -> NED 90
        self.assertAlmostEqual(yaw_enu_to_ned(90.0), 0.0)     # north -> NED 0
        self.assertAlmostEqual(yaw_enu_to_ned(180.0), 270.0)  # west  -> NED 270
        self.assertAlmostEqual(yaw_enu_to_ned(270.0), 180.0)  # south -> NED 180

    def test_yaw_wraps_within_range(self):
        for degrees in (-90, 0, 45, 359, 360, 720):
            got = yaw_enu_to_ned(degrees)
            self.assertTrue(0.0 <= got < 360.0, f"yaw {degrees} -> {got} out of range")


# --------------------------------------------------------------------------- #
# 2. Validation: what must be rejected
# --------------------------------------------------------------------------- #
class TestSpecValidationRejects(unittest.TestCase):
    """The validator is the quality gate; a broken gate is worse than none."""

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
        with self.assertRaises(SpecError) as context:
            parse_spec(data)
        self.assertIn("duplicate", str(context.exception))

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
        with self.assertRaises(SpecError) as context:
            parse_spec(data)
        self.assertIn("does_not_exist", str(context.exception))

    def test_rejects_malformed_position(self):
        data = minimal_spec()
        data["objects"][0]["pos"] = [1.0, 2.0]
        with self.assertRaises(SpecError):
            parse_spec(data)

    def test_accepts_numeric_strings_from_llm(self):
        """Models often write numbers as strings; accept them instead of crashing."""
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
        """Report everything in one pass, not one problem per retry."""
        data = minimal_spec(name="", seed="not-an-int", objects=[])
        with self.assertRaises(SpecError) as context:
            parse_spec(data)
        self.assertGreaterEqual(len(context.exception.problems), 2)


# --------------------------------------------------------------------------- #
# 3. Asset catalogue
# --------------------------------------------------------------------------- #
class TestAssetCatalog(unittest.TestCase):
    def test_catalog_is_not_empty(self):
        self.assertGreaterEqual(len(asset_catalog.CATALOG), 10)
        self.assertGreater(len(asset_catalog.all_assets()), 40)

    def test_whitelist_matches_all_assets(self):
        self.assertEqual(set(asset_catalog.whitelist()), set(asset_catalog.all_assets()))

    def test_every_category_declares_valid_role(self):
        for category in asset_catalog.CATALOG.values():
            self.assertIn(category.role, {"structure", "prop", "target", "hazard"})

    def test_structure_categories_have_physics(self):
        """A structure without collision means the drone flies through it."""
        for category in asset_catalog.CATALOG.values():
            if category.role == "structure":
                self.assertTrue(
                    category.physics,
                    f"category {category.name} is structure but physics=False",
                )

    def test_prompt_text_mentions_every_asset(self):
        text = asset_catalog.prompt_text()
        missing = [a for a in asset_catalog.all_assets() if a not in text]
        self.assertEqual(missing, [], f"assets missing from the prompt: {missing}")

    def test_get_raises_on_unknown_category(self):
        with self.assertRaises(KeyError):
            asset_catalog.get("no_such_category")


# --------------------------------------------------------------------------- #
# 4. The reference scenario must be fully compliant
# --------------------------------------------------------------------------- #
class TestReferenceSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = PROJECT_ROOT / "specs" / "fire_floor2.json"
        cls.spec = load_spec(cls.path)

    def test_loads(self):
        self.assertEqual(self.spec.name, "fire_floor2")

    def test_assets_all_in_catalogue(self):
        problems = llm_compiler.check_assets_whitelisted(self.spec)
        self.assertEqual(problems, [], f"assets outside the catalogue: {problems}")

    def test_passes_scenario_sanity(self):
        problems = llm_compiler.check_scenario_sanity(self.spec)
        self.assertEqual(problems, [], f"sanity checks failed: {problems}")

    def test_has_single_target_and_spawn(self):
        targets = [o for o in self.spec.objects if o.role == "target"]
        self.assertEqual(len(targets), 1)
        self.assertIsNotNone(self.spec.target)
        self.assertEqual(self.spec.target.object_id, targets[0].id)
        self.assertIsNotNone(self.spec.spawn)

    def test_structure_has_collision(self):
        for obj in self.spec.objects:
            if obj.role == "structure":
                self.assertTrue(obj.physics, f"{obj.id} is structure without collision")

    def test_target_is_hidden_behind_structure(self):
        """The goal should sit inside the building footprint, not in the open."""
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
        self.assertGreaterEqual(spawn.pos[2], 3.0, "spawn altitude below 3 m")
        horizontal = (
            (spawn.pos[0] - goal.pos[0]) ** 2 + (spawn.pos[1] - goal.pos[1]) ** 2
        ) ** 0.5
        self.assertGreaterEqual(horizontal, 8.0, "spawn too close to the target")


# --------------------------------------------------------------------------- #
# 5. Offline template: every request must produce a compliant spec
# --------------------------------------------------------------------------- #
class TestTemplateGenerator(unittest.TestCase):
    REQUESTS = [
        "fire in a two-storey apartment, searching for a survivor",
        "city street search for a lost car",
        "outdoor forest search and rescue",
        "earthquake rubble rescue mission",
        "industrial pipe inspection",
        "burning warehouse with smoke",
    ]

    def test_all_requests_produce_valid_spec(self):
        for request in self.REQUESTS:
            with self.subTest(request=request):
                data = llm_compiler.template_spec(request)
                spec = parse_spec(data)
                problems = llm_compiler.validate_all(spec)
                self.assertEqual(problems, [], f"request '{request}' invalid: {problems}")

    def test_template_is_deterministic(self):
        first = llm_compiler.template_spec("a burning building")
        second = llm_compiler.template_spec("a burning building")
        self.assertEqual(first, second)

    def test_fire_keyword_adds_hazard_objects(self):
        with_fire = llm_compiler.template_spec("a burning building")
        without = llm_compiler.template_spec("a plain building")
        fire_hazards = [o for o in with_fire["objects"] if o["role"] == "hazard"]
        plain_hazards = [o for o in without["objects"] if o["role"] == "hazard"]
        self.assertGreater(len(fire_hazards), len(plain_hazards))

    def test_template_name_reflects_request(self):
        self.assertIn("fire", llm_compiler.template_spec("a burning building")["name"])
        self.assertIn("inspection", llm_compiler.template_spec("inspection job")["name"])

    def test_template_scenarios_are_geometrically_usable(self):
        """End-to-end constraint: template output must actually be flyable.

        This raises the bar from "schema-valid" to "geometrically usable". A
        sealed upper floor would pass schema validation and still be worthless.
        """
        requests = [
            "fire in a two-storey apartment, searching for a survivor",
            "city street search for a lost car",
            "outdoor forest search and rescue",
            "earthquake rubble rescue mission",
        ]
        for request in requests:
            with self.subTest(request=request):
                spec = parse_spec(llm_compiler.template_spec(request))
                result = verifier.verify(spec)
                self.assertTrue(
                    result.usable,
                    f"scenario for '{request}' is not geometrically usable: "
                    f"{result.failures}",
                )


# --------------------------------------------------------------------------- #
# 6. Specification round-tripping
# --------------------------------------------------------------------------- #
class TestSpecRoundtrip(unittest.TestCase):
    def test_dict_spec_dict_is_stable(self):
        """spec -> dict -> spec -> dict must match, or fields are silently lost."""
        original = parse_spec(minimal_spec())
        first = llm_compiler.spec_to_dict(original)
        second = llm_compiler.spec_to_dict(parse_spec(first))
        self.assertEqual(first, second)

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
        first = llm_compiler.spec_to_dict(spec)
        second = llm_compiler.spec_to_dict(parse_spec(first))
        self.assertEqual(first, second)


# --------------------------------------------------------------------------- #
# 7. JSON extraction from messy model output
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
# 8. Geometric verifier
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
    """A 10x10x3 room, optionally with a 4 m doorway in the south wall.

    The origin is deliberately far from zero so that mixing local and world
    coordinates is caught: a sealed room would be reported solvable.
    """
    objects = [
        # Ground base. Without it the space under the floor slab is open, so the
        # drone can enter from below and the room is not sealed in 3D.
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
    """The geometry behind the occlusion test."""

    def test_segment_through_box_hits(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertTrue(verifier.segment_hits_aabb((-5, 0, 0), (5, 0, 0), box))

    def test_segment_passing_beside_box_misses(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertFalse(verifier.segment_hits_aabb((-5, 5, 0), (5, 5, 0), box))

    def test_segment_stopping_short_misses(self):
        box = verifier.AABB((10, -1, -1), (12, 1, 1))
        self.assertFalse(verifier.segment_hits_aabb((-5, 0, 0), (5, 0, 0), box))

    def test_segment_above_box_misses(self):
        box = verifier.AABB((0, -1, -1), (2, 1, 1))
        self.assertFalse(verifier.segment_hits_aabb((-5, 0, 5), (5, 0, 5), box))


class TestVerifierOnRooms(unittest.TestCase):
    """The verifier must tell a sealed room from one with a doorway."""

    def test_sealed_room_is_unsolvable(self):
        spec = parse_spec(_room_spec(with_doorway=False))
        result = verifier.verify(spec, resolution=0.5, clearance=0.6)
        self.assertFalse(result.solvable, "a sealed room must not be judged solvable")
        self.assertTrue(result.initially_occluded, "the walls should block the sight line")
        self.assertFalse(result.usable)

    def test_room_with_doorway_is_solvable(self):
        spec = parse_spec(_room_spec(with_doorway=True))
        result = verifier.verify(spec, resolution=0.5, clearance=0.6)
        self.assertTrue(result.solvable, "a room with a 4 m doorway must be solvable")
        self.assertIsNotNone(result.path_length_m)
        self.assertGreater(result.path_length_m, result.straight_distance_m)

    def test_open_scene_is_not_occluded(self):
        """With no occluders the verifier must report "not occluded" and reject."""
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
        self.assertIn(verifier.OCCLUSION_FAILURE, result.failures)

    def test_requirement_max_time_can_fail(self):
        """Requirement checks must be able to fail, or they are decoration."""
        data = _room_spec(
            with_doorway=True,
            requirements=[{"kind": "max_time_s", "value": 1}],
        )
        spec = parse_spec(data)
        result = verifier.verify(spec)
        self.assertFalse(result.usable)
        self.assertTrue(
            any(
                r["kind"] == "max_time_s" and r["satisfiable"] is False
                for r in result.requirement_results
            )
        )

    def test_requirement_min_altitude_can_fail(self):
        data = _room_spec(
            with_doorway=True,
            requirements=[{"kind": "min_altitude_m", "value": 100}],
        )
        spec = parse_spec(data)
        result = verifier.verify(spec)
        self.assertTrue(
            any(
                r["kind"] == "min_altitude_m" and r["satisfiable"] is False
                for r in result.requirement_results
            )
        )


class TestVerifierOnReferenceSpec(unittest.TestCase):
    """The reference scenario must pass the full verification; it is the baseline."""

    def test_reference_scenario_is_usable(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        self.assertTrue(result.solvable, f"unsolvable: {result.failures}")
        self.assertTrue(result.initially_occluded, "the target must start occluded")
        self.assertEqual(result.failures, [], f"failures: {result.failures}")
        self.assertTrue(result.usable)

    def test_reference_path_length_is_plausible(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        # Entering a doorway and climbing to the upper floor must cost distance.
        self.assertGreater(result.path_length_m, result.straight_distance_m * 1.2)
        self.assertLess(result.path_length_m, result.straight_distance_m * 4.0)

    def test_grid_metadata_is_populated(self):
        spec = load_spec(PROJECT_ROOT / "specs" / "fire_floor2.json")
        result = verifier.verify(spec)
        self.assertIn("resolution_m", result.grid_meta)
        self.assertIn("cells", result.grid_meta)
        self.assertGreater(result.grid_meta["cells"], 0)


# --------------------------------------------------------------------------- #
# Entry point
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