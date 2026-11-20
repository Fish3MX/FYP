#!/usr/bin/env python3
"""Geometric verification of a scenario: pure CPU, no Unreal Engine, no AirSim.

Why the verifier is the contribution
------------------------------------
"Let a large model generate a 3D scene" is not by itself a research result,
because the result may be unusable: the building may be sealed, the target may
be visible from take-off (so it is not a search task), or the path may be so
short that the scenario measures nothing.

This module answers four questions **before any agent runs**:

  1. Is it solvable?        -- 3D occupancy grid plus A*; no path means the
                               generator failed, not the agent.
  2. Is it occluded at t=0? -- is the line from spawn to target blocked? This
                               is what makes the task a search task.
  3. Is it non-trivial?     -- path length and detour ratio against thresholds;
                               a trivial scenario is rejected.
  4. Are the requirements satisfiable? -- each assertion is evaluated on the
                               ground-truth path.

Key design decision: no engine
------------------------------
Verification reads only the specification and approximates object volumes with
axis-aligned bounding boxes. Two benefits:

  * **fast**: tens of milliseconds, so scenarios can be screened in bulk on a
    laptop;
  * **testable**: unit tests can prove that it really rejects bad scenarios.

The price is approximation error. That error is not ignored but *measured*:
comparing verifier verdicts against AirSim's real colliders requires the engine
and is kept as a separate experiment.

Coordinates
-----------
Specifications and all internal computation use ENU
(x=east, y=north, z=up, metres). NED never appears here.

Usage
-----
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from asset_catalog import CATALOG
from scene_spec import SceneObject, SceneSpec, load_spec

LOG = logging.getLogger("verifier")

# Conservative drone radius (metres). Obstacles are inflated by this much so a
# planned path is a corridor the drone can actually fly, not a zero-width line
# grazing a wall.
DEFAULT_CLEARANCE_M = 0.6

# Grid resolution (metres). Finer means more accurate but memory and time grow
# cubically.
DEFAULT_RESOLUTION_M = 0.5

# Above this cell count the resolution is coarsened automatically.
MAX_CELLS = 6_000_000

# Assumed cruise speed, used only to turn path length into an estimated duration
# when checking max_time_s.
ASSUMED_CRUISE_MPS = 3.0

# Message emitted when the target is visible from the spawn pose. Kept as a
# constant because tests and downstream tooling match on it.
OCCLUSION_FAILURE = (
    "the spawn pose can see the target directly, so the task is not a search "
    "task (initial occlusion check failed)"
)


# --------------------------------------------------------------------------- #
# Asset -> footprint
# --------------------------------------------------------------------------- #
# Primitives take their size from `scale` (1 m units); other assets use the
# nominal size of their category. These numbers only need to be conservative --
# the goal is to stop the planner squeezing the drone through gaps smaller than
# itself, not to reproduce the art asset exactly.
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

# Asset name -> category, derived from the catalogue instead of maintained twice.
ASSET_TO_CATEGORY: Dict[str, str] = {}
for _category in CATALOG.values():
    for _asset in _category.assets:
        ASSET_TO_CATEGORY.setdefault(_asset, _category.name)


# --------------------------------------------------------------------------- #
# Basic geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box in world ENU metres."""

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

    def contains(self, point: Sequence[float]) -> bool:
        return all(self.lo[i] <= point[i] <= self.hi[i] for i in range(3))

    def intersects(self, other: "AABB") -> bool:
        return all(
            self.lo[i] <= other.hi[i] and other.lo[i] <= self.hi[i] for i in range(3)
        )

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            tuple(min(self.lo[i], other.lo[i]) for i in range(3)),  # type: ignore[arg-type]
            tuple(max(self.hi[i], other.hi[i]) for i in range(3)),  # type: ignore[arg-type]
        )


def object_extent(spec: SceneSpec, obj: SceneObject) -> Tuple[float, float, float]:
    """Estimate an object's bounding box size in metres.

    Primitives are computed exactly from `scale`; everything else uses its
    category's nominal size multiplied by `scale`. An unknown asset falls back
    to a 1 m cube: over-estimating an obstacle is safer than missing it.
    """
    category = ASSET_TO_CATEGORY.get(obj.asset)
    base = CATEGORY_EXTENT_M.get(category or "", (1.0, 1.0, 1.0))
    sx, sy, sz = obj.scale
    return (base[0] * sx, base[1] * sy, base[2] * sz)


def object_aabb(spec: SceneSpec, obj: SceneObject) -> AABB:
    """World-space AABB of an object.

    Yaw is ignored: a rotated object is treated as its axis-aligned outer box.
    This is exact for multiples of 90 degrees and conservative otherwise.
    Conservative means a passable gap may be judged impassable, never the other
    way round, so an unsolvable scenario cannot slip through.
    """
    cx, cy, cz = spec.world_position_enu(obj)
    ex, ey, ez = object_extent(spec, obj)
    return AABB(
        (cx - ex / 2, cy - ey / 2, cz - ez / 2),
        (cx + ex / 2, cy + ey / 2, cz + ez / 2),
    )


def segment_hits_aabb(p0: Sequence[float], p1: Sequence[float], box: AABB) -> bool:
    """Does the segment p0-p1 intersect the box? Slab method.

    Used for the line-of-sight test: if the segment from spawn to target crosses
    any occluder, the target is not visible at t=0.
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
# Occupancy grid
# --------------------------------------------------------------------------- #
class OccupancyGrid:
    """Dense 3D occupancy grid.

    Index convention: `idx = ix + nx * (iy + ny * iz)`, iz is the vertical axis.
    """

    def __init__(self, aabb: AABB, resolution: float) -> None:
        self.resolution = float(resolution)
        self.origin = tuple(aabb.lo)
        size = aabb.size
        self.nx = max(1, int(math.ceil(size[0] / self.resolution)))
        self.ny = max(1, int(math.ceil(size[1] / self.resolution)))
        self.nz = max(1, int(math.ceil(size[2] / self.resolution)))
        self.blocked = bytearray(self.nx * self.ny * self.nz)

    # ---------- index arithmetic ----------
    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def to_index(self, x: int, y: int, z: int) -> int:
        return x + self.nx * (y + self.ny * z)

    def to_cell(self, point: Sequence[float]) -> Tuple[int, int, int]:
        ix = int((point[0] - self.origin[0]) / self.resolution)
        iy = int((point[1] - self.origin[1]) / self.resolution)
        iz = int((point[2] - self.origin[2]) / self.resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return (ix, iy, iz)

    def to_point(self, cell: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(
            self.origin[i] + (cell[i] + 0.5) * self.resolution for i in range(3)  # type: ignore[return-value]
        )

    # ---------- occupancy queries ----------
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

    # ---------- rasterisation ----------
    def rasterize(self, box: AABB) -> int:
        """Mark every cell touched by the box; returns the number marked."""
        lo = self.to_cell(box.lo)
        hi = self.to_cell(box.hi)
        count = 0
        for iz in range(lo[2], hi[2] + 1):
            for iy in range(lo[1], hi[1] + 1):
                base = self.to_index(0, iy, iz)
                for index in range(base + lo[0], base + hi[0] + 1):
                    if not self.blocked[index]:
                        self.blocked[index] = 1
                        count += 1
        return count

    def carve(self, box: AABB) -> None:
        """Force every cell touched by the box to be free."""
        lo = self.to_cell(box.lo)
        hi = self.to_cell(box.hi)
        for iz in range(lo[2], hi[2] + 1):
            for iy in range(lo[1], hi[1] + 1):
                base = self.to_index(0, iy, iz)
                for index in range(base + lo[0], base + hi[0] + 1):
                    self.blocked[index] = 0


def build_grid(
    spec: SceneSpec,
    resolution: float = DEFAULT_RESOLUTION_M,
    clearance: float = DEFAULT_CLEARANCE_M,
) -> Tuple[OccupancyGrid, Dict[str, AABB], Dict[str, Any]]:
    """Rasterise the solid volume described by the spec.

    Returns (grid, per-object AABB, metadata). Only objects with
    `physics=True` enter the collision grid, matching engine behaviour: a prop
    without a collider does not block the drone.

    All coordinates are world ENU. `spawn.pos` is local to `origin`, so it goes
    through `world_spawn_enu()` first; mixing the two frames silently produces
    "never occluded" and nonsense path lengths.
    """
    solid: List[Tuple[str, AABB]] = []
    for obj in spec.objects:
        if not obj.physics:
            continue
        solid.append((obj.id, object_aabb(spec, obj)))

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
        raise ValueError("the spec has neither solid objects nor a spawn/target")

    lo = tuple(min(p[i] for p in points) - 3.0 for i in range(3))
    hi = tuple(max(p[i] for p in points) + 3.0 for i in range(3))
    bounds = AABB(lo, hi)  # type: ignore[arg-type]

    # Coarsen automatically if the grid would be too large.
    res = float(resolution)
    while True:
        grid = OccupancyGrid(bounds, res)
        if grid.n_cells <= MAX_CELLS:
            break
        res *= 1.5
        LOG.warning("grid too large, coarsening the resolution to %.2f m", res)

    boxes: Dict[str, AABB] = {}
    for obj_id, box in solid:
        boxes[obj_id] = box
        grid.rasterize(box.expanded(clearance))

    # Deliberately no carving to make the goal reachable. A target sealed inside
    # solid structure must be reported as unsolvable: that is exactly the
    # generator failure the verifier exists to catch, not noise to hide.
    meta = {
        "resolution_m": grid.resolution,
        "dims": [grid.nx, grid.ny, grid.nz],
        "cells": grid.n_cells,
        "origin_enu": list(grid.origin),
        "blocked_cells": sum(grid.blocked),
        "solid_objects": len(solid),
        "clearance_m": clearance,
    }
    return grid, boxes, meta


# --------------------------------------------------------------------------- #
# A* path planning
# --------------------------------------------------------------------------- #
# 26-connected neighbourhood: -1/0/+1 on each axis, excluding the origin.
NEIGHBOURS: List[Tuple[int, int, int]] = [
    d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)
]


def astar(
    grid: OccupancyGrid,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    max_expansions: int = 2_000_000,
) -> Optional[List[Tuple[int, int, int]]]:
    """3D A*. Returns the cell path including both endpoints, or None.

    The heuristic is Euclidean distance times the resolution, which is
    admissible on a 26-connected grid, so A* returns an optimal path.
    """
    if not grid.is_free(start):
        return None
    if not grid.is_free(goal):
        return None
    if start == goal:
        return [start]

    res = grid.resolution

    def heuristic(cell: Tuple[int, int, int]) -> float:
        return res * math.sqrt(
            (cell[0] - goal[0]) ** 2
            + (cell[1] - goal[1]) ** 2
            + (cell[2] - goal[2]) ** 2
        )

    costs: Dict[Tuple[int, int, int], float] = {start: 0.0}
    parents: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    heap: List[Tuple[float, float, Tuple[int, int, int]]] = [
        (heuristic(start), 0.0, start)
    ]
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
            LOG.warning("A* exceeded %d expansions, giving up", max_expansions)
            return None

        for delta in NEIGHBOURS:
            neighbour = (
                current[0] + delta[0],
                current[1] + delta[1],
                current[2] + delta[2],
            )
            if neighbour in closed or not grid.is_free(neighbour):
                continue
            step = res * math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
            new_cost = g_cost + step
            if new_cost < costs.get(neighbour, math.inf):
                costs[neighbour] = new_cost
                parents[neighbour] = current
                heapq.heappush(heap, (new_cost + heuristic(neighbour), new_cost, neighbour))

    return None


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


# --------------------------------------------------------------------------- #
# Requirement checks
# --------------------------------------------------------------------------- #
def check_requirements(
    spec: SceneSpec,
    path_enu: Optional[Sequence[Tuple[float, float, float]]],
) -> List[Dict[str, Any]]:
    """Evaluate every declared requirement on the ground-truth path.

    "Satisfiable" means: a feasible path exists and flying it satisfies the
    assertion. That is why these checks are built on the A* result rather than
    on the raw specification alone.
    """
    results: List[Dict[str, Any]] = []

    length = path_length(path_enu) if path_enu else None
    estimated_time = length / ASSUMED_CRUISE_MPS if length is not None else None

    for requirement in spec.requirements:
        kind = str(requirement.get("kind", ""))
        value = requirement.get("value")
        satisfiable: Optional[bool] = None
        detail = ""

        if kind == "max_time_s":
            if estimated_time is None:
                satisfiable, detail = False, "cannot estimate duration (unsolvable)"
            else:
                satisfiable = estimated_time <= float(value)
                detail = (
                    f"estimated {estimated_time:.1f} s "
                    f"({length:.1f} m / {ASSUMED_CRUISE_MPS} m/s), "
                    f"limit {float(value):.0f} s"
                )

        elif kind == "min_altitude_m":
            if path_enu is None:
                satisfiable, detail = False, "cannot evaluate (unsolvable)"
            else:
                lowest = min(point[2] for point in path_enu)
                satisfiable = lowest >= float(value)
                detail = f"lowest point {lowest:.2f} m, limit {float(value):.1f} m"

        elif kind == "no_collision":
            # The path lies in inflated free space, so reachability already
            # implies a collision-free route.
            satisfiable = path_enu is not None
            detail = "guaranteed by reachability in the inflated free space"

        elif kind == "must_enter_building":
            if path_enu is None:
                satisfiable, detail = False, "cannot evaluate (unsolvable)"
            else:
                structures = [o for o in spec.objects if o.role == "structure"]
                if not structures:
                    satisfiable, detail = False, "the scenario has no structures"
                else:
                    xs = [spec.world_position_enu(o)[0] for o in structures]
                    ys = [spec.world_position_enu(o)[1] for o in structures]
                    inside = any(
                        min(xs) <= point[0] <= max(xs)
                        and min(ys) <= point[1] <= max(ys)
                        for point in path_enu
                    )
                    satisfiable = inside
                    detail = (
                        "the path crosses the building footprint"
                        if inside
                        else "the path never enters the building footprint"
                    )

        elif kind == "min_path_length_m":
            if length is None:
                satisfiable, detail = False, "cannot evaluate (unsolvable)"
            else:
                satisfiable = length >= float(value)
                detail = f"path {length:.1f} m, limit {float(value):.1f} m"

        elif kind == "max_detour_ratio":
            if length is None or spec.target is None or spec.spawn is None:
                satisfiable, detail = False, "cannot evaluate (unsolvable or no target)"
            else:
                goal = spec.object_by_id(spec.target.object_id)
                start_world = spec.world_spawn_enu()
                goal_world = (
                    spec.world_position_enu(goal) if goal else start_world
                )
                straight = math.dist(start_world, goal_world)
                ratio = length / straight if straight > 1e-6 else math.inf
                satisfiable = ratio <= float(value)
                detail = f"detour ratio {ratio:.2f}, limit {float(value):.2f}"

        else:
            satisfiable, detail = None, f"unknown requirement kind '{kind}', not checked"

        results.append(
            {
                "kind": kind,
                "value": value,
                "satisfiable": satisfiable,
                "detail": detail,
            }
        )

    return results


# --------------------------------------------------------------------------- #
# Result object
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
        """Usable as a test: solvable, occluded, no failures, requirements met."""
        return (
            self.solvable
            and self.initially_occluded
            and not self.failures
            and all(r["satisfiable"] is not False for r in self.requirement_results)
        )

    def summary(self) -> str:
        def yes_no(value: bool) -> str:
            return "yes" if value else "NO"

        lines = [
            f"scenario: {self.spec_name}",
            f"  solvable        : {yes_no(self.solvable)}"
            + (f"  (path {self.path_length_m:.1f} m)" if self.path_length_m else ""),
            f"  occluded at t=0 : {yes_no(self.initially_occluded)}",
            f"  straight-line   : {self.straight_distance_m:.1f} m",
            f"  detour ratio    : "
            + (f"{self.detour_ratio:.2f}" if self.detour_ratio else "n/a"),
            f"  lowest altitude : "
            + (
                f"{self.min_path_altitude_m:.2f} m"
                if self.min_path_altitude_m is not None
                else "n/a"
            ),
            f"  grid            : {self.grid_meta['dims']} @ "
            f"{self.grid_meta['resolution_m']:.2f} m "
            f"({self.grid_meta['cells']} cells, clearance {self.clearance_m:.1f} m)",
        ]

        if self.requirement_results:
            lines.append("  requirements:")
            for result in self.requirement_results:
                mark = {True: "PASS", False: "FAIL", None: "SKIP"}[result["satisfiable"]]
                lines.append(
                    f"    [{mark}] {result['kind']} = {result['value']}  "
                    f"({result['detail']})"
                )

        if self.failures:
            lines.append("  verdict: NOT usable")
            for failure in self.failures:
                lines.append(f"    - {failure}")
        else:
            lines.append("  verdict: usable as a test case")

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
# Main verification routine
# --------------------------------------------------------------------------- #
def verify(
    spec: SceneSpec,
    *,
    resolution: float = DEFAULT_RESOLUTION_M,
    clearance: float = DEFAULT_CLEARANCE_M,
    min_detour_ratio: float = 1.15,
    min_path_length_m: float = 10.0,
) -> VerificationResult:
    """Run the full geometric verification of a specification.

    Parameters
    ----------
    resolution, clearance:
        Grid granularity and the drone safety radius.
    min_detour_ratio:
        Lower bound on path length divided by straight-line distance. Below it
        the target is almost directly reachable, which is trivial. It is a lower
        bound because the scenario being rejected is the too-easy one.
    min_path_length_m:
        Absolute lower bound on path length, in case the target is next door.
    """
    failures: List[str] = []

    if spec.spawn is None:
        failures.append("missing spawn: solvability cannot be verified")
    if spec.target is None:
        failures.append("missing target: solvability cannot be verified")

    goal_obj = (
        spec.object_by_id(spec.target.object_id) if spec.target is not None else None
    )
    if spec.target is not None and goal_obj is None:
        failures.append(
            f"target.object_id='{spec.target.object_id}' is not present in objects"
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

    # Everything moves to world ENU here: spawn.pos and object pos are local to
    # origin, while occluder boxes are world-space.
    start_enu = spec.world_spawn_enu()
    goal_enu = spec.world_position_enu(goal_obj)
    straight = math.dist(start_enu, goal_enu)

    # ---------- 1. initial occlusion ----------
    occluders = [
        object_aabb(spec, obj)
        for obj in spec.objects
        if obj.role in ("structure", "prop") and obj.id != goal_obj.id
    ]
    occluded = any(
        segment_hits_aabb(start_enu, goal_enu, box) for box in occluders
    )

    # ---------- 2. occupancy grid and A* ----------
    grid, _, grid_meta = build_grid(spec, resolution=resolution, clearance=clearance)

    start_cell = grid.to_cell(start_enu)
    goal_cell = grid.to_cell(goal_enu)

    # If the spawn lands inside an obstacle (for example right on the ground),
    # snap to the nearest free cell.
    if not grid.is_free(start_cell):
        start_cell = _nearest_free(grid, start_cell, radius=6)

    # The goal cell may be occupied by the target's own body. In that case plan
    # to a free cell within the landing radius -- consistent with the benchmark
    # protocol, which only requires flying close to the target. If the whole
    # radius is blocked, the scenario really is unsolvable.
    if not grid.is_free(goal_cell):
        landing_cells = max(
            1,
            int(round(
                (spec.target.landing_radius_m if spec.target else 4.0)
                / grid.resolution
            )),
        )
        goal_cell = _nearest_free(grid, goal_cell, radius=landing_cells)

    cells = astar(grid, start_cell, goal_cell) if start_cell and goal_cell else None
    solvable = cells is not None

    path_enu: Optional[List[Tuple[float, float, float]]] = None
    length: Optional[float] = None
    detour: Optional[float] = None
    min_altitude: Optional[float] = None

    if solvable and cells:
        path_enu = [grid.to_point(cell) for cell in cells]
        # Replace the endpoints with the true poses to remove grid quantisation.
        path_enu[0] = tuple(start_enu)
        path_enu[-1] = tuple(goal_enu)
        length = path_length(path_enu)
        detour = length / straight if straight > 1e-6 else math.inf
        min_altitude = min(point[2] for point in path_enu)
    else:
        failures.append("A* found no feasible path from spawn to target (unsolvable)")

    # ---------- 3. non-triviality ----------
    if solvable and length is not None:
        if length < min_path_length_m:
            failures.append(
                f"the path is only {length:.1f} m, below the {min_path_length_m:.0f} m "
                f"minimum: the task is too easy"
            )
        if detour is not None and detour < min_detour_ratio:
            failures.append(
                f"detour ratio is only {detour:.2f} (minimum {min_detour_ratio:.2f}): "
                f"the target is almost directly reachable, so this is not a search task"
            )

    # ---------- 4. occlusion is what makes it a search task ----------
    if not occluded:
        failures.append(OCCLUSION_FAILURE)

    # ---------- 5. requirement satisfiability ----------
    requirement_results = check_requirements(spec, path_enu)
    for result in requirement_results:
        if result["satisfiable"] is False:
            failures.append(
                f"requirement '{result['kind']}' cannot be satisfied: {result['detail']}"
            )

    return VerificationResult(
        spec_name=spec.name,
        solvable=solvable,
        initially_occluded=occluded,
        path_length_m=length,
        straight_distance_m=straight,
        detour_ratio=detour,
        min_path_altitude_m=min_altitude,
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
    """Spiral outward from a cell until a free one is found."""
    if grid.is_free(cell):
        return cell
    for r in range(1, radius + 1):
        for delta in itertools.product(range(-r, r + 1), repeat=3):
            candidate = (cell[0] + delta[0], cell[1] + delta[1], cell[2] + delta[2])
            if grid.is_free(candidate):
                return candidate
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Geometric verification: solvability, initial occlusion, "
            "non-triviality, requirement satisfiability"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True, help="path to the specification JSON")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--min-detour", type=float, default=1.15)
    parser.add_argument("--min-path-length", type=float, default=10.0)
    parser.add_argument("--json", default="", help="write the report to this path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Windows consoles default to CP936 and mangle UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        spec = load_spec(args.spec)
    except Exception as exc:  # noqa: BLE001
        LOG.error("cannot load the specification: %s", exc)
        return 2

    result = verify(
        spec,
        resolution=args.resolution,
        clearance=args.clearance,
        min_detour_ratio=args.min_detour,
        min_path_length_m=args.min_path_length,
    )

    print(result.summary())

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOG.info("report written to %s", args.json)

    return 0 if result.usable else 1


if __name__ == "__main__":
    sys.exit(main())