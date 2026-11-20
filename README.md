# FYP — Task-Driven, Verifiable 3D Scene Generation for UAV Simulation Benchmarking

Automatic generation of flyable UAV test scenarios from natural-language test
requests. A large language model turns a request such as *"test the drone searching
for a survivor in a burning two-storey building"* into a structured, schema-validated
**scene specification**; a builder then instantiates that specification inside
Unreal Engine / AirSim, and a geometric verifier checks that the resulting scenario
is reachable, non-trivial, and actually a search task.

The contribution is **not** "an LLM can generate a 3D scene". It is that scenario
generation can be made *verifiable, reproducible, and domain-parameterisable* —
which is what an evaluation benchmark needs and what end-to-end 3D world models
cannot provide.

---

## Architecture

```
 natural-language test request
              │
              ▼
      ┌───────────────┐   schema-validated JSON
      │  LLM compiler │ ────────────────────┐
      └───────────────┘                     │
              │                             ▼
              │                   ┌──────────────────┐
              │                   │  rule verifier   │──► violation reports
              │                   └──────────────────┘        │
              │                             │ pass            │ feedback
              │                             ▼                 │
              │                   ┌──────────────────┐        │
              │                   │ task instantiator│ ◄──────┘
              │                   └──────────────────┘
              │                             │ episode
              ▼                             ▼
       scene builder ──────────► UE / AirSim ──► agent ──► metrics + report
```

The pipeline is split into a **semantic layer** (spec generation + validation +
verification; pure CPU, runs on a laptop) and an **instantiation layer**
(AirSim-bound; runs on the GPU workstation). Only two interfaces cross the
boundary, so the semantic half is fully testable without an engine.

---

## Repository layout

```
Project/
├── src/
│   ├── scene_spec.py        # spec dataclasses, validation, ENU<->NED conversion
│   ├── asset_catalog.py     # semantic asset vocabulary (what the LLM may use)
│   ├── llm_compiler.py      # natural language -> spec, with validate-and-retry
│   ├── verifier.py          # geometry: solvability, occlusion, non-triviality
│   └── scene_builder.py     # spec -> AirSim objects (spawn, cleanup, readback)
├── scripts/
│   ├── hello_uav.py         # minimal loop: connect, take off, fly, capture, land
│   └── probe_assets.py      # probe which assets can be spawned at runtime
├── specs/
│   └── fire_floor2.json     # hand-written reference scenario
├── tests/
│   └── test_offline.py      # no-engine test suite (spec, coords, catalog, template)
├── requirements.txt
├── .gitattributes           # LF everywhere, so Windows and Linux agree
└── .gitignore
```

---

## Testing

Half of this pipeline is pure data manipulation and must not require a simulator
to check. That half has a test suite:

```bash
python tests/test_offline.py          # or: python -m unittest tests.test_offline -v
```

It covers, with no engine, no GPU and no AirSim connection:

- ENU→NED conversion and yaw mapping, including all four cardinal directions
- spec validation rejecting every malformed case that a generator can produce
- the asset catalogue's internal consistency and prompt completeness
- the reference scenario passing all checks
- the offline template producing valid specs for six different requests
- spec ↔ dict round-tripping (catches silently dropped fields)
- JSON extraction from messy model output (prose, fences, trailing commas)
- segment/AABB intersection used by the occlusion test
- a sealed room being judged **unsolvable** and the same room **solvable** once a
  4 m doorway is cut into it
- requirement assertions that actually fail when they should (`max_time_s=1`)
- every template scenario being geometrically usable, not merely schema-valid

Run it after every change to the spec schema, the catalogue, the compiler or the
verifier, before spending time on the workstation.

## Verification

`src/verifier.py` answers four questions about a spec **before any agent flies it**,
using only geometry — no engine, no rendering, a few hundred milliseconds:

| Check | Method | Rejects |
|---|---|---|
| Solvable | 3D occupancy grid from solid AABBs, inflated by a drone radius, then A* | a target sealed behind walls |
| Initially occluded | segment/AABB intersection from spawn to goal | a goal visible from take-off (not a search task) |
| Non-trivial | path length and detour ratio vs straight-line distance | a goal one straight hop away |
| Requirements satisfiable | each `requirements[]` entry evaluated on the ground-truth path | impossible assertions such as `max_time_s: 1` |

```bash
python src/verifier.py --spec specs/fire_floor2.json
python src/verifier.py --spec specs/fire_floor2.json --json report.json
```

Two conventions this enforces on every spec:

**A ground slab is mandatory.** A floor modelled as a floating slab leaves the
space underneath it open, so a drone can fly *under* the building and reach a
target from below — the room is then not sealed in 3D even though it looks sealed
in plan. Specs must include an explicit ground base.

**Doorways must survive inflation.** A 2 m door becomes 0.8 m after a 0.6 m
clearance is applied and can vanish at 0.5 m grid resolution. Openings should be
at least 4 m wide, or the clearance reduced deliberately.

> The verifier is the reason scene generation here is worth calling research:
> it converts "the model produced a scene" into "the scene is a usable test".

---

## Coordinate convention

This is the single most bug-prone part of the project, so it is stated once and
enforced in exactly one module.

| Layer | Frame | Axes | Yaw |
|---|---|---|---|
| **Spec files (JSON)** | ENU | x = east, y = north, z = **up** | degrees, CCW from east |
| **AirSim** | NED | x = north, y = east, z = **down** | degrees, CW from north |
| **Unreal Engine** | – | centimetres, z up | – |

`scene_spec.enu_to_ned()` and `scene_spec.yaw_enu_to_ned()` are the **only**
places a conversion may occur. No other module may compute one.

Specs are written in ENU because "5 m north, 3 m east, 2 m up" is what a language
model produces naturally; `z = -2` is what it gets wrong.

---

## Requirements

- Unreal Engine 4.27 scene packaged with the AirSim plugin
- Python 3.9+ with `airsim` and `numpy`
- The AirSim API port is read from the scene's `settings.json` (`ApiServerPort`);
  the reference deployment uses `30100`

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Start the simulator

Launch the packaged UE scene. It must print the vehicle name (e.g. `Drone_1`)
before Python can connect.

```bash
bash CleanCityStreet.sh -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter=0 \
  --settings=/path/to/settings/30100/settings.json
```

Verify the API port is listening:

```bash
ss -lntp | grep 30100
```

### 2. Verify the control link

```bash
python scripts/hello_uav.py --connect-only -v   # connection only
python scripts/hello_uav.py                     # full flight + 4-view capture
```

Outputs `shots/cam*_rgb.png`, `shots/cam*_depth.npy`, and `shots/metadata.json`.

### 3. Probe runtime asset availability

```bash
python scripts/probe_assets.py --spawn --json assets_report.json
```

Prints how many assets can be spawned at runtime and emits a verdict
(`runtime_placement_viable` / `limited` / `unavailable`). This determines whether
the runtime-placement route is viable on a given scene.

### 4. Build a scenario from a spec

```bash
python src/scene_spec.py specs/fire_floor2.json                  # validate only
python src/scene_builder.py --spec specs/fire_floor2.json --dry-run
python src/scene_builder.py --spec specs/fire_floor2.json --move-drone
```

`--dry-run` performs the asset pre-check without touching the scene: every
`asset` in the spec must appear in `simListAssets()`.

### 5. Compile a request into a spec

No API key needed to try the pipeline end to end:

```bash
python src/asset_catalog.py                      # browse the allowed vocabulary
python src/llm_compiler.py --request "fire in a two-storey apartment, searching for a survivor" \
       --offline --out specs/generated_fire.json  # deterministic template
python src/scene_builder.py --spec specs/generated_fire.json --dry-run
```

With a model (any OpenAI-compatible endpoint):

```bash
export LLM_API_KEY=sk-xxxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat

python src/llm_compiler.py --request "test the drone searching for a survivor \
       inside a burning two-storey apartment" --out specs/generated_fire.json
```

The compiler generates, validates, and feeds every violation back to the model
for up to `--max-attempts` rounds. A spec only reaches disk once it passes
schema validation, the asset whitelist, and the scenario-sanity checks
(structure has collision, target exists and is unique, drone starts airborne and
far enough away).

---

## Design decisions

**Hand-written specs before LLM-generated ones.** The spec → scene half is
debugged with JSON written by hand, so that when the LLM is introduced, a failure
can be attributed to one half of the pipeline rather than "something in the middle".

**Generated objects are namespaced.** Everything the builder creates is named
`gen_<spec_name>_<object_id>`, so cleanup only ever removes objects this tool
created — the ~6.7k objects of the base city scene are never touched.

**Physics is opt-in per object.** Structure must be spawned with
`physics_enabled=True` or the drone flies through walls; props may skip it.

**Asset availability is checked before anything is built.** A spec referring to
an asset absent from the scene fails fast with the full list of missing names,
instead of half-building a scene and reporting 15 separate spawn errors.

---

## Status

| Milestone | State |
|---|---|
| AirSim control link verified on remote workstation | done |
| 4-view RGB + metric depth capture with pose metadata | done |
| Runtime asset availability probed (487 assets, spawn confirmed) | done |
| Spec schema, validation, ENU→NED conversion | done |
| Spec → AirSim builder with pre-check and readback | done |
| Semantic asset catalogue constraining the generator | done |
| LLM compiler: request → spec, validate-and-retry loop | done |
| Deterministic template baseline (no-API comparison) | done |
| Geometric verifier: solvability, occlusion, non-triviality | done |
| Episode runner + metrics (SR / OSR / SPL / NE) | planned |
| Difficulty-parameter sweep | planned |

---

## References

- OpenUAV / TravelUAV — arXiv:2410.07087 · [project page](https://prince687028.github.io/OpenUAV/)
- AirSim — arXiv:1705.05065
- UAV-ON — arXiv:2508.00288
- Holodeck — arXiv:2312.09067 · SceneCraft — arXiv:2403.01248 · SceneWeaver — arXiv:2509.20414

---

## Licence

See `LICENSE`.