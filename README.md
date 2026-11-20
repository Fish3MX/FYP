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
│   └── scene_builder.py     # spec -> AirSim objects (spawn, cleanup, readback)
├── scripts/
│   ├── hello_uav.py         # minimal loop: connect, take off, fly, capture, land
│   └── probe_assets.py      # probe which assets can be spawned at runtime
├── specs/
│   └── fire_floor2.json     # hand-written reference scenario
├── docs/
│   ├── proposal_en.md       # research proposal (English)
│   ├── proposal_zh.md       # research proposal (Chinese)
│   └── proposal_en.docx
└── requirements.txt
```

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
| LLM compiler: request → spec | next |
| Geometric verifier: solvability, occlusion, non-triviality | planned |
| Episode runner + metrics (SR / OSR / SPL / NE) | planned |
| Difficulty-parameter sweep | planned |

---

## 中文速览

- **目标**：让大模型把"一句话测试需求"变成 AirSim 里可飞行的测试场景。
- **分工**：LLM 负责规划（决定放什么、放哪里），AirSim 负责施工（渲染和物理）。
- **为什么不是"大模型直接生成 3D"**：生成式 mesh 没有碰撞体、没有真实尺度，
  飞不进去。所以场景是**编译**出来的，不是生成出来的 —— 编译才能验证。
- **当前进度**：控制链路已打通，规格书与构建器已就绪，下一步接 LLM。
- **跑起来**：见上面 Quick start 的四步。

---

## References

- OpenUAV / TravelUAV — arXiv:2410.07087 · [project page](https://prince687028.github.io/OpenUAV/)
- AirSim — arXiv:1705.05065
- UAV-ON — arXiv:2508.00288
- Holodeck — arXiv:2312.09067 · SceneCraft — arXiv:2403.01248 · SceneWeaver — arXiv:2509.20414
- Full citation list in `docs/proposal_en.md`.

---

## Licence

See `LICENSE`.