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
│   ├── probe_assets.py      # probe which assets can be spawned at runtime
│   ├── check_llm.py         # verify the LLM endpoint before compiling anything
│   └── batch_compile.py     # batch requests, acceptance rate, failure taxonomy
├── specs/
│   └── fire_floor2.json     # hand-written reference scenario
├── tests/
│   └── test_offline.py      # no-engine test suite (spec, coords, catalog, template)
├── requirements.txt
├── .gitattributes           # LF everywhere, so Windows and Linux agree
└── .gitignore
```

---

## Credentials

The LLM compiler needs an API key, a base URL and a model name. There are two
ways to provide them, and neither puts a secret in the repository.

**Option A — a `.env` file (recommended)**

```bash
cp .env.example .env
$EDITOR .env          # fill in LLM_API_KEY
python scripts/check_llm.py
```

`.env` is git-ignored. Real environment variables always take precedence over
its contents, so a file can hold defaults while a shell export overrides them
for one run.

**Option B — exported environment variables**

```bash
export LLM_API_KEY=sk-xxxx
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
python scripts/check_llm.py
```

`src/llm_env.py` is the only module that reads these, and `scripts/check_llm.py`
makes one five-token request to confirm the endpoint works before any
compilation is attempted. It turns an opaque failure inside the compiler into a
specific diagnosis: bad key, wrong base URL, unknown model, quota, or an
unreachable network.

Never commit a key. If one leaks, revoke it in the provider's console.

## Batch experiments

RQ2 is a quantitative claim — "generation can be made verifiable" — so it needs
numbers, not anecdotes. `scripts/batch_compile.py` runs a request set through the
production compile loop and reports acceptance rate, retries and a failure
taxonomy.

```bash
# quick sanity batch
python scripts/batch_compile.py --limit 3 --tag smoke

# the paper run
python scripts/batch_compile.py --trials 3 --tag full

# the ablation: identical model and prompts, geometric gate removed
python scripts/batch_compile.py --trials 3 --no-verify --tag ablation

# the control: deterministic template instead of a model
python scripts/batch_compile.py --template --tag template
```

It writes `runs/batch_<timestamp>_<tag>/` containing every accepted spec, a
`results.csv` (one row per request×trial), `summary.json` and `failures.log`.
`runs/` is git-ignored, so a batch never pollutes the repository.

The four runs above give the comparison table the evaluation needs: with and
without verification, and model versus template, on the same request set.

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
- the compile loop retrying after a geometric failure and feeding back a
  concrete diagnostic, giving up explicitly when it never converges, and
  accepting the same bad scenario when `--no-verify` is set (the ablation)
- credential parsing: quoted values, `export` prefix, trailing comments, a
  Windows BOM, environment-over-file precedence, and key masking

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
| Verifier wired into the retry loop, with ablation flag | done |
| Credential handling via environment or git-ignored `.env` | done |
| Batch harness: acceptance rate, retries, failure taxonomy | done |
| End-to-end run with a real model | done (1 scenario, first attempt) |
| Episode runner + metrics (SR / OSR / SPL / NE) | next |
| Difficulty-parameter sweep | planned |

### The ablation that measures what verification buys

`compile_request(..., verify_geometry=False)` runs the identical model and
prompts with the geometric gate removed. Comparing the two conditions is the
experiment behind RQ2:

```bash
# with verification (default)
python src/llm_compiler.py --request "..." --out specs/with_verify.json

# without it
python src/llm_compiler.py --request "..." --out specs/no_verify.json --no-verify
```

Acceptance rate, retry counts and the failure taxonomy across a batch of
requests, with and without the gate, are the numbers to report.

---

## References

- OpenUAV / TravelUAV — arXiv:2410.07087 · [project page](https://prince687028.github.io/OpenUAV/)
- AirSim — arXiv:1705.05065
- UAV-ON — arXiv:2508.00288
- Holodeck — arXiv:2312.09067 · SceneCraft — arXiv:2403.01248 · SceneWeaver — arXiv:2509.20414

---

## Licence

See `LICENSE`.

## CN
这个项目的核心目标是：把自然语言测试需求自动编译成一个可在 AirSim 中运行的无人机搜索场景，并在真正生成场景前验证它是否可行、是否具有足够的搜索难度。

整体 pipeline 可以概括为：

┌──────────────────────────────────────────┐
│ 自然语言需求                             │
│                                          ├─────────┐
└──────────────────────────────────────────┘         │
                                                     │
                                                     │
┌──────────────────────────────────────────┐         │
│ LLM Compiler                             │         │
│                                          ├◄────────┘
│                                          ├───────────┐
│                                          ├◄──────────╪─────┐
└──────────────────────────────────────────┘           │     │
                                                       │     │
                                                       │     │
┌──────────────────────────────────────────┐           │     │
│ JSON 场景规格 SceneSpec                  │           │     │
│                                          ├◄──────────┘     │
│                                          ├─────────────┐   │
└──────────────────────────────────────────┘             │   │
                                                         │   │
                                                         │   │
┌──────────────────────────────────────────┐             │   │
│ 结构与语义校验                           │             │   │
│                                          ├◄────────────┘   │
│                                          ├───────────────┐ │
└──────────────────────────────────────────┘               │ │
                                                           │ │
                                                           │ │
┌──────────────────────────────────────────┐               │ │
│ 几何验证                                 │               │ │
│                                          ├◄──────────────┘ │
│                                          ├──失败───────────┘
│                                          ├──通过─────────────┐
└──────────────────────────────────────────┘                   │
                                                               │
                                                               │
┌──────────────────────────────────────────┐                   │
│ AirSim 场景构建                          │                   │
│                                          ├◄──────────────────┘
│                                          ├─────────────────────┐
└──────────────────────────────────────────┘                     │
                                                                 │
                                                                 │
┌──────────────────────────────────────────┐                     │
│ 生成多个视角图像                         │                     │
│                                          ├◄────────────────────┘
│                                          ├───────────────────────┐
└──────────────────────────────────────────┘                       │
                                                                   │
                                                                   │
┌──────────────────────────────────────────┐                       │
│ 保存 spec / verification / render report │                       │
│                                          ├◄──────────────────────┘
└──────────────────────────────────────────┘

## 1. 输入：自然语言需求或已有 JSON

入口主要是 scripts/generate_and_render.py (scripts/generate_and_render.py)。

它支持三种输入方式：

- --request "..."：输入自然语言，由 LLM 生成场景；
- --offline：不调用 LLM，使用确定性的模板生成；
- --spec path.json：直接读取已有场景规格。

例如：

Test a drone searching for a survivor inside a burning two-storey apartment.

该需求最终要变成一个结构化 JSON，主要包括：

- 场景名称和随机种子；
- 世界原点；
- 场景物体；
- 每个物体的位置、旋转、缩放和物理属性；
- 目标物体；
- 无人机初始位置；
- 时间、高度等任务约束；
- 给无人机的任务说明。

这些数据结构定义在 src/scene_spec.py (src/scene_spec.py)。

## 2. 坐标系统统一为 ENU

项目在规格文件中使用 ENU 坐标：

- x：向东；
- y：向北；
- z：向上；
- yaw：从东开始逆时针计算。

AirSim 使用 NED 坐标：

- x：向北；
- y：向东；
- z：向下。

因此所有转换集中在 scene_spec.py：

enu_to_ned()
ned_to_enu()
yaw_enu_to_ned()

这一点很关键，因为场景生成、几何验证和 AirSim 实际摆放必须使用同一套空间语义。物体的本地位置会先加上 origin，转换成世界 ENU，再交给 AirSim。

## 3. LLM 编译自然语言需求

核心逻辑在 src/llm_compiler.py (src/llm_compiler.py) 的 compile_request()。

LLM 并不直接生成 Unreal/AirSim 代码，而是生成受约束的 JSON 配置。编译过程有多个 gate：

### 第一层：Prompt 约束

system_prompt() 和 user_prompt() 会告诉模型：

- JSON schema；
- ENU 坐标约定；
- 可用资产名称；
- 角色类型；
- 目标、出生点和任务要求；
- 物体数量等限制。

可用资产由 src/asset_catalog.py (src/asset_catalog.py) 提供。

### 第二层：JSON 解析

extract_json() 从模型输出中提取 JSON。

如果模型输出了自然语言、错误 JSON 或 Markdown 代码块，当前尝试会失败，然后把错误反馈给模型。

### 第三层：结构校验

parse_spec() 会检查：

- version 是否正确；
- 场景名称是否存在；
- 物体列表是否为空；
- 物体 ID 是否重复；
- 坐标、缩放是否为三个数字；
- scale 是否大于 0；
- role 是否属于规定集合；
- target 是否引用了真实物体；
- requirements 是否为合法数组。

错误会被收集起来一次性反馈，而不是遇到第一个错误就停止。

### 第四层：语义和资产校验

validate_all() 继续检查：

- 资产是否在白名单中；
- 物体数量是否合理；
- spawn 高度是否合理；
- target 是否和 spawn 太近；
- physics 物体是否满足场景要求；
- 任务是否满足基本可执行条件。

如果失败，模型收到具体错误列表并重新生成完整 JSON。

### 第五层：几何验证

如果开启 verify_geometry=True，会调用 src/verifier.py (src/verifier.py) 的 verify()。

如果场景不能形成有效搜索任务，也会反馈给模型重写。最多重试 max_attempts 次，默认是 3 次。

## 4. 几何验证：判断场景是否真的可用

这是项目区别于普通“LLM 生成 3D 场景”的核心部分。

verifier.verify() 主要执行以下步骤。

### 4.1 初始视线遮挡检查

项目把 structure 和 prop 物体转换成 AABB 包围盒，然后检查：

spawn → target

之间的直线是否穿过障碍物。

如果目标从出生点一眼就能看到，场景会被判定为过于简单。

### 4.2 建立三维占据栅格

build_grid() 将场景离散成 3D occupancy grid：

- 障碍物所在单元格标记为 blocked；
- 加入无人机安全 clearance；
- 目标附近允许寻找可降落单元格。

### 4.3 A* 路径搜索

astar() 在三维栅格上寻找 spawn 到 target 的可行路径。

它验证两件事：

- 是否存在路径；
- 路径是否需要绕行。

项目会计算：

detour_ratio = 实际路径长度 / 直线距离

如果路径太短，或者 detour ratio 太小，场景会被判定为“几乎直达”，不符合搜索任务要求。

### 4.4 任务约束验证

check_requirements() 检查规格中的任务要求，例如：

- 最大时间；
- 最低飞行高度；
- 最大路径长度；
- 其他可计算约束。

最终返回 VerificationResult，包括：

- 是否可达；
- 是否初始遮挡；
- 路径长度；
- 直线距离；
- detour ratio；
- 最低路径高度；
- 失败原因列表。

只有 result.usable == True 才能进入 AirSim。

## 5. AirSim 场景构建

场景通过 src/scene_builder.py (src/scene_builder.py) 写入 AirSim。

stage_build() 的步骤是：

1. 连接 AirSim；
2. 调用 simListAssets() 获取当前环境支持的资产；
3. 检查规格中使用的 asset 是否存在；
4. 删除旧场景中带有指定 prefix 的对象；
5. 对每个 SceneObject 调用 simSpawnObject()；
6. 设置：
   - AirSim pose；
   - scale；
   - physics；
   - 对象名称；
7. 读取对象实际 pose，检查是否发生明显位置漂移。

物体的位置处理过程是：

局部 ENU 坐标
    ↓ 加 origin
世界 ENU 坐标
    ↓ ENU → NED
AirSim Pose
    ↓
simSpawnObject()

无人机初始位置也通过 move_drone_to_spawn() 转换后设置到 AirSim。

## 6. 渲染和图像采集

场景构建完成后，stage_render() 会生成三个固定视角：

- 01_spawn_view：从声明的出生点观察；
- 02_approach：从斜前方观察；
- 03_overhead：从上方观察。

每个视角会：

1. 计算相机位置；
2. 计算相机朝向；
3. 将无人机瞬移到该位置；
4. 调用 simGetImages()；
5. 获取多个摄像头画面；
6. 将图像写入：

runs/<run>/images/

同时生成：

- spec.json：最终场景规格；
- verification.json：几何验证结果；
- render_report.json：图像和视角信息。

## 7. 批量实验 pipeline

另一个重要入口是 scripts/batch_compile.py (scripts/batch_compile.py)。

它用于研究和统计，不负责 AirSim 渲染。

批量流程是：

多个自然语言请求
    ↓
重复调用 compile_request()
    ↓
记录每个请求用了几次 LLM 尝试
    ↓
记录接受率、失败类型、路径长度、detour ratio
    ↓
输出 CSV / JSON / failures.log

它支持：

- LLM 条件；
- deterministic template 条件；
- 关闭几何验证的 ablation；
- 每个请求多次 trial；
- 失败原因分类。

输出通常包括：

results.csv
summary.json
failures.log
specs/*.json

失败类型包括：

- JSON 解析失败；
- 结构验证失败；
- 资产幻觉；
- 无可行路径；
- 没有遮挡；
- 场景太简单；
- 任务约束无法满足。

## 8. 一次完整运行的实际顺序

以 generate_and_render.py 为例，完整执行顺序是：

1. 读取自然语言 request
2. 读取 .env 中的 LLM 配置
3. 调用 LLM
4. 提取 JSON
5. parse_spec()
6. 检查资产和语义规则
7. verifier.verify()
8. 失败则把具体问题反馈给 LLM 重试
9. 保存 spec.json
10. 连接 AirSim
11. 检查 simListAssets()
12. 清理旧对象
13. 生成新对象
14. 检查生成结果
15. 将无人机放到多个观察点
16. 采集图像
17. 保存 verification 和 render report

所以这个项目真正实现的是一条闭环：

自然语言任务
→ 结构化场景计划
→ 结构/语义检查
→ 物理几何可行性检查
→ AirSim 场景实例化
→ 图像采集与实验记录

核心设计思想是：LLM 只负责提出场景计划，规则验证器负责把计划约束成可执行、可重复、具有搜索难度的测试场景。