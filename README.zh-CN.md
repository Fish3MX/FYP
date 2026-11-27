# 大模型驱动的无人机测试场景生成

[English](README.md) | [中文](README.zh-CN.md)

把一句自然语言的测试需求，变成 Unreal Engine / AirSim 里可飞行、可验证的无人机测试场景。

---

## 项目做什么

| 层 | 职责 | 在哪个文件 |
|---|---|---|
| 大模型 | 把需求写成一份场景规格书（JSON） | `src/llm_compiler.py` |
| 程序 | 校验规格书，并做几何验证 | `src/verifier.py` |
| 引擎 | 把物体真正生成到场景里，渲染出图 | `src/scene_builder.py` |

**模型不生成 3D 模型。** 场景由模拟器自带的资产拼装而成。

---

## 仓库结构

```
Project/
├── src/                     核心模块
│   ├── scene_spec.py            规格书格式、校验、ENU<->NED 转换
│   ├── scene_profile.py         某个场景提供什么（端口、相机、资产）
│   ├── asset_catalog.py         给模型的语义资产词表
│   ├── llm_env.py               从环境变量或 .env 读 API 凭证
│   ├── llm_compiler.py          需求 -> 规格书，含校验-反馈重试
│   ├── verifier.py              可解性、遮挡、难度、指标可满足性
│   └── scene_builder.py         规格书 -> AirSim 物体
├── scripts/                 入口
│   ├── main.py                  主流程：编译 -> 验证 -> 搭建 -> 渲染
│   ├── batch_compile.py         生成质量统计（接受率、失败类型）
│   ├── inspect_run.py           图像统计 + 资产摘要
│   └── check_llm.py             接口自检
├── tools/                   环境工具
│   ├── onboard_scene.py         接入新场景：体检、写档案、生成最小规格
│   ├── hello_uav.py             最小飞行自检：起飞、拍照、降落
│   └── calibrate_assets.py      标定资产真实尺寸
├── specs/                   场景规格数
├── scenes/settings/         AirSim 配置文件
├── tests/                   离线测试
└── runs/                    实验产物（git 忽略）
```

| 文件 | 解释 |
|---|---|
| `scripts/main.py` | 平时只需要这一条命令。 |
| `tools/onboard_scene.py` | 每接一个新场景跑一次，产出所有后续步骤要读的档案。 |
| `src/verifier.py` | 在任何智能体运行之前，判断这份规格书能不能当测试用。 |
| `src/scene_profile.py` | 让代码不绑定任何一张具体地图。 |

---

## 依赖

- **仿真**：OpenUAV / TravelUAV，仅使用 `closeloop_envs`（目前用 `ModernCityMap`）。
- **Python**：3.10 环境，需要 `airsim`、`numpy`、`pillow`。
- **硬件**：NVIDIA GPU；离屏渲染即可，不需要显示器。

```bash
pip install -r requirements.txt
python -c "import airsim; print('airsim ok')"
```

---

## 配置凭证

凭证不写在代码里。可以导出环境变量，也可以复制模板：

```bash
cp .env.example .env
chmod 600 .env          # 填入 LLM_API_KEY
python scripts/check_llm.py
```

`LLM_BASE_URL` 和 `LLM_MODEL` 默认指向 DeepSeek，任何 OpenAI 兼容接口都能用。
`.env` 已被 git 忽略；真实环境变量优先于文件。

---

## 接入 OpenUAV

### OpenUAV 自身环境需要改两处

这两处改的是 OpenUAV 的 `requirements.txt` 和配置，不属于本仓库。

| 改动 | 原因 |
|---|---|
| 注释掉 `flash-attn==2.5.9.post1` | 它给 OpenUAV 自带的模型用，本项目从不加载该模型 |
| 锁定 `msgpack==0.6.2` | `msgpackrpc` 传了 `encoding=` 参数，msgpack 1.0 已删除；装 1.x 会直接 import 失败 |

锁版本有副作用：0.6.2 下 Python 的 `str` 会打包成 *raw*，到对端变成 `bytes`，而服务器拿它去和 `str` 类型的字典键比较。

**给 `AirVLNSimulatorServerTool.py` 打补丁（约第 591 行）**

不打这个补丁，服务器匹配不到收到的场景名，会抛 `KeyError: can not find scene file`。

```python
for item in scen_id_gpu_list:                      # 第 591 行
    if isinstance(item[0], bytes):                 # 新增
        item[0] = item[0].decode('utf-8')          # 新增
```

### 启动场景

`OPENUAV_ROOT` 换成你自己的路径，每台机器不同。

```bash
export OPENUAV_ROOT=/path/to/OpenUAV     # <-- 改这里
cd $OPENUAV_ROOT/closeloop_envs

bash ./ModernCityMap.sh \
  -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter=0 \
  -settings=$OPENUAV_ROOT/airsim_plugin/settings/30001/settings.json
```

| 参数 | 含义 |
|---|---|
| `-RenderOffscreen` | 不开窗口直接渲染，SSH / 无桌面也能跑 |
| `-NoSound`、`-NoVSync` | 关闭声音和垂直同步 |
| `-GraphicsAdapter=0` | 使用 0 号 GPU |
| `-settings=` | AirSim 配置，决定 `ApiServerPort`、机型名和相机 |

**客户端连的是 settings 文件里的 `ApiServerPort`（这里是 30001），不是启动器的端口。**
这份 settings 由 OpenUAV 的 server tool 生成，其中 `LockStep: true`，异步飞行指令会阻塞。本流程只做"传送 + 拍照"，所以不受影响；要做飞行测试需要换成 `LockStep: false` 的配置。

### 验证连接（可选但建议）

```bash
python tools/onboard_scene.py --port 30001 --scene openuav_moderncity \
    --emit-spec specs/onboard_smoke.json
```

六项检查：导入、连接、机型、资产、生成、成像。产出 `runs/onboard/profile_<scene>.json`。

---

## 闭环流程

```bash
# 1. 每个新场景跑一次：体检并写档案
python tools/onboard_scene.py --port 30001 --scene openuav_moderncity \
    --emit-spec specs/onboard_smoke.json

# 2. 搭建并渲染（不用模型，也不做几何验证）
python scripts/main.py --spec specs/onboard_smoke.json \
    --scene-profile runs/onboard/profile_openuav_moderncity.json \
    --no-verify --out runs/render_1

# 3. 不用人眼看图，直接看统计
python scripts/inspect_run.py --run runs/render_1

# 4. 换成大模型来生成
python scripts/main.py --request "a drone searching for a survivor in a burning two-storey apartment" \
    --scene-profile runs/onboard/profile_openuav_moderncity.json \
    --no-verify --out runs/render_2

# 5. 批量跑需求，统计生成质量
python scripts/batch_compile.py --trials 3 --tag main
```

| 命令 | 产出 |
|---|---|
| `onboard_scene.py` | `profile_*.json`、`onboard_report.json`、一张截图 |
| `main.py` | `spec.json`、`render_report.json`、`images/<机位>/<视角>.png` |
| `inspect_run.py` | 亮度/对比度表，标出全黑或纯色帧 |
| `batch_compile.py` | `results.csv`、`summary.json`、失败类型分布 |

在成品地图上必须加 `--no-verify`：验证器只认识规格书里写出的物体，不认识地图自带的建筑。

---

## 其他说明

- `runs/` 和 `*.npy` 已被 git 忽略，实验产物不会进仓库。
- `tests/test_offline.py` 不需要引擎、不需要 GPU、不需要联网：`python tests/test_offline.py`。
- `specs/fire_floor2.json` 是早期给另一张地图（CityStreet）写的参照场景；测试套件仍用它当固定样本。
- `src/asset_catalog.py` 目前描述的是 CityStreet 的资产，后续会按 OpenUAV 的档案重建。

## 许可证

见 `LICENSE`。