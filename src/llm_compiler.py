#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_compiler.py —— 把自然语言测试需求编译成场景规格书

核心思路
--------
不是"让大模型生成一个 3D 世界"，而是"让大模型写一份受约束的配置文件"。
配置文件的格式由 `scene_spec.py` 定义，资产名由 `asset_catalog.py` 限定。
模型只负责**规划**（放什么、放哪里、任务是什么），不负责渲染。

三层约束，从软到硬
------------------
1. **提示词层**：告诉模型坐标系、字段含义、可用资产、数量上限。
   这是软约束，模型可能违反。
2. **校验层**：`scene_spec.parse_spec()` 做结构校验 —— 字段缺失、类型错误、
   id 重复、target 指向不存在的物体，全部拒绝。
3. **反馈层**：校验失败时，把**具体问题列表**发回给模型让它重写，
   最多 N 轮。这是本模块最关键的设计：模型不是一次成型，而是被
   验证器"逼"到正确。

此外还做一次**资产白名单检查**：规格书里出现的任何 asset 名必须来自
`asset_catalog`，否则视为模型幻觉，同样进入重写循环。

离线模式
--------
没有 API key 时，`--offline` 会用一个确定性模板生成规格书。这不是补救措施，
而是有用的对照基线：论文里"LLM 编译器"的效果应当与"模板生成"对比，
否则无法说明模型到底贡献了什么。

用法
----
    # 需要 API key（任意 OpenAI 兼容端点）
    export LLM_API_KEY=sk-xxxx
    export LLM_BASE_URL=https://api.deepseek.com/v1      # 或其它
    export LLM_MODEL=deepseek-chat

    python src/llm_compiler.py \
        --request "测试无人机在着火的二层公寓里搜索伤员" \
        --out specs/generated_fire.json

    # 无 API，用模板跑通全链路
    python src/llm_compiler.py --request "着火的二层公寓" \
        --offline --out specs/generated_fire.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from asset_catalog import prompt_text, whitelist
from scene_spec import SceneSpec, SpecError, parse_spec

LOG = logging.getLogger("llm_compiler")

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_ATTEMPTS = 3

# 规格书里允许出现的字段，用于提示词
SCHEMA_BRIEF = """
{
  "version": "0.1",
  "name": "snake_case_scenario_name",
  "seed": 42,
  "origin": [x_east, y_north, z_up],
  "objects": [
    {
      "id": "unique_id",
      "asset": "<an asset name from the catalogue>",
      "pos": [x_east, y_north, z_up],
      "yaw_deg": 0.0,
      "scale": [sx, sy, sz],
      "physics": true,
      "role": "structure | prop | target | hazard"
    }
  ],
  "target": { "object_id": "id_of_the_goal_object", "landing_radius_m": 4.0 },
  "spawn": { "pos": [x_east, y_north, z_up], "yaw_deg": 0.0 },
  "requirements": [ { "kind": "max_time_s", "value": 300 } ],
  "instruction": "one or two sentences telling the drone what to do"
}
"""


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
def system_prompt() -> str:
    return textwrap.dedent(
        f"""\
        You are a scenario compiler for a UAV simulation benchmark.

        Your job: turn a natural-language test request into ONE JSON object that
        describes a flyable 3D test scenario for Unreal Engine / AirSim.

        ## Coordinate system (critical)
        All positions use ENU, in METRES:
          - x = east, y = north, z = UP
          - yaw_deg = rotation about the vertical axis, counter-clockwise from east
        The scene is built around `origin`. Object `pos` values are RELATIVE to
        `origin`; `spawn.pos` is also relative to `origin`.
        You never write AirSim's NED frame. A converter handles that.

        ## Scale
        The base primitives (Cube, Plane) are 1 m across at scale 1.
        To make a wall 12 m long, 0.2 m thick and 3 m tall:
          "asset": "Cube", "scale": [12.0, 0.2, 3.0]
        Non-primitive assets ignore this and use their modelled size.

        ## Asset catalogue
        You may ONLY use asset names listed below. Inventing a name is a hard
        failure and the scenario will be rejected.

        {prompt_text()}

        ## Build rules
        1. 12 to 40 objects total. Prefer `Cube` for walls and floors: it is the
           only asset whose dimensions you control exactly.
        2. Walls MUST have "physics": true, otherwise the drone flies through them.
           Decorative props should have "physics": false or omitted.
        3. Every building needs an opening. If you enclose a room completely with
           walls, the scenario is unsolvable. Leave a doorway gap of at least 2 m.
        4. Exactly one object must have "role": "target", and
           `target.object_id` must equal that object's id.
        5. The target must be hidden from `spawn`: place it behind a wall or
           inside the structure, not in open line of sight.
        6. `spawn.pos` must be OUTSIDE the structure and at least 10 m away,
           with z of 5-15 m (i.e. airborne). The drone starts already flying.
        7. Keep the whole scene within roughly 40 m x 40 m around `origin`.

        ## Output format
        Return ONLY the JSON object. No prose, no markdown fences, no comments.
        The object must match this schema:

        {SCHEMA_BRIEF}
        """
    )


def user_prompt(request: str) -> str:
    return (
        "Compile this test request into a scenario specification:\n\n"
        f"{request}\n\n"
        "Return only the JSON object."
    )


# --------------------------------------------------------------------------- #
# HTTP 调用（只用标准库，避免额外依赖）
# --------------------------------------------------------------------------- #
def call_chat_api(
    messages: Sequence[Dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    temperature: float = 0.2,
) -> str:
    """调用 OpenAI 兼容的 /chat/completions，返回助手消息文本。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"LLM 接口返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 LLM 接口 {url}: {exc.reason}\n"
            f"检查 LLM_BASE_URL 与网络/代理设置。"
        ) from exc

    try:
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"LLM 响应格式无法解析: {body[:500]}"
        ) from exc


# --------------------------------------------------------------------------- #
# JSON 提取
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """
    从模型回复里抠出 JSON 对象。

    模型经常不听话：加 markdown 围栏、前言后语、或者尾随逗号。
    这里按"越来越宽容"的顺序尝试。
    """
    candidates: List[str] = []

    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1))

    candidates.append(text)

    # 从第一个 { 到最后一个 } 的切片
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Optional[Exception] = None
    for raw in candidates:
        cleaned = raw.strip()
        if not cleaned:
            continue
        # 去掉行内注释与尾随逗号
        cleaned = re.sub(r"^\s*//.*$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(
        f"模型输出里找不到可解析的 JSON 对象。原始回复前 600 字符:\n"
        f"{text[:600]}\n解析错误: {last_error}"
    )


# --------------------------------------------------------------------------- #
# 约束检查
# --------------------------------------------------------------------------- #
def check_assets_whitelisted(spec: SceneSpec) -> List[str]:
    """返回不在资产目录里的资产名列表（空表示全部合规）。"""
    wl = whitelist()
    return sorted({o.asset for o in spec.objects if o.asset not in wl})


def check_scenario_sanity(spec: SceneSpec) -> List[str]:
    """
    校验器之外的"场景合理性"检查。

    结构校验只能保证字段合法；这里检查的是场景作为测试有没有意义。
    这些规则在论文里对应 RQ2 的"非平凡性"与"初始遮挡"。
    """
    problems: List[str] = []

    if len(spec.objects) < 5:
        problems.append(f"objects 只有 {len(spec.objects)} 个，场景过于简单（至少 5 个）")

    if spec.target is None:
        problems.append("缺少 target：搜索任务必须指定目标物体")
    if spec.spawn is None:
        problems.append("缺少 spawn：必须指定无人机起点")

    targets = [o for o in spec.objects if o.role == "target"]
    if spec.target is not None:
        if len(targets) != 1:
            problems.append(
                f"role=target 的物体应恰好 1 个，实际 {len(targets)} 个"
            )
        elif spec.target.object_id != targets[0].id:
            problems.append(
                f"target.object_id='{spec.target.object_id}' 与 role=target 的 "
                f"物体 '{targets[0].id}' 不一致"
            )

    structures = [o for o in spec.objects if o.role == "structure"]
    if not structures:
        problems.append("没有任何 role=structure 的物体，场景缺少实体环境")
    for s in structures:
        if not s.physics:
            problems.append(
                f"结构物体 '{s.id}' 的 physics=false，无人机会穿墙而过"
            )

    if spec.spawn is not None and spec.target is not None:
        goal = spec.object_by_id(spec.target.object_id)
        if goal is not None:
            # 起点空中、目标在地面附近
            if spec.spawn.pos[2] < 3.0:
                problems.append(
                    f"spawn 高度 {spec.spawn.pos[2]:.1f} m 太低，无人机应在 5-15 m 起飞"
                )
            horizontal = (
                (spec.spawn.pos[0] - goal.pos[0]) ** 2
                + (spec.spawn.pos[1] - goal.pos[1]) ** 2
            ) ** 0.5
            if horizontal < 8.0:
                problems.append(
                    f"起点到目标的水平距离只有 {horizontal:.1f} m，任务太短（至少 8 m）"
                )

    return problems


def validate_all(spec: SceneSpec) -> List[str]:
    """把资产白名单与场景合理性检查合并成一份问题列表。"""
    problems = check_assets_whitelisted(spec)
    problems = [
        f"资产 '{a}' 不在允许的资产目录里，必须换成目录中存在的名字" for a in problems
    ]
    problems += check_scenario_sanity(spec)
    return problems


# --------------------------------------------------------------------------- #
# 离线模板（无 API 时的确定性基线）
# --------------------------------------------------------------------------- #
def template_spec(request: str) -> Dict[str, Any]:
    """
    根据关键词生成一个确定性的规格书。

    用途有两个：
      1. 没有 API key 时仍能跑通整条流水线；
      2. 作为论文里的对照基线 —— 证明 LLM 相对"规则模板"到底多做了什么。
    """
    text = request.lower()

    objects: List[Dict[str, Any]] = []

    def add(**kw: Any) -> None:
        objects.append(kw)

    # --- 地基：规格书约定必须显式给出地面，否则楼板下方是通的，
    #     无人机会从楼底钻进"封闭"的房间 ---
    add(id="ground_base", asset="Cube", pos=[0, 0, -0.4], scale=[24, 20, 1.2],
        physics=True, role="structure")

    # --- 两层楼主体 ---
    add(id="floor_ground", asset="Cube", pos=[0, 0, 0.1], scale=[12, 8, 0.2],
        physics=True, role="structure")

    # 二楼楼板被楼梯井分成四块，留出 3x3 m 开口，否则二楼无法到达
    for tag, pos_, scale_ in [
        ("north", [0, 3.0, 3.1], [12, 2.0, 0.2]),
        ("south", [0, -2.5, 3.1], [12, 3.0, 0.2]),
        ("west", [-5.5, 0.5, 3.1], [1.0, 3.0, 0.2]),
        ("east", [2.0, 0.5, 3.1], [8.0, 3.0, 0.2]),
    ]:
        add(id=f"floor_second_{tag}", asset="Cube", pos=pos_, scale=scale_,
            physics=True, role="structure")

    add(id="roof", asset="Cube", pos=[0, 0, 6.1], scale=[12, 8, 0.2],
        physics=True, role="structure")

    # 一层：南墙留 4 m 门洞（2 m 在 0.6 m 安全半径下会被栅格堵死）
    add(id="l1_wall_north", asset="Cube", pos=[0, 4, 1.6], scale=[12, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_south_left", asset="Cube", pos=[-5, -4, 1.6], scale=[2, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_south_right", asset="Cube", pos=[3, -4, 1.6], scale=[6, 0.2, 3],
        physics=True, role="structure")
    add(id="l1_wall_east", asset="Cube", pos=[6, 0, 1.6], scale=[0.2, 8, 3],
        physics=True, role="structure")
    add(id="l1_wall_west", asset="Cube", pos=[-6, 0, 1.6], scale=[0.2, 8, 3],
        physics=True, role="structure")

    # 二层：全封闭，目标藏在里面
    for tag, pos_, scale_ in [
        ("north", [0, 4, 4.6], [12, 0.2, 3]),
        ("south", [0, -4, 4.6], [12, 0.2, 3]),
        ("east", [6, 0, 4.6], [0.2, 8, 3]),
        ("west", [-6, 0, 4.6], [0.2, 8, 3]),
    ]:
        add(id=f"l2_wall_{tag}", asset="Cube", pos=pos_, scale=scale_,
            physics=True, role="structure")

    # --- 目标 ---
    add(id="survivor_1", asset="BP_ma_construction_worker_01",
        pos=[2, 2, 3.2], yaw_deg=210, physics=True, role="target")

    # --- 按关键词扩展 ---
    if any(k in text for k in ("火", "fire", "burn", "smoke", "烟")):
        add(id="fire_light_2f", asset="PointLightBP", pos=[0, 0, 3.6], role="hazard")
        add(id="fire_marker_2f", asset="Sphere", pos=[0.5, 0.5, 3.5],
            scale=[0.5, 0.5, 0.5], role="hazard")
        add(id="fireplace_1f", asset="SM_Fireplace_01a", pos=[-4, 2, 0.2],
            physics=True, role="hazard")

    if any(k in text for k in ("车", "car", "vehicle", "街道", "street", "城市", "urban")):
        for i, (px, py) in enumerate([(-9, -6), (-9, 6), (9, -5)]):
            add(id=f"car_{i}", asset="SM_Sedan03_Main", pos=[px, py, 0.2],
                yaw_deg=90 * i, physics=True, role="prop")

    if any(k in text for k in ("树", "tree", "forest", "林", "野外", "outdoor")):
        for i, (px, py) in enumerate([(-10, 9), (10, 9), (-11, -9)]):
            add(id=f"tree_{i}", asset="SM_tree_pine_a", pos=[px, py, 0.2],
                physics=True, role="prop")

    if any(k in text for k in ("废墟", "rubble", "塌", "collapse", "disaster", "地震")):
        for i, (px, py) in enumerate([(-3, -6), (4, 5)]):
            add(id=f"rubble_{i}", asset="SM_Rubble01", pos=[px, py, 0.3],
                physics=True, role="prop")

    if any(k in text for k in ("电", "巡检", "inspection", "infrastructure", "管道", "pipe")):
        for i in range(4):
            add(id=f"pillar_{i}", asset="Cube",
                pos=[-4 + i * 2.5, -2, 1.6], scale=[0.4, 0.4, 3],
                physics=True, role="structure")

    # --- 门与窗：纯装饰 ---
    add(id="door_entry", asset="SM_build_b_door_wood_single",
        pos=[-3, -3.9, 1.0], role="prop")
    add(id="window_north", asset="SM_build_b_mod_lvl2_doublewindow",
        pos=[2, 3.85, 4.6], yaw_deg=90, role="prop")

    # --- 任务与指标 ---
    name = "scenario"
    for key, slug in [
        ("fire", "fire"), ("火", "fire"),
        ("search", "search"), ("搜", "search"),
        ("inspection", "inspection"), ("巡检", "inspection"),
    ]:
        if key in text:
            name = f"{slug}_building"
            break

    return {
        "version": "0.1",
        "name": name,
        "seed": 42,
        "origin": [25.0, 0.0, 0.0],
        "objects": objects,
        "target": {"object_id": "survivor_1", "landing_radius_m": 4.0},
        "spawn": {"pos": [-28.0, 0.0, 8.0], "yaw_deg": 0.0},
        "requirements": [
            {"kind": "max_time_s", "value": 600},
            {"kind": "min_altitude_m", "value": 1.5},
            {"kind": "no_collision", "value": True},
        ],
        "instruction": (
            "Take off, fly east toward the two-storey building, enter through the "
            "ground-floor doorway, and locate the person on the second floor. "
            "Land within 4 metres of them."
        ),
    }


# --------------------------------------------------------------------------- #
# 主编译循环
# --------------------------------------------------------------------------- #
def compile_request(
    request: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    temperature: float = 0.2,
) -> SceneSpec:
    """
    反复"生成 -> 校验 -> 反馈"直到产出合规规格书。返回校验通过的 SceneSpec。

    抛出 RuntimeError 表示重试耗尽，异常信息里带最后一轮的完整问题列表。
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(request)},
    ]

    all_problems: List[str] = []

    for attempt in range(1, max_attempts + 1):
        LOG.info("第 %d/%d 轮生成", attempt, max_attempts)
        raw = call_chat_api(
            messages, base_url=base_url, api_key=api_key,
            model=model, temperature=temperature,
        )

        # --- 解析 JSON ---
        try:
            data = extract_json(raw)
        except ValueError as exc:
            problems = [str(exc)]
            LOG.warning("JSON 解析失败，进入重试")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    "Your reply was not valid JSON. Problems:\n- "
                    + "\n- ".join(problems)
                    + "\n\nReturn ONLY the JSON object, no fences, no prose."},
            ]
            all_problems = problems
            continue

        # --- 结构校验 ---
        try:
            spec = parse_spec(data)
        except SpecError as exc:
            problems = exc.problems
            LOG.warning("规格书结构校验失败（%d 个问题），反馈给模型重写", len(problems))
            messages += [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {"role": "user", "content":
                    "That specification failed validation. Fix ALL of these and "
                    "return the complete corrected JSON object:\n- "
                    + "\n- ".join(problems)},
            ]
            all_problems = problems
            continue

        # --- 语义与资产校验 ---
        problems = validate_all(spec)
        if problems:
            LOG.warning("场景合理性检查发现 %d 个问题，反馈给模型重写", len(problems))
            messages += [
                {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)},
                {"role": "user", "content":
                    "The scenario is structurally valid but not usable as a test. "
                    "Fix ALL of these and return the complete corrected JSON:\n- "
                    + "\n- ".join(problems)},
            ]
            all_problems = problems
            continue

        LOG.info("第 %d 轮通过全部检查", attempt)
        return spec

    raise RuntimeError(
        f"{max_attempts} 轮之后仍未得到合规规格书。最后一轮的问题:\n"
        + "\n".join(f"  - {p}" for p in all_problems)
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="自然语言需求 -> 场景规格书 JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--request", required=True, help="自然语言测试需求")
    ap.add_argument("--out", default="", help="规格书输出路径（留空则打印到 stdout）")
    ap.add_argument("--offline", action="store_true", help="不调用 LLM，用确定性模板")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过几何验证（默认会做：可解性、初始遮挡、非平凡性）",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- 离线模板 ----
    if a.offline:
        LOG.info("离线模式：使用确定性模板生成")
        data = template_spec(a.request)
        try:
            spec = parse_spec(data)
        except SpecError as exc:
            LOG.error("模板产物未通过校验（这是本程序的 bug）:\n%s", exc)
            return 1
        problems = validate_all(spec)
        if problems:
            LOG.error("模板产物未通过合理性检查（这是本程序的 bug）:")
            for p in problems:
                LOG.error("  - %s", p)
            return 1
        if not a.no_verify and not _verify_spec(spec):
            LOG.error("模板产物未通过几何验证（这是本程序的 bug）")
            return 1
        return _emit(spec, data, a.out)

    # ---- 在线模式 ----
    if not a.api_key:
        LOG.error(
            "缺少 API key。两种做法：\n"
            "  1) export LLM_API_KEY=sk-xxxx  （可选 LLM_BASE_URL / LLM_MODEL）\n"
            "  2) 加 --offline 用模板跑通链路"
        )
        return 2

    try:
        spec = compile_request(
            a.request,
            base_url=a.base_url,
            api_key=a.api_key,
            model=a.model,
            max_attempts=a.max_attempts,
            temperature=a.temperature,
        )
    except RuntimeError as exc:
        LOG.error("编译失败: %s", exc)
        return 3

    if not a.no_verify and not _verify_spec(spec):
        LOG.error(
            "模型产出的场景未通过几何验证。这说明当前的校验-反馈循环还不够，"
            "需要把验证器接入 compile_request 的重试回路。"
        )
        return 4

    return _emit(spec, spec_to_dict(spec), a.out)


def spec_to_dict(spec: SceneSpec) -> Dict[str, Any]:
    """把 SceneSpec 转回可序列化的 dict（用于落盘）。"""
    return {
        "version": spec.version,
        "name": spec.name,
        "seed": spec.seed,
        "origin": list(spec.origin),
        "objects": [
            {
                "id": o.id,
                "asset": o.asset,
                "pos": list(o.pos),
                **({"yaw_deg": o.yaw_deg} if o.yaw_deg else {}),
                **({"scale": list(o.scale)} if tuple(o.scale) != (1.0, 1.0, 1.0) else {}),
                **({"physics": o.physics} if o.physics else {}),
                "role": o.role,
            }
            for o in spec.objects
        ],
        **(
            {"target": {
                "object_id": spec.target.object_id,
                "landing_radius_m": spec.target.landing_radius_m,
            }}
            if spec.target else {}
        ),
        **(
            {"spawn": {"pos": list(spec.spawn.pos), "yaw_deg": spec.spawn.yaw_deg}}
            if spec.spawn else {}
        ),
        "requirements": spec.requirements,
        "instruction": spec.instruction,
    }


def _verify_spec(spec: SceneSpec) -> bool:
    """
    跑几何验证并打印摘要。返回是否可作为测试用例使用。

    局部导入 verifier：这样只做"编译"而不做"验证"的调用方不会因为
    verifier 的依赖变化而受影响。
    """
    import verifier as _verifier

    result = _verifier.verify(spec)
    print(result.summary())
    print()
    return result.usable


def _emit(spec: SceneSpec, data: Dict[str, Any], out: str) -> int:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        LOG.info("规格书已写入 %s", out)
    else:
        print(text)

    print()
    print(spec.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())