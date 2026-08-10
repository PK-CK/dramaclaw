"""六个步骤的执行体。

每一步都复用界面上那颗按钮背后的同一条路径——直接调对应的路由函数，
让它照常入队既有的 task_type，然后等它跑完。不另起一套生成逻辑：
出图/出片的参数装配有几百行（分辨率、比例、参考图、计费快照），
抄一份必然与上游漂移。

路由函数的 ``user`` 参数在直接调用时不会走 FastAPI 的依赖注入，
所以鉴权由批量流水线自己的入口一次性把关（``require_scope("tasks:submit")``），
这里传的是从 ProjectContext 还原出来的请求者身份，每一步仍会重新校验项目角色。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from novelvideo.batch_pipeline import service
from novelvideo.batch_pipeline.models import (
    GateId,
    PipelineState,
    StepId,
    StepStatus,
)
from novelvideo.batch_pipeline.prompt_enhancer import (
    build_shot_context,
    enhance_video_prompt,
    is_enhanced,
    strip_enhancement,
)
from novelvideo.batch_pipeline.scene_matcher import (
    assign_beats,
    build_assignment_table,
    parse_scene_headings,
)
from novelvideo.project_context import ProjectContext

#: 子任务的终态。轮询到这几个状态就停。
_TERMINAL = {"completed", "failed", "cancelled"}
#: 单个子任务的最长等待。出片最慢，按最慢的给。
_TASK_TIMEOUT_SECONDS = 45 * 60
#: 闸门最长等待。人可能去睡了，超时就停在闸门上，不往下花钱。
_GATE_TIMEOUT_SECONDS = 6 * 60 * 60
_POLL_INTERVAL_SECONDS = 2.0


class StepFailed(RuntimeError):
    """步骤失败。带上给人看的原因，由 runner 落进 step.error。"""


class PipelineCancelled(RuntimeError):
    """用户在闸门上取消，或点了取消按钮。"""


@dataclass
class Runtime:
    ctx: ProjectContext
    episode: int
    output_dir: str
    state: PipelineState
    log: Callable[..., None]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def project(self) -> str:
        return self.ctx.project_id

    @property
    def user(self) -> dict[str, Any]:
        """从 ProjectContext 还原请求者身份。

        只带 id 与 username，不带任何凭据；下游 resolve_project_context
        会用它重新查一次项目角色，权限判定不因为走了批量入口而放宽。
        """
        return {
            "user_id": self.ctx.requester_user_id,
            "username": self.ctx.requester_username,
        }

    async def store(self):
        from novelvideo.api.deps import make_sqlite_store_for_context

        return await make_sqlite_store_for_context(self.ctx)

    def save(self) -> None:
        service.save_state(self.output_dir, self.state)

    def refresh_signals(self) -> None:
        """从盘上重读闸门与取消标记——它们是别的请求写进来的。"""
        latest = service.load_state(self.output_dir, self.episode)
        self.state.gates = latest.gates
        self.state.auto_approve = latest.auto_approve
        self.state.cancelled = latest.cancelled

    def raise_if_cancelled(self) -> None:
        self.refresh_signals()
        if self.state.cancelled:
            raise PipelineCancelled("用户取消了批量流水线")


# ── 通用工具 ──────────────────────────────────────────────────────────────


async def _await_task(
    rt: Runtime,
    task_type: str,
    *,
    beat_num: int | None = None,
    scope: str | None = None,
    label: str = "",
    timeout: float = _TASK_TIMEOUT_SECONDS,
) -> Any:
    """轮询子任务直到终态。返回 TaskState。"""
    from novelvideo.task_state import get_task_manager

    manager = get_task_manager()
    waited = 0.0
    while waited < timeout:
        rt.raise_if_cancelled()
        state = manager.get_task_for_project(
            rt.ctx, task_type, rt.episode, beat_num=beat_num, scope=scope
        )
        if state is not None and state.status in _TERMINAL:
            return state
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
    raise StepFailed(f"{label or task_type} 超时（{int(timeout / 60)} 分钟未结束）")


def _require_ok(payload: Any, what: str) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    if not data.get("ok"):
        raise StepFailed(f"{what}失败：{data.get('error') or data.get('message') or payload}")
    return data


async def _wait_for_gate(rt: Runtime, gate_id: GateId, payload: dict[str, Any]) -> None:
    """挂起等人确认。已放行（或已预授权）就直接过。"""
    gate = rt.state.gate(gate_id)
    gate.payload = payload
    rt.save()

    if rt.state.gate_is_open(gate_id):
        rt.log(f"闸门「{gate_id.value}」已放行")
        return

    rt.log(f"⏸ 等待确认：{gate_id.value}")
    waited = 0.0
    while waited < _GATE_TIMEOUT_SECONDS:
        rt.raise_if_cancelled()
        if rt.state.gate_is_open(gate_id):
            rt.log(f"闸门「{gate_id.value}」已确认，继续")
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
    raise StepFailed(f"闸门「{gate_id.value}」等待超时，流水线停在此处（未产生后续开销）")


async def _load_script(rt: Runtime) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    store = await rt.store()
    script = await store.get_script_as_dict(rt.episode)
    if not script:
        raise StepFailed(f"第 {rt.episode} 集还没有剧本")
    beats = [dict(b) for b in (script.get("beats") or [])]
    if not beats:
        raise StepFailed(f"第 {rt.episode} 集剧本里没有 beat")
    return store, script, beats


# ── 第 0 步：前置体检 ─────────────────────────────────────────────────────


async def run_preflight(rt: Runtime) -> dict[str, Any]:
    """零成本自检。硬缺失直接拦下，软缺失只报告。"""
    store, _script, beats = await _load_script(rt)
    episode = store.get_episode(rt.episode)

    scene_menu = [
        getattr(item, "scene_id", "") or (item.get("scene_id") if isinstance(item, dict) else "")
        for item in (getattr(episode, "scene_menu", None) or [])
    ]
    prop_menu = [
        getattr(item, "prop_id", "") or (item.get("prop_id") if isinstance(item, dict) else "")
        for item in (getattr(episode, "prop_menu", None) or [])
    ]
    raw_content = str(getattr(episode, "raw_content", "") or "")
    headings = parse_scene_headings(raw_content)

    blockers: list[str] = []
    warnings: list[str] = []
    if not raw_content.strip():
        blockers.append("剧集缺少剧本原文，无法推导场次归属")
    if not headings:
        warnings.append("剧本原文里没有解析到场次头（`1-1 地点 时段 内`），场景归属只能靠关键词兜底")
    if not scene_menu:
        warnings.append("场景菜单为空，将在第 1 步规划")
    if not prop_menu:
        warnings.append("道具菜单为空，将在第 1 步规划")

    missing_scene_ref = [
        int(b.get("beat_number") or 0)
        for b in beats
        if not ((b.get("scene_ref") or {}).get("scene_id"))
    ]
    cost = service.estimate_cost(beats, resolution=rt.state.resolution)
    rt.state.cost_estimate = cost

    if blockers:
        raise StepFailed("；".join(blockers))

    for line in warnings:
        rt.log(f"⚠️ {line}")
    rt.log(
        f"体检通过：{len(beats)} 个 beat，{len(headings)} 场，"
        f"{len(missing_scene_ref)} 个 beat 缺场景关联；"
        f"预估视频 {cost['video_seconds']}s ≈ ${cost['video_usd']}"
    )
    return {
        "beats": len(beats),
        "scenes": len(headings),
        "scene_menu": scene_menu,
        "prop_menu": prop_menu,
        "missing_scene_ref": missing_scene_ref,
        "warnings": warnings,
        "cost_estimate": cost,
    }


# ── 第 1 步：场景 + 道具规划 ──────────────────────────────────────────────


async def run_plan_assets(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.episodes import plan_episode_props, plan_episode_scenes

    preflight = rt.state.step(StepId.PREFLIGHT).result
    planned: list[str] = []

    if not preflight.get("scene_menu"):
        rt.log("规划场景菜单...")
        _require_ok(await plan_episode_scenes(rt.project, rt.episode, rt.user), "场景规划")
        await _await_task(rt, "episode_scene_planner", label="场景规划")
        planned.append("scene")

    if not preflight.get("prop_menu"):
        rt.log("规划道具菜单...")
        _require_ok(await plan_episode_props(rt.project, rt.episode, rt.user), "道具规划")
        await _await_task(rt, "episode_prop_planner", label="道具规划")
        planned.append("prop")

    if not planned:
        rt.log("场景与道具菜单都已就绪，跳过")
    return {"planned": planned}


# ── 第 2 步：场次交叉验证 → 补 scene_ref（闸门 1）────────────────────────


async def run_match_scenes(rt: Runtime) -> dict[str, Any]:
    store, _script, beats = await _load_script(rt)
    episode = store.get_episode(rt.episode)
    raw_content = str(getattr(episode, "raw_content", "") or "")

    allowed = {
        str(getattr(item, "scene_id", "") or (item.get("scene_id") if isinstance(item, dict) else ""))
        for item in (getattr(episode, "scene_menu", None) or [])
    }
    allowed.discard("")

    headings = parse_scene_headings(raw_content)
    assignments = assign_beats(
        headings,
        beats,
        allowed_scene_ids=allowed or None,
        script_lines=raw_content.split("\n"),
    )
    table = build_assignment_table(assignments)
    rt.log(
        f"推导出 {len(table['by_scene'])} 段场次，{table['changed']} 个 beat 需要改写，"
        f"{len(table['low_confidence'])} 个低置信"
    )

    await _wait_for_gate(rt, GateId.SCENE_ASSIGNMENT, table)

    written = 0
    for item in assignments:
        if not item.changed:
            continue
        ok = await store.update_beat_asset(
            episode_number=rt.episode,
            beat_number=item.beat_number,
            scene_ref={"scene_id": item.scene_id, "variant_id": ""},
            time_of_day=item.time_of_day,
        )
        if ok:
            written += 1
        else:
            rt.log(f"⚠️ Beat {item.beat_number} 写入场景关联失败")
    rt.log(f"已回填 {written} 个 beat 的场景与时段")
    return {"table": table, "written": written}


# ── 第 3 步：批量草图（闸门 2）───────────────────────────────────────────


async def run_sketches(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import generate_sketches
    from novelvideo.api.schemas import SketchGenerateRequest

    body = SketchGenerateRequest(
        style=rt.options.get("style"),
        model=str(rt.options.get("sketch_model") or "nanobanana"),
        grid_index=-1,  # -1 = 全集所有网格
        sketch_scene_grouping=bool(rt.options.get("sketch_scene_grouping", True)),
        aspect_ratio=str(rt.options.get("aspect_ratio") or "2:3"),  # type: ignore[arg-type]
        image_generation_selection=rt.options.get("image_generation_selection"),
    )
    data = _require_ok(
        await generate_sketches(rt.project, rt.episode, body, rt.user), "草图生成"
    )
    scopes = list((data.get("data") or {}).get("scopes") or [])
    if not scopes and data.get("task_key"):
        scopes = [str(data.get("scope") or "grid_0")]
    rt.log(f"草图任务已入队：{len(scopes)} 个网格")

    failed: list[str] = []
    for index, scope in enumerate(scopes, start=1):
        state = await _await_task(
            rt, "sketch_grid_generation", scope=scope, label=f"草图 {scope}"
        )
        if state.status != "completed":
            failed.append(f"{scope}: {state.error or state.status}")
        rt.log(f"草图 {index}/{len(scopes)} 完成", progress=index / max(len(scopes), 1))
    if failed:
        raise StepFailed("草图生成失败：" + "；".join(failed))

    await _wait_for_gate(
        rt,
        GateId.SKETCH_SAMPLE,
        {"scopes": scopes, "hint": "抽查草图的人物姿势与镜头景别，确认后继续渲染"},
    )
    return {"scopes": scopes}


# ── 第 4 步：批量渲染 ⊕ 视频提示词（闸门 3）─────────────────────────────


async def run_render(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import regenerate_beats
    from novelvideo.api.schemas import BeatsRegenerateRequest

    _store, _script, beats = await _load_script(rt)
    beat_numbers = [int(b.get("beat_number") or 0) for b in beats]

    body = BeatsRegenerateRequest(
        beat_indices=beat_numbers,
        style=rt.options.get("style"),
        model=str(rt.options.get("render_model") or "nanobanana"),
        mode_key=str(rt.options.get("render_mode_key") or "1x1_2-3"),
        image_generation_selection=rt.options.get("image_generation_selection"),
    )
    data = _require_ok(
        await regenerate_beats(rt.project, rt.episode, body, rt.user), "批量渲染"
    )
    scope = str(data.get("scope") or (data.get("data") or {}).get("scope") or "")
    rt.log(f"渲染任务已入队（{len(beat_numbers)} 个 beat）")
    state = await _await_task(rt, "selected_regen", scope=scope or None, label="批量渲染")
    if state.status != "completed":
        raise StepFailed(f"批量渲染失败：{state.error or state.status}")

    prompts = await _build_video_prompts(rt)

    await _wait_for_gate(
        rt,
        GateId.RENDER_SAMPLE,
        {
            "beats": beat_numbers,
            "prompt_samples": prompts["samples"],
            "hint": "抽查画风、角色脸、背景是否一致，并看一眼视频提示词，确认后开始出片",
        },
    )
    return {"beats": len(beat_numbers), **prompts["stats"]}


async def _build_video_prompts(rt: Runtime) -> dict[str, Any]:
    """逐 beat 生成视频提示词并叠加导演级约束。"""
    from novelvideo.api.routes.scripts import _generate_and_save_beat_video_prompt

    store, _script, beats = await _load_script(rt)
    language = str(rt.options.get("language") or "en")

    # 场景分段：同一场里的第几镜决定「首镜定轴」还是「守轴」
    index_in_scene: dict[int, int] = {}
    prev_scene = object()
    counter = 0
    for beat in beats:
        scene_id = str((beat.get("scene_ref") or {}).get("scene_id") or "")
        counter = 0 if scene_id != prev_scene else counter + 1
        prev_scene = scene_id
        index_in_scene[int(beat.get("beat_number") or 0)] = counter

    generated = 0
    enhanced = 0
    skipped: list[int] = []
    samples: list[dict[str, Any]] = []

    for position, beat in enumerate(beats):
        rt.raise_if_cancelled()
        beat_num = int(beat.get("beat_number") or 0)
        try:
            data = await _generate_and_save_beat_video_prompt(
                store=store,
                output_dir=rt.output_dir,
                project_name=rt.ctx.project_name,
                episode_num=rt.episode,
                beat_num=beat_num,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001 — 单镜失败不该拖垮整步
            skipped.append(beat_num)
            rt.log(f"⚠️ Beat {beat_num} 视频提示词生成失败：{exc}")
            continue
        generated += 1

        if data.get("field") != "video_prompt":
            continue  # 首尾帧模式的过渡提示词不叠加运镜约束

        base = strip_enhancement(str(data.get("prompt") or ""))
        shot = build_shot_context(
            beat,
            scene_id=str((beat.get("scene_ref") or {}).get("scene_id") or ""),
            time_of_day=str(beat.get("time_of_day") or ""),
            index_in_scene=index_in_scene.get(beat_num, 0),
            prev_beat=beats[position - 1] if position else None,
        )
        final = enhance_video_prompt(base, shot)
        if final and not is_enhanced(str(data.get("prompt") or "")):
            await store.update_beat_asset(
                episode_number=rt.episode, beat_number=beat_num, video_prompt=final
            )
        enhanced += 1
        if len(samples) < 3:
            samples.append({"beat_number": beat_num, "prompt": final})

        rt.log(
            f"视频提示词 {position + 1}/{len(beats)}",
            progress=(position + 1) / max(len(beats), 1),
        )

    if skipped:
        rt.log(f"⚠️ {len(skipped)} 个 beat 没有提示词：{skipped}，出片时会跳过")
    return {
        "samples": samples,
        "stats": {"prompts_generated": generated, "prompts_enhanced": enhanced,
                  "prompts_skipped": skipped},
    }


# ── 第 5 步：逐 beat 出视频 ──────────────────────────────────────────────


async def run_videos(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import generate_single_video
    from novelvideo.api.schemas import SingleVideoRequest

    _store, _script, beats = await _load_script(rt)
    backend = str(rt.options.get("video_backend") or "grok_720")
    resolution = str(rt.state.resolution or "720p")
    semaphore = asyncio.Semaphore(max(1, int(rt.state.video_concurrency or 2)))

    done = 0
    failed: list[dict[str, Any]] = []
    total = len(beats)

    async def _one(beat: dict[str, Any]) -> None:
        nonlocal done
        beat_num = int(beat.get("beat_number") or 0)
        async with semaphore:
            rt.raise_if_cancelled()
            body = SingleVideoRequest(
                resolution=resolution,
                video_backend=backend,
                duration=rt.options.get("duration"),
                ratio=rt.options.get("ratio"),
            )
            try:
                _require_ok(
                    await generate_single_video(
                        rt.project, rt.episode, beat_num, body, rt.user
                    ),
                    f"Beat {beat_num} 出片",
                )
                state = await _await_task(
                    rt, "single_video", beat_num=beat_num, label=f"Beat {beat_num} 出片"
                )
                if state.status != "completed":
                    raise StepFailed(state.error or state.status)
            except PipelineCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — 单镜失败继续跑完其余
                failed.append({"beat_number": beat_num, "error": str(exc)})
                rt.log(f"❌ Beat {beat_num} 出片失败：{exc}")
                return
        done += 1
        rt.log(f"出片 {done}/{total}", progress=done / max(total, 1))

    await asyncio.gather(*(_one(beat) for beat in beats))

    rt.state.failed_beats = [int(item["beat_number"]) for item in failed]
    if failed:
        rt.log(
            f"⚠️ {len(failed)} 个 beat 未出片：{rt.state.failed_beats}。"
            "这些位置在成片里是缺口，会断剧情连续性，需单独重试后再合成。"
        )
    return {"total": total, "succeeded": done, "failed": failed}


# ── 编排 ────────────────────────────────────────────────────────────────

STEP_RUNNERS: dict[StepId, Callable[[Runtime], Any]] = {
    StepId.PREFLIGHT: run_preflight,
    StepId.PLAN_ASSETS: run_plan_assets,
    StepId.MATCH_SCENES: run_match_scenes,
    StepId.SKETCHES: run_sketches,
    StepId.RENDER: run_render,
    StepId.VIDEOS: run_videos,
}

STEP_LABELS: dict[StepId, str] = {
    StepId.PREFLIGHT: "前置体检",
    StepId.PLAN_ASSETS: "场景与道具规划",
    StepId.MATCH_SCENES: "场次交叉验证",
    StepId.SKETCHES: "批量草图",
    StepId.RENDER: "批量渲染与视频提示词",
    StepId.VIDEOS: "逐镜出片",
}


async def run_pipeline(rt: Runtime, *, start_from: StepId | None = None) -> dict[str, Any]:
    """按顺序跑完六步。已完成的步骤会跳过，支持断点续跑。"""
    started = start_from is None
    for step_id in StepId:
        if not started:
            started = step_id == start_from
            if not started:
                continue

        step = rt.state.step(step_id)
        if step.status == StepStatus.DONE and start_from is None:
            rt.log(f"[{STEP_LABELS[step_id]}] 已完成，跳过")
            continue

        step.status = StepStatus.RUNNING
        step.error = ""
        rt.save()
        rt.log(f"▶ {STEP_LABELS[step_id]}")

        try:
            step.result = await STEP_RUNNERS[step_id](rt) or {}
        except PipelineCancelled:
            step.status = StepStatus.PENDING
            step.message = "已取消"
            rt.save()
            raise
        except Exception as exc:  # noqa: BLE001 — 统一落到 step.error 供前端展示
            step.status = StepStatus.FAILED
            step.error = str(exc)
            rt.save()
            raise

        step.status = StepStatus.DONE
        step.progress = 1.0
        step.message = f"{STEP_LABELS[step_id]}完成"
        rt.save()

    return {
        "steps": {k: v.status.value for k, v in rt.state.steps.items()},
        "failed_beats": list(rt.state.failed_beats),
        "cost_estimate": dict(rt.state.cost_estimate),
    }
