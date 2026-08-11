"""七个步骤的执行体。

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


class GateTimedOutSkip(RuntimeError):
    """闸门超时未放行，且该闸门声明超时=跳过而非失败（闸门 5：不点头就不出片）。"""


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
        """写盘。

        取消与闸门确认是别的请求写进这个文件的；直接整份覆盖会把刚到的
        决定抹掉（本进程内存里的副本可能是 2 秒前的）。这两类信号只会
        false→true，写之前从盘上合并回来即可。
        """
        latest = service.load_state(self.output_dir, self.episode)
        if latest.cancelled:
            self.state.cancelled = True
        # 只合并 approved，不合并 payload：目前 payload 只有流水线自己写，
        # 前端调 approve 时传的是 null，没有第二个写入源。
        # ⚠️ payload 还兼作前端的闸门显示判据（queries/batch-pipeline.ts 的
        # pendingGate 用 payload 非空来判断闸门是否已到达）。日后前端若开始
        # 回传人工修正，这里必须同步补上 payload 合并——否则不只是修正丢失，
        # 闸门会因 payload 被覆盖成空而根本不渲染，用户看不到放行按钮，
        # 流水线一路卡到 6 小时超时，且日志上看不出原因。
        for key, gate in latest.gates.items():
            if gate.approved:
                self.state.gate(GateId(key)).approved = True
        if latest.auto_approve:
            self.state.auto_approve = latest.auto_approve
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

    def step_progress(self, step_id: StepId, message: str, progress: float) -> None:
        """把循环进度写进 state.json（弹窗读这里），并同步任务面板。

        log() 只到任务面板；不落盘的话弹窗里的步骤行永远一片空白。
        """
        step = self.state.step(step_id)
        step.message = message
        step.progress = progress
        self.save()
        self.log(message, progress=progress)


# ── 通用工具 ──────────────────────────────────────────────────────────────


async def _await_task(
    rt: Runtime,
    task_type: str,
    *,
    task_id: str = "",
    beat_num: int | None = None,
    scope: str | None = None,
    label: str = "",
    timeout: float = _TASK_TIMEOUT_SECONDS,
    progress_step: StepId | None = None,
) -> Any:
    """轮询子任务直到终态。返回 TaskState。

    必须带 task_id：任务状态按 (task_type, episode, beat, scope) 做 key，
    上一轮同 key 的记录可能还留在库里且已是 completed；不比对 task_id
    就会一进来就"看见完成"，直接跳过这一步。
    """
    from novelvideo.task_state import get_task_manager

    manager = get_task_manager()
    # 墙钟计时：只累加 sleep 会漏掉每轮的读盘与查询开销，实际超时明显长于标称
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        rt.raise_if_cancelled()
        state = manager.get_task_for_project(
            rt.ctx, task_type, rt.episode, beat_num=beat_num, scope=scope
        )
        matches = state is not None and (not task_id or state.task_id == task_id)
        if matches and state.status in _TERMINAL:
            return state
        if matches and progress_step is not None:
            _mirror_subtask_progress(rt, progress_step, state)
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    raise StepFailed(f"{label or task_type} 超时（{int(timeout / 60)} 分钟未结束）")


def _require_ok(payload: Any, what: str) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    if not data.get("ok"):
        raise StepFailed(f"{what}失败：{data.get('error') or data.get('message') or payload}")
    return data


def _task_id_of(data: dict[str, Any]) -> str:
    return str(data.get("task_id") or "")


def _mirror_subtask_progress(rt: Runtime, step_id: StepId, task_state: Any) -> None:
    """把子任务的进度镜像到流水线步骤上，弹窗才看得到 x/54。

    渲染是一整个大子任务，进度只上报到任务面板。轮询 2 秒一轮、
    一跑几十分钟，必须节流：进度或文案有实质变化才写盘。
    """
    progress = float(getattr(task_state, "progress", 0.0) or 0.0)
    message = str(getattr(task_state, "current_task", "") or "")
    step = rt.state.step(step_id)
    if abs(progress - step.progress) < 0.01 and (not message or message == step.message):
        return
    step.progress = progress
    if message:
        step.message = message
    rt.save()


#: 闸门归属的步骤。挂起等人时把该步标成 waiting_gate，前端才能显示暂停态。
_GATE_STEP: dict[GateId, StepId] = {
    GateId.SCENE_ASSIGNMENT: StepId.MATCH_SCENES,
    GateId.SKETCH_SAMPLE: StepId.SKETCHES,
    GateId.RENDER_SAMPLE: StepId.RENDER,
    GateId.VIDEO_PROMPTS: StepId.VIDEO_PROMPTS,
    GateId.VIDEO_GENERATION: StepId.VIDEOS,
}


async def _wait_for_gate(
    rt: Runtime,
    gate_id: GateId,
    payload: dict[str, Any],
    *,
    skip_on_timeout: bool = False,
) -> None:
    """挂起等人确认。已放行（或已预授权）就直接过。

    skip_on_timeout：超时不算失败，抛 GateTimedOutSkip 让编排层把该步
    标成 skipped 后正常收官——闸门 5 用它实现「不点头就不出片」。
    """
    # 用完即弃：refresh_signals 会整份替换 state.gates，任何跨越它的
    # gate 引用都会脱钩成孤儿（写进去的值再也落不了盘）。step 不受影响，
    # 因为 refresh 不动 steps——但别指望这个巧合。
    rt.state.gate(gate_id).payload = payload
    step = rt.state.step(_GATE_STEP[gate_id])
    rt.save()

    if rt.state.gate_is_open(gate_id):
        rt.log(f"闸门「{gate_id.value}」已放行")
        return

    step.status = StepStatus.WAITING_GATE
    step.message = f"等待确认：{gate_id.value}"
    rt.save()
    rt.log(f"⏸ 等待确认：{gate_id.value}")

    deadline = asyncio.get_running_loop().time() + _GATE_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        rt.raise_if_cancelled()
        if rt.state.gate_is_open(gate_id):
            step.status = StepStatus.RUNNING
            step.message = ""
            rt.save()
            rt.log(f"闸门「{gate_id.value}」已确认，继续")
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    if skip_on_timeout:
        raise GateTimedOutSkip(
            f"闸门「{gate_id.value}」超时未放行，已跳过出片（未产生出片开销）"
        )
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
        data = _require_ok(
            await plan_episode_scenes(rt.project, rt.episode, rt.user), "场景规划"
        )
        await _await_task(
            rt, "episode_scene_planner", task_id=_task_id_of(data), label="场景规划"
        )
        planned.append("scene")

    if not preflight.get("prop_menu"):
        rt.log("规划道具菜单...")
        data = _require_ok(
            await plan_episode_props(rt.project, rt.episode, rt.user), "道具规划"
        )
        await _await_task(
            rt, "episode_prop_planner", task_id=_task_id_of(data), label="道具规划"
        )
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
    tasks = [
        (str(item.get("scope") or ""), str(item.get("task_id") or ""))
        for item in ((data.get("data") or {}).get("tasks") or [])
    ]
    if not tasks and data.get("task_id"):
        tasks = [(str(data.get("scope") or "grid_0"), _task_id_of(data))]
    scopes = [scope for scope, _ in tasks]
    rt.log(f"草图任务已入队：{len(tasks)} 个网格")

    failed: list[str] = []
    for index, (scope, task_id) in enumerate(tasks, start=1):
        state = await _await_task(
            rt,
            "sketch_grid_generation",
            task_id=task_id,
            scope=scope,
            label=f"草图 {scope}",
        )
        if state.status != "completed":
            failed.append(f"{scope}: {state.error or state.status}")
        rt.step_progress(
            StepId.SKETCHES,
            f"草图 {index}/{len(tasks)} 完成",
            index / max(len(tasks), 1),
        )
    if failed:
        raise StepFailed("草图生成失败：" + "；".join(failed))

    await _wait_for_gate(
        rt,
        GateId.SKETCH_SAMPLE,
        {"scopes": scopes, "hint": "抽查草图的人物姿势与镜头景别，确认后继续渲染"},
    )
    return {"scopes": scopes}


# ── 第 4 步：批量渲染（闸门 3）───────────────────────────────────────────


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
    state = await _await_task(
        rt,
        "selected_regen",
        task_id=_task_id_of(data),
        scope=scope or None,
        label="批量渲染",
        progress_step=StepId.RENDER,
    )
    if state.status != "completed":
        raise StepFailed(f"批量渲染失败：{state.error or state.status}")

    await _wait_for_gate(
        rt,
        GateId.RENDER_SAMPLE,
        {
            "beats": beat_numbers,
            "hint": "抽查画风、角色脸、背景是否一致，确认后继续生成视频提示词",
        },
    )
    return {"beats": len(beat_numbers)}


# ── 第 5 步：逐 beat 生成视频提示词（闸门 4）─────────────────────────────


async def run_video_prompts(rt: Runtime) -> dict[str, Any]:
    prompts = await _build_video_prompts(rt)

    await _wait_for_gate(
        rt,
        GateId.VIDEO_PROMPTS,
        {
            "prompt_samples": prompts["samples"],
            "hint": "抽查视频提示词的文案与运镜，确认后进入出片授权（闸门 5）",
        },
    )
    return dict(prompts["stats"])


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

        rt.step_progress(
            StepId.VIDEO_PROMPTS,
            f"视频提示词 {position + 1}/{len(beats)}",
            (position + 1) / max(len(beats), 1),
        )

    if skipped:
        rt.log(f"⚠️ {len(skipped)} 个 beat 没有提示词：{skipped}，出片时会跳过")
    return {
        "samples": samples,
        "stats": {"prompts_generated": generated, "prompts_enhanced": enhanced,
                  "prompts_skipped": skipped},
    }


# ── 第 6 步：逐 beat 出视频（闸门 5）─────────────────────────────────────


async def run_videos(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import generate_single_video
    from novelvideo.api.schemas import SingleVideoRequest

    _store, _script, beats = await _load_script(rt)

    # 闸门 5 是钱闸：过了这道门才开始按秒计费。不预授权、也不来点头，
    # 超时走 GateTimedOutSkip——出片步标 skipped，前面的成果照常保留。
    await _wait_for_gate(
        rt,
        GateId.VIDEO_GENERATION,
        {
            "beats": len(beats),
            "cost_estimate": dict(rt.state.cost_estimate),
            "hint": (
                "确认后开始批量出片（按秒计费）。不确认则到时自动跳过出片，"
                "已生成的图与提示词全部保留"
            ),
        },
        skip_on_timeout=True,
    )

    backend = str(rt.options.get("video_backend") or "grok_720")
    # resolution/duration/ratio 只在显式给了值时才塞进请求体：路由用
    # `"resolution" in model_fields_set` 判断用户有没有动过它，白填一个
    # 默认值会改变计费口径与 seedance2 分支的行为。grok_720 固定 720p，
    # 本来也不吃这三个字段。
    extra: dict[str, Any] = {}
    for key in ("resolution", "duration", "ratio"):
        value = rt.options.get(key)
        if value not in (None, ""):
            extra[key] = value
    semaphore = asyncio.Semaphore(max(1, int(rt.state.video_concurrency or 3)))

    done = 0
    failed: list[dict[str, Any]] = []
    total = len(beats)

    async def _one(beat: dict[str, Any]) -> None:
        nonlocal done
        beat_num = int(beat.get("beat_number") or 0)
        async with semaphore:
            rt.raise_if_cancelled()
            body = SingleVideoRequest(video_backend=backend, **extra)
            try:
                queued = _require_ok(
                    await generate_single_video(
                        rt.project, rt.episode, beat_num, body, rt.user
                    ),
                    f"Beat {beat_num} 出片",
                )
                state = await _await_task(
                    rt,
                    "single_video",
                    task_id=_task_id_of(queued),
                    beat_num=beat_num,
                    label=f"Beat {beat_num} 出片",
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
        rt.step_progress(StepId.VIDEOS, f"出片 {done}/{total}", done / max(total, 1))

    await asyncio.gather(*(_one(beat) for beat in beats))

    rt.state.failed_beats = [int(item["beat_number"]) for item in failed]
    if failed:
        rt.log(
            f"⚠️ {len(failed)} 个 beat 未出片：{rt.state.failed_beats}。"
            "这些位置在成片里是缺口，会断剧情连续性，需单独重试后再合成。"
        )
    if total and done == 0:
        # 一个都没出来还报「完成」，用户会以为可以去合成了
        raise StepFailed(f"{total} 个 beat 全部出片失败，第一条错误：{failed[0]['error']}")
    return {"total": total, "succeeded": done, "failed": failed}


# ── 编排 ────────────────────────────────────────────────────────────────

STEP_RUNNERS: dict[StepId, Callable[[Runtime], Any]] = {
    StepId.PREFLIGHT: run_preflight,
    StepId.PLAN_ASSETS: run_plan_assets,
    StepId.MATCH_SCENES: run_match_scenes,
    StepId.SKETCHES: run_sketches,
    StepId.RENDER: run_render,
    StepId.VIDEO_PROMPTS: run_video_prompts,
    StepId.VIDEOS: run_videos,
}

STEP_LABELS: dict[StepId, str] = {
    StepId.PREFLIGHT: "前置体检",
    StepId.PLAN_ASSETS: "场景与道具规划",
    StepId.MATCH_SCENES: "场次交叉验证",
    StepId.SKETCHES: "批量草图",
    StepId.RENDER: "批量渲染",
    StepId.VIDEO_PROMPTS: "视频提示词",
    StepId.VIDEOS: "逐镜出片",
}


async def run_pipeline(rt: Runtime, *, start_from: StepId | None = None) -> dict[str, Any]:
    """按顺序跑完七步。已完成的步骤会跳过，支持断点续跑。"""
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
        except GateTimedOutSkip as exc:
            # 目前只有闸门 5 走到这：不放行=不出片，是用户的选择而非故障。
            # 标 skipped 后正常收官，前面步骤的成果全部保留。
            step.status = StepStatus.SKIPPED
            step.message = str(exc)
            rt.save()
            rt.log(f"⏭ {STEP_LABELS[step_id]}：{exc}")
            break
        except PipelineCancelled:
            step.status = StepStatus.PENDING
            step.message = "已取消"
            rt.save()
            raise
        except asyncio.CancelledError:
            # 通用任务面板的取消与 deadline 超时走的是 main_task.cancel()，
            # 抛的是 CancelledError —— 它继承 BaseException，`except Exception`
            # 拦不住。不在这里落盘，state.json 会永远停在 running，
            # 前端一直显示「正在跑」而实际早已没有进程。
            step.status = StepStatus.PENDING
            step.message = "已被取消或超时中止"
            rt.state.cancelled = True
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
