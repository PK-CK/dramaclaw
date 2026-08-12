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


#: 出图类子任务一次带几个 beat。默认 1——每张出完立刻落盘。
#:
#: 上游 render.py 的 selected_regen 是「先把所有网格全生成完，再统一切回
#: beat 帧」（`results = await regenerate_selected_beats(...)` 之后才进
#: `for result in results: save_grid_and_split(...)`）。整集丢进去意味着：
#:   ① 54 张 × 60–90 秒 ≈ 70 分钟，必然撞上 ST_PROJECT_TASK_TIMEOUT_S
#:      的 1800 秒硬上限被杀；
#:   ② 死在半路时一张都没切回 beat，已经出的图连同 token 全部作废；
#:   ③ 中途没有任何 per-beat 产物，看不出进度，重试也无从跳过。
#: 上游代码不能动，那就别给它整集——一次只给一个 beat，它出完那一张就
#: 立刻切图落盘。网络抖动最多废掉当前这一张，重跑时已有图的直接跳过。
_DEFAULT_IMAGE_CHUNK = 1


def _image_chunk_size(rt: Runtime) -> int:
    override = rt.options.get("image_chunk_size")
    if override:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_IMAGE_CHUNK


def _chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


#: 连续失败到这个数就停：单张失败多半是网络抖动，连着失败就是配置或额度问题，
#: 再往下跑只是把整集的钱烧完。
_MAX_CONSECUTIVE_FAILURES = 3

#: 同一块的重试次数与退避秒数。
#:
#: 图片要先经 media relay（Cloudinary/OSS）上传成公网 URL 再交给图像模型，
#: 而 relay 那层是零重试的（storage/media_relay.py 的 upload_bytes 一次
#: httpx.post，抛错即抛出）。一次 SSL EOF 就会判这个 beat 失败。实测同样
#: 网络下 2MB 上传 6/6 成功，说明这类错基本都是瞬时抖动，退避重试即可。
_CHUNK_RETRIES = 3
_RETRY_BACKOFF_SECONDS = (5.0, 15.0, 30.0)

#: 判定为「值得重试」的瞬时故障特征。配置错、额度不足这类重试也没用。
_TRANSIENT_HINTS = (
    "ssl",
    "eof",
    "timeout",
    "timed out",
    "connection",
    "reset",
    "temporarily",
    "502",
    "503",
    "504",
    "429",
    "media relay",
)


def _looks_transient(error: str) -> bool:
    lowered = str(error).lower()
    return any(hint in lowered for hint in _TRANSIENT_HINTS)


async def _run_chunk_with_retry(
    rt: Runtime, label: str, attempt_once: Callable[[], Any]
) -> None:
    """跑一块，瞬时故障退避重试。非瞬时故障立刻抛出，不浪费时间。"""
    last: Exception | None = None
    for attempt in range(_CHUNK_RETRIES):
        try:
            await attempt_once()
            return
        except PipelineCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — 由调用方决定这块算不算失败
            last = exc
            if not _looks_transient(str(exc)) or attempt == _CHUNK_RETRIES - 1:
                raise
            wait = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
            rt.log(f"↻ {label} 遇到瞬时故障，{wait:.0f} 秒后重试（{attempt + 1}/{_CHUNK_RETRIES - 1}）：{exc}")
            await asyncio.sleep(wait)
    if last is not None:
        raise last


def _missing_beats(rt: Runtime, beats: list[dict[str, Any]], kind: str) -> list[int]:
    """挑出还没有产物的 beat。

    kind 取 ``sketch`` 或 ``frame``——分别对应 ``sketches/epNNN/beat_NN.png``
    与 ``frames/epNNN/beat_NN.png``。已经有图的一律跳过：重出一遍既花钱，
    又会往图片池里多叠一版历史，正是要消掉的浪费。
    """
    from novelvideo.utils.path_resolver import PathResolver

    paths = PathResolver(rt.output_dir, rt.episode)
    missing: list[int] = []
    for beat in beats:
        number = int(beat.get("beat_number") or 0)
        if number <= 0:
            continue
        target = paths.sketch(number) if kind == "sketch" else paths.frame(number)
        if not target.exists() or target.stat().st_size == 0:
            missing.append(number)
    return missing


async def run_sketches(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import generate_sketches, regenerate_sketches
    from novelvideo.api.schemas import SketchGenerateRequest, SketchRegenerateRequest

    _store, _script, beats = await _load_script(rt)
    missing = _missing_beats(rt, beats, "sketch")

    if not missing:
        rt.log(f"全部 {len(beats)} 个 beat 都已有草图，跳过生成")
        await _wait_for_gate(
            rt,
            GateId.SKETCH_SAMPLE,
            {"scopes": [], "skipped": True, "hint": "草图已齐，抽查确认后继续渲染"},
        )
        return {"scopes": [], "skipped": True, "existing": len(beats)}

    # 只缺一部分时走单 beat 再生，别为了几个缺口把整集重出一遍
    if len(missing) < len(beats):
        rt.log(f"{len(beats) - len(missing)} 个 beat 已有草图，只补 {len(missing)} 个：{missing}")
        filled: list[int] = []
        consecutive = 0
        for chunk in _chunked(missing, _image_chunk_size(rt)):
            rt.raise_if_cancelled()
            label = f"Beat {chunk[0]}" if len(chunk) == 1 else f"Beat {chunk[0]}–{chunk[-1]}"
            rt.step_progress(
                StepId.SKETCHES,
                f"补草图 {len(filled)}/{len(missing)}（{label}）",
                len(filled) / max(len(missing), 1),
            )
            regen = SketchRegenerateRequest(
                beat_indices=chunk,
                style=rt.options.get("style"),
                model=str(rt.options.get("sketch_model") or "nanobanana"),
                mode_key=str(rt.options.get("sketch_mode_key") or "1x1_2-3"),
                image_generation_selection=rt.options.get("image_generation_selection"),
            )
            async def _once(regen=regen, label=label) -> None:
                data = _require_ok(
                    await regenerate_sketches(rt.project, rt.episode, regen, rt.user),
                    f"补草图 {label}",
                )
                state = await _await_task(
                    rt,
                    "sketch_regen",
                    task_id=_task_id_of(data),
                    scope=str(data.get("scope") or "") or None,
                    label=f"补草图 {label}",
                )
                if state.status != "completed":
                    raise StepFailed(state.error or state.status)

            try:
                await _run_chunk_with_retry(rt, label, _once)
            except PipelineCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — 单块失败不该废掉整步
                consecutive += 1
                rt.log(f"❌ {label} 草图失败（第 {consecutive} 次连续失败）：{exc}")
                if consecutive >= _MAX_CONSECUTIVE_FAILURES:
                    raise StepFailed(
                        f"连续 {consecutive} 次草图失败，已停止。最后一条错误：{exc}"
                    ) from exc
                continue
            consecutive = 0
            filled.extend(chunk)
        rt.step_progress(
            StepId.SKETCHES, f"补草图 {len(filled)}/{len(missing)} 完成", 1.0
        )
        await _wait_for_gate(
            rt,
            GateId.SKETCH_SAMPLE,
            {"beats": filled, "hint": "抽查新补的草图，确认后继续渲染"},
        )
        return {"scopes": [], "filled": filled, "existing": len(beats) - len(missing)}

    rt.log(f"{len(missing)} 个 beat 都没有草图，整集生成")
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
    missing = _missing_beats(rt, beats, "frame")

    if not missing:
        rt.log(f"全部 {len(beats)} 个 beat 都已有渲染图，跳过生成")
        await _wait_for_gate(
            rt,
            GateId.RENDER_SAMPLE,
            {
                "beats": [],
                "skipped": True,
                "hint": "渲染图已齐，抽查确认后继续生成视频提示词",
            },
        )
        return {"beats": 0, "skipped": True, "existing": len(beats)}

    existing = len(beats) - len(missing)
    if existing:
        rt.log(f"{existing} 个 beat 已有渲染图，只补 {len(missing)} 个：{missing}")

    chunks = _chunked(missing, _image_chunk_size(rt))
    done_beats: list[int] = []
    failed: list[dict[str, Any]] = []
    consecutive = 0

    for index, chunk in enumerate(chunks, start=1):
        rt.raise_if_cancelled()
        label = f"Beat {chunk[0]}" if len(chunk) == 1 else f"Beat {chunk[0]}–{chunk[-1]}"
        rt.step_progress(
            StepId.RENDER,
            f"渲染 {len(done_beats)}/{len(missing)}（{label}）",
            len(done_beats) / max(len(missing), 1),
        )
        body = BeatsRegenerateRequest(
            beat_indices=chunk,
            style=rt.options.get("style"),
            model=str(rt.options.get("render_model") or "nanobanana"),
            mode_key=str(rt.options.get("render_mode_key") or "1x1_2-3"),
            image_generation_selection=rt.options.get("image_generation_selection"),
        )

        async def _once() -> None:
            data = _require_ok(
                await regenerate_beats(rt.project, rt.episode, body, rt.user),
                f"渲染 {label}",
            )
            state = await _await_task(
                rt,
                "selected_regen",
                task_id=_task_id_of(data),
                scope=str(data.get("scope") or "") or None,
                label=f"渲染 {label}",
            )
            if state.status != "completed":
                raise StepFailed(state.error or state.status)

        try:
            await _run_chunk_with_retry(rt, label, _once)
        except PipelineCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — 单块失败不该废掉整集
            consecutive += 1
            failed.append({"beats": chunk, "error": str(exc)})
            rt.log(f"❌ {label} 渲染失败（第 {consecutive} 次连续失败）：{exc}")
            if consecutive >= _MAX_CONSECUTIVE_FAILURES:
                raise StepFailed(
                    f"连续 {consecutive} 次渲染失败，已停止以免烧完整集额度。"
                    f"最后一条错误：{exc}"
                ) from exc
            continue
        consecutive = 0
        done_beats.extend(chunk)
        rt.step_progress(
            StepId.RENDER,
            f"渲染 {len(done_beats)}/{len(missing)}（{label} 完成）",
            len(done_beats) / max(len(missing), 1),
        )
        _ = index

    if failed:
        rt.log(f"⚠️ {len(failed)} 块渲染失败，未出图的 beat 需重跑（重跑会自动跳过已有的）")

    await _wait_for_gate(
        rt,
        GateId.RENDER_SAMPLE,
        {
            "beats": done_beats,
            "failed": failed,
            "hint": "抽查画风、角色脸、背景是否一致，确认后继续生成视频提示词",
        },
    )
    return {"beats": len(done_beats), "existing": existing, "failed": failed}


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


#: 静默动作镜的默认时长。这类 beat 没台词、剧本也不标时长（实测 54 个 beat
#: 里有 25 个如此，与 audio_type=="silence" 完全重合），给一个够走完一个动作、
#: 又不至于拖沓的值。可用 options.silent_shot_seconds 调。
_SILENT_SHOT_SECONDS = 4


def _beat_video_duration(beat: dict[str, Any], *, silent_seconds: int = _SILENT_SHOT_SECONDS) -> int:
    """这一镜该出多长。

    原先一个都不传，全走模型默认的 5 秒——台词长的镜头话没说完就切走，
    静默镜干等，整集节奏全乱（实测剧本应有 144.8 秒，出成了 270.8 秒）。

    **不在这里夹紧边界**：值交给上游
    ``video_duration.normalize_video_duration_for_backend(backend, value)``
    按后端能力夹紧并取整，自己再夹一次只会与上游漂移。
    """
    from novelvideo.config import TTS_CHARS_PER_SECOND, TTS_DIALOGUE_CHARS_PER_SECOND

    for key in ("duration_seconds", "estimated_duration"):
        try:
            declared = float(beat.get(key) or 0)
        except (TypeError, ValueError):
            declared = 0.0
        if declared > 0:
            return max(1, int(round(declared)))

    text = str(beat.get("narration_segment") or "").strip()
    if not text:
        return max(1, silent_seconds)

    is_dialogue = str(beat.get("audio_type") or "") == "dialogue"
    rate = TTS_DIALOGUE_CHARS_PER_SECOND if is_dialogue else TTS_CHARS_PER_SECOND
    return max(1, int(round(len(text) / max(rate, 0.1))))


def _has_video(rt: Runtime, beat: dict[str, Any]) -> bool:
    """这个 beat 是否已经出过片。

    两处都认：beat 上的 video_url，以及盘上的 mp4。下载成功会写 video_url，
    但历史数据可能只有文件（早期手动跑的、或写库前就中断的），只看一处会漏。
    """
    from novelvideo.utils.path_resolver import PathResolver

    if str(beat.get("video_url") or "").strip():
        return True
    target = PathResolver(rt.output_dir, rt.episode).video(int(beat.get("beat_number") or 0))
    return target.exists() and target.stat().st_size > 0


def _existing_video_prompt(beat: dict[str, Any]) -> str:
    """取 beat 上已有的视频提示词。首尾帧模式用 keyframe_prompt。"""
    field = (
        "keyframe_prompt"
        if str(beat.get("video_mode") or "first_frame") == "keyframe"
        else "video_prompt"
    )
    return str(beat.get(field) or "").strip()


async def _reenhance_existing_prompts(
    rt: Runtime,
    store: Any,
    beats: list[dict[str, Any]],
    index_in_scene: dict[int, int],
    beat_by_number: dict[int, dict[str, Any]],
) -> int:
    """把存量提示词的约束块按当前规则重叠一遍。

    运镜正文（GlobalVideoOptimizer 读草图产出的那段）原样保留——它可以用
    strip_enhancement 从旧提示词里完整剥出来，所以**一次模型调用都不需要**。
    改了 prompt_enhancer 的规则之后想让整集跟上，用这个，别用 regenerate：
    后者每个 beat 都要再读一次草图，54 个 beat 就是 54 次视觉模型调用。
    """
    touched = 0
    for beat in beats:
        beat_num = int(beat.get("beat_number") or 0)
        existing = _existing_video_prompt(beat)
        if not existing:
            continue
        # 首尾帧模式的过渡提示词不叠运镜约束，跳过
        if str(beat.get("video_mode") or "first_frame") == "keyframe":
            continue
        base = strip_enhancement(existing)
        if not base:
            continue
        shot = build_shot_context(
            beat,
            scene_id=str((beat.get("scene_ref") or {}).get("scene_id") or ""),
            time_of_day=str(beat.get("time_of_day") or ""),
            index_in_scene=index_in_scene.get(beat_num, 0),
            prev_beat=beat_by_number.get(beat_num - 1),
        )
        final = enhance_video_prompt(base, shot)
        if not final or final == existing:
            continue
        await store.update_beat_asset(
            episode_number=rt.episode, beat_number=beat_num, video_prompt=final
        )
        beat["video_prompt"] = final   # 让后续的 _existing_video_prompt 看到新值
        touched += 1
    return touched


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

    beat_by_number = {int(b.get("beat_number") or 0): b for b in beats}
    generated = 0
    enhanced = 0
    skipped: list[int] = []
    reused: list[int] = []
    samples: list[dict[str, Any]] = []

    # prompt_refresh 决定已有提示词怎么处理：
    #   skip       —— 默认，已有就不动
    #   reenhance  —— 只重叠约束块：把旧约束剥掉、按当前规则重新叠。
    #                 运镜正文原样保留，**零模型调用、零成本**。改了 prompt_enhancer
    #                 之后想让存量提示词跟上新规则，用这个，别用 regenerate。
    #   regenerate —— 全部推倒重来，每个 beat 都要再调一次视觉模型读草图，花钱
    refresh = str(rt.options.get("prompt_refresh") or "skip").strip().lower()
    if refresh not in {"skip", "reenhance", "regenerate"}:
        rt.log(f"⚠️ 未知的 prompt_refresh「{refresh}」，按 skip 处理")
        refresh = "skip"

    if refresh == "reenhance":
        touched = await _reenhance_existing_prompts(
            rt, store, beats, index_in_scene, beat_by_number
        )
        rt.log(f"已重叠约束块：{touched} 个 beat（未调用任何模型）")

    # 已经有提示词的不重生成：重生成要再调一次视觉模型（读草图出运镜文案），
    # 花钱，而且会把人工在虾镜里改过的文案覆盖掉。
    if refresh == "regenerate":
        pending = list(beats)
        reused = []
        rt.log(f"prompt_refresh=regenerate：{len(pending)} 个 beat 全部重新生成（会调模型）")
    else:
        pending = [beat for beat in beats if not _existing_video_prompt(beat)]
        reused = [
            int(beat.get("beat_number") or 0)
            for beat in beats
            if _existing_video_prompt(beat)
        ]
        if reused:
            rt.log(f"{len(reused)} 个 beat 已有视频提示词，跳过；只生成 {len(pending)} 个")
    if not pending:
        rt.log("全部 beat 都已有视频提示词，整步跳过")
        for beat in beats[:3]:
            samples.append(
                {
                    "beat_number": int(beat.get("beat_number") or 0),
                    "prompt": _existing_video_prompt(beat),
                }
            )
        return {
            "samples": samples,
            "stats": {
                "prompts_generated": 0,
                "prompts_enhanced": 0,
                "prompts_reused": reused,
                "prompts_skipped": [],
            },
        }

    for position, beat in enumerate(pending):
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
            prev_beat=beat_by_number.get(beat_num - 1),
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
            f"视频提示词 {position + 1}/{len(pending)}（Beat {beat_num}）",
            (position + 1) / max(len(pending), 1),
        )

    if skipped:
        rt.log(f"⚠️ {len(skipped)} 个 beat 没有提示词：{skipped}，出片时会跳过")
    return {
        "samples": samples,
        "stats": {
            "prompts_generated": generated,
            "prompts_enhanced": enhanced,
            "prompts_reused": reused,
            "prompts_skipped": skipped,
        },
    }


# ── 第 6 步：逐 beat 出视频（闸门 5）─────────────────────────────────────


async def run_videos(rt: Runtime) -> dict[str, Any]:
    from novelvideo.api.routes.generation import generate_single_video
    from novelvideo.api.schemas import SingleVideoRequest

    _store, _script, beats = await _load_script(rt)

    # 已经有成片的不再出：出片按秒计费，重出一遍就是白烧一次钱。
    # 判定看两处——beat 上的 video_url，以及盘上的 mp4；
    # 任一存在即视为已出（下载成功会写 video_url，但历史数据可能只有文件）。
    pending = [b for b in beats if not _has_video(rt, b)]
    already = [int(b.get("beat_number") or 0) for b in beats if _has_video(rt, b)]
    if already:
        rt.log(f"{len(already)} 个 beat 已有成片，跳过；只出 {len(pending)} 个")
    if not pending:
        rt.log(f"全部 {len(beats)} 个 beat 都已有成片，整步跳过（不进出片闸门，不计费）")
        return {"total": 0, "succeeded": 0, "failed": [], "existing": len(beats)}

    # 闸门 5 是钱闸：过了这道门才开始按秒计费。不预授权、也不来点头，
    # 超时走 GateTimedOutSkip——出片步标 skipped，前面的成果照常保留。
    await _wait_for_gate(
        rt,
        GateId.VIDEO_GENERATION,
        {
            "beats": len(pending),
            "existing": len(already),
            "cost_estimate": dict(rt.state.cost_estimate),
            "hint": (
                "确认后开始批量出片（按秒计费）。不确认则到时自动跳过出片，"
                "已生成的图与提示词全部保留"
            ),
        },
        skip_on_timeout=True,
    )

    backend = str(rt.options.get("video_backend") or "grok_720")
    # resolution/ratio 只在显式给了值时才塞进请求体：路由用
    # `"resolution" in model_fields_set` 判断用户有没有动过它，白填一个
    # 默认值会改变计费口径与 seedance2 分支的行为。grok_720 固定 720p，
    # 本来也不吃这两个字段。
    # duration 不同——它必须逐 beat 给，见 _beat_video_duration。
    extra: dict[str, Any] = {}
    for key in ("resolution", "ratio"):
        value = rt.options.get(key)
        if value not in (None, ""):
            extra[key] = value
    forced_duration = rt.options.get("duration")
    try:
        silent_seconds = int(rt.options.get("silent_shot_seconds") or _SILENT_SHOT_SECONDS)
    except (TypeError, ValueError):
        silent_seconds = _SILENT_SHOT_SECONDS
    semaphore = asyncio.Semaphore(max(1, int(rt.state.video_concurrency or 3)))

    done = 0
    failed: list[dict[str, Any]] = []
    total = len(pending)

    async def _one(beat: dict[str, Any]) -> None:
        nonlocal done
        beat_num = int(beat.get("beat_number") or 0)
        async with semaphore:
            rt.raise_if_cancelled()
            duration = forced_duration or _beat_video_duration(
                beat, silent_seconds=silent_seconds
            )
            body = SingleVideoRequest(
                video_backend=backend, duration=duration, **extra
            )
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

    await asyncio.gather(*(_one(beat) for beat in pending))

    rt.state.failed_beats = [int(item["beat_number"]) for item in failed]
    if failed:
        rt.log(
            f"⚠️ {len(failed)} 个 beat 未出片：{rt.state.failed_beats}。"
            "这些位置在成片里是缺口，会断剧情连续性，需单独重试后再合成。"
        )
    if total and done == 0:
        # 一个都没出来还报「完成」，用户会以为可以去合成了
        raise StepFailed(f"{total} 个 beat 全部出片失败，第一条错误：{failed[0]['error']}")
    return {
        "total": total,
        "succeeded": done,
        "failed": failed,
        "existing": len(already),
    }


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
