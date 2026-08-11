"""批量流水线端点：启动 / 查状态 / 过闸门 / 取消。

鉴权在这里一次性把关（``tasks:submit`` + 项目 editor 角色）。runner 里
直接调用的那些路由函数不会再走 FastAPI 依赖注入，所以这道门必须守住。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from novelvideo.api.auth import get_api_user, require_scope
from novelvideo.api.deps import make_sqlite_store_for_context, resolve_project_scope
from novelvideo.batch_pipeline import service
from novelvideo.batch_pipeline.models import GateId, PipelineState, StepId
from novelvideo.ports import get_task_backend
from novelvideo.task_backend.runners.batch_pipeline import TASK_TYPE
from novelvideo.task_identity import project_task_state_key
from novelvideo.task_state import get_task_manager

logger = logging.getLogger("novelvideo.api.batch_pipeline")

router = APIRouter()

_GATE_VALUES = {gate.value for gate in GateId}
_STEP_VALUES = {step.value for step in StepId}
#: 任务已结束的状态。与 batch_pipeline.steps._TERMINAL 同源。
_TERMINAL_TASK_STATUS = {"completed", "failed", "cancelled"}


class BatchPipelineStartRequest(BaseModel):
    #: 预授权放行的闸门。闸门 1（场景分配表）不接受预授权，服务端会剔除。
    auto_approve: list[str] = Field(default_factory=list)
    start_from: Optional[str] = None
    resolution: str = "720p"
    video_concurrency: int = 3
    #: 透传给各步骤的生成参数（风格、图像模型、视频后端等）
    options: dict[str, Any] = Field(default_factory=dict)


class GateDecisionRequest(BaseModel):
    approved: bool = True
    #: 人工在闸门上做的修正，原样并进 gate.payload 供审计
    payload: Optional[dict[str, Any]] = None


def _max_concurrent_pipelines() -> int:
    """同时最多几条流水线。

    runner 与它的草图/渲染子任务共用 default 车道，所以只能占一半槽，
    另一半留给子任务；至少允许 1 条，否则功能等于关掉。
    """
    from novelvideo.task_backend.limits import global_lane_concurrency

    return max(1, global_lane_concurrency("default") // 2)


def _active_pipeline_count(manager, ctx) -> int:
    """统计这个用户名下还在跑的流水线条数（跨项目、跨集）。

    车道并发是**全实例**的，严格说应该全局统计；这里只统计到用户一级，
    因为跨用户的活跃任务没有现成的查询接口，而真正的准入控制属于
    task backend 的职责，不该在这条业务路由里另造一套。CE 单用户部署下
    两者等价；多用户实例仍可能被多人叠加占满车道。
    """
    seen: set[str] = set()
    active = 0
    for task in manager.list_tasks_for_user(ctx.owner_username):
        if task.task_type != TASK_TYPE or task.status in _TERMINAL_TASK_STATUS:
            continue
        key = f"{task.project_id}:{task.episode}"
        if key in seen:
            continue
        seen.add(key)
        active += 1
    return active


def _validated_auto_approve(values: list[str]) -> list[str]:
    from novelvideo.batch_pipeline.models import MANDATORY_GATES

    mandatory = {gate.value for gate in MANDATORY_GATES}
    return [v for v in values if v in _GATE_VALUES and v not in mandatory]


@router.post("/projects/{project}/episodes/{episode_num}/batch-pipeline/start")
async def start_batch_pipeline(
    project: str,
    episode_num: int,
    body: BatchPipelineStartRequest,
    user: dict = Depends(require_scope("tasks:submit")),
):
    """启动批量流水线。"""
    resolved = await resolve_project_scope(project, user, required_role="editor")
    ctx = resolved.ctx
    if ctx is None:
        raise HTTPException(status_code=400, detail="批量流水线需要 project context")

    if body.start_from and body.start_from not in _STEP_VALUES:
        raise HTTPException(status_code=400, detail=f"未知的起始步骤: {body.start_from}")

    manager = get_task_manager()

    # 同一集只允许有一条在跑：两条会抢写同一份 state.json，闸门与取消
    # 标记互相覆盖，而且各自都在等对方入队的子任务。
    running = manager.get_task_for_project(ctx, TASK_TYPE, episode_num)
    if running is not None and running.status not in _TERMINAL_TASK_STATUS:
        raise HTTPException(
            status_code=409,
            detail=f"第 {episode_num} 集已有批量流水线在跑（{running.status}），请先取消或等它结束",
        )

    # 跨集也要限：runner 自己占 default 车道一个槽，它的草图/渲染子任务
    # 也回 default 抢槽。同时开的集数一旦占满这条车道，所有 runner 都在等
    # 各自永远排不上的子任务——整体饿死到超时为止。留一半槽给子任务。
    active = _active_pipeline_count(manager, ctx)
    if active >= _max_concurrent_pipelines():
        raise HTTPException(
            status_code=409,
            detail=(
                f"已有 {active} 集批量流水线在跑，达到并发上限 "
                f"{_max_concurrent_pipelines()}（再多会把任务车道占满，子任务排不上）。"
                "请等其中一集结束或先取消。"
            ),
        )

    store = await make_sqlite_store_for_context(ctx)
    script = await store.get_script_as_dict(episode_num)
    beats = list((script or {}).get("beats") or [])
    if not beats:
        raise HTTPException(status_code=409, detail=f"第 {episode_num} 集还没有剧本 beat")

    dropped = set(body.auto_approve) - set(_validated_auto_approve(body.auto_approve))
    auto_approve = _validated_auto_approve(body.auto_approve)

    # 重新起跑：清掉上一轮的闸门确认，否则会拿着旧的确认直接冲过去
    state = PipelineState(
        project=ctx.project_id,
        episode=episode_num,
        auto_approve=auto_approve,
        video_concurrency=max(1, body.video_concurrency),
        resolution=body.resolution,
        cost_estimate=service.estimate_cost(beats, resolution=body.resolution),
    )
    # 先备份旧状态：下面这行覆写在入队之前，入队若抛异常，上一轮的步骤
    # 结果与闸门确认已经被抹平，而新任务并没有跑起来。
    previous = service.load_state(resolved.output_dir, episode_num, project=ctx.project_id)
    service.save_state(resolved.output_dir, state)

    try:
        queued = await get_task_backend().enqueue_project_task(
            ctx,
            product_surface="mainline",
            task_type=TASK_TYPE,
            queue_kind="default",
            episode=episode_num,
            payload={
                "episode": episode_num,
                "output_dir": resolved.output_dir,
                "config": {
                    "auto_approve": auto_approve,
                    "start_from": body.start_from or "",
                    "resolution": body.resolution,
                    "video_concurrency": max(1, body.video_concurrency),
                    "options": body.options,
                },
            },
        )
    except Exception:
        service.save_state(resolved.output_dir, previous)
        raise
    logger.info("[%s] EP%d 批量流水线已入队 task=%s", project, episode_num, queued.task_state.task_id)

    return {
        "ok": True,
        "task_type": TASK_TYPE,
        "task_id": queued.task_state.task_id,
        "task_key": project_task_state_key(TASK_TYPE, ctx.project_id, episode_num),
        "backend": queued.backend,
        "queue": queued.queue,
        "state": service.summarize(state),
        "rejected_auto_approve": sorted(dropped),
        "message": f"第 {episode_num} 集批量流水线已进入队列",
    }


@router.get("/projects/{project}/episodes/{episode_num}/batch-pipeline/status")
async def batch_pipeline_status(
    project: str,
    episode_num: int,
    user: dict = Depends(get_api_user),
):
    resolved = await resolve_project_scope(project, user, required_role="viewer")
    state = service.load_state(resolved.output_dir, episode_num, project=project)
    return {"ok": True, "data": service.summarize(state)}


@router.post("/projects/{project}/episodes/{episode_num}/batch-pipeline/gates/{gate_id}")
async def decide_batch_pipeline_gate(
    project: str,
    episode_num: int,
    gate_id: str,
    body: GateDecisionRequest,
    # 放行闸门实质是授权后续子任务提交与开销，语义上属 tasks:submit，
    # 与 start 端点保持同一强度。
    user: dict = Depends(require_scope("tasks:submit")),
):
    """确认或驳回一个闸门。驳回等同取消——后面的步骤都建立在它之上。"""
    if gate_id not in _GATE_VALUES:
        raise HTTPException(status_code=404, detail=f"未知闸门: {gate_id}")
    resolved = await resolve_project_scope(project, user, required_role="editor")

    if not body.approved:
        state = service.request_cancel(resolved.output_dir, episode_num)
        return {"ok": True, "data": service.summarize(state), "message": "已驳回，流水线停止"}

    state = service.approve_gate(
        resolved.output_dir,
        episode_num,
        GateId(gate_id),
        approved=True,
        payload=body.payload,
    )
    return {"ok": True, "data": service.summarize(state), "message": f"闸门 {gate_id} 已放行"}


@router.post("/projects/{project}/episodes/{episode_num}/batch-pipeline/cancel")
async def cancel_batch_pipeline(
    project: str,
    episode_num: int,
    user: dict = Depends(require_scope("tasks:submit")),
):
    """取消。

    协作式取消：已经发出去的出片请求不会被撤回，会跑完并按秒计费。
    这是钱的知情权，必须写进用户可见的 message，不能只留在 docstring 里。
    """
    resolved = await resolve_project_scope(project, user, required_role="editor")
    state = service.request_cancel(resolved.output_dir, episode_num)
    return {
        "ok": True,
        "data": service.summarize(state),
        "message": "已请求取消。注意：此刻已经发出的出片请求会跑完并计费，之后不再有新的开销。",
    }


@router.get("/projects/{project}/episodes/{episode_num}/batch-pipeline/estimate")
async def estimate_batch_pipeline(
    project: str,
    episode_num: int,
    resolution: str = "720p",
    user: dict = Depends(get_api_user),
):
    """开跑前的成本预估，零开销。"""
    resolved = await resolve_project_scope(project, user, required_role="viewer")
    store = await make_sqlite_store_for_context(resolved.ctx)
    script = await store.get_script_as_dict(episode_num)
    beats = list((script or {}).get("beats") or [])
    return {"ok": True, "data": service.estimate_cost(beats, resolution=resolution)}
