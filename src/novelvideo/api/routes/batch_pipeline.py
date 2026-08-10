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

logger = logging.getLogger("novelvideo.api.batch_pipeline")

router = APIRouter()

_GATE_VALUES = {gate.value for gate in GateId}
_STEP_VALUES = {step.value for step in StepId}


class BatchPipelineStartRequest(BaseModel):
    #: 预授权放行的闸门。闸门 1（场景分配表）不接受预授权，服务端会剔除。
    auto_approve: list[str] = Field(default_factory=list)
    start_from: Optional[str] = None
    resolution: str = "720p"
    video_concurrency: int = 2
    #: 透传给各步骤的生成参数（风格、图像模型、视频后端等）
    options: dict[str, Any] = Field(default_factory=dict)


class GateDecisionRequest(BaseModel):
    approved: bool = True
    #: 人工在闸门上做的修正，原样并进 gate.payload 供审计
    payload: Optional[dict[str, Any]] = None


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
    service.save_state(resolved.output_dir, state)

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
    user: dict = Depends(get_api_user),
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
    user: dict = Depends(get_api_user),
):
    """取消。已入队的子任务不会被回滚，但不会再有新的开销。"""
    resolved = await resolve_project_scope(project, user, required_role="editor")
    state = service.request_cancel(resolved.output_dir, episode_num)
    return {"ok": True, "data": service.summarize(state), "message": "已请求取消"}


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
