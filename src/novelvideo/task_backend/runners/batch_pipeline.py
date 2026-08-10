"""批量流水线 runner。

它自己在 default 车道占一个槽，然后把子任务照常入队（草图/渲染走 default
的其余 7 个槽，出片走 video 车道的 2 个槽），所以不会自己等自己。
"""

from __future__ import annotations

import asyncio
from typing import Any

from novelvideo.project_context import ProjectContext
from novelvideo.task_backend.cancel import await_envelope_with_cancel_watch
from novelvideo.task_backend.registry import register_project_task_runner
from novelvideo.task_state import get_task_manager

TASK_TYPE = "batch_pipeline"


async def _run_batch_pipeline_async(
    envelope: dict[str, Any], ctx: ProjectContext
) -> dict[str, Any]:
    from novelvideo.batch_pipeline import service
    from novelvideo.batch_pipeline.models import GateId, StepId
    from novelvideo.batch_pipeline.steps import PipelineCancelled, Runtime, run_pipeline

    payload = envelope.get("payload") or {}
    config = payload.get("config") or {}
    episode = int(payload.get("episode") or 0)
    output_dir = str(payload.get("output_dir") or ctx.output_dir)

    manager = get_task_manager()

    def log(message: str, *, progress: float | None = None) -> None:
        manager.update_progress_for_project(
            ctx,
            TASK_TYPE,
            episode,
            scope=None,
            progress=progress,
            current_task=message,
            logs=[message],
        )

    state = service.load_state(output_dir, episode, project=ctx.project_id)
    state.cancelled = False
    state.auto_approve = [
        gate for gate in (config.get("auto_approve") or []) if gate in {g.value for g in GateId}
    ]
    state.video_concurrency = max(1, int(config.get("video_concurrency") or 2))
    state.resolution = str(config.get("resolution") or "720p")

    start_from = None
    raw_start = str(config.get("start_from") or "").strip()
    if raw_start:
        try:
            start_from = StepId(raw_start)
        except ValueError:
            log(f"⚠️ 未知的起始步骤「{raw_start}」，从头开始")

    rt = Runtime(
        ctx=ctx,
        episode=episode,
        output_dir=output_dir,
        state=state,
        log=log,
        options=dict(config.get("options") or {}),
    )
    rt.save()

    try:
        result = await run_pipeline(rt, start_from=start_from)
    except PipelineCancelled as exc:
        log(f"已取消：{exc}")
        return {"ok": False, "cancelled": True, "state": service.summarize(rt.state)}

    log("批量流水线完成", progress=1.0)
    return {"ok": True, **result, "state": service.summarize(rt.state)}


def run_batch_pipeline(envelope: dict[str, Any], ctx: ProjectContext) -> dict[str, Any]:
    return asyncio.run(
        await_envelope_with_cancel_watch(
            _run_batch_pipeline_async(envelope, ctx),
            envelope,
            task_type=TASK_TYPE,
        )
    )


register_project_task_runner(TASK_TYPE, run_batch_pipeline)
