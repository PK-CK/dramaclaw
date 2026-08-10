"""批量流水线的状态存取与成本预估。

状态落在 ``<output_dir>/batch_pipeline/ep<NNN>/state.json``：闸门确认是由
另一个请求写进来的带外信号（跟 cancel flag 同一个道理），任务在跑的过程中
要能读到，所以不能只放在任务进程的内存里。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from novelvideo.batch_pipeline.models import (
    GateId,
    PipelineState,
    StepId,
    StepStatus,
)

#: 720p grok-imagine-video 官方价（美元/秒）。上游调价时用环境变量覆盖，
#: 不要改这个常量——它同时是「没配置时按什么算」的文档。
DEFAULT_VIDEO_PRICE_PER_SECOND = 0.07


def state_path(output_dir: str | Path, episode: int) -> Path:
    return Path(output_dir) / "batch_pipeline" / f"ep{episode:03d}" / "state.json"


def load_state(output_dir: str | Path, episode: int, project: str = "") -> PipelineState:
    path = state_path(output_dir, episode)
    if path.exists():
        try:
            return PipelineState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            # 状态文件坏了不该让整条流水线打不开——退回全新状态，
            # 大不了重跑一遍（每一步都会先查产物是否已就绪）。
            pass
    return PipelineState(project=project, episode=episode)


def save_state(output_dir: str | Path, state: PipelineState) -> Path:
    """原子写。任务与 REST 请求会并发写这个文件，直接 open(w) 可能读到半截。"""
    path = state_path(output_dir, state.episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def approve_gate(
    output_dir: str | Path,
    episode: int,
    gate_id: GateId,
    *,
    approved: bool = True,
    payload: dict[str, Any] | None = None,
) -> PipelineState:
    state = load_state(output_dir, episode)
    gate = state.gate(gate_id)
    gate.approved = approved
    if payload is not None:
        gate.payload = {**gate.payload, **payload}
    save_state(output_dir, state)
    return state


def request_cancel(output_dir: str | Path, episode: int) -> PipelineState:
    state = load_state(output_dir, episode)
    state.cancelled = True
    save_state(output_dir, state)
    return state


def video_price_per_second() -> float:
    raw = os.environ.get("BATCH_PIPELINE_VIDEO_PRICE_PER_SECOND", "").strip()
    try:
        price = float(raw)
    except ValueError:
        return DEFAULT_VIDEO_PRICE_PER_SECOND
    return price if price > 0 else DEFAULT_VIDEO_PRICE_PER_SECOND


def estimate_cost(beats: list[dict[str, Any]], *, resolution: str = "720p") -> dict[str, Any]:
    """预估这一集要花多少。

    只有视频给出金额——它按秒计价，算得准。草图与渲染走各自的图像模型
    与积分口径，这里只报张数，不编价格。
    """
    total_seconds = 0.0
    for beat in beats:
        raw = beat.get("duration_seconds") or beat.get("estimated_duration") or 0
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            seconds = 0.0
        # grok-imagine-video：不传时长按 8 秒计费，超过 15 秒会被上游夹紧
        total_seconds += min(max(seconds, 8.0), 15.0)

    price = video_price_per_second()
    return {
        "beats": len(beats),
        "video_seconds": round(total_seconds, 1),
        "video_price_per_second": price,
        "video_usd": round(total_seconds * price, 2),
        "resolution": resolution,
        "sketch_images": len(beats),
        "render_images": len(beats),
        "note": "仅视频按秒估价；草图与渲染的费用取决于所选图像模型，未计入",
    }


def summarize(state: PipelineState) -> dict[str, Any]:
    """给前端的精简视图：六步进度 + 三闸门状态。"""
    return {
        "project": state.project,
        "episode": state.episode,
        "cancelled": state.cancelled,
        "resolution": state.resolution,
        "video_concurrency": state.video_concurrency,
        "failed_beats": list(state.failed_beats),
        "cost_estimate": dict(state.cost_estimate),
        "steps": [
            {
                "step_id": step.value,
                "status": state.step(step).status.value,
                "progress": state.step(step).progress,
                "message": state.step(step).message,
                "error": state.step(step).error,
                "result": state.step(step).result,
            }
            for step in StepId
        ],
        "gates": [
            {
                "gate_id": gate.value,
                "required": state.gate(gate).required,
                "approved": state.gate(gate).approved,
                "open": state.gate_is_open(gate),
                "payload": state.gate(gate).payload,
            }
            for gate in GateId
        ],
        "current_step": next(
            (
                step.value
                for step in StepId
                if state.step(step).status
                in (StepStatus.RUNNING, StepStatus.WAITING_GATE)
            ),
            "",
        ),
    }
