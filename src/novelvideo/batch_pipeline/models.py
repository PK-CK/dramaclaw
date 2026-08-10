"""批量流水线的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepId(str, Enum):
    """六个步骤。顺序即执行顺序。"""

    PREFLIGHT = "preflight"          # 第 0 步：前置体检（零成本）
    PLAN_ASSETS = "plan_assets"      # 第 1 步：场景 + 道具规划
    MATCH_SCENES = "match_scenes"    # 第 2 步：场次交叉验证 → 补 scene_ref
    SKETCHES = "sketches"            # 第 3 步：批量草图
    RENDER = "render"                # 第 4 步：批量渲染 ⊕ 视频提示词
    VIDEOS = "videos"                # 第 5 步：逐 beat 出视频


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_GATE = "waiting_gate"    # 卡在闸门，等人确认
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"              # 前置条件已满足，无需执行


class GateId(str, Enum):
    """三个闸门。闸门 1 强制人工确认，2/3 可预授权自动放行。"""

    SCENE_ASSIGNMENT = "scene_assignment"  # 闸门 1：场景分配表（决定后续所有图的背景）
    SKETCH_SAMPLE = "sketch_sample"        # 闸门 2：草图抽查（看姿势）
    RENDER_SAMPLE = "render_sample"        # 闸门 3：渲染抽查（看画风/角色脸/背景）


#: 闸门 1 不允许预授权：它决定后面每一张图的背景，错了要全部重出。
MANDATORY_GATES: frozenset[GateId] = frozenset({GateId.SCENE_ASSIGNMENT})


@dataclass
class SceneHeading:
    """从剧本原文解析出的一个场次头。"""

    order: int              # 在本集内的序号，从 1 起
    scene_id: str           # 地点，如「市立医院急诊走廊」
    time_of_day: str        # 已归一到系统规范值
    raw_time: str           # 剧本原文的时段词，如「凌晨」
    interior: str           # 内 / 外 / 空
    line_no: int            # 在原文中的行号（1 起）


@dataclass
class BeatAssignment:
    """一个 beat 的场景归属判定结果。"""

    beat_number: int
    scene_id: str
    time_of_day: str
    confidence: str         # high / medium / low
    evidence: str           # 判定依据，供人工复核
    changed: bool = False   # 与 beat 现有值相比是否有变化


@dataclass
class GateState:
    gate_id: GateId
    required: bool                       # 是否必须人工确认
    approved: bool = False
    payload: dict[str, Any] = field(default_factory=dict)   # 给人看的材料


@dataclass
class StepState:
    step_id: StepId
    status: StepStatus = StepStatus.PENDING
    progress: float = 0.0
    message: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """整条流水线的状态。可序列化落库，用于断点续跑。"""

    project: str
    episode: int
    steps: dict[str, StepState] = field(default_factory=dict)
    gates: dict[str, GateState] = field(default_factory=dict)
    auto_approve: list[str] = field(default_factory=list)   # 预授权放行的闸门
    video_concurrency: int = 2
    resolution: str = "720p"
    failed_beats: list[int] = field(default_factory=list)
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def step(self, step_id: StepId) -> StepState:
        return self.steps.setdefault(step_id.value, StepState(step_id=step_id))

    def gate(self, gate_id: GateId) -> GateState:
        return self.gates.setdefault(
            gate_id.value,
            GateState(gate_id=gate_id, required=gate_id in MANDATORY_GATES),
        )

    def gate_is_open(self, gate_id: GateId) -> bool:
        """闸门是否放行：已人工确认，或非强制且被预授权。"""
        state = self.gate(gate_id)
        if state.approved:
            return True
        if gate_id in MANDATORY_GATES:
            return False
        return gate_id.value in self.auto_approve

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "episode": self.episode,
            "steps": {
                k: {
                    "step_id": v.step_id.value,
                    "status": v.status.value,
                    "progress": v.progress,
                    "message": v.message,
                    "error": v.error,
                    "result": v.result,
                }
                for k, v in self.steps.items()
            },
            "gates": {
                k: {
                    "gate_id": v.gate_id.value,
                    "required": v.required,
                    "approved": v.approved,
                    "payload": v.payload,
                }
                for k, v in self.gates.items()
            },
            "auto_approve": list(self.auto_approve),
            "video_concurrency": self.video_concurrency,
            "resolution": self.resolution,
            "failed_beats": list(self.failed_beats),
            "cost_estimate": dict(self.cost_estimate),
            "cancelled": self.cancelled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        state = cls(
            project=str(data.get("project") or ""),
            episode=int(data.get("episode") or 0),
            auto_approve=list(data.get("auto_approve") or []),
            video_concurrency=int(data.get("video_concurrency") or 2),
            resolution=str(data.get("resolution") or "720p"),
            failed_beats=list(data.get("failed_beats") or []),
            cost_estimate=dict(data.get("cost_estimate") or {}),
            cancelled=bool(data.get("cancelled")),
        )
        for key, raw in (data.get("steps") or {}).items():
            state.steps[key] = StepState(
                step_id=StepId(raw.get("step_id") or key),
                status=StepStatus(raw.get("status") or "pending"),
                progress=float(raw.get("progress") or 0.0),
                message=str(raw.get("message") or ""),
                error=str(raw.get("error") or ""),
                result=dict(raw.get("result") or {}),
            )
        for key, raw in (data.get("gates") or {}).items():
            gid = GateId(raw.get("gate_id") or key)
            state.gates[key] = GateState(
                gate_id=gid,
                required=bool(raw.get("required", gid in MANDATORY_GATES)),
                approved=bool(raw.get("approved")),
                payload=dict(raw.get("payload") or {}),
            )
        return state
