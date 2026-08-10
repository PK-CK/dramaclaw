"""批量流水线：把虾镜里逐个点的六步串成一次操作。

对外只暴露数据结构与状态服务；步骤实现（steps）依赖 api.routes，
在 runner 里按需导入，避免 import 环。
"""

from novelvideo.batch_pipeline.models import (
    MANDATORY_GATES,
    GateId,
    PipelineState,
    StepId,
    StepStatus,
)

__all__ = [
    "MANDATORY_GATES",
    "GateId",
    "PipelineState",
    "StepId",
    "StepStatus",
]
