// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { jsonWithBackendError } from "@/lib/api-errors";
import { p } from "@/lib/api-path";
import { queryKeys } from "@/lib/query-keys";
import type { OkResponse, TaskResponse } from "@/types/api";

export type BatchStepId =
  | "preflight"
  | "plan_assets"
  | "match_scenes"
  | "sketches"
  | "render"
  | "videos";

export type BatchStepStatus =
  | "pending"
  | "running"
  | "waiting_gate"
  | "done"
  | "failed"
  | "skipped";

export type BatchGateId =
  | "scene_assignment"
  | "sketch_sample"
  | "render_sample";

/** 闸门 1 决定后面每一张图的背景，服务端不接受预授权。 */
export const MANDATORY_GATES: readonly BatchGateId[] = ["scene_assignment"];

export interface BatchStepState {
  step_id: BatchStepId;
  status: BatchStepStatus;
  progress: number;
  message: string;
  error: string;
  result: Record<string, unknown>;
}

export interface BatchGateState {
  gate_id: BatchGateId;
  required: boolean;
  approved: boolean;
  open: boolean;
  payload: Record<string, unknown>;
}

export interface BatchCostEstimate {
  beats: number;
  video_seconds: number;
  video_price_per_second: number;
  video_usd: number;
  resolution: string;
  sketch_images: number;
  render_images: number;
  note: string;
}

export interface BatchPipelineState {
  project: string;
  episode: number;
  cancelled: boolean;
  resolution: string;
  video_concurrency: number;
  failed_beats: number[];
  cost_estimate: Partial<BatchCostEstimate>;
  steps: BatchStepState[];
  gates: BatchGateState[];
  current_step: BatchStepId | "";
}

/** 闸门 1 的确认材料：场景分配表。 */
export interface SceneAssignmentRow {
  beat_number: number;
  scene_id: string;
  time_of_day: string;
  confidence: "high" | "medium" | "low";
  evidence: string;
  changed: boolean;
}

export interface SceneAssignmentTable {
  total: number;
  changed: number;
  low_confidence: number[];
  by_scene: Array<{
    scene_id: string;
    time_of_day: string;
    count: number;
    beat_range: string;
  }>;
  rows: SceneAssignmentRow[];
}

export interface StartBatchPipelineParams {
  autoApprove?: BatchGateId[];
  startFrom?: BatchStepId;
  resolution?: string;
  videoConcurrency?: number;
  options?: Record<string, unknown>;
}

function base(project: string, episode: number) {
  return p`api/v1/projects/${project}/episodes/${episode}/batch-pipeline`;
}

export function useBatchPipelineStatus(
  project: string,
  episode: number,
  /** 弹窗关着就不查——闲置时不打服务端。 */
  enabled = false,
) {
  return useQuery({
    queryKey: queryKeys.batchPipelineStatus(project, episode),
    queryFn: ({ signal }) =>
      jsonWithBackendError<OkResponse<BatchPipelineState>>(
        api.get(`${base(project, episode)}/status`, {
          signal,
          throwHttpErrors: false,
        }),
      ),
    enabled: enabled && !!project && episode > 0,
    // 轮询节奏由数据自己决定：在跑（含卡在闸门）才 2s 一次，否则停。
    // 写成函数而不是外部布尔量，避免同一 key 被两个 useQuery 用不同
    // 间隔订阅时互相覆盖。
    refetchInterval: (query) =>
      isPipelineActive(query.state.data?.data) ? 2000 : false,
  });
}

export function useBatchPipelineEstimate(
  project: string,
  episode: number,
  resolution = "720p",
) {
  return useQuery({
    queryKey: [
      ...queryKeys.batchPipelineStatus(project, episode),
      "estimate",
      resolution,
    ] as const,
    queryFn: ({ signal }) =>
      jsonWithBackendError<OkResponse<BatchCostEstimate>>(
        api.get(`${base(project, episode)}/estimate?resolution=${resolution}`, {
          signal,
          throwHttpErrors: false,
        }),
      ),
    enabled: !!project && episode > 0,
  });
}

export function useStartBatchPipeline(project: string, episode: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: StartBatchPipelineParams = {}) =>
      jsonWithBackendError<
        TaskResponse & { state: BatchPipelineState; rejected_auto_approve: string[] }
      >(
        api.post(`${base(project, episode)}/start`, {
          json: {
            auto_approve: params.autoApprove ?? [],
            start_from: params.startFrom ?? null,
            resolution: params.resolution ?? "720p",
            video_concurrency: params.videoConcurrency ?? 2,
            options: params.options ?? {},
          },
          throwHttpErrors: false,
        }),
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.batchPipelineStatus(project, episode),
      });
    },
  });
}

export function useDecideBatchGate(project: string, episode: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      gateId,
      approved,
      payload,
    }: {
      gateId: BatchGateId;
      approved: boolean;
      payload?: Record<string, unknown>;
    }) =>
      jsonWithBackendError<OkResponse<BatchPipelineState>>(
        api.post(`${base(project, episode)}/gates/${gateId}`, {
          json: { approved, payload: payload ?? null },
          throwHttpErrors: false,
        }),
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.batchPipelineStatus(project, episode),
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.beats(project, episode),
      });
    },
  });
}

export function useCancelBatchPipeline(project: string, episode: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      jsonWithBackendError<OkResponse<BatchPipelineState>>(
        api.post(`${base(project, episode)}/cancel`, { throwHttpErrors: false }),
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.batchPipelineStatus(project, episode),
      });
    },
  });
}

/** 流水线是否还在跑（含卡在闸门上）——决定要不要开轮询。 */
export function isPipelineActive(state: BatchPipelineState | undefined): boolean {
  if (!state || state.cancelled) return false;
  return state.steps.some(
    (step) => step.status === "running" || step.status === "waiting_gate",
  );
}

/** 正在等人确认的闸门。没有则返回 null。 */
export function pendingGate(
  state: BatchPipelineState | undefined,
): BatchGateState | null {
  if (!state) return null;
  const running = state.steps.find((step) => step.status === "running");
  if (!running) return null;
  const gateForStep: Partial<Record<BatchStepId, BatchGateId>> = {
    match_scenes: "scene_assignment",
    sketches: "sketch_sample",
    render: "render_sample",
  };
  const gateId = gateForStep[running.step_id];
  if (!gateId) return null;
  const gate = state.gates.find((g) => g.gate_id === gateId);
  if (!gate || gate.open) return null;
  // payload 是进闸门时才写的；空 payload 说明这一步还没跑到闸门
  return Object.keys(gate.payload ?? {}).length > 0 ? gate : null;
}
