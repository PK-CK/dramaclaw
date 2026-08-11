// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { AlertTriangle, Check, CircleDashed, Loader2, PauseCircle, X } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GLASS_ALERT_DIALOG_CONTENT_CLASS } from "@/lib/dialog-styles";
import { backendErrorToastMessage } from "@/lib/api-errors";
import { cn } from "@/lib/utils";
import {
  isPipelineActive,
  pendingGate,
  useBatchPipelineEstimate,
  useBatchPipelineStatus,
  useCancelBatchPipeline,
  useDecideBatchGate,
  useStartBatchPipeline,
  type BatchGateId,
  type BatchStepId,
  type BatchStepState,
  type SceneAssignmentTable,
} from "@/lib/queries/batch-pipeline";

const STEP_ORDER: BatchStepId[] = [
  "preflight",
  "plan_assets",
  "match_scenes",
  "sketches",
  "render",
  "video_prompts",
  "videos",
];

/** 可预授权的四个闸门。闸门 1 服务端硬拒，不放进来。 */
const PREAUTHORIZABLE: BatchGateId[] = [
  "sketch_sample",
  "render_sample",
  "video_prompts",
  "video_generation",
];

interface BatchPipelineDialogProps {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  project: string;
  episode: number;
}

function StepIcon({ status }: { status: BatchStepState["status"] }) {
  if (status === "running") return <Loader2 className="size-3.5 animate-spin text-primary" />;
  if (status === "waiting_gate") return <PauseCircle className="size-3.5 text-amber-400" />;
  if (status === "done") return <Check className="size-3.5 text-emerald-400" />;
  if (status === "failed") return <X className="size-3.5 text-destructive" />;
  if (status === "skipped") return <Check className="size-3.5 text-muted-foreground/60" />;
  return <CircleDashed className="size-3.5 text-muted-foreground/60" />;
}

export function BatchPipelineDialog({
  open,
  onOpenChange,
  project,
  episode,
}: BatchPipelineDialogProps) {
  const { t } = useTranslation();
  const [preauthorized, setPreauthorized] = useState<BatchGateId[]>([]);

  const status = useBatchPipelineStatus(project, episode, open);
  const state = status.data?.data;
  const active = isPipelineActive(state);

  const estimate = useBatchPipelineEstimate(project, episode, state?.resolution ?? "720p");
  const start = useStartBatchPipeline(project, episode);
  const decide = useDecideBatchGate(project, episode);
  const cancel = useCancelBatchPipeline(project, episode);

  const gate = useMemo(() => pendingGate(state), [state]);
  const sceneTable = useMemo(() => {
    if (gate?.gate_id !== "scene_assignment") return null;
    return gate.payload as unknown as SceneAssignmentTable;
  }, [gate]);

  const cost = estimate.data?.data ?? state?.cost_estimate;

  const handleStart = async () => {
    try {
      const res = await start.mutateAsync({
        autoApprove: preauthorized,
        resolution: "720p",
        videoConcurrency: 3,
      });
      toast.success(res.message);
    } catch (error) {
      toast.error(backendErrorToastMessage(error, t));
    }
  };

  const handleGate = async (approved: boolean) => {
    if (!gate) return;
    try {
      await decide.mutateAsync({ gateId: gate.gate_id, approved });
      toast.success(
        approved
          ? t("episode.workbench.batchPipeline.gateApproved")
          : t("episode.workbench.batchPipeline.gateRejected"),
      );
    } catch (error) {
      toast.error(backendErrorToastMessage(error, t));
    }
  };

  const handleCancel = async () => {
    try {
      await cancel.mutateAsync();
      toast.success(t("episode.workbench.batchPipeline.cancelled"));
    } catch (error) {
      toast.error(backendErrorToastMessage(error, t));
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className={cn(GLASS_ALERT_DIALOG_CONTENT_CLASS, "max-w-3xl")}>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("episode.workbench.batchPipeline.title")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("episode.workbench.batchPipeline.description")}
          </AlertDialogDescription>
        </AlertDialogHeader>

        {/* 七步进度 */}
        <div className="space-y-1.5 rounded-md border border-white/10 p-3">
          {STEP_ORDER.map((stepId, index) => {
            const step = state?.steps.find((s) => s.step_id === stepId);
            const stepStatus = step?.status ?? "pending";
            return (
              <div key={stepId} className="flex items-start gap-2 text-[12px]">
                <span className="w-4 text-right text-muted-foreground/60">{index + 1}</span>
                <StepIcon status={stepStatus} />
                <div className="min-w-0 flex-1">
                  <div
                    className={cn(
                      "font-medium",
                      stepStatus === "done" && "text-emerald-300",
                      stepStatus === "failed" && "text-destructive",
                    )}
                  >
                    {t(`episode.workbench.batchPipeline.steps.${stepId}`)}
                  </div>
                  {(step?.message || step?.error) && (
                    <div
                      className={cn(
                        "truncate text-[11px]",
                        step?.error ? "text-destructive" : "text-muted-foreground",
                      )}
                    >
                      {step?.error || step?.message}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* 闸门材料 */}
        {gate && (
          <div className="space-y-2 rounded-md border border-amber-400/40 bg-amber-400/[0.06] p-3">
            <div className="flex items-center gap-2 text-[12px] font-medium text-amber-200">
              <PauseCircle className="size-3.5" />
              {t(`episode.workbench.batchPipeline.gates.${gate.gate_id}`)}
            </div>

            {sceneTable && (
              <>
                <div className="text-[11px] text-muted-foreground">
                  {t("episode.workbench.batchPipeline.sceneTableSummary", {
                    total: sceneTable.total,
                    changed: sceneTable.changed,
                    low: sceneTable.low_confidence.length,
                  })}
                </div>
                <ScrollArea className="max-h-48">
                  <table className="w-full text-[11px]">
                    <tbody>
                      {sceneTable.by_scene.map((row) => (
                        <tr key={`${row.scene_id}-${row.beat_range}`} className="border-b border-white/5">
                          <td className="py-1 pr-2 text-muted-foreground">
                            Beat {row.beat_range}
                          </td>
                          <td className="py-1 pr-2">{row.scene_id}</td>
                          <td className="py-1 pr-2 text-muted-foreground">{row.time_of_day}</td>
                          <td className="py-1 text-right text-muted-foreground">
                            {row.count}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollArea>
                {sceneTable.low_confidence.length > 0 && (
                  <div className="flex items-start gap-1.5 text-[11px] text-amber-200/90">
                    <AlertTriangle className="mt-0.5 size-3 shrink-0" />
                    {t("episode.workbench.batchPipeline.lowConfidenceBeats", {
                      beats: sceneTable.low_confidence.join(", "),
                    })}
                  </div>
                )}
              </>
            )}

            {!sceneTable && typeof gate.payload.hint === "string" && (
              <div className="text-[11px] text-muted-foreground">{gate.payload.hint}</div>
            )}

            <div className="flex gap-2 pt-1">
              <Button size="sm" onClick={() => void handleGate(true)} disabled={decide.isPending}>
                {t("episode.workbench.batchPipeline.approve")}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => void handleGate(false)}
                disabled={decide.isPending}
              >
                {t("episode.workbench.batchPipeline.reject")}
              </Button>
            </div>
          </div>
        )}

        {/* 预授权 + 成本 */}
        {!active && (
          <div className="space-y-2 rounded-md border border-white/10 p-3">
            <div className="text-[11px] font-medium text-muted-foreground">
              {t("episode.workbench.batchPipeline.preauthTitle")}
            </div>
            {PREAUTHORIZABLE.map((gateId) => (
              <label key={gateId} className="flex items-center gap-2 text-[12px]">
                <Checkbox
                  checked={preauthorized.includes(gateId)}
                  onCheckedChange={(checked) =>
                    setPreauthorized((prev) =>
                      checked ? [...prev, gateId] : prev.filter((g) => g !== gateId),
                    )
                  }
                />
                {t("episode.workbench.batchPipeline.preauthGate", {
                  gate: t(`episode.workbench.batchPipeline.gates.${gateId}`),
                })}
              </label>
            ))}
            <div className="text-[11px] text-muted-foreground">
              {t("episode.workbench.batchPipeline.mandatoryGateNote")}
            </div>
            {cost && (
              <div className="border-t border-white/5 pt-2 text-[11px] text-muted-foreground">
                {t("episode.workbench.batchPipeline.costEstimate", {
                  beats: cost.beats ?? 0,
                  seconds: cost.video_seconds ?? 0,
                  usd: cost.video_usd ?? 0,
                })}
                <div className="text-[10px] opacity-70">{cost.note}</div>
              </div>
            )}
          </div>
        )}

        {state && state.failed_beats.length > 0 && (
          <div className="flex items-start gap-1.5 rounded-md border border-destructive/40 bg-destructive/[0.06] p-2 text-[11px] text-destructive">
            <AlertTriangle className="mt-0.5 size-3 shrink-0" />
            {t("episode.workbench.batchPipeline.failedBeats", {
              beats: state.failed_beats.join(", "),
            })}
          </div>
        )}

        <AlertDialogFooter>
          <AlertDialogCancel>{t("common.close", "关闭")}</AlertDialogCancel>
          {active ? (
            <Button variant="ghost" onClick={() => void handleCancel()} disabled={cancel.isPending}>
              {t("episode.workbench.batchPipeline.cancel")}
            </Button>
          ) : (
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void handleStart();
              }}
              disabled={start.isPending}
            >
              {start.isPending && <Loader2 className="size-3 animate-spin" />}
              {t("episode.workbench.batchPipeline.start")}
            </AlertDialogAction>
          )}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
