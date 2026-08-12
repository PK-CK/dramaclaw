// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { jsonWithBackendError } from "@/lib/api-errors";
import { p } from "@/lib/api-path";
import { queryKeys } from "@/lib/query-keys";
import type { Orientation } from "@/lib/aspect-ratio";
import type { OkResponse } from "@/types/api";

export interface AspectSetInventory {
  aspect: Orientation;
  sketches: number;
  frames: number;
  videos: number;
  total: number;
}

export interface AspectAssetsData {
  current: Orientation;
  supported: Orientation[];
  sets: Record<Orientation, AspectSetInventory>;
}

export interface AspectSwitchResult {
  changed: boolean;
  from: Orientation;
  to: Orientation;
  archived: number;
  restored: number;
}

function base(project: string, episode: number) {
  return p`api/v1/projects/${project}/episodes/${episode}/aspect-assets`;
}

/** 盘点每个画幅各有多少产物。零开销，只数文件。 */
export function useAspectAssets(
  project: string,
  episode: number,
  current: Orientation,
  enabled = true,
) {
  return useQuery({
    queryKey: [...queryKeys.aspectAssets(project, episode), current] as const,
    queryFn: ({ signal }) =>
      jsonWithBackendError<OkResponse<AspectAssetsData>>(
        api.get(`${base(project, episode)}?current=${encodeURIComponent(current)}`, {
          signal,
          throwHttpErrors: false,
        }),
      ),
    enabled: enabled && !!project && episode > 0,
  });
}

/**
 * 切换画幅时把产物成套换过去。
 *
 * 不调这个的话，切了画幅但生效目录里还是旧比例的图，而跳过逻辑只看文件
 * 在不在——点批量流水线会整步跳过，画幅改了等于没改。
 */
export function useSwitchAspectAssets(project: string, episode: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ from, to }: { from: Orientation; to: Orientation }) =>
      jsonWithBackendError<OkResponse<AspectSwitchResult> & { message: string }>(
        api.post(`${base(project, episode)}/switch`, {
          json: { from_aspect: from, to_aspect: to },
          throwHttpErrors: false,
        }),
      ),
    onSuccess: () => {
      // 产物换了一套，beat 卡片上的图必须跟着重取
      void queryClient.invalidateQueries({
        queryKey: queryKeys.beats(project, episode),
      });
      void queryClient.invalidateQueries({
        queryKey: queryKeys.aspectAssets(project, episode),
      });
    },
  });
}
