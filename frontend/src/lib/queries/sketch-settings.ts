// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { p } from "@/lib/api-path";
import { queryKeys } from "@/lib/query-keys";
import type { ErrorResponse, OkResponse } from "@/types/api";

export interface SketchSettingsData {
  sketch_image_selection: string;
  options: Record<string, string>;
}

export interface SketchSettingsUpdate {
  sketch_image_selection?: string;
}

/**
 * 草图生成的画幅参数。
 *
 * 与 `lib/aspect-ratio` 的 Orientation 同一套取值——画幅只该有一个真源，
 * 这里独立维护一份联合类型是历史遗留，两边一旦不同步就会出现
 * 「下拉能选、类型不认」的编译错。直接复用，不再重复声明。
 */
export type { Orientation as SketchAspectRatio } from "@/lib/aspect-ratio";

export function useSketchSettings(project: string) {
  return useQuery({
    queryKey: queryKeys.sketchSettings(project),
    queryFn: ({ signal }) =>
      api
        .get(p`api/v1/projects/${project}/sketch-settings`, { signal })
        .json<OkResponse<SketchSettingsData>>(),
    enabled: !!project,
  });
}

export function useUpdateSketchSettings(project: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: SketchSettingsUpdate) =>
      api
        .patch(p`api/v1/projects/${project}/sketch-settings`, { json: params })
        .json<OkResponse<SketchSettingsData> | ErrorResponse>(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.sketchSettings(project) });
    },
  });
}
