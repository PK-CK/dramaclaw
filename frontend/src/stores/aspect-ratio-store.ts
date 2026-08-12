// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { useCallback } from "react";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { quotaSafeStateStorage } from "@/lib/localStorageQuota";

import {
  aspectSpec,
  DEFAULT_ORIENTATION,
  orientationForAspectRatio,
  type AspectSpec,
  type Orientation,
} from "@/lib/aspect-ratio";

/**
 * Per-project画幅 (aspect ratio) orientation — the single global source of
 * truth that the whole UI follows.
 *
 * Scoped per project so switching between projects never pollutes the other's
 * orientation. Region-scoped: registered in `src/lib/reset-region-state.ts`
 * so a cluster switch clears it.
 *
 * FE-first: this is the authoritative client value. Backend persistence of
 * `project_config.aspect_ratio` (cross-device sync) is added later by the
 * backend; when it lands, hydrate this store from the project config on load.
 */
interface AspectRatioState {
  // project id → orientation
  byProject: Record<string, Orientation>;
  getOrientation: (project: string) => Orientation;
  setOrientation: (project: string, orientation: Orientation) => void;
  reset: () => void;
}

export const useAspectRatioStore = create<AspectRatioState>()(
  persist(
    (set, get) => ({
      byProject: {},
      getOrientation: (project) =>
        normalize(get().byProject[project]) ?? DEFAULT_ORIENTATION,
      setOrientation: (project, orientation) =>
        set((state) => ({
          byProject: { ...state.byProject, [project]: orientation },
        })),
      reset: () => set({ byProject: {} }),
    }),
    {
      name: "supertale-aspect-ratio",
      storage: createJSONStorage(() => quotaSafeStateStorage),
      // 画幅从二元 portrait/landscape 扩成六种比例字面量后，老用户
      // localStorage 里还是旧值。不迁移的话 aspectSpec 查不到、整个 UI
      // 退回默认画幅，用户会以为自己的设置被清了。
      migrate: (persisted) => {
        const state = persisted as { byProject?: Record<string, string> } | undefined;
        const byProject: Record<string, Orientation> = {};
        for (const [project, value] of Object.entries(state?.byProject ?? {})) {
          const migrated = normalize(value);
          if (migrated) byProject[project] = migrated;
        }
        return { byProject } as AspectRatioState;
      },
      version: 1,
    },
  ),
);

/** 把任意存量值收敛成合法画幅；不认识的返回 null 由调用方兜底。 */
function normalize(value: string | undefined): Orientation | null {
  return value ? orientationForAspectRatio(value) : null;
}

/**
 * Ergonomic per-project accessor: returns the current orientation, its derived
 * {@link AspectSpec}, and a stable setter. Use this in components instead of
 * reading the raw store.
 */
export function useProjectAspectRatio(project: string): {
  orientation: Orientation;
  spec: AspectSpec;
  setOrientation: (orientation: Orientation) => void;
} {
  const orientation = useAspectRatioStore(
    (s) => orientationForAspectRatio(s.byProject[project]) ?? DEFAULT_ORIENTATION,
  );
  const setProjectOrientation = useAspectRatioStore((s) => s.setOrientation);
  const setOrientation = useCallback(
    (next: Orientation) => setProjectOrientation(project, next),
    [project, setProjectOrientation],
  );
  return { orientation, spec: aspectSpec(orientation), setOrientation };
}
