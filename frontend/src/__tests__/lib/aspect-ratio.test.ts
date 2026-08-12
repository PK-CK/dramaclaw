// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
import { describe, expect, it } from "vitest";

import {
  aspectRatioForOrientation,
  ASPECT_OPTIONS,
  cropCostForAspect,
  orientationForAspectRatio,
  orientationForSpineTemplate,
  zoomCropBox,
} from "@/lib/aspect-ratio";

describe("aspect ratio helpers", () => {
  it("maps project_config aspect_ratio to UI orientation", () => {
    // 画幅现在就是比例本身，六种都原样返回
    for (const ratio of ASPECT_OPTIONS) {
      expect(orientationForAspectRatio(ratio)).toBe(ratio);
    }
    // 旧的二元枚举还留在老用户的 localStorage 里，必须能收敛回来
    expect(orientationForAspectRatio("landscape")).toBe("16:9");
    expect(orientationForAspectRatio("portrait")).toBe("2:3");
    expect(orientationForAspectRatio(undefined)).toBeNull();
    expect(orientationForAspectRatio("21:9")).toBeNull();
  });

  it("prices the crop cost of each aspect against the 9:16 video output", () => {
    // 9:16 与出片比例一致，零裁切；越偏离裁得越多
    expect(cropCostForAspect("9:16")).toBe(0);
    expect(cropCostForAspect("2:3")).toBeCloseTo(0.078, 3);
    expect(cropCostForAspect("16:9")).toBeCloseTo(0.342, 3);
    // 下拉顺序必须按裁切代价单调递增——排错了用户会以为靠前的更省
    const costs = ASPECT_OPTIONS.map(cropCostForAspect);
    expect(costs[0]).toBe(0);
    expect([...costs].sort((a, b) => a - b)).toEqual(costs);
    expect(costs[costs.length - 1]).toBeCloseTo(cropCostForAspect("16:9"), 6);
  });

  it("maps UI orientation back to persisted project_config aspect_ratio", () => {
    for (const ratio of ASPECT_OPTIONS) {
      expect(aspectRatioForOrientation(ratio)).toBe(ratio);
    }
  });

  it("uses narrated projects as the landscape default", () => {
    expect(orientationForSpineTemplate("narrated")).toBe("16:9");
    expect(orientationForSpineTemplate("drama")).toBe("2:3");
  });

  it("zooms a crop box around its center while clamping to the source image", () => {
    expect(
      zoomCropBox({ x: 100, y: 50, width: 400, height: 200 }, 1000, 600, 0.5),
    ).toEqual({ x: 200, y: 100, width: 200, height: 100 });
    expect(
      zoomCropBox({ x: 200, y: 100, width: 200, height: 100 }, 300, 180, 2),
    ).toEqual({ x: 0, y: 30, width: 300, height: 150 });
  });
});
