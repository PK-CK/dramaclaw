// SPDX-License-Identifier: Elastic-2.0
// Copyright (c) 2026 ClaymoreLab
/**
 * Single source of truth for project画幅 (aspect ratio) derivations.
 *
 * The whole UI follows ONE orientation per project. Every concrete value the
 * app needs — CSS aspect box, display label, crop ratio, generation params —
 * is derived from that single orientation here so callers never hardcode
 * "2:3" / "16:9" / "aspect-video" again.
 */

/**
 * 画幅键。取值直接就是比例字面量，与后端 schema 的 Literal 一一对应。
 *
 * 底层 `generators/nanobanana_grid.py` 的 REGEN_MODE_CONFIGS 对这六种都备了
 * 1x1~4x4 的完整网格模式，早就支持；先前只在这里和后端 schema 里被锁成了两个。
 */
export type Orientation =
  | "1:1"
  | "2:3"
  | "3:4"
  | "4:3"
  | "9:16"
  | "16:9";
export type ProjectAspectRatio = Orientation;
export type SpineTemplate = "drama" | "narrated";

/**
 * 出片比例。grok-imagine-video 固定输出 9:16 竖屏，改不了。
 * 首帧图与它不一致的部分会被裁掉，所以渲染画幅越接近这个值越好。
 */
export const VIDEO_OUTPUT_RATIO = 9 / 16;

export interface AspectSpec {
  orientation: Orientation;
  /** Human-facing ratio label, e.g. "2:3". */
  label: string;
  /** CSS aspect-ratio value, e.g. "2/3" (for arbitrary tailwind classes). */
  cssRatio: string;
  /** Tailwind aspect-box class. */
  aspectClass: string;
  /** width / height — used for crop-box math. */
  ratioValue: number;
  /** Sketch generation aspect param (backend-accepted union). */
  sketchAspect: Orientation;
  /** Render aspect_mode literal sent to the render pipeline. */
  renderAspect: Orientation;
  /** 竖屏/横屏，供仍按方向做布局判断的地方使用。 */
  isPortrait: boolean;
}

function spec(label: Orientation, w: number, h: number, aspectClass: string): AspectSpec {
  return {
    orientation: label,
    label,
    cssRatio: `${w}/${h}`,
    aspectClass,
    ratioValue: w / h,
    sketchAspect: label,
    renderAspect: label,
    isPortrait: w / h < 1,
  };
}

const SPECS: Record<Orientation, AspectSpec> = {
  "1:1": spec("1:1", 1, 1, "aspect-square"),
  "2:3": spec("2:3", 2, 3, "aspect-[2/3]"),
  "3:4": spec("3:4", 3, 4, "aspect-[3/4]"),
  "4:3": spec("4:3", 4, 3, "aspect-[4/3]"),
  "9:16": spec("9:16", 9, 16, "aspect-[9/16]"),
  "16:9": spec("16:9", 16, 9, "aspect-video"),
};

/** 下拉里的顺序：按裁切代价从小到大，与出片比例越接近的排越前。 */
export const ASPECT_OPTIONS: readonly Orientation[] = [
  "9:16",  // 零裁切
  "2:3",   // 左右各裁 7.8%
  "3:4",   // 12.5%
  "1:1",   // 21.9%
  "4:3",   // 28.9%
  "16:9",  // 34.2%
];

/**
 * 选这个画幅出图，送进视频模型时首帧**左右各**要被裁掉的比例。
 * 0 表示与出片比例一致、零裁切。UI 用它把代价标出来。
 *
 * 注意口径：返回的是单侧比例，不是被裁掉的总宽度（总宽 = 2×本值）。
 * 两种口径混用会让 16:9 看起来比 4:3 便宜，实际相反。
 */
export function cropCostForAspect(orientation: Orientation): number {
  const keep = VIDEO_OUTPUT_RATIO / SPECS[orientation].ratioValue;
  if (keep >= 1) return 0;
  return (1 - keep) / 2;
}

export function aspectSpec(orientation: Orientation): AspectSpec {
  return SPECS[orientation] ?? SPECS["2:3"];
}

/** Default orientation for a project before any explicit choice. */
export const DEFAULT_ORIENTATION: Orientation = "2:3";

export function orientationForAspectRatio(
  aspectRatio: string | null | undefined,
): Orientation | null {
  const value = String(aspectRatio ?? "").trim();
  if (value in SPECS) return value as Orientation;
  // 旧值兼容：localStorage 里存量是 portrait/landscape 二元枚举。
  if (value === "portrait") return "2:3";
  if (value === "landscape") return "16:9";
  return null;
}

export function aspectRatioForOrientation(
  orientation: Orientation,
): ProjectAspectRatio {
  return aspectSpec(orientation).renderAspect;
}

export function orientationForSpineTemplate(
  spineTemplate: SpineTemplate | null | undefined,
): Orientation {
  return spineTemplate === "narrated" ? "16:9" : DEFAULT_ORIENTATION;
}

/**
 * Convert a "W:H" ratio label (e.g. "2:3", "16:9") into a CSS
 * `aspect-ratio` value ("W / H"). Use for single-image boxes whose true aspect
 * is variable — pass `spec.sketchAspect` for sketch cells, `spec.renderAspect`
 * for render/video frames.
 */
export function ratioToCss(ratio: string): string {
  const [w, h] = ratio.split(":");
  return `${w} / ${h}`;
}

/**
 * CSS `aspect-ratio` for a composite grid thumbnail. A grid is `cols × rows`
 * cells, each cell having `cellAspect` ("W:H"), so the whole image is
 * `(cols·W) / (rows·H)`. Keeps grid previews from squishing portrait cells
 * into a 16:9 box.
 */
export function gridAspectCss(
  cols: number,
  rows: number,
  cellAspect: string,
): string {
  const [w, h] = cellAspect.split(":").map(Number);
  const safeCols = Math.max(1, cols);
  const safeRows = Math.max(1, rows);
  if (!w || !h) return "1 / 1";
  return `${safeCols * w} / ${safeRows * h}`;
}

export interface CropBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export const FULL_SOURCE_CROP_SIZE = 999_999;

export function centerCropBoxForRatio(
  width: number,
  height: number,
  ratio: number,
): CropBox {
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    !Number.isFinite(ratio) ||
    width <= 0 ||
    height <= 0 ||
    ratio <= 0
  ) {
    return {
      x: 0,
      y: 0,
      width: FULL_SOURCE_CROP_SIZE,
      height: FULL_SOURCE_CROP_SIZE,
    };
  }

  let cropWidth = width;
  let cropHeight = cropWidth / ratio;
  if (cropHeight > height) {
    cropHeight = height;
    cropWidth = cropHeight * ratio;
  }

  return {
    x: Math.max(0, Math.round((width - cropWidth) / 2)),
    y: Math.max(0, Math.round((height - cropHeight) / 2)),
    width: Math.max(1, Math.round(cropWidth)),
    height: Math.max(1, Math.round(cropHeight)),
  };
}

export function zoomCropBox(
  crop: CropBox,
  sourceWidth: number,
  sourceHeight: number,
  scale: number,
): CropBox {
  const safeWidth = Math.max(1, sourceWidth);
  const safeHeight = Math.max(1, sourceHeight);
  const currentWidth = Math.max(1, crop.width);
  const currentHeight = Math.max(1, crop.height);
  const maxScale = Math.min(safeWidth / currentWidth, safeHeight / currentHeight);
  const minScale = Math.min(
    maxScale,
    Math.max(16 / currentWidth, 16 / currentHeight),
  );
  const nextScale = Math.min(Math.max(scale, minScale), maxScale);
  const nextWidth = Math.max(1, Math.round(currentWidth * nextScale));
  const nextHeight = Math.max(1, Math.round(currentHeight * nextScale));
  const centerX = crop.x + currentWidth / 2;
  const centerY = crop.y + currentHeight / 2;

  return {
    x: Math.min(
      Math.max(0, Math.round(centerX - nextWidth / 2)),
      Math.max(0, safeWidth - nextWidth),
    ),
    y: Math.min(
      Math.max(0, Math.round(centerY - nextHeight / 2)),
      Math.max(0, safeHeight - nextHeight),
    ),
    width: nextWidth,
    height: nextHeight,
  };
}
