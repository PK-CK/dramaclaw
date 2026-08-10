"""给视频提示词叠加导演级约束。

`GlobalVideoOptimizer` 产出的是「这一镜动什么」，够用但不管三件事：
画面质感在整集里会飘、相邻镜之间的服化道与光源会跳、机位会越轴。
本模块在它的输出后面追加四段约束，只在批量流水线里生效——
不改共享的系统提示词，避免影响存量的单镜生成。

追加块刻意写得短：视频模型对超长提示词会降质，这里只补
「模型自己猜不出来、而整集必须一致」的信息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: 全集统一的画面质感锚点。整集共用一句，镜与镜之间才不会一段像胶片一段像手机。
AESTHETIC_ANCHOR = (
    "ARRI Alexa 65 电影摄影质感，35mm 胶片颗粒，真实肤色，浅景深，"
    "自然光比，无磨皮无滤镜"
)

#: 时段 → 光线描述。取值与 novelvideo.time_of_day 的规范值一一对应。
_LIGHTING: dict[str, str] = {
    "清晨": "冷调晨光斜射，色温偏蓝，阴影长而软",
    "上午": "明亮日光，色温中性偏冷，阴影清晰",
    "正午": "顶光强烈，阴影短而硬，高光易过曝",
    "午后": "暖调侧光，色温偏黄，阴影拉长",
    "白天": "均匀日光，色温中性，阴影柔和",
    "黄昏": "低角度暖光，色温橙红，逆光轮廓明显",
    "夜晚": "低照度人工光源，色温混杂，高对比暗部",
}

#: 负面词。短剧出片最常见的四类废片，逐条对应实拍返工原因。
NEGATIVE_TERMS: tuple[str, ...] = (
    "多手多指",
    "面部扭曲",
    "五官漂移",
    "肢体穿模",
    "画面闪烁",
    "镜头抖动",
    "文字水印",
    "字幕",
    "慢动作",
    "画面缩放跳变",
)

#: 画面里出现这些字样时不追加口型约束——它们本就不是在说话的镜头。
_SILENT_HINTS = ("闭上眼", "昏睡", "背对", "远景", "空镜")


@dataclass
class ShotContext:
    """一镜的上下文。全部取自 beat 的结构化字段，不靠正则去猜上一镜的提示词。"""

    beat_number: int
    scene_id: str
    time_of_day: str
    index_in_scene: int                 # 本镜在本场里的序号，0 起
    identities: list[str] = field(default_factory=list)   # 如 苏晚_医院陪护
    props: list[str] = field(default_factory=list)
    prev_identities: list[str] = field(default_factory=list)
    prev_props: list[str] = field(default_factory=list)
    is_dialogue: bool = False
    visual_description: str = ""


def _strip_identity_suffix(identity: str) -> tuple[str, str]:
    """``苏晚_医院陪护`` → ``("苏晚", "医院陪护")``。没有下划线时造型为空。"""
    name, _, costume = str(identity or "").partition("_")
    return name.strip(), costume.strip()


def build_shot_context(
    beat: dict[str, Any],
    *,
    scene_id: str,
    time_of_day: str,
    index_in_scene: int,
    prev_beat: dict[str, Any] | None = None,
) -> ShotContext:
    def _ids(source: dict[str, Any] | None) -> list[str]:
        return [str(x) for x in ((source or {}).get("detected_identities") or []) if x]

    def _props(source: dict[str, Any] | None) -> list[str]:
        return [str(x) for x in ((source or {}).get("detected_props") or []) if x]

    return ShotContext(
        beat_number=int(beat.get("beat_number") or 0),
        scene_id=scene_id,
        time_of_day=time_of_day,
        index_in_scene=index_in_scene,
        identities=_ids(beat),
        props=_props(beat),
        prev_identities=_ids(prev_beat),
        prev_props=_props(prev_beat),
        is_dialogue=bool(str(beat.get("narration_segment") or "").strip())
        and str(beat.get("audio_type") or "") != "narration",
        visual_description=str(beat.get("visual_description") or ""),
    )


def build_aesthetic_block(ctx: ShotContext) -> str:
    lighting = _LIGHTING.get(ctx.time_of_day, "")
    parts = [AESTHETIC_ANCHOR]
    if lighting:
        parts.append(lighting)
    return "【质感】" + "；".join(parts)


def build_continuity_block(ctx: ShotContext) -> str:
    """列出必须与上一镜保持一致的点。

    只列**两镜都出现**的角色与道具——上一镜没有的东西谈不上「继承」，
    硬写进去反而会诱导模型把它凭空画出来。
    """
    if ctx.index_in_scene == 0:
        return f"【连贯】本场首镜，确立 {ctx.scene_id} 的空间关系与光源方向"

    carried_ids = [i for i in ctx.identities if i in ctx.prev_identities]
    carried_props = [p for p in ctx.props if p in ctx.prev_props]

    points: list[str] = [f"承接上一镜，同一场景 {ctx.scene_id}，光源方向与色温不变"]
    if carried_ids:
        looks = []
        for identity in carried_ids:
            name, costume = _strip_identity_suffix(identity)
            looks.append(f"{name}（{costume}）" if costume else name)
        points.append("服装、发型、妆容与上一镜完全一致：" + "、".join(looks))
    if carried_props:
        points.append("道具位置与状态延续：" + "、".join(carried_props))
    return "【连贯】" + "；".join(points)


def build_axis_block(ctx: ShotContext) -> str:
    """180 度轴线。首镜定轴，其后各镜守轴。"""
    if ctx.index_in_scene == 0:
        return "【机位】本场首镜确立动作轴线，机位定在轴线一侧"
    return (
        "【机位】严守 180 度轴线，机位保持在上一镜的同一侧，"
        "角色的左右朝向不得翻转；换景别可以，越轴不行"
    )


def build_negative_block() -> str:
    return "【避免】" + "、".join(NEGATIVE_TERMS)


def _needs_lipsync_note(ctx: ShotContext) -> bool:
    return ctx.is_dialogue and not any(h in ctx.visual_description for h in _SILENT_HINTS)


def enhance_video_prompt(base_prompt: str, ctx: ShotContext) -> str:
    """在运动提示词后追加四段约束，返回最终提交给视频模型的提示词。

    base_prompt 为空时返回空串——宁可让上层报「提示词未生成」，
    也不要把一堆只有约束没有动作的文字送去出片。
    """
    base = (base_prompt or "").strip()
    if not base:
        return ""

    blocks = [
        base,
        build_aesthetic_block(ctx),
        build_continuity_block(ctx),
        build_axis_block(ctx),
    ]
    if _needs_lipsync_note(ctx):
        blocks.append("【口型】说话镜头，唇动与台词节奏对齐，下颌与面颊有自然起伏")
    blocks.append(build_negative_block())
    return "\n".join(blocks)


#: 已追加过约束的提示词特征。重跑流水线时用它避免二次叠加。
_ENHANCED_MARK = re.compile(r"^【(?:质感|连贯|机位|口型|避免)】", re.M)


def is_enhanced(prompt: str) -> bool:
    return bool(_ENHANCED_MARK.search(prompt or ""))


def strip_enhancement(prompt: str) -> str:
    """剥掉追加块，取回原始运动提示词。用于重跑时先还原再叠加。"""
    lines = (prompt or "").split("\n")
    kept = [line for line in lines if not _ENHANCED_MARK.match(line.strip())]
    return "\n".join(kept).strip()
