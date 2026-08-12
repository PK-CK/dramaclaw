"""给视频提示词叠加导演级约束。

`GlobalVideoOptimizer` 产出的是「这一镜动什么」，够用但不管三件事：
画面质感在整集里会飘、相邻镜之间的服化道与光源会跳、机位会越轴。
本模块在它的输出后面追加两行约束，只在批量流水线里生效——
不改共享的系统提示词，避免影响存量的单镜生成。

追加块刻意写得短（实测约 100 字，上限 120）：视频模型对超长提示词会降质。
初版拆成质感/连贯/机位/口型四段约 200 字，比运镜指令本身还长，把真正的
镜头调度稀释掉了，验片时表现为分镜跑偏。现在只补「模型自己猜不出来、
而整集必须一致」的信息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: 全集统一的画面质感锚点。整集共用一句，镜与镜之间才不会一段像胶片一段像手机。
AESTHETIC_ANCHOR = "ARRI Alexa 65 电影质感，胶片颗粒，真实肤色，浅景深，无磨皮"

#: 时段 → 光线描述。取值与 novelvideo.time_of_day 的规范值一一对应。
_LIGHTING: dict[str, str] = {
    "清晨": "冷调斜射晨光，长软阴影",
    "上午": "明亮偏冷日光",
    "正午": "顶光强烈，硬阴影",
    "午后": "暖调侧光，长阴影",
    "白天": "均匀日光，柔和阴影",
    "黄昏": "低角度暖光，逆光轮廓",
    "夜晚": "低照度混合光，高对比暗部",
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
    # detected_identities / detected_props 在「无角色/无道具」时存的是哨兵
    # __NO_CHARACTER__ / __NO_PROP__。直接取值会把哨兵原样拼进提示词
    # （实测 50/54 个 beat 出现「道具位置与状态延续：__NO_PROP__」），
    # 视频模型会当成一串要理解的文本。用上游现成的过滤函数，别自己写。
    from novelvideo.models import real_detected_identities, real_detected_props

    def _ids(source: dict[str, Any] | None) -> list[str]:
        return real_detected_identities((source or {}).get("detected_identities"))

    def _props(source: dict[str, Any] | None) -> list[str]:
        return real_detected_props((source or {}).get("detected_props"))

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


def build_negative_block() -> str:
    """负面词块。

    ⚠️ **当前不参与提示词组装**，故意留着不删。

    grok-imagine-video 的 ``POST /videos/generations`` 契约里没有
    negative_prompt 字段（见 generators/video_generator.py 的
    GrokVideoGenerator，只发 prompt / image / duration），这串词只能拼进正文，
    等于当成正向描述发出去——写「多手多指、面部扭曲」反而诱导模型画出来。

    日后接支持负面词的后端（如 Seedance）时，把它塞进那个字段即可直接复用。
    """
    return "、".join(NEGATIVE_TERMS)


def _needs_lipsync_note(ctx: ShotContext) -> bool:
    return ctx.is_dialogue and not any(h in ctx.visual_description for h in _SILENT_HINTS)


def build_constraint_block(ctx: ShotContext) -> str:
    """压成两行的约束。

    早先拆成质感/连贯/机位/口型四段，约 200 字，比运镜指令本身还长，
    把真正的镜头调度稀释掉了（视频模型对超长提示词会降质）。这里只保留
    模型自己猜不出、而整集必须一致的那几件事，实测约 100 字。
    """
    look = [AESTHETIC_ANCHOR]
    lighting = _LIGHTING.get(ctx.time_of_day, "")
    if lighting:
        look.append(lighting)
    if _needs_lipsync_note(ctx):
        look.append("唇动对齐台词")
    lines = ["【画面】" + "，".join(look)]

    if ctx.index_in_scene == 0:
        lines.append(f"【承接】本场首镜，确立 {ctx.scene_id} 的空间与光源，机位定在轴线一侧")
        return "\n".join(lines)

    carried = [
        _strip_identity_suffix(i)[0]
        for i in ctx.identities
        if i in ctx.prev_identities
    ]
    points = [f"承上镜：{ctx.scene_id}，光源不变"]
    if carried:
        points.append("、".join(carried) + " 服化一致")
    points.append("机位同侧，不越轴")
    lines.append("【承接】" + "；".join(points))
    return "\n".join(lines)


def enhance_video_prompt(base_prompt: str, ctx: ShotContext) -> str:
    """在运动提示词后追加约束，返回最终提交给视频模型的提示词。

    base_prompt 为空时返回空串——宁可让上层报「提示词未生成」，
    也不要把一堆只有约束没有动作的文字送去出片。
    """
    base = (base_prompt or "").strip()
    if not base:
        return ""
    return base + "\n" + build_constraint_block(ctx)


#: 已追加过约束的提示词特征。重跑流水线时用它避免二次叠加。
#: 质感/连贯/机位/口型/避免是旧版的五个标记，必须继续认——库里存量提示词
#: 还带着它们，剥不干净就会在重跑时叠加两代约束。
_ENHANCED_MARK = re.compile(r"^【(?:画面|承接|质感|连贯|机位|口型|避免)】", re.M)


def is_enhanced(prompt: str) -> bool:
    return bool(_ENHANCED_MARK.search(prompt or ""))


def strip_enhancement(prompt: str) -> str:
    """剥掉追加块，取回原始运动提示词。用于重跑时先还原再叠加。"""
    lines = (prompt or "").split("\n")
    kept = [line for line in lines if not _ENHANCED_MARK.match(line.strip())]
    return "\n".join(kept).strip()
