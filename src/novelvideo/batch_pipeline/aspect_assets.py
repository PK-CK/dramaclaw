"""按画幅隔离产物：一种画幅一套草图/渲染图/成片。

**为什么不直接把画幅塞进上游目录结构**：`sketches_dir` / `frames_dir` 这类
路径在上游代码里有一百多处引用（实测 sketches_dir 48 处、frames_dir 66 处、
硬编码目录名 71 处），改布局等于大面积动上游，日后合并必冲突。

所以上游路径保持原样、永远代表「当前生效」的那一套，我们在旁边维护归档：

    sketches/epNNN/beat_NN.png              ← 上游路径，当前生效
    frames/epNNN/beat_NN.png
    videos/beats/epNNN/beat_NN.mp4

    aspect_sets/epNNN/2-3/sketches/beat_NN.png     ← 归档
    aspect_sets/epNNN/2-3/frames/beat_NN.png
    aspect_sets/epNNN/2-3/videos/beat_NN.mp4
    aspect_sets/epNNN/9-16/...

切换画幅 = 归档当前 → 清空生效目录 → 从目标画幅的归档恢复（没有就留空）。

用**硬链接**而不是复制：切换是元数据操作，秒级完成、不额外占空间。同一个
卷内安全；跨卷会自动退回复制。

**这套设计顺带让跳过逻辑自动正确**：生效目录里永远只有当前画幅的产物，
`_missing_beats` / `_has_video` 照旧只看文件在不在就够了，不必再关心画幅。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 支持的画幅。与前端 lib/aspect-ratio.ts 的 Orientation、
#: 后端 api/schemas.py 的 AspectRatio 三处必须同步。
SUPPORTED_ASPECTS: tuple[str, ...] = ("1:1", "2:3", "3:4", "4:3", "9:16", "16:9")
DEFAULT_ASPECT = "2:3"

#: 产物类别 → (上游生效目录相对路径, 文件名模板)
_KINDS: dict[str, tuple[str, str]] = {
    "sketches": ("sketches/ep{ep:03d}", "beat_{n:02d}.png"),
    "frames": ("frames/ep{ep:03d}", "beat_{n:02d}.png"),
    "videos": ("videos/beats/ep{ep:03d}", "beat_{n:02d}.mp4"),
}


def normalize_aspect(value: str | None) -> str:
    """把任意输入收敛成受支持的画幅。不认识的一律退到默认值。"""
    text = str(value or "").strip()
    if text in SUPPORTED_ASPECTS:
        return text
    # 兼容前端早期的二元枚举与目录名写法
    alias = {"portrait": "2:3", "landscape": "16:9"}.get(text.lower())
    if alias:
        return alias
    if "-" in text:
        return normalize_aspect(text.replace("-", ":"))
    return DEFAULT_ASPECT


def _slug(aspect: str) -> str:
    """`9:16` → `9-16`。冒号不适合做目录名。"""
    return normalize_aspect(aspect).replace(":", "-")


def live_dir(output_dir: str | Path, episode: int, kind: str) -> Path:
    return Path(output_dir) / _KINDS[kind][0].format(ep=episode)


def archive_dir(output_dir: str | Path, episode: int, aspect: str, kind: str) -> Path:
    return (
        Path(output_dir)
        / "aspect_sets"
        / f"ep{episode:03d}"
        / _slug(aspect)
        / kind
    )


@dataclass
class AspectInventory:
    """某个画幅下各类产物各有几个。"""

    aspect: str
    sketches: int = 0
    frames: int = 0
    videos: int = 0

    @property
    def total(self) -> int:
        return self.sketches + self.frames + self.videos

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "sketches": self.sketches,
            "frames": self.frames,
            "videos": self.videos,
            "total": self.total,
        }


def _link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接；跨卷或已存在同名文件时退回复制。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _files_in(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    suffix = pattern.rsplit(".", 1)[-1]
    return sorted(
        p for p in directory.glob(f"beat_*.{suffix}") if p.is_file() and p.stat().st_size > 0
    )


def inventory(output_dir: str | Path, episode: int) -> dict[str, AspectInventory]:
    """盘点每个画幅各有多少产物。

    **当前生效的那一套不在归档里**，调用方要自己把 `live_inventory` 归到
    当前画幅名下——否则刚生成完还没切换过画幅的项目会显示成「一套都没有」。
    """
    result: dict[str, AspectInventory] = {}
    for aspect in SUPPORTED_ASPECTS:
        inv = AspectInventory(aspect=aspect)
        for kind, (_, pattern) in _KINDS.items():
            count = len(_files_in(archive_dir(output_dir, episode, aspect, kind), pattern))
            setattr(inv, kind, count)
        result[aspect] = inv
    return result


def live_inventory(output_dir: str | Path, episode: int, aspect: str) -> AspectInventory:
    """盘点当前生效目录里的产物，挂在给定画幅名下。"""
    inv = AspectInventory(aspect=normalize_aspect(aspect))
    for kind, (_, pattern) in _KINDS.items():
        inv_count = len(_files_in(live_dir(output_dir, episode, kind), pattern))
        setattr(inv, kind, inv_count)
    return inv


def archive_live(output_dir: str | Path, episode: int, aspect: str) -> int:
    """把当前生效目录的产物归档到给定画幅名下。返回归档的文件数。

    归档用硬链接，所以归档后生效目录里的文件仍然可用——真正的清空由
    `clear_live` 负责，两步分开是为了「归档失败就不清空」。
    """
    moved = 0
    for kind, (_, pattern) in _KINDS.items():
        target = archive_dir(output_dir, episode, aspect, kind)
        for src in _files_in(live_dir(output_dir, episode, kind), pattern):
            _link_or_copy(src, target / src.name)
            moved += 1
    return moved


def clear_live(output_dir: str | Path, episode: int) -> int:
    """清空生效目录。只删 beat_NN.* 这类产物，不碰目录里的其他东西。"""
    removed = 0
    for kind, (_, pattern) in _KINDS.items():
        for path in _files_in(live_dir(output_dir, episode, kind), pattern):
            path.unlink()
            removed += 1
    return removed


def restore(output_dir: str | Path, episode: int, aspect: str) -> int:
    """把给定画幅的归档恢复到生效目录。返回恢复的文件数。"""
    restored = 0
    for kind, (_, pattern) in _KINDS.items():
        source = archive_dir(output_dir, episode, aspect, kind)
        target = live_dir(output_dir, episode, kind)
        for src in _files_in(source, pattern):
            _link_or_copy(src, target / src.name)
            restored += 1
    return restored


def switch(
    output_dir: str | Path, episode: int, *, from_aspect: str, to_aspect: str
) -> dict[str, Any]:
    """从一个画幅切到另一个。

    顺序是「先归档、再清空、最后恢复」——归档若失败就不会走到清空那一步，
    当前这套产物不会凭空消失。
    """
    source = normalize_aspect(from_aspect)
    target = normalize_aspect(to_aspect)
    if source == target:
        return {
            "changed": False,
            "from": source,
            "to": target,
            "archived": 0,
            "restored": 0,
        }

    archived = archive_live(output_dir, episode, source)
    clear_live(output_dir, episode)
    restored = restore(output_dir, episode, target)
    return {
        "changed": True,
        "from": source,
        "to": target,
        "archived": archived,
        "restored": restored,
    }
