"""画幅资产隔离的回归测试。

核心承诺：一种画幅一套产物，来回切换不丢东西，切到没产物的画幅就是空的。
"""

from __future__ import annotations

import pytest

from novelvideo.batch_pipeline import aspect_assets as aa


def _write_live(root, episode: int, beats: range, *, tag: str = "x") -> None:
    """在生效目录造一套完整产物。"""
    for kind, (rel, pattern) in aa._KINDS.items():
        directory = root / rel.format(ep=episode)
        directory.mkdir(parents=True, exist_ok=True)
        for n in beats:
            (directory / pattern.format(n=n)).write_bytes(f"{tag}-{kind}-{n}".encode())


def _live_counts(root, episode: int) -> dict[str, int]:
    return {
        kind: len(list((root / rel.format(ep=episode)).glob("beat_*")))
        if (root / rel.format(ep=episode)).exists()
        else 0
        for kind, (rel, _) in aa._KINDS.items()
    }


def test_normalize_aspect() -> None:
    assert aa.normalize_aspect("9:16") == "9:16"
    assert aa.normalize_aspect("9-16") == "9:16"          # 目录名写法
    assert aa.normalize_aspect("portrait") == "2:3"       # 前端旧枚举
    assert aa.normalize_aspect("landscape") == "16:9"
    assert aa.normalize_aspect("21:9") == aa.DEFAULT_ASPECT   # 不支持的退默认
    assert aa.normalize_aspect(None) == aa.DEFAULT_ASPECT


def test_switch_archives_and_restores(tmp_path) -> None:
    """2:3 → 9:16 → 2:3 来回切，两套产物都不能丢。"""
    _write_live(tmp_path, 1, range(1, 4), tag="portrait")

    # 切到 9:16：旧的进归档，生效目录清空（9:16 还没产物）
    result = aa.switch(tmp_path, 1, from_aspect="2:3", to_aspect="9:16")
    assert result["changed"] is True
    assert result["archived"] == 9        # 3 beat × 3 类
    assert result["restored"] == 0
    assert _live_counts(tmp_path, 1) == {"sketches": 0, "frames": 0, "videos": 0}

    # 按新画幅出一套（模拟批量流水线跑完）
    _write_live(tmp_path, 1, range(1, 6), tag="vertical")

    # 切回 2:3：新的一套进归档，旧的恢复
    back = aa.switch(tmp_path, 1, from_aspect="9:16", to_aspect="2:3")
    assert back["archived"] == 15         # 5 beat × 3 类
    assert back["restored"] == 9
    assert _live_counts(tmp_path, 1) == {"sketches": 3, "frames": 3, "videos": 3}

    # 内容确实是当初那一套，不是新的
    sketch = tmp_path / "sketches" / "ep001" / "beat_01.png"
    assert sketch.read_bytes().startswith(b"portrait")

    # 再切回 9:16，5 个 beat 的那套原样回来
    again = aa.switch(tmp_path, 1, from_aspect="2:3", to_aspect="9:16")
    assert again["restored"] == 15
    assert _live_counts(tmp_path, 1) == {"sketches": 5, "frames": 5, "videos": 5}
    assert (tmp_path / "frames" / "ep001" / "beat_05.png").read_bytes().startswith(b"vertical")


def test_switch_to_same_aspect_is_a_noop(tmp_path) -> None:
    """切到同一个画幅不能动任何东西——否则误触会白归档一次。"""
    _write_live(tmp_path, 1, range(1, 3))
    result = aa.switch(tmp_path, 1, from_aspect="2:3", to_aspect="2:3")
    assert result == {
        "changed": False,
        "from": "2:3",
        "to": "2:3",
        "archived": 0,
        "restored": 0,
    }
    assert _live_counts(tmp_path, 1) == {"sketches": 2, "frames": 2, "videos": 2}


def test_inventory_counts_each_aspect(tmp_path) -> None:
    _write_live(tmp_path, 1, range(1, 4))
    aa.switch(tmp_path, 1, from_aspect="2:3", to_aspect="16:9")
    _write_live(tmp_path, 1, range(1, 2))

    sets = aa.inventory(tmp_path, 1)
    assert sets["2:3"].to_dict() == {
        "aspect": "2:3", "sketches": 3, "frames": 3, "videos": 3, "total": 9,
    }
    assert sets["9:16"].total == 0            # 从没出过
    assert sets["16:9"].total == 0            # 还没归档，只在生效目录里

    live = aa.live_inventory(tmp_path, 1, "16:9")
    assert (live.sketches, live.frames, live.videos) == (1, 1, 1)


def test_empty_files_are_ignored(tmp_path) -> None:
    """0 字节文件是出图失败的残留，不能被当成产物归档或计数。"""
    directory = tmp_path / "sketches" / "ep001"
    directory.mkdir(parents=True)
    (directory / "beat_01.png").write_bytes(b"ok")
    (directory / "beat_02.png").write_bytes(b"")

    assert aa.live_inventory(tmp_path, 1, "2:3").sketches == 1
    assert aa.archive_live(tmp_path, 1, "2:3") == 1


def test_archive_uses_hardlinks_when_possible(tmp_path) -> None:
    """归档走硬链接：切换要秒级完成且不翻倍占空间。"""
    _write_live(tmp_path, 1, range(1, 2))
    aa.archive_live(tmp_path, 1, "2:3")

    live = tmp_path / "sketches" / "ep001" / "beat_01.png"
    archived = aa.archive_dir(tmp_path, 1, "2:3", "sketches") / "beat_01.png"
    assert archived.exists()
    # 同一 inode 才是硬链接；跨卷会退回复制，那种情况下这条断言不适用，
    # 但 tmp_path 与归档目录同卷，必然是硬链接
    assert live.stat().st_ino == archived.stat().st_ino


@pytest.mark.parametrize("aspect,slug", [("9:16", "9-16"), ("2:3", "2-3"), ("1:1", "1-1")])
def test_archive_dir_slug(tmp_path, aspect: str, slug: str) -> None:
    """冒号不能进目录名。"""
    path = aa.archive_dir(tmp_path, 1, aspect, "frames")
    assert path.parent.name == slug
    assert ":" not in str(path)
