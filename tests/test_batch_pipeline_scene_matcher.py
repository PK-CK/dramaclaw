"""场次归属推导的回归测试。

用一段结构完整的短剧本（4 场、台词+动作混排、同一地点出现两次、
末场剧本标「凌晨」而画面写「天蒙蒙亮」）覆盖以下已修过的缺陷：

* 同名场景两段被聚合成一个虚假大区间（15–26 与 40–54 并成 15–54）
* 场次边界靠均分兜底而整体偏移一两个 beat
* 时段覆盖只命中单个 beat，导致同场戏前半夜后半晨
"""

from __future__ import annotations

import pytest

from novelvideo.batch_pipeline.scene_matcher import (
    assign_beats,
    build_assignment_table,
    parse_scene_headings,
)

SCRIPT = """第1集
1-1 市立医院急诊走廊 深夜 内
人物:苏晚、周明远
△一支圆珠笔的笔尖抵在苏晚眼前,笔杆上全是汗。
周明远:签。妈在里面等着开颅,四十万,现在。
苏晚 OS:结婚十年,他第一次用这种眼神看我。
△苏晚绕开他,快步走向缴费大厅。
苏晚:上一次你这么叫我,是让我把陪嫁的镯子当了。

1-2 市立医院12床病房 深夜 内
人物:苏晚、何桂芳、周雪
△何桂芳躺在床上,输液管一滴一滴,人昏睡着。
△床头柜上摊着一个牛皮纸文件袋,袋口露出一角合同纸。
周雪:嫂子,你翻我妈东西呢?

1-3 市立医院楼梯间 深夜 内
人物:苏晚、陈晓
△苏晚缩在楼梯拐角,手机贴着耳朵。
苏晚:陈晓,我知道现在是凌晨三点。
△黑暗里,苏晚听见楼上有脚步声停住。

1-4 市立医院12床病房 凌晨 内
人物:苏晚、何桂芳
△天蒙蒙亮,何桂芳醒了,嘴角歪着,手抖得厉害。
何桂芳(含糊):晚……晚啊……
△苏晚握住她的手。
周雪:凉了,我去热热。
"""

#: (画面, 台词, 说话人)。画面里的角色名按线上数据的写法替换成占位符。
BEAT_SOURCE: list[tuple[str, str, str]] = [
    ("一支圆珠笔的笔尖抵在{{苏晚_医院陪护}}眼前，笔杆上全是汗。", "", ""),
    ("{{周明远_落魄西装}}开口说话。", "签。妈在里面等着开颅,四十万,现在。", "周明远"),
    ("{{苏晚_医院陪护}}的独白。", "结婚十年,他第一次用这种眼神看我。", "苏晚"),
    ("{{苏晚_医院陪护}}绕开他，快步走向缴费大厅。", "", ""),
    ("{{苏晚_医院陪护}}开口说话。", "上一次你这么叫我,是让我把陪嫁的镯子当了。", "苏晚"),
    ("{{何桂芳_住院病号服}}躺在病床上昏睡，输液管中的液体一滴一滴落下。", "", ""),
    ("床头柜上摊着一个牛皮纸文件袋，袋口露出一角合同纸。", "", ""),
    ("{{周雪_精致职业装}}站在门口笑着开口说话。", "嫂子,你翻我妈东西呢?", "周雪"),
    ("{{苏晚_医院陪护}}缩在楼梯拐角，手机贴着耳朵。", "", ""),
    ("{{苏晚_医院陪护}}对着电话开口说话。", "陈晓,我知道现在是凌晨三点。", "苏晚"),
    ("黑暗里，{{苏晚_医院陪护}}听见楼上传来的脚步声停住。", "", ""),
    ("天蒙蒙亮，{{何桂芳_住院病号服}}醒来，嘴角歪斜，双手抖得厉害。", "", ""),
    ("{{何桂芳_住院病号服}}含糊地开口说话。", "晚……晚啊……", "何桂芳"),
    ("{{苏晚_医院陪护}}握住{{何桂芳_住院病号服}}的手。", "", ""),
    ("{{周雪_精致职业装}}笑着开口说话。", "凉了,我去热热。", "周雪"),
]

SCENES = {"市立医院急诊走廊", "市立医院12床病房", "市立医院楼梯间"}


@pytest.fixture
def beats() -> list[dict]:
    return [
        {
            "beat_number": i,
            "visual_description": visual,
            "narration_segment": narration,
            "speaker": speaker,
            "scene_ref": {},
            "time_of_day": "",
        }
        for i, (visual, narration, speaker) in enumerate(BEAT_SOURCE, start=1)
    ]


def test_parse_scene_headings() -> None:
    headings = parse_scene_headings(SCRIPT)
    assert [(h.scene_id, h.raw_time, h.interior) for h in headings] == [
        ("市立医院急诊走廊", "深夜", "内"),
        ("市立医院12床病房", "深夜", "内"),
        ("市立医院楼梯间", "深夜", "内"),
        ("市立医院12床病房", "凌晨", "内"),
    ]
    # 「人物:」行、集标题、正文都不能被当成场次头
    assert [h.line_no for h in headings] == [2, 10, 16, 22]


def test_assign_beats_matches_manual_segmentation(beats: list[dict]) -> None:
    headings = parse_scene_headings(SCRIPT)
    table = build_assignment_table(
        assign_beats(headings, beats, SCENES, script_lines=SCRIPT.split("\n"))
    )

    assert [
        (row["beat_range"], row["scene_id"], row["time_of_day"])
        for row in table["by_scene"]
    ] == [
        ("1–5", "市立医院急诊走廊", "夜晚"),
        ("6–8", "市立医院12床病房", "夜晚"),
        ("9–11", "市立医院楼梯间", "夜晚"),
        # 剧本写「凌晨」（归一为夜晚），但画面写「天蒙蒙亮」，整场改判清晨
        ("12–15", "市立医院12床病房", "清晨"),
    ]
    assert table["total"] == len(BEAT_SOURCE)
    assert table["changed"] == len(BEAT_SOURCE)  # 原本 scene_ref 全空
    assert table["low_confidence"] == []


def test_daybreak_override_covers_whole_scene(beats: list[dict]) -> None:
    """只有第 12 个 beat 写了「天蒙蒙亮」，同场其余 beat 也必须一起改判。"""
    headings = parse_scene_headings(SCRIPT)
    rows = assign_beats(headings, beats, SCENES, script_lines=SCRIPT.split("\n"))
    last_scene = [r for r in rows if r.beat_number >= 12]
    assert len(last_scene) == 4
    assert {r.time_of_day for r in last_scene} == {"清晨"}


def test_unknown_scene_is_skipped(beats: list[dict]) -> None:
    """场景库里没有的场次直接跳过，不能写野指针进 scene_ref。"""
    headings = parse_scene_headings(SCRIPT)
    rows = assign_beats(
        headings, beats, {"市立医院急诊走廊"}, script_lines=SCRIPT.split("\n")
    )
    assert {r.scene_id for r in rows} == {"市立医院急诊走廊"}
    assert [r.beat_number for r in rows] == [1, 2, 3, 4, 5]


def test_falls_back_to_keywords_without_script_lines(beats: list[dict]) -> None:
    """拿不到剧本原文时退回关键词兜底，仍要给出 4 段且顺序正确。"""
    headings = parse_scene_headings(SCRIPT)
    table = build_assignment_table(assign_beats(headings, beats, SCENES))
    assert [row["scene_id"] for row in table["by_scene"]] == [
        "市立医院急诊走廊",
        "市立医院12床病房",
        "市立医院楼梯间",
        "市立医院12床病房",
    ]
    assert sum(row["count"] for row in table["by_scene"]) == len(BEAT_SOURCE)


def test_falls_back_when_anchors_too_sparse(beats: list[dict]) -> None:
    """有原文但对不上（beat 文案被大改）时也要退回兜底，不能拿几个孤点硬分段。

    覆盖率门槛是 max(3, (总数+1)//2)；这里把画面与台词全部换掉，
    只留最后一个 beat 能对上，锚点数远低于门槛。
    """
    headings = parse_scene_headings(SCRIPT)
    scrambled = [
        {**beat, "visual_description": f"无关画面{i}", "narration_segment": "", "speaker": ""}
        for i, beat in enumerate(beats[:-1])
    ] + [beats[-1]]

    table = build_assignment_table(
        assign_beats(headings, scrambled, SCENES, script_lines=SCRIPT.split("\n"))
    )
    # 退回兜底后仍必须是 4 段、顺序正确、beat 不重不漏
    assert [row["scene_id"] for row in table["by_scene"]] == [
        "市立医院急诊走廊",
        "市立医院12床病房",
        "市立医院楼梯间",
        "市立医院12床病房",
    ]
    assert sum(row["count"] for row in table["by_scene"]) == len(BEAT_SOURCE)
    assert [row["beat_number"] for row in table["rows"]] == list(
        range(1, len(BEAT_SOURCE) + 1)
    )


def test_missing_beats_skips_existing_artifacts(tmp_path) -> None:
    """已有产物的 beat 必须被跳过——重出既花钱，又往图片池多叠一版历史。"""
    from novelvideo.batch_pipeline.steps import Runtime, _missing_beats
    from novelvideo.batch_pipeline.models import PipelineState

    beats = [{"beat_number": i} for i in range(1, 6)]
    sketches = tmp_path / "sketches" / "ep001"
    frames = tmp_path / "frames" / "ep001"
    sketches.mkdir(parents=True)
    frames.mkdir(parents=True)
    for number in (1, 2, 4):
        (sketches / f"beat_{number:02d}.png").write_bytes(b"x")
    # 空文件视同缺失：出图失败留下的 0 字节文件不能算数
    (sketches / "beat_03.png").write_bytes(b"")
    for number in (1, 2, 3, 4, 5):
        (frames / f"beat_{number:02d}.png").write_bytes(b"x")

    rt = Runtime(
        ctx=None,  # type: ignore[arg-type]
        episode=1,
        output_dir=str(tmp_path),
        state=PipelineState(project="p", episode=1),
        log=lambda *a, **k: None,
    )
    assert _missing_beats(rt, beats, "sketch") == [3, 5]
    assert _missing_beats(rt, beats, "frame") == []


def test_existing_video_prompt_and_video_are_skipped(tmp_path) -> None:
    """提示词与成片都要按已有产物跳过——两者重做都是白花钱。"""
    from novelvideo.batch_pipeline.models import PipelineState
    from novelvideo.batch_pipeline.steps import (
        Runtime,
        _existing_video_prompt,
        _has_video,
    )

    # 提示词：首帧模式看 video_prompt，首尾帧模式看 keyframe_prompt
    assert _existing_video_prompt({"video_prompt": "镜头推近"}) == "镜头推近"
    assert _existing_video_prompt({"video_prompt": "   "}) == ""
    assert _existing_video_prompt({}) == ""
    assert (
        _existing_video_prompt({"video_mode": "keyframe", "keyframe_prompt": "过渡"})
        == "过渡"
    )
    # 首尾帧模式下 video_prompt 有值也不算数——出片用的是 keyframe_prompt
    assert (
        _existing_video_prompt({"video_mode": "keyframe", "video_prompt": "旧文案"}) == ""
    )

    videos = tmp_path / "videos" / "beats" / "ep001"
    videos.mkdir(parents=True)
    (videos / "beat_01.mp4").write_bytes(b"x")
    (videos / "beat_02.mp4").write_bytes(b"")  # 空文件视同没出

    rt = Runtime(
        ctx=None,  # type: ignore[arg-type]
        episode=1,
        output_dir=str(tmp_path),
        state=PipelineState(project="p", episode=1),
        log=lambda *a, **k: None,
    )
    assert _has_video(rt, {"beat_number": 1}) is True           # 盘上有 mp4
    assert _has_video(rt, {"beat_number": 2}) is False          # 空文件不算
    assert _has_video(rt, {"beat_number": 3}) is False          # 什么都没有
    # 库里有 video_url 就算，即使盘上没文件（换过存储、或只留了远端地址）
    assert _has_video(rt, {"beat_number": 3, "video_url": "https://x/a.mp4"}) is True


def test_empty_inputs_return_empty() -> None:
    assert assign_beats([], [{"beat_number": 1}]) == []
    assert assign_beats(parse_scene_headings(SCRIPT), []) == []
