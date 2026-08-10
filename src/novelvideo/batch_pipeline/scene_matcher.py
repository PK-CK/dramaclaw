"""从剧本场次头推导每个 beat 的场景归属。

拆 beat 时场景关联会丢失（beat 的 scene_ref 为空），导致渲染阶段
「当前 Beat 没有关联场景，不能选择背景」。本模块用剧本原文的场次头
把这条关联补回来，并给出判定依据供人工复核。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from novelvideo.batch_pipeline.models import BeatAssignment, SceneHeading
from novelvideo.time_of_day import normalize_time_of_day

#: 场次头：`1-1 市立医院急诊走廊 深夜 内` / `1.1 地点 内 深夜`
_HEADING_RE = re.compile(
    r"^\s*(\d+)[-.](\d+)\s+(.+?)\s*$",
)

#: 场次头行尾常见的「时段 内/外」组合，用于从地点里剥离
_TAIL_RE = re.compile(r"\s*(?P<a>\S+)\s+(?P<b>内|外)\s*$|\s*(?P<c>内|外)\s+(?P<d>\S+)\s*$")

#: 画面里出现这些词，说明天已经亮了；此时不能沿用剧本的「凌晨/深夜」。
#: 起因：剧本写「凌晨」，normalize_time_of_day("凌晨") == "夜晚"，
#: 但该场首句是「天蒙蒙亮」，按夜晚打光会与画面矛盾。
_DAYBREAK_HINTS = ("天蒙蒙亮", "天亮", "天已亮", "晨光", "曙光", "天色泛白", "东方发白")

#: 地点关键词 → 用于给边界判定加权的次要信号
_MIN_KEYWORD_LEN = 2


def parse_scene_headings(raw_content: str) -> list[SceneHeading]:
    """解析剧本原文里的场次头。

    只认 `场号 地点 时段 内/外` 这一族写法（DramaClaw 剧本规范），
    解析不出来的行一律忽略——宁可少认，也不要把正文误判成场次头。
    """
    headings: list[SceneHeading] = []
    for line_no, line in enumerate(raw_content.split("\n"), start=1):
        matched = _HEADING_RE.match(line)
        if not matched:
            continue
        body = matched.group(3).strip()
        if not body:
            continue

        interior = ""
        raw_time = ""
        tail = _TAIL_RE.search(body)
        if tail:
            if tail.group("b"):
                raw_time, interior = tail.group("a"), tail.group("b")
            else:
                interior, raw_time = tail.group("c"), tail.group("d")
            body = body[: tail.start()].strip()

        if not body:
            continue

        headings.append(
            SceneHeading(
                order=len(headings) + 1,
                scene_id=body,
                time_of_day=normalize_time_of_day(raw_time) if raw_time else "",
                raw_time=raw_time,
                interior=interior,
                line_no=line_no,
            )
        )
    return headings


#: 场所类后缀。剧本正文常写「楼梯拐角」而非「楼梯间」，匹配时需剥掉这些字，
#: 否则关键词比正文长一个字就永远匹配不上（实测：Beat 27「缩在楼梯拐角」
#: 匹配不上「楼梯间」，导致场次边界只能靠均分兜底而整体偏移）。
_PLACE_SUFFIXES = ("间", "室", "厅", "房", "内", "里", "口", "处")


def _scene_keywords(scene_id: str) -> list[str]:
    """从场景名里抽可匹配的关键词，由长到短，如
    「市立医院楼梯间」→ 市立医院楼梯间 / 院楼梯间 / 楼梯间 / 楼梯 / 梯间。
    """
    words: list[str] = []
    text = scene_id.strip()
    if len(text) >= _MIN_KEYWORD_LEN:
        words.append(text)
    for size in (4, 3, 2):
        if len(text) > size:
            words.append(text[-size:])
    # 再补一组剥掉场所后缀的短词，覆盖「楼梯间」→「楼梯」这类写法差异
    for word in list(words):
        stripped = word
        while len(stripped) > _MIN_KEYWORD_LEN and stripped[-1] in _PLACE_SUFFIXES:
            stripped = stripped[:-1]
        if stripped != word and len(stripped) >= _MIN_KEYWORD_LEN:
            words.append(stripped)
    # 去重并保持「长词优先」，长词命中更可信
    seen: set[str] = set()
    ordered: list[str] = []
    for word in sorted(words, key=len, reverse=True):
        if word not in seen:
            seen.add(word)
            ordered.append(word)
    return ordered


def _resolve_time_of_day(heading: SceneHeading, description: str) -> tuple[str, str]:
    """确定该 beat 的时段，并返回判定说明。

    剧本时段优先，但画面明说天亮时以画面为准（见 _DAYBREAK_HINTS 注释）。
    """
    base = heading.time_of_day or ""
    for hint in _DAYBREAK_HINTS:
        if hint in description:
            if base != "清晨":
                return "清晨", f"画面出现「{hint}」，覆盖剧本时段「{heading.raw_time or '空'}」"
            break
    if heading.raw_time and base and heading.raw_time != base:
        return base, f"剧本时段「{heading.raw_time}」归一为「{base}」"
    return base, (f"取自剧本场次头「{heading.raw_time}」" if heading.raw_time else "剧本未标时段")


def _norm(text: str) -> str:
    """归一化用于比对的文本：去标点空白，中英文标点等价。"""
    return re.sub(r"[\s，。、！？；：“”‘’「」『』()（）—…·,.!?;:\"'\-]", "", text or "")


#: 剧本里不属于正文的行：集标题、人物表。场次头另由 _HEADING_RE 识别。
_TITLE_RE = re.compile(r"^\s*第\s*\d+\s*[集话話幕场場]\s*$")
_ROSTER_PREFIXES = ("人物", "角色", "出场人物", "出場人物")
#: 动作行的起始标记（剧本规范用 △）
_ACTION_MARKS = "△▲◇◆*＊"
#: `角色:台词` / `角色(括注):台词`
_SPEAKER_SPLIT_RE = re.compile(r"^([^:：]{1,24})[:：](.+)$", re.S)
#: 判定为「同一句」的相似度下限。低于此值宁可不锚，让插值兜底。
_MATCH_FLOOR = 0.34


@dataclass(frozen=True)
class _ContentLine:
    """剧本正文里的一行（已剔除场次头/人物表/空行）。"""

    line_no: int        # 原文行号，1 起
    speaker: str        # 台词行的说话人原文，动作行为空
    payload: str        # 已归一化的正文，用于比对
    is_action: bool


def _content_lines(script_lines: list[str]) -> list[_ContentLine]:
    """抽出剧本正文行。beat 与正文行本就是一一生成的，这个序列即对齐的另一端。"""
    out: list[_ContentLine] = []
    for line_no, raw in enumerate(script_lines, start=1):
        line = raw.strip()
        if not line or _TITLE_RE.match(line) or _HEADING_RE.match(line):
            continue
        if any(line.startswith(prefix) for prefix in _ROSTER_PREFIXES):
            continue

        is_action = line[0] in _ACTION_MARKS
        body = line.lstrip(_ACTION_MARKS).strip() if is_action else line
        speaker = ""
        if not is_action:
            matched = _SPEAKER_SPLIT_RE.match(body)
            if matched:
                speaker, body = matched.group(1).strip(), matched.group(2).strip()
            else:
                is_action = True  # 无「角色:」前缀的裸行按动作行处理
        if not body:
            continue
        out.append(_ContentLine(line_no, speaker, _norm(body), is_action))
    return out


def _beat_texts(beat: dict[str, Any]) -> tuple[str, str, str]:
    """取 beat 的三段可比对文本：台词 / 画面 / 说话人。

    画面描述里的角色名被替换成了 ``{{角色_身份}}`` 占位符，比对前要剥掉，
    否则占位符本身的字数会把相似度稀释掉。
    """
    narration = _norm(
        str(beat.get("narration_segment") or "") or str(beat.get("dialogue") or "")
    )
    visual = _norm(re.sub(r"\{\{[^}]*\}\}", "", str(beat.get("visual_description") or "")))
    return narration, visual, str(beat.get("speaker") or "").strip()


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _pair_score(texts: tuple[str, str, str], line: _ContentLine) -> float:
    """一个 beat 与一行正文的匹配得分。

    台词行主要看台词（逐字取自原文，通常 1.0），动作行主要看画面描述
    （经过改写，只能求相似）。交叉方向按半权计入，避免把「开口说话。」
    这类通用画面词误配到动作行上。
    """
    narration, visual, speaker = texts
    if line.is_action:
        return max(_ratio(visual, line.payload), _ratio(narration, line.payload) * 0.6)

    score = max(_ratio(narration, line.payload), _ratio(visual, line.payload) * 0.5)
    if speaker and line.speaker and (speaker in line.speaker or line.speaker in speaker):
        score = min(1.0, score + 0.1)
    return score


def _anchor_beats_to_lines(
    ordered: list[dict[str, Any]], script_lines: list[str]
) -> dict[int, int]:
    """把 beat 单调对齐到剧本正文行，返回 ``{beat 下标: 行号}``。

    beat 是照剧本顺序逐行生成的，所以两个序列天然单调；用带自由空位的
    序列对齐（DP）求全局最优匹配，比逐条贪心找子串稳得多——贪心一旦
    在某条上错锚，后面的游标就整体跑偏。
    """
    lines = _content_lines(script_lines)
    if not lines or not ordered:
        return {}

    texts = [_beat_texts(beat) for beat in ordered]
    n, m = len(texts), len(lines)

    #: score[i][j] < 0 表示该组合不允许配对
    score = [
        [(s if (s := _pair_score(texts[i], lines[j])) >= _MATCH_FLOOR else -1.0)
         for j in range(m)]
        for i in range(n)
    ]

    # dp[i][j]：前 i 个 beat 与前 j 行的最优单调匹配总分（空位不罚分）
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev, srow = dp[i], dp[i - 1], score[i - 1]
        for j in range(1, m + 1):
            best = row[j - 1] if row[j - 1] > prev[j] else prev[j]
            cell = srow[j - 1]
            if cell >= 0.0 and prev[j - 1] + cell > best:
                best = prev[j - 1] + cell
            row[j] = best

    anchors: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        cell = score[i - 1][j - 1]
        if cell >= 0.0 and abs(dp[i][j] - (dp[i - 1][j - 1] + cell)) < 1e-9:
            anchors[i - 1] = lines[j - 1].line_no
            i -= 1
            j -= 1
        elif abs(dp[i][j] - dp[i - 1][j]) < 1e-9:
            i -= 1
        else:
            j -= 1
    return anchors


def _interpolate_lines(anchors: dict[int, int], total: int) -> list[float]:
    """给每个 beat 一个（可能是插值出来的）行号，让未锚定的 beat 也能排序。"""
    idxs = sorted(anchors)
    positions = [0.0] * total
    for idx in idxs:
        positions[idx] = float(anchors[idx])

    first, last = idxs[0], idxs[-1]
    for idx in range(first):
        positions[idx] = anchors[first] - (first - idx)
    for left, right in zip(idxs, idxs[1:]):
        if right - left <= 1:
            continue
        span = anchors[right] - anchors[left]
        for idx in range(left + 1, right):
            positions[idx] = anchors[left] + span * (idx - left) / (right - left)
    for idx in range(last + 1, total):
        positions[idx] = anchors[last] + (idx - last)
    return positions


def _bounds_from_anchors(
    anchors: dict[int, int], headings: list[SceneHeading], total: int
) -> list[int] | None:
    """由锚点行号推出各场次在 beat 序列上的起始下标。

    覆盖率不足一半时返回 None，交由调用方走关键词兜底。
    """
    if len(anchors) < max(3, (total + 1) // 2):
        return None

    positions = _interpolate_lines(anchors, total)
    bounds = [0]
    for heading in headings[1:]:
        # 本场第一个 beat = 首个行号落在场次头之后的 beat
        start = next(
            (idx for idx in range(bounds[-1] + 1, total) if positions[idx] > heading.line_no),
            None,
        )
        if start is None:
            return None
        bounds.append(start)
    bounds.append(total)

    # 单调且不越界才算有效
    if any(bounds[i] >= bounds[i + 1] for i in range(len(bounds) - 1)):
        return None
    return bounds


def assign_beats(
    headings: list[SceneHeading],
    beats: list[dict[str, Any]],
    allowed_scene_ids: set[str] | None = None,
    script_lines: list[str] | None = None,
) -> list[BeatAssignment]:
    """把 beat 按顺序切分到各场次，并用画面文本校正边界。

    做法：
      1. 先按场次数量把 beat 均分（顺序切分——beat 本就是按剧本顺序生成的）；
      2. 再用画面描述里的地点关键词校正边界：某个 beat 明确提到下一场的
         地点时，把边界前移到它。
      3. 只输出 allowed_scene_ids 里存在的场景，避免写入野指针。
    """
    if not headings or not beats:
        return []

    ordered = sorted(beats, key=lambda b: int(b.get("beat_number") or 0))
    total = len(ordered)
    count = len(headings)

    # ① 首选：把 beat 锚回剧本原文行号。beat 是按剧本顺序生成的，
    #    行号是最硬的 ground truth，比关键词匹配可靠得多。
    anchors = _anchor_beats_to_lines(ordered, script_lines or [])
    bounds = _bounds_from_anchors(anchors, headings, total)

    if bounds is None:
        # ② 兜底：无原文可锚时，顺序均分 + 地点关键词校正边界
        bounds = [round(i * total / count) for i in range(count)]
        bounds.append(total)
        window = max(1, total // (count * 5))
        for i in range(1, count):
            keywords = _scene_keywords(headings[i].scene_id)
            for idx in range(max(bounds[i - 1] + 1, bounds[i] - window),
                             min(bounds[i + 1], bounds[i] + window + 1)):
                text = (
                    f"{ordered[idx].get('visual_description') or ''}"
                    f"{ordered[idx].get('narration_segment') or ''}"
                )
                if any(kw and kw in text for kw in keywords):
                    bounds[i] = idx
                    break

    assignments: list[BeatAssignment] = []
    for i, heading in enumerate(headings):
        if allowed_scene_ids is not None and heading.scene_id not in allowed_scene_ids:
            # 剧本里有、但场景库还没有的场次：跳过，交由第 1 步先建库
            continue

        # 时段按「场次」整体判定：只要本场任一 beat 表明天已亮，
        # 整场都改判，避免同一场戏前半夜后半晨的割裂。
        scene_text = "".join(
            f"{ordered[k].get('visual_description') or ''}"
            f"{ordered[k].get('narration_segment') or ''}"
            for k in range(bounds[i], bounds[i + 1])
        )
        scene_time, scene_time_reason = _resolve_time_of_day(heading, scene_text)

        for idx in range(bounds[i], bounds[i + 1]):
            beat = ordered[idx]
            desc = str(beat.get("visual_description") or "")
            narration = str(beat.get("narration_segment") or "")
            text = f"{desc}{narration}"

            time_value, time_reason = scene_time, scene_time_reason

            # 置信度三档：锚回原文行 > 画面点名场景 > 纯顺序推定
            hit = any(kw and kw in text for kw in _scene_keywords(heading.scene_id))
            line_no = anchors.get(idx)
            if line_no is not None:
                confidence = "high"
                evidence = f"锚定剧本第 {line_no} 行"
            elif hit:
                confidence = "high"
                evidence = f"画面点名「{heading.scene_id}」"
            else:
                confidence = "medium"
                evidence = f"按剧本第 {heading.order} 场顺序推定"

            current = beat.get("scene_ref") or {}
            changed = (
                str(current.get("scene_id") or "") != heading.scene_id
                or str(beat.get("time_of_day") or "") != time_value
            )

            assignments.append(
                BeatAssignment(
                    beat_number=int(beat.get("beat_number") or 0),
                    scene_id=heading.scene_id,
                    time_of_day=time_value,
                    confidence=confidence,
                    evidence=f"第{heading.order}场·{evidence}；{time_reason}",
                    changed=changed,
                )
            )
    return assignments


def build_assignment_table(assignments: list[BeatAssignment]) -> dict[str, Any]:
    """汇总成给人看的分配表（闸门 1 的确认材料）。"""
    # 同一个场景可能在一集里出现多次（如「病房」在开头和结尾各一场），
    # 必须按连续段落分组，不能按 scene_id 聚合，否则两段会被合并成
    # 一个虚假的大区间（实测：15–26 与 40–54 被并成 15–54）。
    segments: list[dict[str, Any]] = []
    for item in assignments:
        if segments and segments[-1]["scene_id"] == item.scene_id and \
                item.beat_number == segments[-1]["_last"] + 1:
            segments[-1]["count"] += 1
            segments[-1]["_last"] = item.beat_number
        else:
            segments.append({
                "scene_id": item.scene_id,
                "time_of_day": item.time_of_day,
                "count": 1,
                "_first": item.beat_number,
                "_last": item.beat_number,
            })

    return {
        "total": len(assignments),
        "changed": sum(1 for a in assignments if a.changed),
        "low_confidence": [a.beat_number for a in assignments if a.confidence != "high"],
        "by_scene": [
            {
                "scene_id": seg["scene_id"],
                "time_of_day": seg["time_of_day"],
                "count": seg["count"],
                "beat_range": f"{seg['_first']}–{seg['_last']}",
            }
            for seg in segments
        ],
        "rows": [
            {
                "beat_number": a.beat_number,
                "scene_id": a.scene_id,
                "time_of_day": a.time_of_day,
                "confidence": a.confidence,
                "evidence": a.evidence,
                "changed": a.changed,
            }
            for a in assignments
        ],
    }
