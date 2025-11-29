# app/core/transcription_utils.py
import uuid
from typing import List, Dict, Any


def normalize_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 Aliyun 返回的 segments（start/end 为毫秒）转换为秒制浮点数。
    期望输入格式：
    [
      {"start": 1234, "end": 4567, "text": "xxx", ...},
      ...
    ]
    """
    normalized = []
    for s in raw_segments or []:
        start_ms = float(s.get("start", 0))
        end_ms = float(s.get("end", 0))
        text = s.get("text", "") or ""
        start = start_ms / 1000.0
        end = end_ms / 1000.0
        if end < start:
            # 兜底：至少保证不为负
            end = start
        normalized.append({
            "start": start,
            "end": end,
            "text": text.strip()
        })
    return normalized


def build_sentences(
    segments: List[Dict[str, Any]],
    target_sentence_duration: float = 30.0,
    max_sentence_duration: float = 45.0
) -> List[Dict[str, Any]]:
    """
    把原始 segments 拼成 sentence：
    - 尽量让每句时长 ≈ target_sentence_duration（默认 30s）
    - 绝不超过 max_sentence_duration（默认 45s）
    返回：
    [
      {
        "sentenceId": "uuid",
        "start": float,
        "end": float,
        "text": "xxx"
      },
      ...
    ]
    """
    sentences: List[Dict[str, Any]] = []
    if not segments:
        return sentences

    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    current_text_parts: List[str] = []

    def flush_sentence():
        nonlocal current_start, current_end, current_text_parts
        if not current_text_parts:
            return
        sentence_text = " ".join(t for t in current_text_parts if t).strip()
        if not sentence_text:
            current_text_parts = []
            return
        sentences.append({
            "sentenceId": str(uuid.uuid4()),
            "start": current_start,
            "end": current_end,
            "text": sentence_text,
        })
        current_text_parts = []

    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        t = (seg.get("text") or "").strip()
        if not t:
            continue

        if not current_text_parts:
            # 新句子的起点
            current_start = s
            current_end = e
            current_text_parts.append(t)
            continue

        # 如果加入本段后的时长
        new_end = e
        duration = new_end - current_start

        if duration <= target_sentence_duration:
            # 正常追加
            current_text_parts.append(t)
            current_end = new_end
        elif duration <= max_sentence_duration:
            # 已经偏长，但还没超过上限 —— 也允许合并
            current_text_parts.append(t)
            current_end = new_end
            flush_sentence()
        else:
            # 超过 max_sentence_duration，先把当前句子 flush，再单独起一条
            flush_sentence()
            current_start = s
            current_end = e
            current_text_parts = [t]

    # 最后一条
    flush_sentence()
    return sentences


def build_groups(
    sentences: List[Dict[str, Any]],
    target_group_duration: float = 90.0,
    max_group_duration: float = 120.0
) -> List[Dict[str, Any]]:
    """
    把 sentences 再聚合成 group：
    - group 时长 ≈ 90s
    - 不超过 max_group_duration（默认 120s）
    返回：
    [
      {
        "groupId": "uuid",
        "start": float,
        "end": float,
        "text": "xxx",
        "sentenceIds": ["...", ...]
      },
      ...
    ]
    """
    groups: List[Dict[str, Any]] = []
    if not sentences:
        return groups

    current_group_id = str(uuid.uuid4())
    current_start = sentences[0]["start"]
    current_end = sentences[0]["end"]
    current_text_parts: List[str] = []
    current_sentence_ids: List[str] = []

    def flush_group():
        nonlocal current_group_id, current_start, current_end, current_text_parts, current_sentence_ids
        if not current_sentence_ids:
            return
        group_text = " ".join(t for t in current_text_parts if t).strip()
        groups.append({
            "groupId": current_group_id,
            "start": current_start,
            "end": current_end,
            "text": group_text,
            "sentenceIds": list(current_sentence_ids),
        })
        # 开一个新的 group，后续需要重新赋值
        current_group_id = str(uuid.uuid4())
        current_text_parts = []
        current_sentence_ids = []

    for sent in sentences:
        sid = sent["sentenceId"]
        s = float(sent["start"])
        e = float(sent["end"])
        t = (sent.get("text") or "").strip()
        if not t:
            continue

        if not current_sentence_ids:
            # 第一个 sentence
            current_start = s
            current_end = e
            current_sentence_ids.append(sid)
            current_text_parts.append(t)
            continue

        new_end = e
        duration = new_end - current_start

        if duration <= target_group_duration:
            current_end = new_end
            current_sentence_ids.append(sid)
            current_text_parts.append(t)
        elif duration <= max_group_duration:
            current_end = new_end
            current_sentence_ids.append(sid)
            current_text_parts.append(t)
            flush_group()
        else:
            # 太长了，先 flush，再单独起一个新的 group
            flush_group()
            current_start = s
            current_end = e
            current_sentence_ids = [sid]
            current_text_parts = [t]

    flush_group()
    return groups


def format_srt_time(seconds: float) -> str:
    """
    秒 → SRT 时间串：HH:MM:SS,mmm
    """
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt_from_segments(segments: List[Dict[str, Any]]) -> str:
    """
    根据标准化后的 segments（秒制）生成 SRT。
    规则：
    - 对于文本太长的段落（> 20 字符），按中英文逗号切分；如果还长，就尽量平均拆分；
    - 拆分后用线性插值在原 start/end 上分配新的时间区间。
    """
    if not segments:
        return ""

    # 1. 按 start 排序
    segs = sorted(segments, key=lambda x: float(x["start"]))

    processed = []

    for seg in segs:
        start = float(seg["start"])
        end = float(seg["end"])
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if len(text) <= 20:
            processed.append({"start": start, "end": end, "text": text})
            continue

        # 2. 按逗号拆分
        total_len = len(text)
        fragments = []
        cur = []
        last_idx = 0
        for idx, ch in enumerate(text):
            cur.append(ch)
            if ch in [",", "，", "。", "！", "？", ";", "；"]:
                frag_text = "".join(cur).strip()
                if frag_text:
                    fragments.append((last_idx, idx + 1, frag_text))
                cur = []
                last_idx = idx + 1
        # 末尾剩余
        if cur:
            frag_text = "".join(cur).strip()
            if frag_text:
                fragments.append((last_idx, total_len, frag_text))

        if not fragments:
            # 兜底：按长度强拆 20 字/段
            chunk_size = 20
            for i in range(0, total_len, chunk_size):
                frag = text[i:i + chunk_size].strip()
                if frag:
                    fragments.append((i, min(i + chunk_size, total_len), frag))

        # 3. 线性插值时间
        for (st_idx, ed_idx, frag_text) in fragments:
            ratio_start = st_idx / total_len
            ratio_end = ed_idx / total_len
            frag_start = start + (end - start) * ratio_start
            frag_end = start + (end - start) * ratio_end
            processed.append({
                "start": frag_start,
                "end": frag_end,
                "text": frag_text,
            })

    processed.sort(key=lambda x: float(x["start"]))

    # 4. 拼 SRT 字符串
    lines = []
    for i, seg in enumerate(processed, start=1):
        st = format_srt_time(float(seg["start"]))
        ed = format_srt_time(float(seg["end"]))
        txt = seg["text"]
        lines.append(str(i))
        lines.append(f"{st} --> {ed}")
        lines.append(txt)
        lines.append("")  # 空行

    return "\n".join(lines)
