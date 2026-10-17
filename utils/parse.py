"""Gemini Web 响应解析与正文清洗。"""

from __future__ import annotations

import json
import re
from typing import Any

from core.exceptions import GeminiError
from utils.logger import Logger


LOGGER: Logger = Logger("parse")


def strip_artifacts(text: str) -> str:
    """去掉代码执行和卡片伪影, 保留正文两端的空白。"""
    cleaned: str = re.sub(
        r"```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?",
        "",
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"http://googleusercontent\.com/card_content/\d+\n?", "", cleaned)


def clean_text(text: str) -> str:
    """非流式清洗: 去伪影并去掉完整正文的首尾空白。"""
    return strip_artifacts(text).strip()


def pop_complete_lines(buffer: str) -> tuple[list[str], str]:
    """取出已经以换行结束的行, 留下未完成片段。"""
    if "\n" not in buffer:
        return [], buffer
    parts: list[str] = buffer.split("\n")
    return parts[:-1], parts[-1]


def detect_bard_error(raw: str) -> str | None:
    """检测上游 BardErrorInfo 错误码。"""
    match: re.Match[str] | None = re.search(r"BardErrorInfo[^\[]*\[(\d+)\]", raw)
    return match.group(1) if match else None


def format_bard_error(code: str) -> str:
    """格式化 BardErrorInfo, 对齐 Node formatBardError。"""
    hints: dict[str, str] = {
        "1003": " (image ref structure or binding failed)",
        "1060": " (region/IP blocked or bl outdated)",
    }
    return f"Gemini upstream rejected request: BardErrorInfo [{code}]{hints.get(code, '')}"


def raise_if_bard_error(raw: str) -> None:
    """发现 BardErrorInfo 时中断解析, 避免把错误页当正文。"""
    if "BardErrorInfo" not in raw:
        return
    code: str | None = detect_bard_error(raw)
    if code:
        raise GeminiError(format_bard_error(code))


class StreamTextAssembler:
    """把 Gemini 累积正文切成增量, 保留分片边界上的空白。"""

    def __init__(self) -> None:
        """记录已见过的原文和已经下发的清洗文本。"""
        self._previous_raw: str = ""
        self._emitted: str = ""

    def push(self, text: str) -> str | None:
        """接收一条更长的累积原文, 返回尚未下发的增量。"""
        if len(text) <= len(self._previous_raw):
            return None
        cleaned: str = strip_artifacts(text)
        self._previous_raw = text
        if not cleaned.startswith(self._emitted):
            # 伪影正则跨过已发送前缀时, 全文不再是简单追加, 不能把旧正文再发一遍.
            return None
        delta: str = cleaned[len(self._emitted):]
        self._emitted = cleaned
        if not delta:
            return None
        return delta


def extract_texts_from_line(line: str) -> list[str]:
    """从单行 wrb.fr 响应提取文本片段, 对齐 Node extractTextsFromLine。"""
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        outer: Any = json.loads(line)
        row: Any = outer[0]
        inner_text: Any = row[2]
        if not isinstance(inner_text, str) or len(inner_text) < 50:
            return []
        inner: Any = json.loads(inner_text)
        if not isinstance(inner, list) or len(inner) <= 4 or not inner[4]:
            return []
        texts: list[str] = []
        for part in inner[4]:
            if not isinstance(part, list) or len(part) <= 1 or not isinstance(part[1], list):
                continue
            for text in part[1]:
                if isinstance(text, str) and text:
                    texts.append(text)
        return texts
    except Exception:
        # Node 版在任意 JSON/结构异常时直接返回空数组。
        return []


def extract_response_text(raw: str) -> str:
    """解析完整响应并返回最长候选文本, 对齐 Node extractResponseText。"""
    code: str | None = detect_bard_error(raw)
    if code:
        raise GeminiError(format_bard_error(code))

    last_text: str = ""
    for line in raw.split("\n"):
        for text in extract_texts_from_line(line):
            if len(text) > len(last_text):
                last_text = text
    if not last_text:
        LOGGER.warning("Gemini upstream returned no parseable text (raw len=%d)", len(raw))
    return clean_text(last_text)
