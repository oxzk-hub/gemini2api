"""Gemini Web 请求构造。"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import uuid
from typing import Any


# StreamGenerate 与页面 token 都发往这个源站.
GEMINI_WEB_BASE_URL: str = "https://gemini.google.com"

_TOKEN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("push_id", r'"qKIAYe":"([^"]+)"'),
    ("pctx", r'"Ylro7b":"([^"]+)"'),
    ("at", r'"thykhd":"([^"]+)"'),
    ("bl", r'"cfb2h":"([^"]+)"'),
)


def extract_page_tokens(html: str) -> dict[str, str]:
    """从 Gemini 页面 HTML 提取 push_id, pctx, at 和 bl。"""
    tokens: dict[str, str] = {}
    for key, pattern in _TOKEN_PATTERNS:
        match: re.Match[str] | None = re.search(pattern, html)
        if match:
            tokens[key] = match.group(1)
    if "bl" not in tokens:
        fallback: re.Match[str] | None = re.search(r"boq_assistant-bard-web-server_[0-9._p]+", html)
        if fallback:
            tokens["bl"] = fallback.group(0)
    return tokens


def build_headers() -> dict[str, str]:
    """构建 StreamGenerate 请求头, 对齐 Node buildHeaders。"""
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": "https://gemini.google.com/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }


def build_payload(
    prompt: str,
    model_id: int,
    think_mode: int,
    xsrf_token: str | None,
) -> str:
    """构建 f.req 表单体, 对齐 Node buildPayload。"""
    inner: list[Any] = [None] * 102
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id

    # JSON.stringify(inner), followed by JSON.stringify([null, innerString]).
    inner_json: str = json.dumps(inner, ensure_ascii=False, separators=(",", ":"))
    outer_json: str = json.dumps(
        [None, inner_json],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    params: dict[str, str] = {"f.req": outer_json}
    if xsrf_token:
        params["at"] = xsrf_token
    return urllib.parse.urlencode(params)


def build_stream_url(gemini_bl: str) -> str:
    """构建 StreamGenerate URL, 对齐 Node getUrl。"""
    request_id: int = int(time.time() * 1000) % 1_000_000
    return (
        f"{GEMINI_WEB_BASE_URL}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={gemini_bl}&hl=en&_reqid={request_id}&rt=c"
    )
