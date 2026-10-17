"""Gemini Web StreamGenerate 客户端。"""

from __future__ import annotations

import codecs
import time
from typing import Any, AsyncIterator

from curl_cffi.requests import AsyncSession, Response
from curl_cffi.requests.exceptions import RequestException, Timeout

from core.config import Settings, settings
from core.exceptions import GeminiError, GeminiTimeoutError
from utils.parse import (
    StreamTextAssembler,
    extract_response_text,
    extract_texts_from_line,
    pop_complete_lines,
    raise_if_bard_error,
)
from utils.logger import Logger
from utils.request import GEMINI_WEB_BASE_URL, build_headers, build_payload, build_stream_url, extract_page_tokens


LOGGER: Logger = Logger("upstream")

TOKEN_FETCH_TIMEOUT_SECONDS: float = 15.0


class GeminiService:
    """管理 Gemini Web HTTP 客户端, 请求构建和响应解析。"""

    def __init__(
        self,
        app_settings: Settings = settings,
        client: AsyncSession | None = None,
    ) -> None:
        """创建 Gemini 上游服务。"""
        self.settings: Settings = app_settings
        self._client: AsyncSession = client or AsyncSession(
            timeout=self.settings.request_timeout_sec,
            allow_redirects=True,
            trust_env=False,
            impersonate="chrome",
        )
        self._owns_client: bool = client is None

    async def close(self) -> None:
        """关闭服务创建的 HTTP 客户端。"""
        if self._owns_client:
            await self._client.close()

    async def _resolve_runtime(self) -> tuple[str, str | None]:
        """请求时从 Gemini 页面读取 bl/at token, 不使用缓存。"""
        started: float = time.perf_counter()
        try:
            response: Response = await self._client.get(
                f"{GEMINI_WEB_BASE_URL}/app",
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36"
                    ),
                },
                timeout=TOKEN_FETCH_TIMEOUT_SECONDS,
            )
            tokens: dict[str, str] = extract_page_tokens(response.text)
            LOGGER.info(
                "Gemini token fetch: %.2fs (bl=%s at=%s)",
                time.perf_counter() - started,
                tokens.get("bl"),
                tokens.get("at"),
            )
            if not tokens:
                LOGGER.warning("Gemini page token extraction found nothing, falling back to GEMINI_BL")
            return tokens.get("bl") or self.settings.gemini_bl, tokens.get("at") or None
        except Exception as exc:
            LOGGER.warning(
                "Gemini page token fetch failed after %.2fs: %s",
                time.perf_counter() - started,
                exc,
            )
            return self.settings.gemini_bl, None

    @staticmethod
    def _http_error(response: Response, raw: str) -> GeminiError:
        """上游 HTTP 失败统一映射为 502. raw 只留给调用方记日志。"""
        del raw
        return GeminiError(
            f"Gemini upstream HTTP {response.status_code}",
            status_code=502,
        )

    async def generate(
        self,
        prompt: str,
        model_id: int,
        think_mode: int,
        extra_fields: dict[int, Any] | None = None,
    ) -> str:
        """非流式生成。"""
        gemini_bl, xsrf_token = await self._resolve_runtime()
        body: str = build_payload(prompt, model_id, think_mode, xsrf_token)
        url: str = build_stream_url(gemini_bl)
        del extra_fields

        started: float = time.perf_counter()
        try:
            response: Response = await self._client.post(
                url,
                headers=build_headers(),
                data=body,
            )
        except Timeout:
            LOGGER.warning(
                "Gemini generate timeout after %.2fs (%ss)",
                time.perf_counter() - started,
                self.settings.request_timeout_sec,
            )
            raise GeminiTimeoutError()
        except RequestException as exc:
            LOGGER.warning("Gemini generate request failed after %.2fs: %s", time.perf_counter() - started, exc)
            raise GeminiError(f"Gemini upstream request failed: {exc}")

        elapsed: float = time.perf_counter() - started
        raw: str = response.text
        LOGGER.info("Gemini generate: %.2fs (HTTP %s)", elapsed, response.status_code)
        if not response.ok:
            LOGGER.warning("Gemini generate HTTP %s: %s", response.status_code, raw[:200])
            raise self._http_error(response, raw)
        return extract_response_text(raw)

    async def stream(
        self,
        prompt: str,
        model_id: int,
        think_mode: int,
        extra_fields: dict[int, Any] | None = None,
    ) -> AsyncIterator[str]:
        """流式生成增量文本。"""
        gemini_bl, xsrf_token = await self._resolve_runtime()
        body: str = build_payload(prompt, model_id, think_mode, xsrf_token)
        url: str = build_stream_url(gemini_bl)
        del extra_fields

        assembler: StreamTextAssembler = StreamTextAssembler()
        started: float = time.perf_counter()
        first_chunk_at: float | None = None

        def take_deltas(lines: list[str]) -> list[str]:
            """从协议行里取出尚未下发的正文增量。"""
            deltas: list[str] = []
            for line in lines:
                for text in extract_texts_from_line(line):
                    delta: str | None = assembler.push(text)
                    if delta:
                        deltas.append(delta)
            return deltas

        try:
            async with self._client.stream(
                "POST",
                url,
                headers=build_headers(),
                data=body,
            ) as response:
                if not response.ok:
                    raw_bytes: bytes = await response.acontent()
                    raw: str = raw_bytes.decode(response.encoding or "utf-8", errors="replace")
                    LOGGER.warning("Gemini stream HTTP %s: %s", response.status_code, raw[:200])
                    raise self._http_error(response, raw)

                buffer: str = ""
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                async for chunk in response.aiter_content():
                    buffer += decoder.decode(chunk)
                    raise_if_bard_error(buffer)
                    lines, buffer = pop_complete_lines(buffer)
                    for delta in take_deltas(lines):
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        yield delta
                buffer += decoder.decode(b"", final=True)
                raise_if_bard_error(buffer)
                lines, rest = pop_complete_lines(buffer)
                if rest:
                    # 最后一帧经常没有换行, 不补上会把结尾正文留在缓冲里.
                    lines.append(rest)
                for delta in take_deltas(lines):
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    yield delta
            LOGGER.info(
                "Gemini stream: first=%s total=%.2fs",
                f"{first_chunk_at - started:.2f}s" if first_chunk_at else "n/a",
                time.perf_counter() - started,
            )
        except Timeout:
            LOGGER.warning(
                "Gemini stream timeout after %.2fs (%ss)",
                time.perf_counter() - started,
                self.settings.request_timeout_sec,
            )
            raise GeminiTimeoutError()
        except RequestException as exc:
            LOGGER.warning("Gemini stream request failed after %.2fs: %s", time.perf_counter() - started, exc)
            raise GeminiError(f"Gemini upstream stream failed: {exc}")
