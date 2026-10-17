"""Gemini 原生 API 协议模型。"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class GeminiGenerateRequest(BaseModel):
    """表示 Gemini generateContent 请求, 保留未知字段供上游参数使用。"""

    model_config = ConfigDict(extra="allow")
    contents: list[dict[str, Any]]
    systemInstruction: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    toolConfig: dict[str, Any] | None = None
    generationConfig: dict[str, Any] | None = None


class GeminiErrorBody(BaseModel):
    """表示 Gemini 原生错误主体。"""

    error: dict[str, Any]
