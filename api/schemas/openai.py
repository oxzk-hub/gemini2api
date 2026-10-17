"""OpenAI Chat Completions 协议模型。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIMessage(BaseModel):
    """表示一条 OpenAI 对话消息。"""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    """表示 Chat Completions 请求。"""

    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"


class OpenAIResponsesRequest(BaseModel):
    """表示 OpenAI Responses API 请求。"""

    model_config = ConfigDict(extra="allow")
    model: str = ""
    input: str | list[Any] | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    stream: bool = False


class OpenAIErrorBody(BaseModel):
    """表示 OpenAI 风格错误主体。"""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None
