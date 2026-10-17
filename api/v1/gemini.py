"""Gemini 原生 API 路由。"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas.gemini import GeminiGenerateRequest
from core.exceptions import GeminiError
from services.gemini import GeminiService
from utils import MODELS, resolve_model

router: APIRouter = APIRouter(tags=["Gemini"])


def _service(request: Request) -> GeminiService:
    """获取应用生命周期创建的 Gemini 服务。"""
    service: GeminiService | None = getattr(request.app.state, "gemini_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Gemini service is unavailable")
    return service


def _tool_prompt(request_body: GeminiGenerateRequest) -> str:
    """把 Gemini Function Calling 定义转换为原项目提示格式。"""
    tool_config: dict[str, Any] = request_body.toolConfig or {}
    calling: dict[str, Any] = tool_config.get("functionCallingConfig", {})
    mode: str = str(calling.get("mode", "AUTO"))
    if mode == "NONE" or not request_body.tools:
        return ""
    definitions: list[dict[str, Any]] = []
    for group in request_body.tools:
        for function in group.get("functionDeclarations", []):
            item: dict[str, Any] = {"name": function.get("name", ""), "description": function.get("description", "")}
            parameters: Any = function.get("parameters") or function.get("parametersJsonSchema")
            if parameters:
                item["parameters"] = parameters
            definitions.append(item)
    if not definitions:
        return ""
    constraint: str = ""
    if mode == "ANY":
        names: list[str] = calling.get("allowedFunctionNames", [])
        constraint = f"\n\nIMPORTANT: You MUST call one of these tools: {', '.join(names)}." if names else "\n\nIMPORTANT: You MUST call at least one tool."
    return (
        "# Tool Use\n\n"
        "You can call the following tools to help accomplish tasks.\n"
        "Call format (use this exact format):\n"
        "```function_call\n"
        '{"name": "<tool_name>", "args": {<arguments>}}\n'
        "```\n\n"
        f"Available tools:\n{json.dumps(definitions, ensure_ascii=False, indent=2)}{constraint}"
    )


def contents_to_prompt(body: GeminiGenerateRequest) -> str:
    """按原项目规则将 Gemini 原生请求转换为 Gemini Web 提示文本。"""
    parts: list[str] = []
    if body.systemInstruction:
        system_text: str = " ".join(str(part.get("text", "")) for part in body.systemInstruction.get("parts", []) if part.get("text"))
        if system_text:
            parts.append(system_text)
    tool_text: str = _tool_prompt(body)
    if tool_text:
        parts.append(tool_text)
    for content in body.contents:
        role: str = str(content.get("role", "user"))
        text_parts: list[str] = []
        for part in content.get("parts", []):
            if part.get("text"):
                text_parts.append(str(part["text"]))
            elif part.get("functionCall"):
                text_parts.append(f"```function_call\n{json.dumps(part['functionCall'], ensure_ascii=False)}\n```")
            elif part.get("functionResponse"):
                response: dict[str, Any] = part["functionResponse"]
                text_parts.append(f"[Tool result for {response.get('name', '')}]: {json.dumps(response.get('response', {}), ensure_ascii=False)}")
        text: str = "\n".join(text_parts)
        parts.append(f"[Assistant]: {text}" if role == "model" else text)
    return "\n\n".join(item for item in parts if item)


def parse_function_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """解析原项目支持的 function_call 代码块。"""
    import re

    calls: list[dict[str, Any]] = []
    clean: str = text
    patterns: tuple[str, ...] = (r"```function_call\s*\n(.*?)\n```", r"(?:^|\n)function_call\s*\n(\{[^`]*?\})")
    for pattern in patterns:
        matches = re.findall(pattern, clean, re.DOTALL)
        for value in matches:
            try:
                data: dict[str, Any] = json.loads(value.strip())
                if "name" in data:
                    calls.append({"name": data["name"], "args": data.get("args", data.get("arguments", {}))})
            except (json.JSONDecodeError, TypeError):
                continue
        clean = re.sub(pattern, "", clean, flags=re.DOTALL).strip()
    return clean, calls


def _available_models(request: Request) -> tuple[str, ...]:
    """返回源码确认的模型列表, 可由环境配置显式缩小。"""
    return tuple(MODELS)


@router.get("/v1beta/models")
async def list_gemini_models(request: Request) -> dict[str, Any]:
    """返回 Gemini 原生模型列表。"""
    return {"models": [_model_info(request, model) for model in _available_models(request)]}


def _model_info(request: Request, model_name: str) -> dict[str, Any]:
    """构建 Gemini 原生模型详情。"""
    prefix: str = "/v1" if request.url.path.startswith("/v1/") else "/v1beta"
    return {
        "name": f"models/{model_name}",
        "baseModelId": model_name,
        "displayName": model_name,
        "description": MODELS[model_name]["description"],
        "version": "1.0.0",
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent", "countTokens"],
        "baseUrl": f"{request.url.scheme}://{request.url.netloc}{prefix}",
    }


@router.get("/v1beta/models/{model}")
@router.get("/v1/models/{model}")
async def get_gemini_model(request: Request, model: str) -> dict[str, Any]:
    """返回 Gemini 原生模型详情。"""
    model_name, _, _, _ = resolve_model(model)
    return _model_info(request, model_name)


def _native_response(model_name: str, prompt: str, text: str, has_tools: bool) -> dict[str, Any]:
    """构建 Gemini 原生非流式响应。"""
    parts: list[dict[str, Any]] = []
    if has_tools:
        clean, calls = parse_function_calls(text)
        if clean:
            parts.append({"text": clean})
        parts.extend({"functionCall": call} for call in calls)
        if not parts:
            parts.append({"text": text})
    else:
        parts.append({"text": text or "I apologize, but I was unable to generate a response. Please try again."})
    prompt_tokens: int = len(prompt) // 4
    output_tokens: int = len(text) // 4
    return {
        "candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": output_tokens, "totalTokenCount": prompt_tokens + output_tokens},
        "modelVersion": model_name,
    }


async def _native_stream(service: GeminiService, prompt: str, model_name: str, model_id: int, think_mode: int, extra: dict[int, Any] | None) -> AsyncIterator[str]:
    """生成 Gemini 原生 SSE 数据事件, 并处理已发送后的上游异常。"""
    full_text: str = ""
    try:
        async for delta in service.stream(prompt, model_id, think_mode, extra):
            full_text += delta
            event: dict[str, Any] = {"candidates": [{"content": {"parts": [{"text": delta}], "role": "model"}, "index": 0}], "modelVersion": model_name}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        prompt_tokens: int = len(prompt) // 4
        output_tokens: int = len(full_text) // 4
        final: dict[str, Any] = {"candidates": [{"finishReason": "STOP", "index": 0}], "usageMetadata": {"promptTokenCount": prompt_tokens, "candidatesTokenCount": output_tokens, "totalTokenCount": prompt_tokens + output_tokens}, "modelVersion": model_name}
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    except GeminiError as exc:
        error: dict[str, Any] = {"error": {"code": exc.status_code, "message": exc.message, "status": "UPSTREAM_ERROR"}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"


@router.post("/v1beta/models/{model}:generateContent")
async def generate_content(request: Request, model: str, body: GeminiGenerateRequest) -> dict[str, Any]:
    """处理 Gemini 原生非流式内容生成。"""
    model_name, model_id, think_mode, extra = resolve_model(model)
    prompt: str = contents_to_prompt(body)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty content")
    text: str = await _service(request).generate(prompt, model_id, think_mode, extra)
    has_tools: bool = bool(body.tools) and (body.toolConfig or {}).get("functionCallingConfig", {}).get("mode", "AUTO") != "NONE"
    return _native_response(model_name, prompt, text, has_tools)


@router.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(request: Request, model: str, body: GeminiGenerateRequest) -> StreamingResponse:
    """处理 Gemini 原生 SSE 流式内容生成。"""
    model_name, model_id, think_mode, extra = resolve_model(model)
    prompt: str = contents_to_prompt(body)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty content")
    has_tools: bool = bool(body.tools) and (body.toolConfig or {}).get("functionCallingConfig", {}).get("mode", "AUTO") != "NONE"
    if has_tools:
        text: str = await _service(request).generate(prompt, model_id, think_mode, extra)

        async def one_event() -> AsyncIterator[str]:
            """将工具调用结果包装为 Gemini 单事件流。"""
            yield f"data: {json.dumps(_native_response(model_name, prompt, text, True), ensure_ascii=False)}\n\n"

        return StreamingResponse(one_event(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    return StreamingResponse(
        _native_stream(_service(request), prompt, model_name, model_id, think_mode, extra),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/v1beta/models/{model}")
@router.post("/v1beta/models/{model}/generateContent")
@router.post("/v1/models/{model}")
@router.post("/v1/models/{model}/generateContent")
async def generate_content_variant(request: Request, model: str, body: GeminiGenerateRequest) -> dict[str, Any]:
    """兼容斜杠与裸 model 形式, 复用 generateContent 逻辑。"""
    return await generate_content(request, model, body)


@router.post("/v1beta/models/{model}/streamGenerateContent")
@router.post("/v1/models/{model}/streamGenerateContent")
async def stream_generate_content_variant(request: Request, model: str, body: GeminiGenerateRequest) -> StreamingResponse:
    """兼容斜杠形式, 复用 streamGenerateContent 逻辑。"""
    return await stream_generate_content(request, model, body)
