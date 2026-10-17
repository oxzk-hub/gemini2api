"""OpenAI 兼容 API 路由。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas.openai import OpenAIChatRequest, OpenAIMessage, OpenAIResponsesRequest
from core.exceptions import GeminiError
from services.gemini import GeminiService
from utils import MODELS, resolve_model
from utils.logger import Logger

router: APIRouter = APIRouter(tags=["OpenAI"])
LOGGER: Logger = Logger("api")


def _service(request: Request) -> GeminiService:
    """获取应用生命周期创建的 Gemini 服务。"""
    service: GeminiService | None = getattr(request.app.state, "gemini_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Gemini service is unavailable")
    return service


def _tool_prompt(tools: list[dict[str, Any]] | None, tool_choice: Any) -> str:
    """将 OpenAI 工具定义转换为原项目使用的提示文本。"""
    if not tools or tool_choice == "none":
        return ""
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = tool.get("function", tool)
        definitions.append({
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        })
    instruction: str = ""
    if tool_choice == "required":
        instruction = "\n\nIMPORTANT: You MUST call at least one tool. Do not respond with text only."
    elif isinstance(tool_choice, dict):
        name: str = str(tool_choice.get("function", {}).get("name", ""))
        if name:
            instruction = f'\n\nIMPORTANT: You MUST call the tool "{name}". Do not call other tools.'
    return (
        "# Tool Use\n\n"
        "You can call the following tools. Call format:\n"
        '```tool_call\n{"name": "func_name", "arguments": {...}}\n```\n'
        "When calling tools, output ONLY the tool_call block(s).\n\n"
        f"Available tools:\n{json.dumps(definitions, ensure_ascii=False, indent=2)}{instruction}"
    )


def _message_content(content: str | list[dict[str, Any]] | None) -> str:
    """把 OpenAI 消息内容转换为纯文本, 明确拒绝本次范围外的图片能力。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if part.get("type") in {"text", "input_text"}:
            parts.append(str(part.get("text", "")))
    return " ".join(parts)


def messages_to_prompt(messages: list[OpenAIMessage], tools: list[dict[str, Any]] | None, tool_choice: Any) -> str:
    """按原项目规则将 OpenAI 消息转换为 Gemini Web 提示文本。"""
    parts: list[str] = []
    tool_text: str = _tool_prompt(tools, tool_choice)
    if tool_text:
        parts.append(tool_text)
    for message in messages:
        content: str = _message_content(message.content)
        if message.role == "system":
            parts.append(f"[System instruction]: {content}")
        elif message.role == "assistant":
            calls: list[dict[str, Any]] = message.tool_calls or []
            call_text: list[str] = []
            for call in calls:
                function: dict[str, Any] = call.get("function", {})
                raw_arguments: Any = function.get("arguments", "{}")
                try:
                    arguments: Any = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except (json.JSONDecodeError, TypeError):
                    arguments = raw_arguments
                call_text.append(
                    "```tool_call\n"
                    + json.dumps({"name": function.get("name", ""), "arguments": arguments}, ensure_ascii=False)
                    + "\n```"
                )
            parts.append(f"[Assistant]: {content or ''}\n" + "\n".join(call_text))
        elif message.role == "tool":
            parts.append(f"[Tool result for {message.name or ''}]: {content}")
        else:
            parts.append(content)
    return "\n\n".join(item for item in parts if item)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """提取原项目约定的 tool_call 代码块。"""
    import re

    calls: list[dict[str, Any]] = []
    pattern: str = r"```tool_call\s*\n(.*?)\n```"
    clean_parts: list[str] = []
    last_end: int = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        clean_parts.append(text[last_end:match.start()])
        last_end = match.end()
        try:
            data: dict[str, Any] = json.loads(match.group(1).strip())
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": data["name"], "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)},
            })
        except (json.JSONDecodeError, KeyError):
            continue
    clean_parts.append(text[last_end:])
    return "".join(clean_parts).strip(), calls


def _usage(prompt: str, text: str) -> dict[str, int]:
    """按原项目字符近似算法生成用量字段。"""
    prompt_tokens: int = len(prompt) // 4
    completion_tokens: int = len(text) // 4
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def _available_models(request: Request) -> tuple[str, ...]:
    """返回源码确认的模型列表, 允许部署配置显式缩小范围。"""
    return tuple(MODELS)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    """返回 OpenAI 兼容模型列表。"""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "google",
                "description": MODELS[model]["description"],
            }
            for model in _available_models(request)
        ],
    }


async def _openai_stream(service: GeminiService, prompt: str, model_name: str, model_id: int, think_mode: int, extra: dict[int, Any] | None, completion_id: str) -> AsyncIterator[str]:
    """生成 OpenAI SSE 事件, 并将流中异常转换为安全事件。"""
    created: int = int(time.time())
    try:
        async for delta in service.stream(prompt, model_id, think_mode, extra):
            chunk: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        end: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
    except GeminiError as exc:
        error: dict[str, Any] = {"error": {"message": exc.message, "type": "upstream_error", "code": None}}
        yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        return
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: OpenAIChatRequest) -> Any:
    """处理 OpenAI Chat Completions 非流式和 SSE 流式请求。"""
    model_name, model_id, think_mode, extra = resolve_model(body.model)
    prompt: str = messages_to_prompt(body.messages, body.tools, body.tool_choice)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    service: GeminiService = _service(request)
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    native_stream: bool = body.stream and (not body.tools or body.tool_choice == "none")
    LOGGER.info(
        "chat: model=%s stream=%s tools=%s prompt_len=%d",
        model_name,
        body.stream,
        bool(body.tools) and body.tool_choice != "none",
        len(prompt),
    )
    if native_stream:
        return StreamingResponse(
            _openai_stream(service, prompt, model_name, model_id, think_mode, extra, completion_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    text: str = await service.generate(prompt, model_id, think_mode, extra)
    tool_calls: list[dict[str, Any]] = []
    if body.tools and body.tool_choice != "none":
        text, tool_calls = parse_tool_calls(text)
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish_reason: str = "tool_calls" if tool_calls else "stop"
    result: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(prompt, text),
    }
    if body.stream:
        async def one_event() -> AsyncIterator[str]:
            """将工具调用或禁用原生流的结果包装为单个 SSE 事件。"""
            chunk: dict[str, Any] = {"id": completion_id, "object": "chat.completion.chunk", "created": result["created"], "model": model_name, "choices": [{"index": 0, "delta": message, "finish_reason": finish_reason}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(one_event(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    return result


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 Responses 扁平 function 工具规范为 Chat 格式。"""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function" and "function" not in tool and isinstance(tool.get("name"), str):
            normalized.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
        else:
            normalized.append(tool)
    return normalized


def _function_call_arguments(arguments: Any) -> str:
    """把工具参数收成 JSON 字符串。"""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments or {}, ensure_ascii=False)


def _function_output_text(output: Any) -> str:
    """把 function_call_output 收成文本。"""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text", "")))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(item for item in parts if item)
    return json.dumps(output, ensure_ascii=False)


def _responses_input_to_messages(body: OpenAIResponsesRequest) -> list[OpenAIMessage]:
    """把 Responses input/instructions 转换为 Chat 消息。"""
    messages: list[OpenAIMessage] = []
    if body.instructions:
        messages.append(OpenAIMessage(role="system", content=body.instructions))

    items: Any = body.input
    if isinstance(items, str):
        messages.append(OpenAIMessage(role="user", content=items))
        return messages
    if not isinstance(items, list):
        return messages

    # Codex 把 function_call 做成独立 input item, 用 call_id 对应后面的 output.
    call_names: dict[str, str] = {}

    for item in items:
        if isinstance(item, str):
            messages.append(OpenAIMessage(role="user", content=item))
            continue
        if not isinstance(item, dict):
            continue

        if item.get("type") == "function_call":
            call_id: str = str(item.get("call_id") or item.get("id") or "")
            name: str = str(item.get("name", ""))
            for key in (item.get("call_id"), item.get("id")):
                if key:
                    call_names[str(key)] = name
            messages.append(OpenAIMessage(
                role="assistant",
                content=None,
                tool_calls=[{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _function_call_arguments(item.get("arguments")),
                    },
                }],
            ))
            continue

        if item.get("type") == "function_call_output":
            call_id = str(item.get("call_id") or item.get("id") or "")
            messages.append(OpenAIMessage(
                role="tool",
                name=str(item.get("name") or call_names.get(call_id, "")),
                content=_function_output_text(item.get("output")),
            ))
            continue

        role: str = str(item.get("role", "user"))
        content: Any = item.get("content")

        if role == "assistant":
            text_acc: str = ""
            tool_calls: list[dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "output_text":
                        text_acc += str(part.get("text", ""))
                    elif part.get("type") == "function_call":
                        part_name: str = str(part.get("name", ""))
                        part_id: str = str(part.get("call_id") or part.get("id") or "")
                        if part_id:
                            call_names[part_id] = part_name
                        tool_calls.append({
                            "id": part_id,
                            "type": "function",
                            "function": {
                                "name": part_name,
                                "arguments": _function_call_arguments(part.get("arguments")),
                            },
                        })
            elif isinstance(content, str):
                text_acc = content
            messages.append(OpenAIMessage(
                role="assistant",
                content=text_acc or None,
                tool_calls=tool_calls or None,
            ))
            continue

        if isinstance(content, list):
            text: str = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
            )
        else:
            text = "" if content is None else str(content)
        messages.append(OpenAIMessage(role=role, content=text))

    return messages


def _responses_stream_events(
    response_id: str,
    model_name: str,
    created_at: int,
    output: list[dict[str, Any]],
    usage: dict[str, int],
) -> list[tuple[str, dict[str, Any]]]:
    """把已完成的 Responses 输出展开成 SDK 状态机需要的 SSE 事件。"""

    def response_body(status: str, items: list[dict[str, Any]], with_usage: bool) -> dict[str, Any]:
        """构造响应资源. 完成前不带 usage。"""
        payload: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": model_name,
            "output": items,
        }
        if with_usage:
            payload["usage"] = usage
        return payload

    events: list[tuple[str, dict[str, Any]]] = [
        ("response.created", {
            "type": "response.created",
            "response": response_body("in_progress", [], False),
        }),
        ("response.in_progress", {
            "type": "response.in_progress",
            "response": response_body("in_progress", [], False),
        }),
    ]

    for output_index, item in enumerate(output):
        if item["type"] == "function_call":
            events.append(("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": item["id"],
                    "call_id": item["call_id"],
                    "name": item["name"],
                    "arguments": "",
                    "status": "in_progress",
                },
            }))
            arguments: str = str(item.get("arguments", ""))
            if arguments:
                events.append(("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": arguments,
                }))
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": output_index,
                "name": item["name"],
                "call_id": item["call_id"],
                "arguments": arguments,
            }))
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            }))
            continue

        events.append(("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": item["id"],
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }))
        for content_index, part in enumerate(item.get("content", [])):
            text: str = str(part.get("text", ""))
            events.append(("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))
            if text:
                events.append(("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text,
                }))
            events.append(("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
                "text": text,
            }))
            events.append(("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": item["id"],
                "output_index": output_index,
                "content_index": content_index,
                "part": {
                    "type": "output_text",
                    "text": text,
                    "annotations": part.get("annotations", []),
                },
            }))
        events.append(("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }))

    events.append(("response.completed", {
        "type": "response.completed",
        "response": response_body("completed", output, True),
    }))
    return events


@router.post("/v1/responses")
async def responses(request: Request, body: OpenAIResponsesRequest) -> Any:
    """处理 OpenAI Responses API (Codex CLI) 请求。"""
    service: GeminiService = _service(request)
    tools: list[dict[str, Any]] | None = _normalize_tools(body.tools) if body.tools else None
    messages: list[OpenAIMessage] = _responses_input_to_messages(body)
    prompt: str = messages_to_prompt(messages, tools, body.tool_choice)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="empty input")

    model_name, model_id, think_mode, extra = resolve_model(body.model)
    LOGGER.info(
        "responses: model=%s stream=%s tools=%s prompt_len=%d",
        model_name,
        body.stream,
        bool(tools),
        len(prompt),
    )
    text: str = await service.generate(prompt, model_id, think_mode, extra)

    content: str = text or ""
    tool_calls: list[dict[str, Any]] = []
    if tools and body.tool_choice != "none" and text:
        content, tool_calls = parse_tool_calls(text)

    response_id: str = f"resp_{uuid.uuid4().hex[:16]}"
    message_id: str = f"msg_{uuid.uuid4().hex[:12]}"
    output: list[dict[str, Any]] = []
    for call in tool_calls:
        function: dict[str, Any] = call["function"]
        output.append({
            "type": "function_call",
            "id": call["id"],
            "call_id": call["id"],
            "name": function["name"],
            "arguments": function["arguments"],
            "status": "completed",
        })
    if content or not tool_calls:
        output.append({
            "type": "message",
            "id": message_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })

    usage: dict[str, int] = {
        "input_tokens": len(prompt) // 4,
        "output_tokens": len(text) // 4,
        "total_tokens": (len(prompt) + len(text)) // 4,
    }
    created_at: int = int(time.time())

    if body.stream:
        async def event_stream() -> AsyncIterator[str]:
            """按 Responses 事件顺序下发, 每个事件单独一帧。"""
            for name, data in _responses_stream_events(response_id, model_name, created_at, output, usage):
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model_name,
        "output": output,
        "usage": usage,
    }
