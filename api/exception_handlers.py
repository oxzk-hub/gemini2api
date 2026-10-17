"""全局异常处理器。"""

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions import GeminiError
from utils.logger import Logger

LOGGER: Logger = Logger()


def _is_gemini_request(request: Request) -> bool:
    """判断请求是否使用 Gemini 原生协议。"""
    return request.url.path.startswith("/v1beta/")


def _error_response(request: Request, status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    """按请求协议生成安全错误响应。"""
    if _is_gemini_request(request):
        payload: dict[str, Any] = {"error": {"code": status_code, "message": message, "status": error_type.upper()}}
    else:
        payload = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return JSONResponse(status_code=status_code, content=payload)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """处理显式 HTTP 异常。"""
    detail: str = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return _error_response(request, exc.status_code, detail)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """处理请求校验异常, 不泄露内部堆栈。"""
    return _error_response(request, 422, "Request validation failed", "invalid_request_error")


async def gemini_exception_handler(request: Request, exc: GeminiError) -> JSONResponse:
    """处理上游 Gemini 内部异常。"""
    LOGGER.warning("Gemini upstream error (%s): %s", exc.status_code, exc.message)
    return _error_response(request, exc.status_code, exc.message, "upstream_error")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """记录未预期异常并返回安全错误。"""
    LOGGER.exception("Unhandled request error")
    return _error_response(request, 500, "Internal server error", "internal_error")


def register_exception_handlers(app: FastAPI) -> None:
    """为应用注册统一异常处理器。"""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(GeminiError, gemini_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
