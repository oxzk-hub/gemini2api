"""请求级认证中间件。"""

import hmac
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from core.config import INSECURE_API_KEY_VALUES, Settings, settings
from utils.logger import Logger

LOGGER: Logger = Logger("auth")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """对 /v1 与 /v1beta 接口执行单 key 认证, 支持多种传递方式。"""

    def __init__(self, app: Any, app_settings: Settings = settings) -> None:
        """保存配置并初始化中间件。"""
        super().__init__(app)
        self.settings: Settings = app_settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """校验受保护路径。"""
        if not self._is_protected(request.url.path):
            return await call_next(request)
        configured_key: str = (self.settings.api_key or "").strip()
        if not configured_key or configured_key in INSECURE_API_KEY_VALUES:
            # 未配置密钥时拒绝, 避免监听公网时变成开放代理.
            LOGGER.warning("API_KEY is not configured; rejecting %s", request.url.path)
            return JSONResponse(status_code=401, content={"error": {"message": "invalid api key"}})
        supplied_key: str | None = self._extract_key(request)
        if supplied_key and hmac.compare_digest(supplied_key, configured_key):
            return await call_next(request)
        return JSONResponse(status_code=401, content={"error": {"message": "invalid api key"}})

    @staticmethod
    def _is_protected(path: str) -> bool:
        """与 Node 版一致: /v1 与 /v1beta 下全部接口需要鉴权。"""
        return path.startswith("/v1/") or path.startswith("/v1beta/")

    @staticmethod
    def _extract_key(request: Request) -> str | None:
        """依次读取 Bearer、x-api-key、x-goog-api-key 与 ?key=。"""
        authorization: str = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and token:
            return token
        for header in ("x-api-key", "x-goog-api-key"):
            value: str | None = request.headers.get(header)
            if value:
                return value
        return request.query_params.get("key") or None
