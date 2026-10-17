"""FastAPI 应用创建与资源组装入口。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import api_router
from api.exception_handlers import register_exception_handlers
from api.middleware import APIKeyMiddleware
from core.config import Settings, settings
from services.gemini import GeminiService
from utils.logger import Logger

# 站点图标目录, 路径相对本文件, 不依赖进程工作目录.
_PUBLIC_DIR: Path = Path(__file__).resolve().parent / "public"

Logger.configure()


def create_app(app_settings: Settings | None = None) -> FastAPI:
    """创建 FastAPI 应用实例。"""
    selected_settings: Settings = app_settings or settings

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """用本次 create_app 的配置创建 Gemini 服务, token 惰性获取。"""
        service: GeminiService = GeminiService(selected_settings)
        app.state.gemini_service = service
        try:
            yield
        finally:
            await service.close()

    app: FastAPI = FastAPI(
        title="gemini2api",
        lifespan=lifespan,
    )
    app.state.settings = selected_settings
    # 后添加的中间件在外层: 保持 CORS 包住鉴权, 401 响应也带跨域头。
    app.add_middleware(APIKeyMiddleware, app_settings=selected_settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    _register_public_routes(app)
    register_exception_handlers(app)
    return app


def _register_public_routes(app: FastAPI) -> None:
    """把 public 目录中的站点图标挂到浏览器会直接请求的路径。"""

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_ico() -> FileResponse:
        """返回 ICO 站点图标。"""
        return FileResponse(_PUBLIC_DIR / "favicon.ico", media_type="image/x-icon")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon_svg() -> FileResponse:
        """返回 SVG 站点图标。"""
        return FileResponse(_PUBLIC_DIR / "favicon.svg", media_type="image/svg+xml")


app: FastAPI = create_app()
