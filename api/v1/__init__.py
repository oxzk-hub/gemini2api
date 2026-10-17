"""版本化 API 路由聚合。"""

import tomllib
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from api.v1.gemini import router as gemini_router
from api.v1.openai import router as openai_router
from utils import DEFAULT_MODEL_ID

api_router: APIRouter = APIRouter()


@lru_cache(maxsize=1)
def _project_version() -> str:
    """读取 pyproject.toml 的 project.version。"""
    project_file: Path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with project_file.open("rb") as handle:
        document: dict[str, object] = tomllib.load(handle)
    raw_project: object = document["project"]
    if not isinstance(raw_project, dict):
        raise RuntimeError("pyproject.toml project table is missing")
    raw_version: object = raw_project["version"]
    if not isinstance(raw_version, str) or not raw_version:
        raise RuntimeError("pyproject.toml project.version is missing")
    return raw_version


@api_router.get("/")
async def root() -> dict[str, object]:
    """健康检查。"""
    return {"status": "ok", "version": _project_version(), "models": [DEFAULT_MODEL_ID]}


api_router.include_router(openai_router)
api_router.include_router(gemini_router)

__all__ = ["api_router"]
