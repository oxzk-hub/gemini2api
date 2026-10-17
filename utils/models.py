"""客户端模型名到 Gemini Web 上游参数的映射。"""

from typing import Any


MODELS: dict[str, dict[str, Any]] = {
    "gemini-3.6-flash": {"mode": 1, "think": 4, "description": "All-around model"},
}
DEFAULT_MODEL_ID: str = "gemini-3.6-flash"


def resolve_model(
    model_name: str,
    default: str | None = None,
) -> tuple[str, int, int, dict[int, Any] | None]:
    """忽略客户端模型标识, 固定返回默认 Gemini Flash 上游参数。"""
    del model_name, default
    model_config: dict[str, Any] = MODELS[DEFAULT_MODEL_ID]
    return DEFAULT_MODEL_ID, int(model_config["mode"]), int(model_config["think"]), None
