"""应用配置。"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 文档和示例里的公开字符串, 不能当作已配置的密钥.
INSECURE_API_KEY_VALUES: frozenset[str] = frozenset({"replace-with-your-secret-key"})


class Settings(BaseSettings):
    """从环境变量和 .env 文件读取应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    api_key: str = Field(default="", validation_alias="API_KEY")
    request_timeout_sec: float = Field(default=180.0, validation_alias="REQUEST_TIMEOUT_SEC", gt=0)
    gemini_bl: str = Field(
        default="boq_assistant-bard-web-server_20260716.08_p0",
        validation_alias="GEMINI_BL",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _require_api_key(cls, value: object) -> str:
        """拒绝空密钥和示例占位符, 未配置时进程不能启动."""
        text: str = "" if value is None else str(value).strip()
        if not text or text in INSECURE_API_KEY_VALUES:
            raise ValueError("API_KEY must be set to a non-placeholder secret")
        return text


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """创建并缓存应用统一配置实例。"""
    return Settings()


settings: Settings = get_settings()
