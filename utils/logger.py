"""项目统一日志。"""

from __future__ import annotations

import logging


def _qualified_name(name: str) -> str:
    """把模块后缀收成 gemini2api 下的日志名。"""
    cleaned: str = name.strip().strip(".")
    if not cleaned or cleaned == "gemini2api":
        return "gemini2api"
    if cleaned.startswith("gemini2api."):
        return cleaned
    return f"gemini2api.{cleaned}"


class Logger:
    """gemini2api 命名空间下的日志器。"""

    _configured: bool = False
    """是否已经写入根日志格式。"""

    def __init__(self, name: str = "") -> None:
        """按模块后缀创建日志器。空名称对应根日志 gemini2api。"""
        self.configure()
        self._logger: logging.Logger = logging.getLogger(_qualified_name(name))

    @property
    def name(self) -> str:
        """返回底层日志器名称。"""
        return self._logger.name

    @classmethod
    def configure(cls) -> None:
        """初始化根日志格式。重复调用不会叠加 handler。"""
        if cls._configured:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        cls._configured = True

    def debug(self, message: str, *args: object) -> None:
        """记录调试日志。"""
        self._logger.debug(message, *args)

    def info(self, message: str, *args: object) -> None:
        """记录信息日志。"""
        self._logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        """记录警告日志。"""
        self._logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        """记录错误日志。"""
        self._logger.error(message, *args)

    def exception(self, message: str, *args: object) -> None:
        """记录错误日志并附带当前异常堆栈。"""
        self._logger.exception(message, *args)
