"""项目内部异常。"""


class GeminiError(Exception):
    """表示 Gemini Web 上游调用失败。"""

    def __init__(self, message: str, status_code: int = 502) -> None:
        """保存安全的客户端消息和 HTTP 状态码。"""
        super().__init__(message)
        self.message: str = message
        self.status_code: int = status_code


class GeminiTimeoutError(GeminiError):
    """表示 Gemini Web 上游请求超时。"""

    def __init__(self, message: str = "Gemini upstream request timed out") -> None:
        """创建超时异常。"""
        super().__init__(message, status_code=504)


class GeminiInvalidResponseError(GeminiError):
    """表示 Gemini Web 上游返回无法解析的响应。"""

    def __init__(self, message: str = "Invalid response from Gemini upstream") -> None:
        """创建无效响应异常。"""
        super().__init__(message, status_code=502)
