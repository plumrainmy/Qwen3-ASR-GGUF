import time
from starlette.types import ASGIApp, Receive, Scope, Send
from core.logger import logger


class AccessLogMiddleware:
    """
    纯 ASGI 访问日志中间件

    重要: 不继承 BaseHTTPMiddleware！
    BaseHTTPMiddleware 会通过 anyio.MemoryObjectStream 管道包装 StreamingResponse 的 body，
    对长时间运行的 SSE 流 (如 1 小时音频转写) 极易导致管道断裂 / 背压超时。
    改为纯 ASGI 中间件可让 StreamingResponse 直通客户端，彻底避免此类问题。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        # 从 scope 中提取请求信息
        client = scope.get("client")
        client_host = client[0] if client else "unknown"
        method = scope.get("method", "?")
        path = scope.get("path", "/")

        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            cost_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[REQ] 500 {method} {path} from {client_host} " f"cost={cost_ms:.2f}ms"
            )
            raise

        cost_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"[REQ] {status_code} {method} {path} "
            f"from {client_host} cost={cost_ms:.2f}ms"
        )
