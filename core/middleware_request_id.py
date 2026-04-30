import uuid
from starlette.types import ASGIApp, Receive, Scope, Send
from core.logger import request_id_ctx


class RequestIDMiddleware:
    """
    纯 ASGI 请求 ID 中间件

    不继承 BaseHTTPMiddleware，避免 StreamingResponse 管道包装问题。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        request_id_ctx.set(request_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)
