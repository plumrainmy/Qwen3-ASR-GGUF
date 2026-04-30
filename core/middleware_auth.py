import json
from starlette.types import ASGIApp, Receive, Scope, Send
from core.config import args

# 可配置无需认证的路径
EXCLUDE_PATHS = {"/", "/docs", "/openapi.json", "/favicon.ico", "/static"}


class TokenAuthMiddleware:
    """
    纯 ASGI Token 认证中间件

    不继承 BaseHTTPMiddleware，避免 StreamingResponse 管道包装问题。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")

        # 跳过无需认证的路径
        if path in EXCLUDE_PATHS or path.startswith("/public"):
            await self.app(scope, receive, send)
            return

        # 从 scope headers 中提取 Authorization
        headers_dict = dict(scope.get("headers", []))
        token = headers_dict.get(b"authorization", b"").decode()

        if token != f"Bearer {args.web_secret_key}":
            body = json.dumps(
                {"code": 401, "msg": "Invalid or missing token", "data": None}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)
