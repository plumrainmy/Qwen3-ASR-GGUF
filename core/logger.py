import sys
import os
from loguru import logger
from contextvars import ContextVar
from pathlib import Path

request_id_ctx = ContextVar("request_id", default="-")

BASE_DIR = Path(__file__).resolve().parent
LOG_ROOT = BASE_DIR / "logs"

logger.remove()

# 生产级配置：根据环境变量控制日志级别
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")  # production 或 development
CONSOLE_LEVEL = "WARNING" if ENVIRONMENT == "production" else "DEBUG"
FILE_LEVEL = "INFO"  # 文件日志保留 INFO 级别用于审计

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level:<8} | "
    "{extra[request_id]} | "
    "{message}"
)

# 控制台：生产环境只输出 WARNING 及以上，开发环境输出 DEBUG 及以上
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=CONSOLE_LEVEL,
)

# 应用日志：所有 INFO 及以上级别的日志
logger.add(
    LOG_ROOT / "{time:YYYY-MM-DD}/app.log",
    format=LOG_FORMAT,
    level=FILE_LEVEL,
    rotation="100 MB",  # 单文件最大 100MB
    retention="7 days",
    compression="zip",
    enqueue=True,
)

# 错误日志：单独收集 ERROR 及以上级别
logger.add(
    LOG_ROOT / "{time:YYYY-MM-DD}/error.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="100 MB",  # 单文件最大 100MB
    retention="14 days",
    enqueue=True,
)

# 调试日志：仅在开发环境写入 DEBUG 及以上级别的日志
if ENVIRONMENT == "development":
    logger.add(
        LOG_ROOT / "{time:YYYY-MM-DD}/debug.log",
        format=LOG_FORMAT,
        level="DEBUG",
        rotation="100 MB",
        retention="3 days",
        enqueue=True,
    )


# 拦截器：自动注入 request_id
def inject_request_id(record):
    record["extra"]["request_id"] = request_id_ctx.get()
    return record


logger = logger.patch(inject_request_id)
