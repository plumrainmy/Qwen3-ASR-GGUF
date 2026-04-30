import argparse
import torch
import os
import yaml


def _str2bool(value):
    """兼容 argparse 的布尔参数解析。"""
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {value}")


def __build_parser_args() -> argparse.Namespace:
    """
    构建参数解析器
    """
    parser = argparse.ArgumentParser(description="Qwen3 ASR GGUF Configuration")

    # 通用参数
    parser.add_argument(
        "--use_gpu",
        type=_str2bool,
        default=torch.cuda.is_available(),
        help="是否使用GPU预测",
    )
    parser.add_argument(
        "--web_secret_key", type=str, default="qwen3-asr-token", help="接口请求秘钥"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务启动IP地址")
    parser.add_argument("--port", type=int, default=8002, help="服务启动端口")
    parser.add_argument(
        "--base_url", type=str, default="/qwen3-asr/api/v1", help="接口基础路径"
    )
    parser.add_argument("--configs", type=str, default="", help="配置文件路径")

    # 使用 parse_known_args 防止未知参数报错
    args, _ = parser.parse_known_args()
    return args


args = __build_parser_args()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API configuration mapping args
    HOST: str = args.host
    PORT: int = args.port

    # Model configuration paths
    MODEL_DIR: str = "./models"
    DATA_DIR: str = "./datasets"
    HOTWORDS_PATH: str = "./hot-word.txt"

    # Transcription settings
    SIMILAR_THRESHOLD: float = 0.6
    MAX_HOTWORDS: int = 10
    ENABLE_CTC: bool = True

    # Global ASR engine defaults
    ASR_CHUNK_SIZE: float = 30.0
    ASR_MEMORY_NUM: int = 1
    ASR_DYNAMIC_CHUNK_THRESHOLD: float = (
        10.0  # 音频时长超过此阈值时自动启用 VAD 动态分片
    )
    DEFAULT_LANGUAGE: str = "Chinese"

    # Global aligner defaults
    ALIGNER_USE_GPU: bool = args.use_gpu

    # Global VAD defaults
    # VAD 由 ASR_DYNAMIC_CHUNK_THRESHOLD 自动触发（音频 > 阈值时延迟加载），无需手动开关
    VAD_MODEL_DIR: str = "./models/FireRedVAD/VAD"
    VAD_USE_GPU: bool = False
    VAD_SPEECH_THRESHOLD: float = 0.35  # 初始帧语音概率阈值（自适应算法会动态调整）
    VAD_MIN_DURATION: float = 10.0
    VAD_SMOOTH_WINDOW_SIZE: int = 5
    VAD_MIN_SPEECH_FRAME: int = 15  # 150ms 最短语音段，兼顾短促词语
    VAD_MAX_SPEECH_FRAME: int = 3000  # 30s 单语音段上限
    VAD_MIN_SILENCE_FRAME: int = 40  # 400ms 最短静音，避免句内短停顿导致语义割裂
    VAD_MERGE_SILENCE_FRAME: int = 30  # 合并 <300ms 间隔的相邻语音段
    VAD_EXTEND_SPEECH_FRAME: int = 8  # 语音边界向外扩展 80ms，捕捉词首/尾音
    VAD_CHUNK_MAX_FRAME: int = 30000

    # Upload settings
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 120

    # Inference CPU thread settings
    # 0 = 自动 (n_threads = cpu_count//4, n_threads_batch = cpu_count)
    INFERENCE_CPU_THREADS: int = 16
    INFERENCE_CPU_THREADS_BATCH: int = 32

    # Default context for ASR
    DEFAULT_CONTEXT: str = ""

    @property
    def upload_dir_path(self) -> str:
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        return self.UPLOAD_DIR

    class Config:
        env_prefix = "ASR_"


settings = Settings()
