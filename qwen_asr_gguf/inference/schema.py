# coding=utf-8
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Tuple
import numpy as np


class MsgType(Enum):
    CMD_ENCODE = auto()  # 主进程 -> Encoder: 编码请求
    CMD_ALIGN = auto()  # 主进程 -> Aligner: 对齐请求
    CMD_STOP = auto()  # 主进程 -> Worker: 停止请求
    MSG_EMBD = auto()  # Worker -> 主进程: 返回特征 (Encoder)
    MSG_ALIGN = auto()  # Worker -> 主进程: 返回对齐结果 (Aligner)
    MSG_READY = auto()  # Worker -> 主进程: 就绪信号
    MSG_DONE = auto()  # Worker -> 主进程: 已退出信号
    MSG_ERROR = auto()  # Worker -> 主进程: 错误信号


@dataclass
class StreamingMessage:
    """音频编码/对齐进程通用通信协议"""
    msg_type: MsgType
    data: Any = None  # 存放音频 chunk 或 embedding/align 结果
    text: Optional[str] = None  # 用于对齐的文本
    offset_sec: float = 0.0  # 对齐的时间轴偏移
    language: Optional[str] = None  # 语言
    is_last: bool = False  # 标记是否为最后一段音频
    encode_time: float = 0.0  # 耗时统计


@dataclass
class DecodeResult:
    """LLM 解码内核输出标准化"""
    text: str = ""  # 包含前缀的完整文本
    new_text: str = ""  # 本次增量生成的文本
    stable_tokens: List[int] = field(default_factory=list)
    t_prefill: float = 0.0  # 预填充耗时 (ms)
    t_generate: float = 0.0  # 生成耗时 (ms)
    n_prefill: int = 0  # 预填充 token 数
    n_generate: int = 0  # 生成 token 数
    is_aborted: bool = False  # 是否因重复或其他原因熔断中断


@dataclass(frozen=True)
class ForcedAlignItem:
    """单个词/字符的对齐结果"""
    text: str
    start_time: float  # 单位：秒
    end_time: float  # 单位：秒


@dataclass
class ForcedAlignResult:
    """对齐结果标准化集合 (官方结构化输出格式)"""
    items: List[ForcedAlignItem]
    performance: Optional[dict] = None

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


@dataclass
class AlignerConfig:
    """对齐引擎配置"""
    model_dir: str
    # 拆分为 Frontend 和 Backend
    encoder_frontend_fn: str = "qwen3_aligner_encoder_frontend.fp16.onnx"
    encoder_backend_fn: str = "qwen3_aligner_encoder_backend.fp16.onnx"

    llm_fn: str = "qwen3_aligner_llm.f16.gguf"
    onnx_provider: str = 'CPU'  # CPU, CUDA, DML, TensorRT
    llm_use_gpu: bool = True
    n_ctx: int = 2048  # 对于 Aligner Decoder，每秒音频+文字，约占 30 个 token
    dml_pad_to: int = 40  # Encoder 填充时长


@dataclass
class VADConfig:
    """
    VAD (语音活动检测) 引擎配置

    基于 FireRedVAD Non-Streaming 模式，在 ASR 推理前对音频进行语音活动检测，
    自动跳过静音片段以降低 RTF 并减少幻觉。
    VAD 由 ASREngineConfig.dynamic_chunk_threshold 控制，当音频时长超过阈值时
    自动延迟加载，无需手动开关。

    speech_threshold: 初始帧语音概率阈值（自适应算法会根据概率分布动态调整）
    """

    model_dir: str = "models/FireRedVAD/VAD"
    use_gpu: bool = False
    # FireRedVadConfig 参数 (生产级默认值)
    smooth_window_size: int = 5
    speech_threshold: float = 0.35  # 初始阈值，自适应检测会动态调整
    min_speech_frame: int = 15  # 150ms 最短语音段
    max_speech_frame: int = 3000  # 30s 单语音段上限
    min_silence_frame: int = 40  # 400ms 最短静音，避免句内割裂
    merge_silence_frame: int = 30  # 合并 <300ms 间隔的近邻语音段
    extend_speech_frame: int = 8  # 语音边界向外扩展 80ms，捕捉词首/尾音
    chunk_max_frame: int = 30000  # 单分片最大帧数 (300s)
    # 控制参数：仅对超过此时长的片段启用 VAD 前置过滤 (秒)
    vad_min_duration: float = 10.0


@dataclass
class VADResult:
    """VAD 单次检测结果"""

    has_speech: bool  # 是否含有语音
    timestamps: List[Tuple[float, float]]  # 语音区间列表 [(start_sec, end_sec), ...]
    duration: float  # 音频总时长 (秒)
    detect_time: float = 0.0  # VAD 检测耗时 (秒)

    @property
    def speech_ratio(self) -> float:
        """语音占总时长的比例 (0~1)"""
        if self.duration <= 0:
            return 0.0
        speech_dur = sum(e - s for s, e in self.timestamps)
        return speech_dur / self.duration


@dataclass
class VADChunk:
    """VAD 引导分片：记录音频中一个逻辑分片的时间范围及语音情况。

    由 QwenVADEngine.build_chunks() 生成，供 _asr_core 主循环使用。
    has_speech=True  → 含有语音，需送入 Encoder + LLM 解码
    has_speech=False → 纯静音区间，直接跳过 ASR 推理
    """

    idx: int
    start_sec: float  # 在原始音频中的起始秒
    end_sec: float  # 在原始音频中的结束秒
    has_speech: bool  # 是否含有语音
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)

    # speech_segments: VAD 检测到的语音段列表 [(start_sec, end_sec), ...]，坐标相对于原始音频

    @property
    def span_sec(self) -> float:
        """分片的物理时间跨度（秒）"""
        return self.end_sec - self.start_sec

    @property
    def speech_sec(self) -> float:
        """分片内实际语音时长（秒）；无语音段列表时退化为 span_sec"""
        if self.speech_segments:
            return sum(e - s for s, e in self.speech_segments)
        return self.span_sec if self.has_speech else 0.0


@dataclass
class ASREngineConfig:
    """ASR 识别引擎配置"""
    model_dir: str
    encoder_frontend_fn: str = "qwen3_asr_encoder_frontend.fp16.onnx"
    encoder_backend_fn: str = "qwen3_asr_encoder_backend.fp16.onnx"
    llm_fn: str = "qwen3_asr_llm.f16.gguf"

    onnx_provider: str = 'CPU'  # CPU, CUDA, DML, TensorRT
    llm_use_gpu: bool = True
    dml_pad_to: int = 40  # 使用 DirectML 加速 onnx 时，Encoder 填充时长
    n_ctx: int = 2048  # 对于 ASR Decoder，每秒音频+文字，约占 20 个 token
    chunk_size: float = 40.0  # 每个片段 40s，对应 800 个 token
    memory_num: int = 1  # 记忆一个片段，转录一个片段，对应 1600 个 token
    verbose: bool = True
    enable_aligner: bool = False
    align_config: Optional[AlignerConfig] = None

    # VAD 与动态分片 ──────────────────────────────────────────────────
    # VAD 无需手动开关：当音频时长 > dynamic_chunk_threshold 时自动延迟加载
    vad_config: Optional[VADConfig] = None  # VAD 配置，None 时使用默认值
    dynamic_chunk_threshold: float = (
        10.0  # 音频时长 (秒) 超过此阈值时自动启用 VAD 动态分片
    )

    # CPU 推理线程配置 (0 = 自动)
    n_threads: int = 0
    n_threads_batch: int = 0

    def __post_init__(self):
        # 如果没有显式设置 Encoder 填充时长，则默认与 LLM 分段识别时长对齐
        if self.dml_pad_to is None:
            object.__setattr__(self, 'pad_to', int(self.chunk_size))

        if self.align_config is None:
            object.__setattr__(self, 'align_config', AlignerConfig(
                model_dir=self.model_dir,
                onnx_provider=self.onnx_provider,
                llm_use_gpu=self.llm_use_gpu,
                dml_pad_to=self.dml_pad_to  # Aligner 默认也跟随主 pad_to
            ))
        elif self.align_config.dml_pad_to is None:
            object.__setattr__(self.align_config, 'dml_pad_to', int(self.chunk_size))

        # VAD：始终保留默认配置（用于长音频自动启用动态分片时延迟加载）
        if self.vad_config is None:
            object.__setattr__(self, "vad_config", VADConfig())


@dataclass
class TranscribeResult:
    """ASR 转录结果 (含可选的对齐信息)"""
    text: str
    alignment: Optional[ForcedAlignResult] = None
    performance: Optional[dict] = None


@dataclass
class StreamChunkResult:
    """流式转录中单个音频分片的实时结果"""

    segment_idx: int  # 分片序号 (0-based)
    text: str  # 本片段转写文本
    start_sec: float  # 分片音频起始时间 (秒)
    end_sec: float  # 分片音频结束时间 (秒)
    is_last: bool  # 是否为最后一个分片
    skipped_by_vad: bool = False  # 是否被 VAD 判定为静音并跳过
    full_text: str = ""  # 截至当前的累积全文 (仅 is_last=True 时填充完整值)
    # 性能
    encode_time: float = 0.0
    decode_time: float = 0.0
    prefill_time: float = 0.0
