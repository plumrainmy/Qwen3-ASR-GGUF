# coding=utf-8
"""
vad.py — 语音活动检测 (VAD) 引擎封装

基于 FireRedVAD Non-Streaming 模式，为 ASR 流水线提供分片级静音前置过滤：
- 对每个音频分片执行 VAD 检测，判断是否含有效语音
- 跳过纯静音片段，避免 ASR 模型在无语音输入时产生幻觉并浪费算力
- 接口设计与 asr.py / aligner.py 保持一致，便于统一管理
- 使用 (int16_array, sample_rate) 元组直接传递音频，无临时文件 I/O

使用示例:
    config = VADConfig(model_dir="models/FireRedVAD/VAD")
    vad = QwenVADEngine(config)

    audio = np.zeros(16000 * 30, dtype=np.float32)   # 30s 静音 (float32 PCM)
    result = vad.detect(audio)
    print(result.has_speech)   # False
"""

import os
import time
import numpy as np
from typing import Optional, List, Tuple

from .schema import VADConfig, VADResult
from core.logger import logger


class QwenVADEngine:
    """
    FireRedVAD 非流式语音活动检测引擎封装。

    职责:
      - 加载 FireRedVAD 模型
      - 将 numpy 音频片段写入临时 WAV 文件供 FireRedVAD 使用
      - 解析检测结果，返回标准化的 VADResult 数据类

    线程安全: 每次 detect() 调用使用独立临时文件，可安全用于多线程场景。
    """

    def __init__(self, config: VADConfig):
        self.config = config
        self._vad = None
        self._last_probs = None  # 缓存最近一次 detect() 的帧级概率，供自适应阈值使用
        self._load_model()

    # ──────────────────────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        """加载 FireRedVAD 模型（延迟导入，避免无 fireredvad 时整包不可用）"""
        try:
            from fireredvad import FireRedVad, FireRedVadConfig
        except ImportError as exc:
            raise ImportError(
                "fireredvad 未安装，请执行: pip install fireredvad\n"
                "或参考 https://github.com/FireRedTeam/FireRedVAD 手动安装"
            ) from exc

        cfg = self.config
        vad_cfg = FireRedVadConfig(
            use_gpu=cfg.use_gpu,
            smooth_window_size=cfg.smooth_window_size,
            speech_threshold=cfg.speech_threshold,
            min_speech_frame=cfg.min_speech_frame,
            max_speech_frame=cfg.max_speech_frame,
            min_silence_frame=cfg.min_silence_frame,
            merge_silence_frame=cfg.merge_silence_frame,
            extend_speech_frame=cfg.extend_speech_frame,
            chunk_max_frame=cfg.chunk_max_frame,
        )

        logger.info(f"[VAD] 正在加载 FireRedVAD 模型: {cfg.model_dir}")
        t0 = time.time()
        self._vad = FireRedVad.from_pretrained(cfg.model_dir, vad_cfg)
        logger.info(
            f"[VAD] 模型加载完成，耗时 {time.time() - t0:.2f}s "
            f"(GPU: {cfg.use_gpu}, threshold: {cfg.speech_threshold})"
        )

    # ──────────────────────────────────────────────────────────────────
    # 核心检测接口
    # ──────────────────────────────────────────────────────────────────

    def detect(self, audio: np.ndarray, sr: int = 16000) -> VADResult:
        """
        对 numpy 音频数组执行语音活动检测。

        Args:
            audio: 单声道 float32 PCM 音频，采样率需与 sr 参数一致。
            sr:    采样率 (Hz)，默认 16000。

        Returns:
            VADResult: 包含 has_speech / timestamps / duration / detect_time。

        注意:
            FireRedVAD 原生接受文件路径，此方法会将 numpy 数组写入系统临时目录
            的 WAV 文件，检测完成后立即删除，不留残留文件。
        """
        if self._vad is None:
            raise RuntimeError("[VAD] 引擎未初始化，请先调用 _load_model()")

        audio_dur = len(audio) / sr
        t0 = time.time()

        # FireRedVAD 的 detect() 支持 (wav_np, sample_rate) 元组接口，
        # 完全避免临时文件 I/O，与文件路径接口结果完全一致。
        # 内部 AudioFeat 以 int16 范围处理波形，故需从 float32 PCM (-1..1) 转换。
        audio_in = np.asarray(audio, dtype=np.float32)
        if audio_in.ndim > 1:
            audio_in = audio_in.mean(axis=1)
        audio_int16 = (audio_in * 32767.0).clip(-32768, 32767).astype(np.int16)

        raw_result, _probs = self._vad.detect((audio_int16, sr))
        self._last_probs = _probs  # 缓存帧级概率，供 adaptive_detect() 使用

        detect_time = time.time() - t0

        # 解析 FireRedVAD 返回格式:
        # {'dur': float, 'timestamps': [(start_sec, end_sec), ...], 'wav_path': str}
        timestamps: List[Tuple[float, float]] = raw_result.get("timestamps", [])
        reported_dur: float = raw_result.get("dur", audio_dur)

        result = VADResult(
            has_speech=len(timestamps) > 0,
            timestamps=timestamps,
            duration=reported_dur,
            detect_time=detect_time,
        )

        if result.has_speech:
            segs = ", ".join(f"[{s:.2f}s~{e:.2f}s]" for s, e in timestamps)
            logger.debug(
                f"[VAD] 检测到语音 | 时长={reported_dur:.2f}s "
                f"| 语音比={result.speech_ratio:.1%} | 片段: {segs} "
                f"| 耗时={detect_time:.3f}s"
            )
        else:
            logger.debug(
                f"[VAD] 未检测到语音 | 时长={reported_dur:.2f}s "
                f"| 耗时={detect_time:.3f}s"
            )

        return result

    def has_speech(self, audio: np.ndarray, sr: int = 16000) -> bool:
        """
        快速判断音频分片是否含有语音（仅返回布尔值，忽略时间戳细节）。

        等价于 ``self.detect(audio, sr).has_speech``，适合在 ASR 热路径中
        进行简单的通过 / 跳过判断。
        """
        return self.detect(audio, sr).has_speech

    # ──────────────────────────────────────────────────────────────────
    # 自适应阈值检测
    # ──────────────────────────────────────────────────────────────────

    def adaptive_detect(self, audio: np.ndarray, sr: int = 16000) -> VADResult:
        """
        自适应阈值 VAD 检测（两遍法）。

        算法流程：
          1. 以配置阈值执行首次检测，获取帧级语音概率分布
          2. 分析概率分布，取高于噪声底的帧概率的 30% 分位数作为自适应阈值
          3. 若自适应阈值与初始值差异显著，用新阈值对帧概率重新分割语音段
          4. 否则直接返回首次检测结果

        适用场景：长音频离线转写，可更精确地适配不同录音环境的信噪比。
        """
        # 第一遍：标准检测
        result = self.detect(audio, sr)
        probs = self._last_probs

        # 无概率数据时直接返回
        if probs is None:
            return result

        try:
            probs_arr = np.asarray(probs, dtype=np.float32).flatten()
        except (ValueError, TypeError):
            return result

        if len(probs_arr) == 0:
            return result

        # 仅考虑高于噪声底 (>0.1) 的帧概率，避免静音帧拉低分位数
        speech_probs = probs_arr[probs_arr > 0.1]
        if len(speech_probs) == 0:
            return result

        initial_threshold = self.config.speech_threshold
        # 取 30% 分位数，限制在 [0.25, 0.65] 安全区间
        adaptive_threshold = float(np.clip(np.percentile(speech_probs, 30), 0.25, 0.65))

        # 阈值变化不足 0.05 → 无需重新分割
        if abs(adaptive_threshold - initial_threshold) < 0.05:
            logger.debug(
                f"[VAD] 自适应阈值与初始值接近 "
                f"({adaptive_threshold:.3f} vs {initial_threshold:.3f})，保持原结果"
            )
            return result

        logger.debug(
            f"[VAD] 自适应阈值调整: {initial_threshold:.3f} -> {adaptive_threshold:.3f}"
        )

        # 第二遍：用自适应阈值对帧概率重新分割
        audio_dur = len(audio) / sr
        timestamps = self._probs_to_timestamps(probs_arr, adaptive_threshold, audio_dur)

        if not timestamps:
            # 自适应分割无结果 → 保留首次结果
            return result

        return VADResult(
            has_speech=True,
            timestamps=timestamps,
            duration=audio_dur,
            detect_time=result.detect_time,
        )

    def _probs_to_timestamps(
        self,
        probs: np.ndarray,
        threshold: float,
        audio_duration: float,
    ) -> List[Tuple[float, float]]:
        """
        基于帧级概率和阈值重新生成语音时间戳。

        实现逻辑：
          1. 滑动窗口平滑帧级概率，消除毛刺
          2. 应用阈值判定每帧是否为语音
          3. 连续语音帧合并为片段
          4. 过滤过短片段、合并近邻片段、扩展语音边界
        """
        if len(probs) == 0:
            return []

        frame_dur = audio_duration / len(probs)

        # 平滑概率
        window = max(1, self.config.smooth_window_size)
        if window > 1 and len(probs) > window:
            kernel = np.ones(window, dtype=np.float32) / window
            smoothed = np.convolve(probs, kernel, mode="same")
        else:
            smoothed = probs

        # 应用阈值 → 二值语音掩码
        is_speech = smoothed >= threshold

        # 连续语音帧 → 时间段
        segments: List[Tuple[float, float]] = []
        in_speech = False
        start_idx = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                start_idx = i
                in_speech = True
            elif not is_speech[i] and in_speech:
                segments.append((start_idx * frame_dur, i * frame_dur))
                in_speech = False

        if in_speech:
            segments.append((start_idx * frame_dur, len(probs) * frame_dur))

        # 过滤过短片段 (min_speech_frame × frame_dur)
        min_speech_sec = self.config.min_speech_frame * frame_dur
        segments = [(s, e) for s, e in segments if e - s >= min_speech_sec]

        # 合并近邻片段 (min_silence_frame × frame_dur)
        min_silence_sec = self.config.min_silence_frame * frame_dur
        merged: List[Tuple[float, float]] = []
        for s, e in segments:
            if merged and s - merged[-1][1] < min_silence_sec:
                merged.append((merged[-1][0], max(merged[-1][1], e)))
                merged.pop(-2)
            else:
                merged.append((s, e))

        # 扩展语音边界 (extend_speech_frame × frame_dur)
        extend_sec = self.config.extend_speech_frame * frame_dur
        if extend_sec > 0:
            merged = [
                (max(0.0, s - extend_sec), min(audio_duration, e + extend_sec))
                for s, e in merged
            ]

        return merged

    # ──────────────────────────────────────────────────────────────────
    # 高级工具方法
    # ──────────────────────────────────────────────────────────────────

    def build_chunks(
        self,
        timestamps: List[Tuple[float, float]],
        total_dur: float,
        max_span_sec: float = 30.0,
        merge_gap_sec: float = 1.0,
        context_pre_sec: float = 0.2,
        context_post_sec: float = 0.3,
    ) -> list:
        """
        根据 VAD 时间戳构建对齐语音边界的音频分片列表（VADChunk）。

        算法：
          1. 合并间隔 < merge_gap_sec 的相邻语音段
          2. 贪心打包：在不超过 max_span_sec 的前提下，将连续语音段组合为一个
             分片；每个分片在首段前补 context_pre_sec、末段后补 context_post_sec
          3. 在语音分片之间插入静音分片，使输出完整覆盖 0 ~ total_dur 全域

        Args:
            timestamps:       VAD 检测到的语音区间列表 [(start, end), ...]
            total_dur:        音频总时长（秒）
            max_span_sec:     单个分片的最大时间跨度（秒），默认 30s
            merge_gap_sec:    小于此间隔的相邻段自动合并（秒），默认 0.5s
            context_pre_sec:  每个分片首段前的预留缓冲（秒），默认 0.2s
            context_post_sec: 每个分片末段后的预留缓冲（秒），默认 0.3s

        Returns:
            List[VADChunk]，按时间顺序排列；
            has_speech=False 的分片代表静音区，用于进度上报和跳过 ASR。
        """
        from .schema import VADChunk

        if not timestamps:
            return [VADChunk(idx=0, start_sec=0.0, end_sec=total_dur, has_speech=False)]

        # Step 1: 合并近邻语音段
        merged: List[List[float]] = []
        for s, e in sorted(timestamps):
            if merged and s - merged[-1][1] < merge_gap_sec:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        # Step 2: 贪心打包 → speech_chunks = [(chunk_start, chunk_end, segs), ...]
        speech_chunks: list = []
        chunk_segs: List[Tuple[float, float]] = []
        chunk_start_sec: Optional[float] = None

        for raw_s, raw_e in merged:
            c_start = max(0.0, raw_s - context_pre_sec)
            c_end = min(total_dur, raw_e + context_post_sec)

            if chunk_start_sec is None:
                chunk_start_sec = c_start
                chunk_segs = [(raw_s, raw_e)]
            else:
                # 加入当前段后是否超限？
                if c_end - chunk_start_sec > max_span_sec:
                    # Flush 当前分片
                    last_end = min(total_dur, chunk_segs[-1][1] + context_post_sec)
                    speech_chunks.append((chunk_start_sec, last_end, list(chunk_segs)))
                    chunk_start_sec = c_start
                    chunk_segs = [(raw_s, raw_e)]
                else:
                    chunk_segs.append((raw_s, raw_e))

        if chunk_segs:
            last_end = min(total_dur, chunk_segs[-1][1] + context_post_sec)
            speech_chunks.append((chunk_start_sec, last_end, list(chunk_segs)))

        # Step 3: 插入静音分片，完整覆盖 [0, total_dur]
        result: list = []
        cursor = 0.0

        for cs, ce, segs in speech_chunks:
            # 前置静音（间隔 > 0.5s 才记录，避免极短噪声分片）
            if cs > cursor + 0.5:
                result.append(
                    VADChunk(
                        idx=len(result),
                        start_sec=cursor,
                        end_sec=cs,
                        has_speech=False,
                    )
                )
            result.append(
                VADChunk(
                    idx=len(result),
                    start_sec=cs,
                    end_sec=ce,
                    has_speech=True,
                    speech_segments=segs,
                )
            )
            cursor = ce

        # 尾部静音
        if cursor < total_dur - 0.5:
            result.append(
                VADChunk(
                    idx=len(result),
                    start_sec=cursor,
                    end_sec=total_dur,
                    has_speech=False,
                )
            )

        return result

    # ──────────────────────────────────────────────────────────────────

    def get_speech_segments(
        self,
        audio: np.ndarray,
        sr: int = 16000,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        提取音频中所有语音片段的 numpy 子数组。

        Returns:
            List of (segment_audio, start_sec, end_sec) 三元组，
            按时间顺序排列。若无语音则返回空列表。
        """
        result = self.detect(audio, sr)
        if not result.has_speech:
            return []

        segments = []
        for start_sec, end_sec in result.timestamps:
            s = int(start_sec * sr)
            e = int(end_sec * sr)
            s = max(0, s)
            e = min(len(audio), e)
            if e > s:
                segments.append((audio[s:e], start_sec, end_sec))

        return segments

    def should_run_vad(self, chunk_duration: float) -> bool:
        """
        判断当前音频分片时长是否达到启用 VAD 过滤的阈值。

        Args:
            chunk_duration: 分片时长 (秒)。

        Returns:
            True 表示应当执行 VAD；False 表示分片过短，直接送入 ASR 即可。
        """
        return chunk_duration >= self.config.vad_min_duration

    # ──────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────

    def shutdown(self):
        """释放引擎资源（当前 FireRedVAD 无显式释放接口，预留扩展用）"""
        self._vad = None
        logger.info("[VAD] 引擎已关闭")

    def __repr__(self) -> str:
        status = "ready" if self._vad is not None else "unloaded"
        return (
            f"QwenVADEngine("
            f"model='{self.config.model_dir}', "
            f"threshold={self.config.speech_threshold}, "
            f"gpu={self.config.use_gpu}, "
            f"status={status})"
        )
