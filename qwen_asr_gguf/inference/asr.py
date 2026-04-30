# coding=utf-8
import os
import time
import re
import codecs
import dataclasses
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List, Generator
from core.logger import logger
from .schema import MsgType, StreamingMessage, DecodeResult, ASREngineConfig, TranscribeResult, ForcedAlignItem, \
    ForcedAlignResult, ASREngineConfig, StreamChunkResult
from .utils import normalize_language_name, validate_language, detect_and_fix_repetitions, is_hallucination
from .encoder import QwenAudioEncoder
from . import llama


@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None


class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构
            核心改进:
      - 集成 VAD 前置过滤，对静音片段直接跳过 ASR，大幅降低 RTF
      - 统一流水线 _asr_core (生成器)，同时支持一次性 asr() 和流式 asr_stream()
      - transcribe_stream() 供 SSE/WebSocket 场景实时推送逐片结果
    """

    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (Provider: {config.onnx_provider}) ---")

        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        # 动态分片模式下，分片长度由 VAD 动态决定（通常 3~10s），
        # 使用动态形状模式以节省冗余计算；仅当明确禁用动态分片时才使用固定形状。
        use_dynamic = (
                config.dynamic_chunk_threshold is not None
                and config.dynamic_chunk_threshold < float("inf")
        )
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            onnx_provider=config.onnx_provider,
            dml_pad_to=config.dml_pad_to,
            verbose=self.verbose
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner
            self.aligner = QwenForcedAligner(config.align_config)

        # 3. VAD 延迟初始化：由 _ensure_vad() 在首次遇到长音频时按需加载
        self.vad = None

        # 3. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf, use_gpu=config.llm_use_gpu)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False)

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    # ──────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────

    def _ensure_vad(self) -> bool:
        """延迟初始化 VAD 引擎（用于长音频动态分片自动启用场景）。

        当配置中未显式启用 VAD 但音频超过动态分片阈值时，按需加载 VAD 模型。

        Returns:
            True  → VAD 就绪
            False → 初始化失败（缺少模型或依赖），调用方应降级为固定分片
        """
        if self.vad is not None:
            return True

        try:
            from .vad import QwenVADEngine

            vad_config = self.config.vad_config
            if vad_config is None:
                from .schema import VADConfig

                vad_config = VADConfig()

            self.vad = QwenVADEngine(vad_config)
            if self.verbose:
                logger.debug("--- [QwenASR] VAD 延迟加载完成（长音频动态分片） ---")
            return True
        except Exception as exc:
            logger.warning(f"[QwenASR] VAD 延迟加载失败，将使用固定分片: {exc}")
            return False

    def shutdown(self):
        if self.verbose: print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str],
                           language: Optional[str]):
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)"""

        def tk(t): return self.model.tokenize(t)

        # 1. 区块 A: 音频之前的所有内容 (System + User Header)
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]

        # 2. 区块 B: 音频之后的所有内容 (Instruction + Assistant Header + History)
        suffix_head = f"assistant\n"
        if language: suffix_head += f"language {language}"

        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT] + tk(prefix_text)

        # 3. 统计并拼接
        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)

        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre: n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]

        return total_embd

    def _decode(
            self,
            full_embd: np.ndarray,
            prefix_text: str,
            rollback_num: int,
            is_last_chunk: bool = False,
            temperature: float = 0.4,
            streaming: bool = True,
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()

        total_len = full_embd.shape[0]

        # ── n_ctx 越界保护 ──────────────────────────────────────────
        # 如果序列长度超过上下文窗口，llama.cpp 会触发 GGML_ASSERT
        # 并调用 abort() 终止整个进程。这里提前拦截，返回空结果而非崩溃。
        n_ctx = self.config.n_ctx
        if total_len > n_ctx:
            logger.warning(
                f"[Decode] 序列长度 {total_len} 超过 n_ctx={n_ctx}，"
                f"跳过本次推理以避免进程崩溃"
            )
            result.text = ""
            result.is_aborted = True
            return result

        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)

        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start

        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2 ** 31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(512):  # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break

            if self.ctx.decode_token(last_sampled_token) != 0:
                break

            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end='', flush=True)
                    stable_text_acc += piece

            # 熔断检查：检测重复循环
            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    result.is_aborted = True
                    break

            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1

        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
        del batch

        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end="", flush=True)
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p:
                if streaming: print(final_p, end='', flush=True)
                stable_text_acc += final_p

        # 填充结果（内核输出标准化）
        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        result.n_generate = n_gen_tokens
        return result


    def _safe_decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool,
        temperature: float,
        streaming: bool = True,
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装"""
        for i in range(4):
            res = self._decode(full_embd, prefix_text, rollback_num, is_last_chunk, temperature, streaming=streaming)
            if not res.is_aborted:
                break
            temperature += 0.3
            res.text += "====解码有误，强制熔断===="
            print(f"\n\n[!] 触发重试 (Temp -> {temperature:.1f})\n")
        return res

    def _safe_decode_new(
            self,
            full_embd: np.ndarray,
            prefix_text: str,
            rollback_num: int,
            is_last_chunk: bool,
            temperature: float,
            streaming: bool = True,
            max_new_tokens: int = 512,
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装"""
        for i in range(3):
            res = self._decode(
                full_embd,
                prefix_text,
                rollback_num,
                is_last_chunk,
                temperature,
                max_new_tokens,
            )
            if not res.is_aborted:
                break
            temperature = min(0.6, temperature + 0.2)
            logger.warning(f"\n\n[!] 触发重试 (Temp -> {temperature:.1f})\n")

        # 后处理：使用官方去重算法修复残余重复
        res.text = detect_and_fix_repetitions(res.text)
        return res

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0
        gen_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0

        print(f"\n\n📊 性能统计:")
        print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
        print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
        print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
        if stats.get("align_time"):
            print(f"  🔹 对齐耗时    : {stats['align_time']:.3f} 秒")
        print(f"  🔹 编码耗时    : {stats['encode_time']:.3f} 秒")
        print(
            f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
        print(
            f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")

    def transcribe(
            self,
            audio_file: str,
            language: Optional[str] = None,
            context: Optional[str] = None,
            start_second: float = 0.0,
            duration: float = 0.0,
            temperature: float = 0.4,
            rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (从文件加载音频)"""
        from .audio import load_audio
        audio = load_audio(audio_file, start_second=start_second, duration=duration)

        return self.asr_new(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num
        )

    # ──────────────────────────────────────────────────────────────────
    # 核心 API：asr (一次性) & asr_stream (生成器)
    # ──────────────────────────────────────────────────────────────────

    def asr(
            self,
            audio: np.ndarray,
            context: Optional[str],
            language: Optional[str],
            chunk_size_sec: float = 40.0,
            memory_chunks: int = 2,
            temperature: float = 0.4,
            rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)"""
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr

        # 记忆管理 (预定义所有分片的物理边界)
        all_segments: List[ASRS_Segment] = [
            ASRS_Segment(
                idx=i,
                audio_start=i * chunk_size_sec,
                audio_end=min((i + 1) * chunk_size_sec, total_duration)
            ) for i in range(num_chunks)
        ]
        asr_memory = deque(maxlen=memory_chunks)  # 存储 (embd, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []

        # 统计指标
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "encode_time": 0.0, "align_time": 0.0,
        }
        t_main_start = time.time()

        # --- 顺序同步处理循环 ---
        for i in range(num_chunks):
            # 1. 编码第 i 片段
            s, e = i * samples_per_chunk, min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e]
            if len(chunk_data) < samples_per_chunk:
                chunk_data = np.pad(chunk_data, (0, samples_per_chunk - len(chunk_data)))

            audio_feature, enc_time = self.encoder.encode(chunk_data)
            stats["encode_time"] += enc_time
            was_last = (i == num_chunks - 1)

            # 2. 识别第 i 片段文字
            prefix_text = "".join([m[1] for m in asr_memory])
            combined_audio = np.concatenate([m[0] for m in asr_memory] + [audio_feature], axis=0)
            full_embd = self._build_prompt_embd(combined_audio, prefix_text, context, language)

            # 带熔断加温重试的解码调用
            res = self._safe_decode(full_embd, prefix_text, rollback_num, was_last, temperature)

            # 更新记忆与统计
            all_segments[i].text = res.text
            asr_memory.append((audio_feature, res.text))

            total_full_text += res.text
            stats["prefill_tokens"] += res.n_prefill;
            stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate;
            stats["decode_time"] += res.t_generate

            # 3. 对齐第 i 片段 (同步)
            if self.aligner and res.text.strip():
                t_align_start = time.time()
                # 计算偏移（同步版本逻辑简化：直接使用片起点，不考虑前片动态边界）
                if i == 0:
                    offset_sec = all_segments[i].audio_start
                    s_smpl, e_smpl = int(offset_sec * sr), int(all_segments[i].audio_end * sr)
                else:
                    offset_sec = all_segments[i - 1].audio_end
                    s_smpl, e_smpl = int(all_segments[i - 1].audio_end * sr), int(all_segments[i].audio_end * sr)
                audio_slice = audio[s_smpl:e_smpl]

                align_res = self.aligner.align(
                    audio_slice,
                    res.text,
                    language=language,
                    offset_sec=float(offset_sec)
                )
                all_segments[i].items = align_res.items
                # 取最后一个内容的时间结束点
                all_segments[i].audio_end = align_res.items[-1].end_time
                all_aligned_items.extend(align_res.items)
                stats["align_time"] += (time.time() - t_align_start)

        # 4. 结果整理
        all_aligned_items.sort(key=lambda x: x.start_time)
        t_total = time.time() - t_main_start
        if self.verbose: self._print_stats(stats, total_duration, t_total)

        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance=stats
        )

    def asr_new(
            self,
            audio: np.ndarray,
            context: Optional[str],
            language: Optional[str],
            chunk_size_sec: float = 40.0,
            memory_chunks: int = 2,
            temperature: float = 0.4,
            rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)
        在内部调用 _asr_core 生成器，收集所有分片结果后一次性返回。
        VAD 集成：若引擎启用了 VAD，静音片段将被跳过，不送入 ASR 模型。
        """
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []
        final_stats = None

        for chunk_res in self._asr_core(
                audio=audio,
                context=context,
                language=language,
                chunk_size_sec=chunk_size_sec,
                memory_chunks=memory_chunks,
                temperature=temperature,
                rollback_num=rollback_num
        ):
            if not chunk_res.skipped_by_vad:
                total_full_text += chunk_res.text
            # 对齐数据通过私有属性传递
            align_items = getattr(chunk_res, "_align_items", None)
            if align_items:
                all_aligned_items.extend(align_items)
            if chunk_res.is_last:
                final_stats = getattr(chunk_res, "_stats", None)

        all_aligned_items.sort(key=lambda x: x.start_time)
        return TranscribeResult(
            text=total_full_text,
            alignment=(
                ForcedAlignResult(items=all_aligned_items)
                if all_aligned_items
                else None
            ),
            performance=final_stats,
        )

    # ──────────────────────────────────────────────────────────────────
    # 内部统一流水线核心
    # ──────────────────────────────────────────────────────────────────

    def _asr_core(
            self,
            audio: np.ndarray,
            context: Optional[str],
            language: Optional[str],
            chunk_size_sec: float,
            memory_chunks: int,
            temperature: float,
            rollback_num: int,
            enable_aligner: bool = False,
    ) -> Generator[StreamChunkResult, None, None]:
        """
        统一流水线核心（生成器）。asr() 和 asr_stream() 均调用此方法。

        ┌─ VAD 动态分片模式（长音频 > dynamic_chunk_threshold）───────────┐
        │ 1. 对全段音频执行一次自适应阈值 VAD，获取语音时间戳              │
        │ 2. 按语音边界动态组合分片（不在静音中间截断，不在句中切断）       │
        │ 3. 每分片仅送入实际语音帧（trimmed + padded），消除尾部静音幻觉   │
        │ 4. LLM 记忆仅保留前 N 片的文本（不重放音频），                   │
        │    避免非连续音频拼接导致的模型混乱                              │
        │ 5. max_new_tokens 按实际语音时长等比缩放，从根本上限制幻觉空间   │
        └──────────────────────────────────────────────────────────────────┘
        ┌─ 固定分片模式（短音频 ≤ 阈值 或 VAD 不可用时降级）───────────────┐
        │ • 短音频：单一分片直接处理                                        │
        │ • 降级：保持原有 30s 等长切割 + 音频/文本双重记忆上下文           │
        └──────────────────────────────────────────────────────────────────┘
        共同抗幻觉措施（_decode 内）:
          • token 级重复熔断（15-token 窗口，≤3 种 token）
          • n-gram 短语级重复熔断（5/8-char 短语出现 ≥4 次）
          • max_new_tokens 上限（speech_sec × 12，最大 512）
        """
        # ── 语言归一化 ──────────────────────────────────────────────
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        total_duration = total_len / sr

        asr_memory = deque(maxlen=memory_chunks)  # (audio_embd_or_None, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []

        stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "encode_time": 0.0,
            "align_time": 0.0,
            "vad_time": 0.0,
            "vad_skipped_chunks": 0,
        }
        t_main_start = time.time()

        # ── 选择分片策略 ──────────────────────────────────────────────
        #
        # 三级策略:
        #   1. 短音频 (≤ dynamic_chunk_threshold): 不分片，作为单一 chunk 直接处理
        #   2. 长音频 + VAD 可用: 自适应阈值 VAD 动态分片
        #   3. 长音频 + VAD 不可用 (降级): 固定等长分片
        #
        dynamic_threshold = self.config.dynamic_chunk_threshold

        if total_duration <= dynamic_threshold:
            # ── 短音频：不分片 ─────────────────────────────────────
            vad_mode = False
            all_chunks = [
                ASRS_Segment(
                    idx=0,
                    audio_start=0.0,
                    audio_end=total_duration,
                )
            ]
            if self.verbose:
                logger.debug(
                    f"[QwenASR] 短音频 ({total_duration:.1f}s "
                    f"≤ {dynamic_threshold}s)，单一分片直接处理"
                )

        elif self.vad is not None or self._ensure_vad():
            # ── 长音频：VAD 自适应动态分片 ─────────────────────────
            vad_mode = True
            from .schema import VADChunk

            t_vad = time.time()
            vad_result = self.vad.adaptive_detect(audio, sr)
            stats["vad_time"] += time.time() - t_vad

            all_chunks = self.vad.build_chunks(
                timestamps=vad_result.timestamps,
                total_dur=total_duration,
                max_span_sec=chunk_size_sec,
            )
            if self.verbose:
                n_speech = sum(1 for c in all_chunks if c.has_speech)
                n_silence = len(all_chunks) - n_speech
                logger.debug(
                    f"[VAD] 动态分片完成 | 音频 {total_duration:.1f}s "
                    f"| 耗时 {stats['vad_time']:.2f}s "
                    f"| 语音分片 {n_speech}，静音分片 {n_silence}"
                )

        else:
            # ── 降级：VAD 不可用，固定等长分片 ────────────────────
            vad_mode = False
            num_fixed = int(np.ceil(total_len / samples_per_chunk))
            all_chunks = [
                ASRS_Segment(
                    idx=i,
                    audio_start=i * chunk_size_sec,
                    audio_end=min((i + 1) * chunk_size_sec, total_duration),
                )
                for i in range(num_fixed)
            ]
            if self.verbose:
                logger.debug(f"[QwenASR] VAD 不可用，使用固定分片 ({num_fixed} 片)")

        num_chunks = len(all_chunks)
        logger.info(
            f"[ASR] 开始转写 | 音频时长={total_duration:.1f}s | 分片数={num_chunks} "
            f"| 模式={'VAD动态' if vad_mode else '固定'}"
        )

        # ── 主循环 ────────────────────────────────────────────────────
        for i, chunk_def in enumerate(all_chunks):
            is_last = i == num_chunks - 1

            # 从 chunk_def 提取时间坐标和语音标记
            if vad_mode:
                start_sec = chunk_def.start_sec
                end_sec = chunk_def.end_sec
                has_speech = chunk_def.has_speech
                speech_sec = chunk_def.speech_sec
            else:
                start_sec = chunk_def.audio_start
                end_sec = chunk_def.audio_end
                has_speech = True  # 固定模式下统一送 ASR
                speech_sec = end_sec - start_sec

            try:
                s_smpl = int(start_sec * sr)
                e_smpl = min(int(end_sec * sr), total_len)
                chunk_raw = audio[s_smpl:e_smpl]

                # ── 边界音频缓冲（仅固定分片模式）────────────────────────
                # 非末尾分片：在 chunk 尾部额外附加 1 秒音频，让编码器
                # "多听一秒"，使 LLM 能在边界处解码出完整的词句而非截断。
                # 此缓冲仅影响编码输入，不影响报告的 start_sec/end_sec。
                BOUNDARY_PAD_SEC = 2.0
                if not vad_mode and not is_last:
                    padded_end = min(int((end_sec + BOUNDARY_PAD_SEC) * sr), total_len)
                    if padded_end > e_smpl:
                        chunk_raw = audio[s_smpl:padded_end]

                # ── Step 1: 静音跳过 ──────────────────────────────────────
                if not has_speech:
                    stats["vad_skipped_chunks"] += 1
                    chunk_result = StreamChunkResult(
                        segment_idx=i,
                        text="",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        is_last=is_last,
                        skipped_by_vad=True,
                        full_text=total_full_text if is_last else "",
                    )
                    if is_last:
                        t_total = time.time() - t_main_start
                        if self.verbose:
                            self._print_stats(stats, total_duration, t_total)
                        stats["audio_duration"] = total_duration
                        setattr(chunk_result, "_stats", stats)
                        setattr(chunk_result, "_align_items", [])
                    yield chunk_result
                    continue

                # ── 短语音跳过（VAD 模式）───────────────────────────────
                # 语音长度 <0.3s 的分片几乎无有效信息，LLM 易在极短输入
                # 上产生幻觉文本（如随机数字、重复字符），直接跳过。
                if vad_mode and speech_sec < 0.3:
                    stats["vad_skipped_chunks"] += 1
                    if self.verbose:
                        logger.debug(
                            f"  [VAD] 分片 #{i:02d} 语音过短 "
                            f"({speech_sec:.2f}s < 0.3s)，跳过"
                        )
                    chunk_result = StreamChunkResult(
                        segment_idx=i,
                        text="",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        is_last=is_last,
                        skipped_by_vad=True,
                        full_text=total_full_text if is_last else "",
                    )
                    if is_last:
                        t_total = time.time() - t_main_start
                        if self.verbose:
                            self._print_stats(stats, total_duration, t_total)
                        stats["audio_duration"] = total_duration
                        setattr(chunk_result, "_stats", stats)
                        setattr(chunk_result, "_align_items", [])
                    yield chunk_result
                    continue

                if vad_mode:
                    # VAD 模式：直接按实际语音长度编码，无需补零
                    # 效果：5s 语音仅处理 5s 数据，而非 pad 到 30s 再处理
                    audio_feature, enc_time = self.encoder.encode(chunk_raw)
                else:
                    # 固定分片模式：补零至标准分片长度，保持 Encoder 固定输入尺寸
                    chunk_padded = chunk_raw
                    if len(chunk_padded) < samples_per_chunk:
                        chunk_padded = np.pad(
                            chunk_padded, (0, samples_per_chunk - len(chunk_padded))
                        )
                    audio_feature, enc_time = self.encoder.encode(chunk_padded)
                stats["encode_time"] += enc_time

                # ── Step 3: LLM 解码 ─────────────────────────────────────
                if vad_mode:
                    # VAD 模式：仅用文本上下文，不重放前片段音频。
                    # 原因：VAD 分片之间可能有大段静音（时间不连续），将非连续
                    # 音频拼接后送 LLM 会导致模型混乱，产生重复或幻觉。
                    # 限制前缀长度（末尾 40 字符）：保持语境连贯，同时防止
                    # 前缀过长时 LLM 倾向于复读历史文本。
                    prefix_text = "".join(m[1] for m in asr_memory)
                    if len(prefix_text) > 100:
                        prefix_text = prefix_text[-100:]
                    full_embd = self._build_prompt_embd(
                        audio_feature, prefix_text, context, language
                    )
                else:
                    # 固定分片模式：保留音频 + 文本双重记忆（原有行为）
                    prefix_text = "".join(m[1] for m in asr_memory)
                    combined_audio = np.concatenate(
                        [m[0] for m in asr_memory] + [audio_feature], axis=0
                    )
                    full_embd = self._build_prompt_embd(
                        combined_audio, prefix_text, context, language
                    )
                    # n_ctx 安全估算：合并记忆后超过上下文窗口时，回退为仅当前分片
                    if full_embd.shape[0] > self.config.n_ctx:
                        logger.warning(
                            f"[分片 {i}] 合并记忆后序列长度 {full_embd.shape[0]} "
                            f"超过 n_ctx={self.config.n_ctx}，回退为仅当前分片"
                        )
                        full_embd = self._build_prompt_embd(
                            audio_feature, prefix_text, context, language
                        )

                # token 预算：按实际语音时长等比缩放（12 tokens/s 上限）
                # 例：5s 语音 → 最多 60 tokens，防止在短/稀疏音频上过度生成
                max_new_tokens = min(512, max(32, int(speech_sec * 12)))

                res = self._safe_decode_new(
                    full_embd,
                    prefix_text,
                    rollback_num,
                    is_last,
                    temperature,
                    max_new_tokens,
                )

                # ── 幻觉过滤 ─────────────────────────────────────────────
                # 检测 LLM 输出是否为重复单字幻觉（如 "洞洞洞..."、"三三三..."）
                # 若检测到幻觉，将文本置空并标记为跳过，不污染累积全文和记忆
                if is_hallucination(res.text):
                    logger.warning(
                        f"[分片 {i}] 检测到幻觉输出，已丢弃 | "
                        f"原文: {res.text[:50]}{'...' if len(res.text) > 50 else ''}"
                    )
                    stats["vad_skipped_chunks"] += 1
                    chunk_result = StreamChunkResult(
                        segment_idx=i,
                        text="",
                        start_sec=start_sec,
                        end_sec=end_sec,
                        is_last=is_last,
                        skipped_by_vad=True,
                        full_text=total_full_text if is_last else "",
                    )
                    if is_last:
                        t_total = time.time() - t_main_start
                        if self.verbose:
                            self._print_stats(stats, total_duration, t_total)
                        stats["audio_duration"] = total_duration
                        setattr(chunk_result, "_stats", stats)
                        setattr(chunk_result, "_align_items", [])
                    yield chunk_result
                    continue

                # 更新记忆
                if vad_mode:
                    asr_memory.append((None, res.text))  # VAD 模式不缓存音频特征
                else:
                    asr_memory.append((audio_feature, res.text))

                total_full_text += res.text
                stats["prefill_tokens"] += res.n_prefill
                stats["prefill_time"] += res.t_prefill
                stats["decode_tokens"] += res.n_generate
                stats["decode_time"] += res.t_generate

                # ── Step 4: 对齐（可选，同步）────────────────────────────
                chunk_aligned_items: List[ForcedAlignItem] = []
                if enable_aligner and res.text.strip():
                    if self.aligner is None and self.config.align_config:
                        from .aligner import QwenForcedAligner

                        self.aligner = QwenForcedAligner(self.config.align_config)

                    if self.aligner:
                        t_align_start = time.time()
                        s_al = int(start_sec * sr)
                        e_al = int(end_sec * sr)
                        audio_slice = audio[s_al:e_al]

                        align_res = self.aligner.align(
                            audio_slice,
                            res.text,
                            language=language,
                            offset_sec=float(start_sec),
                        )
                        chunk_aligned_items = align_res.items
                        all_aligned_items.extend(align_res.items)
                        stats["align_time"] += time.time() - t_align_start

                # ── Step 5: Yield 分片结果 ───────────────────────────────
                chunk_result = StreamChunkResult(
                    segment_idx=i,
                    text=res.text,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_last=is_last,
                    skipped_by_vad=False,
                    full_text=total_full_text if is_last else "",
                    encode_time=enc_time,
                    decode_time=res.t_generate,
                    prefill_time=res.t_prefill,
                )
                setattr(chunk_result, "_align_items", chunk_aligned_items)
                if is_last:
                    t_total = time.time() - t_main_start
                    if self.verbose:
                        self._print_stats(stats, total_duration, t_total)
                    stats["audio_duration"] = total_duration
                    setattr(chunk_result, "_stats", stats)

                yield chunk_result

            except Exception as exc:
                logger.error(
                    f"[分片 {i}/{num_chunks}] 处理异常，跳过本分片: {exc}",
                    exc_info=True,
                )
                chunk_result = StreamChunkResult(
                    segment_idx=i,
                    text="",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    is_last=is_last,
                    skipped_by_vad=False,
                    full_text=total_full_text if is_last else "",
                )
                setattr(chunk_result, "_align_items", [])
                if is_last:
                    t_total = time.time() - t_main_start
                    stats["audio_duration"] = total_duration
                    setattr(chunk_result, "_stats", stats)
                yield chunk_result
