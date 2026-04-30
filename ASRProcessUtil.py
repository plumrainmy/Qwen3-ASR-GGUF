# coding=utf-8
import os
import sys
import re
import time
from pathlib import Path
import subprocess

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio, ASREngineConfig, AlignerConfig, VADConfig
from qwen_asr_gguf.inference import exporters

# VAD 模型路径（FireRedVAD 非流式版本）
# VAD 由 dynamic_chunk_threshold 自动控制：音频 > 阈值时延迟加载
VAD_MODEL_DIR = "models/FireRedVAD/VAD"


def processASR(audio_path):
    context = "这是一个保险公司客服和客户之间的对话。关键词：美保、评残、小程序、再见、理赔、老师、雇主险、啊、喂你好"

    """构造 ASR 引擎配置

    VAD 动态分片工作流（音频 > dynamic_chunk_threshold 时自动启用）：
      1. 在转写开始前对全段音频执行一次自适应阈值 VAD 检测
      2. 按语音边界动态划分分片（不在句中截断、不在静音中切割）
      3. 每分片的 token 预算随实际语音时长等比缩放，从根本上抑制幻觉
      4. LLM 上下文仅保留前片段文本（不重放音频），避免非连续音频拼接干扰
    """
    vad_cfg = VADConfig(
        model_dir=VAD_MODEL_DIR,
        use_gpu=True,  # VAD 模型较小，CPU 即可满足实时需求
        speech_threshold=0.35,  # 初始语音帧判定阈值（自适应算法会动态调整）
    )

    # 配置引擎
    config = ASREngineConfig(
        model_dir="model",
        llm_use_gpu=True,
        enable_aligner=True,
        align_config=AlignerConfig(
            llm_use_gpu=True,
            model_dir="model",
        ),
        vad_config=vad_cfg,
        dynamic_chunk_threshold=10.0,  # 音频 > 10s 
    )

    # 初始化引擎
    t0 = time.time()
    engine = QwenASREngine(config=config)
    print(f"--- [QwenASR] 引擎初始化耗时: {time.time() - t0:.2f} 秒 ---")

    # 执行转录
    res = engine.transcribe(
        audio_file=audio_path,
        context=context,
        language="Chinese",
        duration=None
    )

    # 导出文本（每行一句）
    txt_path = str(Path(audio_path).with_suffix('.txt'))
    exporters.export_to_txt(txt_path, res)

    # 导出 SRT（仅当有对齐时间戳时，才会有内容输出）
    srt_path = str(Path(audio_path).with_suffix('.srt'))
    exporters.export_to_srt(srt_path, res)

    # 导出 JSON（仅当有对齐时间戳时，才会有内容输出）
    json_path = str(Path(audio_path).with_suffix('.json'))
    exporters.export_to_json(json_path, res)

    # 优雅退出
    engine.shutdown()
    return res


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    processASR('wav/13623461.wav')
