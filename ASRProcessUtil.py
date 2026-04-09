# coding=utf-8
import os
import sys
import re
import time
from pathlib import Path
import subprocess

# 添加项目路径
sys.path.append(str(Path(__file__).parent.absolute()))

from qwen_asr_gguf.inference import QwenASREngine, itn, load_audio, ASREngineConfig, AlignerConfig
from qwen_asr_gguf.inference import exporters


def processASR(audio_path):
    context = "这是一个保险公司客服和客户之间的对话。关键词：美保、评残、小程序、再见、理赔、老师"

    # 配置引擎
    config = ASREngineConfig(
        model_dir="model",
        llm_use_gpu=True,
        enable_aligner=True,
        align_config=AlignerConfig(
            llm_use_gpu=True,
            model_dir="model",
        )
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
    processASR('./input.wav')
