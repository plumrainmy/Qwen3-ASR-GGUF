# coding=utf-8
import os
import sys
import re
import time
from pathlib import Path
import subprocess


def check_ffmpeg():
    """检测系统是否安装 ffmpeg"""
    import shutil
    return shutil.which('ffmpeg') is not None


def convert_to_mp3_16k(input_path, output_path=None, bitrate='128k'):
    """
    将音频文件转换为16k采样率的MP3格式

    Args:
        input_path: 输入音频文件路径
        output_path: 输出MP3文件路径(可选,默认为同名.mp3)
        bitrate: MP3比特率,默认128k

    Returns:
        str: 输出文件路径
    """
    if not check_ffmpeg():
        raise RuntimeError("系统未发现 ffmpeg。请先安装 ffmpeg 并将其添加到系统环境变量 PATH 中。")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_path}")

    # 生成输出路径
    if output_path is None:
        output_path = input_path.with_suffix('.mp3')
    else:
        output_path = Path(output_path)

    print(f"正在转换音频...")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"目标采样率: 16000 Hz")
    print(f"MP3比特率: {bitrate}")

    # 构建 ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i', str(input_path),  # 输入文件
        '-ar', '16000',  # 采样率 16kHz
        str(output_path)
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"\n✓ 转换成功!")
        print(f"输出文件: {output_path}")
        print(f"文件大小: {output_path.stat().st_size / 1024:.2f} KB")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"ffmpeg 转换失败: {error_msg}")


if __name__ == '__main__':
    convert_to_mp3_16k('../wav/13623461.wav')
