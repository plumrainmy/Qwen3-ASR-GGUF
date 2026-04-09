# coding=utf-8
"""
双音轨音频分离工具
将立体声(双声道)音频文件分离为左声道和右声道两个独立的单声道文件
"""
import os
import sys
from pathlib import Path
import subprocess


def check_ffmpeg():
    """检测系统是否安装 ffmpeg"""
    import shutil
    return shutil.which('ffmpeg') is not None


def split_stereo_audio(input_path, output_dir=None, format='wav', sample_rate=16000):
    """
    将双音轨音频文件分离为左右声道两个独立文件

    Args:
        input_path: 输入音频文件路径
        output_dir: 输出目录(可选,默认为输入文件所在目录)
        format: 输出格式,支持 wav/mp3/flac,默认wav
        sample_rate: 采样率,默认16000Hz

    Returns:
        dict: 包含左右声道输出文件路径 {'left': left_path, 'right': right_path}
    """
    if not check_ffmpeg():
        raise RuntimeError("系统未发现 ffmpeg。请先安装 ffmpeg 并将其添加到系统环境变量 PATH 中。")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_path}")

    # 设置输出目录
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 生成输出文件路径
    stem = input_path.stem
    left_path = output_dir / f"{stem}_left.{format}"
    right_path = output_dir / f"{stem}_right.{format}"

    print(f"正在分离双音轨音频...")
    print(f"输入文件: {input_path}")
    print(f"输出左声道: {left_path}")
    print(f"输出右声道: {right_path}")
    print(f"格式: {format}, 采样率: {sample_rate}Hz")

    # 构建编码器参数
    if format == 'mp3':
        codec = 'libmp3lame'
        bitrate_param = ['-b:a', '128k']
    elif format == 'flac':
        codec = 'flac'
        bitrate_param = []
    else:  # wav
        codec = 'pcm_s16le'
        bitrate_param = []

    try:
        # 提取左声道 (channel 0)
        print(f"\n[1/2] 提取左声道...")
        cmd_left = [
            'ffmpeg',
            '-y',
            '-i', str(input_path),
            '-af', 'pan=stereo|c0=c0|c1=0c0',  # 提取左声道
            '-ac', '2',
        ] + bitrate_param + [str(left_path)]

        subprocess.run(cmd_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✓ 左声道提取成功: {left_path}")

        # 提取右声道 (channel 1)
        print(f"\n[2/2] 提取右声道...")
        cmd_right = [
            'ffmpeg',
            '-y',
            '-i', str(input_path),
            '-ac', '2',
            '-af', 'pan=stereo|c0=0c1|c1=c1',  # 提取右声道
        ] + bitrate_param + [str(right_path)]

        subprocess.run(cmd_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"✓ 右声道提取成功: {right_path}")

        print(f"\n{'='*60}")
        print(f"✓ 双音轨分离完成!")
        print(f"左声道文件大小: {left_path.stat().st_size / 1024:.2f} KB")
        print(f"右声道文件大小: {right_path.stat().st_size / 1024:.2f} KB")
        print(f"{'='*60}")

        return {
            'left': str(left_path),
            'right': str(right_path)
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"ffmpeg 分离失败: {error_msg}")


if __name__ == '__main__':
    # 示例用法
    audio_file = '../wav/13623755.wav'  # 修改为你的音频文件路径

    if os.path.exists(audio_file):
        result = split_stereo_audio(audio_file, format='wav')
        print(f"\n输出结果:")
        print(result)
    else:
        print(f"文件不存在: {audio_file}")
        print("请修改 audio_file 变量为你的双音轨音频文件路径")
