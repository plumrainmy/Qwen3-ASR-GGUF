# coding=utf-8
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class TimeInterval:
    """时间区间"""
    start: float  # 开始时间(秒)
    end: float  # 结束时间(秒)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start:.3f}s - {self.end:.3f}s] ({self.duration:.3f}s)"


@dataclass
class LabeledTimeInterval(TimeInterval):
    """带标签的时间区间"""
    label: str = ""  # 标签,如"客服"或"客户"

    def __str__(self) -> str:
        if self.label:
            return f"[{self.start:.3f}s - {self.end:.3f}s] ({self.duration:.3f}s) - {self.label}"
        return super().__str__()


@dataclass
class ChannelInfo:
    """声道信息"""
    channel_name: str  # 声道名称 (左/右)
    intervals: List[TimeInterval] = field(default_factory=list)  # 活跃时间区间列表
    total_active_duration: float = 0.0  # 总活跃时长
    total_silent_duration: float = 0.0  # 总静音时长

    def add_interval(self, interval: TimeInterval):
        self.intervals.append(interval)
        self.total_active_duration += interval.duration

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"{self.channel_name}声道分析结果:",
            f"{'=' * 60}",
            f"活跃区间数量: {len(self.intervals)}",
            f"总活跃时长: {self.total_active_duration:.3f}s",
            f"总静音时长: {self.total_silent_duration:.3f}s",
            f"活跃占比: {(self.total_active_duration / (self.total_active_duration + self.total_silent_duration) * 100) if (self.total_active_duration + self.total_silent_duration) > 0 else 0:.2f}%",
        ]
        # for i, interval in enumerate(self.intervals, 1):
        #     lines.append(f"  {i:3d}. {interval}")
        # lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


def load_stereo_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    加载立体声音频文件

    Args:
        audio_path: 音频文件路径

    Returns:
        Tuple[np.ndarray, int]: 音频数据(shape为[n_samples, n_channels])和采样率

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 音频不是立体声
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    audio_data, sample_rate = sf.read(audio_path, dtype='float32')

    if audio_data.ndim == 1:
        raise ValueError(f"音频是单声道,需要立体声音频文件: {audio_path}")

    if audio_data.shape[1] != 2:
        raise ValueError(f"音频声道数为 {audio_data.shape[1]},需要立体声(2声道): {audio_path}")

    return audio_data, sample_rate


def extract_channel(audio_data: np.ndarray, channel: int) -> np.ndarray:
    """
    提取指定声道

    Args:
        audio_data: 音频数据 [n_samples, n_channels]
        channel: 声道索引 (0=左, 1=右)

    Returns:
        np.ndarray: 单声道音频数据
    """
    return audio_data[:, channel]


def detect_silence_regions(
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
        min_active_duration: float = 0.1
) -> List[TimeInterval]:
    """
    检测音频中的活跃区间(非静音区间)

    Args:
        audio: 单声道音频数据
        sample_rate: 采样率
        threshold_db: 静音阈值(dB),低于此值视为静音
        min_silence_duration: 最小静音持续时间(秒),短于此值的静音会被忽略
        min_active_duration: 最小活跃持续时间(秒),短于此值的活跃段会被忽略

    Returns:
        List[TimeInterval]: 活跃时间区间列表
    """
    # 计算音频能量(RMS)
    frame_size = int(sample_rate * 0.025)  # 25ms帧长
    hop_size = int(sample_rate * 0.010)  # 10ms帧移

    # 分帧
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)

    frames = np.array(frames)

    # 转换为dB
    epsilon = 1e-10
    frames_db = 20 * np.log10(frames + epsilon)

    # 二值化:高于阈值为活跃(1),否则为静音(0)
    is_active = frames_db > threshold_db

    # 转换帧索引到时间
    time_per_frame = hop_size / sample_rate

    # 查找连续活跃区间
    intervals = []
    in_active = False
    start_frame = 0

    for i, active in enumerate(is_active):
        if active and not in_active:
            # 进入活跃状态
            in_active = True
            start_frame = i
        elif not active and in_active:
            # 离开活跃状态
            in_active = False
            end_frame = i
            start_time = start_frame * time_per_frame
            end_time = end_frame * time_per_frame

            # 过滤过短的活跃段
            if end_time - start_time >= min_active_duration:
                intervals.append(TimeInterval(start=start_time, end=end_time))

    # 处理最后一个活跃段
    if in_active:
        start_time = start_frame * time_per_frame
        end_time = len(audio) / sample_rate
        if end_time - start_time >= min_active_duration:
            intervals.append(TimeInterval(start=start_time, end=end_time))

    # 合并间隔过短的区间
    if intervals:
        merged_intervals = [intervals[0]]
        for interval in intervals[1:]:
            gap = interval.start - merged_intervals[-1].end
            if gap < min_silence_duration:
                # 合并区间
                merged_intervals[-1] = TimeInterval(
                    start=merged_intervals[-1].start,
                    end=interval.end
                )
            else:
                merged_intervals.append(interval)
        intervals = merged_intervals

    return intervals


def process_stereo_audio(
        audio_path: str,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
        min_active_duration: float = 0.0,
        output_file: str = None
) -> Dict[str, ChannelInfo]:
    """
    处理立体声音频,提取左右声道的时间区间

    Args:
        audio_path: 音频文件路径
        threshold_db: 静音阈值(dB)
        min_silence_duration: 最小静音持续时间(秒)
        min_active_duration: 最小活跃持续时间(秒)
        output_file: 输出文件路径(可选,支持.txt/.json格式)

    Returns:
        Dict[str, ChannelInfo]: 包含左右声道信息的字典
    """
    print(f"正在处理音频文件: {audio_path}")

    # 加载音频
    audio_data, sample_rate = load_stereo_audio(audio_path)
    duration = len(audio_data) / sample_rate
    print(f"音频时长: {duration:.3f}s, 采样率: {sample_rate}Hz")

    # 提取左右声道
    left_channel = extract_channel(audio_data, 0)
    right_channel = extract_channel(audio_data, 1)

    print("正在分析左声道...")
    left_intervals = detect_silence_regions(
        left_channel,
        sample_rate,
        threshold_db,
        min_silence_duration,
        min_active_duration
    )

    print("正在分析右声道...")
    right_intervals = detect_silence_regions(
        right_channel,
        sample_rate,
        threshold_db,
        min_silence_duration,
        min_active_duration
    )

    # 构建结果
    left_info = ChannelInfo(channel_name="左声道")
    for interval in left_intervals:
        left_info.add_interval(interval)
    left_info.total_silent_duration = duration - left_info.total_active_duration

    right_info = ChannelInfo(channel_name="右声道")
    for interval in right_intervals:
        right_info.add_interval(interval)
    right_info.total_silent_duration = duration - right_info.total_active_duration

    #  如果左声道中的开始结束时间在右声道某一个区间的范围中,则将左声道的时间区间删除
    i = 0
    while i < len(left_info.intervals):
        left_interval = left_info.intervals[i]
        should_remove = False
        for right_interval in right_info.intervals:
            if left_interval.start >= right_interval.start and left_interval.end <= right_interval.end:
                should_remove = True
                break

        if should_remove:
            left_info.intervals.pop(i)
        else:
            i += 1

    # 如果右声道中的开始结束时间在左声道某一个区间的范围中,则将右声道的时间区间删除
    i = 0
    while i < len(right_info.intervals):
        right_interval = right_info.intervals[i]
        should_remove = False
        for left_interval in left_info.intervals:
            if right_interval.start >= left_interval.start and right_interval.end <= left_interval.end:
                should_remove = True
                break

        if should_remove:
            right_info.intervals.pop(i)
        else:
            i += 1

    result = {
        "left": left_info,
        "right": right_info
    }

    # 打印结果
    print(left_info.summary())
    print(right_info.summary())

    # 输出到文件
    if output_file:
        export_to_file(result, duration, output_file)

    return result


def export_to_file(result: Dict[str, ChannelInfo], total_duration: float, output_file: str):
    """
    导出结果到文件

    Args:
        result: 声道信息字典
        total_duration: 音频总时长
        output_file: 输出文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == '.json':
        _export_json(result, total_duration, output_path)
    else:
        _export_text(result, total_duration, output_path)

    print(f"结果已保存到: {output_file}")


def _export_text(result: Dict[str, ChannelInfo], total_duration: float, output_path: Path):
    """导出为文本格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"音频分析报告\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"总时长: {total_duration:.3f}s\n\n")

        for channel_key in ['left', 'right']:
            info = result[channel_key]
            f.write(info.summary())


def _export_json(result: Dict[str, ChannelInfo], total_duration: float, output_path: Path):
    """导出为JSON格式"""
    import json

    data = {
        "total_duration": round(total_duration, 3),
        "channels": {}
    }

    for channel_key in ['left', 'right']:
        info = result[channel_key]
        data["channels"][channel_key] = {
            "channel_name": info.channel_name,
            "total_active_duration": round(info.total_active_duration, 3),
            "total_silent_duration": round(info.total_silent_duration, 3),
            "active_ratio": round(
                (info.total_active_duration / (info.total_active_duration + info.total_silent_duration) * 100)
                if (info.total_active_duration + info.total_silent_duration) > 0 else 0,
                2
            ),
            "intervals": [
                {
                    "start": round(iv.start, 3),
                    "end": round(iv.end, 3),
                    "duration": round(iv.duration, 3)
                }
                for iv in info.intervals
            ]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # 示例用法
    audio_file = './input.mp3'

    if os.path.exists(audio_file):
        # 处理音频并输出结果
        result = process_stereo_audio(
            audio_path=audio_file,
            threshold_db=-40.0,  # 静音阈值
            min_silence_duration=0.3,  # 最小静音时长
            min_active_duration=0.1,  # 最小活跃时长
        )

        # 也可以单独访问左右声道信息
        print("\n左声道活跃区间数:", len(result['left'].intervals))
        print("右声道活跃区间数:", len(result['right'].intervals))
    else:
        print(f"测试文件不存在: {audio_file}")
        print("请提供一个立体声音频文件进行测试")
