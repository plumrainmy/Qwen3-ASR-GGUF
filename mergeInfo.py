import os

from RoleAndTimerProcessUtil import process_stereo_audio, TimeInterval
from ASRProcessUtil import processASR
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from typing import Optional
from qwen_asr_gguf.inference.schema import ForcedAlignItem
from qwen_asr_gguf.inference.chinese_itn import chinese_to_num
import json
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class LabeledTimeInterval(TimeInterval):
    """带标签的时间区间"""
    label: str = ""  # 标签,如"客服"或"客户"
    content: str = ""  # 内容,如"你好"

    def __str__(self) -> str:
        if self.label:
            return f"[{self.start:.3f}s - {self.end:.3f}s] ({self.duration:.3f}s) - {self.label}"
        return super().__str__()


def merge_and_sort_intervals(
        left_intervals: List[TimeInterval],
        right_intervals: List[TimeInterval],
        left_label: str = "左声道",
        right_label: str = "右声道"
) -> List[LabeledTimeInterval]:
    """
    合并左右声道的时间区间并按开始时间排序

    Args:
        left_intervals: 左声道时间区间列表
        right_intervals: 右声道时间区间列表
        left_label: 左声道标签
        right_label: 右声道标签

    Returns:
        List[LabeledTimeInterval]: 按开始时间排序的带标签时间区间列表
    """
    merged = []

    # 添加左声道区间
    for interval in left_intervals:
        merged.append(LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=left_label
        ))

    # 添加右声道区间
    for interval in right_intervals:
        merged.append(LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=right_label
        ))

    # 按开始时间排序
    merged.sort(key=lambda x: x.start)

    return merged


def merge_and_sort_intervals_with_text(
        left_intervals: List[TimeInterval],
        right_intervals: List[TimeInterval],
        left_align_items: Optional[List[ForcedAlignItem]] = None,
        right_align_items: Optional[List[ForcedAlignItem]] = None,
        left_label: str = "客服",
        right_label: str = "客户"
) -> List[LabeledTimeInterval]:
    """
    合并左右声道的时间区间,添加标签和文本内容,并按开始时间排序

    Args:
        left_intervals: 左声道时间区间列表
        right_intervals: 右声道时间区间列表
        left_align_items: 左声道ASR对齐结果
        right_align_items: 右声道ASR对齐结果
        left_label: 左声道标签
        right_label: 右声道标签

    Returns:
        List[LabeledTimeInterval]: 按开始时间排序的带标签和时间区间列表
    """
    merged = []
    left_matched_indices = set()
    right_matched_indices = set()

    # 添加左声道区间
    for idx, interval in enumerate(left_intervals):
        content = ""
        if left_align_items:
            content = find_matching_text(
                interval.start,
                interval.end,
                left_align_items,
                left_matched_indices,
                is_last_call=idx == len(left_intervals) - 1
            )

        merged.append(LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=left_label,
            content=content
        ))

    # 添加右声道区间
    for idx, interval in enumerate(right_intervals):
        content = ""
        if right_align_items:
            content = find_matching_text(
                interval.start,
                interval.end,
                right_align_items,
                right_matched_indices,
                is_last_call=idx == len(right_intervals) - 1
            )

        merged.append(LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=right_label,
            content=content
        ))

    # 按开始时间排序
    merged.sort(key=lambda x: x.start)

    return merged


def find_matching_text(
        start_time: float,
        end_time: float,
        align_items: List[ForcedAlignItem],
        matched_indices: Optional[set[int]] = None,
        is_last_call: bool = False
) -> str:
    """
    根据时间区间查找匹配的文本内容

    Args:
        start_time: 开始时间
        end_time: 结束时间
        align_items: 对齐结果列表

    Returns:
        str: 匹配的文本内容,如果没有匹配则返回空字符串
    """
    matched_texts = []
    if matched_indices is None:
        matched_indices = set()

    for idx, item in enumerate(align_items):
        if idx in matched_indices:
            continue

        # 如果当前区间开始前还有未匹配内容，优先并入本次结果，避免前面的漏字一直滞留。
        if item.end_time <= start_time:
            matched_texts.append(item.text)
            matched_indices.add(idx)
            continue

        # 判断是否有时间重叠
        if item.start_time < end_time and item.end_time > start_time:
            matched_texts.append(item.text)
            matched_indices.add(idx)

    if is_last_call:
        for idx, item in enumerate(align_items):
            if idx in matched_indices:
                continue
            if item.start_time >= end_time:
                matched_texts.append(item.text)
                matched_indices.add(idx)

    return "".join(matched_texts)


def print_sorted_intervals_with_text(
        sorted_intervals: List[LabeledTimeInterval],
        title: str = "对话记录(按时间排序)"
):
    """
    打印带文本内容的排序时间区间

    Args:
        sorted_intervals: 排序后的时间区间列表
        title: 标题
    """
    merged_intervals: List[LabeledTimeInterval] = []
    for interval in sorted_intervals:
        current_content = interval.content if interval.content else ""
        leading_punctuation = ""
        while current_content and current_content[0] in "，。！？、；：,.!?)]）】":
            leading_punctuation += current_content[0]
            current_content = current_content[1:]

        if leading_punctuation:
            for index in range(len(merged_intervals) - 1, -1, -1):
                previous = merged_intervals[index]
                if previous.label != interval.label:
                    continue

                previous_content = previous.content if previous.content else ""
                merged_intervals[index] = LabeledTimeInterval(
                    start=previous.start,
                    end=previous.end,
                    label=previous.label,
                    content=f"{previous_content}{leading_punctuation}"
                )
                break

        normalized_interval = LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=interval.label,
            content=current_content
        )

        if not merged_intervals or merged_intervals[-1].label != normalized_interval.label:
            merged_intervals.append(normalized_interval)
            continue

        previous = merged_intervals[-1]
        merged_content = previous.content if previous.content else ""
        merged_intervals[-1] = LabeledTimeInterval(
            start=previous.start,
            end=normalized_interval.end,
            label=previous.label,
            content=f"{merged_content}{current_content}"
        )

    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'序号':<6} {'时间区间':<25} {'声道':<8} {'内容'}")
    print(f"{'-' * 80}")

    for i, interval in enumerate(merged_intervals, 1):
        time_str = f"[{interval.start:.3f}s - {interval.end:.3f}s]"
        content_display = interval.content if interval.content else "(静音)"
        # print(f"{i:<6} {time_str:<25} {interval.label:<8} {chinese_to_num(content_display)}")
        print(f"{interval.label:<8} {chinese_to_num(content_display)}")

    print(f"{'=' * 80}")
    print(f"总计: {len(merged_intervals)} 个片段\n")


def trim_trailing_silence(
        audio_path: str,
        threshold_db: float = -40.0,
        min_trailing_silence: float = 0.3,
        keep_tail_padding: float = 0.2,
        min_active_tail_duration: Optional[float] = 0.1
) -> str:
    """
    如果音频尾部是静音，则裁掉尾静音并返回新文件路径；否则返回原路径。
    """
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    if audio_data.ndim == 1:
        channel_data = audio_data[:, np.newaxis]
    else:
        channel_data = audio_data

    frame_size = max(1, int(sample_rate * 0.025))
    hop_size = max(1, int(sample_rate * 0.010))
    time_per_frame = hop_size / sample_rate
    epsilon = 1e-10
    active_frames = []

    for frame_idx, start in enumerate(range(0, len(channel_data) - frame_size + 1, hop_size)):
        frame = channel_data[start:start + frame_size]
        channel_rms = np.sqrt(np.mean(frame ** 2, axis=0))
        frame_db = 20 * np.log10(np.max(channel_rms) + epsilon)
        active_frames.append(frame_db > threshold_db)

    last_active_frame = None
    if min_active_tail_duration is None or min_active_tail_duration <= 0:
        for frame_idx in range(len(active_frames) - 1, -1, -1):
            if active_frames[frame_idx]:
                last_active_frame = frame_idx
                break
    else:
        min_active_frames = max(1, int(np.ceil(min_active_tail_duration / time_per_frame)))
        consecutive_active = 0

        for frame_idx in range(len(active_frames) - 1, -1, -1):
            if active_frames[frame_idx]:
                consecutive_active += 1
                if consecutive_active >= min_active_frames:
                    last_active_frame = frame_idx + min_active_frames - 1
                    break
            else:
                consecutive_active = 0

    if last_active_frame is None:
        return audio_path

    audio_duration = len(channel_data) / sample_rate
    trim_end_time = min((last_active_frame + 1) * time_per_frame + keep_tail_padding, audio_duration)
    trailing_silence = audio_duration - trim_end_time
    if trailing_silence < min_trailing_silence:
        return audio_path

    trim_end_sample = min(len(audio_data), int(trim_end_time * sample_rate))
    trimmed_audio = audio_data[:trim_end_sample]
    trimmed_path = str(Path(audio_path).with_name(f"{Path(audio_path).stem}_trimmed{Path(audio_path).suffix}"))
    sf.write(trimmed_path, trimmed_audio, sample_rate)
    return trimmed_path


def process_channel_audio(
        audio_path: str,
        trim_min_active_tail_duration: Optional[float] = None,
        left_label: str = "客服：",
        right_label: str = "客户："
) -> List[LabeledTimeInterval]:
    if trim_min_active_tail_duration is not None:
        audio_path = trim_trailing_silence(
            audio_path,
            min_active_tail_duration=trim_min_active_tail_duration
        )

    result = process_stereo_audio(audio_path)
    left = result['left'].intervals
    right = result['right'].intervals

    res = processASR(audio_path)
    print(res.alignment)
    return merge_and_sort_intervals_with_text(
        left,
        right,
        res.alignment,
        res.alignment,
        left_label=left_label,
        right_label=right_label
    )


if __name__ == '__main__':
    audio_base_path = './wav/13623481'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    merged_and_sort_left = process_channel_audio(
        f'{audio_base_path}_left.wav',
        trim_min_active_tail_duration=0.1
    )

    merged_and_sort_right = process_channel_audio(f'{audio_base_path}_right.wav')

    # 新创建一个数组合并merged_and_sort_left merged_and_sort_right 这两个数据都是返回的List[LabeledTimeInterval] 按照start 排序合并展示
    merged_all = merged_and_sort_left + merged_and_sort_right
    merged_all.sort(key=lambda x: x.start)

    print_sorted_intervals_with_text(merged_all, title="合并后的对话记录(左右声道音频整合)")
