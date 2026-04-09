from RoleAndTimerProcessUtil import process_stereo_audio, TimeInterval
from ASRProcessUtil import processASR
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from typing import Optional
from qwen_asr_gguf.inference.schema import ForcedAlignItem
from qwen_asr_gguf.inference.chinese_itn import chinese_to_num
import json


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

    # 添加左声道区间
    for interval in left_intervals:
        content = ""
        if left_align_items:
            content = find_matching_text(interval.start, interval.end, left_align_items)

        merged.append(LabeledTimeInterval(
            start=interval.start,
            end=interval.end,
            label=left_label,
            content=content
        ))

    # 添加右声道区间
    for interval in right_intervals:
        content = ""
        if right_align_items:
            content = find_matching_text(interval.start, interval.end, right_align_items)

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
        align_items: List[ForcedAlignItem]
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

    for item in align_items:
        # 判断是否有时间重叠
        if item.start_time < end_time and item.end_time > start_time:
            matched_texts.append(item.text)

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
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'序号':<6} {'时间区间':<25} {'声道':<8} {'内容'}")
    print(f"{'-' * 80}")

    for i, interval in enumerate(sorted_intervals, 1):
        time_str = f"[{interval.start:.3f}s - {interval.end:.3f}s]"
        content_display = interval.content if interval.content else "(静音)"
        # print(f"{i:<6} {time_str:<25} {interval.label:<8} {chinese_to_num(content_display)}")
        print(f"{interval.label:<8} {chinese_to_num(content_display)}")

    print(f"{'=' * 80}")
    print(f"总计: {len(sorted_intervals)} 个片段\n")


if __name__ == '__main__':
    path_left = './wav/13623755_left.wav'
    result = process_stereo_audio(path_left)
    left = result['left'].intervals
    right = result['right'].intervals

    res_left = processASR(path_left)
    print(res_left.alignment)
    merged_and_sort_left = merge_and_sort_intervals_with_text(
        left, right, res_left.alignment, res_left.alignment, left_label="客服：", right_label="客户：")



    path_right = './wav/13623755_right.wav'
    result = process_stereo_audio(path_right)
    left = result['left'].intervals
    right = result['right'].intervals

    res_right = processASR(path_right)
    print(res_right.alignment)
    merged_and_sort_right = merge_and_sort_intervals_with_text(
        left, right, res_right.alignment, res_right.alignment, left_label="客服：", right_label="客户：")

    # 新创建一个数组合并merged_and_sort_left merged_and_sort_right 这两个数据都是返回的List[LabeledTimeInterval] 按照start 排序合并展示
    merged_all = merged_and_sort_left + merged_and_sort_right
    merged_all.sort(key=lambda x: x.start)

    print_sorted_intervals_with_text(merged_all, title="合并后的对话记录(左右声道音频整合)")
