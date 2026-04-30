# coding=utf-8
import numpy as np
from typing import List, Optional

SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]


def normalize_language_name(language: str) -> str:
    """
    将语言名称归一化为 Qwen3-ASR 使用的标准格式：
    首字母大写，其余小写（例如 'cHINese' -> 'Chinese'）。
    """
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    """
    验证语言是否在支持列表中。
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def detect_and_fix_repetitions(text: str, threshold: int = 4) -> str:
    """
    检测并修复文本中的异常重复模式（移植自官方 qwen_asr/inference/utils.py）。

    适用于 ASR 解码后的后处理，去除模型幻觉产生的大量重复字符或短语。
    threshold=4 表示同一模式连续出现 ≥4 次才触发去重。
    降低阈值以捕获 " 洞 洞 洞 洞" 等空格间隔的重复幻觉模式。

    Args:
        text: 待处理文本
        threshold: 重复阈值，连续出现 ≥ threshold 次才触发去重

    Returns:
        去重后的文本
    """

    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i + count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s

        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break

                pattern = s[i:i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx + k] != pattern:
                        valid = False
                        break

                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break

            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)

    text = fix_char_repeats(text, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def is_hallucination(text: str, min_length: int = 4, max_unique_ratio: float = 0.1) -> bool:
    """
    检测 ASR 输出文本是否为幻觉（单字/双字高频重复的无意义输出）。

    判定规则（满足任一即为幻觉）：
      1. 去除空格和标点后，唯一字符种类 ≤ 2 且文本长度 ≥ min_length
      2. 单个字符占比 ≥ (1 - max_unique_ratio)，即 90%+ 为同一字符

    典型幻觉模式：
      - " 洞 洞 洞 洞 洞 洞 洞 洞 洞 洞"
      - "三三三三三三三三三三"
      - "的的的的的的的的"

    Args:
        text: 待检测文本
        min_length: 触发检测的最小字符数（去空格后）
        max_unique_ratio: 字符多样性阈值，低于此值则判定为幻觉

    Returns:
        True → 文本疑似幻觉，应丢弃
        False → 文本正常
    """
    if not text:
        return False

    # 去除空格和常见标点
    import re
    cleaned = re.sub(r'[\s，。、！？,.!?\-—]+', '', text)

    if len(cleaned) < min_length:
        return False

    unique_chars = set(cleaned)

    # 规则 1：唯一字符 ≤ 2（如全是 "洞" 或 "三洞" 交替）
    if len(unique_chars) <= 2:
        return True

    # 规则 2：某个字符占比 ≥ 90%（如 "三洞洞洞洞洞洞洞洞洞"，"洞"占 90%）
    from collections import Counter
    counts = Counter(cleaned)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count / len(cleaned) >= (1 - max_unique_ratio):
        return True

    return False
