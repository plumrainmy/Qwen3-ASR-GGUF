# coding=utf-8
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import List, Sequence, Tuple


LEFT_PATTERN = re.compile(
    r"^\s*(?P<index>\d+)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<speaker>[^：:\s]+)\s+"
    r"(?P<content>.+?)\s*$"
)
RIGHT_PATTERN = re.compile(r"^\s*(?P<speaker>[^：:]+?)\s*[：:]\s*(?P<content>.+?)\s*$")

SPEAKER_ALIASES = {
    "客服": "客服",
    "坐席": "客服",
    "专员": "客服",
    "销售": "客服",
    "客户": "客户",
    "用户": "客户",
    "来电人": "客户",
    "顾客": "客户",
}


@dataclass
class DialogueTurn:
    speaker: str
    content: str
    raw_line: str
    line_no: int


@dataclass
class MatchResult:
    left_index: int
    right_index: int
    speaker: str
    left_content: str
    right_content: str
    similarity: float


def normalize_speaker(speaker: str) -> str:
    cleaned = re.sub(r"\s+", "", speaker.strip())
    return SPEAKER_ALIASES.get(cleaned, cleaned)


def normalize_content(content: str) -> str:
    text = content.strip()
    text = re.sub(r"\（.*?\）|\(.*?\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"[，。！？、；：,.!?\-~\s]+", "", text)
    return text


def clean_turn_content(content: str) -> str:
    text = content.strip()
    text = re.sub(r"^[，。！？、；：,.!?]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningful_turn(content: str) -> bool:
    normalized = normalize_content(content)
    if not normalized:
        return False
    silence_tokens = {"静音", "silence"}
    return normalized.lower() not in silence_tokens


def should_merge_turns(previous: DialogueTurn, current: DialogueTurn) -> bool:
    if previous.speaker != current.speaker:
        return False

    previous_text = previous.content.strip()
    current_text = current.content.strip()
    if not previous_text or not current_text:
        return False

    # 很短且没有完整句末时，通常是半句碎片。
    if len(normalize_content(previous_text)) <= 3 and previous_text[-1] not in "。！？!?":
        return True

    # 前一句未以完整句末结束，通常是ASR切碎后的续接。
    if previous_text[-1] not in "。！？!?":
        return True

    # 如果前一句已经完整结束，只合并明显是残片的下一句。
    current_normalized = normalize_content(current_text)
    if len(current_normalized) <= 2:
        return True

    fragment_prefixes = ("，", "。", "、", "？", "！")
    if current_text.startswith(fragment_prefixes):
        return True

    filler_only_prefixes = ("嗯", "呃", "哦", "啊")
    return current_text.startswith(filler_only_prefixes) and len(current_normalized) <= 4


def parse_line(line: str, line_no: int) -> DialogueTurn | None:
    if not line.strip():
        return None

    left_match = LEFT_PATTERN.match(line)
    if left_match:
        speaker = normalize_speaker(left_match.group("speaker"))
        content = clean_turn_content(left_match.group("content"))
        return DialogueTurn(speaker=speaker, content=content, raw_line=line.rstrip("\n"), line_no=line_no)

    right_match = RIGHT_PATTERN.match(line)
    if right_match:
        speaker = normalize_speaker(right_match.group("speaker"))
        content = clean_turn_content(right_match.group("content"))
        return DialogueTurn(speaker=speaker, content=content, raw_line=line.rstrip("\n"), line_no=line_no)

    return None


def merge_adjacent_turns(turns: Sequence[DialogueTurn]) -> List[DialogueTurn]:
    merged: List[DialogueTurn] = []

    for turn in turns:
        if not merged or not should_merge_turns(merged[-1], turn):
            merged.append(turn)
            continue

        previous = merged[-1]
        merged[-1] = DialogueTurn(
            speaker=previous.speaker,
            content=f"{previous.content}{turn.content}",
            raw_line=f"{previous.raw_line}\n{turn.raw_line}",
            line_no=previous.line_no,
        )

    return merged


def parse_dialogues(
    text: str,
    merge_consecutive_same_speaker: bool = False,
    drop_non_meaningful: bool = False,
) -> Tuple[List[DialogueTurn], List[str]]:
    turns: List[DialogueTurn] = []
    skipped_lines: List[str] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        turn = parse_line(line, line_no)
        if turn is None:
            if line.strip():
                skipped_lines.append(f"第{line_no}行: {line}")
            continue
        if drop_non_meaningful and not is_meaningful_turn(turn.content):
            continue
        turns.append(turn)

    if merge_consecutive_same_speaker:
        turns = merge_adjacent_turns(turns)

    return turns, skipped_lines


def content_similarity(left: str, right: str) -> float:
    left_clean = normalize_content(left)
    right_clean = normalize_content(right)
    if not left_clean and not right_clean:
        return 1.0
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def build_score_matrix(left_turns: Sequence[DialogueTurn], right_turns: Sequence[DialogueTurn]) -> List[List[float]]:
    rows = len(left_turns) + 1
    cols = len(right_turns) + 1
    dp = [[0.0] * cols for _ in range(rows)]
    gap_penalty = 0.45

    for i in range(1, rows):
        dp[i][0] = dp[i - 1][0] - gap_penalty
    for j in range(1, cols):
        dp[0][j] = dp[0][j - 1] - gap_penalty

    for i in range(1, rows):
        for j in range(1, cols):
            left_turn = left_turns[i - 1]
            right_turn = right_turns[j - 1]
            speaker_bonus = 1.0 if left_turn.speaker == right_turn.speaker else -0.7
            similarity = content_similarity(left_turn.content, right_turn.content)
            match_score = dp[i - 1][j - 1] + speaker_bonus + similarity
            delete_score = dp[i - 1][j] - gap_penalty
            insert_score = dp[i][j - 1] - gap_penalty
            dp[i][j] = max(match_score, delete_score, insert_score)

    return dp


def align_dialogues(left_turns: Sequence[DialogueTurn], right_turns: Sequence[DialogueTurn]) -> Tuple[List[MatchResult], List[int], List[int]]:
    dp = build_score_matrix(left_turns, right_turns)
    matches: List[MatchResult] = []
    unmatched_left: List[int] = []
    unmatched_right: List[int] = []
    gap_penalty = 0.45
    i = len(left_turns)
    j = len(right_turns)

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            left_turn = left_turns[i - 1]
            right_turn = right_turns[j - 1]
            speaker_bonus = 1.0 if left_turn.speaker == right_turn.speaker else -0.7
            similarity = content_similarity(left_turn.content, right_turn.content)
            match_score = dp[i - 1][j - 1] + speaker_bonus + similarity
            if abs(dp[i][j] - match_score) < 1e-9:
                if left_turn.speaker == right_turn.speaker:
                    matches.append(
                        MatchResult(
                            left_index=i - 1,
                            right_index=j - 1,
                            speaker=left_turn.speaker,
                            left_content=left_turn.content,
                            right_content=right_turn.content,
                            similarity=similarity,
                        )
                    )
                else:
                    unmatched_left.append(i - 1)
                    unmatched_right.append(j - 1)
                i -= 1
                j -= 1
                continue

        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] - gap_penalty)) < 1e-9:
            unmatched_left.append(i - 1)
            i -= 1
            continue

        if j > 0:
            unmatched_right.append(j - 1)
            j -= 1

    matches.reverse()
    unmatched_left.reverse()
    unmatched_right.reverse()
    return matches, unmatched_left, unmatched_right


def recover_exact_short_matches(
    left_turns: Sequence[DialogueTurn],
    right_turns: Sequence[DialogueTurn],
    matches: Sequence[MatchResult],
    unmatched_left: Sequence[int],
    unmatched_right: Sequence[int],
) -> Tuple[List[MatchResult], List[int], List[int]]:
    recovered_matches = list(matches)
    remaining_left = list(unmatched_left)
    remaining_right = list(unmatched_right)
    used_right: set[int] = set()

    for left_index in unmatched_left:
        left_turn = left_turns[left_index]
        left_normalized = normalize_content(left_turn.content)
        if len(left_normalized) > 4:
            continue

        for right_index in remaining_right:
            if right_index in used_right:
                continue
            right_turn = right_turns[right_index]
            if left_turn.speaker != right_turn.speaker:
                continue
            if left_normalized != normalize_content(right_turn.content):
                continue

            recovered_matches.append(
                MatchResult(
                    left_index=left_index,
                    right_index=right_index,
                    speaker=left_turn.speaker,
                    left_content=left_turn.content,
                    right_content=right_turn.content,
                    similarity=1.0,
                )
            )
            used_right.add(right_index)
            break

    matched_left_indexes = {item.left_index for item in recovered_matches}
    matched_right_indexes = {item.right_index for item in recovered_matches}
    remaining_left = [index for index in remaining_left if index not in matched_left_indexes]
    remaining_right = [index for index in remaining_right if index not in matched_right_indexes]
    recovered_matches.sort(key=lambda item: (item.left_index, item.right_index))
    return recovered_matches, remaining_left, remaining_right


def compare_texts(left_text: str, right_text: str) -> dict:
    left_turns, left_skipped = parse_dialogues(left_text)
    right_turns, right_skipped = parse_dialogues(
        right_text,
        merge_consecutive_same_speaker=True,
        drop_non_meaningful=True,
    )

    matches, unmatched_left, unmatched_right = align_dialogues(left_turns, right_turns)
    matches, unmatched_left, unmatched_right = recover_exact_short_matches(
        left_turns,
        right_turns,
        matches,
        unmatched_left,
        unmatched_right,
    )
    matched_similarity_sum = sum(item.similarity for item in matches)
    denominator = max(len(left_turns), len(right_turns), 1)
    overall_similarity = matched_similarity_sum / denominator
    role_accuracy = (len(matches) / denominator) if denominator else 0.0

    return {
        "summary": {
            "left_turns": len(left_turns),
            "right_turns": len(right_turns),
            "matched_turns": len(matches),
            "left_unmatched": len(unmatched_left),
            "right_unmatched": len(unmatched_right),
            "role_accuracy": round(role_accuracy, 4),
            "overall_similarity": round(overall_similarity, 4),
        },
        "matches": [
            {
                "left_line_no": left_turns[item.left_index].line_no,
                "right_line_no": right_turns[item.right_index].line_no,
                "speaker": item.speaker,
                "similarity": round(item.similarity, 4),
                "left_content": item.left_content,
                "right_content": item.right_content,
            }
            for item in matches
        ],
        "unmatched_left": [asdict(left_turns[index]) for index in unmatched_left],
        "unmatched_right": [asdict(right_turns[index]) for index in unmatched_right],
        "skipped_lines": {
            "left": left_skipped,
            "right": right_skipped,
        },
    }


def print_report(result: dict) -> None:
    summary = result["summary"]
    print("=== 对话相似度结果 ===")
    print(f"结果1句数: {summary['left_turns']}")
    print(f"结果2句数: {summary['right_turns']}")
    print(f"匹配句数: {summary['matched_turns']}")
    print(f"角色匹配率: {summary['role_accuracy']:.2%}")
    print(f"整体相似度: {summary['overall_similarity']:.2%}")
    print("")
    print("=== 逐句匹配 ===")
    for item in result["matches"]:
        print(
            f"[结果1 第{item['left_line_no']}行] <-> [结果2 第{item['right_line_no']}行] "
            f"{item['speaker']} 相似度={item['similarity']:.2%}"
        )
        print(f"  结果1: {item['left_content']}")
        print(f"  结果2: {item['right_content']}")

    if result["unmatched_left"]:
        print("")
        print("=== 结果1未匹配 ===")
        for item in result["unmatched_left"]:
            print(f"第{item['line_no']}行 {item['speaker']}: {item['content']}")

    if result["unmatched_right"]:
        print("")
        print("=== 结果2未匹配 ===")
        for item in result["unmatched_right"]:
            print(f"第{item['line_no']}行 {item['speaker']}: {item['content']}")

    skipped = result["skipped_lines"]
    if skipped["left"] or skipped["right"]:
        print("")
        print("=== 未识别行 ===")
        for side, lines in (("结果1", skipped["left"]), ("结果2", skipped["right"])):
            for line in lines:
                print(f"{side} {line}")


def main() -> None:
    file1_text = """
1 2026-04-08 17:51:31 客服 李先生您好，请问您是需要咨询这个冯秀玲员工的这个理赔案件进度事宜是吗？
2 2026-04-08 17:51:38 客户 对。
3 2026-04-08 17:51:39 客服 好的，呃我查看了一下这个呃理赔案件呢，目前还没有上传资料申请理赔的，请问您与这个冯秀玲是什么关系啊？
4 2026-04-08 17:51:50 客户 雇佣关系。
5 2026-04-08 17:51:51 客服 好的，就您是这个呃被保公司的负责人是吗？
6 2026-04-08 17:51:56 客户 嗯也不是吧，就是公司的人嗯。
7 2026-04-08 17:51:59 客服 嗯好的嗯。嗯那您看一下呃就是这个伤者有没有治疗康复，如果有治疗康复的话呢，呃您联
8 2026-04-08 17:52:00 客户 嗯。
9 2026-04-08 17:52:06 客户 那法院已经判了，他没有走走这个流程吗？
10 2026-04-08 17:52:09 客服 但我看了这个案件还没有提交资料的呀，是有把资料提交给哪个人员了吗？
11 2026-04-08 17:52:16 客户 嗯这这这是当地就是我上保险那个人，他是应该是他提供保险吧。
12 2026-04-08 17:52:22 客服 我看了一下呃就是这个案件呢我看是配置给您呃公司的一个法务155尾号为7245这个人员呢提交申请理赔的，但我看了一下案件还没有提交资料。
13 2026-04-08 17:52:37 客户 还没有提交资料啊。
14 2026-04-08 17:52:38 客服 对，您可以联系一下噢您公司的那个法务呢问一下他这个啊情况。
15 2026-04-08 17:52:45 客户 嗯帮帮能帮我说一下哪个手机号吗？我看是谁啊。
16 2026-04-08 17:52:49 客服 嗯我只能看到前3位和后4位中间部分隐藏的是13155开头，尾号为7245嗯。
17 2026-04-08 17:52:57 客户 7245是吧？
18 2026-04-08 17:52:58 客服 对，尾号为7245这个人员提交资料。
19 2026-04-08 17:53:02 客户 噢。前面是155，
20 2026-04-08 17:53:05 客服 我看155开头嗯。72。
21 2026-04-08 17:53:08 客户 155开头。
22 2026-04-08 17:53:09 客服 对，7245结尾。
23 2026-04-08 17:53:11 客户 724。噢明白了。
24 2026-04-08 17:53:15 客户 好的我知道了嗯嗯。
25 2026-04-08 17:53:15 客服 那还有其他方面可以帮您的吗？
26 2026-04-08 17:53:18 客户 啊没有我问一下啊嗯。
27 2026-04-08 17:53:19 客服 嗯好的，那不打扰您了，再见。
""".strip()

    file2_text = """
客户：      (静音)
客服：      哎，您好，我这边是那个美保理赔中心的，就是咱单位之前报那个林立国的案件，想问一下现在这个案件是进行到哪一步了呀？
客户：      他这个家伙啊，在医院里住了3天医院，他现在
客服：      嗯
客户：      要评残，
客服：      ，
客户：      傻逼人
客服：      嗯，呃，嗯，
客户：      ！你看他，他评得到残不？傻逼人
客服：      这个，呃，我看之前说是伤者是起诉单位是吗
客户：      ！他，对呀，打打官司了，
客服：      ？现在
客户：      开庭了开了两两两三次了。
客服：      ，呃，那现在有没有判下来呀？还是
客户：      他怎么他怎么能评评得到残？我就问你
客服：      ，呃，但是我看他这个确实是骨折啊，因为一般来说有骨折，在这个工伤标准它也是能评得上的，所以我说想问一下这个法院咋判的
客户：      ，他有什么骨折啊？他骨折伤筋动骨100天，他3天了就去我去我工地里闹事，去报警，带他老婆
客服：      ？哦，了解
客户：      。他有骨折吗
客服：      ，呃，那现在就是说咱法院现在是怎，就是有没有说给出判决啊？还是呃咋调解的？
客户：      ？没有，没有
客服：      就是就是他还现在还在上诉是吧
客户：      ，对
客服：      ？哦，行，呃，那这个也只有说等到后续看一下，那这个先和他调解喽
客户：      ，对对对，这个
客服：      ，嗯
客户：      如果是他能评得到残了，那就没有没有王法了，好吧
客服：      ，
客户：      ？他总共住院住了两天半，第3天就去我工地里闹事
客服：      嗯，嗯，
客户：      ，啊
客服：      啊，
客户：      ，报警啊，带他老婆啊
客服：      哎，
客户：      ，不让装设备，你知道吧
客服：      我我我看当时提供的资料他是住了10天院来的不
客户：      ？没有
客服：      ？
客户：      ，这个东西有证据的，你知道吧？可以对上日期的，住了两天半，跑到我工地去去去要工钱，去报警，你知道吗？啊
客服：      哦，哦，就是他当时提供的这个资料不是说在工地受了伤的吗？还是啥意思啊？为啥？您说的情况和提供的资料对不上
客户：      ，我也不知道，反正他很很轻很轻，知道吧
客服：      啊？哦，行，行
客户：      ？他是他是想在里面嗯要钱嘞，他要得到吗？嗯，
客服：      行
客户：      傻逼啊，嗯
客服：      行，好的，那我这边了解了，那就麻烦您这边再和他沟通一下吧。然后后续如果说，嗯，好好，那先不打电话，再见啊。
客户：      ，哎，行行行，我知道知道，嗯，啊，嗯。

""".strip()

    output_json = False
    result = compare_texts(file1_text, file2_text)
    if output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print_report(result)


if __name__ == "__main__":
    main()
