"""汉字层面的文本比对。纯标准库，不碰模型，毫秒级。

为什么只比汉字：闸门 1 要回答的是「内容有没有被吞掉」，而标点、空格、
英文大小写在 ASR 回环里本来就不稳定。把它们纳入比较只会制造噪声，
把真实的丢字信号淹掉。

为什么用 difflib 而不是双指针：见 `coverage` 的说明。
"""

from difflib import SequenceMatcher

_HAN_START = "一"
_HAN_END = "鿿"


def han_only(text: str) -> str:
    """只保留基本区汉字（U+4E00–U+9FFF）。"""
    return "".join(c for c in text if _HAN_START <= c <= _HAN_END)


def _matcher(reference: str, heard: str) -> SequenceMatcher:
    # autojunk 会把出现频率高的元素当「垃圾」跳过。中文长文本里高频字（的、了、
    # 是）恰恰是正常内容，一旦被当垃圾，覆盖率会无故偏低。必须关掉。
    return SequenceMatcher(None, han_only(reference), han_only(heard), autojunk=False)


def coverage(reference: str, heard: str) -> float:
    """参考文本中有多少比例的汉字在听到的文本里找到了对应物。

    用 difflib 的最长公共子序列而不是手写贪心双指针：贪心版在遇到第一处
    不匹配后会永久失步，把「第一百二十二天 vs 第122天」这种局部替换读成
    3.5% 的相似度（设计 §6 实测）。difflib 会重新对齐后续内容。

    参考为空返回 0.0——空参考下的「100% 覆盖」是个没有意义的真空真理，
    让它通过闸门比让它失败更危险。
    """
    ref = han_only(reference)
    if not ref:
        return 0.0

    matched = sum(block.size for block in _matcher(reference, heard).get_matching_blocks())
    return matched / len(ref)


def missing_spans(reference: str, heard: str) -> tuple[tuple[int, str], ...]:
    """参考侧未被匹配上的连续片段，返回 ((汉字下标, 片段), ...)。

    下标是 `han_only(reference)` 内的下标——诊断时要能直接定位到原句，
    所以给的是「第几个汉字」而不是原始字符偏移。
    """
    ref = han_only(reference)
    if not ref:
        return ()

    return tuple(
        (i1, ref[i1:i2])
        for tag, i1, i2, _j1, _j2 in _matcher(reference, heard).get_opcodes()
        if tag in ("delete", "replace")
    )
