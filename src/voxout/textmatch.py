"""汉字层面的文本比对。不碰模型，毫秒级。

为什么只比汉字：闸门 1 要回答的是「内容有没有被吞掉」，而标点、空格、
英文大小写在 ASR 回环里本来就不稳定。把它们纳入比较只会制造噪声，
把真实的丢字信号淹掉。

同理还要消掉繁简：Whisper-zh 对同一段音频有时写简体有时写繁体，同一次渲染里
两种混着出现。「權」和「权」是一个字的两种写法，把它算成丢字，闸门量的就是
ASR 的书写偏好而不是内容。T13 实测（帖2，同一批音频只换比较口径）：裸比较
89.44% → 数字双侧归一化 95.37% → 再加繁转简 97.59%。

比较口径的通则：**参考侧与听到侧永远走同一个归一化**。单侧归一化会把等价关系
弄成有方向的，反方向的差异照旧算成丢字。数字（`normalize`，在 verify 里对两侧
调用）与繁简（本模块）都遵守这一条。

为什么用 difflib 而不是双指针：见 `coverage` 的说明。
"""

from difflib import SequenceMatcher

from opencc import OpenCC

_HAN_START = "一"
_HAN_END = "鿿"

_T2S = OpenCC("t2s")
"""繁→简转换器。构造耗时 2.3 ms（一次），转换 540 汉字 0.55 ms，可以放在比较路径上。"""


def han_only(text: str) -> str:
    """只保留基本区汉字（U+4E00–U+9FFF）。

    **不做繁简归一**：`chunker` 用这个函数数汉字来定切段长度，而繁简转换在词组
    层面可能改变字数，一旦改变，切段的「汉字数守恒」就不再成立。归一化只属于
    比较路径（`_canonical`）。
    """
    return "".join(c for c in text if _HAN_START <= c <= _HAN_END)


def _canonical(text: str) -> str:
    """比较用的规范形：只留汉字，再统一到简体。

    实测两篇真实短帖（213 / 540 汉字）经此转换后与 `han_only` 恒等——简体输入是
    不动点。这一点对 `missing_spans` 很重要，理由见那里。
    """
    return _T2S.convert(han_only(text))


def _matcher(reference: str, heard: str) -> SequenceMatcher:
    # autojunk 会把出现频率高的元素当「垃圾」跳过。中文长文本里高频字（的、了、
    # 是）恰恰是正常内容，一旦被当垃圾，覆盖率会无故偏低。必须关掉。
    return SequenceMatcher(None, _canonical(reference), _canonical(heard), autojunk=False)


def coverage(reference: str, heard: str) -> float:
    """参考文本中有多少比例的汉字在听到的文本里找到了对应物。

    用 difflib 的最长公共子序列而不是手写贪心双指针：贪心版在遇到第一处
    不匹配后会永久失步，把「第一百二十二天 vs 第122天」这种局部替换读成
    3.5% 的相似度（设计 §6 实测）。difflib 会重新对齐后续内容。

    参考为空返回 0.0——空参考下的「100% 覆盖」是个没有意义的真空真理，
    让它通过闸门比让它失败更危险。
    """
    ref = _canonical(reference)
    if not ref:
        return 0.0

    matched = sum(block.size for block in _matcher(reference, heard).get_matching_blocks())
    return matched / len(ref)


def missing_spans(reference: str, heard: str) -> tuple[tuple[int, str], ...]:
    """参考侧未被匹配上的连续片段，返回 ((汉字下标, 片段), ...)。

    下标是 `_canonical(reference)` 内的下标——诊断时要能直接定位到原句，
    所以给的是「第几个汉字」而不是原始字符偏移。

    差异报告里与这些下标并列展示的是**未**归一化的朗读文本，所以两串等长这件事
    是报告可读的前提。简体输入下归一化是恒等（`_canonical` 有实测），而朗读文本
    是我们自己写的简体，所以前提成立；哪天参考侧真出现繁体，报告里的位置会指向
    归一化后的串，片段文本本身仍然正确。
    """
    ref = _canonical(reference)
    if not ref:
        return ()

    return tuple(
        (i1, ref[i1:i2])
        for tag, i1, i2, _j1, _j2 in _matcher(reference, heard).get_opcodes()
        if tag in ("delete", "replace")
    )
