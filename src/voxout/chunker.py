"""长文切段。

存在的唯一理由是绕开生成循环的 token 上限（budget.py 把那个上限算成事前断言）。
因此第一条纪律：**切段不是丢内容的许可**。无标点的超长句宁可整段送出并告警，
也不许在字中间硬切——硬切会同时破坏韵律和闸门 1 的可读性。

实测 1050 汉字单次调用零截断，阈值取 900 是保守值。本轮两篇真实短帖
283 / 572 汉字，都走单段路径。
"""

import logging

from voxout.config import CHUNK_THRESHOLD_HAN
from voxout.textmatch import han_only

_LOG = logging.getLogger(__name__)

_SENTENCE_ENDS = "。！？；\n"
"""切段点。markdown 清洗后段落界是 `\\n`，与句末标点同等对待。"""


def _split_sentences(text: str) -> tuple[str, ...]:
    """按句末标点切成句子，标点跟在前一句尾部（韵律属于前一句）。"""
    sentences: list[str] = []
    current: list[str] = []

    for char in text:
        current.append(char)
        if char in _SENTENCE_ENDS:
            sentences.append("".join(current))
            current = []

    if current:
        sentences.append("".join(current))
    return tuple(sentences)


def _pack(sentences: tuple[str, ...], threshold_han: int) -> tuple[str, ...]:
    """贪心打包：句子装得下就装，装不下才开新段。"""
    segments: list[str] = []
    current: list[str] = []
    current_han = 0

    for index, sentence in enumerate(sentences):
        sentence_han = len(han_only(sentence))
        if current and current_han + sentence_han > threshold_han:
            segments.append("".join(current))
            current, current_han = [], 0

        if not current and sentence_han > threshold_han:
            _LOG.warning(
                "第 %d 句共 %d 汉字，超过阈值 %d 且内部无句末标点可切；整句送出以免丢字",
                index + 1,
                sentence_han,
                threshold_han,
            )

        current.append(sentence)
        current_han += sentence_han

    if current:
        segments.append("".join(current))
    return tuple(segments)


def chunk(text: str, threshold_han: int = CHUNK_THRESHOLD_HAN) -> tuple[str, ...]:
    """把文本切成不超过 `threshold_han` 汉字的段（无标点可切时会超）。

    未超阈值时原样返回单段，不做任何重新拼装——短帖是主流路径，不该被改动。

    Raises:
        ValueError: 文本为空。
    """
    if not text.strip():
        raise ValueError("待切段的文本为空")

    if len(han_only(text)) <= threshold_han:
        return (text,)

    segments = (segment.strip() for segment in _pack(_split_sentences(text), threshold_han))
    return tuple(segment for segment in segments if segment)
